"""
Step 5: Train Projection Matrix
Train a linear projection matrix to map small model hidden states to large model space
"""

import os
import json
import torch
import numpy as np
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.linear_model import Ridge

from config import MODELS, DATA_CONFIG, OUTPUT_PATHS, INJECTION_CONFIG


def load_alignment_samples() -> List[Dict]:
    """
    Load alignment samples for training projection matrix
    """
    alignment_path = DATA_CONFIG["alignment_data_path"]
    
    if not os.path.exists(alignment_path):
        print(f"Warning: Alignment data not found at {alignment_path}")
        print("Using test samples instead...")
        alignment_path = DATA_CONFIG["sampled_data_path"]
    
    with open(alignment_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    return samples


def collect_hidden_state_pairs(
    small_model_key: str,
    large_model_key: str,
    num_samples: int,
    layer_idx_small: int = -2,
    layer_idx_large: int = -2,
    step_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect paired hidden states from both models at specified layers
    
    Args:
        small_model_key: Key for small model (e.g., "qwen1.5B")
        large_model_key: Key for large model (e.g., "qwen7B")
        num_samples: Number of samples to collect
        layer_idx_small: Which layer to extract from small model (-1 = last, -2 = second-to-last)
        layer_idx_large: Which layer to extract from large model
        step_idx: Which generation step (0 = first token)
    
    Returns:
        X: Small model hidden states, shape (n_samples, dim_small)
        Y: Large model hidden states, shape (n_samples, dim_large)
    """
    small_output_dir = MODELS[small_model_key]["output_dir"]
    large_output_dir = MODELS[large_model_key]["output_dir"]
    
    X_list = []
    Y_list = []
    
    print(f"Collecting hidden state pairs from layer {layer_idx_small} (small) and {layer_idx_large} (large)...")
    
    for i in tqdm(range(num_samples)):
        # Load small model output
        small_path = os.path.join(small_output_dir, f"sample_{i:03d}.pt")
        if not os.path.exists(small_path):
            continue
        
        try:
            small_output = torch.load(small_path, map_location="cpu")
            small_hidden = small_output.get("hidden_states_per_step", [])
            
            if len(small_hidden) <= step_idx:
                continue
            
            small_layer_hidden = small_hidden[step_idx][layer_idx_small]  # [hidden_dim]
            
            # Load large model output
            large_path = os.path.join(large_output_dir, f"sample_{i:03d}.pt")
            if not os.path.exists(large_path):
                continue
            
            large_output = torch.load(large_path, map_location="cpu")
            large_hidden = large_output.get("hidden_states_per_step", [])
            
            if len(large_hidden) <= step_idx:
                continue
            
            large_layer_hidden = large_hidden[step_idx][layer_idx_large]  # [hidden_dim]
            
            X_list.append(small_layer_hidden.numpy())
            Y_list.append(large_layer_hidden.numpy())
            
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
    
    if not X_list or not Y_list:
        print("Error: No valid hidden state pairs collected!")
        return None, None
    
    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    
    print(f"Collected {X.shape[0]} hidden state pairs")
    print(f"Small model hidden dim: {X.shape[1]}")
    print(f"Large model hidden dim: {Y.shape[1]}")
    
    return X, Y


def train_linear_projection(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train linear projection using Ridge regression
    
    Y = W @ X + b
    
    Args:
        X: Source features, shape (n_samples, dim_small)
        Y: Target features, shape (n_samples, dim_large)
        alpha: Ridge regularization parameter
    
    Returns:
        W: Projection matrix, shape (dim_large, dim_small)
        b: Bias vector, shape (dim_large,)
    """
    print(f"Training linear projection with Ridge (alpha={alpha})...")
    
    # Use Ridge regression
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X, Y)
    
    W = ridge.coef_  # Shape: (dim_large, dim_small)
    b = ridge.intercept_  # Shape: (dim_large,)
    
    # Calculate training error
    Y_pred = ridge.predict(X)
    mse = np.mean((Y - Y_pred) ** 2)
    
    print(f"Training MSE: {mse:.6f}")
    print(f"Projection matrix shape: {W.shape}")
    print(f"Bias shape: {b.shape}")
    
    return W, b


def evaluate_projection(
    W: np.ndarray,
    b: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate projection quality
    """
    # Project X to Y space
    Y_pred = X_test @ W.T + b
    
    # Calculate metrics
    mse = np.mean((Y_test - Y_pred) ** 2)
    
    # Cosine similarity
    Y_test_norm = Y_test / (np.linalg.norm(Y_test, axis=1, keepdims=True) + 1e-8)
    Y_pred_norm = Y_pred / (np.linalg.norm(Y_pred, axis=1, keepdims=True) + 1e-8)
    cosine_sim = np.mean(np.sum(Y_test_norm * Y_pred_norm, axis=1))
    
    return {
        "mse": float(mse),
        "cosine_similarity": float(cosine_sim)
    }


def save_projection_matrix(W: np.ndarray, b: np.ndarray, output_dir: str):
    """
    Save projection matrix and bias
    """
    os.makedirs(output_dir, exist_ok=True)
    
    W_tensor = torch.from_numpy(W).float()
    b_tensor = torch.from_numpy(b).float()
    
    W_path = OUTPUT_PATHS["projection_matrix"]
    b_path = OUTPUT_PATHS["projection_bias"]
    
    torch.save(W_tensor, W_path)
    torch.save(b_tensor, b_path)
    
    print(f"Saved projection matrix to: {W_path}")
    print(f"Saved bias vector to: {b_path}")


def train_projection_for_multiple_layers(
    small_model_key: str,
    large_model_key: str,
    num_samples: int,
    layers_to_try: List[int] = [-1, -2, -3, -4]
) -> Dict[int, Dict]:
    """
    Train projection matrices for multiple layers and select the best one
    """
    results = {}
    
    for layer_idx in layers_to_try:
        print(f"\n{'='*60}")
        print(f"Training projection for layer {layer_idx}")
        print('='*60)
        
        # Collect hidden states
        X, Y = collect_hidden_state_pairs(
            small_model_key,
            large_model_key,
            num_samples,
            layer_idx_small=layer_idx,
            layer_idx_large=layer_idx,
            step_idx=0
        )
        
        if X is None or Y is None:
            print(f"Skipping layer {layer_idx} (no data)")
            continue
        
        # Split into train/test
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]
        
        # Train projection
        W, b = train_linear_projection(X_train, Y_train, alpha=1.0)
        
        # Evaluate
        train_metrics = evaluate_projection(W, b, X_train, Y_train)
        test_metrics = evaluate_projection(W, b, X_test, Y_test)
        
        print(f"\nTrain metrics: MSE={train_metrics['mse']:.6f}, Cosine Sim={train_metrics['cosine_similarity']:.4f}")
        print(f"Test metrics: MSE={test_metrics['mse']:.6f}, Cosine Sim={test_metrics['cosine_similarity']:.4f}")
        
        results[layer_idx] = {
            "W": W,
            "b": b,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics
        }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train projection matrix")
    parser.add_argument("--small_model", type=str, default="qwen1.5B",
                      help="Small model key")
    parser.add_argument("--large_model", type=str, default="qwen7B",
                      help="Large model key")
    parser.add_argument("--num_samples", type=int, default=100,
                      help="Number of alignment samples to use")
    parser.add_argument("--alpha", type=float, default=1.0,
                      help="Ridge regularization parameter")
    parser.add_argument("--layer_idx", type=int, default=-2,
                      help="Which layer to use for projection (-1=last, -2=second-to-last, etc.)")
    parser.add_argument("--try_multiple_layers", action="store_true",
                      help="Try multiple layers and select best")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Step 5: Train Projection Matrix")
    print("="*80)
    
    if args.try_multiple_layers:
        # Train for multiple layers
        layers_to_try = INJECTION_CONFIG["injection_layers"]
        layers_to_try = [-i for i in layers_to_try]  # Convert to negative indices
        
        results = train_projection_for_multiple_layers(
            args.small_model,
            args.large_model,
            args.num_samples,
            layers_to_try
        )
        
        # Select best based on test cosine similarity
        best_layer = None
        best_cosine = -1
        
        for layer_idx, result in results.items():
            test_cosine = result["test_metrics"]["cosine_similarity"]
            if test_cosine > best_cosine:
                best_cosine = test_cosine
                best_layer = layer_idx
        
        print(f"\n{'='*80}")
        print("Best layer selection")
        print('='*80)
        print(f"Best layer: {best_layer}")
        print(f"Best test cosine similarity: {best_cosine:.4f}")
        
        # Save best projection
        best_result = results[best_layer]
        save_projection_matrix(
            best_result["W"],
            best_result["b"],
            INJECTION_CONFIG["projection_dir"]
        )
        
        # Save all results
        results_summary = {
            layer: {
                "train_mse": result["train_metrics"]["mse"],
                "train_cosine": result["train_metrics"]["cosine_similarity"],
                "test_mse": result["test_metrics"]["mse"],
                "test_cosine": result["test_metrics"]["cosine_similarity"]
            }
            for layer, result in results.items()
        }
        
        summary_path = os.path.join(INJECTION_CONFIG["projection_dir"], "layer_comparison.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"Saved layer comparison to: {summary_path}")
        
    else:
        # Train for single layer
        X, Y = collect_hidden_state_pairs(
            args.small_model,
            args.large_model,
            args.num_samples,
            layer_idx_small=args.layer_idx,
            layer_idx_large=args.layer_idx,
            step_idx=0
        )
        
        if X is None or Y is None:
            print("Error: Could not collect hidden states")
            return
        
        # Split into train/test
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]
        
        # Train projection
        W, b = train_linear_projection(X_train, Y_train, alpha=args.alpha)
        
        # Evaluate
        train_metrics = evaluate_projection(W, b, X_train, Y_train)
        test_metrics = evaluate_projection(W, b, X_test, Y_test)
        
        print(f"\nTrain metrics:")
        print(f"  MSE: {train_metrics['mse']:.6f}")
        print(f"  Cosine Similarity: {train_metrics['cosine_similarity']:.4f}")
        
        print(f"\nTest metrics:")
        print(f"  MSE: {test_metrics['mse']:.6f}")
        print(f"  Cosine Similarity: {test_metrics['cosine_similarity']:.4f}")
        
        # Save projection
        save_projection_matrix(W, b, INJECTION_CONFIG["projection_dir"])
    
    print("\n" + "="*80)
    print("Projection Matrix Training Complete")
    print("="*80)


if __name__ == "__main__":
    main()
