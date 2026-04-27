"""
Step 4: CKA Analysis
Calculate Centered Kernel Alignment (CKA) between different models' hidden states
"""

import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from tqdm import tqdm

from config import MODELS, OUTPUT_PATHS, ANALYSIS_CONFIG


def center_kernel(K: np.ndarray) -> np.ndarray:
    """
    Center a kernel matrix
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Calculate linear CKA between two sets of features
    
    Args:
        X: Features from model 1, shape (n_samples, dim1)
        Y: Features from model 2, shape (n_samples, dim2)
    
    Returns:
        CKA similarity score
    """
    # Compute kernel matrices
    K_X = X @ X.T
    K_Y = Y @ Y.T
    
    # Center the kernels
    K_X_centered = center_kernel(K_X)
    K_Y_centered = center_kernel(K_Y)
    
    # Calculate HSIC (Hilbert-Schmidt Independence Criterion)
    hsic = np.sum(K_X_centered * K_Y_centered)
    
    # Normalize
    norm_x = np.sqrt(np.sum(K_X_centered * K_X_centered))
    norm_y = np.sqrt(np.sum(K_Y_centered * K_Y_centered))
    
    if norm_x * norm_y == 0:
        return 0.0
    
    cka = hsic / (norm_x * norm_y)
    return float(cka)


def load_hidden_states_for_layer(
    model_key: str,
    num_samples: int,
    layer_idx: int,
    step_idx: int = 0
) -> np.ndarray:
    """
    Load hidden states for a specific layer across all samples
    
    Args:
        model_key: Model identifier (e.g., "qwen1.5B")
        num_samples: Number of samples to load
        layer_idx: Which layer to extract (0 = embedding layer, -1 = last layer)
        step_idx: Which generation step (0 = first generated token)
    
    Returns:
        Array of shape (n_samples, hidden_dim)
    """
    output_dir = MODELS[model_key]["output_dir"]
    hidden_states_list = []
    
    for i in range(num_samples):
        output_path = os.path.join(output_dir, f"sample_{i:03d}.pt")
        
        if not os.path.exists(output_path):
            continue
        
        try:
            output = torch.load(output_path, map_location="cpu")
            hidden_states = output.get("hidden_states_per_step", [])
            
            if len(hidden_states) > step_idx:
                step_hidden = hidden_states[step_idx]  # List of layer hiddens
                
                if len(step_hidden) > layer_idx:
                    layer_hidden = step_hidden[layer_idx]  # [hidden_dim]
                    hidden_states_list.append(layer_hidden.numpy())
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
    
    if not hidden_states_list:
        return None
    
    return np.stack(hidden_states_list, axis=0)


def calculate_cka_matrix(
    model1_key: str,
    model2_key: str,
    num_samples: int,
    step_idx: int = 0
) -> Tuple[np.ndarray, int, int]:
    """
    Calculate CKA matrix between all layers of two models
    
    Returns:
        cka_matrix: Array of shape (n_layers1, n_layers2)
        n_layers1: Number of layers in model 1
        n_layers2: Number of layers in model 2
    """
    print(f"Calculating CKA between {model1_key} and {model2_key}...")
    
    # First, determine the number of layers
    output_path = os.path.join(MODELS[model1_key]["output_dir"], "sample_000.pt")
    if not os.path.exists(output_path):
        print(f"Error: Sample file not found at {output_path}")
        return None, 0, 0
    
    output = torch.load(output_path, map_location="cpu")
    hidden_states = output.get("hidden_states_per_step", [])
    
    if not hidden_states or len(hidden_states) == 0:
        print("Error: No hidden states found")
        return None, 0, 0
    
    n_layers1 = len(hidden_states[0])
    
    # Check model 2
    output_path2 = os.path.join(MODELS[model2_key]["output_dir"], "sample_000.pt")
    output2 = torch.load(output_path2, map_location="cpu")
    hidden_states2 = output2.get("hidden_states_per_step", [])
    n_layers2 = len(hidden_states2[0])
    
    print(f"{model1_key} has {n_layers1} layers")
    print(f"{model2_key} has {n_layers2} layers")
    
    # Initialize CKA matrix
    cka_matrix = np.zeros((n_layers1, n_layers2))
    
    # Calculate CKA for each layer pair
    for i in tqdm(range(n_layers1), desc=f"Processing {model1_key} layers"):
        X = load_hidden_states_for_layer(model1_key, num_samples, i, step_idx)
        
        if X is None:
            print(f"Warning: Could not load layer {i} from {model1_key}")
            continue
        
        for j in range(n_layers2):
            Y = load_hidden_states_for_layer(model2_key, num_samples, j, step_idx)
            
            if Y is None:
                print(f"Warning: Could not load layer {j} from {model2_key}")
                continue
            
            # Ensure same number of samples
            min_samples = min(X.shape[0], Y.shape[0])
            X_sub = X[:min_samples]
            Y_sub = Y[:min_samples]
            
            # Calculate CKA
            cka_value = linear_cka(X_sub, Y_sub)
            cka_matrix[i, j] = cka_value
    
    return cka_matrix, n_layers1, n_layers2


def plot_cka_matrix(
    cka_matrix: np.ndarray,
    model1_name: str,
    model2_name: str,
    save_path: str
):
    """
    Plot CKA matrix as a heatmap
    """
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        cka_matrix,
        annot=False,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'CKA Similarity'}
    )
    
    plt.xlabel(f"{model2_name} Layers", fontsize=12)
    plt.ylabel(f"{model1_name} Layers", fontsize=12)
    plt.title(f"CKA Similarity: {model1_name} vs {model2_name}", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved CKA heatmap to {save_path}")


def plot_layer_similarity_curves(
    cka_matrices: Dict[str, np.ndarray],
    save_path: str
):
    """
    Plot CKA similarity curves showing diagonal and max similarity per layer
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for pair_name, cka_matrix in cka_matrices.items():
        # Extract diagonal (if square) or closest diagonal
        n_rows, n_cols = cka_matrix.shape
        min_dim = min(n_rows, n_cols)
        
        # Diagonal similarity
        diagonal = np.array([cka_matrix[i, i] if i < n_cols else 0 for i in range(n_rows)])
        
        # Max similarity per layer
        max_sim = np.max(cka_matrix, axis=1)
        
        # Plot diagonal
        axes[0].plot(range(len(diagonal)), diagonal, marker='o', label=pair_name, linewidth=2)
        
        # Plot max similarity
        axes[1].plot(range(len(max_sim)), max_sim, marker='s', label=pair_name, linewidth=2)
    
    axes[0].set_xlabel("Layer Index", fontsize=12)
    axes[0].set_ylabel("CKA Similarity", fontsize=12)
    axes[0].set_title("Diagonal CKA Similarity", fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel("Layer Index", fontsize=12)
    axes[1].set_ylabel("Max CKA Similarity", fontsize=12)
    axes[1].set_title("Maximum CKA Similarity per Layer", fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved similarity curves to {save_path}")


def find_most_similar_layers(cka_matrix: np.ndarray, top_k: int = 5) -> List[Tuple[int, int, float]]:
    """
    Find the most similar layer pairs
    
    Returns:
        List of (layer1_idx, layer2_idx, cka_score) tuples
    """
    # Flatten matrix and get top-k indices
    flat_indices = np.argsort(cka_matrix.ravel())[::-1][:top_k]
    
    # Convert back to 2D indices
    results = []
    for flat_idx in flat_indices:
        i, j = np.unravel_index(flat_idx, cka_matrix.shape)
        results.append((int(i), int(j), float(cka_matrix[i, j])))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Calculate CKA similarity")
    parser.add_argument("--num_samples", type=int, default=200,
                      help="Number of samples to use for CKA calculation")
    parser.add_argument("--step_idx", type=int, default=0,
                      help="Which generation step to analyze (0=first token)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Step 4: CKA Analysis")
    print("="*80)
    
    # Calculate CKA matrices for different model pairs
    model_pairs = [
        ("qwen1.5B", "qwen7B"),
        ("qwen7B", "qwen14B"),
        ("qwen1.5B", "qwen14B")
    ]
    
    cka_results = {}
    
    for model1, model2 in model_pairs:
        pair_name = f"{model1}_vs_{model2}"
        print(f"\nProcessing pair: {pair_name}")
        
        cka_matrix, n_layers1, n_layers2 = calculate_cka_matrix(
            model1,
            model2,
            args.num_samples,
            args.step_idx
        )
        
        if cka_matrix is not None:
            cka_results[pair_name] = cka_matrix
            
            # Save matrix
            if model1 == "qwen1.5B" and model2 == "qwen7B":
                output_path = OUTPUT_PATHS["cka_matrix_1.5B_vs_7B"]
            elif model1 == "qwen7B" and model2 == "qwen14B":
                output_path = OUTPUT_PATHS["cka_matrix_7B_vs_14B"]
            elif model1 == "qwen1.5B" and model2 == "qwen14B":
                output_path = OUTPUT_PATHS["cka_matrix_1.5B_vs_14B"]
            
            np.save(output_path, cka_matrix)
            print(f"Saved CKA matrix to {output_path}")
            
            # Plot individual heatmap
            plot_path = output_path.replace('.npy', '.png')
            plot_cka_matrix(cka_matrix, model1, model2, plot_path)
            
            # Find most similar layers
            similar_layers = find_most_similar_layers(cka_matrix, top_k=5)
            print(f"\nTop 5 most similar layer pairs for {pair_name}:")
            for layer1, layer2, score in similar_layers:
                print(f"  {model1} layer {layer1} <-> {model2} layer {layer2}: CKA = {score:.4f}")
    
    # Plot combined similarity curves
    if cka_results:
        curves_path = os.path.join(ANALYSIS_CONFIG["cka_output_dir"], "cka_curves.png")
        plot_layer_similarity_curves(cka_results, curves_path)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("CKA Analysis Summary")
    print("="*80)
    
    for pair_name, cka_matrix in cka_results.items():
        print(f"\n{pair_name}:")
        print(f"  Matrix shape: {cka_matrix.shape}")
        print(f"  Mean CKA: {np.mean(cka_matrix):.4f}")
        print(f"  Max CKA: {np.max(cka_matrix):.4f}")
        print(f"  Min CKA: {np.min(cka_matrix):.4f}")
        
        # Diagonal statistics
        n_rows, n_cols = cka_matrix.shape
        min_dim = min(n_rows, n_cols)
        diagonal = np.array([cka_matrix[i, i] for i in range(min_dim)])
        print(f"  Mean diagonal CKA: {np.mean(diagonal):.4f}")
    
    print("="*80)


if __name__ == "__main__":
    main()
