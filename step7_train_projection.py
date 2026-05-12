"""
Step 6: Train Projection Matrix (V3 — GPU accelerated)

Train a linear projection matrix to map small model hidden states to large model space.

V3 Optimizations:
  - Replaced sklearn Ridge with torch.linalg.lstsq on GPU (closed-form solution)
  - All matrix operations (evaluation, cosine similarity) run on GPU
  - Removed sklearn dependency entirely
  - Data stays on GPU from loading through training to evaluation
  - Removed unused cleanup_gpu import
"""

import os
import json
import torch
import numpy as np
import argparse
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from config import MODELS, DATA_CONFIG, INJECTION_CONFIG, get_projection_paths
from utils import load_model_output

# Embedding layer skip constant
EMBED_SKIP = 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_alignment_samples() -> List[Dict]:
    """Load alignment samples for training projection matrix."""
    alignment_path = DATA_CONFIG["alignment_data_path"]

    if not os.path.exists(alignment_path):
        print(f"  Warning: alignment data not found at {alignment_path}, using test data")
        alignment_path = DATA_CONFIG["sampled_data_path"]

    with open(alignment_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Hidden state pair collection
# ---------------------------------------------------------------------------

def collect_hidden_state_pairs(
    small_model_key: str,
    large_model_key: str,
    num_samples: int,
    layer_idx_small: int = -2,
    layer_idx_large: int = -2,
    step_idx: int = 0,
    sub_dir: str = "",
    device: str = 'cuda',
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Collect paired hidden states from both models at specified generation steps.

    Returns:
        X: Small model hidden states (n_samples, dim_small) on device, or None
        Y: Large model hidden states (n_samples, dim_large) on device, or None
    """
    small_output_dir = os.path.join(MODELS[small_model_key]["output_dir"], sub_dir) if sub_dir else MODELS[small_model_key]["output_dir"]
    large_output_dir = os.path.join(MODELS[large_model_key]["output_dir"], sub_dir) if sub_dir else MODELS[large_model_key]["output_dir"]

    X_list = []
    Y_list = []

    print(f"  Collecting pairs: step={step_idx}, small_layer={layer_idx_small}, large_layer={layer_idx_large}")

    for i in tqdm(range(num_samples), desc="  Loading pairs"):
        small_output = load_model_output(small_output_dir, i)
        large_output = load_model_output(large_output_dir, i)

        if small_output is None or large_output is None:
            continue

        # Extract small model hidden state
        small_hidden = small_output.get("hidden_states_per_step", [])
        if len(small_hidden) <= step_idx:
            continue
        step_layers = small_hidden[step_idx]
        if isinstance(step_layers, list):
            if abs(layer_idx_small) >= len(step_layers):
                continue
            small_hs = step_layers[layer_idx_small]
        else:
            small_hs = step_layers

        # Extract large model hidden state
        large_hidden = large_output.get("hidden_states_per_step", [])
        if len(large_hidden) <= step_idx:
            continue
        step_layers = large_hidden[step_idx]
        if isinstance(step_layers, list):
            if abs(layer_idx_large) >= len(step_layers):
                continue
            large_hs = step_layers[layer_idx_large]
        else:
            large_hs = step_layers

        # Keep as tensors — stack to GPU later
        if small_hs is None or large_hs is None:
            continue

        # Hidden states from step2 are already 1D vectors (hidden_dim,)
        # No need to average over token dimension
        if isinstance(small_hs, torch.Tensor):
            X_list.append(small_hs.float())
        else:
            X_list.append(torch.tensor(np.array(small_hs), dtype=torch.float32))

        if isinstance(large_hs, torch.Tensor):
            Y_list.append(large_hs.float())
        else:
            Y_list.append(torch.tensor(np.array(large_hs), dtype=torch.float32))

    if not X_list or not Y_list:
        print("  Error: no valid hidden state pairs collected!")
        return None, None

    # Stack and move to GPU in one transfer
    X = torch.stack(X_list, dim=0).to(device)   # (n_samples, dim_small)
    Y = torch.stack(Y_list, dim=0).to(device)   # (n_samples, dim_large)
    print(f"  Collected {X.shape[0]} pairs: dim_small={X.shape[1]}, dim_large={Y.shape[1]} [on {device}]")
    return X, Y


# ---------------------------------------------------------------------------
# Projection training (GPU — closed-form Ridge solution)
# ---------------------------------------------------------------------------

def train_linear_projection(
    X: torch.Tensor, Y: torch.Tensor, alpha: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Train linear projection using Ridge regression (closed-form on GPU).

    Solves: W = Y^T @ X @ (X^T @ X + αI)^{-1}
    With bias: augment X with ones column.

    All operations on GPU.
    """
    # Augment X with bias column: [X, 1]
    ones = torch.ones(X.shape[0], 1, device=X.device, dtype=X.dtype)
    X_aug = torch.cat([X, ones], dim=1)  # (n, d+1)

    d_small = X.shape[1]

    # Ridge closed-form: W = (X^T X + αI)^{-1} X^T Y
    XtX = X_aug.t() @ X_aug  # (d+1, d+1)
    XtX += alpha * torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)

    # Solve using Cholesky decomposition (faster and more stable than inverse)
    try:
        L = torch.linalg.cholesky(XtX)
        # Solve L Z = X^T Y  then  L^T W = Z
        XtY = X_aug.t() @ Y  # (d+1, d_large)
        Z = torch.linalg.solve_triangular(L, XtY, upper=False)
        W_aug = torch.linalg.solve_triangular(L.t(), Z, upper=True)
    except RuntimeError:
        # Fallback to lstsq if Cholesky fails
        W_aug = torch.linalg.lstsq(X_aug, Y).solution

    # Split weight and bias
    W = W_aug[:d_small, :].t()  # (d_large, d_small)
    b = W_aug[d_small, :]       # (d_large,)

    # Evaluate training MSE
    Y_pred = X_aug @ W_aug
    mse = torch.mean((Y - Y_pred) ** 2).item()

    print(f"  Training MSE: {mse:.6f}")
    print(f"  W shape: {W.shape}, b shape: {b.shape}")
    return W, b


def evaluate_projection(
    W: torch.Tensor, b: torch.Tensor, X_test: torch.Tensor, Y_test: torch.Tensor,
) -> Dict[str, float]:
    """Evaluate projection quality on test data (GPU)."""
    Y_pred = X_test @ W.t() + b

    mse = torch.mean((Y_test - Y_pred) ** 2).item()

    Y_test_norm = torch.nn.functional.normalize(Y_test, p=2, dim=1)
    Y_pred_norm = torch.nn.functional.normalize(Y_pred, p=2, dim=1)
    cosine_sim = torch.sum(Y_test_norm * Y_pred_norm, dim=1).mean().item()

    return {"mse": float(mse), "cosine_similarity": float(cosine_sim)}


# ---------------------------------------------------------------------------
# Save (uses dynamic paths based on model pair)
# ---------------------------------------------------------------------------

def save_projection_matrix(
    W: torch.Tensor, b: torch.Tensor,
    small_model: str, large_model: str,
) -> Dict[str, str]:
    """Save projection matrix and bias using model-pair-specific paths."""
    paths = get_projection_paths(small_model, large_model)
    os.makedirs(os.path.dirname(paths["projection_matrix"]), exist_ok=True)

    torch.save(W.float().cpu(), paths["projection_matrix"])
    torch.save(b.float().cpu(), paths["projection_bias"])

    print(f"  Saved W: {paths['projection_matrix']}")
    print(f"  Saved b: {paths['projection_bias']}")
    return paths


# ---------------------------------------------------------------------------
# Multi-layer training
# ---------------------------------------------------------------------------

def train_projection_for_multiple_layers(
    small_model_key: str,
    large_model_key: str,
    num_samples: int,
    layers_to_try: List[int] = [-1, -2, -3, -4],
    sub_dir: str = "",
    step_idx: int = 0,
    alpha: float = 1.0,
    device: str = 'cuda',
) -> Dict[int, Dict]:
    """Train projection matrices for multiple layers and return all results."""
    results = {}

    for layer_idx in layers_to_try:
        print(f"\n--- Training for layer {layer_idx} ---")

        X, Y = collect_hidden_state_pairs(
            small_model_key, large_model_key, num_samples,
            layer_idx_small=layer_idx, layer_idx_large=layer_idx,
            step_idx=step_idx, sub_dir=sub_dir, device=device,
        )

        if X is None or Y is None:
            print(f"  Skipping layer {layer_idx} (no data)")
            continue

        # 80/20 train/test split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]

        W, b = train_linear_projection(X_train, Y_train, alpha=alpha)

        train_metrics = evaluate_projection(W, b, X_train, Y_train)
        test_metrics = evaluate_projection(W, b, X_test, Y_test)

        print(f"  Train: MSE={train_metrics['mse']:.6f}, CosSim={train_metrics['cosine_similarity']:.4f}")
        print(f"  Test:  MSE={test_metrics['mse']:.6f}, CosSim={test_metrics['cosine_similarity']:.4f}")

        results[layer_idx] = {
            "W": W, "b": b,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 6: Train Projection Matrix")
    parser.add_argument("--small_model", type=str, default="qwen1.5B")
    parser.add_argument("--large_model", type=str, default="qwen7B")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--layer_idx", type=int, default=-2)
    parser.add_argument("--try_multiple_layers", action="store_true")
    parser.add_argument("--use_alignment_dir", action="store_true")
    parser.add_argument("--step_idx", type=int, default=0)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 80)
    print("Step 6: Train Projection Matrix (V3 — GPU)")
    print(f"  {args.small_model} -> {args.large_model}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 80)

    sub_dir = "alignment" if args.use_alignment_dir else ""

    if args.try_multiple_layers:
        # Convert positive layer numbers to negative indices
        layers_config = INJECTION_CONFIG["injection_layers"]
        layers_to_try = [-i for i in layers_config]

        results = train_projection_for_multiple_layers(
            args.small_model, args.large_model, args.num_samples,
            layers_to_try, sub_dir=sub_dir, step_idx=args.step_idx,
            alpha=args.alpha, device=device,
        )

        if not results:
            print("\nError: no layers had valid data for training.")
            return

        # Select best by test cosine similarity
        best_layer = max(results.keys(), key=lambda k: results[k]["test_metrics"]["cosine_similarity"])
        best_cosine = results[best_layer]["test_metrics"]["cosine_similarity"]

        print(f"\n{'=' * 60}")
        print(f"Best layer: {best_layer} (test cosine sim: {best_cosine:.4f})")
        print("=" * 60)

        # Save best projection
        saved_paths = save_projection_matrix(
            results[best_layer]["W"], results[best_layer]["b"],
            args.small_model, args.large_model,
        )

        # Save layer comparison summary
        comparison = {
            str(layer): {
                "train_mse": r["train_metrics"]["mse"],
                "train_cosine": r["train_metrics"]["cosine_similarity"],
                "test_mse": r["test_metrics"]["mse"],
                "test_cosine": r["test_metrics"]["cosine_similarity"],
            }
            for layer, r in results.items()
        }
        comparison_path = get_projection_paths(args.small_model, args.large_model)["layer_comparison"]
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"  Saved comparison: {comparison_path}")

    else:
        # Single layer training
        X, Y = collect_hidden_state_pairs(
            args.small_model, args.large_model, args.num_samples,
            layer_idx_small=args.layer_idx, layer_idx_large=args.layer_idx,
            step_idx=args.step_idx, sub_dir=sub_dir, device=device,
        )

        if X is None or Y is None:
            print("Error: could not collect hidden states")
            return

        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        Y_train, Y_test = Y[:split_idx], Y[split_idx:]

        W, b = train_linear_projection(X_train, Y_train, alpha=args.alpha)

        train_m = evaluate_projection(W, b, X_train, Y_train)
        test_m = evaluate_projection(W, b, X_test, Y_test)

        print(f"\n  Train MSE: {train_m['mse']:.6f}, CosSim={train_m['cosine_similarity']:.4f}")
        print(f"  Test  MSE: {test_m['mse']:.6f}, CosSim={test_m['cosine_similarity']:.4f}")

        save_projection_matrix(W, b, args.small_model, args.large_model)

    print("\n" + "=" * 80)
    print("Projection Matrix Training Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
