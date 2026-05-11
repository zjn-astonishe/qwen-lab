"""
Step 5: CKA Analysis (V3 — heavily optimized)

Calculate Centered Kernel Alignment (CKA) between different models' hidden states.

V3 Optimizations:
  - Load each .pt file ONCE, extract ALL layers in memory (eliminates ~55K redundant I/O)
  - Batch all CKA computations on GPU with vectorized kernel matrix math
  - Remove cleanup_gpu() from inner loop; call once per model pair
  - Pre-compute centering matrix H once instead of per-CKA call
  - Keep data on GPU throughout; minimize CPU↔GPU transfers
"""

import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from config import MODELS, OUTPUT_PATHS, ANALYSIS_CONFIG
from utils import load_model_output

# Embedding layer skip constant
EMBED_SKIP = 1


# ---------------------------------------------------------------------------
# Hidden state loading (optimized: load each file once, extract all layers)
# ---------------------------------------------------------------------------

def load_all_layers_for_model(
    model_key: str, num_samples: int, step_idx: int = 0,
) -> Optional[np.ndarray]:
    """Load hidden states for ALL layers across all samples in ONE pass.

    Instead of opening each .pt file 28-36 times (once per layer), this opens
    each file exactly once and extracts all layers into a 3-D array.

    Returns:
        Array of shape (n_layers, n_samples, hidden_dim), or None if no data.
    """
    output_dir = MODELS[model_key]["output_dir"]
    all_layers_per_sample = []   # list of (n_layers, hidden_dim) arrays
    n_layers = None

    for i in range(num_samples):
        output = load_model_output(output_dir, i)
        if output is None:
            continue

        hidden_states = output.get("hidden_states_per_step", [])
        if len(hidden_states) <= step_idx:
            continue

        step_layers = hidden_states[step_idx]

        if isinstance(step_layers, list):
            # Skip embedding layer
            actual_layers = step_layers[EMBED_SKIP:]
        elif isinstance(step_layers, torch.Tensor):
            actual_layers = [step_layers]
        else:
            continue

        # Convert tensors to numpy
        layer_arrays = []
        for layer_hidden in actual_layers:
            if isinstance(layer_hidden, torch.Tensor):
                layer_arrays.append(layer_hidden.numpy())
            elif layer_hidden is not None:
                layer_arrays.append(np.array(layer_hidden))
            else:
                layer_arrays.append(None)

        if n_layers is None:
            n_layers = len(layer_arrays)

        all_layers_per_sample.append(layer_arrays)

    if not all_layers_per_sample or n_layers is None or n_layers == 0:
        return None

    # Build (n_layers, n_samples, hidden_dim) array
    # Some layers may be None for some samples; filter those out
    hidden_dim = None
    for layer_list in all_layers_per_sample:
        for arr in layer_list:
            if arr is not None:
                hidden_dim = arr.shape[-1]
                break
        if hidden_dim is not None:
            break

    if hidden_dim is None:
        return None

    n_samples_actual = len(all_layers_per_sample)
    result = np.full((n_layers, n_samples_actual, hidden_dim), np.nan, dtype=np.float32)

    for s_idx, layer_list in enumerate(all_layers_per_sample):
        for l_idx, arr in enumerate(layer_list):
            if arr is not None and l_idx < n_layers:
                result[l_idx, s_idx, :arr.shape[0]] = arr[:hidden_dim]

    return result


def _get_num_transformer_layers_from_data(data: np.ndarray) -> int:
    """Get number of layers from pre-loaded data."""
    return data.shape[0] if data is not None else 0


# ---------------------------------------------------------------------------
# CKA computation (vectorized GPU batch)
# ---------------------------------------------------------------------------

def batch_linear_cka(
    layers1: np.ndarray, layers2: np.ndarray, device: str = 'cuda',
) -> np.ndarray:
    """Compute the full CKA matrix between two models' layers using batched GPU ops.

    Args:
        layers1: (n1, n_samples, d1) — model 1 layer activations
        layers2: (n2, n_samples, d2) — model 2 layer activations

    Returns:
        CKA matrix of shape (n1, n2)
    """
    n1, n_samples, d1 = layers1.shape
    n2, _, d2 = layers2.shape

    # Normalize each layer's features (mean=0, std=1 per feature dimension)
    # layers1: (n1, n_samples, d1)
    X = torch.from_numpy(layers1).float()
    X_mean = X.mean(dim=1, keepdim=True)
    X_std = X.std(dim=1, keepdim=True).clamp(min=1e-8)
    X = (X - X_mean) / X_std
    del X_mean, X_std

    Y = torch.from_numpy(layers2).float()
    Y_mean = Y.mean(dim=1, keepdim=True)
    Y_std = Y.std(dim=1, keepdim=True).clamp(min=1e-8)
    Y = (Y - Y_mean) / Y_std
    del Y_mean, Y_std

    # Move to GPU
    X = X.to(device)
    Y = Y.to(device)

    # Pre-compute centering matrix H = I - (1/n) * 11^T
    H = torch.eye(n_samples, device=device) - torch.ones(
        (n_samples, n_samples), device=device
    ) / n_samples

    # Compute CKA matrix — process in chunks to manage GPU memory
    cka_matrix = torch.zeros((n1, n2), device=device)
    chunk_size = 8  # process 8 layers of model1 at a time

    for i_start in range(0, n1, chunk_size):
        i_end = min(i_start + chunk_size, n1)
        # X_batch: (chunk, n_samples, d1)
        X_batch = X[i_start:i_end]

        # Kernel matrices: K_X = X @ X^T  →  (chunk, n, n)
        K_X = torch.bmm(X_batch, X_batch.transpose(1, 2))

        for j_start in range(0, n2, chunk_size):
            j_end = min(j_start + chunk_size, n2)
            Y_chunk = Y[j_start:j_end]

            K_Y = torch.bmm(Y_chunk, Y_chunk.transpose(1, 2))

            # Center: K_Xc = H @ K_X @ H  (broadcast H over batch)
            # (n, n) @ (chunk, n, n) @ (n, n) → use einsum or loop
            # Efficient: K_Xc[k] = H @ K_X[k] @ H
            K_Xc = torch.bmm(H.unsqueeze(0).expand(K_X.shape[0], -1, -1), K_X)
            K_Xc = torch.bmm(K_Xc, H.unsqueeze(0).expand(K_X.shape[0], -1, -1))

            K_Yc = torch.bmm(H.unsqueeze(0).expand(K_Y.shape[0], -1, -1), K_Y)
            K_Yc = torch.bmm(K_Yc, H.unsqueeze(0).expand(K_Y.shape[0], -1, -1))

            # HSIC(k, l) = sum(K_Xc[k] * K_Yc[l])
            # For all pairs in chunk: use broadcasting
            # K_Xc: (chunk_x, n, n), K_Yc: (chunk_y, n, n)
            # hsic_pairs: (chunk_x, chunk_y) via einsum
            hsic = torch.einsum('xij,yij->xy', K_Xc, K_Yc)

            # Norms: ||K_Xc[k]||_F
            norm_x = torch.sqrt(torch.einsum('xij,xij->x', K_Xc, K_Xc))
            norm_y = torch.sqrt(torch.einsum('yij,yij->y', K_Yc, K_Yc))

            # CKA = HSIC / (||K_Xc|| * ||K_Yc||)
            denom = norm_x.unsqueeze(1) * norm_y.unsqueeze(0)  # (chunk_x, chunk_y)
            denom = denom.clamp(min=1e-10)
            cka_block = hsic / denom

            cka_matrix[i_start:i_end, j_start:j_end] = cka_block

            del K_Y, K_Yc, hsic, norm_y, denom, cka_block

        del X_batch, K_X, K_Xc, norm_x

    # Single GPU→CPU transfer for the entire result
    result = cka_matrix.cpu().numpy()

    # Clean up GPU once
    del X, Y, H, cka_matrix
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# CKA matrix computation (orchestrator)
# ---------------------------------------------------------------------------

def calculate_cka_matrix(
    model1_key: str, model2_key: str, num_samples: int, step_idx: int = 0,
) -> Tuple[Optional[np.ndarray], int, int]:
    """Calculate CKA similarity matrix between two models' layers."""
    print(f"\nCalculating CKA: {model1_key} vs {model2_key}...")

    # Load all layers for both models in ONE pass each
    print(f"  Loading {model1_key} (all layers in single pass)...")
    data1 = load_all_layers_for_model(model1_key, num_samples, step_idx)

    if data1 is None:
        print(f"  Error: no hidden states found for {model1_key}")
        return None, 0, 0

    n_layers1 = data1.shape[0]
    n_samples1 = data1.shape[1]
    print(f"  {model1_key}: {n_layers1} layers, {n_samples1} samples, dim={data1.shape[2]}")

    print(f"  Loading {model2_key} (all layers in single pass)...")
    data2 = load_all_layers_for_model(model2_key, num_samples, step_idx)

    if data2 is None:
        print(f"  Error: no hidden states found for {model2_key}")
        return None, 0, 0

    n_layers2 = data2.shape[0]
    n_samples2 = data2.shape[1]
    print(f"  {model2_key}: {n_layers2} layers, {n_samples2} samples, dim={data2.shape[2]}")

    # Align sample counts (use minimum)
    min_samples = min(n_samples1, n_samples2)
    data1 = data1[:, :min_samples, :]
    data2 = data2[:, :min_samples, :]

    # Check for NaN-only layers and replace with zeros
    # (layers with all NaN means that layer had no valid data)
    for l in range(n_layers1):
        if np.all(np.isnan(data1[l])):
            data1[l] = 0.0
    for l in range(n_layers2):
        if np.all(np.isnan(data2[l])):
            data2[l] = 0.0

    # Handle NaN within layers: replace with 0 (mean-centered, so 0 is neutral)
    np.nan_to_num(data1, copy=False, nan=0.0)
    np.nan_to_num(data2, copy=False, nan=0.0)

    # Compute CKA matrix (fully batched on GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Computing CKA matrix ({n_layers1}x{n_layers2}) on {device}...")
    cka_matrix = batch_linear_cka(data1, data2, device=device)

    print(f"  Mean CKA: {cka_matrix.mean():.4f}")

    del data1, data2
    return cka_matrix, n_layers1, n_layers2


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_cka_matrix(cka_matrix: np.ndarray, model1_name: str, model2_name: str,
                    save_path: str):
    """Plot CKA matrix as a heatmap."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cka_matrix, annot=False, cmap="YlOrRd", vmin=0, vmax=1,
                cbar_kws={'label': 'CKA Similarity'})
    plt.xlabel(f"{model2_name} Layers", fontsize=12)
    plt.ylabel(f"{model1_name} Layers", fontsize=12)
    plt.title(f"CKA Similarity: {model1_name} vs {model2_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap: {save_path}")


def plot_layer_similarity_curves(cka_matrices: Dict[str, np.ndarray], save_path: str):
    """Plot diagonal and max CKA similarity curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for pair_name, cka_matrix in cka_matrices.items():
        n_rows, n_cols = cka_matrix.shape
        min_dim = min(n_rows, n_cols)
        diagonal = np.array([cka_matrix[i, i] if i < n_cols else 0 for i in range(n_rows)])
        max_sim = np.max(cka_matrix, axis=1)

        axes[0].plot(range(len(diagonal)), diagonal, marker='o', label=pair_name, linewidth=2)
        axes[1].plot(range(len(max_sim)), max_sim, marker='s', label=pair_name, linewidth=2)

    axes[0].set_xlabel("Layer Index", fontsize=12)
    axes[0].set_ylabel("CKA Similarity", fontsize=12)
    axes[0].set_title("Diagonal CKA Similarity", fontsize=14)
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Layer Index", fontsize=12)
    axes[1].set_ylabel("Max CKA Similarity", fontsize=12)
    axes[1].set_title("Maximum CKA Similarity per Layer", fontsize=14)
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved curves: {save_path}")


def find_most_similar_layers(cka_matrix: np.ndarray, top_k: int = 5) -> List[Tuple[int, int, float]]:
    """Find the top-k most similar layer pairs."""
    flat_indices = np.argsort(cka_matrix.ravel())[::-1][:top_k]
    results = []
    for flat_idx in flat_indices:
        i, j = np.unravel_index(flat_idx, cka_matrix.shape)
        results.append((int(i), int(j), float(cka_matrix[i, j])))
    return results


# ---------------------------------------------------------------------------
# Model pair -> output path mapping
# ---------------------------------------------------------------------------

_CKA_PATH_MAP = {
    ("qwen1.5B", "qwen7B"): "cka_matrix_1.5B_vs_7B",
    ("qwen7B", "qwen3B"): "cka_matrix_7B_vs_3B",
    ("qwen1.5B", "qwen3B"): "cka_matrix_1.5B_vs_3B",
}


def _get_cka_output_path(model1: str, model2: str) -> str:
    """Get the CKA matrix output path for a model pair."""
    key = (model1, model2)
    if key in _CKA_PATH_MAP:
        return OUTPUT_PATHS[_CKA_PATH_MAP[key]]
    # Dynamic fallback
    return os.path.join(ANALYSIS_CONFIG["cka_output_dir"], f"cka_matrix_{model1}_vs_{model2}.npy")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 5: CKA Analysis")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--step_idx", type=int, default=0)
    args = parser.parse_args()

    print("=" * 80)
    print("Step 5: CKA Analysis (V3 — optimized)")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU detected, CKA will run on CPU (slower)")

    model_pairs = [
        ("qwen1.5B", "qwen7B"),
        ("qwen7B", "qwen3B"),
        ("qwen1.5B", "qwen3B"),
    ]

    cka_results = {}

    for model1, model2 in model_pairs:
        pair_name = f"{model1}_vs_{model2}"
        print(f"\n--- {pair_name} ---")

        cka_matrix, n1, n2 = calculate_cka_matrix(model1, model2, args.num_samples, args.step_idx)

        if cka_matrix is not None:
            cka_results[pair_name] = cka_matrix

            # Save matrix
            output_path = _get_cka_output_path(model1, model2)
            np.save(output_path, cka_matrix)
            print(f"  Saved: {output_path}")

            # Plot heatmap
            plot_path = output_path.replace('.npy', '.png')
            plot_cka_matrix(cka_matrix, model1, model2, plot_path)

            # Top similar layers
            top_pairs = find_most_similar_layers(cka_matrix, top_k=5)
            print(f"  Top 5 similar layers:")
            for l1, l2, score in top_pairs:
                print(f"    {model1} layer {l1} <-> {model2} layer {l2}: CKA = {score:.4f}")

    # Combined similarity curves
    if cka_results:
        curves_path = OUTPUT_PATHS["cka_plot"]
        plot_layer_similarity_curves(cka_results, curves_path)

    # Summary
    print(f"\n{'=' * 80}")
    print("CKA Analysis Summary")
    print("=" * 80)
    for pair_name, matrix in cka_results.items():
        min_dim = min(matrix.shape)
        diagonal = np.array([matrix[i, i] for i in range(min_dim)])
        print(f"\n  {pair_name}:")
        print(f"    Shape: {matrix.shape}, mean: {np.mean(matrix):.4f}, "
              f"diag_mean: {np.mean(diagonal):.4f}")
        print(f"    Range: [{np.min(matrix):.4f}, {np.max(matrix):.4f}]")

    print("=" * 80)


if __name__ == "__main__":
    main()