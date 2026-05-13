"""
Step 6b: Feature Cosine Distance Measurement (Geometric Experiment — Phase II)

Measures the geometric distance between correct and incorrect option tokens in
the hidden state space of each model.  If the hypothesis is correct, small models
(1.5B) will have correct and incorrect options "squeezed together" (high cosine
similarity), while 7B will have them "pushed apart" (low cosine similarity).

Methodology:
  1. For each error sample, locate the positions of GT token (e.g. "B") and the
     small-model's predicted token (e.g. "D") in the input sequence.
  2. Extract the hidden state vectors at these positions from the last layer
     (and optionally all layers for evolution analysis).
  3. Compute cosine similarity, L2 distance, and logit difference between
     the GT and predicted option vectors.
  4. Compare these metrics across 1.5B, 3B, and 7B.

Input:  step2 .pt files (prefill_hidden_states or hidden_states_per_step)
Output: option_distance_analysis.json, 3× PNG visualizations

No additional inference required — all data is already saved by step2.
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from config import MODELS, ANALYSIS_CONFIG, INJECTION_CONFIG, MEMORY_CONFIG
from qa_utils import (
    get_gt_answer, get_answer_type, get_clean_answer, get_answer_token_id,
)
from utils import load_model_outputs, save_json


# ---------------------------------------------------------------------------
# Core analysis logic
# ---------------------------------------------------------------------------


def analyze_sample_distances(
    sample_idx: int,
    outputs_list: Dict[str, Dict],
    tokenizer,
    option_letters: List[str],
    device: str = "cuda",
    proj_matrices: Optional[Dict[str, torch.Tensor]] = None,
) -> Optional[Dict[str, Any]]:
    """Analyze hidden state distances for a single error sample across models.

    Data structure from step2:
      - prefill_hidden_states: list[num_layers] of [hidden_dim] (last input token)
      - hidden_states_per_step: list[num_gen_steps] of list[num_layers] of [hidden_dim]

    Cross-model cosine requires projection matrices (from step7) since
    hidden dims differ (1.5B=1536, 3B=2048, 7B=3584).
    proj_matrices: dict mapping f"{src}_to_{tgt}" → W tensor [tgt_dim, src_dim]
    """
    # Get GT answer and predicted answer from 1.5B
    out_1_5B = outputs_list.get("1.5B")
    if out_1_5B is None:
        return None

    gt_answer = get_gt_answer(out_1_5B.get("ground_truth", {}))
    answer_type = get_answer_type(out_1_5B.get("ground_truth", {}))
    pred_answer, _ = get_clean_answer(out_1_5B)

    if not gt_answer or not pred_answer:
        return None

    if answer_type != "multiple_choice":
        return None

    # Determine GT token and predicted token
    gt_letter = gt_answer.strip().upper()
    pred_letter = pred_answer.strip().upper()

    if gt_letter not in option_letters or pred_letter not in option_letters:
        return None

    if gt_letter == pred_letter:
        return None  # Not an error sample

    result = {
        "sample_idx": sample_idx,
        "sample_id": out_1_5B.get("sample_id", f"sample_{sample_idx}"),
        "dataset": out_1_5B.get("ground_truth", {}).get("dataset", "unknown"),
        "gt_answer": gt_letter,
        "pred_answer": pred_letter,
        "per_model": {},
    }

    for model_name, out in outputs_list.items():
        if out is None:
            continue

        hs_per_step = out.get("hidden_states_per_step", [])
        if not hs_per_step:
            result["per_model"][model_name] = {"error": "no hidden_states_per_step"}
            continue

        # Normalize: list of list → list of tensors [num_layers, hidden_dim]
        step_tensors = []
        for step_data in hs_per_step:
            if isinstance(step_data, (list, tuple)):
                step_tensors.append(
                    torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in step_data])
                )
            elif torch.is_tensor(step_data):
                step_tensors.append(step_data)
            else:
                continue

        if not step_tensors:
            result["per_model"][model_name] = {"error": "empty hidden_states_per_step"}
            continue

        num_steps = len(step_tensors)
        num_layers = step_tensors[0].shape[0]
        hidden_dim = step_tensors[0].shape[1]

        # --- Last generation step, last layer: the "answer vector" ---
        h_answer = step_tensors[-1][-1].to(device)  # [hidden_dim]

        # --- Layer-wise evolution at the answer step ---
        layer_evolution = []
        layer_vectors = []
        for layer_idx in range(num_layers):
            h_l = step_tensors[-1][layer_idx].to(device)
            layer_vectors.append(h_l)

        # Stack all layers: [num_layers, hidden_dim]
        all_layers = torch.stack(layer_vectors, dim=0)

        # Cosine similarity between adjacent layers (measures "resolution change")
        if num_layers > 1:
            for i in range(num_layers - 1):
                cos_l = F.cosine_similarity(
                    all_layers[i].unsqueeze(0), all_layers[i + 1].unsqueeze(0), dim=-1
                ).item()
                layer_evolution.append(float(cos_l))

        # --- Step-wise evolution: cosine between consecutive generation steps (last layer) ---
        step_evolution = []
        for step_i in range(min(num_steps - 1, 20)):  # Cap at 20 steps
            h_s1 = step_tensors[step_i][-1].to(device)
            h_s2 = step_tensors[step_i + 1][-1].to(device)
            cos_s = F.cosine_similarity(
                h_s1.unsqueeze(0), h_s2.unsqueeze(0), dim=-1
            ).item()
            step_evolution.append(float(cos_s))

        # --- Cross-model reference: cosine between this model's answer vector
        #     and 7B's answer vector (projected to 7B space if needed) ---
        cross_model_cos = {}
        # Save original h_answer — must NOT be mutated across iterations
        h_answer_orig = h_answer

        for other_name, other_out in outputs_list.items():
            if other_name == model_name or other_out is None:
                continue
            other_hs = other_out.get("hidden_states_per_step", [])
            if not other_hs:
                continue
            other_last = other_hs[-1]
            if isinstance(other_last, (list, tuple)):
                other_last = torch.stack(
                    [t if torch.is_tensor(t) else torch.tensor(t) for t in other_last]
                )
            if not torch.is_tensor(other_last):
                continue

            h_other = other_last[-1].to(device)

            # Check dimension compatibility
            if h_answer_orig.shape[-1] != h_other.shape[-1]:
                # Need projection: project h_answer to other model's space
                pair_key = f"{model_name.replace('B', '')}B_to_{other_name.replace('B', '')}B"
                # Try common key formats
                pair_keys_to_try = [
                    f"{model_name.replace('B', '')}B_to_{other_name.replace('B', '')}B",
                    f"qwen{model_name}_to_qwen{other_name}",
                ]
                W = None
                if proj_matrices:
                    for pk in pair_keys_to_try:
                        if pk in proj_matrices:
                            W = proj_matrices[pk]
                            break
                if W is None:
                    # Try reverse projection: project h_other to h_answer's space
                    pair_keys_rev = [
                        f"{other_name.replace('B', '')}B_to_{model_name.replace('B', '')}B",
                        f"qwen{other_name}_to_qwen{model_name}",
                    ]
                    if proj_matrices:
                        for pk in pair_keys_rev:
                            if pk in proj_matrices:
                                W = proj_matrices[pk]
                                # Use reverse projection: project h_other → h_answer space
                                h_other_proj = F.linear(h_other.float(), W.float())
                                cos_cross = F.cosine_similarity(
                                    h_answer_orig.unsqueeze(0), h_other_proj.unsqueeze(0), dim=-1
                                ).item()
                                cross_model_cos[f"{model_name}_vs_{other_name}"] = float(cos_cross)
                                W = None  # Mark as handled to skip the forward block below
                                break

                if W is not None:
                    # Forward projection: project h_answer → h_other space
                    h_answer_proj = F.linear(h_answer_orig.float(), W.float())
                    cos_cross = F.cosine_similarity(
                        h_answer_proj.unsqueeze(0), h_other.unsqueeze(0), dim=-1
                    ).item()
                    cross_model_cos[f"{model_name}_vs_{other_name}"] = float(cos_cross)
                elif f"{model_name}_vs_{other_name}" not in cross_model_cos:
                    # No projection found in either direction
                    cross_model_cos[f"{model_name}_vs_{other_name}"] = None
            else:
                cos_cross = F.cosine_similarity(
                    h_answer_orig.unsqueeze(0), h_other.unsqueeze(0), dim=-1
                ).item()
                cross_model_cos[f"{model_name}_vs_{other_name}"] = float(cos_cross)

        result["per_model"][model_name] = {
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_gen_steps": num_steps,
            "answer_vector_norm": float(torch.norm(h_answer_orig).item()),
            "layer_evolution_cosine": layer_evolution,
            "step_evolution_cosine": step_evolution,
            "cross_model_cosine": cross_model_cos,
        }

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_cross_model_cosine(
    all_results: List[Dict],
    output_dir: str,
):
    """Plot cross-model cosine similarity distribution.

    Three panels:
      (a) Box plot of cross-model cosine similarity
      (b) Histogram of cosine similarity distribution
      (c) Scatter: 1.5B-vs-7B vs 3B-vs-7B cosine
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    model_colors = {"1.5B_vs_7B": "#2196F3", "3B_vs_7B": "#4CAF50", "1.5B_vs_3B": "#FF9800"}
    pair_order = ["1.5B_vs_7B", "3B_vs_7B", "1.5B_vs_3B"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Collect data per pair (filter out None values)
    data_cosine = {p: [] for p in pair_order}
    for r in all_results:
        for mn, pm in r.get("per_model", {}).items():
            for pair_key, cos_val in pm.get("cross_model_cosine", {}).items():
                if pair_key in data_cosine and cos_val is not None:
                    data_cosine[pair_key].append(cos_val)

    # (a) Box plot
    valid_pairs = [p for p in pair_order if data_cosine[p]]
    if valid_pairs:
        bp = axes[0].boxplot(
            [data_cosine[p] for p in valid_pairs],
            tick_labels=valid_pairs, patch_artist=True, widths=0.5,
        )
        for patch, pk in zip(bp["boxes"], valid_pairs):
            patch.set_facecolor(model_colors[pk])
            patch.set_alpha(0.6)

        for i, pk in enumerate(valid_pairs):
            x = np.random.normal(i + 1, 0.04, len(data_cosine[pk]))
            axes[0].scatter(x, data_cosine[pk], alpha=0.3, s=20, color=model_colors[pk])

    axes[0].set_ylabel("Cosine Similarity (Answer Vectors)")
    axes[0].set_title("(a) Cross-Model Cosine Similarity\n(Higher = more similar)")
    axes[0].grid(axis="y", alpha=0.3)

    # (b) Histogram overlay
    for pk in valid_pairs:
        axes[1].hist(data_cosine[pk], bins=20, alpha=0.5,
                     label=pk, color=model_colors[pk], edgecolor="white")
    axes[1].set_xlabel("Cosine Similarity")
    axes[1].set_ylabel("Count")
    axes[1].set_title("(b) Cosine Similarity Distribution")
    axes[1].legend(loc="best", fontsize=10)
    axes[1].grid(alpha=0.3)

    # (c) Scatter: 1.5B-vs-7B vs 3B-vs-7B
    if data_cosine["1.5B_vs_7B"] and data_cosine["3B_vs_7B"]:
        # Align by sample index
        cos_15 = []
        cos_3B = []
        for r in all_results:
            c15 = None
            c3 = None
            for mn, pm in r.get("per_model", {}).items():
                if "1.5B_vs_7B" in pm.get("cross_model_cosine", {}):
                    c15 = pm["cross_model_cosine"]["1.5B_vs_7B"]
                if "3B_vs_7B" in pm.get("cross_model_cosine", {}):
                    c3 = pm["cross_model_cosine"]["3B_vs_7B"]
            if c15 is not None and c3 is not None:
                cos_15.append(c15)
                cos_3B.append(c3)
        if cos_15:
            axes[2].scatter(cos_15, cos_3B, alpha=0.5, s=30, color="#9C27B0")
            axes[2].plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
    axes[2].set_xlabel("1.5B vs 7B Cosine")
    axes[2].set_ylabel("3B vs 7B Cosine")
    axes[2].set_title("(c) 1.5B vs 3B Similarity to 7B\n(Points above line = 3B closer to 7B)")
    axes[2].legend(loc="best", fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "option_distance_cosine_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_layer_evolution(
    all_results: List[Dict],
    output_dir: str,
):
    """Plot adjacent-layer cosine similarity evolution across layers.

    Shows how much each layer changes the hidden representation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_colors = {"1.5B": "#2196F3", "3B": "#4CAF50", "7B": "#FF9800"}

    fig, ax = plt.subplots(figsize=(10, 6))

    for mn in ["1.5B", "3B", "7B"]:
        evolutions = []
        for r in all_results:
            pm = r.get("per_model", {}).get(mn, {})
            evo = pm.get("layer_evolution_cosine", [])
            if evo:
                evolutions.append(evo)

        if not evolutions:
            continue

        max_len = max(len(e) for e in evolutions)
        padded = []
        for e in evolutions:
            padded.append(e + [np.nan] * (max_len - len(e)))
        padded = np.array(padded)

        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)

        x = range(len(mean_curve))
        ax.plot(x, mean_curve, label=mn, color=model_colors[mn], linewidth=2)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        color=model_colors[mn], alpha=0.15)

    ax.set_xlabel("Layer Transition (i → i+1)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Adjacent-Layer Cosine Similarity at Answer Step\n"
                 "(Lower = more information change per layer)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "option_distance_layer_evolution.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_step_evolution(
    all_results: List[Dict],
    output_dir: str,
):
    """Plot step-wise cosine similarity between consecutive generation steps.

    Shows how much the hidden state changes at each generation step.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_colors = {"1.5B": "#2196F3", "3B": "#4CAF50", "7B": "#FF9800"}

    fig, ax = plt.subplots(figsize=(10, 6))

    for mn in ["1.5B", "3B", "7B"]:
        evolutions = []
        num_gen_steps_list = []
        for r in all_results:
            pm = r.get("per_model", {}).get(mn, {})
            evo = pm.get("step_evolution_cosine", [])
            n_steps = pm.get("num_gen_steps", 0)
            if evo:
                evolutions.append(evo)
            num_gen_steps_list.append(n_steps)

        if not evolutions:
            continue

        max_len = max(len(e) for e in evolutions)
        padded = []
        for e in evolutions:
            padded.append(e + [np.nan] * (max_len - len(e)))
        padded = np.array(padded)

        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)

        x = range(len(mean_curve))
        ax.plot(x, mean_curve, label=mn, color=model_colors[mn], linewidth=2)
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                        color=model_colors[mn], alpha=0.15)

    ax.set_xlabel("Generation Step Transition (i → i+1)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Step-wise Hidden State Change (Last Layer)\n"
                 "(Lower = more change between consecutive generation steps)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    # If no data was plotted, show an informative message
    if not ax.lines:
        # Collect avg gen steps info
        info_parts = []
        for mn in ["1.5B", "3B", "7B"]:
            steps_list = []
            for r in all_results:
                pm = r.get("per_model", {}).get(mn, {})
                ns = pm.get("num_gen_steps", 0)
                if ns:
                    steps_list.append(ns)
            if steps_list:
                info_parts.append(f"{mn}: avg={np.mean(steps_list):.1f}")
        info_text = "No step-wise data (all samples have only 1 generation step)\n"
        if info_parts:
            info_text += "Avg generation steps: " + ", ".join(info_parts)
        info_text += "\nStep evolution requires ≥2 generation steps (e.g., numerical answers)"
        ax.text(0.5, 0.5, info_text, transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="#666",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", edgecolor="#ccc"))

    plt.tight_layout()
    path = os.path.join(output_dir, "option_distance_step_evolution.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    num_samples: int,
    device: str = "cuda",
    output_dir: str = "experiment_results/option_distance",
):
    """Run the full option distance analysis."""
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    model_name = MODELS["qwen1.5B"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load three-model comparison to identify error samples
    cmp_path = ANALYSIS_CONFIG["three_model_comparison"]
    if not os.path.exists(cmp_path):
        print(f"  Error: three_model_comparison.csv not found at {cmp_path}")
        print("  Please run step3 first.")
        return

    df_cmp = pd.read_csv(cmp_path)
    error_samples = df_cmp[(df_cmp["correct_7B"]) & (~df_cmp["correct_1.5B"])].copy()
    print(f"\nFound {len(error_samples)} injection targets")

    if len(error_samples) == 0:
        print("  No error samples to analyze.")
        return

    # Try to load projection matrices from step7 (needed for cross-model cosine)
    proj_matrices = {}
    proj_dir = INJECTION_CONFIG.get("projection_dir", "experiment_results/projection_matrices")
    if os.path.isdir(proj_dir):
        for fname in os.listdir(proj_dir):
            if fname.startswith("W_up_") and fname.endswith(".pt"):
                # Parse: W_up_qwen1.5B_to_qwen7B.pt → qwen1.5B_to_qwen7B
                pair_tag = fname[5:-3]  # strip "W_up_" and ".pt"
                W_path = os.path.join(proj_dir, fname)
                try:
                    W = torch.load(W_path, map_location=device)
                    proj_matrices[pair_tag] = W.to(device)
                    print(f"  Loaded projection: {pair_tag} {W.shape}")
                except Exception as e:
                    print(f"  Warning: failed to load {W_path}: {e}")
        if not proj_matrices:
            print("  Warning: No projection matrices found in step7 output.")
            print("  Cross-model cosine comparison will be skipped.")
            print("  Run step7 first for full cross-model analysis.")

    # Load model outputs
    print(f"Loading model outputs to {device}...")
    outputs_1_5B = load_model_outputs("qwen1.5B", num_samples, map_location=device)
    outputs_3B = load_model_outputs("qwen3B", num_samples, map_location=device)
    outputs_7B = load_model_outputs("qwen7B", num_samples, map_location=device)

    option_letters = ["A", "B", "C", "D"]

    # Analyze each error sample
    all_results = []
    for _, row in tqdm(error_samples.iterrows(), total=len(error_samples),
                       desc="Measuring option distances"):
        idx = int(row["sample_idx"])
        outputs_list = {
            "1.5B": outputs_1_5B[idx] if idx < len(outputs_1_5B) else None,
            "3B": outputs_3B[idx] if idx < len(outputs_3B) else None,
            "7B": outputs_7B[idx] if idx < len(outputs_7B) else None,
        }
        result = analyze_sample_distances(
            idx, outputs_list, tokenizer, option_letters,
            device=device, proj_matrices=proj_matrices,
        )
        if result is not None:
            all_results.append(result)

    # --- Aggregate statistics ---
    print(f"\n{'=' * 80}")
    print("Feature Cosine Distance Analysis Results")
    print(f"{'=' * 80}")

    summary = {}

    # 1. Cross-model cosine similarity (answer vectors)
    cross_pairs = ["1.5B_vs_7B", "3B_vs_7B", "1.5B_vs_3B"]
    for pair_key in cross_pairs:
        cos_vals = []
        for r in all_results:
            for mn, pm in r.get("per_model", {}).items():
                if pair_key in pm.get("cross_model_cosine", {}):
                    cos_vals.append(pm["cross_model_cosine"][pair_key])
        if cos_vals:
            cos_vals = [v for v in cos_vals if v is not None]  # Filter None (no projection)
        if cos_vals:
            summary[pair_key] = {
                "n_samples": len(cos_vals),
                "cosine_mean": float(np.mean(cos_vals)),
                "cosine_median": float(np.median(cos_vals)),
                "cosine_std": float(np.std(cos_vals)),
                "cosine_min": float(min(cos_vals)),
                "cosine_max": float(max(cos_vals)),
            }
            print(f"\n  {pair_key} ({len(cos_vals)} samples):")
            print(f"    Cosine Similarity (answer vectors):")
            print(f"      mean={np.mean(cos_vals):.4f}, median={np.median(cos_vals):.4f}")
            print(f"      std={np.std(cos_vals):.4f}, range=[{min(cos_vals):.4f}, {max(cos_vals):.4f}]")

    # 2. Per-model statistics
    for mn in ["1.5B", "3B", "7B"]:
        valid = [r for r in all_results
                 if mn in r.get("per_model", {}) and "answer_vector_norm" in r["per_model"][mn]]
        n = len(valid)
        if n == 0:
            print(f"\n  {mn}: no valid results")
            continue

        norms = [r["per_model"][mn]["answer_vector_norm"] for r in valid]
        print(f"\n  {mn} ({n} samples):")
        print(f"    Answer vector norm: mean={np.mean(norms):.2f}, median={np.median(norms):.2f}")

        summary[mn] = {
            "n_samples": n,
            "answer_vector_norm_mean": float(np.mean(norms)),
        }

    # 3. Hypothesis test: Resolution Compression
    cos_15_7B = summary.get("1.5B_vs_7B", {}).get("cosine_mean", 0)
    cos_3B_7B = summary.get("3B_vs_7B", {}).get("cosine_mean", 0)
    cos_15_3B = summary.get("1.5B_vs_3B", {}).get("cosine_mean", 0)

    print(f"\n  *** Hypothesis Test: Resolution Compression ***")
    print(f"    1.5B vs 7B cosine: {cos_15_7B:.4f}")
    print(f"    3B vs 7B cosine:   {cos_3B_7B:.4f}")
    print(f"    1.5B vs 3B cosine: {cos_15_3B:.4f}")
    print(f"    (Higher cosine = more similar hidden states)")
    if cos_15_7B > 0.95:
        print(f"    >>> STRONG: 1.5B and 7B answer vectors nearly identical — confirms")
        print(f"        small model 'knows' the answer but fails at final output step")
    elif cos_15_7B > 0.85:
        print(f"    >>> MODERATE: 1.5B and 7B answer vectors highly similar")

    # Save results
    save_path = os.path.join(output_dir, "option_distance_analysis.json")
    save_json({"summary": summary, "results": all_results}, save_path)
    print(f"\n  Results saved to: {save_path}")

    # Generate plots
    print("\n  Generating plots...")
    plot_cross_model_cosine(all_results, output_dir)
    plot_layer_evolution(all_results, output_dir)
    plot_step_evolution(all_results, output_dir)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 6b: Feature Cosine Distance Measurement (Phase II)"
    )
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--output_dir", type=str,
                        default="experiment_results/option_distance")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for computation (default: cuda if available)")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print("=" * 80)
    print("Step 6b: Feature Cosine Distance Measurement (Phase II)")
    print("  Proving: small model's hidden space 'squeezes' options together")
    print("=" * 80)

    run_analysis(num_samples=args.num_samples, device=device, output_dir=args.output_dir)

    print("\n" + "=" * 80)
    print("Step 6b Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()