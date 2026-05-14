"""
Step 7b: Cross-Model LM Head Decoding (Causal Intervention Experiment — Phase III)

Ultimate causal proof: if the small model's error is purely due to "resolution
compression at the last step", then borrowing the large model's high-resolution
"magnifying glass" (LM Head) to decode the small model's hidden states should
instantly correct the output.

Methodology:
  1. Load the 7B model's LM Head weights and final RMSNorm.
  2. Load the projection matrix (W, b) trained by step7 (1.5B -> 7B).
  3. For each error sample (1.5B wrong, 7B correct):
     a. Extract 1.5B's hidden state at the last layer, generation position.
     b. Project it to 7B's space: h_proj = W @ h_small + b
     c. Apply 7B's RMSNorm: h_norm = RMSNorm(h_proj)
     d. Compute logits via 7B's LM Head: logits = h_norm @ W_lm_head.T
     e. Compare: does cross-decoded prediction match GT?
  4. Also test WITHOUT projection (raw 1.5B hidden state -> 7B LM Head) to
     isolate the effect of projection.
  5. Test at multiple layers to find the optimal intervention point.

Input:  step7 projection matrices (W.pt, b.pt), step2 .pt files
Output: cross_model_decode_results.json, 3x PNG visualizations

Requires loading 7B's LM Head weights (~14GB in bf16) but NO forward pass.
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
# Model loading (LM Head only, no forward pass)
# ---------------------------------------------------------------------------

def load_lm_head_and_norm(model_key: str, device: str = "cpu"):
    """Load only the LM Head and final RMSNorm from a model.

    Loads the full model on CPU (not CUDA) to avoid meta tensor issues,
    extracts the two needed weight tensors, then frees the model.
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    model_name = MODELS[model_key]["model_name"]
    print(f"  Loading LM Head from {model_name} (on CPU, extracting weights)...")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # ALWAYS load on CPU to avoid meta tensor issues from device_map="auto"
    # We only need two tensors (~2GB for 7B), then we free the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Extract and clone weights while model is alive
    lm_head_weight = model.lm_head.weight.detach().clone()   # [vocab_size, hidden_dim]
    norm_weight = model.model.norm.weight.detach().clone()    # [hidden_dim]

    # Get norm epsilon from config
    eps = 1e-6
    if hasattr(config, "rms_norm_eps"):
        eps = config.rms_norm_eps
    elif hasattr(config, "norm_eps"):
        eps = config.norm_eps

    hidden_dim = MODELS[model_key]["hidden_dim"]

    # Free the full model
    del model
    import gc
    gc.collect()

    print(f"    LM Head shape: {lm_head_weight.shape}")
    print(f"    Hidden dim: {hidden_dim}")
    print(f"    Norm weight shape: {norm_weight.shape}")

    return {
        "lm_head_weight": lm_head_weight,
        "norm_weight": norm_weight,
        "norm_eps": eps,
        "hidden_dim": hidden_dim,
        "vocab_size": lm_head_weight.shape[0] if lm_head_weight is not None else 0,
    }


def apply_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm: x * weight / sqrt(mean(x^2) + eps)."""
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    return (x.float() / rms).to(x.dtype) * weight.float()


# ---------------------------------------------------------------------------
# Cross-model decoding logic
# ---------------------------------------------------------------------------

def cross_model_decode(
    h_source: torch.Tensor,
    proj_W: Optional[torch.Tensor],
    proj_b: Optional[torch.Tensor],
    target_lm_head: Dict,
    source_lm_head: Optional[Dict] = None,
    use_projection: bool = True,
) -> Dict[str, Any]:
    """Decode a hidden state using a target model's LM Head.

    Args:
        h_source: Source model's hidden state [hidden_dim_src]
        proj_W: Projection matrix [hidden_dim_tgt, hidden_dim_src] or None
        proj_b: Projection bias [hidden_dim_tgt] or None
        target_lm_head: Dict with lm_head_weight, norm_weight, norm_eps
        source_lm_head: Optional source model's LM Head for baseline comparison
        use_projection: Whether to apply projection before decoding

    Returns:
        Dict with decoded predictions, probabilities, etc.
    """
    result = {}

    # --- Source model baseline (if available) ---
    if source_lm_head is not None:
        h_src_normed = apply_rms_norm(
            h_source, source_lm_head["norm_weight"], source_lm_head["norm_eps"]
        )
        logits_src = F.linear(h_src_normed.float(), source_lm_head["lm_head_weight"].float())
        probs_src = F.softmax(logits_src, dim=-1)
        top1_src = probs_src.argmax(dim=-1).item()
        result["source_top1"] = top1_src
        result["source_top1_prob"] = probs_src[top1_src].item()

    # --- Target model decoding ---
    if use_projection and proj_W is not None:
        # Project source hidden state to target space
        h_proj = F.linear(h_source.float(), proj_W.float(),
                          proj_b.float() if proj_b is not None else None)
    else:
        # Direct: use source hidden state with target LM Head
        # This requires dimensions to match (padded/truncated)
        src_dim = h_source.shape[-1]
        tgt_dim = target_lm_head["hidden_dim"]
        if src_dim > tgt_dim:
            h_proj = h_source.float()[:, :tgt_dim] if h_source.dim() == 2 else h_source.float()[:tgt_dim]
        elif src_dim < tgt_dim:
            padding = torch.zeros(tgt_dim - src_dim, dtype=torch.float32, device=h_source.device)
            h_proj = torch.cat([h_source.float(), padding], dim=-1)
        else:
            h_proj = h_source.float()

    # Apply target's RMSNorm
    h_normed = apply_rms_norm(
        h_proj, target_lm_head["norm_weight"], target_lm_head["norm_eps"]
    )

    # Compute logits via target's LM Head
    logits_tgt = F.linear(h_normed, target_lm_head["lm_head_weight"].float())
    probs_tgt = F.softmax(logits_tgt, dim=-1)

    top1_tgt = probs_tgt.argmax(dim=-1).item()

    result["cross_top1"] = top1_tgt
    result["cross_top1_prob"] = probs_tgt[top1_tgt].item()
    result["cross_top5"] = torch.topk(probs_tgt, 5).indices.tolist()
    result["cross_top5_probs"] = torch.topk(probs_tgt, 5).values.tolist()

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    num_samples: int,
    small_model: str = "qwen1.5B",
    large_model: str = "qwen7B",
    layers_to_test: Optional[List[int]] = None,
    device: str = "cuda",
    output_dir: str = "experiment_results/cross_model_decode",
):
    """Run the full cross-model decoding experiment."""
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    model_name = MODELS[small_model]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load projection matrix
    proj_paths = INJECTION_CONFIG.get("projection_dir",
                                       "experiment_results/projection_matrices")
    pair_tag = f"{small_model}_to_{large_model}"
    W_path = os.path.join(proj_paths, f"W_up_{pair_tag}.pt")
    b_path = os.path.join(proj_paths, f"bias_{pair_tag}.pt")

    if not os.path.exists(W_path):
        print(f"  Error: Projection matrix not found at {W_path}")
        print(f"  Please run step7 first.")
        return

    proj_W = torch.load(W_path, map_location=device)
    proj_b = torch.load(b_path, map_location=device) if os.path.exists(b_path) else None

    if proj_W is not None:
        proj_W = proj_W.to(device)
    if proj_b is not None:
        proj_b = proj_b.to(device)

    print(f"\n  Loaded projection matrix: {proj_W.shape} -> {device}")
    if proj_b is not None:
        print(f"  Loaded projection bias: {proj_b.shape} -> {device}")

    # Load 7B LM Head and norm
    print(f"\n  Loading {large_model} LM Head and RMSNorm on {device}...")
    target_lm = load_lm_head_and_norm(large_model, device=device)

    if target_lm["lm_head_weight"] is None or target_lm["norm_weight"] is None:
        print("  Error: Failed to load 7B LM Head or RMSNorm weights.")
        print("  Check that the model is cached locally.")
        return

    # Move weights to device
    for key in ["lm_head_weight", "norm_weight"]:
        if target_lm[key] is not None:
            target_lm[key] = target_lm[key].to(device)

    # Load 1.5B LM Head for baseline
    print(f"\n  Loading {small_model} LM Head and RMSNorm (baseline) on {device}...")
    source_lm = load_lm_head_and_norm(small_model, device=device)

    if source_lm["lm_head_weight"] is None or source_lm["norm_weight"] is None:
        print("  Error: Failed to load source model LM Head or RMSNorm weights.")
        return

    for key in ["lm_head_weight", "norm_weight"]:
        if source_lm[key] is not None:
            source_lm[key] = source_lm[key].to(device)

    # Load error samples
    cmp_path = ANALYSIS_CONFIG["three_model_comparison"]
    if not os.path.exists(cmp_path):
        print(f"  Error: three_model_comparison.csv not found at {cmp_path}")
        return

    df_cmp = pd.read_csv(cmp_path)
    # Column names use model size only (e.g. "correct_1.5B"), not full key (e.g. "correct_qwen1.5B")
    small_tag = small_model.replace("qwen", "")  # "qwen1.5B" -> "1.5B"
    large_tag = large_model.replace("qwen", "")
    error_samples = df_cmp[(df_cmp[f"correct_{large_tag}"]) & (~df_cmp[f"correct_{small_tag}"])].copy()
    print(f"\nFound {len(error_samples)} injection targets")

    if len(error_samples) == 0:
        print("  No error samples.")
        return

    # Load model outputs
    print(f"Loading model outputs to {device}...")
    outputs_small = load_model_outputs(small_model, num_samples, map_location=device)

    # Determine layers to test
    small_num_layers = MODELS[small_model]["num_layers"]
    if layers_to_test is None:
        # Test last 5 layers and specific key layers
        layers_to_test = list(range(max(0, small_num_layers - 5), small_num_layers))

    print(f"  Testing layers: {layers_to_test}")

    # Run cross-model decoding
    results = []
    for _, row in tqdm(error_samples.iterrows(), total=len(error_samples),
                       desc="Cross-model decoding"):
        idx = int(row["sample_idx"])
        if idx >= len(outputs_small) or outputs_small[idx] is None:
            continue

        out = outputs_small[idx]
        gt_answer = get_gt_answer(out.get("ground_truth", {}))
        answer_type = get_answer_type(out.get("ground_truth", {}))
        pred_answer, _ = get_clean_answer(out)

        if not gt_answer or answer_type != "multiple_choice":
            continue

        gt_token_id = get_answer_token_id(tokenizer, gt_answer, answer_type)
        pred_token_id = get_answer_token_id(tokenizer, pred_answer, answer_type) if pred_answer else None

        # Extract hidden states
        hs_per_step = out.get("hidden_states_per_step", [])
        prefill_hs = out.get("prefill_hidden_states")

        # Normalize to tensors (data may be stored as list)
        if prefill_hs is not None and isinstance(prefill_hs, (list, tuple)):
            prefill_hs = torch.stack([t if torch.is_tensor(t) else torch.tensor(t) for t in prefill_hs])
        if isinstance(hs_per_step, list) and len(hs_per_step) > 0:
            _new = []
            for t in hs_per_step:
                if torch.is_tensor(t):
                    _new.append(t)
                else:
                    try:
                        _new.append(torch.tensor(t))
                    except Exception:
                        pass
            hs_per_step = _new

        if not hs_per_step and prefill_hs is None:
            continue

        sample_result = {
            "sample_idx": idx,
            "sample_id": out.get("sample_id", f"sample_{idx}"),
            "dataset": out.get("ground_truth", {}).get("dataset", "unknown"),
            "gt_answer": gt_answer,
            "pred_answer": pred_answer if pred_answer else "",
            "gt_token_id": gt_token_id,
            "pred_token_id": pred_token_id,
            "per_layer": {},
        }

        # Decode at each test layer
        # Data structure from step2:
        #   hidden_states_per_step: list[num_gen_steps] of list[num_layers] of [hidden_dim]
        #   So hs_per_step[step_idx] = list of per-layer vectors [hidden_dim]
        for layer_idx in layers_to_test:
            h_gen = None

            # Try hidden_states_per_step (generation step hidden states)
            # Use step 0 (answer generation step) to match step7 training data
            if hs_per_step:
                answer_step = 0  # Answer generation step (matches step7 projection training)
                step_data = hs_per_step[answer_step]

                # step_data is list[num_layers] of [hidden_dim] or tensor [num_layers, hidden_dim]
                if isinstance(step_data, (list, tuple)):
                    if layer_idx >= len(step_data):
                        continue
                    h_vec = step_data[layer_idx]
                    if torch.is_tensor(h_vec):
                        h_gen = h_vec.to(device)
                    else:
                        h_gen = torch.tensor(h_vec, dtype=torch.float32, device=device)
                elif torch.is_tensor(step_data):
                    if step_data.dim() == 2 and layer_idx < step_data.shape[0]:
                        h_gen = step_data[layer_idx].to(device)

            # Fallback: try prefill_hidden_states
            if h_gen is None and prefill_hs is not None:
                if isinstance(prefill_hs, torch.Tensor) and prefill_hs.dim() == 2:
                    if layer_idx < prefill_hs.shape[0]:
                        h_gen = prefill_hs[layer_idx].to(device)

            if h_gen is None:
                continue

            # Cross-model decode WITH projection
            decode_result = cross_model_decode(
                h_source=h_gen,
                proj_W=proj_W,
                proj_b=proj_b,
                target_lm_head=target_lm,
                source_lm_head=source_lm,
                use_projection=True,
            )

            # Cross-model decode WITHOUT projection (direct)
            decode_direct = cross_model_decode(
                h_source=h_gen,
                proj_W=None,
                proj_b=None,
                target_lm_head=target_lm,
                use_projection=False,
            )

            # Check corrections
            corrected = decode_result["cross_top1"] == gt_token_id
            direct_corrected = decode_direct["cross_top1"] == gt_token_id

            # GT probability
            gt_prob_cross = F.softmax(
                F.linear(
                    apply_rms_norm(
                        F.linear(h_gen.float(), proj_W.float(),
                                proj_b.float() if proj_b is not None else None),
                        target_lm["norm_weight"], target_lm["norm_eps"]
                    ),
                    target_lm["lm_head_weight"].float()
                ), dim=-1
            )[gt_token_id].item() if gt_token_id is not None else 0.0

            gt_prob_source = decode_result.get("source_top1_prob", 0.0)

            sample_result["per_layer"][str(layer_idx)] = {
                "corrected": corrected,
                "direct_corrected": direct_corrected,
                "cross_top1": decode_result["cross_top1"],
                "cross_top1_prob": round(decode_result["cross_top1_prob"], 6),
                "cross_top5": decode_result["cross_top5"],
                "source_top1": decode_result.get("source_top1", -1),
                "source_top1_prob": round(decode_result.get("source_top1_prob", 0), 6),
                "gt_prob_cross": round(gt_prob_cross, 6),
                "gt_prob_source": round(gt_prob_source, 6),
                "gt_prob_gain": round(gt_prob_cross / max(gt_prob_source, 1e-10), 4),
            }

        results.append(sample_result)

    # --- Aggregate statistics ---
    print(f"\n{'=' * 80}")
    print("Cross-Model Decode Results (Causal Intervention)")
    print(f"{'=' * 80}")

    # Load projection layer comparison for best layer info
    layer_comp_path = os.path.join(proj_paths, f"layer_comparison_{pair_tag}.json")
    best_proj_layer = -1
    if os.path.exists(layer_comp_path):
        with open(layer_comp_path, "r") as f:
            layer_comp = __import__("json").load(f)
        if layer_comp:
            best_proj_layer = layer_comp.get("best_layer", -1)

    summary = {
        "total_error_samples": len(error_samples),
        "analyzed_samples": len(results),
        "best_projection_layer": best_proj_layer,
        "per_layer": {},
    }

    all_layer_keys = set()
    for r in results:
        all_layer_keys.update(r.get("per_layer", {}).keys())

    for layer_str in sorted(all_layer_keys, key=int):
        layer_results = [r["per_layer"][layer_str] for r in results if layer_str in r.get("per_layer", {})]
        if not layer_results:
            continue

        n = len(layer_results)
        n_corrected = sum(1 for r in layer_results if r["corrected"])
        n_direct_corrected = sum(1 for r in layer_results if r["direct_corrected"])
        mean_gain = np.mean([r["gt_prob_gain"] for r in layer_results])

        print(f"\n  Layer {layer_str} ({n} samples):")
        print(f"    With projection:    {n_corrected}/{n} corrected = {n_corrected/n*100:.1f}%")
        print(f"    Without projection: {n_direct_corrected}/{n} corrected = {n_direct_corrected/n*100:.1f}%")
        print(f"    Mean P(GT) gain:    {mean_gain:.2f}x")

        summary["per_layer"][layer_str] = {
            "n_samples": n,
            "corrected_with_proj": n_corrected,
            "corrected_without_proj": n_direct_corrected,
            "correction_rate_with_proj": n_corrected / n,
            "correction_rate_without_proj": n_direct_corrected / n,
            "mean_gt_prob_gain": float(mean_gain),
        }

    # Find best layer
    if not summary["per_layer"]:
        print("\n  No layer results to analyze — all samples failed at hidden state extraction.")
        print("  Check step2 data structure (hidden_states_per_step / prefill_hidden_states).")
        save_json({"summary": summary, "results": []}, save_path)
        return summary

    best_layer = max(summary["per_layer"].items(),
                      key=lambda x: x[1]["corrected_with_proj"])
    print(f"\n  *** Best intervention layer: {best_layer[0]} ***")
    print(f"      Correction rate (with projection):    {best_layer[1]['correction_rate_with_proj']*100:.1f}%")
    print(f"      Correction rate (without projection): {best_layer[1]['correction_rate_without_proj']*100:.1f}%")

    # Causal conclusion
    total = len(results)
    best_corrected = best_layer[1]["corrected_with_proj"]
    print(f"\n  *** CAUSAL CONCLUSION ***")
    if best_corrected / total > 0.3:
        print(f"  STRONG SUPPORT: {best_corrected}/{total} samples ({best_corrected/total*100:.1f}%)")
        print(f"  corrected by borrowing 7B's LM Head.")
        print(f"  This confirms the 'last-step resolution compression' hypothesis.")
    elif best_corrected / total > 0.1:
        print(f"  MODERATE SUPPORT: {best_corrected}/{total} samples ({best_corrected/total*100:.1f}%)")
        print(f"  corrected. Partial support for the hypothesis.")
    else:
        print(f"  WEAK SUPPORT: {best_corrected}/{total} samples ({best_corrected/total*100:.1f}%)")
        print(f"  corrected. The error may involve more than just resolution compression.")

    # Save results
    save_path = os.path.join(output_dir, "cross_model_decode_results.json")
    # Convert to serializable format
    serializable_results = []
    for r in results:
        sr = {
            "sample_idx": r["sample_idx"],
            "sample_id": r["sample_id"],
            "dataset": r["dataset"],
            "gt_answer": r["gt_answer"],
            "pred_answer": r["pred_answer"],
            "per_layer": r["per_layer"],
        }
        serializable_results.append(sr)

    save_json({"summary": summary, "results": serializable_results}, save_path)
    print(f"\n  Results saved to: {save_path}")

    # Generate plots
    print("\n  Generating plots...")
    _plot_correction_rates(summary, output_dir)
    _plot_prob_gain_comparison(results, output_dir)

    return summary


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_correction_rates(
    summary: Dict[str, Any],
    output_dir: str,
):
    """Plot correction rates across layers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    layers = sorted(summary["per_layer"].keys(), key=int)
    rates_proj = [summary["per_layer"][l]["correction_rate_with_proj"] * 100 for l in layers]
    rates_direct = [summary["per_layer"][l]["correction_rate_without_proj"] * 100 for l in layers]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(layers))

    ax.bar([i - 0.2 for i in x], rates_proj, width=0.4,
           label="With Projection", color="#FF9800", alpha=0.8, edgecolor="white")
    ax.bar([i + 0.2 for i in x], rates_direct, width=0.4,
           label="Without Projection (Direct)", color="#2196F3", alpha=0.8, edgecolor="white")

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Correction Rate (%)")
    ax.set_title("Cross-Model Decode: Correction Rate by Layer\n"
                 "(1.5B Hidden State -> 7B LM Head)")
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend(loc="best", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(max(rates_proj + [10]), max(rates_direct + [10])) * 1.2)

    plt.tight_layout()
    path = os.path.join(output_dir, "cross_decode_correction_rates.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_prob_gain_comparison(
    results: List[Dict],
    output_dir: str,
):
    """Plot P(GT) probability gain: source vs cross-decoded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Use the last available layer's results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for r in results:
        per_layer = r.get("per_layer", {})
        if not per_layer:
            continue
        last_layer = max(per_layer.keys(), key=int)
        lr = per_layer[last_layer]

        axes[0].scatter(
            lr.get("gt_prob_source", 0),
            lr.get("gt_prob_cross", 0),
            alpha=0.5, s=30, color="#FF9800",
        )

    # Diagonal reference
    max_val = 1.0
    axes[0].plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="No gain")
    axes[0].set_xlabel("P(GT) — 1.5B Self-Decoded")
    axes[0].set_ylabel("P(GT) — Cross-Decoded (7B LM Head)")
    axes[0].set_title("(a) P(GT) Comparison\n(Points above diagonal = improvement)")
    axes[0].legend(loc="best", fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, max_val)
    axes[0].set_ylim(0, max_val)

    # (b) Gain distribution
    gains = []
    for r in results:
        per_layer = r.get("per_layer", {})
        if not per_layer:
            continue
        last_layer = max(per_layer.keys(), key=int)
        lr = per_layer[last_layer]
        gains.append(lr.get("gt_prob_gain", 0))

    axes[1].hist(gains, bins=30, color="#4CAF50", alpha=0.7, edgecolor="white")
    axes[1].axvline(x=1.0, color="red", linestyle="--", alpha=0.7,
                     label="No gain (1x)")
    axes[1].set_xlabel("P(GT) Gain Ratio (Cross / Self)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("(b) P(GT) Gain Distribution\n(Bars right of red line = improvement)")
    axes[1].legend(loc="best", fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "cross_decode_prob_gain.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 7b: Cross-Model LM Head Decoding (Phase III — Causal Proof)"
    )
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--small_model", type=str, default="qwen1.5B")
    parser.add_argument("--large_model", type=str, default="qwen7B")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to test (e.g., '23,24,25,26,27')")
    parser.add_argument("--output_dir", type=str,
                        default="experiment_results/cross_model_decode")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for model weights and computation (default: cuda if available)")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    layers = None
    if args.layers:
        layers = [int(l.strip()) for l in args.layers.split(",")]

    print("=" * 80)
    print("Step 7b: Cross-Model LM Head Decoding (Phase III)")
    print("  Causal proof: does borrowing 7B's LM Head fix 1.5B's errors?")
    print("=" * 80)

    run_analysis(
        num_samples=args.num_samples,
        small_model=args.small_model,
        large_model=args.large_model,
        layers_to_test=layers,
        device=device,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 80)
    print("Step 7b Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
