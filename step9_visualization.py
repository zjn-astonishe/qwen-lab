"""
Step 8: Visualization (V2 — optimized)

Generate visualizations for probability distributions and injection effects.

Optimized:
  - Fixed field name mismatch: now correctly reads 'original_large_probs' from step6 output
  - Uses ast.literal_eval for safe details parsing
  - Uses config.OUTPUT_PATHS for consistent plot paths
  - Uses utils.load_model_output() for centralized loading
  - Better error handling for missing data
"""

import os
import ast
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List, Optional
from transformers import AutoTokenizer

from config import (
    MODELS, ANALYSIS_CONFIG, INJECTION_CONFIG, OUTPUT_PATHS,
    get_error_analysis_path, get_injection_results_path,
)
from utils import load_model_output, cleanup_gpu


def load_tokenizer(model_key: str):
    """Load tokenizer for decoding tokens."""
    model_name = MODELS[model_key]["model_name"]
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def _safe_parse_details(raw) -> Dict:
    """Safely parse details field using ast.literal_eval."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip() not in ("{}", "", "nan", "None"):
        try:
            return ast.literal_eval(raw)
        except (ValueError, TypeError, SyntaxError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Probability distribution comparison
# ---------------------------------------------------------------------------

def plot_probability_distribution_comparison(
    sample_idx: int,
    small_probs: torch.Tensor,
    large_probs: torch.Tensor,
    tokenizer,
    top_k: int = 30,
    save_path: str = None,
    indices1: torch.Tensor = None,
    indices2: torch.Tensor = None,
):
    """Plot side-by-side comparison of probability distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    k_val = min(top_k, small_probs.size(0), large_probs.size(0))
    small_topk = torch.topk(small_probs, k=k_val)
    large_topk = torch.topk(large_probs, k=k_val)

    # Decode tokens — use provided indices for sparse top-k
    if indices1 is not None:
        small_tokens = [tokenizer.decode([indices1[idx.item()].item()]) for idx in small_topk.indices]
    else:
        small_tokens = [tokenizer.decode([idx.item()]) for idx in small_topk.indices]

    if indices2 is not None:
        large_tokens = [tokenizer.decode([indices2[idx.item()].item()]) for idx in large_topk.indices]
    else:
        large_tokens = [tokenizer.decode([idx.item()]) for idx in large_topk.indices]

    small_vals = small_topk.values.numpy()
    large_vals = large_topk.values.numpy()

    axes[0].barh(range(top_k), small_vals[::-1], color='coral')
    axes[0].set_yticks(range(top_k))
    axes[0].set_yticklabels(small_tokens[::-1], fontsize=8)
    axes[0].set_xlabel('Probability', fontsize=12)
    axes[0].set_title('Small Model (1.5B) - Top Tokens', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)

    axes[1].barh(range(top_k), large_vals[::-1], color='skyblue')
    axes[1].set_yticks(range(top_k))
    axes[1].set_yticklabels(large_tokens[::-1], fontsize=8)
    axes[1].set_xlabel('Probability', fontsize=12)
    axes[1].set_title('Large Model (7B) - Top Tokens', fontsize=14)
    axes[1].grid(axis='x', alpha=0.3)

    plt.suptitle(f'Probability Distribution Comparison - Sample {sample_idx}', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_injection_effect(
    sample_idx: int,
    original_probs: torch.Tensor,
    injected_probs: torch.Tensor,
    gt_token: int,
    tokenizer,
    alpha: float,
    injection_layer: int,
    top_k: int = 30,
    save_path: str = None,
):
    """Plot probability distribution before and after injection."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    original_topk = torch.topk(original_probs, k=top_k)
    injected_topk = torch.topk(injected_probs, k=top_k)

    original_tokens = [tokenizer.decode([idx.item()]) for idx in original_topk.indices]
    injected_tokens = [tokenizer.decode([idx.item()]) for idx in injected_topk.indices]

    gt_token_str = tokenizer.decode([gt_token]) if gt_token >= 0 else None

    colors_orig = ['red' if tokenizer.decode([idx.item()]) == gt_token_str else 'skyblue'
                   for idx in original_topk.indices]
    axes[0].barh(range(top_k), original_topk.values.numpy()[::-1], color=colors_orig[::-1])
    axes[0].set_yticks(range(top_k))
    axes[0].set_yticklabels(original_tokens[::-1], fontsize=8)
    axes[0].set_xlabel('Probability', fontsize=12)
    axes[0].set_title('Before Injection (7B Original)', fontsize=14)
    axes[0].grid(axis='x', alpha=0.3)

    colors_inj = ['red' if tokenizer.decode([idx.item()]) == gt_token_str else 'coral'
                  for idx in injected_topk.indices]
    axes[1].barh(range(top_k), injected_topk.values.numpy()[::-1], color=colors_inj[::-1])
    axes[1].set_yticks(range(top_k))
    axes[1].set_yticklabels(injected_tokens[::-1], fontsize=8)
    axes[1].set_xlabel('Probability', fontsize=12)
    axes[1].set_title(f'After Injection (a={alpha}, layer={injection_layer})', fontsize=14)
    axes[1].grid(axis='x', alpha=0.3)

    title = f'Injection Effect - Sample {sample_idx}'
    if gt_token_str:
        title += f' (GT: {gt_token_str})'
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# B-ball case visualization
# ---------------------------------------------------------------------------

def visualize_b_ball_cases(error_analysis_path: str, num_cases: int = 10):
    """Visualize typical B-ball dilemma cases."""
    print("Visualizing B-ball dilemma cases...")

    df = pd.read_csv(error_analysis_path)
    b_ball_samples = df[df['is_b_ball_dilemma'] == True].head(num_cases)
    if len(b_ball_samples) == 0:
        print("  No B-ball cases found.")
        return

    tokenizer = load_tokenizer("qwen1.5B")
    plot_dir = ANALYSIS_CONFIG["prob_plots_dir"]
    os.makedirs(plot_dir, exist_ok=True)

    count = 0
    for _, row in b_ball_samples.iterrows():
        sample_idx = int(row['sample_idx'])

        small_output = load_model_output(MODELS["qwen1.5B"]["output_dir"], sample_idx)
        large_output = load_model_output(MODELS["qwen7B"]["output_dir"], sample_idx)

        if small_output is None or large_output is None:
            continue

        # Get probs — prefer full logits, fall back to top-k
        sp = small_output.get("probs_per_step", [])
        lp = large_output.get("probs_per_step", [])
        si = small_output.get("top_k_info", [])
        li = large_output.get("top_k_info", [])

        small_p = sp[0] if sp else None
        large_p = lp[0] if lp else None
        small_i = si[0]["indices"] if si else None
        large_i = li[0]["indices"] if li else None

        if small_p is None and (not si or si[0].get("probs") is None):
            continue
        if large_p is None and (not li or li[0].get("probs") is None):
            continue

        save_path = os.path.join(plot_dir, f"case_{sample_idx:03d}_comparison.png")
        plot_probability_distribution_comparison(
            sample_idx,
            small_p if small_p is not None else si[0]["probs"],
            large_p if large_p is not None else li[0]["probs"],
            tokenizer, top_k=30, save_path=save_path,
            indices1=small_i, indices2=large_i,
        )
        count += 1

    print(f"  Saved {count} B-ball case plots to {plot_dir}")


# ---------------------------------------------------------------------------
# Injection result visualization (FIXED: correct field name)
# ---------------------------------------------------------------------------

def visualize_injection_results(injection_results_path: str, num_cases: int = 10):
    """Visualize injection experiment results with before/after comparison."""
    print("Visualizing injection results...")

    if not os.path.exists(injection_results_path):
        print(f"  Injection results not found at {injection_results_path}")
        return

    probs_data_path = injection_results_path.replace('.csv', '_probs.pt')
    if not os.path.exists(probs_data_path):
        print(f"  Probability data not found at {probs_data_path}")
        print("  Re-run step6 to generate probability data for visualization")
        return

    print(f"  Loading probability data from {probs_data_path}")
    all_probs_data = torch.load(probs_data_path, map_location="cpu")
    df = pd.read_csv(injection_results_path)

    valid_results = df[df['gt_rank_after_injection'] > 0]
    if len(valid_results) == 0:
        print("  No valid injection results to visualize")
        return

    # Get unique samples and their best configurations
    best_samples = valid_results.groupby('sample_idx')['gt_rank_after_injection'].min()
    best_samples = best_samples.nsmallest(num_cases)

    tokenizer = load_tokenizer("qwen7B")
    plot_dir = ANALYSIS_CONFIG["prob_plots_dir"]
    os.makedirs(plot_dir, exist_ok=True)

    count = 0
    for sample_idx in best_samples.index:
        # Find probs_data for this sample
        sample_probs = None
        for pd_entry in all_probs_data:
            if pd_entry["sample_idx"] == sample_idx:
                sample_probs = pd_entry
                break
        if sample_probs is None:
            continue

        # Get original probs (FIXED: correct field name from step6)
        original_probs = sample_probs.get('original_large_probs')
        if original_probs is None:
            continue

        # Get the best configuration
        sample_results = valid_results[valid_results['sample_idx'] == sample_idx]
        best_config = sample_results.nsmallest(1, 'gt_rank_after_injection').iloc[0]

        injection_layer = int(best_config['injection_layer'])
        alpha = float(best_config['alpha'])
        gt_token = int(best_config['gt_token_id'])
        gt_rank = int(best_config['gt_rank_after_injection'])

        key = f"layer{injection_layer}_alpha{alpha}"
        injected_probs = sample_probs.get("injected_probs", {}).get(key)

        if injected_probs is None:
            continue

        save_path = os.path.join(
            plot_dir, f"injection_sample_{sample_idx:03d}_layer{injection_layer}_alpha{alpha}.png"
        )
        plot_injection_effect(
            sample_idx, original_probs, injected_probs,
            gt_token, tokenizer, alpha, injection_layer,
            top_k=30, save_path=save_path,
        )
        count += 1

    print(f"  Saved {count} injection comparison plots to {plot_dir}")


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------

def create_summary_plots(
    error_analysis_path: str,
    injection_results_path: str,
    output_dir: str,
):
    """Create summary visualization panel."""
    print("Creating summary plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Error type distribution
    if os.path.exists(error_analysis_path):
        df_errors = pd.read_csv(error_analysis_path)
        error_counts = df_errors['error_type'].value_counts()

        axes[0, 0].bar(error_counts.index, error_counts.values, color='coral')
        axes[0, 0].set_xlabel('Error Type', fontsize=12)
        axes[0, 0].set_ylabel('Count', fontsize=12)
        axes[0, 0].set_title('Error Type Distribution (Small Model)', fontsize=14)
        axes[0, 0].tick_params(axis='x', rotation=45)

        total_errors = df_errors['has_error'].sum()
        if total_errors > 0:
            axes[0, 0].text(0.5, 0.95, f'Total errors: {total_errors}/{len(df_errors)}',
                             transform=axes[0, 0].transAxes, ha='center', va='top',
                             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2 & 3: Injection effectiveness (share valid_inj scope)
    valid_inj = None
    if os.path.exists(injection_results_path):
        df_inj = pd.read_csv(injection_results_path)
        valid_inj = df_inj[df_inj['gt_rank_after_injection'] > 0]

        if len(valid_inj) > 0:
            # Plot 2: Injection effectiveness by alpha
            alpha_perf = valid_inj.groupby('alpha')['gt_rank_after_injection'].mean()
            axes[0, 1].plot(alpha_perf.index, alpha_perf.values,
                             marker='o', linewidth=2, markersize=8, color='steelblue')
            axes[0, 1].set_xlabel('Alpha (Mixing Coefficient)', fontsize=12)
            axes[0, 1].set_ylabel('Average GT Token Rank', fontsize=12)
            axes[0, 1].set_title('Injection Effectiveness vs Alpha', fontsize=14)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].invert_yaxis()

            # Plot 3: Injection effectiveness by layer
            layer_perf = valid_inj.groupby('injection_layer')['gt_rank_after_injection'].mean()
            axes[1, 0].bar(layer_perf.index.astype(str), layer_perf.values, color='mediumseagreen')
            axes[1, 0].set_xlabel('Injection Layer', fontsize=12)
            axes[1, 0].set_ylabel('Average GT Token Rank', fontsize=12)
            axes[1, 0].set_title('Injection Effectiveness by Layer', fontsize=14)
            axes[1, 0].grid(axis='y', alpha=0.3)

    # Plot 4: Error entropy distribution
    if os.path.exists(error_analysis_path):
        error_entropies = []
        for _, row in df_errors.iterrows():
            detail = row.get('details', None)
            d = _safe_parse_details(detail)
            ent = d.get('entropy')
            if ent is not None and row.get('has_error', False):
                error_entropies.append(ent)

        if error_entropies:
            axes[1, 1].hist(error_entropies, bins=20, alpha=0.8, color='steelblue',
                             edgecolor='white', linewidth=0.5)
            axes[1, 1].set_xlabel('Entropy (nats)', fontsize=11)
            axes[1, 1].set_ylabel('Count', fontsize=11)
            axes[1, 1].set_title('Error Sample Entropy Distribution', fontsize=13)
            # Annotate median
            med_val = float(np.median(error_entropies))
            mean_val = float(np.mean(error_entropies))
            axes[1, 1].axvline(x=med_val, color='red', linestyle='--', linewidth=1.2)
            axes[1, 1].text(0.97, 0.95,
                             f'n={len(error_entropies)}\nmedian={med_val:.3f}\nmean={mean_val:.3f}',
                             transform=axes[1, 1].transAxes, ha='right', va='top',
                             fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient entropy data', ha='center', va='center',
                             transform=axes[1, 1].transAxes, fontsize=14, color='gray')

    plt.tight_layout()
    summary_path = os.path.join(output_dir, "experiment_summary.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary plot to {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 8: Visualization")
    parser.add_argument("--num_cases", type=int, default=10)
    parser.add_argument("--skip_individual", action="store_true")
    parser.add_argument("--small_model", type=str, default="qwen1.5B",
                        help="Small model key for dynamic path resolution")
    parser.add_argument("--large_model", type=str, default="qwen7B",
                        help="Large model key for dynamic path resolution")
    args = parser.parse_args()

    print("=" * 80)
    print("Step 8: Visualization")
    print("=" * 80)

    # Resolve paths: try dynamic model-pair paths first, fall back to static config
    error_analysis_path = get_error_analysis_path(args.small_model, args.large_model)
    injection_results_path = get_injection_results_path(args.small_model, args.large_model)
    cka_output_dir = ANALYSIS_CONFIG["cka_output_dir"]

    # If dynamic paths don't exist, fall back to static config paths
    if not os.path.exists(error_analysis_path):
        error_analysis_path = ANALYSIS_CONFIG["error_analysis_output"]
    if not os.path.exists(injection_results_path):
        injection_results_path = INJECTION_CONFIG["results_output"]

    print(f"  Error analysis path: {error_analysis_path}  (exists: {os.path.exists(error_analysis_path)})")
    print(f"  Injection results path: {injection_results_path}  (exists: {os.path.exists(injection_results_path)})")

    # Visualize B-ball cases
    if not args.skip_individual and os.path.exists(error_analysis_path):
        visualize_b_ball_cases(error_analysis_path, num_cases=args.num_cases)

    # Visualize injection results
    if not args.skip_individual and os.path.exists(injection_results_path):
        visualize_injection_results(injection_results_path, num_cases=args.num_cases)

    # Summary plots
    create_summary_plots(error_analysis_path, injection_results_path, cka_output_dir)

    print(f"\nPlots saved to: {ANALYSIS_CONFIG['prob_plots_dir']}")
    print("=" * 80)
    print("Visualization Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()