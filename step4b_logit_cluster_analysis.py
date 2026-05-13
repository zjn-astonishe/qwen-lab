"""
Step 4b: Logit Semantic Cluster Analysis (Observation Experiment — Phase I)

Proves that when a small model (1.5B) makes an error, it does NOT guess randomly.
Instead, it assigns probability to tokens within the SAME semantic "macro-category"
as the correct answer.  This is the "right drawer, wrong file" hypothesis.

Methodology:
  1. For each error sample (small model wrong, 7B correct), extract the Top-20 logits
     from the answer step of ALL three models.
  2. Decode the Top-20 token IDs into text and classify each token into a semantic
     category (option letter, number, format token, domain term, etc.).
  3. Check whether the GT token is present in the small model's Top-20.
  4. Compare the semantic distributions of Top-20 across models.

Input:  step2 .pt files (top_k_info), step3 three_model_comparison.csv
Output: logit_cluster_analysis.json, 3× PNG visualizations

No additional inference required — all data is already saved by step2.
"""

import os
import re
import json
import argparse
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from collections import Counter

from config import MODELS, ANALYSIS_CONFIG, MEMORY_CONFIG
from qa_utils import (
    get_gt_answer, get_answer_type, get_clean_answer, compare_answers,
    get_answer_token_id, find_answer_step,
)
from utils import load_model_outputs, save_json


# ---------------------------------------------------------------------------
# Token semantic classification
# ---------------------------------------------------------------------------

# Precompiled regex patterns for token category classification
_PAT_SINGLE_LETTER = re.compile(r'^[A-E]$')
_PAT_NUMBER = re.compile(r'^[\d,]+(?:\.\d+)?$')
_PAT_NEWLINE = re.compile(r'^\s*$|^<\|.*?\|>$|^(\\n)+$')
_PAT_SPECIAL = re.compile(r'^[^\w\s]$|^[.,:;!?()\[\]{}\'"\\]$')
_PAT_DOMAIN_WORD = re.compile(r'^[a-zA-Z]{3,}$')


def classify_token(text: str, answer_type: str) -> str:
    """Classify a decoded token into a semantic category.

    Categories:
      - "option_letter": Single uppercase letter A-E (MC answer options)
      - "number": Numeric token (digits, commas, decimals)
      - "format_token": Special/formatting tokens (newlines, control tokens)
      - "domain_term": Multi-character alphabetic strings (subject words)
      - "symbol": Punctuation and special characters
      - "other": Anything else
    """
    if not text:
        return "other"

    text_stripped = text.strip()

    # Special / control tokens
    if _PAT_NEWLINE.match(text_stripped) or text_stripped.startswith("<|"):
        return "format_token"

    # Option letters (only meaningful for MC tasks)
    if answer_type == "multiple_choice" and _PAT_SINGLE_LETTER.match(text_stripped):
        return "option_letter"

    # Numbers
    if _PAT_NUMBER.match(text_stripped.replace(",", "").replace("$", "")):
        return "number"

    # Punctuation / symbols
    if _PAT_SPECIAL.match(text_stripped):
        return "symbol"

    # Domain words (multi-character alphabetic strings)
    if _PAT_DOMAIN_WORD.match(text_stripped):
        return "domain_term"

    return "other"


# ---------------------------------------------------------------------------
# Core analysis logic
# ---------------------------------------------------------------------------

def extract_top_k_info(
    model_output: Dict[str, Any],
    tokenizer,
    answer_step: int,
    k: int = 20,
) -> Optional[Dict[str, Any]]:
    """Extract Top-k tokens with decoded text and semantic categories.

    Args:
        model_output: Step2 output dict (contains top_k_info).
        tokenizer: HuggingFace tokenizer for decoding.
        answer_step: Generation step index to analyze.
        k: Number of top tokens to extract.

    Returns:
        Dict with:
          - "top_tokens": list of (token_id, decoded_text, prob, category)
          - "gt_in_top_k": bool
          - "gt_rank": int (-1 if not found)
          - "gt_prob": float
          - "top1_prob": float
          - "top1_token": str
    """
    top_k_info = model_output.get("top_k_info", [])
    if answer_step >= len(top_k_info):
        return None

    entry = top_k_info[answer_step]
    probs = entry.get("probs")
    indices = entry.get("indices")

    if probs is None or indices is None:
        return None

    k = min(k, len(probs))
    probs = probs[:k]
    indices = indices[:k]

    # Decode tokens
    top_tokens = []
    for i in range(k):
        token_id = int(indices[i].item()) if torch.is_tensor(indices[i]) else int(indices[i])
        prob = float(probs[i].item()) if torch.is_tensor(probs[i]) else float(probs[i])
        decoded = tokenizer.decode([token_id]).strip()
        top_tokens.append({
            "token_id": token_id,
            "text": decoded,
            "prob": prob,
        })

    # Ground truth check
    ground_truth = model_output.get("ground_truth", {})
    gt_answer = get_gt_answer(ground_truth)
    answer_type = get_answer_type(ground_truth)
    gt_token_id = get_answer_token_id(tokenizer, gt_answer, answer_type) if gt_answer else None

    gt_in_top_k = False
    gt_rank = -1
    gt_prob = 0.0

    if gt_token_id is not None:
        for i, t in enumerate(top_tokens):
            if t["token_id"] == gt_token_id:
                gt_in_top_k = True
                gt_rank = i + 1
                gt_prob = t["prob"]
                break

    return {
        "top_tokens": top_tokens,
        "gt_in_top_k": gt_in_top_k,
        "gt_rank": gt_rank,
        "gt_prob": gt_prob,
        "top1_prob": float(top_tokens[0]["prob"]) if top_tokens else 0.0,
        "top1_text": top_tokens[0]["text"] if top_tokens else "",
        "answer_type": answer_type,
    }


def analyze_semantic_distribution(
    top_tokens: List[Dict],
    answer_type: str,
) -> Dict[str, Any]:
    """Compute semantic category distribution for a list of top tokens."""
    categories = []
    for t in top_tokens:
        cat = classify_token(t["text"], answer_type)
        categories.append(cat)

    counter = Counter(categories)
    total = len(categories)
    distribution = {cat: count / total for cat, count in counter.items()}

    return {
        "categories": categories,
        "counter": dict(counter),
        "distribution": distribution,
    }


def find_answer_step_for_output(
    model_output: Dict[str, Any],
    tokenizer,
    answer_type: str,
) -> Tuple[int, Optional[int]]:
    """Find the generation step where the model outputs its answer."""
    generated_ids = model_output.get("generated_ids")
    input_ids = model_output.get("input_ids")
    if generated_ids is None or input_ids is None:
        return 0, None

    input_len = input_ids.shape[0] if input_ids.dim() == 1 else input_ids.shape[1]
    pred_answer, _ = get_clean_answer(model_output)

    if not pred_answer:
        return 0, None

    return find_answer_step(generated_ids, input_len, tokenizer, pred_answer, answer_type)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_semantic_distribution_comparison(
    results_by_model: Dict[str, List[Dict]],
    output_dir: str,
):
    """Plot semantic distribution comparison across three models.

    Two panels:
      (a) Grouped bar chart — semantic category proportions per model (error samples)
      (b) GT-in-Top-k cumulative ratio curve
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    all_categories = ["option_letter", "number", "domain_term", "format_token", "symbol", "other"]
    cat_labels = {
        "option_letter": "Option\nLetter",
        "number": "Number",
        "domain_term": "Domain\nTerm",
        "format_token": "Format\nToken",
        "symbol": "Symbol",
        "other": "Other",
    }
    model_colors = {"1.5B": "#2196F3", "3B": "#4CAF50", "7B": "#FF9800"}

    # --- (a) Grouped bar chart ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Aggregate distributions
    agg_dists = {}
    for model_name, results in results_by_model.items():
        dist_sum = {cat: 0.0 for cat in all_categories}
        count = 0
        for r in results:
            if r is None or r.get("semantic_dist") is None:
                continue
            for cat, val in r["semantic_dist"]["distribution"].items():
                dist_sum[cat] += val
            count += 1
        if count > 0:
            agg_dists[model_name] = {cat: dist_sum[cat] / count for cat in all_categories}
        else:
            agg_dists[model_name] = {cat: 0.0 for cat in all_categories}

    x = np.arange(len(all_categories))
    width = 0.25
    for i, model_name in enumerate(["1.5B", "3B", "7B"]):
        if model_name not in agg_dists:
            continue
        vals = [agg_dists[model_name].get(cat, 0) for cat in all_categories]
        axes[0].bar(x + i * width, vals, width, label=model_name,
                    color=model_colors[model_name], alpha=0.85, edgecolor="white")

    axes[0].set_xlabel("Semantic Category")
    axes[0].set_ylabel("Proportion in Top-20")
    axes[0].set_title("(a) Semantic Distribution of Top-20 Tokens\n(Error Samples)")
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([cat_labels[c] for c in all_categories], fontsize=9)
    axes[0].legend(loc="best", fontsize=10)
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", alpha=0.3)

    # --- (b) GT-in-Top-k cumulative ratio ---
    for model_name in ["1.5B", "3B", "7B"]:
        if model_name not in results_by_model:
            continue
        ratios = []
        for r in results_by_model[model_name]:
            if r is None:
                continue
            ratios.append(1.0 if r.get("gt_in_top_k") else 0.0)
        if not ratios:
            continue
        # Cumulative mean (sorted by ratio, which is binary here)
        cumulative = np.cumsum(sorted(ratios, reverse=True)) / np.arange(1, len(ratios) + 1)
        axes[1].plot(range(1, len(cumulative) + 1), cumulative,
                     label=model_name, color=model_colors[model_name],
                     linewidth=2, marker="o", markersize=4)

    axes[1].set_xlabel("Number of Samples (sorted)")
    axes[1].set_ylabel("Cumulative GT-in-Top-20 Ratio")
    axes[1].set_title("(b) GT Token Present in Top-20\n(Cumulative)")
    axes[1].legend(loc="best", fontsize=10)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "logit_cluster_semantic_distribution.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_topk_rank_distribution(
    results_by_model: Dict[str, List[Dict]],
    output_dir: str,
):
    """Plot GT rank distribution across models.

    Shows where GT token ranks in the probability distribution.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    model_colors = {"1.5B": "#2196F3", "3B": "#4CAF50", "7B": "#FF9800"}

    for model_name in ["1.5B", "3B", "7B"]:
        if model_name not in results_by_model:
            continue
        ranks = []
        for r in results_by_model[model_name]:
            if r is None:
                continue
            rank = r.get("gt_rank", -1)
            if rank > 0:
                ranks.append(rank)

        if not ranks:
            continue

        # Histogram (log-scaled x-axis)
        bins = [1, 2, 3, 5, 10, 20, 50, 100]
        ax.hist(ranks, bins=bins, label=model_name, color=model_colors[model_name],
                alpha=0.6, edgecolor="white", linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("GT Token Rank (log scale)")
    ax.set_ylabel("Number of Samples")
    ax.set_title("GT Token Rank Distribution in Top-100 Logits\n(Error Samples)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "logit_cluster_gt_rank_distribution.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_same_category_ratio(
    results_by_model: Dict[str, List[Dict]],
    output_dir: str,
):
    """Plot the ratio of Top-20 tokens that share the GT's semantic category."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Box plot: proportion of same-category tokens
    data_by_model = {}
    for model_name in ["1.5B", "3B", "7B"]:
        if model_name not in results_by_model:
            continue
        proportions = []
        for r in results_by_model[model_name]:
            if r is None or r.get("same_category_ratio") is None:
                continue
            proportions.append(r["same_category_ratio"])
        if proportions:
            data_by_model[model_name] = proportions

    if data_by_model:
        model_colors = {"1.5B": "#2196F3", "3B": "#4CAF50", "7B": "#FF9800"}
        positions = []
        data_list = []
        labels = []
        colors = []
        for i, (mn, vals) in enumerate(data_by_model.items()):
            positions.append(i)
            data_list.append(vals)
            labels.append(mn)
            colors.append(model_colors[mn])

        bp = axes[0].boxplot(data_list, positions=positions, patch_artist=True,
                              labels=labels, widths=0.5)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    axes[0].set_ylabel("Proportion of Same-Category Tokens in Top-20")
    axes[0].set_title("(a) Same-Category Ratio\n(Higher = more 'focused' on correct drawer)")
    axes[0].grid(axis="y", alpha=0.3)

    # (b) Scatter: same_category_ratio vs gt_rank
    model_colors = {"1.5B": "#2196F3", "3B": "#4CAF50", "7B": "#FF9800"}
    for model_name in ["1.5B", "3B", "7B"]:
        if model_name not in results_by_model:
            continue
        x_vals = []
        y_vals = []
        for r in results_by_model[model_name]:
            if r is None:
                continue
            rank = r.get("gt_rank", -1)
            ratio = r.get("same_category_ratio")
            if rank > 0 and ratio is not None:
                x_vals.append(rank)
                y_vals.append(ratio)
        if x_vals:
            axes[1].scatter(x_vals, y_vals, label=model_name,
                           color=model_colors[model_name], alpha=0.5, s=40)

    axes[1].set_xlabel("GT Token Rank")
    axes[1].set_ylabel("Same-Category Ratio")
    axes[1].set_title("(b) Same-Category Ratio vs GT Rank")
    axes[1].legend(loc="best", fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "logit_cluster_same_category.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    num_samples: int,
    top_k: int = 20,
    device: str = "cuda",
    output_dir: str = "experiment_results/logit_cluster",
):
    """Run the full logit semantic cluster analysis."""
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

    # Filter: 7B correct, 1.5B wrong (injection targets)
    error_samples = df_cmp[(df_cmp["correct_7B"]) & (~df_cmp["correct_1.5B"])].copy()
    print(f"\nFound {len(error_samples)} injection targets (7B correct, 1.5B wrong)")

    if len(error_samples) == 0:
        print("  No error samples to analyze.")
        return

    # Load model outputs
    print(f"Loading model outputs to {device}...")
    outputs_1_5B = load_model_outputs("qwen1.5B", num_samples, map_location=device)
    outputs_3B = load_model_outputs("qwen3B", num_samples, map_location=device)
    outputs_7B = load_model_outputs("qwen7B", num_samples, map_location=device)

    # Analyze each error sample
    results_by_model = {"1.5B": [], "3B": [], "7B": []}

    for _, row in tqdm(error_samples.iterrows(), total=len(error_samples),
                       desc="Analyzing logit clusters"):
        idx = int(row["sample_idx"])
        gt_answer = str(row["gt_answer"]).strip()
        dataset = row.get("dataset", "unknown")

        for model_name, outputs in [("1.5B", outputs_1_5B),
                                     ("3B", outputs_3B),
                                     ("7B", outputs_7B)]:
            if idx >= len(outputs) or outputs[idx] is None:
                results_by_model[model_name].append(None)
                continue

            out = outputs[idx]
            answer_type = get_answer_type(out.get("ground_truth", {}))

            # Find answer step
            answer_step, _ = find_answer_step_for_output(out, tokenizer, answer_type)

            # Extract top-k info
            info = extract_top_k_info(out, tokenizer, answer_step, k=top_k)

            if info is None:
                results_by_model[model_name].append(None)
                continue

            # Classify tokens
            for t in info["top_tokens"]:
                t["category"] = classify_token(t["text"], answer_type)

            # Compute semantic distribution
            semantic_dist = analyze_semantic_distribution(info["top_tokens"], answer_type)

            # Compute same-category ratio
            gt_cat = classify_token(
                tokenizer.decode([get_answer_token_id(tokenizer, gt_answer, answer_type)]
                                ).strip() if gt_answer else "",
                answer_type,
            ) if gt_answer else "other"

            same_cat_count = sum(
                1 for t in info["top_tokens"] if t["category"] == gt_cat
            )
            same_category_ratio = same_cat_count / len(info["top_tokens"]) if info["top_tokens"] else 0.0

            result = {
                "sample_idx": idx,
                "sample_id": row.get("sample_id", f"sample_{idx}"),
                "dataset": dataset,
                "gt_answer": gt_answer,
                "gt_category": gt_cat,
                "answer_step": answer_step,
                "gt_in_top_k": info["gt_in_top_k"],
                "gt_rank": info["gt_rank"],
                "gt_prob": info["gt_prob"],
                "top1_prob": info["top1_prob"],
                "top1_text": info["top1_text"],
                "top_tokens": info["top_tokens"],
                "semantic_dist": semantic_dist,
                "same_category_ratio": same_category_ratio,
            }
            results_by_model[model_name].append(result)

    # --- Aggregate statistics ---
    print(f"\n{'=' * 80}")
    print("Logit Cluster Analysis Results")
    print(f"{'=' * 80}")

    summary = {}
    for model_name in ["1.5B", "3B", "7B"]:
        valid = [r for r in results_by_model[model_name] if r is not None]
        n = len(valid)
        if n == 0:
            print(f"\n  {model_name}: no valid results")
            continue

        gt_in_top_k_count = sum(1 for r in valid if r["gt_in_top_k"])
        same_cat_ratios = [r["same_category_ratio"] for r in valid if r["same_category_ratio"] is not None]
        gt_ranks = [r["gt_rank"] for r in valid if r["gt_rank"] > 0]
        gt_probs = [r["gt_prob"] for r in valid if r["gt_prob"] > 0]

        print(f"\n  {model_name} ({n} error samples):")
        print(f"    GT in Top-{top_k}: {gt_in_top_k_count}/{n} = {gt_in_top_k_count/n*100:.1f}%")
        print(f"    Same-category ratio (mean): {np.mean(same_cat_ratios):.3f}")
        if gt_ranks:
            print(f"    GT rank: median={np.median(gt_ranks):.0f}, "
                  f"mean={np.mean(gt_ranks):.1f}, "
                  f"range=[{min(gt_ranks)}, {max(gt_ranks)}]")
        if gt_probs:
            print(f"    GT prob: mean={np.mean(gt_probs):.4f}, "
                  f"median={np.median(gt_probs):.4f}")

        # Aggregate semantic distribution
        agg_dist = {}
        for r in valid:
            for cat, val in r["semantic_dist"]["distribution"].items():
                agg_dist[cat] = agg_dist.get(cat, 0) + val
        agg_dist = {cat: val / n for cat, val in agg_dist.items()}
        print(f"    Semantic distribution: {agg_dist}")

        summary[model_name] = {
            "n_samples": n,
            "gt_in_top_k_ratio": gt_in_top_k_count / n,
            "mean_same_category_ratio": float(np.mean(same_cat_ratios)),
            "median_gt_rank": float(np.median(gt_ranks)) if gt_ranks else -1,
            "mean_gt_rank": float(np.mean(gt_ranks)) if gt_ranks else -1,
            "mean_gt_prob": float(np.mean(gt_probs)) if gt_probs else 0.0,
            "semantic_distribution": agg_dist,
        }

    # --- Hypothesis test: "same category" for 1.5B ---
    valid_1_5B = [r for r in results_by_model["1.5B"] if r is not None]
    if valid_1_5B:
        high_ratio = sum(1 for r in valid_1_5B if r.get("same_category_ratio", 0) > 0.3)
        print(f"\n  *** Hypothesis Test: 1.5B Top-20 shares GT's category ***")
        print(f"    Samples with >30% same-category tokens: {high_ratio}/{len(valid_1_5B)} "
              f"= {high_ratio/len(valid_1_5B)*100:.1f}%")
        print(f"    (If >50%, this supports the 'right drawer, wrong file' hypothesis)")

    # --- Save results ---
    save_path = os.path.join(output_dir, "logit_cluster_analysis.json")
    save_json(summary, save_path)
    print(f"\n  Results saved to: {save_path}")

    # --- Generate plots ---
    print("\n  Generating plots...")
    plot_semantic_distribution_comparison(results_by_model, output_dir)
    plot_topk_rank_distribution(results_by_model, output_dir)
    plot_same_category_ratio(results_by_model, output_dir)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 4b: Logit Semantic Cluster Analysis (Phase I)"
    )
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--top_k", type=int, default=20,
                        help="Number of top logits to analyze per sample")
    parser.add_argument("--output_dir", type=str,
                        default="experiment_results/logit_cluster")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for tensor ops (default: cuda if available)")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print("=" * 80)
    print("Step 4b: Logit Semantic Cluster Analysis (Phase I)")
    print("  Proving: small model errors are 'right drawer, wrong file'")
    print("=" * 80)

    run_analysis(
        num_samples=args.num_samples,
        top_k=args.top_k,
        device=device,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 80)
    print("Step 4b Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()