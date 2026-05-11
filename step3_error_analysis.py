"""
Step 3: Error Analysis (V3 — QA tasks, optimized)

Analyze errors from small models (1.5B, 3B) compared to the large model (7B)
on QA tasks (GSM8K, ARC-Challenge, MMLU).

Optimized:
  - All answer extraction / comparison logic delegated to qa_utils (no duplication)
  - eval() replaced with ast.literal_eval via qa_utils.parse_details()
  - Uses utils.load_model_outputs() for centralized loading
  - Cleaner per-dataset breakdown without duplicate iteration
  - Three-model comparison uses qa_utils for consistency
"""

import os
import torch
import pandas as pd
import argparse
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
from scipy.stats import entropy

from config import MODELS, ANALYSIS_CONFIG, MEMORY_CONFIG, DATA_CONFIG
from qa_utils import (
    extract_answer, compare_answers, get_gt_answer, get_answer_type,
    get_answer_token_id, find_answer_step, normalize_numerical_answer,
    parse_details,
)
from utils import load_model_outputs


# ---------------------------------------------------------------------------
# Probability analysis helpers
# ---------------------------------------------------------------------------

def calculate_top_k_overlap(
    probs1: torch.Tensor, probs2: torch.Tensor, k: int = 20,
    indices1: Optional[torch.Tensor] = None,
    indices2: Optional[torch.Tensor] = None,
) -> float:
    """Jaccard similarity between top-k token sets."""
    if indices1 is not None and indices2 is not None:
        set1 = set(indices1.tolist()[:k])
        set2 = set(indices2.tolist()[:k])
    else:
        set1 = set(torch.topk(probs1, k=min(k, probs1.size(0))).indices.tolist())
        set2 = set(torch.topk(probs2, k=min(k, probs2.size(0))).indices.tolist())

    union = len(set1 | set2)
    return len(set1 & set2) / union if union > 0 else 0.0


def calculate_entropy(probs: torch.Tensor) -> float:
    """Shannon entropy of a probability distribution."""
    p = probs.cpu().numpy()
    p = p[p > 0]
    return float(entropy(p))


def _get_probs_and_indices(
    output: Dict[str, Any], step: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract (probs, indices) at a given step from model output."""
    # Full logits path
    if len(output.get("probs_per_step", [])) > step:
        return output["probs_per_step"][step], None

    # Top-k path
    top_k_info = output.get("top_k_info", [])
    if len(top_k_info) > step:
        entry = top_k_info[step]
        return entry["probs"], entry["indices"]

    return None, None


# ---------------------------------------------------------------------------
# B-ball dilemma identification
# ---------------------------------------------------------------------------

def identify_b_ball_dilemma(
    small_probs: torch.Tensor,
    large_probs: torch.Tensor,
    gt_token_id: int,
    predicted_token_id: int,
    small_indices: Optional[torch.Tensor] = None,
    large_indices: Optional[torch.Tensor] = None,
    top_k: int = 20,
    saved_window: int = 100,
    entropy_threshold: float = 2.5,
    topk_overlap_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Determine whether an error qualifies as a 'B-ball dilemma'.

    Criteria:
      1. predicted_token != gt_token
      2. gt_token is in small model's top-k (for B-ball) or at least in saved window
      3. High entropy (flat distribution) in small model
      4. Moderate top-k overlap with large model
    """
    result = {
        "is_b_ball_dilemma": False,
        "gt_in_topk": False,
        "gt_in_saved_window": False,
        "gt_rank": -1,
        "entropy": 0.0,
        "topk_overlap": 0.0,
        "predicted_prob": 0.0,
        "gt_prob": 0.0,
    }

    result["entropy"] = calculate_entropy(small_probs)

    # --- Find GT rank and probability in small-model distribution ---
    if small_indices is not None:
        idx_list = small_indices.tolist() if isinstance(small_indices, torch.Tensor) else list(small_indices)
        if gt_token_id in idx_list:
            pos = idx_list.index(gt_token_id)
            result["gt_in_saved_window"] = True
            result["gt_rank"] = pos + 1
            result["gt_prob"] = float(small_probs[pos])
            if result["gt_rank"] <= top_k:
                result["gt_in_topk"] = True
        if predicted_token_id in idx_list:
            result["predicted_prob"] = float(small_probs[idx_list.index(predicted_token_id)])
    else:
        if 0 <= gt_token_id < len(small_probs):
            sorted_idx = torch.argsort(small_probs, descending=True)
            positions = (sorted_idx == gt_token_id).nonzero(as_tuple=True)
            if len(positions[0]) > 0:
                result["gt_rank"] = int(positions[0][0].item()) + 1
                result["gt_in_topk"] = result["gt_rank"] <= top_k
                result["gt_in_saved_window"] = result["gt_rank"] <= saved_window
                result["gt_prob"] = float(small_probs[gt_token_id])
        result["predicted_prob"] = float(small_probs[predicted_token_id]) \
            if 0 <= predicted_token_id < len(small_probs) else 0.0

    # --- Top-k overlap with large model ---
    result["topk_overlap"] = calculate_top_k_overlap(
        small_probs, large_probs, k=top_k,
        indices1=small_indices, indices2=large_indices,
    )

    # --- Sparse data diagnostics ---
    result["is_sparse_topk"] = (small_indices is not None)
    if result["is_sparse_topk"] and not result["gt_in_saved_window"]:
        result["gt_rank"] = -2  # sentinel: "outside saved window"

    # --- Final decision ---
    if (result["gt_in_topk"]
            and result["entropy"] > entropy_threshold
            and result["topk_overlap"] > topk_overlap_threshold):
        result["is_b_ball_dilemma"] = True

    return result


# ---------------------------------------------------------------------------
# Per-sample error analysis
# ---------------------------------------------------------------------------

def analyze_sample_errors(
    sample_idx: int,
    small_model_output: Dict[str, Any],
    large_model_output: Dict[str, Any],
    tokenizer,
) -> Dict[str, Any]:
    """Analyze errors for a single sample comparing small vs large model."""
    analysis = {
        "sample_idx": sample_idx,
        "sample_id": small_model_output.get("sample_id", f"sample_{sample_idx}"),
        "has_error": False,
        "error_type": None,
        "is_b_ball_dilemma": False,
        "details": {},
        "skip_reason": None,
    }

    # --- Extract and compare answers ---
    generated_text = small_model_output.get("generated_text", "")
    ground_truth = small_model_output.get("ground_truth", {})
    answer_type = get_answer_type(ground_truth)
    predicted_answer = extract_answer(generated_text, answer_type)
    comparison = compare_answers(predicted_answer, ground_truth)

    analysis["has_error"] = comparison["has_error"]
    analysis["error_type"] = comparison["error_subtype"]
    analysis["comparison"] = comparison

    if not comparison["has_error"]:
        return analysis

    # --- Prepare for B-ball analysis ---
    pred_answer = comparison.get("predicted_answer")
    gt_answer = comparison.get("gt_answer")

    if not pred_answer or not gt_answer:
        analysis["skip_reason"] = f"no pred or gt answer (pred={pred_answer}, gt={gt_answer})"
        return analysis

    pred_token_id = get_answer_token_id(tokenizer, pred_answer, answer_type)
    gt_token_id = get_answer_token_id(tokenizer, gt_answer, answer_type)

    if pred_token_id is None or gt_token_id is None:
        analysis["skip_reason"] = f"token id missing (pred={pred_token_id}, gt={gt_token_id})"
        return analysis

    generated_ids = small_model_output.get("generated_ids")
    input_ids = small_model_output.get("input_ids")
    if generated_ids is None or input_ids is None:
        analysis["skip_reason"] = "missing generated_ids or input_ids"
        return analysis

    input_len = input_ids.shape[0] if input_ids.dim() == 1 else input_ids.shape[1]

    # Find the answer step
    answer_step, _ = find_answer_step(
        generated_ids, input_len, tokenizer, pred_answer, answer_type
    )

    # Get probability distributions at that step
    small_probs, small_indices = _get_probs_and_indices(small_model_output, answer_step)
    large_probs, large_indices = _get_probs_and_indices(large_model_output, answer_step)

    if small_probs is None or large_probs is None:
        small_steps = len(small_model_output.get("top_k_info",
                                                 small_model_output.get("probs_per_step", [])))
        large_steps = len(large_model_output.get("top_k_info",
                                                 large_model_output.get("probs_per_step", [])))
        analysis["skip_reason"] = (
            f"probs missing at answer_step={answer_step} "
            f"(small has {small_steps} steps, large has {large_steps} steps)"
        )
        return analysis

    saved_window = MEMORY_CONFIG.get("save_top_k_logits", 100)
    b_ball_info = identify_b_ball_dilemma(
        small_probs, large_probs,
        gt_token_id=gt_token_id,
        predicted_token_id=pred_token_id,
        small_indices=small_indices,
        large_indices=large_indices,
        top_k=ANALYSIS_CONFIG["top_k_overlap"],
        saved_window=saved_window,
        entropy_threshold=ANALYSIS_CONFIG["entropy_threshold"],
    )

    analysis["is_b_ball_dilemma"] = b_ball_info["is_b_ball_dilemma"]
    analysis["answer_step"] = answer_step
    analysis["pred_answer"] = pred_answer
    analysis["gt_answer"] = gt_answer
    analysis["details"] = b_ball_info

    return analysis


# ---------------------------------------------------------------------------
# Three-model comparison
# ---------------------------------------------------------------------------

def compare_three_models(
    outputs_1_5B: List[Optional[Dict]],
    outputs_3B: List[Optional[Dict]],
    outputs_7B: List[Optional[Dict]],
    num_samples: int,
    tokenizer,
) -> None:
    """Compare prediction distributions across all three models."""
    results = []

    for i in range(num_samples):
        if outputs_1_5B[i] is None or outputs_3B[i] is None or outputs_7B[i] is None:
            continue

        gt = outputs_1_5B[i].get("ground_truth", {})
        answer_type = get_answer_type(gt)
        gt_answer = get_gt_answer(gt)
        if gt_answer is None:
            continue

        answers = {}
        for model_name, outputs in [("1.5B", outputs_1_5B), ("3B", outputs_3B), ("7B", outputs_7B)]:
            text = outputs[i].get("generated_text", "")
            answers[model_name] = extract_answer(text, answer_type)

        correct = {}
        for model_name, pred in answers.items():
            if pred is None:
                correct[model_name] = False
            elif answer_type == "numerical":
                pn, gn = normalize_numerical_answer(pred, gt_answer)
                correct[model_name] = (pn == gn)
            else:
                correct[model_name] = (pred.strip().upper() == gt_answer.strip().upper())

        results.append({
            "sample_idx": i,
            "sample_id": outputs_1_5B[i].get("sample_id", f"sample_{i}"),
            "dataset": gt.get("dataset", "unknown"),
            "gt_answer": gt_answer,
            "pred_1.5B": answers["1.5B"],
            "pred_3B": answers["3B"],
            "pred_7B": answers["7B"],
            "correct_1.5B": correct["1.5B"],
            "correct_3B": correct["3B"],
            "correct_7B": correct["7B"],
        })

    if not results:
        print("\n  No evaluable samples for three-model comparison.")
        return

    df = pd.DataFrame(results)
    n = len(df)

    print(f"\n{'=' * 80}")
    print("Three-Model Comparison Summary")
    print(f"{'=' * 80}")
    print(f"\nPer-model accuracy ({n} evaluable samples):")
    for m in ["1.5B", "3B", "7B"]:
        acc = df[f"correct_{m}"].sum() / n * 100
        print(f"  {m:>4s}: {df[f'correct_{m}'].sum():>4d}/{n} = {acc:.1f}%")

    # Per-dataset accuracy
    print(f"\nPer-dataset accuracy:")
    for ds_name in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds_name]
        parts = []
        for m in ["1.5B", "3B", "7B"]:
            acc = sub[f"correct_{m}"].sum() / len(sub) * 100
            parts.append(f"{m}={acc:.1f}%")
        print(f"  {ds_name:>20s}: {'  '.join(parts)} ({len(sub)} samples)")

    # Consensus matrices
    print(f"\nConsensus matrices:")
    for pair_name, m1, m2 in [("1.5B vs 3B", "1.5B", "3B"),
                                ("1.5B vs 7B", "1.5B", "7B"),
                                ("3B vs 7B", "3B", "7B")]:
        bc = ((df[f"correct_{m1}"]) & (df[f"correct_{m2}"])).sum()
        m1o = (df[f"correct_{m1}"] & ~df[f"correct_{m2}"]).sum()
        m2o = (~df[f"correct_{m1}"] & df[f"correct_{m2}"]).sum()
        bw = (~df[f"correct_{m1}"] & ~df[f"correct_{m2}"]).sum()
        print(f"  {pair_name}: both_ok={bc}, {m1}_only={m1o}, {m2}_only={m2o}, both_wrong={bw}")

    # Injection target analysis
    for small_m in ["1.5B", "3B"]:
        inj_targets = df[(df["correct_7B"]) & (~df[f"correct_{small_m}"])]
        print(f"\n  {small_m} vs 7B injection targets (7B ok, {small_m} wrong): {len(inj_targets)}")
        if len(inj_targets) > 0:
            for _, row in inj_targets.head(10).iterrows():
                print(f"    {row['sample_id']:40s}  "
                      f"GT={row['gt_answer']:10s}  "
                      f"{small_m}={str(row[f'pred_{small_m}']):10s}  "
                      f"7B={str(row['pred_7B']):10s}")

    # Save to CSV
    csv_path = ANALYSIS_CONFIG["three_model_comparison"]
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"\nThree-model comparison saved to: {csv_path}")


# ---------------------------------------------------------------------------
# Per-dataset breakdown helper
# ---------------------------------------------------------------------------

def _build_per_dataset_accuracy(
    results: List[Dict],
    outputs: List[Optional[Dict]],
    model_label: str = "small",
) -> None:
    """Print per-dataset accuracy breakdown for the small model."""
    ds_correct = {}
    ds_total = {}
    for i in range(len(results)):
        if outputs[i] is None:
            continue
        gt = outputs[i].get("ground_truth", {})
        ds_name = gt.get("dataset", "unknown") if isinstance(gt, dict) else "unknown"
        has_error = results[i]["has_error"]
        ds_total[ds_name] = ds_total.get(ds_name, 0) + 1
        if not has_error:
            ds_correct[ds_name] = ds_correct.get(ds_name, 0) + 1

    for ds_name in sorted(ds_total.keys()):
        n_ds = ds_total[ds_name]
        n_ok = ds_correct.get(ds_name, 0)
        print(f"  {ds_name:>20s}: {n_ok}/{n_ds} correct ({n_ok / n_ds * 100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 3: Error Analysis (QA tasks)")
    parser.add_argument("--small_model", type=str, default=None)
    parser.add_argument("--large_model", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--three_model", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 80)
    print("Step 3: Error Analysis (QA tasks)")
    print("=" * 80)

    # Load tokenizer (shared between Qwen models)
    from transformers import AutoTokenizer
    model_name = MODELS["qwen1.5B"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Determine which pairs to analyze
    if args.small_model and args.large_model:
        pairs = [(args.small_model, args.large_model)]
    else:
        pairs = [("qwen3B", "qwen7B"), ("qwen1.5B", "qwen7B")]

    for small_key, large_key in pairs:
        pair_name = f"{small_key}_vs_{large_key}"
        base_dir = os.path.dirname(ANALYSIS_CONFIG["error_analysis_output"])
        output_path = os.path.join(base_dir, f"error_analysis_{pair_name}.csv")

        print(f"\n{'=' * 80}")
        print(f"Analyzing: {small_key} (small) vs {large_key} (large)")
        print(f"{'=' * 80}")

        small_outputs = load_model_outputs(small_key, args.num_samples)
        large_outputs = load_model_outputs(large_key, args.num_samples)

        results = []
        for i in tqdm(range(args.num_samples), desc=pair_name):
            if small_outputs[i] is None or large_outputs[i] is None:
                continue
            results.append(analyze_sample_errors(i, small_outputs[i], large_outputs[i], tokenizer))

        df = pd.DataFrame(results)
        os.makedirs(base_dir, exist_ok=True)
        df.to_csv(output_path, index=False)

        # --- Large model accuracy check ---
        large_correct_small_wrong = []
        both_wrong = []
        both_correct = []
        large_only_errors = []

        for i in tqdm(range(args.num_samples), desc=f"{large_key} accuracy", leave=False):
            if large_outputs[i] is None or (i >= len(results)):
                continue
            large_text = large_outputs[i].get("generated_text", "")
            large_gt = large_outputs[i].get("ground_truth", {})
            large_pred = extract_answer(large_text, get_answer_type(large_gt))
            large_cmp = compare_answers(large_pred, large_gt)

            if large_cmp["error_subtype"] == "gt_unavailable":
                continue

            large_has_error = large_cmp["has_error"]
            small_has_error = results[i]["has_error"]

            entry = {
                "sample_id": large_outputs[i].get("sample_id", f"sample_{i}"),
                "large_pred": large_pred,
                "large_gt": large_cmp.get("gt_answer"),
                "large_error": large_has_error,
                "dataset": large_gt.get("dataset", "unknown") if isinstance(large_gt, dict) else "unknown",
            }

            if not large_has_error and small_has_error:
                large_correct_small_wrong.append(entry)
            elif large_has_error and small_has_error:
                both_wrong.append(entry)
            elif not large_has_error and not small_has_error:
                both_correct.append(entry)
            else:
                large_only_errors.append(entry)

        # --- Statistics ---
        print(f"\n{'=' * 80}")
        print(f"Error Analysis: {pair_name}")
        print(f"{'=' * 80}")
        print(f"Total analyzed: {len(results)}")

        gt_unavail = df[df["error_type"] == "gt_unavailable"]
        evaluable = df[df["error_type"] != "gt_unavailable"]
        n_errors = int(evaluable["has_error"].sum()) if len(evaluable) > 0 else 0
        n_evaluable = len(evaluable)

        print(f"GT unavailable: {len(gt_unavail)}")
        print(f"Evaluable: {n_evaluable}")
        if n_evaluable > 0:
            print(f"Errors: {n_errors} ({n_errors / n_evaluable * 100:.1f}%)")

        if n_evaluable > 0:
            print(f"\nPer-dataset accuracy ({small_key}):")
            _build_per_dataset_accuracy(results, small_outputs)

        # Large model report
        n_large_evaluable = len(large_correct_small_wrong) + len(both_wrong) + len(both_correct) + len(large_only_errors)
        n_large_errors = len(both_wrong) + len(large_only_errors)
        print(f"\n--- {large_key} Accuracy ---")
        print(f"  Evaluable: {n_large_evaluable}")
        if n_large_evaluable > 0:
            print(f"  Errors: {n_large_errors} ({n_large_errors / n_large_evaluable * 100:.1f}%)")

        # Injection targets
        print(f"\n  Both correct:               {len(both_correct):>3d}")
        print(f"  {large_key} ok, {small_key} wrong (INJECTION): {len(large_correct_small_wrong):>3d}")
        print(f"  Both wrong:                 {len(both_wrong):>3d}")
        print(f"  {small_key} ok, {large_key} wrong: {len(large_only_errors):>3d}")

        if large_correct_small_wrong:
            print(f"\n  *** INJECTION TARGETS ({len(large_correct_small_wrong)}) ***")
            for e in large_correct_small_wrong:
                match = df[df["sample_id"] == e["sample_id"]]
                rank_str = ""
                if len(match) > 0:
                    det = parse_details(match.iloc[0]["details"]) if "details" in match.columns else {}
                    if det.get("gt_rank", -1) > 0:
                        rank_str = f", gt_rank={det['gt_rank']}"
                    if det.get("entropy", 0) > 0:
                        rank_str += f", entropy={det['entropy']:.2f}"
                print(f"    {e['sample_id']:40s}  "
                      f"{e['dataset']:15s}  "
                      f"gt={e['large_gt']}{rank_str}")

        # B-ball analysis
        if n_errors > 0:
            n_bball = int(df['is_b_ball_dilemma'].sum())
            print(f"\nB-ball dilemma: {n_bball} ({n_bball / n_errors * 100:.1f}% of errors)")

            # GT rank diagnostic
            gt_ranks = []
            for _, r in df[df["has_error"]].iterrows():
                det = parse_details(r.get("details", {}))
                rank = det.get("gt_rank", -1)
                if rank > 0:
                    gt_ranks.append(rank)
            if gt_ranks:
                gt_ranks.sort()
                print(f"GT rank distribution ({len(gt_ranks)} with known rank):")
                print(f"  min={gt_ranks[0]}, median={gt_ranks[len(gt_ranks) // 2]}, "
                      f"max={gt_ranks[-1]}, mean={sum(gt_ranks) / len(gt_ranks):.0f}")

    # --- Three-model comparison ---
    if args.three_model:
        print(f"\n\n{'=' * 80}")
        print("Three-Model Comparison (1.5B vs 3B vs 7B)")
        print("=" * 80)
        compare_three_models(
            load_model_outputs("qwen1.5B", args.num_samples),
            load_model_outputs("qwen3B", args.num_samples),
            load_model_outputs("qwen7B", args.num_samples),
            args.num_samples, tokenizer,
        )

    print("\n" + "=" * 80)
    print("Step 3 Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()