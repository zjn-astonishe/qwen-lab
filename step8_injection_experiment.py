"""
Step 8: Injection Experiment (V3 — QA tasks, optimized)

Inject small model's hidden states into large model and evaluate the effect.

Optimized:
  - All answer extraction / comparison delegated to qa_utils (no duplication)
  - eval() replaced with ast.literal_eval via qa_utils.parse_details()
  - Uses config.get_projection_paths() for dynamic model-pair file naming
  - Uses config.get_injection_results_path() for dynamic output naming
  - Cleaner hook mechanism with proper resource cleanup
"""

import os
import json
import torch
import pandas as pd
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    MODELS, INJECTION_CONFIG, ANALYSIS_CONFIG, HARDWARE_CONFIG,
    get_projection_paths, get_injection_results_path, get_error_analysis_path,
)
from qa_utils import (
    extract_answer, compare_answers, get_gt_answer, get_answer_type,
    get_answer_token_id, find_answer_step, get_clean_answer,
)
from utils import load_model_output, cleanup_gpu


# ---------------------------------------------------------------------------
# Projection matrix I/O
# ---------------------------------------------------------------------------

def load_projection_matrix(
    small_model: str, large_model: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load projection matrix and bias for a specific model pair."""
    paths = get_projection_paths(small_model, large_model)
    W_path = paths["projection_matrix"]
    b_path = paths["projection_bias"]

    if not os.path.exists(W_path) or not os.path.exists(b_path):
        raise FileNotFoundError(
            f"Projection matrix not found for {small_model}->{large_model}. "
            f"Expected: {W_path}. Run step5 first."
        )

    W = torch.load(W_path, map_location="cpu")
    b = torch.load(b_path, map_location="cpu")
    print(f"  Loaded projection: W={W.shape}, b={b.shape}")
    return W, b


# ---------------------------------------------------------------------------
# Injection target selection
# ---------------------------------------------------------------------------

def _model_key_to_col(model_key: str) -> str:
    """Convert model key (e.g. 'qwen3B') to CSV column suffix (e.g. '3B')."""
    return model_key.replace("qwen", "")


def load_injection_samples(
    error_analysis_path: str,
    max_samples: int = 50,
    small_model: str = "qwen3B",
    large_model: str = "qwen7B",
) -> List[int]:
    """Load samples where large model is correct but small model is wrong."""
    df = pd.read_csv(error_analysis_path)
    targets = []

    small_col = f"correct_{_model_key_to_col(small_model)}"
    large_col = f"correct_{_model_key_to_col(large_model)}"

    if small_col in df.columns and large_col in df.columns:
        # Three-model comparison format
        for _, row in df.iterrows():
            if row.get(large_col, False) and not row.get(small_col, False):
                targets.append(int(row["sample_idx"]))
        print(f"  Found {len(targets)} injection targets ({large_model} ok, {small_model} wrong)")
    else:
        # Legacy error analysis format
        for _, row in df.iterrows():
            if row.get("has_error", False) and row.get("error_type") != "gt_unavailable":
                targets.append(int(row["sample_idx"]))
        print(f"  Found {len(targets)} error samples")

    return targets[:max_samples]


# ---------------------------------------------------------------------------
# Injection hook mechanism
# ---------------------------------------------------------------------------

def inject_and_generate(
    model,
    input_ids: torch.Tensor,
    injection_hidden: torch.Tensor,
    injection_layer: int,
    alpha: float,
    max_new_tokens: int = 30,
) -> Dict[str, Any]:
    """Run model.generate() with hidden-state injection at every decode step."""
    device = model.device
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)
    inj = injection_hidden.to(device)

    def make_hook(projected):
        call_count = {"n": 0}

        def hook(module, input, output):
            # Only inject during the first forward pass (prefill).
            # On subsequent decode steps the input is a single token and
            # injecting would cascade errors across the generation.
            call_count["n"] += 1
            if call_count["n"] > 1:
                return  # pass-through for all decode steps

            target_device = output[0].device if isinstance(output, tuple) else output.device
            proj = projected.to(target_device)
            if isinstance(output, tuple):
                hs = output[0].clone()
                hs[:, -1, :] = (1 - alpha) * hs[:, -1, :] + proj.unsqueeze(0)
                return (hs,) + output[1:]
            else:
                hs = output.clone()
                hs[:, -1, :] = (1 - alpha) * hs[:, -1, :] + proj.unsqueeze(0)
                return hs
        return hook

    layers = model.model.layers
    target = layers[injection_layer]
    handle = target.register_forward_hook(make_hook(inj))

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=2,
            )

        probs_per_step = []
        for s in outputs.scores:
            probs_per_step.append(torch.softmax(s[0], dim=-1).cpu())

        result = {
            "generated_ids": outputs.sequences[0].cpu(),
            "probs_per_step": probs_per_step,
        }
    finally:
        handle.remove()

    return result


# ---------------------------------------------------------------------------
# Per-sample injection experiment
# ---------------------------------------------------------------------------

def run_injection_experiment(
    sample_idx: int,
    small_model_output: Dict,
    W: torch.Tensor,
    b: torch.Tensor,
    model,
    tokenizer,
    alpha_values: List[float],
    injection_layers: List[int],
    step_idx: int = 0,
) -> Tuple[List[Dict], Optional[Dict]]:
    """Run injection experiments for a single sample across alpha/layer combos."""
    results = []
    probs_data = {"sample_idx": sample_idx}

    hidden_states = small_model_output.get("hidden_states_per_step", [])
    if not hidden_states:
        return results, None

    input_ids = small_model_output["input_ids"]
    ground_truth = small_model_output.get("ground_truth", {})
    answer_type = get_answer_type(ground_truth)
    gt_answer = get_gt_answer(ground_truth)

    # Load original large model output for baseline
    large_path = os.path.join(MODELS["qwen7B"]["output_dir"], f"sample_{sample_idx:03d}.pt")
    original_probs = None
    orig_out = load_model_output(MODELS["qwen7B"]["output_dir"], sample_idx)
    if orig_out is not None:
        probs_list = orig_out.get("probs_per_step", [])
        if probs_list:
            original_probs = probs_list[0]
        else:
            tk = orig_out.get("top_k_info", [])
            if tk:
                original_probs = tk[0]["probs"]

    probs_data["original_large_probs"] = original_probs

    # Parse predicted answer using centralized get_clean_answer
    pred_answer, answer_type = get_clean_answer(small_model_output)
    if not answer_type:
        answer_type = get_answer_type(ground_truth)

    if not pred_answer or not gt_answer:
        return results, probs_data

    generated_ids = small_model_output.get("generated_ids")
    if generated_ids is None:
        return results, probs_data

    input_len = input_ids.shape[0] if input_ids.dim() == 1 else input_ids.shape[1]
    answer_step, actual_pred_token = find_answer_step(
        generated_ids, input_len, tokenizer, pred_answer, answer_type
    )

    pred_token_id = actual_pred_token if actual_pred_token is not None else -1
    gt_token_id = get_answer_token_id(tokenizer, gt_answer, answer_type) or -1

    probs_data["pred_answer"] = pred_answer
    probs_data["gt_answer"] = gt_answer
    probs_data["answer_step"] = answer_step
    probs_data["answer_type"] = answer_type
    probs_data["ground_truth"] = ground_truth
    probs_data["injected_probs"] = {}

    for injection_layer in injection_layers:
        layer_idx = -injection_layer

        inj_step = answer_step if 0 < answer_step < len(hidden_states) else step_idx
        step_hidden = hidden_states[inj_step]
        if abs(layer_idx) >= len(step_hidden):
            continue

        small_hidden = step_hidden[layer_idx].float()
        projected = (small_hidden @ W.T + b)[-1]  # last token only

        for alpha in alpha_values:
            try:
                inj_result = inject_and_generate(
                    model, input_ids, projected,
                    injection_layer=layer_idx, alpha=alpha,
                    max_new_tokens=512,
                )

                inj_text = tokenizer.decode(inj_result["generated_ids"], skip_special_tokens=True)
                inj_answer = extract_answer(inj_text, answer_type)

                inj_cmp = compare_answers(inj_answer, ground_truth)
                pred_cmp = compare_answers(pred_answer, ground_truth)

                # GT rank after injection — find rank at the first decode step
                # (step 0), which is the most meaningful: it reflects how the
                # injected prefill influences the very first generated token.
                gt_rank = -1
                probs_list = inj_result.get("probs_per_step", [])
                if probs_list and gt_token_id >= 0:
                    first_probs = probs_list[0]
                    if gt_token_id < len(first_probs):
                        sorted_idx = torch.argsort(first_probs, descending=True)
                        positions = (sorted_idx == gt_token_id).nonzero(as_tuple=True)
                        if len(positions[0]) > 0:
                            gt_rank = int(positions[0][0].item()) + 1

                # Save probs for visualization (first decode step only)
                if probs_list:
                    key = f"layer{injection_layer}_alpha{alpha}"
                    probs_data["injected_probs"][key] = probs_list[0].clone()

                # Original GT rank (at first decode step of original large model)
                orig_gt_rank = -1
                if original_probs is not None and gt_token_id >= 0 and gt_token_id < len(original_probs):
                    sorted_idx = torch.argsort(original_probs, descending=True)
                    positions = (sorted_idx == gt_token_id).nonzero(as_tuple=True)
                    if len(positions[0]) > 0:
                        orig_gt_rank = int(positions[0][0].item()) + 1

                results.append({
                    "sample_idx": sample_idx,
                    "injection_layer": injection_layer,
                    "alpha": alpha,
                    "pred_answer": pred_answer,
                    "injected_answer": inj_answer,
                    "gt_answer": gt_answer,
                    "answer_type": answer_type,
                    "gt_token_id": gt_token_id,
                    "original_correct": not pred_cmp["has_error"],
                    "injected_correct": not inj_cmp["has_error"],
                    "original_gt_rank": orig_gt_rank,
                    "gt_rank_after_injection": gt_rank,
                    "answer_changed": inj_answer != pred_answer,
                    "error_corrected": pred_cmp["has_error"] and not inj_cmp["has_error"],
                })

            except Exception as e:
                print(f"  Error (sample={sample_idx}, layer={injection_layer}, alpha={alpha}): {e}")
                continue

    return results, probs_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 8: Injection Experiment (QA tasks)")
    parser.add_argument("--small_model", type=str, default="qwen3B")
    parser.add_argument("--large_model", type=str, default="qwen7B")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--error_analysis", type=str, default=None)
    args = parser.parse_args()

    print("=" * 80)
    print(f"Step 8: Injection Experiment ({args.small_model} -> {args.large_model})")
    print("=" * 80)

    # Resolve error analysis path
    if args.error_analysis is None:
        three_model_path = ANALYSIS_CONFIG.get("three_model_comparison")
        if three_model_path and os.path.exists(three_model_path):
            args.error_analysis = three_model_path
        else:
            args.error_analysis = get_error_analysis_path(args.small_model, args.large_model)

    # Load projection matrix
    W, b = load_projection_matrix(args.small_model, args.large_model)

    # Load injection targets
    injection_samples = load_injection_samples(
        args.error_analysis, args.max_samples, args.small_model, args.large_model
    )
    if not injection_samples:
        print("No injection targets found. Check step 3 output.")
        return

    # Load large model
    print(f"\nLoading large model: {MODELS[args.large_model]['model_name']}")
    model_name = MODELS[args.large_model]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="./models")
    # transformers 5.x uses 'dtype' instead of 'torch_dtype'
    _dtype = torch.float16 if HARDWARE_CONFIG["dtype"] == "float16" else torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=_dtype,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="./models",
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=_dtype,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="./models",
        )
    model.eval()
    print(f"  Loaded on: {model.device}")

    all_results = []
    all_probs_data = []
    small_output_dir = MODELS[args.small_model]["output_dir"]

    for sample_idx in tqdm(injection_samples, desc="Injection"):
        small_output = load_model_output(small_output_dir, sample_idx)
        if small_output is None:
            print(f"  Warning: missing sample {sample_idx}")
            continue

        sample_results, probs_data = run_injection_experiment(
            sample_idx, small_output, W, b, model, tokenizer,
            INJECTION_CONFIG["alpha_values"],
            INJECTION_CONFIG["injection_layers"],
        )
        all_results.extend(sample_results)
        if probs_data is not None:
            all_probs_data.append(probs_data)

    del model
    cleanup_gpu()

    if not all_results:
        print("No results collected.")
        return

    # Save results
    output_path = get_injection_results_path(args.small_model, args.large_model)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    if all_probs_data:
        probs_path = output_path.replace(".csv", "_probs.pt")
        torch.save(all_probs_data, probs_path)
        print(f"Probabilities saved to: {probs_path}")

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("Injection Experiment Results")
    print(f"{'=' * 80}")
    print(f"Total experiments: {len(all_results)}")

    n_corrected = df["error_corrected"].sum()
    print(f"Errors corrected: {n_corrected}/{len(df)} ({n_corrected / len(df) * 100:.1f}%)")

    print(f"\nCorrection rate by alpha:")
    for alpha in sorted(df["alpha"].unique()):
        sub = df[df["alpha"] == alpha]
        corrected = sub["error_corrected"].sum()
        print(f"  alpha={alpha}: {corrected}/{len(sub)} ({corrected / len(sub) * 100:.1f}%)")

    print(f"\nCorrection rate by layer:")
    for layer in sorted(df["injection_layer"].unique()):
        sub = df[df["injection_layer"] == layer]
        corrected = sub["error_corrected"].sum()
        print(f"  layer={layer}: {corrected}/{len(sub)} ({corrected / len(sub) * 100:.1f}%)")

    valid = df[df["gt_rank_after_injection"] > 0]
    if len(valid) > 0:
        print(f"\nAvg GT rank after injection: {valid['gt_rank_after_injection'].mean():.2f}")
        print(f"Answer changed: {df['answer_changed'].sum()}/{len(df)}")

        best = valid.groupby(["injection_layer", "alpha"])["gt_rank_after_injection"].mean()
        best = best.sort_values()
        print("\nBest configurations (lowest avg GT rank):")
        print(best.head(10).to_string())


if __name__ == "__main__":
    main()