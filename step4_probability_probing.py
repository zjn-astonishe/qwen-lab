"""
Step 4: Probability Probing (3-Model)
=================================================

For each error sample (small model wrong, large model correct), run ALL three models
(1.5B, 3B, 7B) through a single forward pass and probe the probability distribution at
EVERY layer by temporarily applying the LM head to each layer's hidden state.

This reveals:
  1. At which layer (progress %) each model starts diverging from the others
  2. At which layer the GT token gets "locked out" in the small/medium model
  3. Where the optimal switch points would be for early-exit / cascade inference:
     e.g. 1.5B推理到犹豫层 → 投影到7B推理少量层 → 投影回1.5B完成

Core insight (from experiment discussion):
  - 词表 = 原子箱子 (vocab_size=151,936), 输出空间一致
  - 隐藏状态 = 聚类箱子, 不同模型聚类划分不同
  - 1.5B: 28层×1536维, 3B: 36层×2048维, 7B: 28层×3584维
  - Probing的目的: 观察不同"聚类分辨率"下GT token的概率变化

Output:
  - JSON: per-sample, per-model, per-layer P(GT) and P(pred)
  - CSV:  flat per-layer probabilities
  - Plots: 5种可视化图
    (1) Aggregate P(GT) curves — 3模型对比
    (2) Individual sample evolution
    (3) Divergence & switch point analysis
    (4) Entropy comparison
    (5) Early exit decision matrix (heatmap)
"""

import json
import os
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

from config import MODELS, DATA_CONFIG, HARDWARE_CONFIG, ANALYSIS_CONFIG, PROBING_CONFIG
from utils import cleanup_gpu, load_model_output

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Model loading (lightweight — no generate, just forward pass)
# ---------------------------------------------------------------------------

def load_model_for_probing(model_key: str):
    """Load model and tokenizer for probing (no gradient)."""
    cfg = MODELS[model_key]
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(HARDWARE_CONFIG["dtype"], torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"], trust_remote_code=True, cache_dir="./models"
    )
    # transformers 5.x uses 'dtype' instead of 'torch_dtype'
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="./models",
            max_memory=HARDWARE_CONFIG.get("max_memory", None),
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="./models",
            max_memory=HARDWARE_CONFIG.get("max_memory", None),
        )
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Core probing function
# ---------------------------------------------------------------------------

def probe_all_layers(
    model,
    tokenizer,
    input_text: str,
    gt_token_id: int,
    pred_token_id: int,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Run a single prefill forward pass and probe probability at every layer.

    At each layer l, we take the hidden state of the LAST input token and
    apply the LM head to get a "pseudo-logit" distribution.  This tells us
    what the model would output if it stopped at layer l.

    Args:
        model: HuggingFace CausalLM
        tokenizer: corresponding tokenizer
        input_text: the full prompt
        gt_token_id: ground truth answer token id
        pred_token_id: small model's predicted (wrong) answer token id
        top_k: how many top tokens to record per layer

    Returns:
        Dict with per-layer probing results.
    """
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # Get the LM head (final linear layer: hidden_dim -> vocab_size)
    lm_head = model.get_output_embeddings()

    # Get the final RMSNorm layer.
    # In Qwen2, the forward path is: layers -> model.norm() -> lm_head.
    # outputs.hidden_states[-1] already has norm applied (Qwen2 applies it
    # internally), but intermediate layers are RAW outputs without norm.
    # We must apply norm before lm_head for every layer to get meaningful logits.
    final_norm = getattr(model.model, 'norm', None)
    if final_norm is None:
        final_norm = getattr(model.model, 'final_layernorm', None)

    # --- Diagnostic: verify norm is found and hidden_states structure ---
    _norm_found = final_norm is not None
    _norm_cls = type(final_norm).__name__ if _norm_found else 'None'
    print(f"    [DIAG] final_norm found: {_norm_found}, class: {_norm_cls}")

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    # outputs.hidden_states: tuple of (batch, seq_len, hidden_dim)
    # Index 0 = embedding layer, 1..N = transformer layers
    # NOTE: In Qwen2, hidden_states[-1] already includes model.norm().
    # Intermediate hidden_states [1..N-1] are RAW (no norm).
    all_hidden = outputs.hidden_states
    last_hidden_idx = len(all_hidden) - 1  # this one already has norm

    num_transformer_layers = len(all_hidden) - 1  # exclude embedding

    results = {
        "num_layers": num_transformer_layers,
        "layers": [],
    }

    # Also record final-layer actual logits for reference
    final_logits = outputs.logits[0, -1, :]  # (vocab_size,)
    final_probs = F.softmax(final_logits, dim=-1)
    results["final_gt_prob"] = float(final_probs[gt_token_id])
    results["final_pred_prob"] = float(final_probs[pred_token_id])
    results["final_gt_rank"] = int((final_logits.argsort(descending=True) == gt_token_id).nonzero()[0]) + 1

    # --- Diagnostic: verify hidden_states structure ---
    print(f"    [DIAG] hidden_states count: {len(all_hidden)}, "
          f"last_idx: {last_hidden_idx}, num_transformer_layers: {num_transformer_layers}")
    # Spot-check: compare probing at last layer (skip norm) vs outputs.logits
    _last_hidden = all_hidden[last_hidden_idx][0, -1, :]
    _last_hidden_dev = _last_hidden.device
    _lm_w = lm_head.weight.to(_last_hidden_dev) if lm_head.weight.device != _last_hidden_dev else lm_head.weight
    _lm_b = lm_head.bias.to(_last_hidden_dev) if lm_head.bias is not None and lm_head.bias.device != _last_hidden_dev else lm_head.bias
    _last_logits_check = F.linear(_last_hidden, _lm_w, _lm_b)
    _last_probs_check = F.softmax(_last_logits_check, dim=-1)
    _logits_match = torch.allclose(_last_probs_check[gt_token_id], final_probs[gt_token_id], atol=1e-4)
    print(f"    [DIAG] last-layer probe vs outputs.logits P(GT) match: {_logits_match} "
          f"({float(_last_probs_check[gt_token_id]):.6f} vs {float(final_probs[gt_token_id]):.6f})")
    del _last_hidden, _last_logits_check, _last_probs_check, _lm_w, _lm_b
    # Spot-check: compare layer 5 with norm vs without norm
    if len(all_hidden) > 6 and final_norm is not None:
        _raw_h = all_hidden[5][0, -1, :]
        _raw_h_dev = _raw_h.device
        _lm_w2 = lm_head.weight.to(_raw_h_dev) if lm_head.weight.device != _raw_h_dev else lm_head.weight
        _lm_b2 = lm_head.bias.to(_raw_h_dev) if lm_head.bias is not None and lm_head.bias.device != _raw_h_dev else lm_head.bias
        # Cast to float32 before norm to prevent fp16 overflow in RMSNorm
        _normed_h = final_norm(_raw_h.float().unsqueeze(0)).squeeze(0).to(_raw_h.dtype)
        _raw_logits = F.linear(_raw_h, _lm_w2, _lm_b2)
        _normed_logits = F.linear(_normed_h, _lm_w2, _lm_b2)
        _raw_p = F.softmax(_raw_logits, dim=-1)
        _normed_p = F.softmax(_normed_logits, dim=-1)
        _raw_entropy = float(-torch.sum(_raw_p * torch.log(_raw_p + 1e-10)))
        _normed_entropy = float(-torch.sum(_normed_p * torch.log(_normed_p + 1e-10)))
        print(f"    [DIAG] layer 4 (idx=5) without norm: entropy={_raw_entropy:.2f}, "
              f"P(GT)={float(_raw_p[gt_token_id]):.6f}")
        print(f"    [DIAG] layer 4 (idx=5) with norm:    entropy={_normed_entropy:.2f}, "
              f"P(GT)={float(_normed_p[gt_token_id]):.6f}")
        del _raw_h, _normed_h, _raw_logits, _normed_logits, _raw_p, _normed_p, _lm_w2, _lm_b2, _raw_entropy, _normed_entropy

    # Probe each transformer layer (skip index 0 = embedding)
    for layer_idx in range(1, len(all_hidden)):
        # Hidden state of last input token at this layer
        hidden = all_hidden[layer_idx][0, -1, :].unsqueeze(0)  # (1, 1, hidden_dim)

        # Apply final RMSNorm if this is NOT the last hidden state.
        # (The last one already has norm applied by the model internally.)
        # Cast to float32 before norm to prevent fp16 overflow in RMSNorm.
        if layer_idx != last_hidden_idx and final_norm is not None:
            hidden = final_norm(hidden.float()).to(hidden.dtype)

        hidden = hidden.squeeze(0).squeeze(0)  # back to (hidden_dim,)

        # Find the device of this hidden state (important for multi-GPU)
        hidden_device = hidden.device

        # Apply LM head: hidden_dim -> vocab_size
        if lm_head.weight.device != hidden_device:
            lm_head_weight = lm_head.weight.to(hidden_device)
            if lm_head.bias is not None:
                lm_head_bias = lm_head.bias.to(hidden_device)
            else:
                lm_head_bias = None
        else:
            lm_head_weight = lm_head.weight
            lm_head_bias = lm_head.bias

        probing_logits = F.linear(hidden, lm_head_weight, lm_head_bias)
        probing_probs = F.softmax(probing_logits, dim=-1)

        gt_prob = float(probing_probs[gt_token_id])
        pred_prob = float(probing_probs[pred_token_id])

        # Top-k tokens at this layer
        topk_probs, topk_ids = torch.topk(probing_probs, k=min(top_k, probing_probs.size(0)))
        topk_tokens = [tokenizer.decode([t]).strip() for t in topk_ids.tolist()]

        # GT rank at this layer
        gt_rank = int((probing_logits.argsort(descending=True) == gt_token_id).nonzero(as_tuple=True)[0]) + 1

        # Entropy (compute in float32 to avoid fp16 underflow on 150k+ vocab)
        probs_f32 = probing_probs.float()
        log_probs = torch.log(probs_f32 + 1e-10)
        entropy = float(-torch.sum(probs_f32 * log_probs))

        # Progress percentage through the model (0-100)
        progress = (layer_idx - 1) / max(num_transformer_layers - 1, 1) * 100

        results["layers"].append({
            "layer": layer_idx - 1,  # 0-indexed transformer layer
            "progress": progress,
            "gt_prob": gt_prob,
            "pred_prob": pred_prob,
            "prob_ratio": gt_prob / max(pred_prob, 1e-10),  # >1 means GT ahead
            "gt_rank": gt_rank,
            "entropy": entropy,
            "top1_token": topk_tokens[0],
            "top1_prob": float(topk_probs[0]),
            "gt_in_topk": gt_rank <= top_k,
            "topk_tokens": topk_tokens,
            "topk_probs": [float(p) for p in topk_probs.tolist()],
        })

    del outputs, inputs, final_logits, final_probs
    cleanup_gpu()

    return results


# ---------------------------------------------------------------------------
# Identify error samples to probe
# ---------------------------------------------------------------------------

def find_probe_targets(
    comparison_csv: str,
    small_model: str = "qwen1.5B",
    large_model: str = "qwen7B",
    max_samples: Optional[int] = None,
    error_pair: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Find samples where small model is wrong and large model is correct.

    When error_pair is None, finds ALL samples where at least one smaller model
    is wrong but 7B is correct (union of 1.5B-error and 3B-error sets).
    Each target records the wrong model's prediction as pred_wrong for probing.

    Returns list of dicts with sample_idx, sample_id, gt_answer, pred_wrong,
    pred_large, answer_type, wrong_model.
    """
    df = pd.read_csv(comparison_csv)

    large_tag = large_model.replace("qwen", "")
    large_correct_col = f"correct_{large_tag}"
    large_pred_col = f"pred_{large_tag}"

    if error_pair is not None:
        # Single pair mode: find samples for this specific pair
        wrong_tag = error_pair.replace("qwen", "")
        wrong_correct_col = f"correct_{wrong_tag}"
        wrong_pred_col = f"pred_{wrong_tag}"
        mask = (df[wrong_correct_col] == False) & (df[large_correct_col] == True)
        mask &= df[wrong_pred_col].notna() & (df[wrong_pred_col] != "")
        targets = []
        for _, row in df[mask].iterrows():
            targets.append({
                "sample_idx": int(row["sample_idx"]),
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "gt_answer": str(row["gt_answer"]),
                "pred_wrong": str(row[wrong_pred_col]),
                "pred_large": str(row[large_pred_col]),
                "wrong_model": wrong_tag,
                "answer_type": row.get("answer_type", "multiple_choice"),
            })
    else:
        # Union mode: find all samples where 7B is correct but at least one smaller model is wrong
        wrong_models = ["1.5B", "3B"]
        seen_indices = set()
        targets = []
        for wrong_tag in wrong_models:
            wrong_correct_col = f"correct_{wrong_tag}"
            wrong_pred_col = f"pred_{wrong_tag}"
            mask = (df[wrong_correct_col] == False) & (df[large_correct_col] == True)
            mask &= df[wrong_pred_col].notna() & (df[wrong_pred_col] != "")
            for _, row in df[mask].iterrows():
                idx = int(row["sample_idx"])
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    targets.append({
                        "sample_idx": idx,
                        "sample_id": row["sample_id"],
                        "dataset": row["dataset"],
                        "gt_answer": str(row["gt_answer"]),
                        "pred_wrong": str(row[wrong_pred_col]),
                        "pred_large": str(row[large_pred_col]),
                        "wrong_model": wrong_tag,
                        "answer_type": row.get("answer_type", "multiple_choice"),
                    })

    if max_samples and len(targets) > max_samples:
        targets = targets[:max_samples]

    return targets


def get_token_ids(
    tokenizer, answer: str, answer_type: str
) -> Optional[int]:
    """Get the first token ID for an answer string."""
    answer = answer.strip()
    if not answer or answer == "None":
        return None

    if answer_type == "numerical":
        tokens = tokenizer.encode(answer, add_special_tokens=False)
        return tokens[0] if tokens else None
    else:
        tokens = tokenizer.encode(answer, add_special_tokens=False)
        tokens = [t for t in tokens if tokenizer.decode([t]).strip() != ""]
        return tokens[0] if tokens else None


# ---------------------------------------------------------------------------
# Main probing pipeline
# ---------------------------------------------------------------------------

def run_probing(
    small_model_key: str = "qwen1.5B",
    large_model_key: str = "qwen7B",
    probe_models: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    output_dir: str = "experiment_results/probing",
    comparison_csv: Optional[str] = None,
):
    """Run layer-wise probability probing for error samples.

    Args:
        small_model_key: Model key for the "small" model in error analysis
        large_model_key: Model key for the "large" model in error analysis
        probe_models: List of model keys to actually probe (default: all 3)
        max_samples: Maximum number of samples to probe
        output_dir: Directory for output files
        comparison_csv: Path to three_model_comparison.csv
    """
    # Determine which models to probe
    if probe_models is None:
        probe_models = ["qwen1.5B", "qwen3B", "qwen7B"]

    print("=" * 80)
    print("Step 4: Probability Probing (3-Model)")
    print(f"  Error pair: {small_model_key} (wrong) vs {large_model_key} (correct)")
    print(f"  Probing models: {probe_models}")
    print("=" * 80)

    # --- Load comparison CSV ---
    if comparison_csv is None:
        comparison_csv = ANALYSIS_CONFIG["three_model_comparison"]

    print(f"\nLoading comparison CSV: {comparison_csv}")
    targets = find_probe_targets(comparison_csv, small_model_key, large_model_key, max_samples)
    print(f"  Found {len(targets)} error samples to probe")
    # Show breakdown by wrong_model
    from collections import Counter
    _model_counts = Counter(t.get("wrong_model", "unknown") for t in targets)
    print(f"  Breakdown: {dict(_model_counts)}")

    # Collect all output dirs we may need
    needed_output_dirs = set()
    for t in targets:
        wm = t.get("wrong_model", "1.5B")
        for mk in probe_models:
            needed_output_dirs.add(MODELS[mk]["output_dir"])
    print(f"  Output dirs needed: {needed_output_dirs}")

    if not targets:
        print("  No targets found. Exiting.")
        return

    # --- Prepare tokenizer (use first model's tokenizer for token ID lookup) ---
    # Load tokenizer only (no model weights) to get token IDs.
    # All Qwen2.5 models share the same vocabulary, so any tokenizer works.
    print(f"\nLoading tokenizer from {probe_models[0]}...")
    _ref_tokenizer = AutoTokenizer.from_pretrained(
        MODELS[probe_models[0]]["model_name"],
        trust_remote_code=True, cache_dir="./models"
    )

    # --- Prepare input text from model outputs ---
    # Try each model's output_dir until we find one that has the data
    output_dir_candidates = [MODELS[mk]["output_dir"] for mk in probe_models]

    # --- Pre-compute per-sample metadata (input_text, token IDs) ---
    # This is model-independent, so we do it once before any model is loaded.
    os.makedirs(output_dir, exist_ok=True)

    sample_meta = []  # list of (target_dict, input_text, gt_token_id, pred_token_id)
    for i, target in enumerate(tqdm(targets, desc="Preparing samples")):
        sample_idx = target["sample_idx"]

        # Load model output to get input_text (try multiple output dirs)
        input_text = None
        for _odir in output_dir_candidates:
            output = load_model_output(_odir, sample_idx)
            if output is not None:
                input_text = output.get("input_text", "")
                if input_text:
                    break
        if not input_text:
            print(f"  [SKIP] sample {sample_idx}: no input_text")
            continue

        # Get token IDs for GT and wrong model's pred answer
        pred_wrong = target.get("pred_wrong", target.get("pred_small", ""))
        gt_token_id = get_token_ids(_ref_tokenizer, target["gt_answer"], target["answer_type"])
        pred_token_id = get_token_ids(_ref_tokenizer, pred_wrong, target["answer_type"])

        if gt_token_id is None or pred_token_id is None:
            print(f"  [SKIP] sample {sample_idx}: cannot get token IDs "
                  f"(gt={target['gt_answer']}, pred={pred_wrong})")
            continue

        # Store token info
        target["gt_token_id"] = gt_token_id
        target["pred_token_id"] = pred_token_id
        target["gt_token_str"] = _ref_tokenizer.decode([gt_token_id]).strip()
        target["pred_token_str"] = _ref_tokenizer.decode([pred_token_id]).strip()

        sample_meta.append({
            "target": target,
            "input_text": input_text,
            "gt_token_id": gt_token_id,
            "pred_token_id": pred_token_id,
        })

    print(f"\n  {len(sample_meta)} samples ready for probing.")

    # --- Sequential model probing: load one model → probe all samples → unload ---
    # --- Pre-initialize all result dicts to avoid None entries in intermediate saves ---
    all_results = []
    for idx, meta in enumerate(sample_meta):
        target = meta["target"]
        all_results.append({
            "sample_idx": target["sample_idx"],
            "sample_id": target["sample_id"],
            "dataset": target["dataset"],
            "gt_answer": target["gt_answer"],
            "pred_wrong": target.get("pred_wrong", ""),
            "pred_large": target["pred_large"],
            "wrong_model": target.get("wrong_model", "unknown"),
            "answer_type": target["answer_type"],
            "gt_token_id": meta["gt_token_id"],
            "pred_token_id": meta["pred_token_id"],
            "gt_token_str": target["gt_token_str"],
            "pred_token_str": target["pred_token_str"],
        })

    for model_key in probe_models:
        model_tag = model_key.replace("qwen", "")  # "1.5B", "3B", "7B"
        print(f"\n{'=' * 60}")
        print(f"  Loading {model_key} ({model_tag}) for probing...")
        print(f"{'=' * 60}")

        try:
            model, tokenizer = load_model_for_probing(model_key)
        except Exception as e:
            print(f"  [FATAL] Failed to load {model_key}: {e}")
            print(f"  Skipping {model_tag}. All samples will have empty layers for this model.")
            cleanup_gpu()
            # Mark all samples as failed for this model
            for idx in range(len(sample_meta)):
                all_results[idx][f"{model_tag}_layers"] = []
                all_results[idx][f"{model_tag}_final_gt_prob"] = 0
                all_results[idx][f"{model_tag}_final_gt_rank"] = 999
                all_results[idx][f"{model_tag}_final_pred_prob"] = 0
                all_results[idx][f"{model_tag}_num_layers"] = 0
            continue

        print(f"  Device: {next(model.parameters()).device}")
        print(f"  Layers: {MODELS[model_key]['num_layers']}, "
              f"Hidden dim: {MODELS[model_key]['hidden_dim']}")

        for idx in range(len(sample_meta)):
            meta = sample_meta[idx]
            target = meta["target"]
            sample_idx = target["sample_idx"]

            try:
                probing = probe_all_layers(
                    model, tokenizer, meta["input_text"],
                    meta["gt_token_id"], meta["pred_token_id"],
                    top_k=PROBING_CONFIG.get("top_k_probe", 10),
                )
                all_results[idx][f"{model_tag}_final_gt_prob"] = probing["final_gt_prob"]
                all_results[idx][f"{model_tag}_final_pred_prob"] = probing["final_pred_prob"]
                all_results[idx][f"{model_tag}_final_gt_rank"] = probing["final_gt_rank"]
                all_results[idx][f"{model_tag}_layers"] = probing["layers"]
                all_results[idx][f"{model_tag}_num_layers"] = probing["num_layers"]
            except Exception as e:
                print(f"  [ERROR] {model_key} probing sample {sample_idx}: {e}")
                cleanup_gpu()
                all_results[idx][f"{model_tag}_layers"] = []
                all_results[idx][f"{model_tag}_final_gt_prob"] = 0
                all_results[idx][f"{model_tag}_final_gt_rank"] = 999
                all_results[idx][f"{model_tag}_final_pred_prob"] = 0
                all_results[idx][f"{model_tag}_num_layers"] = 0

            # Periodic cleanup within model loop
            if (idx + 1) % 5 == 0:
                cleanup_gpu()

            # Save intermediate results after each model finishes a batch
            if (idx + 1) % 10 == 0:
                _save_probing_results(all_results, output_dir)

        # --- Unload this model before loading the next ---
        print(f"  Unloading {model_key}...")
        del model
        cleanup_gpu()

        print(f"  {model_tag} probing complete.")

    # --- Final save ---
    _save_probing_results(all_results, output_dir)
    _save_probing_csv(all_results, output_dir, probe_models)
    _generate_probing_plots(all_results, output_dir, probe_models)

    # --- Summary statistics ---
    _print_summary(all_results, probe_models)

    cleanup_gpu()

    print(f"\nDone. Results saved to: {output_dir}")
    return all_results


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_probing_results(results: List[Dict], output_dir: str):
    """Save raw probing results as JSON."""
    path = os.path.join(output_dir, "probing_results.json")
    clean = []
    for r in results:
        item = {}
        for k, v in r.items():
            if isinstance(v, torch.Tensor):
                item[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                item[k] = v.item()
            else:
                item[k] = v
        clean.append(item)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False, default=str)


def _save_probing_csv(results: List[Dict], output_dir: str, probe_models: List[str]):
    """Save a flat CSV with per-sample, per-model, per-layer probabilities."""
    rows = []
    for r in results:
        base = {
            "sample_idx": r["sample_idx"],
            "sample_id": r["sample_id"],
            "dataset": r["dataset"],
            "answer_type": r["answer_type"],
            "gt_answer": r["gt_answer"],
            "pred_wrong": r.get("pred_wrong", r.get("pred_small", "")),
            "pred_large": r["pred_large"],
            "wrong_model": r.get("wrong_model", "unknown"),
        }
        # Each model's layers
        for model_key in probe_models:
            model_tag = model_key.replace("qwen", "")
            for layer_info in r.get(f"{model_tag}_layers", []):
                row = base.copy()
                row.update({
                    "model": model_tag,
                    "layer": layer_info["layer"],
                    "progress": layer_info["progress"],
                    "gt_prob": layer_info["gt_prob"],
                    "pred_prob": layer_info["pred_prob"],
                    "prob_ratio": layer_info["prob_ratio"],
                    "gt_rank": layer_info["gt_rank"],
                    "entropy": layer_info["entropy"],
                    "top1_token": layer_info["top1_token"],
                    "top1_prob": layer_info["top1_prob"],
                    "gt_in_topk": layer_info["gt_in_topk"],
                })
                rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "probing_per_layer.csv")
    df.to_csv(path, index=False)
    print(f"  Saved per-layer CSV: {path} ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Color map for models
MODEL_COLORS = {
    "1.5B": "steelblue",
    "3B": "mediumseagreen",
    "7B": "darkorange",
}

MODEL_LABELS = {
    "1.5B": "Qwen-1.5B",
    "3B": "Qwen-3B",
    "7B": "Qwen-7B",
}


def _generate_probing_plots(results: List[Dict], output_dir: str, probe_models: List[str]):
    """Generate all probing visualization plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        # Font setup — auto-detect available fonts
        _font_candidates = [
            "/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf",
            "/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.otf",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/lxgw-wenkai/LXGWWenKai-Regular.ttf",
        ]
        _has_cjk = False
        for fp in _font_candidates:
            if os.path.exists(fp):
                try:
                    fm.fontManager.addfont(fp)
                    _has_cjk = True
                except Exception:
                    pass

        _sans = []
        if _has_cjk:
            _sans.append("Noto Sans SC")
        _sans.append("DejaVu Sans")
        plt.rcParams["font.sans-serif"] = _sans
        plt.rcParams["axes.unicode_minus"] = False

        # Detect which models actually have data (some may have failed)
        available_tags = _detect_available_model_tags(results, probe_models)
        if not available_tags:
            print("  [WARN] No model layer data found in results, skipping plots.")
            return
        if len(available_tags) < len(probe_models):
            missing = set(_get_model_tags(probe_models)) - set(available_tags)
            print(f"  [INFO] Only {available_tags} have layer data; {missing} missing. "
                  f"Right panels will show single-model fallback.")

        _plot_aggregate_curves(results, output_dir, plt, available_tags)
        _plot_individual_samples(results, output_dir, plt, available_tags)
        _plot_divergence_analysis(results, output_dir, plt, available_tags)
        _plot_entropy_comparison(results, output_dir, plt, available_tags)
        _plot_early_exit_matrix(results, output_dir, plt, available_tags)

        plt.close("all")
        print("  All probing plots generated.")
    except ImportError as e:
        print(f"  [WARN] matplotlib not available, skipping plots: {e}")


def _get_model_tags(probe_models: List[str]) -> List[str]:
    """Convert model keys to tags."""
    return [m.replace("qwen", "") for m in probe_models]


def _detect_available_model_tags(results: List[Dict], probe_models: List[str]) -> List[str]:
    """Detect which probe models actually have layer data in results.

    When some models fail during probing (OOM, etc.), their *_layers keys
    will be absent from results.  This function filters probe_models down
    to only those that have at least one sample with layer data.
    """
    available = []
    for model_key in probe_models:
        model_tag = model_key.replace("qwen", "")
        if any(r.get(f"{model_tag}_layers") for r in results):
            available.append(model_tag)
    return available


def _plot_aggregate_curves(results: List[Dict], output_dir: str, plt, probe_models: List[str]):
    """Plot 1: Aggregate P(GT), GT Rank, and Prob Ratio curves across all samples.

    For models with different layer counts, x-axis is progress percentage (0-100).
    """
    if not results:
        return

    model_tags = _get_model_tags(probe_models)
    n_models = len(model_tags)

    fig, axes = plt.subplots(1, 3, figsize=(7 * n_models + 4, 6))

    # --- Panel A: Mean P(GT) across progress ---
    ax = axes[0]
    for model_tag in model_tags:
        color = MODEL_COLORS.get(model_tag, "gray")
        label = MODEL_LABELS.get(model_tag, model_tag)

        # Plot individual samples (light)
        for r in results:
            layers = r.get(f"{model_tag}_layers", [])
            if layers:
                xs = [l["progress"] for l in layers]
                ys = [l["gt_prob"] for l in layers]
                ax.plot(xs, ys, color=color, alpha=0.08, linewidth=0.6)

        # Mean curve
        mean = _compute_mean_curve_by_progress(results, model_tag, "gt_prob")
        if mean:
            ax.plot(mean[0], mean[1], color=color, linewidth=2.5, label=f"{label} (n={len(results)})")

    ax.set_xlabel("Progress Through Model (%)", fontsize=12)
    ax.set_ylabel("P(Ground Truth Token)", fontsize=12)
    ax.set_title("(a) Mean P(GT) vs Progress", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    # --- Panel B: Mean GT Rank ---
    ax = axes[1]
    for model_tag in model_tags:
        color = MODEL_COLORS.get(model_tag, "gray")
        label = MODEL_LABELS.get(model_tag, model_tag)
        mean = _compute_mean_curve_by_progress(results, model_tag, "gt_rank")
        if mean:
            ax.plot(mean[0], mean[1], color=color, linewidth=2.5, label=label)

    ax.set_xlabel("Progress Through Model (%)", fontsize=12)
    ax.set_ylabel("GT Token Rank (lower = better)", fontsize=12)
    ax.set_title("(b) Mean GT Rank vs Progress", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    # --- Panel C: Mean P(GT)/P(pred) ratio ---
    ax = axes[2]
    for model_tag in model_tags:
        color = MODEL_COLORS.get(model_tag, "gray")
        label = MODEL_LABELS.get(model_tag, model_tag)
        mean = _compute_mean_curve_by_progress(results, model_tag, "prob_ratio")
        if mean:
            ax.plot(mean[0], mean[1], color=color, linewidth=2.5, label=label)

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Ratio = 1 (tie)")
    ax.set_xlabel("Progress Through Model (%)", fontsize=12)
    ax.set_ylabel("P(GT) / P(pred) Ratio", fontsize=12)
    ax.set_title("(c) Mean P(GT)/P(pred) vs Progress", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(output_dir, "probing_aggregate.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_individual_samples(results: List[Dict], output_dir: str, plt, probe_models: List[str]):
    """Plot 2: Individual sample P(GT) evolution — all probed models."""
    if not results:
        return

    model_tags = _get_model_tags(probe_models)
    n_show = min(len(results), 12)
    fig, axes = plt.subplots(3, 4, figsize=(24, 15))
    axes = axes.flatten()

    for i in range(n_show):
        r = results[i]
        ax = axes[i]

        for model_tag in model_tags:
            color = MODEL_COLORS.get(model_tag, "gray")
            label = MODEL_LABELS.get(model_tag, model_tag)
            layers = r.get(f"{model_tag}_layers", [])
            if not layers:
                continue

            xs = [l["progress"] for l in layers]
            ys_gt = [l["gt_prob"] for l in layers]
            ys_pred = [l["pred_prob"] for l in layers]

            ax.plot(xs, ys_gt, "-", color=color, linewidth=1.5, label=f"{label} P(GT)")
            ax.plot(xs, ys_pred, "--", color=color, linewidth=0.8, alpha=0.6)

        ax.set_title(f"#{r['sample_idx']} {r['dataset']}\n"
                     f"GT={r['gt_answer']} wrong={r.get('pred_wrong', r.get('pred_small', '?'))}"
                     f" [{r.get('wrong_model', '?')}]",
                     fontsize=9)
        ax.legend(loc="best", fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Progress (%)", fontsize=8)
        ax.set_ylabel("Prob", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide empty subplots
    for i in range(n_show, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Individual Sample: Layer-wise P(GT) (solid) and P(pred) (dashed)\n"
                 "All Probed Models",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "probing_individual.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_divergence_analysis(results: List[Dict], output_dir: str, plt, probe_models: List[str]):
    """Plot 3: Divergence point analysis — where do models start to diverge?

    Panel A: Per-sample divergence layer histogram (for each model pair)
    Panel B: Optimal switch point scatter (progress-aligned)
    """
    if not results:
        return

    model_tags = _get_model_tags(probe_models)

    # Build pairs: each model vs the largest (reference / correct) model.
    # Divergence only makes sense when comparing a wrong model against the
    # correct model — comparing two wrong models (e.g. 1.5B vs 3B) is
    # meaningless since both have low P(GT).
    pairs = []
    if len(model_tags) >= 2:
        tag_ref = model_tags[-1]  # largest / correct model (e.g. 7B)
        for i in range(len(model_tags) - 1):
            pairs.append((model_tags[i], tag_ref))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel A: Divergence histogram for each pair ---
    ax = axes[0]
    threshold = PROBING_CONFIG.get("divergence_threshold", 0.1)

    if pairs:
        # Multi-model: show divergence point between each model pair
        for tag_small, tag_large in pairs:
            divergence_layers = []
            for r in results:
                dl = _find_divergence_layer(r, tag_small, tag_large, threshold)
                if dl is not None:
                    divergence_layers.append(dl)

            if divergence_layers:
                color_s = MODEL_COLORS.get(tag_small, "gray")
                color_l = MODEL_COLORS.get(tag_large, "gray")
                median = np.median(divergence_layers)
                ax.hist(divergence_layers, bins=20, alpha=0.5,
                        label=f"{tag_small} vs {tag_large} (median={median:.0f}%)",
                        edgecolor="black", linewidth=0.5)

        ax.set_xlabel("Divergence Progress (%)", fontsize=12)
        ax.set_ylabel("Number of Samples", fontsize=12)
        ax.set_title(f"(a) Divergence Point (P(GT) gap > {threshold})", fontsize=13)
        ax.legend(loc="best", fontsize=9)
    else:
        # Single-model fallback: show GT emergence progress per wrong_model
        # (at what progress does GT token first enter top-10?)
        probe_tag = model_tags[0] if model_tags else "7B"
        emergence_any = False
        for wm in sorted(set(r.get("wrong_model", "unknown") for r in results)):
            emergence_progress = []
            for r in results:
                if r.get("wrong_model") != wm:
                    continue
                layers = r.get(f"{probe_tag}_layers", [])
                for l in layers:
                    if l["gt_rank"] <= 10 and l["gt_prob"] >= 0.001:
                        emergence_progress.append(l["progress"])
                        break
            if emergence_progress:
                emergence_any = True
                color = MODEL_COLORS.get(wm, "gray")
                median = np.median(emergence_progress)
                ax.hist(emergence_progress, bins=15, alpha=0.55, color=color,
                        edgecolor="black", linewidth=0.5,
                        label=f"{wm} wrong → GT visible (med={median:.0f}%)")
        if emergence_any:
            ax.set_xlabel("GT Emergence Progress (%)", fontsize=12)
            ax.set_ylabel("Number of Samples", fontsize=12)
            ax.set_title(f"(a) When GT Enters Top-10 ({probe_tag})", fontsize=13)
            ax.legend(loc="best", fontsize=9)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    # --- Panel B: Per-sample best switch point ---
    ax = axes[1]

    # Try to find the best switch pair (smallest vs largest available model)
    switch_plotted = False
    if len(model_tags) >= 2:
        # Use smallest and largest available models for switch analysis
        tag_small, tag_large = model_tags[0], model_tags[-1]
        switch_info = []
        for r in results:
            sw = _find_best_switch_layer(r, tag_small, tag_large)
            if sw is not None:
                switch_info.append(sw)

        if switch_info:
            switch_progress = [s["progress"] for s in switch_info]
            ax.hist(switch_progress, bins=20, color="coral", edgecolor="black", alpha=0.8)
            ax.axvline(x=np.median(switch_progress), color="red", linestyle="--",
                       linewidth=2, label=f"Median = {np.median(switch_progress):.0f}%")
            ax.axvline(x=np.mean(switch_progress), color="blue", linestyle=":",
                       linewidth=2, label=f"Mean = {np.mean(switch_progress):.0f}%")
            ax.set_xlabel("Optimal Switch Progress (%)", fontsize=12)
            ax.set_ylabel("Number of Samples", fontsize=12)
            ax.set_title(f"(b) When to Switch: {tag_small} → {tag_large}", fontsize=13)
            ax.legend(loc="best", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)
            switch_plotted = True

    # Fallback: show GT-vs-wrong divergence per wrong_model (works with single model)
    if not switch_plotted:
        probe_tag = model_tags[0] if model_tags else "7B"
        div_layers_any = False
        for wm in sorted(set(r.get("wrong_model", "unknown") for r in results)):
            div_layers = []
            for r in results:
                if r.get("wrong_model") != wm:
                    continue
                layers = r.get(f"{probe_tag}_layers", [])
                for l in layers:
                    if l["gt_rank"] <= 5 and l["gt_prob"] >= 0.001:
                        div_layers.append(l["progress"])
                        break
            if div_layers:
                div_layers_any = True
                color = MODEL_COLORS.get(wm, "gray")
                ax.hist(div_layers, bins=15, alpha=0.55, color=color,
                        edgecolor="black", linewidth=0.5,
                        label=f"{wm} wrong (n={len(div_layers)}, med={np.median(div_layers):.0f}%)")
        if div_layers_any:
            ax.set_xlabel("Divergence Progress (%)", fontsize=12)
            ax.set_ylabel("Number of Samples", fontsize=12)
            ax.set_title("(b) GT Surpasses Wrong (by wrong_model)", fontsize=13)
            ax.legend(loc="best", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)

    plt.tight_layout()
    path = os.path.join(output_dir, "probing_divergence.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_entropy_comparison(results: List[Dict], output_dir: str, plt, probe_models: List[str]):
    """Plot 4: Entropy evolution — confidence trajectory per model."""
    if not results:
        return

    model_tags = _get_model_tags(probe_models)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel A: Mean entropy ---
    ax = axes[0]
    for model_tag in model_tags:
        color = MODEL_COLORS.get(model_tag, "gray")
        label = MODEL_LABELS.get(model_tag, model_tag)
        mean = _compute_mean_curve_by_progress(results, model_tag, "entropy")
        if mean:
            ax.plot(mean[0], mean[1], color=color, linewidth=2.5, label=label)

    ax.set_xlabel("Progress Through Model (%)", fontsize=12)
    ax.set_ylabel("Entropy (nats)", fontsize=12)
    ax.set_title("(a) Entropy Evolution (Lower = More Confident)", fontsize=13)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    # --- Panel B: Entropy gap between models ---
    ax = axes[1]
    pairs = []
    for i in range(len(model_tags)):
        for j in range(i + 1, len(model_tags)):
            pairs.append((model_tags[i], model_tags[j]))

    if pairs:
        # Multi-model: show entropy gap between model pairs
        for tag_s, tag_l in pairs:
            mean_s = _compute_mean_curve_by_progress(results, tag_s, "entropy")
            mean_l = _compute_mean_curve_by_progress(results, tag_l, "entropy")
            if mean_s and mean_l:
                min_len = min(len(mean_s[0]), len(mean_l[0]))
                gap = [mean_s[1][k] - mean_l[1][k] for k in range(min_len)]
                ax.plot(mean_s[0][:min_len], gap, linewidth=2,
                        label=f"{tag_s} - {tag_l}")
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

        ax.set_xlabel("Progress Through Model (%)", fontsize=12)
        ax.set_ylabel("Entropy Difference (nats)", fontsize=12)
        ax.set_title("(b) Entropy Gap Between Models", fontsize=13)
    else:
        # Single-model fallback: show entropy split by wrong_model
        probe_tag = model_tags[0] if model_tags else "7B"
        wrong_models = sorted(set(r.get("wrong_model", "unknown") for r in results))
        for wm in wrong_models:
            wm_results = [r for r in results if r.get("wrong_model") == wm]
            mean = _compute_mean_curve_by_progress(wm_results, probe_tag, "entropy")
            if mean:
                color = MODEL_COLORS.get(wm, "gray")
                ax.plot(mean[0], mean[1], color=color, linewidth=2.5,
                        label=f"wrong_model={wm}")
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.set_xlabel("Progress Through Model (%)", fontsize=12)
        ax.set_ylabel("Entropy (nats)", fontsize=12)
        ax.set_title("(b) Entropy by wrong_model", fontsize=13)

    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    plt.tight_layout()
    path = os.path.join(output_dir, "probing_entropy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_early_exit_matrix(results: List[Dict], output_dir: str, plt, probe_models: List[str]):
    """Plot 5: Early Exit Decision Matrix.

    A heatmap showing, for each progress bucket × model pair, the mean P(GT)
    of the larger model minus the mean P(GT) of the smaller model.
    This indicates where switching would be most beneficial.

    Also includes: "how many layers of 7B are needed" analysis.
    """
    if not results:
        return

    model_tags = _get_model_tags(probe_models)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel A: P(GT) heatmap by progress bucket and model ---
    ax = axes[0]

    # Create progress buckets
    n_buckets = 20
    bucket_edges = np.linspace(0, 100, n_buckets + 1)
    bucket_centers = (bucket_edges[:-1] + bucket_edges[1:]) / 2

    # Build matrix: models × progress buckets
    matrix = np.zeros((len(model_tags), n_buckets))

    for m_idx, model_tag in enumerate(model_tags):
        # Collect all progress and gt_prob values for this model
        all_prog = []
        all_gt_prob = []
        for r in results:
            layers = r.get(f"{model_tag}_layers", [])
            if layers:
                all_prog.append([l["progress"] for l in layers])
                all_gt_prob.append([l["gt_prob"] for l in layers])

        if not all_prog:
            matrix[m_idx, :] = np.nan
            continue

        # Flatten
        flat_prog = np.concatenate(all_prog)
        flat_vals = np.concatenate(all_gt_prob)

        # Bin by progress using digitize
        bin_indices = np.digitize(flat_prog, bucket_edges) - 1  # 0-indexed bins
        for b_idx in range(n_buckets):
            mask = bin_indices == b_idx
            if mask.any():
                matrix[m_idx, b_idx] = np.mean(flat_vals[mask])
            else:
                matrix[m_idx, b_idx] = np.nan

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   extent=[0, 100, len(model_tags) - 0.5, -0.5],
                   vmin=0, vmax=0.5)
    ax.set_yticks(range(len(model_tags)))
    ax.set_yticklabels([MODEL_LABELS.get(t, t) for t in model_tags])
    ax.set_xlabel("Progress Through Model (%)", fontsize=12)
    ax.set_title("(a) Mean P(GT) Heatmap", fontsize=13)
    plt.colorbar(im, ax=ax, label="Mean P(GT)")

    # --- Panel B: How many layers are needed? ---
    ax = axes[1]

    # Use ALL available models for threshold comparison
    if len(model_tags) >= 2:
        threshold_values = [0.1, 0.15, 0.2, 0.3, 0.4]
        x = np.arange(len(threshold_values))
        width = min(0.2, 0.8 / len(model_tags))

        for m_idx, model_tag in enumerate(model_tags):
            reaches = []
            for thresh in threshold_values:
                count = 0
                for r in results:
                    layers = r.get(f"{model_tag}_layers", [])
                    if any(l["gt_prob"] >= thresh for l in layers):
                        count += 1
                reaches.append(count / max(len(results), 1) * 100)

            offset = (m_idx - (len(model_tags) - 1) / 2) * width
            ax.bar(x + offset, reaches, width,
                   color=MODEL_COLORS.get(model_tag, "gray"),
                   label=MODEL_LABELS.get(model_tag, model_tag),
                   alpha=0.8, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f"P(GT) >= {t}" for t in threshold_values])
        ax.set_xlabel("P(GT) Threshold", fontsize=12)
        ax.set_ylabel("Samples Reaching Threshold (%)", fontsize=12)
        ax.set_title("(b) How Many Samples Reach P(GT) Threshold?", fontsize=13)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    else:
        # Single-model fallback: show P(GT) threshold reachability split by wrong_model
        probe_tag = model_tags[0] if model_tags else "7B"
        threshold_values = [0.01, 0.05, 0.1, 0.2, 0.5, 0.9]
        x = np.arange(len(threshold_values))
        width = 0.35
        wrong_models = sorted(set(r.get("wrong_model", "unknown") for r in results))

        for m_idx, wm in enumerate(wrong_models):
            reaches = []
            wm_results = [r for r in results if r.get("wrong_model") == wm]
            for thresh in threshold_values:
                count = 0
                for r in wm_results:
                    layers = r.get(f"{probe_tag}_layers", [])
                    if any(l["gt_prob"] >= thresh for l in layers):
                        count += 1
                reaches.append(count / max(len(wm_results), 1) * 100)

            color = MODEL_COLORS.get(wm, "gray")
            ax.bar(x + (m_idx - 0.5) * width, reaches, width,
                   color=color, label=f"wrong_model={wm}",
                   alpha=0.8, edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f">={t}" for t in threshold_values])
        ax.set_xlabel("P(GT) Threshold", fontsize=12)
        ax.set_ylabel("Samples Reaching (%)", fontsize=12)
        ax.set_title(f"(b) P(GT) Threshold Reachability ({probe_tag})", fontsize=13)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "probing_early_exit.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _compute_mean_curve_by_progress(
    results: List[Dict], model_tag: str, field: str, n_points: int = 50
) -> Optional[Tuple[List[float], List[float]]]:
    """Compute mean curve across all samples, aligned by progress percentage.

    Vectorized using numpy interpolation — O(N*L) instead of O(n_points*N*L).
    """
    # Build 2D arrays: each row is one sample's progress-values pairs
    progress_arrays = []
    value_arrays = []

    for r in results:
        layers = r.get(f"{model_tag}_layers", [])
        if layers:
            progress_arrays.append([l["progress"] for l in layers])
            value_arrays.append([l[field] for l in layers])

    if not progress_arrays:
        return None

    # Target progress points
    target_progress = np.linspace(0, 100, n_points)
    # For each sample, interpolate to target progress points
    # Stack interpolated values and take mean across samples (ignoring NaN)
    interp_matrix = np.full((len(progress_arrays), n_points), np.nan)

    for idx, (p_list, v_list) in enumerate(zip(progress_arrays, value_arrays)):
        p_arr = np.array(p_list)
        v_arr = np.array(v_list)
        # np.interp requires strictly increasing x, our progress values are monotonic
        interp_matrix[idx] = np.interp(target_progress, p_arr, v_arr)

    # Mean across samples, ignoring NaN
    mean_values = np.nanmean(interp_matrix, axis=0).tolist()
    target_list = target_progress.tolist()

    return target_list, mean_values


def _find_divergence_layer(
    r: Dict, tag_small: str, tag_large: str, threshold: float = 0.1
) -> Optional[float]:
    """Find the progress point where small model's P(GT) falls below large model's by threshold.

    Vectorized using numpy interpolation.

    When both models are wrong (e.g. 1.5B vs 3B, neither correct), absolute P(GT)
    values are tiny so the fixed threshold never fires.  In that case we fall back
    to an adaptive threshold based on the observed maximum gap, so that the first
    proportionally significant divergence point is still detected.
    """
    small_layers = r.get(f"{tag_small}_layers", [])
    large_layers = r.get(f"{tag_large}_layers", [])

    if not small_layers or not large_layers:
        return None

    # Build progress-value pairs
    small_p = np.array([l["progress"] for l in small_layers])
    small_v = np.array([l["gt_prob"] for l in small_layers])
    large_p = np.array([l["progress"] for l in large_layers])
    large_v = np.array([l["gt_prob"] for l in large_layers])

    # Interpolate both to a fine grid
    target_pct = np.arange(0, 101, 5, dtype=float)
    small_interp = np.interp(target_pct, small_p, small_v)
    large_interp = np.interp(target_pct, large_p, large_v)

    # --- Pass 1: absolute threshold (works when a correct model like 7B exists) ---
    gap = large_interp - small_interp
    exceed_indices = np.where(gap > threshold)[0]
    if len(exceed_indices) > 0:
        return float(target_pct[exceed_indices[0]])

    # --- Pass 2: adaptive relative threshold ---
    # Useful when both models are wrong (e.g. 1.5B vs 3B), where absolute P(GT)
    # values are low (~0.001-0.01) and the gap never reaches the fixed threshold.
    max_gap = float(np.max(gap))
    if max_gap > 0:
        # Use 20% of max_gap as the adaptive threshold, but also require the
        # large model's P(GT) to be at least 1e-4 to avoid early-layer noise.
        adaptive_threshold = max(max_gap * 0.2, 1e-5)
        for idx in range(len(target_pct)):
            if gap[idx] >= adaptive_threshold and large_interp[idx] >= 1e-4:
                return float(target_pct[idx])

    return None


def _find_best_switch_layer(
    r: Dict, tag_small: str = "1.5B", tag_large: str = "7B"
) -> Optional[Dict]:
    """Find the progress point where the GT probability gap between large and small is maximized.

    Vectorized using numpy interpolation.
    """
    small_layers = r.get(f"{tag_small}_layers", [])
    large_layers = r.get(f"{tag_large}_layers", [])

    if not small_layers or not large_layers:
        return None

    small_p = np.array([l["progress"] for l in small_layers])
    small_v = np.array([l["gt_prob"] for l in small_layers])
    large_p = np.array([l["progress"] for l in large_layers])
    large_v = np.array([l["gt_prob"] for l in large_layers])

    target_pct = np.arange(0, 101, 2, dtype=float)
    small_interp = np.interp(target_pct, small_p, small_v)
    large_interp = np.interp(target_pct, large_p, large_v)

    gap = large_interp - small_interp
    best_idx = int(np.argmax(gap))

    # Find the corresponding large model layer at this progress
    best_progress = float(target_pct[best_idx])
    closest_layer_idx = int(np.argmin(np.abs(large_p - best_progress)))

    return {
        "progress": best_progress,
        "gap": float(gap[best_idx]),
        "large_layer_at_switch": large_layers[closest_layer_idx]["layer"],
    }


def _print_summary(results: List[Dict], probe_models: List[str]):
    """Print summary statistics."""
    if not results:
        return

    model_tags = _get_model_tags(probe_models)

    print(f"\n{'=' * 80}")
    print("Probing Summary")
    print(f"{'=' * 80}")
    print(f"  Samples probed: {len(results)}")

    # Final layer stats per model
    print(f"\n  Final Layer Stats:")
    for model_tag in model_tags:
        final_gt = [r.get(f"{model_tag}_final_gt_prob", 0) for r in results]
        final_rank = [r.get(f"{model_tag}_final_gt_rank", 999) for r in results]
        if final_gt:
            print(f"    {model_tag:>4s} P(GT): mean={np.mean(final_gt):.4f}, "
                  f"median={np.median(final_gt):.4f}")
            print(f"    {model_tag:>4s} GT rank: mean={np.mean(final_rank):.1f}, "
                  f"median={np.median(final_rank):.1f}")

    # Divergence analysis: each model vs the largest (correct) model
    threshold = PROBING_CONFIG.get("divergence_threshold", 0.1)
    pairs = []
    if len(model_tags) >= 2:
        tag_ref = model_tags[-1]
        for i in range(len(model_tags) - 1):
            pairs.append((model_tags[i], tag_ref))

    print(f"\n  Divergence Analysis (threshold={threshold}):")
    for tag_s, tag_l in pairs:
        divergence_progress = []
        for r in results:
            dl = _find_divergence_layer(r, tag_s, tag_l, threshold)
            if dl is not None:
                divergence_progress.append(dl)

        if divergence_progress:
            print(f"    {tag_s} vs {tag_l}: "
                  f"{len(divergence_progress)}/{len(results)} samples diverge, "
                  f"median at {np.median(divergence_progress):.0f}% progress")

    # Switch point analysis (1.5B → 7B)
    if "1.5B" in model_tags and "7B" in model_tags:
        switch_info = [_find_best_switch_layer(r, "1.5B", "7B") for r in results]
        switch_info = [s for s in switch_info if s is not None]

        if switch_info:
            switch_progress = [s["progress"] for s in switch_info]
            switch_gaps = [s["gap"] for s in switch_info]
            switch_layers = [s["large_layer_at_switch"] for s in switch_info]

            print(f"\n  Early Exit Switch Analysis (1.5B -> 7B):")
            print(f"    Mean optimal switch progress: {np.mean(switch_progress):.1f}%")
            print(f"    Median optimal switch progress: {np.median(switch_progress):.1f}%")
            print(f"    Mean GT prob gap at switch: {np.mean(switch_gaps):.4f}")
            print(f"    Mean 7B layer at switch: {np.mean(switch_layers):.1f} / "
                  f"{MODELS['qwen7B']['num_layers']}")

            # How many layers of 7B needed after switch?
            remaining_7b = [MODELS['qwen7B']['num_layers'] - s['large_layer_at_switch']
                           for s in switch_info]
            print(f"    Mean remaining 7B layers after switch: {np.mean(remaining_7b):.1f}")

    # 3B intermediate analysis
    if "3B" in model_tags:
        # How many samples does 3B get right that 1.5B gets wrong?
        gt_above_threshold_3b = 0
        gt_above_threshold_15b = 0
        gt_above_threshold_7b = 0
        for r in results:
            final_3b = r.get("3B_final_gt_prob", 0)
            final_15b = r.get("1.5B_final_gt_prob", 0)
            final_7b = r.get("7B_final_gt_prob", 0)
            if final_3b > final_15b:
                gt_above_threshold_3b += 1
            if final_15b > 0:
                gt_above_threshold_15b += 1
            if final_7b > 0:
                gt_above_threshold_7b += 1

        print(f"\n  3B Intermediate Analysis:")
        print(f"    Samples where 3B final P(GT) > 1.5B: {gt_above_threshold_3b}/{len(results)}")
        print(f"    Suggests 3B could serve as intermediate for cascade inference")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 4: Probability Probing (3-Model)"
    )
    parser.add_argument("--small_model", type=str, default="qwen1.5B",
                        choices=list(MODELS.keys()),
                        help="Small model for error sample selection")
    parser.add_argument("--large_model", type=str, default="qwen7B",
                        choices=list(MODELS.keys()),
                        help="Large model for error sample selection")
    parser.add_argument("--models", nargs="+", type=str,
                        choices=list(MODELS.keys()), default=["qwen1.5B", "qwen3B", "qwen7B"],
                        help="Models to probe (default: all 3)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to probe (default: all error samples)")
    parser.add_argument("--output_dir", type=str,
                        default="experiment_results/probing")
    parser.add_argument("--comparison_csv", type=str, default=None,
                        help="Path to three_model_comparison.csv")
    args = parser.parse_args()

    run_probing(
        small_model_key=args.small_model,
        large_model_key=args.large_model,
        probe_models=args.models,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        comparison_csv=args.comparison_csv,
    )


if __name__ == "__main__":
    main()
