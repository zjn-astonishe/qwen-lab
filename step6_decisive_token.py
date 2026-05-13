"""
Step 6: Decisive Token Analysis (V2)

Identify "decisive tokens" — the PROMPT tokens that receive the most attention
from the model when producing the FINAL ANSWER token.

Core Question:
When the model generates its final answer, which tokens in the original
question/prompt did it attend to the most? These are the "decisive tokens" —
the information the model relied on to make its decision.

Methodology:
1. Load model + tokenizer (eager attention for output_attentions=True)
2. For each sample, load the full generated sequence (prompt + answer) from step2
3. Run a FORWARD PASS on the full sequence with output_attentions=True
4. Extract attention from the LAST MEANINGFUL ANSWER TOKEN position
   (not the prompt end, not EOS — the actual answer token)
5. Slice attention to prompt tokens only → "which prompt tokens decided the answer"
6. Compute entropy at the answer position (model's uncertainty when producing answer)
7. Filter: keep only high-entropy samples (model is uncertain)
8. Per-layer decisive tokens → aggregated ranking → visualizations

Key change from V1:
  V1 analyzed attention from the LAST PROMPT position (predicting 1st generated token)
  V2 analyzes attention from the ANSWER TOKEN position (final decision point)

Usage:
  python run_experiment.py --steps 10
  python run_experiment.py --steps 10 --decisive_model qwen7B
  python run_experiment.py --steps 10 --decisive_model qwen3B --entropy_threshold 3.0
"""

import os
import json
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from tqdm import tqdm

from config import MODELS, HARDWARE_CONFIG
from utils import load_model_output


# ============================================================
# 1. Model Loading (compatible with transformers 5.x)
# ============================================================

def load_model_for_attention(model_key: str):
    """Load model and tokenizer, reusing step2's pattern."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = MODELS[model_key]["model_name"]
    dtype_str = HARDWARE_CONFIG.get("dtype", "float16")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype_str, torch.float16)

    print(f"  Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir="./models",
    )

    print(f"  Loading model: {model_name} ({dtype_str})")
    # CRITICAL: must use 'eager' attention to support output_attentions=True.
    # The default 'sdpa' does NOT return attention weights.
    attn_impl = "eager"
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="./models",
            max_memory=HARDWARE_CONFIG.get("max_memory", None),
            attn_implementation=attn_impl,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="./models",
            max_memory=HARDWARE_CONFIG.get("max_memory", None),
            attn_implementation=attn_impl,
        )

    model.eval()

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"  Device: {model.device}, dtype: {model.dtype}")
    print(f"  Num layers: {model.config.num_hidden_layers}")
    print(f"  Num heads: {model.config.num_attention_heads}")
    print(f"  Num KV heads: {getattr(model.config, 'num_key_value_heads', 'N/A')}")

    return model, tokenizer


# ============================================================
# 2. Forward Pass with Attention Capture
# ============================================================

# Structured format tokens to skip when finding the answer position.
# These are tokens that carry formatting/structural meaning (newlines, colons,
# spaces, thinking tags, chat template tokens) rather than substantive answer
# content.  We skip them so that entropy is measured at the first *meaningful*
# answer token, not at a formatting token the model is almost certain about.
_FORMAT_TOKEN_STRINGS = frozenset({
    '\n', '\r', '\t', ' ',       # whitespace
    ':', '：',                    # colons (answer prefix separators)
    '.', '。', ',', '，',        # punctuation used as formatting
    ';', '；',                    # semicolons
})

# Substring patterns for multi-token format markers (e.g. <think思考>)
_FORMAT_TOKEN_PATTERNS = [
    '<think', '</think',
    '<|im_start|>', '<|im_end|>',
    '<|im_start|', '<|im_end|',
    'assistant',                  # role label inside template
]


def _is_format_token(decoded_str: str) -> bool:
    """Return True if the decoded token string is a structural format token."""
    s = decoded_str.strip()
    if not s:
        return True  # pure whitespace token
    if s in _FORMAT_TOKEN_STRINGS:
        return True
    for pat in _FORMAT_TOKEN_PATTERNS:
        if pat in s:
            return True
    return False


def _find_answer_token_pos(tokenizer, token_ids, gen_start: int, gen_end: int):
    """Skip structured format tokens and find the first meaningful answer token.

    Starting from ``gen_start``, scans forward through the generated tokens
    until a non-format token is found.  Also scans backward from ``gen_end``
    (skipping trailing EOS / format tokens) to find the last meaningful token.

    Strategy: for entropy analysis we typically want the FIRST content token
    of the answer (the moment the model starts committing to an answer).
    But if the model uses <think思考> tags, we want to skip past the thinking
    block and use the first token AFTER </think思考>.

    Returns:
        (first_content_pos, last_content_pos) — absolute positions in token_ids.
        Falls back to (gen_start, gen_end - 1) if no content token found.
    """
    total_len = len(token_ids)
    gen_start = max(gen_start, 0)
    gen_end = min(gen_end, total_len)

    # --- Find first content token (skip leading format tokens) ---
    first_pos = gen_start
    for pos in range(gen_start, gen_end):
        decoded = tokenizer.decode([token_ids[pos]])
        if not _is_format_token(decoded):
            first_pos = pos
            break
    else:
        # All tokens are format tokens — fall back
        first_pos = gen_start

    # --- Find last content token (skip trailing EOS / format tokens) ---
    last_pos = max(gen_end - 1, gen_start)
    for pos in range(gen_end - 1, gen_start - 1, -1):
        decoded = tokenizer.decode([token_ids[pos]])
        if not _is_format_token(decoded):
            last_pos = pos
            break

    return first_pos, last_pos


# ============================================================
# 2b. Fast entropy from Step2 (skip forward pass)
# ============================================================

def _fast_entropy_from_step2(step2_output):
    """Compute entropy from step2's saved generation scores, avoiding a forward pass.

    Because of causal masking, step2's ``scores[0]`` (the logits at the first
    generation step) is mathematically identical to what step6 would compute
    via ``logits[input_length - 1]`` in a full forward pass.  We can therefore
    reuse step2's data directly.

    Priority:
      1. ``logits_per_step[0]`` — full vocab logits from step2 (exact).
      2. ``top_k_info[0]``     — top-k probs from step2 (approximate, lower bound).

    Returns:
        (entropy: float | None, source: str | None)
        *source* indicates which path was used (for diagnostics).
    """
    # --- Path 1: full logits (exact) ---
    logits_per_step = step2_output.get("logits_per_step")
    if logits_per_step and len(logits_per_step) > 0:
        logits = logits_per_step[0].float().clamp(max=80)
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-30)
        entropy = -(probs * log_probs).sum().item()
        return entropy, "logits_per_step[0] (exact)"

    # --- Path 2: top-k probs (approximate) ---
    top_k_info = step2_output.get("top_k_info")
    if top_k_info and len(top_k_info) > 0:
        info = top_k_info[0]
        probs = info["probs"].float()
        k = probs.numel()
        coverage = probs.sum().item()

        # Shannon entropy from observed top-k
        log_probs = torch.log(probs + 1e-30)
        topk_ent = -(probs * log_probs).sum().item()

        # Approximate the tail: assume remaining probability mass is uniform
        remaining = max(0.0, 1.0 - coverage)
        V = 151643  # Qwen2.5 vocab size
        if remaining > 1e-10 and V > k:
            # H_tail = -R * log(R / (V - k))
            topk_ent += -remaining * np.log(remaining / (V - k))

        return topk_ent, f"top_k_info[0] (k={k}, coverage={coverage:.4f})"

    return None, None


def _extract_fast_entropy_data(step2_output):
    """Extract minimal data from a step2 output for fast-entropy computation.

    Returns a lightweight dict that _fast_entropy_from_step2 can consume,
    or None if neither logits_per_step nor top_k_info is available.
    Tensors are moved to CPU to free GPU memory.
    """
    logits_per_step = step2_output.get("logits_per_step")
    if logits_per_step and len(logits_per_step) > 0:
        return {"logits_per_step": [logits_per_step[0].cpu()]}

    top_k_info = step2_output.get("top_k_info")
    if top_k_info and len(top_k_info) > 0:
        info = top_k_info[0]
        return {"top_k_info": [{
            "probs": info["probs"].cpu(),
        }]}

    return None


def _build_prompt_content_mask(tokenizer, prompt_ids):
    """Build a boolean mask that identifies content (non-format) positions in the prompt.

    Format tokens include:
      - Chat template special tokens: <|im_start|>, <|im_end|>
      - Role labels: system, user, assistant
      - Their immediate neighbors (±1) to capture surrounding \\n / whitespace

    Returns:
        content_mask: np.ndarray of shape (prompt_len,), True = content position.
    """
    import torch as _torch
    total = len(prompt_ids)
    is_format = np.zeros(total, dtype=bool)

    # Known Qwen2.5 special token IDs
    _SPECIAL_TIDS = {151644, 151645}  # <|im_start|>, <|im_end|>

    # Role label strings (exact match after stripping whitespace)
    _ROLE_LABELS = {'system', 'user', 'assistant'}

    for pos in range(total):
        tid = prompt_ids[pos].item() if isinstance(prompt_ids[pos], _torch.Tensor) else int(prompt_ids[pos])

        if tid in _SPECIAL_TIDS:
            is_format[pos] = True
            continue

        decoded = tokenizer.decode([tid]).strip().lower()
        if decoded in _ROLE_LABELS:
            is_format[pos] = True

    # Expand ±1 to also capture surrounding newlines / whitespace
    expanded = is_format.copy()
    for pos in range(total):
        if is_format[pos]:
            if pos > 0:
                expanded[pos - 1] = True
            if pos < total - 1:
                expanded[pos + 1] = True

    content_mask = ~expanded
    return content_mask


def forward_pass_with_attention(model, tokenizer, input_text: str,
                                 generated_ids=None, input_length=None,
                                 max_length: int = 2048,
                                 extract_per_step: bool = False):
    """Run a forward pass and capture logits + ALL layers' attention.

    V2 (generated_ids + input_length provided):
        Forward pass on the FULL sequence (prompt + generated answer).
        Extracts attention from the ANSWER token position.
        Returns attention sliced to prompt tokens only.

    V1 (fallback, generated_ids not provided):
        Forward pass on prompt only, extracts from last prompt position.

    Args:
        extract_per_step: If True, also extract attention at every generation
            step (not just the answer position).  This enables per-step
            attention evolution analysis.  Adds per_step_data to the return dict.

    Returns dict with:
        logits: (vocab_size,) float tensor on CPU (NaN-free)
        prompt_attn: (num_layers, num_heads, prompt_len) float numpy
        input_ids: (seq_len,) int tensor on CPU
        analyze_pos: int — position where attention was extracted
        input_length: int — number of prompt tokens
        total_len: int — total sequence length
        per_step_data: dict | None — per-step attention/entropy summary
    """
    if generated_ids is not None and input_length is not None:
        # === V2: Full sequence (prompt + generated answer) ===
        full_ids = generated_ids.unsqueeze(0).to(model.device)  # (1, total_len)
        total_len = full_ids.shape[1]

        if total_len > max_length:
            full_ids = full_ids[:, :max_length]
            total_len = max_length
            input_length = min(input_length, max_length - 1)

        attention_mask = torch.ones_like(full_ids)

        with torch.no_grad():
            outputs = model(
                input_ids=full_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        num_generated = total_len - input_length

        # Skip structured format tokens (newlines, <think思考>, etc.)
        # to find the actual answer token positions.
        gen_start = input_length
        gen_end = total_len
        first_content_pos, last_content_pos = _find_answer_token_pos(
            tokenizer, full_ids[0].tolist(), gen_start, gen_end,
        )
        skipped_format = first_content_pos - gen_start
        if skipped_format > 0:
            # Decode skipped tokens for diagnostics
            skipped_ids = full_ids[0, gen_start:first_content_pos].tolist()
            skipped_text = repr(tokenizer.decode(skipped_ids, clean_up_tokenization_spaces=False))
            if len(skipped_text) > 80:
                skipped_text = skipped_text[:77] + "..."
            print(f"    [format-skip] skipped {skipped_format} format tokens: {skipped_text}")

        # analyze_pos: last meaningful content token (for attention extraction)
        analyze_pos = last_content_pos

        # CRITICAL: logits[pos] predicts the token at pos+1.
        # We want the model's uncertainty WHEN CHOOSING the first content token.
        # So we use logits[first_content_pos - 1] which predicts first_content_pos.
        entropy_pos = max(first_content_pos - 1, 0)
        logits = outputs.logits[0, entropy_pos, :].float().cpu()

        nan_cnt = int(torch.isnan(logits).sum().item())
        inf_cnt = int(torch.isinf(logits).sum().item())
        if nan_cnt > 0 or inf_cnt > 0:
            print(f"    [WARN] logits: {nan_cnt} NaN, {inf_cnt} inf / {logits.numel()} total")
            logits = torch.nan_to_num(logits, nan=-80.0, posinf=80.0, neginf=-80.0)

        # Attention from analyze_pos (last content token) to ALL positions
        all_layers_attn = np.stack(
            [layer_attn[0, :, analyze_pos, :].float().cpu().numpy()
             for layer_attn in outputs.attentions],
            axis=0,
        )  # (num_layers, num_heads, total_len)

        if np.isnan(all_layers_attn).any():
            nan_count = int(np.isnan(all_layers_attn).sum())
            print(f"    [WARN] attention: {nan_count} NaN / {all_layers_attn.size} total values")
            all_layers_attn = np.nan_to_num(all_layers_attn, nan=0.0)

        prompt_attn = all_layers_attn[:, :, :input_length]  # (num_layers, num_heads, prompt_len)
        input_ids_cpu = full_ids[0].cpu()

        # --- Per-step attention evolution (V2 only) ---
        # Because of causal masking, attention at position input_length + step
        # only depends on tokens 0..input_length+step — identical to what the
        # model saw during autoregressive generation at that step.
        # We extract this by slicing the full attention matrix already computed.
        per_step_data = None
        if extract_per_step and num_generated > 0:
            gen_token_texts = []
            per_step_entropies = []
            # Average attention over layers & heads: (num_steps, prompt_len)
            per_step_avg_prompt_attn = np.zeros((num_generated, input_length), dtype=np.float32)
            num_layers_actual = len(outputs.attentions)

            for step in range(num_generated):
                pos = input_length + step
                gen_token_texts.append(
                    tokenizer.decode([full_ids[0, pos].item()], clean_up_tokenization_spaces=False)
                )

                # Per-step entropy from the forward pass logits
                step_logits = outputs.logits[0, pos, :].float().cpu()
                step_logits = torch.nan_to_num(step_logits, nan=-80.0, posinf=80.0, neginf=-80.0)
                per_step_entropies.append(compute_entropy(step_logits))
                del step_logits

                # Average attention from all layers & heads at this position → prompt
                step_attn_sum = np.zeros(input_length, dtype=np.float32)
                for layer_attn in outputs.attentions:
                    # (1, heads, total_len, total_len) → (heads, prompt_len) → mean → (prompt_len,)
                    step_attn_sum += layer_attn[0, :, pos, :input_length].float().mean(dim=0).cpu().numpy()
                per_step_avg_prompt_attn[step] = step_attn_sum / num_layers_actual

            per_step_data = {
                "num_steps": num_generated,
                "gen_tokens": gen_token_texts,
                "per_step_entropy": per_step_entropies,
                "avg_prompt_attn": per_step_avg_prompt_attn,  # (num_steps, prompt_len)
            }

        return {
            "logits": logits,
            "prompt_attn": prompt_attn,
            "input_ids": input_ids_cpu,
            "analyze_pos": analyze_pos,
            "first_content_pos": first_content_pos,
            "input_length": input_length,
            "total_len": total_len,
            "per_step_data": per_step_data,
        }
    else:
        # === V1: Prompt-only fallback ===
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        logits = outputs.logits[0, -1, :].float().cpu()

        nan_cnt = int(torch.isnan(logits).sum().item())
        inf_cnt = int(torch.isinf(logits).sum().item())
        if nan_cnt > 0 or inf_cnt > 0:
            print(f"    [WARN] logits: {nan_cnt} NaN, {inf_cnt} inf / {logits.numel()} total  |  dtype={logits.dtype}")
            logits = torch.nan_to_num(logits, nan=-80.0, posinf=80.0, neginf=-80.0)

        all_layers_attn = np.stack(
            [layer_attn[0, :, -1, :].float().cpu().numpy() for layer_attn in outputs.attentions],
            axis=0,
        )

        if np.isnan(all_layers_attn).any():
            nan_count = int(np.isnan(all_layers_attn).sum())
            print(f"    [WARN] attention: {nan_count} NaN / {all_layers_attn.size} total values")
            all_layers_attn = np.nan_to_num(all_layers_attn, nan=0.0)

        input_ids_cpu = input_ids[0].cpu()
        seq_len = input_ids_cpu.shape[0]

        return {
            "logits": logits,
            "prompt_attn": all_layers_attn,
            "input_ids": input_ids_cpu,
            "analyze_pos": seq_len - 1,
            "input_length": seq_len,
            "total_len": seq_len,
        }


def quick_entropy_scan(model, tokenizer, input_text: str,
                       generated_ids=None, input_length=None,
                       max_length: int = 2048) -> float:
    """Fast forward pass WITHOUT attention — just compute output entropy.

    V2: entropy at the answer token position (when generated_ids provided).
    V1: entropy at the last prompt position (fallback).

    Returns:
        entropy: float (NaN if forward pass fails)
    """
    if generated_ids is not None and input_length is not None:
        # V2: Full sequence — entropy at answer position
        full_ids = generated_ids.unsqueeze(0).to(model.device)
        total_len = full_ids.shape[1]
        if total_len > max_length:
            full_ids = full_ids[:, :max_length]
            total_len = max_length
            input_length = min(input_length, max_length - 1)

        num_generated = total_len - input_length

        # Skip structured format tokens to find the first content token
        first_content_pos, _ = _find_answer_token_pos(
            tokenizer, full_ids[0].tolist(), input_length, total_len,
        )

        # logits[pos] predicts token at pos+1; use first_content_pos - 1
        # to capture the model's uncertainty when choosing the first content token.
        entropy_pos = max(first_content_pos - 1, 0)

        attention_mask = torch.ones_like(full_ids)
        with torch.no_grad():
            outputs = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, entropy_pos, :].float().cpu()
    else:
        # V1: Prompt only
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[0, -1, :].float().cpu()

    nan_cnt = int(torch.isnan(logits).sum().item())
    inf_cnt = int(torch.isinf(logits).sum().item())
    if nan_cnt > 0 or inf_cnt > 0:
        print(f"    [WARN] entropy_scan logits: {nan_cnt} NaN, {inf_cnt} inf / {logits.numel()} total")
        logits = torch.nan_to_num(logits, nan=-80.0, posinf=80.0, neginf=-80.0)

    return compute_entropy(logits)


# ============================================================
# 3. Entropy & Candidate Extraction
# ============================================================

def compute_entropy(logits: torch.Tensor) -> float:
    """Shannon entropy H(p) = -Σ p(x) log p(x) in nats.

    Note: callers should run nan_to_num on logits BEFORE calling this
    function. The clamp here is a safety net for extreme-but-finite values.
    """
    logits = logits.float().clamp(max=80)
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-30)
    return -(probs * log_probs).sum().item()


def get_top_candidates(logits: torch.Tensor, tokenizer, top_k: int = 5):
    """Get top-k candidate tokens with their probabilities."""
    logits = logits.float().clamp(max=80)
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, min(top_k, probs.shape[0]))
    candidates = []
    for prob, tid in zip(top_probs, top_ids):
        token_text = tokenizer.decode([tid.item()], clean_up_tokenization_spaces=False)
        candidates.append({"token": token_text, "token_id": tid.item(), "prob": prob.item()})
    return candidates


# ============================================================
# 4. Decisive Token Extraction
# ============================================================

def extract_decisive_tokens_from_attn(
    layer_attn: np.ndarray,   # (num_heads, seq_len)
    input_ids: torch.Tensor,    # (seq_len,)
    tokenizer,
    top_k: int = 10,
) -> List[Dict]:
    """From a single layer's attention at the last position, find the most attended input tokens.

    Returns:
        List of dicts: {token_text, token_id, position, avg_attention, per_head_attention}
        Sorted by avg_attention descending.
    """
    avg_attn = layer_attn.mean(axis=0)  # (seq_len,)

    k = min(top_k, len(avg_attn))
    top_positions = np.argsort(avg_attn)[::-1][:k]

    results = []
    for pos in top_positions:
        tid = input_ids[pos].item()
        token_text = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        results.append({
            "token_text": token_text,
            "token_id": tid,
            "position": int(pos),
            "avg_attention": float(avg_attn[pos]),
            "per_head_attention": layer_attn[:, pos].tolist(),
        })

    return results


# Backwards-compatible alias
def extract_decisive_tokens(attn_matrix, input_ids, tokenizer, top_k=10):
    return extract_decisive_tokens_from_attn(attn_matrix, input_ids, tokenizer, top_k)


def decode_all_tokens(input_ids: torch.Tensor, tokenizer) -> List[str]:
    """Decode all input tokens individually."""
    return [
        tokenizer.decode([tid.item()], clean_up_tokenization_spaces=False)
        for tid in input_ids
    ]


# ============================================================
# 5. Per-Sample Analysis
# ============================================================

def analyze_one_sample(model, tokenizer, input_text: str,
                       top_k_candidates: int, top_k_tokens: int,
                       generated_ids=None, input_length=None,
                       max_length: int = 2048,
                       extract_per_step: bool = False) -> Optional[Dict]:
    """Full multi-layer analysis: forward pass WITH attention, extract decisive tokens per layer.

    V2 (Redesigned): Analyzes attention from ANSWER token position WITHOUT filtering format tokens.
    This version keeps ALL input tokens (including <|im_end|>, whitespace, etc.) to enable
    full exploration of attention patterns across layers during reasoning.

    Args:
        extract_per_step: If True, extract attention at every generation step
            to track how the model's focus evolves during generation.

    Returns analysis dict with per-layer data or None if forward pass fails.
    """
    try:
        fp_result = forward_pass_with_attention(
            model, tokenizer, input_text,
            generated_ids=generated_ids,
            input_length=input_length,
            max_length=max_length,
            extract_per_step=extract_per_step,
        )
    except Exception as e:
        print(f"    Forward pass failed: {e}")
        return None

    logits = fp_result["logits"]
    prompt_attn = fp_result["prompt_attn"]  # (num_layers, num_heads, prompt_len)
    input_ids = fp_result["input_ids"]
    analyze_pos = fp_result["analyze_pos"]
    first_content_pos = fp_result.get("first_content_pos", analyze_pos)
    inp_len = fp_result["input_length"]
    total_len = fp_result["total_len"]

    entropy = compute_entropy(logits)
    candidates = get_top_candidates(logits, tokenizer, top_k_candidates)
    all_tokens = decode_all_tokens(input_ids, tokenizer)

    # Decode prompt and answer tokens separately
    prompt_tokens = decode_all_tokens(input_ids[:inp_len], tokenizer)
    answer_tokens = decode_all_tokens(input_ids[inp_len:], tokenizer)

    num_layers, num_heads, prompt_len = prompt_attn.shape

    # NO FILTERING - Keep all tokens for full attention analysis
    # This allows exploration of how model attends to ALL tokens (including format tokens)
    # across different layers during reasoning process

    # Per-layer decisive tokens (ALL prompt tokens, no filtering)
    per_layer_decisive = []
    per_layer_attn_entropy = []
    per_layer_attn_matrix = []  # Store full attention matrix for each layer

    for layer_idx in range(num_layers):
        layer_attn = prompt_attn[layer_idx]  # (num_heads, prompt_len)

        # Average attention across heads
        avg_attn = layer_attn.mean(axis=0)  # (prompt_len,)
        
        # Compute attention entropy (uncertainty in attention distribution)
        dist = avg_attn / (avg_attn.sum() + 1e-10)
        layer_ent = -(dist * np.log(dist + 1e-10)).sum()
        per_layer_attn_entropy.append(float(layer_ent))

        # Store full attention distribution for this layer
        per_layer_attn_matrix.append(avg_attn.copy())

        # Extract top-k most attended tokens (ALL tokens, no filtering)
        layer_decisive = []
        top_positions = np.argsort(avg_attn)[::-1][:top_k_tokens]
        for pos in top_positions:
            tid = input_ids[pos].item()
            token_text = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
            layer_decisive.append({
                "token_text": token_text,
                "token_id": tid,
                "position": int(pos),
                "avg_attention": float(avg_attn[pos]),
                "attention_rank": int(np.where(np.argsort(avg_attn)[::-1] == pos)[0][0] + 1),
            })
        per_layer_decisive.append((layer_idx, layer_decisive))

    # Last layer analysis
    last_layer_attn = prompt_attn[-1]
    last_layer_decisive = extract_decisive_tokens_from_attn(
        last_layer_attn, input_ids, tokenizer, top_k_tokens
    )

    return {
        "input_text": input_text,
        "entropy": entropy,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "seq_len": prompt_len,
        "analyze_pos": analyze_pos,
        "first_content_pos": first_content_pos,
        "input_length": inp_len,
        "total_len": total_len,
        "prompt_tokens": prompt_tokens,
        "answer_tokens": answer_tokens,
        "candidates": candidates,
        "all_tokens": all_tokens[:total_len],
        "all_layers_attn": prompt_attn,  # (num_layers, num_heads, prompt_len)
        "per_layer_decisive": per_layer_decisive,
        "per_layer_attn_entropy": per_layer_attn_entropy,
        "per_layer_attn_matrix": per_layer_attn_matrix,  # List of (prompt_len,) arrays
        "per_head_attn": last_layer_attn,
        "decisive_tokens": last_layer_decisive,
        "per_step_data": fp_result.get("per_step_data"),
    }


# ============================================================
# 6. Visualizations
# ============================================================

def plot_attention_heatmap_grid(samples: List[Dict], save_path: str, max_display: int = 9):
    """Grid of per-sample attention heatmaps: heads × input tokens.

    Each subplot shows which input tokens each attention head focuses on
    at the generation position (last input token).
    """
    n = min(len(samples), max_display)
    if n == 0:
        print("  [SKIP] No high-entropy samples for heatmap grid")
        return

    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 3.5 * rows))
    if n == 1:
        axes = np.array([[axes]])
    axes_flat = axes.flatten()

    for idx in range(n):
        ax = axes_flat[idx]
        s = samples[idx]

        attn = s["per_head_attn"]   # (num_heads, seq_len)

        # For readability: show top-30 most-attended positions only
        avg_a = attn.mean(axis=0)
        if attn.shape[1] > 30:
            top_pos = np.argsort(avg_a)[-30:]
            top_pos = np.sort(top_pos)
            attn_disp = attn[:, top_pos]
            tok_labels = [s["all_tokens"][p] for p in top_pos]
        else:
            attn_disp = attn
            tok_labels = s["all_tokens"]

        # Abbreviate labels
        short_labels = [t[:10].replace('\n', '↵') for t in tok_labels]

        sns.heatmap(attn_disp, ax=ax, cmap="YlOrRd", cbar=False,
                    xticklabels=short_labels, yticklabels=False)

        ax.set_xticks(range(len(short_labels)))
        ax.set_xticklabels(short_labels, rotation=55, ha='right', fontsize=5)
        ax.set_title(f"H={s['entropy']:.2f}  "
                     f"top: {s['candidates'][0]['token']}={s['candidates'][0]['prob']:.2f}  "
                     f"{s['candidates'][1]['token']}={s['candidates'][1]['prob']:.2f}",
                     fontsize=8)
        ax.set_xlabel("Input tokens", fontsize=6)
        if idx % cols == 0:
            ax.set_ylabel("Attn heads", fontsize=7)

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Decisive Token Heatmaps — Last Layer Attention at Generation Position\n"
                 "(Rows = attention heads, Columns = input tokens, Color = attention weight)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved heatmap grid: {save_path}")


def plot_decisive_token_ranking(samples: List[Dict], save_path: str, top_n: int = 25):
    """Bar chart: most frequently decisive tokens across all high-entropy samples.

    For each sample, its top-5 attended tokens are counted. The tokens that
    appear most often in top-5 across samples are the "consistently decisive" tokens.
    """
    token_counter = Counter()

    for s in samples:
        for dt in s["decisive_tokens"][:5]:
            text = dt["token_text"].strip()
            if text:
                token_counter[text] += 1

    top_tokens = token_counter.most_common(top_n)
    if not top_tokens:
        print("  [SKIP] No decisive tokens for ranking")
        return

    labels, counts = zip(*top_tokens)
    labels = list(labels)[::-1]
    counts = list(counts)[::-1]

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.45), max(6, len(labels) * 0.28)))

    colors = plt.cm.YlOrRd(np.linspace(0.25, 0.85, len(labels)))[::-1]
    bars = ax.barh(range(len(labels)), counts, color=colors, edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Frequency (times ranked in top-5 attended)", fontsize=10)
    ax.set_title("Decisive Token Ranking — Most Attended Input Tokens\n"
                 "(aggregated across all high-entropy samples)",
                 fontsize=12)

    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                str(cnt), va='center', fontsize=8, color='#555')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved ranking: {save_path}")


def plot_entropy_vs_attention(samples: List[Dict], save_path: str):
    """Scatter: output entropy vs attention concentration.

    X-axis: Shannon entropy of the output distribution
    Y-axis: how concentrated the attention is (top-3 attention sum)

    Insight: Are uncertain samples more diffuse (attending to everything)
    or more focused (attending to specific misleading tokens)?
    """
    entropies, top3_conc, top1_attn = [], [], []

    for s in samples:
        attn = s["per_head_attn"]       # (num_heads, seq_len)
        avg = attn.mean(axis=0)          # (seq_len,)
        sorted_a = np.sort(avg)[::-1]
        entropies.append(s["entropy"])
        top3_conc.append(sorted_a[:3].sum())
        top1_attn.append(sorted_a[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.scatter(entropies, top3_conc, alpha=0.6, s=35, c='coral', edgecolors='white', linewidth=0.5)
    ax1.set_xlabel("Output Entropy (nats)")
    ax1.set_ylabel("Top-3 Attention Sum")
    ax1.set_title("Entropy vs Attention Concentration\n(sum of top-3 attended positions)")
    ax1.grid(True, alpha=0.3)

    # Trend line (skip if entropy variance is too low — polyfit becomes ill-conditioned)
    if len(entropies) > 2 and np.std(entropies) > 1e-6:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.RankWarning)
            z = np.polyfit(entropies, top3_conc, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(entropies), max(entropies), 50)
        ax1.plot(x_line, p(x_line), '--', color='gray', alpha=0.6, linewidth=1)

    ax2.scatter(entropies, top1_attn, alpha=0.6, s=35, c='steelblue', edgecolors='white', linewidth=0.5)
    ax2.set_xlabel("Output Entropy (nats)")
    ax2.set_ylabel("Max Single Attention Weight")
    ax2.set_title("Entropy vs Top-1 Attention\n(most attended single token)")
    ax2.grid(True, alpha=0.3)

    if len(entropies) > 2 and np.std(entropies) > 1e-6:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.RankWarning)
            z = np.polyfit(entropies, top1_attn, 1)
        p = np.poly1d(z)
        ax2.plot(x_line, p(x_line), '--', color='gray', alpha=0.6, linewidth=1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved entropy vs attention: {save_path}")


def plot_head_specialization(samples: List[Dict], save_path: str):
    """Analyze head specialization: which heads are focused vs diffuse?

    For each head, compute the entropy of its attention distribution at the
    generation position, averaged across all high-entropy samples.

    - Low entropy head → focused on specific tokens (specialist)
    - High entropy head → spread across many tokens (generalist)
    """
    head_entropies = defaultdict(list)

    for s in samples:
        attn = s["per_head_attn"]   # (num_heads, seq_len)
        for h in range(attn.shape[0]):
            dist = attn[h] / (attn[h].sum() + 1e-10)
            ent = -(dist * np.log(dist + 1e-10)).sum()
            head_entropies[h].append(ent)

    if not head_entropies:
        print("  [SKIP] No head data for specialization analysis")
        return

    heads = sorted(head_entropies.keys())
    avg_ent = [np.mean(head_entropies[h]) for h in heads]
    std_ent = [np.std(head_entropies[h]) for h in heads]

    fig, ax = plt.subplots(figsize=(max(10, len(heads) * 0.4), 5))

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.85, len(heads)))
    bars = ax.bar(range(len(heads)), avg_ent, yerr=std_ent, color=colors,
                  edgecolor='white', linewidth=0.5, capsize=2)

    ax.set_xlabel("Attention Head Index")
    ax.set_ylabel("Avg Attention Entropy (nats)")
    ax.set_title("Head Specialization at Generation Position\n"
                 "(Low = focused on few tokens, High = spread across many)")
    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels([f"H{h}" for h in heads], fontsize=6, rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate extremes
    min_idx = int(np.argmin(avg_ent))
    max_idx = int(np.argmax(avg_ent))
    ax.annotate(f"Most focused\nH{heads[min_idx]} (H={avg_ent[min_idx]:.1f})",
                xy=(min_idx, avg_ent[min_idx]),
                xytext=(min_idx + max(2, len(heads) // 10), avg_ent[min_idx] + 0.4),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=8, color='green')
    ax.annotate(f"Most diffuse\nH{heads[max_idx]} (H={avg_ent[max_idx]:.1f})",
                xy=(max_idx, avg_ent[max_idx]),
                xytext=(max_idx - max(2, len(heads) // 10), avg_ent[max_idx] - 0.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=8, color='red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved head specialization: {save_path}")


def plot_candidate_attention_map(samples: List[Dict], save_path: str, max_samples: int = 20):
    """For each sample's top-2 competing candidates, show which decisive tokens
    are associated with which candidate via attention heads.

    Concept: different attention heads may "support" different candidates.
    If candidate A and B have similar probability, head H1 (which focuses on
    tokens related to A) competes with head H2 (which focuses on tokens related to B).

    We show: for the top-2 candidates per sample, list the top-3 decisive tokens
    and which heads attended to them most strongly.
    """
    if not samples:
        return

    n_show = min(len(samples), max_samples)
    # Ensure minimum figure height so text annotations don't overflow
    fig_height = max(5, n_show * 0.8)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    y_pos = 0
    y_labels = []

    for s_idx, s in enumerate(samples[:n_show]):
        c1 = s["candidates"][0]
        c2 = s["candidates"][1] if len(s["candidates"]) > 1 else {"token": "?", "prob": 0}
        dt = s["decisive_tokens"][:5]

        # Token string with attention info (truncate each token to 12 chars)
        token_strs = []
        for d in dt:
            t = (d["token_text"].strip() or repr(d["token_text"]))[:12]
            token_strs.append(f"{t}({d['avg_attention']:.3f})")
        tokens_text = " | ".join(token_strs)

        label = f"S{s_idx}: H={s['entropy']:.2f}"
        y_labels.append(label)
        ax.barh(y_pos, c1["prob"], height=0.6, color='coral', alpha=0.8,
                label=f'{c1["token"]}' if s_idx == 0 else "")
        ax.barh(y_pos - 0.7, c2["prob"], height=0.6, color='steelblue', alpha=0.8,
                label=f'{c2["token"]}' if s_idx == 0 else "")

        # Annotate decisive tokens — clip to right edge of axes to prevent overflow
        ax.text(min(max(c1["prob"], c2["prob"]) + 0.02, 0.85), y_pos - 0.35,
                tokens_text, va='center', fontsize=5, color='#333',
                family='monospace', clip_on=True)

        y_pos -= 2.0

    # Set y-tick positions to match the midpoints of each sample pair
    ytick_positions = [y - 0.35 for y in range(0, -2 * n_show, -2)]
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(reversed(y_labels[:n_show]), fontsize=7)
    ax.set_xlabel("Probability")
    ax.set_title("Candidate Competition & Decisive Tokens\n"
                 "(orange = top-1 candidate, blue = top-2, text = top-5 decisive tokens with avg attn)",
                 fontsize=11)
    ax.set_xlim(0, 1.15)  # Ensure space for annotation text
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.2, axis='x')

    # Use constrained_layout instead of tight_layout to avoid the overflow warning
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved candidate attention map: {save_path}")


# ============================================================
# 6b. Multi-Layer Visualizations
# ============================================================

def plot_layer_attention_evolution(samples: List[Dict], save_path: str, max_samples: int = 6):
    """How attention focus evolves across layers for each high-entropy sample.

    For each sample, compute per-layer attention entropy (how focused the
    average attention is at that layer). Plot as a line chart: layer (x) vs
    attention entropy (y), one line per sample.

    Insight: do early layers have diffuse attention (processing all tokens)
    that gradually focuses, or does it stay diffuse → sudden focus at the end?
    """
    if not samples:
        print("  [SKIP] No samples for layer evolution")
        return

    fig, ax = plt.subplots(figsize=(12, max(5, max_samples * 0.5)))

    for idx, s in enumerate(samples[:max_samples]):
        n_layers = len(s["per_layer_attn_entropy"])
        layers = list(range(n_layers))
        ent = s["per_layer_attn_entropy"]
        ax.plot(layers, ent, 'o-', markersize=3, linewidth=1.5, alpha=0.7,
                label=f"S{idx} (out_H={s['entropy']:.2f})")

    ax.set_xlabel("Layer Index", fontsize=10)
    ax.set_ylabel("Attention Entropy at Generation Position (nats)", fontsize=10)
    ax.set_title("Attention Focus Evolution Across Layers\n"
                 "(Low = focused on few tokens, High = spread across many)",
                 fontsize=12)
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # low entropy (focused) at top

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved layer evolution: {save_path}")


def plot_layer_decisive_token_heatmap(samples: List[Dict], save_path: str,
                                       max_samples: int = 4, top_positions: int = 15):
    """For each sample, show a layer × token heatmap of average attention.

    Rows = layers (0 to L-1), Columns = top-N most-attended input tokens (anywhere).
    Color = average attention weight.

    Insight: does a specific token "light up" at a particular layer, then fade?
    Or does it stay consistently attended across all layers?
    """
    if not samples:
        print("  [SKIP] No samples for layer-token heatmap")
        return

    n = min(len(samples), max_samples)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 8))
    if n == 1:
        axes = [axes]

    for sample_idx, s in enumerate(samples[:n]):
        ax = axes[sample_idx]
        all_layers_attn = s["all_layers_attn"]  # (num_layers, num_heads, seq_len)

        # Average over heads: (num_layers, seq_len)
        avg_over_heads = all_layers_attn.mean(axis=1)

        # Find top positions by max attention across all layers
        max_attn_per_pos = avg_over_heads.max(axis=0)
        top_pos = np.argsort(max_attn_per_pos)[-top_positions:]
        top_pos = np.sort(top_pos)

        heatmap_data = avg_over_heads[:, top_pos]  # (num_layers, top_positions)
        tok_labels = [s["all_tokens"][p][:12].replace('\n', '↵') for p in top_pos]

        sns.heatmap(heatmap_data, ax=ax, cmap="YlOrRd",
                    xticklabels=tok_labels, yticklabels=False,
                    cbar_kws={"label": "Avg attention", "shrink": 0.8})
        ax.set_xticklabels(tok_labels, rotation=55, ha='right', fontsize=6)
        ax.set_xlabel("Input tokens", fontsize=7)
        ax.set_ylabel("Layer")
        ax.set_title(f"S{sample_idx}: H={s['entropy']:.2f}\n"
                     f"top: {s['candidates'][0]['token']}={s['candidates'][0]['prob']:.2f}",
                     fontsize=9)

    fig.suptitle("Layer × Token Attention Heatmap — Which tokens are attended at which layer?\n"
                 "(Rows = layers from bottom to top, Columns = top-attended tokens)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved layer-token heatmap: {save_path}")


def plot_layer_consistency_ranking(samples: List[Dict], save_path: str, top_n: int = 20):
    """Across all high-entropy samples, which tokens are decisive at which layers?

    For each layer, count how often each token appears in the layer's top-3.
    Then show: for the overall most-decisive tokens, at which layers they are top-3.

    This reveals whether certain tokens are "early deciders" (attended in layers 0-5)
    or "late deciders" (only attended in the final layers).
    """
    if not samples:
        print("  [SKIP] No samples for layer consistency")
        return

    # Collect per-layer token frequency
    # layer_token_freq[layer_idx][token_text] = count
    layer_token_freq = defaultdict(lambda: defaultdict(int))

    for s in samples:
        for layer_idx, decisive_list in s["per_layer_decisive"]:
            for dt in decisive_list[:3]:  # top-3 per layer
                text = dt["token_text"].strip()
                if text:
                    layer_token_freq[layer_idx][text] += 1

    # Find global top-N tokens (summed across all layers)
    global_counter = Counter()
    for layer_idx, tf in layer_token_freq.items():
        for token, cnt in tf.items():
            global_counter[token] += cnt

    top_global = [t for t, _ in global_counter.most_common(top_n)]
    if not top_global:
        print("  [SKIP] No tokens for layer consistency")
        return

    # Build matrix: (top_n_tokens, num_layers)
    all_layers = sorted(layer_token_freq.keys())
    matrix = np.zeros((len(top_global), len(all_layers)))

    for row, token in enumerate(top_global):
        for col, layer_idx in enumerate(all_layers):
            matrix[row, col] = layer_token_freq[layer_idx].get(token, 0)

    # Plot
    fig, ax = plt.subplots(figsize=(max(10, len(all_layers) * 0.4),
                                     max(6, len(top_global) * 0.35)))

    sns.heatmap(matrix, ax=ax, cmap="YlOrRd",
                xticklabels=[f"L{l}" for l in all_layers],
                yticklabels=top_global, cbar_kws={"label": "Top-3 frequency", "shrink": 0.8})
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Token")
    ax.set_title("Decisive Token × Layer Frequency Matrix\n"
                 "(How often each token is in top-3 attended at each layer, across all samples)",
                 fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved layer consistency: {save_path}")


# ============================================================
# 6c. Per-Step Attention Evolution Visualizations
# ============================================================

def plot_generation_attention_evolution(samples: List[Dict], save_path: str,
                                         max_samples: int = 6, top_prompt_tokens: int = 8):
    """How attention to key prompt tokens evolves across generation steps.

    For each high-entropy sample, the model generates a sequence of tokens
    (answer letter for MC, chain-of-thought for GSM8K).  This plot shows,
    at each generation step, how much attention the model gives to the top-K
    most-attended prompt tokens.

    Insight: does the model's focus shift from one part of the question to
    another as it reasons?  Or does it consistently attend to the same tokens?
    """
    samples_with_steps = [s for s in samples if s.get("per_step_data") is not None]
    if not samples_with_steps:
        print("  [SKIP] No per-step attention data for generation evolution plot")
        return

    n = min(len(samples_with_steps), max_samples)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), squeeze=False)
    axes_flat = axes.flatten()

    for idx, s in enumerate(samples_with_steps[:n]):
        ax = axes_flat[idx]
        psd = s["per_step_data"]
        num_steps = psd["num_steps"]
        gen_tokens = psd["gen_tokens"]
        per_step_entropy = psd["per_step_entropy"]
        avg_prompt_attn = psd["avg_prompt_attn"]  # (num_steps, prompt_len)

        # Build content mask to exclude format tokens from ranking
        input_ids = s.get("_input_ids")  # will be set by caller
        content_mask = np.ones(avg_prompt_attn.shape[1], dtype=bool)
        if input_ids is not None:
            tokenizer = None  # not available here; use simple heuristic
            # Filter positions with near-zero average attention across all steps
            step_mean = avg_prompt_attn.mean(axis=0)
            content_mask = step_mean > np.percentile(step_mean, 10)

        # Find top-K prompt tokens by max attention across steps
        max_attn_per_pos = avg_prompt_attn[:, content_mask].max(axis=0) if content_mask.any() else avg_prompt_attn.max(axis=0)
        if content_mask.any():
            content_indices = np.where(content_mask)[0]
            top_local = np.argsort(max_attn_per_pos)[-top_prompt_tokens:]
            top_positions = content_indices[top_local]
        else:
            top_positions = np.argsort(avg_prompt_attn.max(axis=0))[-top_prompt_tokens:]

        top_positions = np.sort(top_positions)

        # Get token labels from all_tokens
        all_tokens = s.get("all_tokens", [])
        token_labels = []
        for p in top_positions:
            if p < len(all_tokens):
                label = all_tokens[p].strip()[:15].replace('\n', '↵')
            else:
                label = f"pos{p}"
            token_labels.append(label)

        # Plot lines
        steps = list(range(num_steps))
        colors = plt.cm.tab10(np.linspace(0, 1, len(top_positions)))
        for k, (pos, label) in enumerate(zip(top_positions, token_labels)):
            ax.plot(steps, avg_prompt_attn[:, pos], '-o', markersize=2,
                    linewidth=1.5, color=colors[k], label=label, alpha=0.8)

        # Overlay entropy on twin axis
        ax2 = ax.twinx()
        ax2.bar(steps, per_step_entropy, alpha=0.15, color='gray', width=0.8, label='entropy')
        ax2.set_ylabel("Entropy (nats)", fontsize=7, color='gray')
        ax2.tick_params(axis='y', labelsize=6, colors='gray')
        ax2.set_ylim(0, max(per_step_entropy) * 1.5 + 0.1)

        # Generation token labels on x-axis
        short_gen = [t.strip()[:6].replace('\n', '↵') for t in gen_tokens]
        ax.set_xticks(steps)
        ax.set_xticklabels(short_gen, rotation=45, ha='right', fontsize=5)
        ax.set_ylabel("Avg attention to prompt", fontsize=7)
        ax.set_title(f"S{idx}: H_final={s['entropy']:.2f}  "
                     f"top={s['candidates'][0]['token']}={s['candidates'][0]['prob']:.2f}  "
                     f"({num_steps} gen steps)",
                     fontsize=9)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=5, ncol=2, framealpha=0.8)

    fig.suptitle("Generation-Step Attention Evolution\n"
                 "(Lines = attention to top prompt tokens at each step, "
                 "Bars = output entropy per step)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved generation attention evolution: {save_path}")


def plot_per_step_entropy_profile(samples: List[Dict], save_path: str,
                                   max_samples: int = 20):
    """Entropy profile across generation steps for all high-entropy samples.

    X-axis: generation step index
    Y-axis: entropy at each step
    One line per sample, colored by whether the answer is correct.

    Insight: for MC tasks, step 0 (the answer letter) should have high entropy
    for uncertain samples.  For GSM8K, entropy may spike at key reasoning steps.
    """
    samples_with_steps = [s for s in samples if s.get("per_step_data") is not None]
    if not samples_with_steps:
        print("  [SKIP] No per-step data for entropy profile")
        return

    n = min(len(samples_with_steps), max_samples)
    fig, ax = plt.subplots(figsize=(13, max(5, n * 0.25)))

    for idx, s in enumerate(samples_with_steps[:n]):
        psd = s["per_step_data"]
        steps = list(range(psd["num_steps"]))
        entropies = psd["per_step_entropy"]
        is_correct = s.get("is_correct")
        color = 'steelblue' if is_correct else 'coral'
        marker = 'o' if is_correct else 'x'
        label = f"S{idx}" if idx < 10 else ""
        ax.plot(steps, entropies, f'-{marker}', markersize=3, linewidth=1.2,
                color=color, alpha=0.7, label=label)

    # Legend proxy
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='steelblue', marker='o', linestyle='-', markersize=4, label='Correct'),
        Line2D([0], [0], color='coral', marker='x', linestyle='-', markersize=4, label='Wrong'),
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=8)

    ax.set_xlabel("Generation step", fontsize=10)
    ax.set_ylabel("Entropy (nats)", fontsize=10)
    ax.set_title("Per-Step Entropy Profile During Generation\n"
                 "(How model uncertainty evolves as it generates each token)",
                 fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved per-step entropy profile: {save_path}")


# ============================================================
# 7. Text Report
# ============================================================

def generate_text_report(samples: List[Dict], save_path: str, max_display: int = 20):
    """Generate a detailed text report for each high-entropy sample."""
    lines = []
    lines.append("=" * 80)
    lines.append("DECISIVE TOKEN ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append(f"\nTotal high-entropy samples: {len(samples)}")

    if samples:
        ents = [s["entropy"] for s in samples]
        lines.append(f"Entropy range: [{min(ents):.3f}, {max(ents):.3f}] nats")
        lines.append(f"Entropy mean:  {np.mean(ents):.3f} nats")
        lines.append(f"Entropy std:   {np.std(ents):.3f} nats")

    # Accuracy summary at the top
    samples_with_gt = [s for s in samples if s.get("is_correct") is not None]
    if samples_with_gt:
        correct = sum(1 for s in samples_with_gt if s["is_correct"])
        lines.append(f"\nAccuracy among high-entropy samples: {correct}/{len(samples_with_gt)} "
                     f"({100*correct/len(samples_with_gt):.1f}%)")
        lines.append(f"  Correct & uncertain: {correct}")
        lines.append(f"  Wrong & uncertain:   {len(samples_with_gt) - correct}")
    
    lines.append(f"\nAnalysis mode: ALL tokens (including format tokens like <|im_end|>, whitespace, etc.)")
    lines.append(f"This enables full exploration of attention patterns across all layers.")

    for idx, s in enumerate(samples[:max_display]):
        lines.append(f"\n{'─' * 70}")
        gt_answer = s.get("ground_truth", {}).get("answer", "?") if s.get("ground_truth") else "?"
        model_ans = s.get("model_answer", "?")
        is_correct = s.get("is_correct", None)
        correct_tag = "" if is_correct is None else (" CORRECT" if is_correct else " WRONG")
        lines.append(f"Sample {idx + 1}  |  Entropy = {s['entropy']:.3f} nats  |  "
                     f"Layers = {s['num_layers']}  |  Heads = {s['num_heads']}  |  "
                     f"Seq = {s['seq_len']}{correct_tag}")
        lines.append(f"{'─' * 70}")

        # Ground truth and model answer
        if gt_answer != "?":
            lines.append(f"  Ground truth: {gt_answer}  |  Model answer: {model_ans}")

        # Input text (truncated for readability)
        inp = s["input_text"]
        if len(inp) > 300:
            lines.append(f"Input: {inp[:300]}...")
        else:
            lines.append(f"Input: {inp}")

        # Candidates
        lines.append("\n  Competing candidates (top-5 by probability):")
        for rank, c in enumerate(s["candidates"], 1):
            bar = "█" * int(c["prob"] * 40)
            lines.append(f"    {rank}. '{c['token']}'  "
                         f"P = {c['prob']:.4f}  "
                         f"logit_id = {c['token_id']}  {bar}")

        # Multi-layer decisive token summary
        lines.append(f"\n  Layer attention entropy (how focused each layer is):")
        entropies = s["per_layer_attn_entropy"]
        # Find most and least focused layers
        min_ent_layer = int(np.argmin(entropies))
        max_ent_layer = int(np.argmax(entropies))
        lines.append(f"    Most focused:    L{min_ent_layer} (H={entropies[min_ent_layer]:.3f})")
        lines.append(f"    Most diffuse:    L{max_ent_layer} (H={entropies[max_ent_layer]:.3f})")
        lines.append(f"    Last layer (L{s['num_layers']-1}): H={entropies[-1]:.3f}")

        # Per-layer top-3 decisive tokens
        lines.append(f"\n  Per-layer top-3 decisive tokens:")
        # Show a few key layers: first, middle, most-focused, last
        key_layers = set()
        key_layers.add(0)                          # first layer
        key_layers.add(s["num_layers"] - 1)       # last layer
        key_layers.add(s["num_layers"] // 2)      # middle layer
        key_layers.add(min_ent_layer)             # most focused
        key_layers = sorted(key_layers)

        for layer_idx, decisive_list in s["per_layer_decisive"]:
            if layer_idx not in key_layers:
                continue
            marker = ""
            if layer_idx == 0:
                marker = " [FIRST]"
            elif layer_idx == s["num_layers"] - 1:
                marker = " [LAST]"
            elif layer_idx == min_ent_layer:
                marker = " [MOST FOCUSED]"
            elif layer_idx == s["num_layers"] // 2:
                marker = " [MIDDLE]"

            tokens_str = " | ".join(
                [f"[{dt['token_text'].strip() or '?'}]({dt['avg_attention']:.4f})"
                 for dt in decisive_list[:3]]
            )
            lines.append(f"    L{layer_idx:2d}{marker}: {tokens_str}")

        # Last-layer detailed decisive tokens (for reference)
        lines.append(f"\n  Last-layer decisive tokens (top-8, detailed):")
        for rank, dt in enumerate(s["decisive_tokens"][:8], 1):
            t = dt["token_text"].strip() or repr(dt["token_text"])
            lines.append(f"    {rank}. [{t}]  pos={dt['position']:4d}  "
                         f"avg_attn={dt['avg_attention']:.5f}")

            if rank <= 3:
                hw = dt["per_head_attention"]
                top_heads = sorted(range(len(hw)), key=lambda h: hw[h], reverse=True)[:5]
                head_str = "  ".join([f"H{h}={hw[h]:.4f}" for h in top_heads])
                lines.append(f"        Top heads: {head_str}")

        # Per-step generation summary (new)
        psd = s.get("per_step_data")
        if psd and psd.get("num_steps", 0) > 1:
            lines.append(f"\n  Per-step generation analysis ({psd['num_steps']} steps):")
            gen_tokens = psd["gen_tokens"]
            per_step_ent = psd["per_step_entropy"]
            for step_i in range(min(len(gen_tokens), 12)):
                tok = gen_tokens[step_i].strip()[:10].replace('\n', '↵')
                ent_val = per_step_ent[step_i] if step_i < len(per_step_ent) else -1
                bar = "█" * int(ent_val * 8) if ent_val > 0 else ""
                lines.append(f"    Step {step_i:2d}: '{tok}'  H={ent_val:.3f}  {bar}")
            if len(gen_tokens) > 12:
                lines.append(f"    ... ({len(gen_tokens) - 12} more steps)")

    lines.append(f"\n{'=' * 80}")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Saved text report: {save_path}")
    return report


# ============================================================
# 8. JSON Data Export
# ============================================================

def save_analysis_json(samples: List[Dict], save_path: str):
    """Save structured analysis data as JSON for further processing."""
    json_data = []
    for s in samples:
        json_data.append({
            "sample_idx": int(s.get("sample_idx", -1)),
            "input_text": s["input_text"],
            "entropy": float(s["entropy"]),
            "num_layers": int(s["num_layers"]),
            "num_heads": int(s["num_heads"]),
            "seq_len": int(s["seq_len"]),
            "input_length": int(s.get("input_length", s["seq_len"])),
            "analyze_pos": int(s.get("analyze_pos", -1)),
            "first_content_pos": int(s.get("first_content_pos", -1)),
            "total_len": int(s.get("total_len", s["seq_len"])),
            "ground_truth_answer": s.get("ground_truth", {}).get("answer", "") if s.get("ground_truth") else "",
            "model_answer": s.get("model_answer", ""),
            "is_correct": s.get("is_correct", None),
            "candidates": s["candidates"],
            "per_layer_attn_entropy": [float(x) for x in s["per_layer_attn_entropy"]],
            "per_layer_top3": {
                str(layer_idx): [dt["token_text"].strip() for dt in decisive[:3]]
                for layer_idx, decisive in s["per_layer_decisive"]
            },
            "last_layer_decisive_tokens": s["decisive_tokens"],
            "per_step_entropy": s.get("per_step_data", {}).get("per_step_entropy", []),
            "per_step_gen_tokens": s.get("per_step_data", {}).get("gen_tokens", []),
        })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Use default handler to catch any remaining numpy/torch types
    def json_default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=json_default)
    print(f"  Saved JSON data: {save_path}")


# ============================================================
# 9. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step 6: Decisive Token Analysis — find the input tokens that the model's "
                    "last-layer attention focuses on when making uncertain predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (standalone):
  python step6_decisive_token.py --model qwen7B
  python step6_decisive_token.py --model qwen7B --entropy_percentile 95
  python step6_decisive_token.py --model qwen7B --entropy_threshold 0.5
  python step6_decisive_token.py --model qwen3B --max_display 20
        """,
    )
    parser.add_argument("--model", type=str, default="qwen7B",
                        choices=["qwen1.5B", "qwen3B", "qwen7B"],
                        help="Model to analyze")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to process")
    parser.add_argument("--entropy_threshold", type=float, default=None,
                        help="Fixed min entropy (nats). Mutually exclusive with --entropy_percentile.")
    parser.add_argument("--entropy_percentile", type=float, default=90,
                        help="Auto-select threshold: top (100-P)%% most uncertain samples. "
                             "Default: 90 → top 10%%. Set to None to use --entropy_threshold.")
    parser.add_argument("--top_k_candidates", type=int, default=5,
                        help="Number of candidate tokens to record")
    parser.add_argument("--top_k_tokens", type=int, default=15,
                        help="Number of decisive tokens to extract per sample")
    parser.add_argument("--max_display", type=int, default=12,
                        help="Max samples in plots and text report")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Max input sequence length (truncate longer)")
    parser.add_argument("--correct_only", action="store_true", default=True,
                        help="Only analyze correct answers (default: True). "
                             "Use --no-correct_only to include wrong answers.")
    parser.add_argument("--no-correct_only", dest="correct_only", action="store_false",
                        help="Include both correct and wrong answers in analysis.")
    args = parser.parse_args()

    # Determine thresholding mode
    use_percentile = args.entropy_threshold is None and args.entropy_percentile is not None
    if args.entropy_threshold is not None and args.entropy_percentile is not None:
        print("Warning: both --entropy_threshold and --entropy_percentile set. "
              "Using --entropy_threshold.")

    print("=" * 80)
    print("Step 6: Decisive Token Analysis")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.num_samples}")
    if use_percentile:
        print(f"  Threshold mode: auto (top {100 - args.entropy_percentile:.0f}%% by entropy)")
    else:
        thr = args.entropy_threshold if args.entropy_threshold is not None else 0.5
        print(f"  Threshold mode: fixed ({thr} nats)")
    print(f"  Max display: {args.max_display}")
    print(f"  Correct only: {args.correct_only}")
    print("=" * 80)

    # Output directory
    output_dir = f"experiment_results/analysis/decisive_token/{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model_for_attention(args.model)

    # Load input texts from step2 outputs
    model_output_dir = MODELS[args.model]["output_dir"]
    print(f"\nLoading input texts from: {model_output_dir}")

    # Pre-load all (index, input_text, gt, model_answer, is_correct, generated_ids, input_length, step2_output) tuples
    # The step2_output is kept only for fast-entropy extraction in Pass 1.
    # sample_pairs: [(index, input_text, gt, model_answer, is_correct, generated_ids, input_length, step2_output), ...]
    sample_pairs = []
    from qa_utils import get_clean_answer, get_gt_answer, compare_answers

    correct_count = 0
    total_with_gt = 0
    for i in range(args.num_samples):
        output = load_model_output(model_output_dir, i)
        if output is None:
            continue
        input_text = output.get("input_text")
        if not input_text:
            continue

        # Extract ground truth and model answer using centralized get_clean_answer
        gt = output.get("ground_truth", {})
        gt_answer = get_gt_answer(gt)
        model_answer, answer_type = get_clean_answer(output)

        # Check correctness if ground truth is available
        is_correct = None
        if gt_answer:
            total_with_gt += 1
            comparison = compare_answers(model_answer, gt)
            is_correct = comparison.get("match", False)
            if is_correct:
                correct_count += 1

        # Load generated_ids and input_length for V2 forward pass
        generated_ids = output.get("generated_ids")
        input_length = None
        if generated_ids is not None:
            input_length = len(generated_ids) - output.get("num_generated_tokens", 0)
            if input_length <= 0:
                input_length = None
                generated_ids = None

        # Keep step2 output for fast-entropy path; extract only the minimal
        # data needed (logits_per_step or top_k_info) to avoid holding large
        # tensors for all 100 samples simultaneously.
        fast_data = _extract_fast_entropy_data(output)

        sample_pairs.append((i, input_text, gt, model_answer, is_correct, generated_ids, input_length, fast_data))

    print(f"  Loaded {len(sample_pairs)} valid samples")
    v2_count = sum(1 for _, _, _, _, _, gid, il, _ in sample_pairs if gid is not None)
    print(f"  V2 (answer-position): {v2_count}, V1 (prompt-only fallback): {len(sample_pairs) - v2_count}")
    fast_count_pre = sum(1 for _, _, _, _, _, _, _, fd in sample_pairs if fd is not None)
    print(f"  Fast-entropy eligible: {fast_count_pre}/{len(sample_pairs)}")
    if total_with_gt > 0:
        print(f"  Accuracy: {correct_count}/{total_with_gt} ({100*correct_count/total_with_gt:.1f}%)")

    # Filter: correct_only — only analyze samples where the model answered correctly
    if args.correct_only:
        before_count = len(sample_pairs)
        sample_pairs = [(i, txt, gt, ma, ic, gid, il, fd)
                        for i, txt, gt, ma, ic, gid, il, fd in sample_pairs
                        if ic is None or ic]  # keep correct + unknown (no GT)
        skipped = before_count - len(sample_pairs)
        print(f"  [correct_only] Filtered out {skipped} wrong-answer samples, "
              f"{len(sample_pairs)} remaining")
        if skipped > 0:
            rem_with_gt = [p for p in sample_pairs if p[4] is not None]
            rem_correct = sum(1 for p in rem_with_gt if p[4])
            print(f"  [correct_only] Remaining accuracy: {rem_correct}/{len(rem_with_gt)}")

    print()

    # ================================================================
    # Pass 1: Fast entropy scan — prefer step2 data (NO forward pass)
    # ================================================================
    filter_note = " [correct_only]" if args.correct_only else ""
    print(f"Pass 1/2: Scanning entropy distribution ({len(sample_pairs)} samples){filter_note}...")
    print(f"  Strategy: use step2 scores first, forward pass as fallback")
    sample_entropies = []  # [(index, input_text, entropy, gt, model_answer, is_correct, generated_ids, input_length), ...]
    nan_count = 0
    fast_hit = 0
    fp_fallback = 0
    for idx, input_text, gt, model_answer, is_correct, gen_ids, inp_len, fast_data in tqdm(sample_pairs, desc="  Scanning"):
        try:
            # --- Fast path: reuse step2's generation scores ---
            ent = None
            source = None
            if fast_data is not None:
                ent, source = _fast_entropy_from_step2(fast_data)
                if ent is not None:
                    fast_hit += 1

            # --- Fallback: full forward pass ---
            if ent is None:
                ent = quick_entropy_scan(model, tokenizer, input_text,
                                          generated_ids=gen_ids, input_length=inp_len,
                                          max_length=args.max_seq_length)
                fp_fallback += 1

            if np.isnan(ent):
                nan_count += 1
            else:
                sample_entropies.append((idx, input_text, ent, gt, model_answer, is_correct, gen_ids, inp_len))
        except Exception as e:
            print(f"    Sample {idx} failed: {e}")
            nan_count += 1

    # Entropy distribution report
    valid_entropies = np.array([e for _, _, e, _, _, _, _, _ in sample_entropies])

    print(f"\n{'=' * 60}")
    print(f"Pass 1 complete: {len(sample_entropies)} valid, {nan_count} NaN")
    print(f"  Fast path (step2 scores): {fast_hit}, Forward pass fallback: {fp_fallback}")
    if len(valid_entropies) > 0:
        print(f"  Entropy distribution (all valid samples):")
        print(f"    min={valid_entropies.min():.3f}  median={np.median(valid_entropies):.3f}  "
              f"mean={valid_entropies.mean():.3f}  max={valid_entropies.max():.3f}  "
              f"std={valid_entropies.std():.3f}")
        for p in [25, 50, 75, 90, 95, 99]:
            print(f"    P{p} = {np.percentile(valid_entropies, p):.3f}")
        for thr in [0.1, 0.3, 0.5, 1.0, 1.5, 2.0]:
            cnt = int(np.sum(valid_entropies >= thr))
            print(f"    H >= {thr}: {cnt}/{len(valid_entropies)} ({100*cnt/len(valid_entropies):.1f}%)")

    # Determine threshold
    if use_percentile and len(valid_entropies) > 0:
        threshold = float(np.percentile(valid_entropies, args.entropy_percentile))
        print(f"\n  Auto threshold: P{args.entropy_percentile:.0f} = {threshold:.3f} nats")
    else:
        threshold = args.entropy_threshold if args.entropy_threshold is not None else 0.5
        print(f"\n  Fixed threshold: {threshold:.3f} nats")

    # Select high-entropy samples for Pass 2
    high_entropy_pairs = [(idx, text, ent, gt, model_answer, is_correct, gen_ids, inp_len)
                          for idx, text, ent, gt, model_answer, is_correct, gen_ids, inp_len in sample_entropies
                          if ent >= threshold]
    # Sort by entropy descending
    high_entropy_pairs.sort(key=lambda x: x[2], reverse=True)

    # Accuracy breakdown for high-entropy samples
    he_with_gt = [(idx, text, ent, gt, ma, ic, gi, il) for idx, text, ent, gt, ma, ic, gi, il in high_entropy_pairs if ic is not None]
    if he_with_gt:
        he_correct = sum(1 for _, _, _, _, _, ic, _, _ in he_with_gt if ic)
        print(f"  High-entropy accuracy: {he_correct}/{len(he_with_gt)} ({100*he_correct/len(he_with_gt):.1f}%)")
        he_wrong = [(idx, text, ent, gt, ma, ic, gi, il) for idx, text, ent, gt, ma, ic, gi, il in he_with_gt if not ic]
        print(f"  High-entropy wrong:   {len(he_wrong)}/{len(he_with_gt)}")

    n_high = len(high_entropy_pairs)
    print(f"  Selected: {n_high} samples with H >= {threshold:.3f} "
          f"({100*n_high/max(len(valid_entropies),1):.1f}% of valid)")
    print(f"{'=' * 60}")

    if n_high == 0:
        print("\nNo high-entropy samples found. Try lowering the threshold:")
        if len(valid_entropies) > 0:
            print(f"  Suggestion: --entropy_threshold {valid_entropies.max():.2f}  "
                  f"(includes top-1 sample)")
            print(f"  Suggestion: --entropy_percentile 95  "
                  f"(P95={np.percentile(valid_entropies, 95):.3f})")
        print(f"\nAll outputs saved to: {output_dir}/")
        print("=" * 80)
        return

    # ================================================================
    # Pass 2: Detailed attention analysis (ONLY for high-entropy samples)
    # ================================================================
    print(f"\nPass 2/2: Detailed attention analysis ({n_high} high-entropy samples)...")
    high_entropy_samples = []
    all_entropies = []
    for idx, input_text, ent, gt, model_answer, is_correct, gen_ids, inp_len in tqdm(high_entropy_pairs, desc="  Analyzing"):
        result = analyze_one_sample(
            model, tokenizer, input_text,
            top_k_candidates=args.top_k_candidates,
            top_k_tokens=args.top_k_tokens,
            generated_ids=gen_ids,
            input_length=inp_len,
            max_length=args.max_seq_length,
            extract_per_step=True,
        )
        if result is not None:
            # Attach ground truth and correctness info
            result["sample_idx"] = idx
            result["ground_truth"] = gt
            result["model_answer"] = model_answer
            result["is_correct"] = is_correct
            high_entropy_samples.append(result)
            all_entropies.append(result["entropy"])

    # Accuracy summary in Pass 2
    he2_with_gt = [s for s in high_entropy_samples if s["is_correct"] is not None]
    if he2_with_gt:
        he2_correct = sum(1 for s in he2_with_gt if s["is_correct"])
        print(f"  Pass2 accuracy: {he2_correct}/{len(he2_with_gt)} ({100*he2_correct/len(he2_with_gt):.1f}%)")

    print(f"\n{'=' * 60}")
    print(f"Results: {len(high_entropy_samples)} samples analyzed successfully")
    if all_entropies:
        print(f"  Entropy range: min={min(all_entropies):.3f}  "
              f"max={max(all_entropies):.3f}  "
              f"mean={np.mean(all_entropies):.3f}  "
              f"std={np.std(all_entropies):.3f}")
    print(f"{'=' * 60}")

    if high_entropy_samples:
        # Text report
        generate_text_report(
            high_entropy_samples,
            os.path.join(output_dir, "decisive_token_report.txt"),
            max_display=args.max_display,
        )

        # JSON data
        save_analysis_json(
            high_entropy_samples,
            os.path.join(output_dir, "decisive_token_data.json"),
        )

        # Visualizations
        print("\nGenerating visualizations...")
        # --- Last-layer visualizations (original) ---
        plot_attention_heatmap_grid(
            high_entropy_samples,
            os.path.join(output_dir, "attention_heatmaps.png"),
            max_display=min(args.max_display, 9),
        )

        plot_decisive_token_ranking(
            high_entropy_samples,
            os.path.join(output_dir, "decisive_token_ranking.png"),
        )

        plot_entropy_vs_attention(
            high_entropy_samples,
            os.path.join(output_dir, "entropy_vs_attention.png"),
        )

        plot_head_specialization(
            high_entropy_samples,
            os.path.join(output_dir, "head_specialization.png"),
        )

        plot_candidate_attention_map(
            high_entropy_samples,
            os.path.join(output_dir, "candidate_attention_map.png"),
            max_samples=min(args.max_display, 20),
        )

        # --- Multi-layer visualizations (new) ---
        print("  Generating multi-layer analysis...")
        plot_layer_attention_evolution(
            high_entropy_samples,
            os.path.join(output_dir, "layer_attention_evolution.png"),
            max_samples=min(args.max_display, 6),
        )

        plot_layer_decisive_token_heatmap(
            high_entropy_samples,
            os.path.join(output_dir, "layer_token_heatmap.png"),
            max_samples=min(4, len(high_entropy_samples)),
            top_positions=15,
        )

        plot_layer_consistency_ranking(
            high_entropy_samples,
            os.path.join(output_dir, "layer_consistency_ranking.png"),
            top_n=20,
        )

        # --- Per-step generation visualizations (new) ---
        print("  Generating per-step generation analysis...")
        plot_generation_attention_evolution(
            high_entropy_samples,
            os.path.join(output_dir, "generation_attention_evolution.png"),
            max_samples=min(args.max_display, 6),
            top_prompt_tokens=8,
        )

        plot_per_step_entropy_profile(
            high_entropy_samples,
            os.path.join(output_dir, "per_step_entropy_profile.png"),
            max_samples=min(len(high_entropy_samples), 20),
        )

    # Cleanup
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nAll outputs saved to: {output_dir}/")
    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
