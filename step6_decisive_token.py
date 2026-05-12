"""
Step 6: Decisive Token Analysis (V1)

Identify "decisive tokens" — the input tokens that receive the most attention
from the model's last layer when the model is uncertain about its answer.

Core Question:
When the final probability distribution has high entropy (multiple candidates
competing with similar probability), which input tokens did the model's last-layer
query vectors (Q) attend to the most? These are the "decisive tokens" — the
information the model relied on to make its uncertain decision.

Methodology:
1. Load model + tokenizer (reusing step2's loading pattern for transformers 5.x)
2. For each sample, run a FORWARD PASS (not generate) with output_attentions=True
3. Compute Shannon entropy of the output probability distribution
4. Filter: keep only high-entropy samples (model is uncertain)
5. Extract last-layer attention at the last input position:
   - Each attention head h has a Q vector at the last position
   - Q_h attends to all input positions via softmax(Q_h @ K^T / sqrt(d))
   - The positions with highest attention are the "decisive tokens" for head h
6. Per-head decisive tokens → aggregated ranking → "which tokens decide the answer"
7. Visualizations:
   (a) Per-sample attention heatmap (heads × input tokens)
   (b) Decisive token ranking (frequency across all high-entropy samples)
   (c) Entropy vs attention concentration scatter
   (d) Head specialization analysis (focused vs diffuse heads)
   (e) Per-sample text report with decisive tokens highlighted

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

def forward_pass_with_attention(model, tokenizer, input_text: str, max_length: int = 2048):
    """Run a forward pass on the input and capture logits + ALL layers' attention.

    This is NOT generation — it's a single forward pass to get:
    - logits at the last position (prediction for the NEXT token)
    - attention weights from ALL layers at the last position

    Returns:
        logits: (vocab_size,) float tensor on CPU (NaN-free)
        all_layers_attn: (num_layers, num_heads, seq_len) float numpy on CPU
            attention from the LAST position across all layers
        input_ids: (seq_len,) int tensor on CPU
    """
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

    # outputs.logits: (1, seq_len, vocab_size)
    # outputs.attentions: tuple of (1, num_heads, seq_len, seq_len), one per layer
    logits = outputs.logits[0, -1, :].float().cpu()          # (vocab_size,) — next-token logits

    # FIX: fp16 can produce inf/nan in logits.
    # NaN → -80 (large negative so softmax gives ~zero probability — these tokens
    #        should not influence the distribution)
    # ±inf → ±80 (prevent exp overflow while preserving sign/ordering)
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        nan_cnt = torch.isnan(logits).sum().item()
        logits = torch.nan_to_num(logits, nan=-80.0, posinf=80.0, neginf=-80.0)

    # Extract attention from LAST position across ALL layers
    # (num_layers, 1, num_heads, seq_len, seq_len) → (num_layers, num_heads, seq_len)
    all_layers_attn = np.stack(
        [layer_attn[0, :, -1, :].float().cpu().numpy() for layer_attn in outputs.attentions],
        axis=0,
    )  # (num_layers, num_heads, seq_len)

    # FIX: fp16 attention can produce NaN in specific heads (e.g., Head 13 with GQA).
    # Replace NaN with 0 (meaning "no attention from this head") so that
    # mean across heads doesn't propagate NaN into all avg_attention values.
    if np.isnan(all_layers_attn).any():
        nan_count = int(np.isnan(all_layers_attn).sum())
        all_layers_attn = np.nan_to_num(all_layers_attn, nan=0.0)

    input_ids_cpu = input_ids[0].cpu()  # (seq_len,)

    return logits, all_layers_attn, input_ids_cpu


def quick_entropy_scan(model, tokenizer, input_text: str, max_length: int = 2048) -> float:
    """Fast forward pass WITHOUT attention — just compute output entropy.

    Used in Pass 1 to scan the entropy distribution across all samples.
    Much faster than forward_pass_with_attention because attention weights
    are not materialized.

    Returns:
        entropy: float (NaN if forward pass fails)
    """
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    logits = outputs.logits[0, -1, :].float().cpu()

    # Same NaN/inf fix: nan → -80 so they get ~zero probability
    if torch.isnan(logits).any() or torch.isinf(logits).any():
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
                       max_length: int = 2048) -> Optional[Dict]:
    """Full multi-layer analysis: forward pass WITH attention, extract decisive tokens per layer.

    Assumes the caller has already determined this sample is high-entropy.
    Returns analysis dict with per-layer data or None if forward pass fails.
    """
    try:
        logits, all_layers_attn, input_ids = forward_pass_with_attention(
            model, tokenizer, input_text, max_length,
        )
    except Exception as e:
        print(f"    Forward pass failed: {e}")
        return None

    entropy = compute_entropy(logits)
    candidates = get_top_candidates(logits, tokenizer, top_k_candidates)
    all_tokens = decode_all_tokens(input_ids, tokenizer)

    num_layers, num_heads, seq_len = all_layers_attn.shape

    # Per-layer decisive tokens
    per_layer_decisive = []  # [(layer_idx, [decisive_token_dicts])] 
    per_layer_attn_entropy = []  # attention entropy per layer

    for layer_idx in range(num_layers):
        layer_attn = all_layers_attn[layer_idx]  # (num_heads, seq_len)

        # Layer attention entropy (how focused is this layer overall?)
        avg_attn = layer_attn.mean(axis=0)  # (seq_len,)
        dist = avg_attn / (avg_attn.sum() + 1e-10)
        layer_ent = -(dist * np.log(dist + 1e-10)).sum()
        per_layer_attn_entropy.append(layer_ent)

        # Top decisive tokens for this layer
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
            })
        per_layer_decisive.append((layer_idx, layer_decisive))

    # Last-layer data (for backwards-compatible single-layer visualizations)
    last_layer_attn = all_layers_attn[-1]  # (num_heads, seq_len)
    last_layer_decisive = extract_decisive_tokens_from_attn(last_layer_attn, input_ids, tokenizer, top_k_tokens)

    return {
        "input_text": input_text,
        "entropy": entropy,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "candidates": candidates,
        "all_tokens": all_tokens,
        # Multi-layer data
        "all_layers_attn": all_layers_attn,      # (num_layers, num_heads, seq_len)
        "per_layer_decisive": per_layer_decisive, # [(layer_idx, [token_dicts])]
        "per_layer_attn_entropy": per_layer_attn_entropy,
        # Last-layer data (for backwards compat)
        "per_head_attn": last_layer_attn,          # (num_heads, seq_len)
        "decisive_tokens": last_layer_decisive,
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

    for idx, s in enumerate(samples[:max_display]):
        lines.append(f"\n{'─' * 70}")
        lines.append(f"Sample {idx + 1}  |  Entropy = {s['entropy']:.3f} nats  |  "
                     f"Layers = {s['num_layers']}  |  Heads = {s['num_heads']}  |  "
                     f"Seq = {s['seq_len']}")
        lines.append(f"{'─' * 70}")

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
            "input_text": s["input_text"],
            "entropy": float(s["entropy"]),
            "num_layers": int(s["num_layers"]),
            "num_heads": int(s["num_heads"]),
            "seq_len": int(s["seq_len"]),
            "candidates": s["candidates"],
            "per_layer_attn_entropy": [float(x) for x in s["per_layer_attn_entropy"]],
            "per_layer_top3": {
                str(layer_idx): [dt["token_text"].strip() for dt in decisive[:3]]
                for layer_idx, decisive in s["per_layer_decisive"]
            },
            "last_layer_decisive_tokens": s["decisive_tokens"],
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
    print("=" * 80)

    # Output directory
    output_dir = f"experiment_results/analysis/decisive_token/{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, tokenizer = load_model_for_attention(args.model)

    # Load input texts from step2 outputs
    model_output_dir = MODELS[args.model]["output_dir"]
    print(f"\nLoading input texts from: {model_output_dir}")

    # Pre-load all (index, input_text) pairs
    sample_pairs = []  # [(index, input_text), ...]
    for i in range(args.num_samples):
        output = load_model_output(model_output_dir, i)
        if output is None:
            continue
        input_text = output.get("input_text")
        if not input_text:
            continue
        sample_pairs.append((i, input_text))

    print(f"  Loaded {len(sample_pairs)} valid samples\n")

    # ================================================================
    # Pass 1: Fast entropy scan (NO attention output — much faster)
    # ================================================================
    print(f"Pass 1/2: Scanning entropy distribution ({len(sample_pairs)} samples)...")
    sample_entropies = []  # [(index, input_text, entropy), ...]
    nan_count = 0
    for idx, input_text in tqdm(sample_pairs, desc="  Scanning"):
        try:
            ent = quick_entropy_scan(model, tokenizer, input_text, args.max_seq_length)
            if np.isnan(ent):
                nan_count += 1
            else:
                sample_entropies.append((idx, input_text, ent))
        except Exception as e:
            print(f"    Sample {idx} failed: {e}")
            nan_count += 1

    # Entropy distribution report
    valid_entropies = np.array([e for _, _, e in sample_entropies])

    print(f"\n{'=' * 60}")
    print(f"Pass 1 complete: {len(sample_entropies)} valid, {nan_count} NaN")
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
    high_entropy_pairs = [(idx, text, ent) for idx, text, ent in sample_entropies
                          if ent >= threshold]
    # Sort by entropy descending
    high_entropy_pairs.sort(key=lambda x: x[2], reverse=True)

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
    for idx, input_text, _ in tqdm(high_entropy_pairs, desc="  Analyzing"):
        result = analyze_one_sample(
            model, tokenizer, input_text,
            top_k_candidates=args.top_k_candidates,
            top_k_tokens=args.top_k_tokens,
            max_length=args.max_seq_length,
        )
        if result is not None:
            high_entropy_samples.append(result)
            all_entropies.append(result["entropy"])

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