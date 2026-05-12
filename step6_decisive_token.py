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
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="./models",
            max_memory=HARDWARE_CONFIG.get("max_memory", None),
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="./models",
            max_memory=HARDWARE_CONFIG.get("max_memory", None),
        )

    model.eval()

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
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
    """Run a forward pass on the input and capture logits + last-layer attention.

    This is NOT generation — it's a single forward pass to get:
    - logits at the last position (prediction for the NEXT token)
    - attention weights from all layers (we use the last layer)

    Returns:
        logits: (vocab_size,) float tensor on CPU
        last_layer_attn: (num_heads, seq_len, seq_len) float numpy on CPU
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
    last_layer_attn = outputs.attentions[-1][0].float().cpu().numpy()  # (heads, seq, seq)
    input_ids_cpu = input_ids[0].cpu()                         # (seq_len,)

    return logits, last_layer_attn, input_ids_cpu


# ============================================================
# 3. Entropy & Candidate Extraction
# ============================================================

def compute_entropy(logits: torch.Tensor) -> float:
    """Shannon entropy H(p) = -Σ p(x) log p(x) in nats."""
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs + 1e-30)
    return -(probs * log_probs).sum().item()


def get_top_candidates(logits: torch.Tensor, tokenizer, top_k: int = 5):
    """Get top-k candidate tokens with their probabilities."""
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, top_k)
    candidates = []
    for prob, tid in zip(top_probs, top_ids):
        token_text = tokenizer.decode([tid.item()], clean_up_tokenization_spaces=False)
        candidates.append({"token": token_text, "token_id": tid.item(), "prob": prob.item()})
    return candidates


# ============================================================
# 4. Decisive Token Extraction
# ============================================================

def extract_decisive_tokens(
    attn_matrix: np.ndarray,   # (num_heads, seq_len, seq_len)
    input_ids: torch.Tensor,    # (seq_len,)
    tokenizer,
    top_k: int = 10,
) -> List[Dict]:
    """From last-layer attention at the last position, find the most attended input tokens.

    The last position is the generation position — its Q vectors decide what
    information to aggregate before producing the next-token prediction.

    Returns:
        List of dicts: {token_text, token_id, position, avg_attention, per_head_attention}
        Sorted by avg_attention descending.
    """
    # Attention from last position to all positions: (num_heads, seq_len)
    last_pos_attn = attn_matrix[:, -1, :]

    # Average across heads: (seq_len,)
    avg_attn = last_pos_attn.mean(axis=0)

    # Top-k positions
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
            "per_head_attention": last_pos_attn[:, pos].tolist(),
        })

    return results


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
                       entropy_threshold: float,
                       top_k_candidates: int, top_k_tokens: int,
                       max_length: int = 2048) -> Optional[Dict]:
    """Run forward pass, compute entropy, extract decisive tokens if high entropy."""
    try:
        logits, attn_matrix, input_ids = forward_pass_with_attention(
            model, tokenizer, input_text, max_length,
        )
    except Exception as e:
        print(f"    Forward pass failed: {e}")
        return None

    entropy = compute_entropy(logits)
    if entropy < entropy_threshold:
        return None

    candidates = get_top_candidates(logits, tokenizer, top_k_candidates)
    decisive_tokens = extract_decisive_tokens(attn_matrix, input_ids, tokenizer, top_k_tokens)
    all_tokens = decode_all_tokens(input_ids, tokenizer)

    # Per-head attention at last position (for heatmap visualization)
    per_head_attn = attn_matrix[:, -1, :]  # (num_heads, seq_len)

    return {
        "input_text": input_text,
        "entropy": entropy,
        "num_layers": attn_matrix.shape[0],  # wait this is num_heads
        "num_heads": attn_matrix.shape[0],
        "seq_len": attn_matrix.shape[1],
        "candidates": candidates,
        "decisive_tokens": decisive_tokens,
        "all_tokens": all_tokens,
        "per_head_attn": per_head_attn,
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

    # Trend line
    if len(entropies) > 2:
        z = np.polyfit(entropies, top3_conc, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(entropies), max(entropies), 50)
        ax1.plot(x_line, p(x_line), '--', color='gray', alpha=0.6, linewidth=1)

    ax2.scatter(entropies, top1_attn, alpha=0.6, s=35, c='steelblue', edgecolors='white', linewidth=0.5)
    ax2.set_xlabel("Output Entropy (nats)")
    ax2.set_ylabel("Max Single Attention Weight")
    ax2.set_title("Entropy vs Top-1 Attention\n(most attended single token)")
    ax2.grid(True, alpha=0.3)

    if len(entropies) > 2:
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

    fig, ax = plt.subplots(figsize=(14, max(4, len(samples[:max_samples]) * 0.35)))

    y_pos = 0
    y_labels = []

    for s_idx, s in enumerate(samples[:max_samples]):
        c1 = s["candidates"][0]
        c2 = s["candidates"][1]
        dt = s["decisive_tokens"][:5]

        # Token string with attention info
        token_strs = []
        for d in dt:
            t = d["token_text"].strip() or repr(d["token_text"])
            token_strs.append(f"{t}({d['avg_attention']:.3f})")
        tokens_text = " | ".join(token_strs)

        label = f"S{s_idx}: H={s['entropy']:.2f}"
        y_labels.append(label)
        ax.barh(y_pos, c1["prob"], height=0.6, color='coral', alpha=0.8,
                label=f'{c1["token"]}' if s_idx == 0 else "")
        ax.barh(y_pos - 0.7, c2["prob"], height=0.6, color='steelblue', alpha=0.8,
                label=f'{c2["token"]}' if s_idx == 0 else "")

        # Annotate decisive tokens
        ax.text(max(c1["prob"], c2["prob"]) + 0.02, y_pos - 0.35,
                tokens_text, va='center', fontsize=5, color='#333',
                family='monospace')

        y_pos -= 2.0

    ax.set_yticks([y - 0.35 for y in range(0, -2 * min(len(samples[:max_samples]), max_samples), -2)])
    ax.set_yticklabels(reversed(y_labels[:max_samples]), fontsize=7)
    ax.set_xlabel("Probability")
    ax.set_title("Candidate Competition & Decisive Tokens\n"
                 "(orange = top-1 candidate, blue = top-2, text = top-5 decisive tokens with avg attn)",
                 fontsize=11)
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved candidate attention map: {save_path}")


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
                     f"Seq length = {s['seq_len']}  |  Heads = {s['num_heads']}")
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

        # Decisive tokens
        lines.append(f"\n  Decisive tokens (top-8 by avg attention from last position):")
        for rank, dt in enumerate(s["decisive_tokens"][:8], 1):
            t = dt["token_text"].strip() or repr(dt["token_text"])
            lines.append(f"    {rank}. [{t}]  pos={dt['position']:4d}  "
                         f"avg_attn={dt['avg_attention']:.5f}")

            # For top-3 decisive tokens, show which heads attend most
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
            "entropy": s["entropy"],
            "num_heads": s["num_heads"],
            "seq_len": s["seq_len"],
            "candidates": s["candidates"],
            "decisive_tokens": s["decisive_tokens"],
        })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
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
  python step10_decisive_token.py --model qwen7B
  python step10_decisive_token.py --model qwen7B --num_samples 200 --entropy_threshold 3.0
  python step10_decisive_token.py --model qwen3B --max_display 20
        """,
    )
    parser.add_argument("--model", type=str, default="qwen7B",
                        choices=["qwen1.5B", "qwen3B", "qwen7B"],
                        help="Model to analyze")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to process")
    parser.add_argument("--entropy_threshold", type=float, default=2.0,
                        help="Minimum output entropy (nats) to count as 'uncertain'")
    parser.add_argument("--top_k_candidates", type=int, default=5,
                        help="Number of candidate tokens to record")
    parser.add_argument("--top_k_tokens", type=int, default=15,
                        help="Number of decisive tokens to extract per sample")
    parser.add_argument("--max_display", type=int, default=12,
                        help="Max samples in plots and text report")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                        help="Max input sequence length (truncate longer)")
    args = parser.parse_args()

    print("=" * 80)
    print("Step 6: Decisive Token Analysis")
    print(f"  Model: {args.model}")
    print(f"  Samples: {args.num_samples}")
    print(f"  Entropy threshold: {args.entropy_threshold} nats")
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

    high_entropy_samples = []
    all_entropies = []

    print(f"\nProcessing {args.num_samples} samples...")
    for i in tqdm(range(args.num_samples)):
        output = load_model_output(model_output_dir, i)
        if output is None:
            continue

        input_text = output.get("input_text")
        if not input_text:
            continue

        result = analyze_one_sample(
            model, tokenizer, input_text,
            entropy_threshold=args.entropy_threshold,
            top_k_candidates=args.top_k_candidates,
            top_k_tokens=args.top_k_tokens,
            max_length=args.max_seq_length,
        )

        if result is not None:
            high_entropy_samples.append(result)
            all_entropies.append(result["entropy"])

    # Sort by entropy (most uncertain first)
    high_entropy_samples.sort(key=lambda x: x["entropy"], reverse=True)

    print(f"\n{'=' * 60}")
    print(f"Results: {len(high_entropy_samples)} high-entropy samples "
          f"(threshold >= {args.entropy_threshold} nats)")
    if all_entropies:
        print(f"  Entropy: min={min(all_entropies):.3f}  "
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