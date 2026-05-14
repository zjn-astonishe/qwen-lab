"""
Step 6c: Attention Head Ablation Analysis

哪些attention head在深层"搞破坏"？

核心方法：
  1. 加载完整模型，对每个错误样本做一次forward pass
  2. 通过hook捕获每层attention的per-head输出（o_proj的输入 = 拼接的各head输出）
  3. 用o_proj权重分解出每个head对hidden state的贡献向量:
     contribution_i = per_head_output[:, :, i, :] @ o_proj_weight[:, i*head_dim:(i+1)*head_dim].T
  4. 对每个head计算GT-Pred对齐贡献分:
     score_i = cos(contribution_i, w_GT) - cos(contribution_i, w_Pred)
     score > 0 → 该head帮助对齐GT（"好头"）
     score < 0 → 该head帮助对齐Pred（"坏头"）
  5. 聚合所有错误样本，绘制 layer × head 热力图

设计决策：
  - 分析prefill最后一个token位置的hidden state（模型读完所有输入，准备生成答案的时刻）
  - 重点分析深层（后半数层），因为step6b已证明偏差在深层引入
  - 仅需一次forward pass per sample（高效），通过后处理分解per-head贡献
  - 对三个模型（1.5B/3B/7B）分别分析，对比"坏头"分布差异

数据流：
  - 加载 sampled_300.json → 重建prompt → 模型forward → hook捕获 → 分解 → 分析
  - 依赖 step3 的 three_model_comparison.csv 识别错误样本
  - 依赖 LM Head 权重获取 w_GT / w_Pred

输出：
  - attention_head_heatmap_{model}.png  (layer × head 热力图, 每个模型一张)
  - attention_head_comparison.png      (三模型对比图)
  - bad_head_ranking.json              (按"破坏力"排序的head列表)
"""

import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict

from config import MODELS, ANALYSIS_CONFIG, HARDWARE_CONFIG
from qa_utils import (
    get_gt_answer, get_answer_type, get_clean_answer, get_answer_token_id,
)
from utils import save_json


# ---------------------------------------------------------------------------
# 模型加载 + per-head hook
# ---------------------------------------------------------------------------

def load_model_for_head_analysis(model_key: str, device: str = "cuda"):
    """加载模型（不compile，因为需要hook），返回 (model, tokenizer)。

    注意：不使用torch.compile()，因为compile会改变模块结构，导致hook失效。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = MODELS[model_key]["model_name"]
    dtype_str = HARDWARE_CONFIG.get("dtype", "bfloat16")
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)

    print(f"  Loading model {model_name} ({torch_dtype}) to {device}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir="./models"
    )

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
    print(f"    Model loaded: {model.device}, dtype: {model.dtype}")
    return model, tokenizer


def get_model_head_info(model_key: str) -> Dict[str, Any]:
    """获取模型的attention head配置信息。

    Qwen2.5使用GQA (Grouped Query Attention):
      - num_attention_heads: query头数
      - num_key_value_heads: kv头数（通常 < num_attention_heads）
      - head_dim: 每个头的维度
      - num_layers: transformer层数
    """
    from transformers import AutoConfig

    model_name = MODELS[model_key]["model_name"]
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads
    hidden_dim = config.hidden_size

    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden_dim": hidden_dim,
        "use_gqa": num_kv_heads < num_heads,
    }


# ---------------------------------------------------------------------------
# Per-head 贡献提取（核心算法）
# ---------------------------------------------------------------------------

class PerHeadContributionExtractor:
    """通过hook捕获attention per-head输出，并用o_proj分解per-head贡献。

    核心原理：
      attention层输出 = o_proj(concat(head_0_out, head_1_out, ..., head_{n-1}_out))
      其中 concat后的形状 = [batch, seq, num_heads * head_dim]
      o_proj的权重形状 = [hidden_dim, num_heads * head_dim]

      因此，head i 的贡献 = head_i_out @ o_proj_weight[:, i*head_dim:(i+1)*head_dim].T
      head_i_out 形状 = [batch, seq, head_dim]
      contribution_i 形状 = [batch, seq, hidden_dim]
    """

    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        self.hooks = []
        # storage[layer_idx] = per_head_outputs tensor [batch, seq, num_heads, head_dim]
        self.storage = {}

    def _make_hook(self, layer_idx: int, num_heads: int, head_dim: int):
        """创建hook函数，捕获o_proj的输入（即concatenated per-head outputs）。"""
        def hook_fn(module, args, kwargs, output):
            # args[0] 是 o_proj 的输入: [batch, seq, num_heads * head_dim]
            concat_output = args[0]
            batch, seq_len, _ = concat_output.shape
            # 重塑为 [batch, seq, num_heads, head_dim]
            per_head = concat_output.view(batch, seq_len, num_heads, head_dim)
            # 仅保存最后一个位置（prefill最后一个token）
            self.storage[layer_idx] = per_head[:, -1, :, :].detach().cpu()  # [batch, num_heads, head_dim]
        return hook_fn

    def register_hooks(self, layers_to_hook: List[int], num_heads: int, head_dim: int):
        """在指定层的o_proj上注册forward hook。"""
        self.remove_hooks()
        self.storage = {}

        for layer_idx in layers_to_hook:
            o_proj = self.model.model.layers[layer_idx].self_attn.o_proj
            hook = o_proj.register_forward_hook(
                self._make_hook(layer_idx, num_heads, head_dim),
                with_kwargs=True,
            )
            self.hooks.append(hook)

    def remove_hooks(self):
        """移除所有hook。"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_per_head_outputs(self) -> Dict[int, torch.Tensor]:
        """获取各层的per-head输出。"""
        return dict(self.storage)


def compute_per_head_contributions(
    per_head_outputs: Dict[int, torch.Tensor],
    o_proj_weight: torch.Tensor,
    head_dim: int,
) -> Dict[int, torch.Tensor]:
    """从per-head outputs和o_proj权重计算每个head的贡献向量。

    Args:
        per_head_outputs: {layer_idx: tensor[num_heads, head_dim]}
        o_proj_weight: tensor[hidden_dim, num_heads * head_dim]
        head_dim: 每个头的维度

    Returns:
        {layer_idx: tensor[num_heads, hidden_dim]} — 每层每个head的贡献向量
    """
    num_heads = list(per_head_outputs.values())[0].shape[0] if per_head_outputs else 0
    contributions = {}

    # o_proj_weight可能在GPU上，但per_head_outputs在CPU上（hook中detach().cpu()）
    # 统一移到CPU进行计算（per-head贡献分解计算量很小）
    if o_proj_weight.device.type != "cpu":
        o_proj_weight = o_proj_weight.float().cpu()

    for layer_idx, ph_out in per_head_outputs.items():
        # ph_out: [num_heads, head_dim]
        # o_proj_weight: [hidden_dim, num_heads * head_dim]
        # 对每个head i: contribution_i = ph_out[i] @ o_proj_weight[:, i*head_dim:(i+1)*head_dim].T
        contrib_list = []
        for h in range(num_heads):
            start = h * head_dim
            end = (h + 1) * head_dim
            w_slice = o_proj_weight[:, start:end]  # [hidden_dim, head_dim]
            head_out = ph_out[h].float()  # [head_dim]
            contrib = head_out @ w_slice.T  # [hidden_dim]
            contrib_list.append(contrib)
        contributions[layer_idx] = torch.stack(contrib_list, dim=0)  # [num_heads, hidden_dim]

    return contributions


# ---------------------------------------------------------------------------
# 核心分析
# ---------------------------------------------------------------------------

def compute_head_alignment_scores(
    contributions: Dict[int, torch.Tensor],
    w_gt: torch.Tensor,
    w_pred: torch.Tensor,
) -> Dict[int, torch.Tensor]:
    """计算每个head的GT-Pred对齐贡献分。

    score_i = cos(contribution_i, w_GT) - cos(contribution_i, w_Pred)
    > 0 → head帮助对齐GT
    < 0 → head帮助对齐Pred（"坏头"）

    Args:
        contributions: {layer_idx: tensor[num_heads, hidden_dim]}
        w_gt: [hidden_dim]
        w_pred: [hidden_dim]

    Returns:
        {layer_idx: tensor[num_heads]} — 每层每个head的score
    """
    scores = {}
    # contributions在CPU上（来自compute_per_head_contributions），统一w到CPU
    w_gt = w_gt.float().cpu()
    w_pred = w_pred.float().cpu()
    for layer_idx, contrib in contributions.items():
        # contrib: [num_heads, hidden_dim]
        # w_gt, w_pred: [hidden_dim]
        cos_gt = F.cosine_similarity(contrib, w_gt.unsqueeze(0), dim=-1)  # [num_heads]
        cos_pred = F.cosine_similarity(contrib, w_pred.unsqueeze(0), dim=-1)  # [num_heads]
        scores[layer_idx] = cos_gt - cos_pred
    return scores


def analyze_one_sample(
    model,
    tokenizer,
    extractor: PerHeadContributionExtractor,
    sample: Dict[str, Any],
    lm_head_weight: torch.Tensor,
    option_token_ids: Dict[str, int],
    model_key: str,
    layers_to_hook: List[int],
    head_info: Dict[str, Any],
    device: str,
) -> Optional[Dict[str, Any]]:
    """对一个样本执行完整的per-head分析。

    Returns:
        {
            "sample_idx": int,
            "sample_id": str,
            "per_layer_head_scores": {layer_idx: [head_scores]},
            "top_bad_heads": [(layer, head, score)],
            "top_good_heads": [(layer, head, score)],
        }
    """
    num_heads = head_info["num_heads"]
    head_dim = head_info["head_dim"]

    # ---- 获取GT/Pred ----
    gt_answer = get_gt_answer(sample.get("ground_truth", {}))
    answer_type = get_answer_type(sample.get("ground_truth", {}))
    if not gt_answer or answer_type != "multiple_choice":
        return None

    gt_letter = gt_answer.strip().upper()
    gt_tid = option_token_ids.get(gt_letter)
    if gt_tid is None:
        return None

    # 构造prompt
    messages = sample["messages"]
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        # fallback
        parts = []
        role_map = {"system": "System", "user": "User", "assistant": "Assistant"}
        for msg in messages:
            parts.append(f"{role_map.get(msg['role'], 'User')}: {msg.get('content', '')}")
        parts.append("Assistant: ")
        input_text = "\n\n".join(parts)

    # tokenize
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    # ---- 获取Pred token（从已有模型输出推断，或用logit最高选项）----
    # 注意：这里无法直接获取pred_answer（需要原始step2输出）
    # 我们只用GT进行分析——寻找哪些head帮助对齐GT，哪些head反对GT
    # 但为了与step6b一致，也计算Pred
    # 用一个简单的heuristic：从model forward pass获取logits，取非GT最高选项
    pred_tid = None
    pred_letter = None

    # ---- Forward pass with hooks ----
    extractor.register_hooks(layers_to_hook, num_heads, head_dim)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            output_attentions=False,
        )
        # 用最后一层的hidden state计算logits来推断pred
        last_hidden = outputs.hidden_states[-1][:, -1, :] if outputs.hidden_states else None

    # 获取per-head outputs
    per_head_outputs = extractor.get_per_head_outputs()
    extractor.remove_hooks()

    if not per_head_outputs:
        return None

    # ---- 获取w_GT ----
    w_gt = lm_head_weight[gt_tid].float().to(device)  # [hidden_dim]

    # ---- 推断Pred选项 ----
    if last_hidden is not None:
        last_h = last_hidden.float().to(device)
        logits = last_h @ lm_head_weight.float().to(device).T  # [1, vocab]
        # 找非GT选项中logit最高的
        best_score = -float("inf")
        for letter, tid in option_token_ids.items():
            if letter == gt_letter:
                continue
            score = logits[0, tid].item()
            if score > best_score:
                best_score = score
                pred_tid = tid
                pred_letter = letter

    if pred_tid is None:
        # fallback: 用第一个非GT选项
        for letter, tid in option_token_ids.items():
            if letter != gt_letter:
                pred_tid = tid
                pred_letter = letter
                break

    w_pred = lm_head_weight[pred_tid].float().to(device)

    # ---- 获取o_proj权重 ----
    # 取任意一个hooked层的o_proj权重（所有层shape相同）
    sample_layer = layers_to_hook[0]
    o_proj_weight = model.model.layers[sample_layer].self_attn.o_proj.weight.data.float().to(device)
    # [hidden_dim, num_heads * head_dim]

    # ---- 计算per-head贡献 ----
    contributions = compute_per_head_contributions(
        per_head_outputs, o_proj_weight, head_dim
    )

    # ---- 计算对齐分数 ----
    head_scores = compute_head_alignment_scores(contributions, w_gt, w_pred)

    # ---- 整理结果 ----
    all_head_entries = []  # (layer, head, score)
    per_layer_scores = {}
    for layer_idx, scores_tensor in head_scores.items():
        score_list = scores_tensor.flatten().tolist()
        per_layer_scores[layer_idx] = score_list
        for h_idx, s in enumerate(score_list):
            all_head_entries.append((layer_idx, h_idx, float(s)))

    # 排序
    sorted_entries = sorted(all_head_entries, key=lambda x: x[2])
    top_bad = sorted_entries[:10]  # 最负 = 最"坏"
    top_good = sorted_entries[-10:][::-1]  # 最正 = 最"好"

    del outputs
    del last_hidden

    return {
        "sample_idx": sample.get("_idx", -1),
        "sample_id": sample.get("id", "unknown"),
        "gt_answer": gt_letter,
        "pred_answer": pred_letter,
        "per_layer_head_scores": per_layer_scores,
        "top_bad_heads": [(l, h, round(float(s), 6)) for l, h, s in top_bad],
        "top_good_heads": [(l, h, round(float(s), 6)) for l, h, s in top_good],
    }


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import matplotlib.font_manager as fm
    _font_candidates = [
        '/usr/share/fonts/truetype/chinese/NotoSansSC[wght].ttf',
        '/usr/share/fonts/truetype/noto-serif-sc/NotoSerifSC-Regular.otf',
        '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    ]
    for _fp in _font_candidates:
        if os.path.exists(_fp):
            try:
                fm.fontManager.addfont(_fp)
            except Exception:
                pass
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def plot_head_heatmap(
    all_sample_scores: List[Dict[str, Any]],
    num_layers: int,
    num_heads: int,
    model_name: str,
    output_dir: str,
):
    """绘制单个模型的 layer × head 对齐分数热力图。

    颜色：红色 = 坏头（帮助Pred），蓝色 = 好头（帮助GT）
    """
    plt = _setup_plt()
    os.makedirs(output_dir, exist_ok=True)

    # 构建聚合矩阵: [num_layers, num_heads]
    score_matrix = np.zeros((num_layers, num_heads))
    count_matrix = np.zeros((num_layers, num_heads))

    for sample_result in all_sample_scores:
        pls = sample_result["per_layer_head_scores"]
        for layer_idx_str, scores in pls.items():
            layer_idx = int(layer_idx_str)
            if 0 <= layer_idx < num_layers:
                for h_idx, s in enumerate(scores):
                    if h_idx < num_heads:
                        score_matrix[layer_idx, h_idx] += s
                        count_matrix[layer_idx, h_idx] += 1

    # 平均
    mask = count_matrix > 0
    avg_matrix = np.where(mask, score_matrix / count_matrix, 0)

    fig, ax = plt.subplots(figsize=(max(12, num_heads * 0.4), max(6, num_layers * 0.2)))

    im = ax.imshow(avg_matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-np.max(np.abs(avg_matrix)) - 0.001,
                   vmax=np.max(np.abs(avg_matrix)) + 0.001)

    ax.set_xlabel("Head Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(f"{model_name}: Attention Head GT-Pred Alignment Score\n"
                 f"(Red = helps Pred/bad, Blue = helps GT/good)")

    # 标注关键数值
    for i in range(num_layers):
        for j in range(num_heads):
            if count_matrix[i, j] > 0:
                val = avg_matrix[i, j]
                # 只标注极端值
                if abs(val) > np.percentile(np.abs(avg_matrix[mask]), 90):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=5, color="black" if abs(val) < np.max(np.abs(avg_matrix)) * 0.7 else "white")

    plt.colorbar(im, ax=ax, label="cos(contribution, w_GT) - cos(contribution, w_Pred)")

    # 标记深层区域
    deep_start = num_layers // 2
    ax.axhline(y=deep_start - 0.5, color="yellow", linewidth=2, linestyle="--", alpha=0.8)
    ax.text(num_heads - 1, deep_start + 0.5, "Deep Layers (key region)", ha="right",
            fontsize=8, color="yellow", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))

    plt.tight_layout()
    path = os.path.join(output_dir, f"attention_head_heatmap_{model_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path, avg_matrix


def plot_model_comparison(
    avg_matrices: Dict[str, np.ndarray],
    output_dir: str,
):
    """绘制三个模型的head对齐分数对比图。

    (a) 每层平均分数对比（层间对比）
    (b) 每层中坏头占比对比（层间对比）
    (c) 三个模型最坏的top-10 head排名
    """
    plt = _setup_plt()
    os.makedirs(output_dir, exist_ok=True)

    model_colors = {"1.5B": "#2196F3", "3B": "#4CAF50", "7B": "#FF9800"}
    model_tags = {"1.5B": "qwen1.5B", "3B": "qwen3B", "7B": "qwen7B"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ---- (a) 每层平均head score ----
    for tag, matrix in avg_matrices.items():
        layer_mean = np.mean(matrix, axis=1)  # [num_layers]
        axes[0].plot(range(len(layer_mean)), layer_mean, label=tag,
                     color=model_colors.get(tag, "gray"), linewidth=2)

    axes[0].axhline(y=0, color="black", linestyle="--", alpha=0.4)
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("Mean Head Score (GT-Pred)")
    axes[0].set_title("(a) Per-Layer Mean Head Alignment Score\n(>0 = GT-aligned, <0 = Pred-aligned)")
    axes[0].legend(loc="best")
    axes[0].grid(alpha=0.3)

    # ---- (b) 每层坏头占比 ----
    for tag, matrix in avg_matrices.items():
        bad_frac_per_layer = []
        for layer_scores in matrix:
            bad_count = np.sum(layer_scores < 0)
            bad_frac_per_layer.append(bad_count / len(layer_scores))
        axes[1].plot(range(len(bad_frac_per_layer)), bad_frac_per_layer,
                     label=tag, color=model_colors.get(tag, "gray"), linewidth=2)

    axes[1].axhline(y=0.5, color="red", linestyle="--", alpha=0.4, label="50% line")
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("Fraction of 'Bad' Heads (score < 0)")
    axes[1].set_title("(b) Per-Layer 'Bad Head' Fraction\n(Higher = more heads help Pred)")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.3)

    # ---- (c) Top-10 worst heads across all layers ----
    all_bad_entries = {tag: [] for tag in avg_matrices}
    for tag, matrix in avg_matrices.items():
        for layer_idx in range(matrix.shape[0]):
            for head_idx in range(matrix.shape[1]):
                all_bad_entries[tag].append((layer_idx, head_idx, matrix[layer_idx, head_idx]))

    for tag, entries in all_bad_entries.items():
        entries.sort(key=lambda x: x[2])
        top10 = entries[:10]
        labels = [f"L{l}_H{h}" for l, h, _ in top10]
        values = [s for _, _, s in top10]
        axes[2].barh(
            range(len(labels)), values, alpha=0.7,
            color=model_colors.get(tag, "gray"), label=tag,
        )

    axes[2].axvline(x=0, color="black", linestyle="--", alpha=0.4)
    axes[2].set_xlabel("Head Score (GT-Pred)")
    axes[2].set_ylabel("Worst Heads")
    axes[2].set_title("(c) Top-10 Worst Heads (All Layers)\n(Most negative = most 'destructive')")
    axes[2].legend(loc="best")
    axes[2].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    path = os.path.join(output_dir, "attention_head_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_analysis(
    model_key: str,
    num_samples: int,
    output_dir: str = "experiment_results/attention_heads",
    device: str = "cpu",
):
    """运行单个模型的attention head分析。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_tag = model_key.replace("qwen", "")  # "qwen1.5B" → "1.5B"
    model_output_dir = os.path.join(output_dir, model_tag)
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"Step 6c: Attention Head Analysis — {model_key}")
    print(f"{'=' * 80}")

    # ---- 获取head配置 ----
    head_info = get_model_head_info(model_key)
    num_layers = head_info["num_layers"]
    num_heads = head_info["num_heads"]
    head_dim = head_info["head_dim"]
    deep_start = num_layers // 2
    print(f"  Layers: {num_layers}, Heads: {num_heads}, Head dim: {head_dim}")
    print(f"  GQA: {head_info['use_gqa']} (kv_heads={head_info['num_kv_heads']})")
    print(f"  Deep layers start at: {deep_start}")

    # ---- 加载数据 ----
    data_path = "./experiment_results/sampled_data/sampled_300.json"
    if not os.path.exists(data_path):
        print(f"  Error: {data_path} not found")
        return None

    with open(data_path, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)

    # ---- 加载错误样本列表 ----
    cmp_path = ANALYSIS_CONFIG["three_model_comparison"]
    if not os.path.exists(cmp_path):
        print(f"  Error: {cmp_path} not found")
        return None

    df_cmp = pd.read_csv(cmp_path)
    error_indices = set(
        df_cmp[(df_cmp["correct_7B"]) & (~df_cmp["correct_1.5B"])]["sample_idx"]
        .astype(int).tolist()
    )
    print(f"  Error samples: {len(error_indices)}")

    if not error_indices:
        print("  No error samples to analyze.")
        return None

    # ---- 加载模型 ----
    model, tokenizer = load_model_for_head_analysis(model_key, device)

    # ---- 加载LM Head ----
    print(f"  Loading LM Head for {model_key}...")
    lm_head_weight = model.lm_head.weight.detach().clone()  # keep on GPU

    # ---- 获取选项token IDs ----
    option_letters = ["A", "B", "C", "D"]
    option_token_ids = {}
    for letter in option_letters:
        tid = get_answer_token_id(tokenizer, letter, "multiple_choice")
        if tid is not None:
            option_token_ids[letter] = tid
    print(f"  Option token IDs: {option_token_ids}")

    # ---- 创建hook extractor ----
    # 分析所有层（但可视化会区分浅层/深层）
    layers_to_hook = list(range(num_layers))
    extractor = PerHeadContributionExtractor(model, device)

    # ---- 逐样本分析 ----
    all_sample_results = []
    analyzed_count = 0

    for idx in tqdm(sorted(error_indices), desc=f"Analyzing {model_key} heads"):
        if idx >= len(all_samples):
            continue

        sample = all_samples[idx]
        sample["_idx"] = idx  # 附加索引

        # 只分析multiple_choice
        answer_type = get_answer_type(sample.get("ground_truth", {}))
        if answer_type != "multiple_choice":
            continue

        try:
            result = analyze_one_sample(
                model=model,
                tokenizer=tokenizer,
                extractor=extractor,
                sample=sample,
                lm_head_weight=lm_head_weight,
                option_token_ids=option_token_ids,
                model_key=model_key,
                layers_to_hook=layers_to_hook,
                head_info=head_info,
                device=device,
            )
            if result is not None:
                all_sample_results.append(result)
                analyzed_count += 1
        except Exception as e:
            print(f"  Error analyzing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 定期清理GPU
        if analyzed_count % 5 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    extractor.remove_hooks()

    print(f"  Analyzed: {analyzed_count} samples")

    if not all_sample_results:
        print("  No results to visualize.")
        del model, tokenizer
        return None

    # ---- 聚合统计 ----
    # 构建聚合矩阵
    score_matrix = np.zeros((num_layers, num_heads))
    count_matrix = np.zeros((num_layers, num_heads))

    for sr in all_sample_results:
        pls = sr["per_layer_head_scores"]
        for layer_idx_str, scores in pls.items():
            li = int(layer_idx_str)
            if 0 <= li < num_layers:
                for hi, s in enumerate(scores):
                    if hi < num_heads:
                        score_matrix[li, hi] += s
                        count_matrix[li, hi] += 1

    mask = count_matrix > 0
    avg_matrix = np.where(mask, score_matrix / count_matrix, 0)

    # ---- 找出最坏的heads ----
    all_entries = []
    for li in range(num_layers):
        for hi in range(num_heads):
            if count_matrix[li, hi] > 0:
                all_entries.append({
                    "layer": li,
                    "head": hi,
                    "avg_score": round(float(avg_matrix[li, hi]), 6),
                    "is_deep": li >= deep_start,
                    "n_samples": int(count_matrix[li, hi]),
                })

    all_entries.sort(key=lambda x: x["avg_score"])
    top_bad = all_entries[:20]
    top_good = all_entries[-20:][::-1]

    # 深层坏头 vs 浅层坏头
    deep_bad = [e for e in all_entries if e["is_deep"] and e["avg_score"] < 0]
    shallow_bad = [e for e in all_entries if not e["is_deep"] and e["avg_score"] < 0]
    deep_good = [e for e in all_entries if e["is_deep"] and e["avg_score"] > 0]
    shallow_good = [e for e in all_entries if not e["is_deep"] and e["avg_score"] > 0]

    # ---- 打印关键发现 ----
    print(f"\n  *** Attention Head Analysis: {model_key} ***")
    print(f"  Total heads: {num_layers * num_heads}")
    print(f"  Bad heads (score < 0): {len([e for e in all_entries if e['avg_score'] < 0])}/{len(all_entries)}")
    print(f"    Shallow layers (0-{deep_start-1}): {len(shallow_bad)} bad")
    print(f"    Deep layers ({deep_start}-{num_layers-1}): {len(deep_bad)} bad")

    if deep_bad and shallow_bad:
        deep_bad_mean = np.mean([e["avg_score"] for e in deep_bad])
        shallow_bad_mean = np.mean([e["avg_score"] for e in shallow_bad])
        print(f"    Deep bad mean score: {deep_bad_mean:.6f}")
        print(f"    Shallow bad mean score: {shallow_bad_mean:.6f}")

    print(f"\n  Top 10 Most Destructive Heads:")
    for i, e in enumerate(top_bad[:10]):
        region = "DEEP" if e["is_deep"] else "shallow"
        print(f"    #{i+1}: Layer {e['layer']:2d}, Head {e['head']:2d} ({region}) "
              f"score={e['avg_score']:.6f}")

    # ---- 可视化 ----
    print(f"\n  Generating visualizations...")
    heatmap_path, _ = plot_head_heatmap(
        all_sample_results, num_layers, num_heads, model_tag, model_output_dir
    )

    # ---- 保存JSON结果 ----
    result_json = {
        "model": model_key,
        "model_tag": model_tag,
        "head_info": head_info,
        "n_samples_analyzed": analyzed_count,
        "deep_start_layer": deep_start,
        "total_heads": num_layers * num_heads,
        "n_bad_total": len([e for e in all_entries if e["avg_score"] < 0]),
        "n_bad_shallow": len(shallow_bad),
        "n_bad_deep": len(deep_bad),
        "top_20_bad_heads": top_bad,
        "top_20_good_heads": top_good,
        "per_layer_mean_score": {str(li): round(float(np.mean(avg_matrix[li])), 6)
                                 for li in range(num_layers)},
        "per_layer_bad_fraction": {str(li): round(float(
            np.sum(avg_matrix[li] < 0) / num_heads), 4)
            for li in range(num_layers)},
    }

    json_path = os.path.join(model_output_dir, f"head_analysis_{model_tag}.json")
    save_json(result_json, json_path)
    print(f"  Results saved to: {json_path}")

    # ---- 释放模型 ----
    del model, tokenizer, lm_head_weight
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return avg_matrix


def run_all_models(
    num_samples: int,
    output_dir: str = "experiment_results/attention_heads",
    device: str = "cpu",
):
    """对三个模型分别运行attention head分析，并生成对比图。"""
    model_keys = ["qwen1.5B", "qwen3B", "qwen7B"]
    avg_matrices = {}

    for model_key in model_keys:
        matrix = run_analysis(model_key, num_samples, output_dir, device)
        if matrix is not None:
            tag = model_key.replace("qwen", "")
            avg_matrices[tag] = matrix

    # ---- 对比图 ----
    if len(avg_matrices) >= 2:
        print(f"\n  Generating cross-model comparison...")
        plot_model_comparison(avg_matrices, output_dir)

    # ---- 综合JSON ----
    summary = {
        "n_models": len(avg_matrices),
        "models": list(avg_matrices.keys()),
        "key_finding": {},
    }

    for tag, matrix in avg_matrices.items():
        num_layers = matrix.shape[0]
        deep_start = num_layers // 2
        deep_mean = np.mean(matrix[deep_start:])
        shallow_mean = np.mean(matrix[:deep_start])
        deep_bad_frac = np.mean(matrix[deep_start:] < 0)
        shallow_bad_frac = np.mean(matrix[:deep_start] < 0)

        summary["key_finding"][tag] = {
            "deep_layers_mean_score": round(float(deep_mean), 6),
            "shallow_layers_mean_score": round(float(shallow_mean), 6),
            "deep_bad_head_fraction": round(float(deep_bad_frac), 4),
            "shallow_bad_head_fraction": round(float(shallow_bad_frac), 4),
            "deep_more_destructive": bool(deep_mean < shallow_mean),
        }

    summary_path = os.path.join(output_dir, "head_analysis_summary.json")
    save_json(summary, summary_path)
    print(f"\n  Summary saved to: {summary_path}")

    # ---- 打印综合发现 ----
    print(f"\n{'=' * 80}")
    print("Step 6c: Attention Head Analysis — Cross-Model Summary")
    print(f"{'=' * 80}")

    for tag, findings in summary["key_finding"].items():
        print(f"\n  {tag}:")
        print(f"    Deep layers mean score:   {findings['deep_layers_mean_score']:.6f}")
        print(f"    Shallow layers mean score: {findings['shallow_layers_mean_score']:.6f}")
        print(f"    Deep bad head fraction:    {findings['deep_bad_head_fraction']*100:.1f}%")
        print(f"    Shallow bad head fraction: {findings['shallow_bad_head_fraction']*100:.1f}%")
        if findings["deep_more_destructive"]:
            print(f"    >>> Deep layers are MORE destructive than shallow layers")

    # 假说检验
    print(f"\n  *** Hypothesis Test: Deep Layer Head Destruction ***")
    if all(f["deep_more_destructive"] for f in summary["key_finding"].values()):
        print("    >>> STRONG SUPPORT: All models show deep layers have more destructive heads")
    elif any(f["deep_more_destructive"] for f in summary["key_finding"].values()):
        supported = [t for t, f in summary["key_finding"].items() if f["deep_more_destructive"]]
        print(f"    >>> SUPPORT: {', '.join(supported)} show deep layer head destruction")

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 6c: Attention Head Ablation Analysis"
    )
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--model", type=str, default="all",
                        choices=["qwen1.5B", "qwen3B", "qwen7B", "all"],
                        help="Which model(s) to analyze (default: all)")
    parser.add_argument("--output_dir", type=str,
                        default="experiment_results/attention_heads")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print("=" * 80)
    print("Step 6c: Attention Head Ablation Analysis")
    print("  Identifies which attention heads help GT vs help Pred")
    print("  Method: per-head contribution decomposition via o_proj")
    print("  Key question: are deep-layer heads more 'destructive'?")
    print("=" * 80)

    if args.model == "all":
        run_all_models(
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            device=device,
        )
    else:
        run_analysis(
            model_key=args.model,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            device=device,
        )

    print("\n" + "=" * 80)
    print("Step 6c Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()