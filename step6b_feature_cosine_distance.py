"""
Step 6b: Feature Cosine Distance (Phase II)

在模型内部，测量prefill阶段各层隐藏状态与GT/Pred选项embedding行之间的余弦距离。

核心假说（Resolution Compression）：
  小模型（1.5B）经过NLP处理后，在表示空间中将GT选项和预测选项"挤压"在一起
  （余弦相似度趋近），而大模型（7B）能保持两者的区分度。
  注意：不使用logit（h·W），因为logit是最终输出，已经被NLP过程"污染"。
  我们直接用余弦距离在表示空间中分析。

数据结构（来自step2）：
  - prefill_hidden_states = list[num_layers], each [hidden_dim]
    → 输入处理完成后、最后一个token位置的各层隐藏状态
  - hidden_states_per_step[step_idx][layer_idx] = [hidden_dim]
    → 第step_idx步生成时的各层隐藏状态

方法：
  对每个错误样本（7B正确，1.5B/3B错误）：
  1. 加载各模型的LM Head权重，提取GT/Pred选项的embedding行：
     w_GT = lm_head.weight[gt_token_id]  → [hidden_dim]
     w_Pred = lm_head.weight[pred_token_id]  → [hidden_dim]
  2. 计算baseline: cosine(w_GT, w_Pred) — 选项本身在LM空间中的距离
  3. 对prefill的每一层 l:
     cos_GT[l] = cosine(h[l], w_GT)    — 该层表征与GT选项的对齐度
     cos_Pred[l] = cosine(h[l], w_Pred)  — 该层表征与Pred选项的对齐度
     cos_diff[l] = cos_GT[l] - cos_Pred[l] — GT vs Pred的区分度
  4. 聚合对比 1.5B vs 3B vs 7B 的逐层余弦距离曲线
  5. 正确样本作为对照组

输出：
  - option_distance_analysis_v2.json（聚合统计）
  - 3x PNG可视化

需要加载LM Head权重（仅weight矩阵，不需要forward pass）。
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm

from config import MODELS, ANALYSIS_CONFIG
from qa_utils import (
    get_gt_answer, get_answer_type, get_clean_answer, get_answer_token_id,
)
from utils import load_model_outputs, save_json


# ---------------------------------------------------------------------------
# LM Head loading (仅weight矩阵，不需要forward pass)
# ---------------------------------------------------------------------------

def load_lm_head_weight(model_key: str, device: str = "cpu") -> Optional[torch.Tensor]:
    """加载模型的LM Head权重。返回 [vocab_size, hidden_dim]。"""
    from transformers import AutoModelForCausalLM

    model_name = MODELS[model_key]["model_name"]
    print(f"  Loading LM Head from {model_name}...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        lm_head_weight = model.lm_head.weight.detach().clone().to(device)  # [vocab_size, hidden_dim]
        del model
        import gc
        gc.collect()

        print(f"    LM Head shape: {lm_head_weight.shape} -> {device}")
        return lm_head_weight
    except Exception as e:
        print(f"    Failed to load: {e}")
        return None


# ---------------------------------------------------------------------------
# 选项token ID获取
# ---------------------------------------------------------------------------

def get_option_token_ids(tokenizer, option_letters: List[str], answer_type: str) -> Dict[str, int]:
    """获取所有选项字母的token ID。"""
    option_token_ids = {}
    for letter in option_letters:
        tid = get_answer_token_id(tokenizer, letter, answer_type)
        if tid is not None:
            option_token_ids[letter] = tid
        else:
            print(f"  Warning: could not get token ID for '{letter}'")
    return option_token_ids


# ---------------------------------------------------------------------------
# 核心分析：模型内特征余弦距离
# ---------------------------------------------------------------------------

def analyze_sample_feature_cosine(
    sample_idx: int,
    outputs_map: Dict[str, Optional[Dict]],
    option_token_ids: Dict[str, int],
    lm_heads: Dict[str, torch.Tensor],
    device: str = "cpu",
) -> Optional[Dict[str, Any]]:
    """分析一个样本在三个模型中的逐层特征余弦距离。

    对每个模型：
    1. 提取GT和Pred选项的LM Head embedding行
    2. 计算baseline: cosine(w_GT, w_Pred)
    3. 对prefill每一层: cosine(h[l], w_GT), cosine(h[l], w_Pred), diff
    4. 对答案生成步的各层也做同样分析
    """
    out_1_5B = outputs_map.get("1.5B")
    if out_1_5B is None:
        return None

    gt_answer = get_gt_answer(out_1_5B.get("ground_truth", {}))
    answer_type = get_answer_type(out_1_5B.get("ground_truth", {}))

    if not gt_answer or answer_type != "multiple_choice":
        return None

    gt_letter = gt_answer.strip().upper()
    gt_tid = option_token_ids.get(gt_letter)
    if gt_tid is None:
        return None

    result = {
        "sample_idx": sample_idx,
        "sample_id": out_1_5B.get("sample_id", f"sample_{sample_idx}"),
        "dataset": out_1_5B.get("ground_truth", {}).get("dataset", "unknown"),
        "gt_answer": gt_letter,
        "per_model": {},
    }

    for mn in ["1.5B", "3B", "7B"]:
        out = outputs_map.get(mn)
        lm_head = lm_heads.get(mn)
        if out is None or lm_head is None:
            continue

        pred_answer, _ = get_clean_answer(out)
        pred_letter = pred_answer.strip().upper() if pred_answer else ""
        is_error = (pred_letter != gt_letter and pred_letter in option_token_ids)

        pred_tid = option_token_ids.get(pred_letter)

        # ---- 选项embedding行 ----
        w_gt = lm_head[gt_tid].float()  # [hidden_dim] (already on device)

        # 如果pred_letter不在选项中或没有token ID，用logit最高的错误选项
        if pred_tid is None or pred_letter == gt_letter:
            # 从top_k_info中找到logit最高的非GT选项
            top_k_info = out.get("top_k_info", [])
            best_wrong_tid = None
            best_wrong_letter = None
            best_wrong_logit = -float("inf")
            for step_k in top_k_info[:3]:  # 检查前几步
                if step_k is None:
                    continue
                indices = step_k.get("indices")
                logits = step_k.get("logits")
                if indices is None or logits is None:
                    continue
                idx_list = indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
                logit_list = logits.tolist() if isinstance(logits, torch.Tensor) else list(logits)
                # 找logit最高且不是GT的选项
                for letter, tid in option_token_ids.items():
                    if letter == gt_letter:
                        continue
                    try:
                        pos = idx_list.index(tid)
                        if logit_list[pos] > best_wrong_logit:
                            best_wrong_tid = tid
                            best_wrong_letter = letter
                            best_wrong_logit = logit_list[pos]
                    except (ValueError, IndexError):
                        pass
                if best_wrong_tid is not None:
                    break

            if best_wrong_tid is not None:
                pred_tid = best_wrong_tid
                pred_letter = best_wrong_letter
                is_error = True
            else:
                result["per_model"][mn] = {"error": "no valid pred option", "is_error": False}
                continue

        w_pred = lm_head[pred_tid].float()  # [hidden_dim]

        # ---- Baseline: 选项embedding行之间的余弦距离 ----
        baseline_cos = F.cosine_similarity(
            w_gt.unsqueeze(0), w_pred.unsqueeze(0), dim=-1
        ).item()

        # ---- Prefill阶段逐层余弦距离 ----
        prefill_hs = out.get("prefill_hidden_states", [])
        prefill_data = []
        if prefill_hs:
            for layer_idx in range(len(prefill_hs)):
                h = prefill_hs[layer_idx].float().to(device)  # [hidden_dim]

                cos_gt = F.cosine_similarity(
                    h.unsqueeze(0), w_gt.unsqueeze(0), dim=-1
                ).item()
                cos_pred = F.cosine_similarity(
                    h.unsqueeze(0), w_pred.unsqueeze(0), dim=-1
                ).item()

                # 所有选项的余弦距离
                all_option_cos = {}
                for letter, tid in option_token_ids.items():
                    w_opt = lm_head[tid].float()
                    all_option_cos[letter] = round(
                        F.cosine_similarity(
                            h.unsqueeze(0), w_opt.unsqueeze(0), dim=-1
                        ).item(), 4
                    )

                prefill_data.append({
                    "layer": layer_idx,
                    "cos_gt": round(cos_gt, 4),
                    "cos_pred": round(cos_pred, 4),
                    "cos_diff": round(cos_gt - cos_pred, 4),
                    "all_option_cos": all_option_cos,
                    "option_cos_spread": round(
                        float(np.std(list(all_option_cos.values()))), 4
                    ),
                })

        # ---- 答案生成步逐层余弦距离 ----
        gen_step_data = []
        hidden_states_per_step = out.get("hidden_states_per_step", [])
        if hidden_states_per_step:
            # 使用第一步生成（答案生成步）的各层隐藏状态
            answer_step_layers = hidden_states_per_step[0] if hidden_states_per_step else []
            for layer_idx in range(len(answer_step_layers)):
                h = answer_step_layers[layer_idx].float().to(device)  # [hidden_dim]

                cos_gt = F.cosine_similarity(
                    h.unsqueeze(0), w_gt.unsqueeze(0), dim=-1
                ).item()
                cos_pred = F.cosine_similarity(
                    h.unsqueeze(0), w_pred.unsqueeze(0), dim=-1
                ).item()

                all_option_cos = {}
                for letter, tid in option_token_ids.items():
                    w_opt = lm_head[tid].float()
                    all_option_cos[letter] = round(
                        F.cosine_similarity(
                            h.unsqueeze(0), w_opt.unsqueeze(0), dim=-1
                        ).item(), 4
                    )

                gen_step_data.append({
                    "layer": layer_idx,
                    "cos_gt": round(cos_gt, 4),
                    "cos_pred": round(cos_pred, 4),
                    "cos_diff": round(cos_gt - cos_pred, 4),
                    "all_option_cos": all_option_cos,
                })

        # ---- 最终生成步（最后一步）的各层余弦距离 ----
        final_step_data = []
        if hidden_states_per_step and len(hidden_states_per_step) > 1:
            final_step_layers = hidden_states_per_step[-1]
            for layer_idx in range(len(final_step_layers)):
                h = final_step_layers[layer_idx].float().to(device)

                cos_gt = F.cosine_similarity(
                    h.unsqueeze(0), w_gt.unsqueeze(0), dim=-1
                ).item()
                cos_pred = F.cosine_similarity(
                    h.unsqueeze(0), w_pred.unsqueeze(0), dim=-1
                ).item()

                final_step_data.append({
                    "layer": layer_idx,
                    "cos_gt": round(cos_gt, 4),
                    "cos_pred": round(cos_pred, 4),
                    "cos_diff": round(cos_gt - cos_pred, 4),
                })

        # ---- Prefill最终层的综合指标 ----
        final_layer_metrics = {}
        if prefill_data:
            fl = prefill_data[-1]
            final_layer_metrics = {
                "cos_gt_final": fl["cos_gt"],
                "cos_pred_final": fl["cos_pred"],
                "cos_diff_final": fl["cos_diff"],
                "option_cos_spread_final": fl["option_cos_spread"],
            }

        result["per_model"][mn] = {
            "pred_answer": pred_letter,
            "is_error": is_error,
            "baseline_cos_gt_pred": round(baseline_cos, 4),
            "hidden_dim": w_gt.shape[0],
            "num_prefill_layers": len(prefill_hs) if prefill_hs else 0,
            "num_gen_steps": len(hidden_states_per_step) if hidden_states_per_step else 0,
            "prefill_layer_data": prefill_data,
            "gen_step0_layer_data": gen_step_data,
            "final_step_layer_data": final_step_data,
            "final_layer_metrics": final_layer_metrics,
        }

    return result


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def _setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import matplotlib.font_manager as fm
    # Try to register fonts, skip if not available on server
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


def plot_cosine_distance_curves(all_results: List[Dict], output_dir: str):
    """图1: 逐层余弦距离曲线（主要图）

    (a) Prefill阶段: cosine(h[l], w_GT) vs cosine(h[l], w_Pred) 逐层变化
    (b) cos_diff = cos_GT - cos_Pred 逐层变化（区分度曲线）
    (c) 选项余弦距离spread逐层变化
    """
    plt = _setup_plt()
    os.makedirs(output_dir, exist_ok=True)

    model_colors = {"1.5B": "#2196F3", "3B": "#4CAF50", "7B": "#FF9800"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # ---- (a) Prefill cos_GT and cos_Pred per layer ----
    for mn in ["1.5B", "3B", "7B"]:
        cos_gt_curves = []
        cos_pred_curves = []
        for r in all_results:
            ld = r["per_model"].get(mn, {})
            layers = ld.get("prefill_layer_data", [])
            if not layers:
                continue
            cos_gt_curves.append([l["cos_gt"] for l in layers])
            cos_pred_curves.append([l["cos_pred"] for l in layers])

        if not cos_gt_curves:
            continue

        max_len = max(max(len(c) for c in cos_gt_curves), max(len(c) for c in cos_pred_curves))
        pad_gt = [c + [np.nan] * (max_len - len(c)) for c in cos_gt_curves]
        pad_pr = [c + [np.nan] * (max_len - len(c)) for c in cos_pred_curves]
        pad_gt = np.array(pad_gt)
        pad_pr = np.array(pad_pr)

        x = range(max_len)
        mean_gt = np.nanmean(pad_gt, axis=0)
        mean_pr = np.nanmean(pad_pr, axis=0)
        std_gt = np.nanstd(pad_gt, axis=0)
        std_pr = np.nanstd(pad_pr, axis=0)

        axes[0].plot(x, mean_gt, label=f"{mn} cos(h, w_GT)", color=model_colors[mn], linewidth=2)
        axes[0].fill_between(x, mean_gt - std_gt, mean_gt + std_gt,
                             color=model_colors[mn], alpha=0.1)
        axes[0].plot(x, mean_pr, label=f"{mn} cos(h, w_Pred)", color=model_colors[mn],
                     linewidth=2, linestyle="--")
        axes[0].fill_between(x, mean_pr - std_pr, mean_pr + std_pr,
                             color=model_colors[mn], alpha=0.05)

    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("Cosine Similarity")
    axes[0].set_title("(a) Prefill: cosine(h[layer], w_option)\n(Solid=GT, Dashed=Pred)")
    axes[0].legend(loc="best", fontsize=7)
    axes[0].grid(alpha=0.3)

    # ---- (b) cos_diff = cos_GT - cos_Pred per layer ----
    for mn in ["1.5B", "3B", "7B"]:
        diff_curves = []
        for r in all_results:
            ld = r["per_model"].get(mn, {})
            layers = ld.get("prefill_layer_data", [])
            if not layers:
                continue
            diff_curves.append([l["cos_diff"] for l in layers])

        if not diff_curves:
            continue

        max_len = max(len(c) for c in diff_curves)
        padded = [c + [np.nan] * (max_len - len(c)) for c in diff_curves]
        padded = np.array(padded)

        x = range(max_len)
        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)

        axes[1].plot(x, mean_curve, label=mn, color=model_colors[mn], linewidth=2)
        axes[1].fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                             color=model_colors[mn], alpha=0.15)

    axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.4, label="0 (no separation)")
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("cos(h, w_GT) - cos(h, w_Pred)")
    axes[1].set_title("(b) Layer-wise GT-Pred Separation\n(Higher = GT more aligned, Lower = Pred more aligned)")
    axes[1].legend(loc="best", fontsize=10)
    axes[1].grid(alpha=0.3)

    # ---- (c) Option cosine spread per layer ----
    for mn in ["1.5B", "3B", "7B"]:
        spread_curves = []
        for r in all_results:
            ld = r["per_model"].get(mn, {})
            layers = ld.get("prefill_layer_data", [])
            if not layers:
                continue
            spread_curves.append([l["option_cos_spread"] for l in layers])

        if not spread_curves:
            continue

        max_len = max(len(c) for c in spread_curves)
        padded = [c + [np.nan] * (max_len - len(c)) for c in spread_curves]
        padded = np.array(padded)

        x = range(max_len)
        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)

        axes[2].plot(x, mean_curve, label=mn, color=model_colors[mn], linewidth=2)
        axes[2].fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                             color=model_colors[mn], alpha=0.15)

    axes[2].set_xlabel("Layer Index")
    axes[2].set_ylabel("Std of cos(h, w_option) across A/B/C/D")
    axes[2].set_title("(c) Option Cosine Spread\n(Lower = options 'squeezed' together)")
    axes[2].legend(loc="best", fontsize=10)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_cosine_distance_layers.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_final_layer_comparison(all_results: List[Dict], output_dir: str):
    """图2: 最终层余弦距离分布对比（error vs correct）"""
    plt = _setup_plt()
    os.makedirs(output_dir, exist_ok=True)

    model_colors = {"1.5B": "#2196F3", "3B": "#4CAF50", "7B": "#FF9800"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # ---- (a) cos_diff (final layer) boxplot per model ----
    cos_diffs = {m: {"error": [], "correct": []} for m in ["1.5B", "3B", "7B"]}
    cos_gt_finals = {m: {"error": [], "correct": []} for m in ["1.5B", "3B", "7B"]}
    spreads_finals = {m: {"error": [], "correct": []} for m in ["1.5B", "3B", "7B"]}

    for r in all_results:
        for mn in ["1.5B", "3B", "7B"]:
            pm = r["per_model"].get(mn, {})
            flm = pm.get("final_layer_metrics", {})
            if not flm:
                continue

            split = "error" if pm.get("is_error", False) else "correct"
            if "cos_diff_final" in flm:
                cos_diffs[mn][split].append(flm["cos_diff_final"])
            if "cos_gt_final" in flm:
                cos_gt_finals[mn][split].append(flm["cos_gt_final"])
            if "option_cos_spread_final" in flm:
                spreads_finals[mn][split].append(flm["option_cos_spread_final"])

    # (a) cos_diff boxplot
    data_groups = []
    labels = []
    colors_list = []
    for mn in ["1.5B", "3B", "7B"]:
        for split in ["error", "correct"]:
            vals = cos_diffs[mn][split]
            if vals:
                data_groups.append(vals)
                labels.append(f"{mn}\n{split}")
                alpha = 0.7 if split == "error" else 0.4
                colors_list.append(model_colors[mn])

    if data_groups:
        bp = axes[0].boxplot(data_groups, labels=labels, patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.5, label="0 (no separation)")
    axes[0].set_ylabel("cos(h_final, w_GT) - cos(h_final, w_Pred)")
    axes[0].set_title("(a) Final Layer GT-Pred Cosine Diff\n(Higher = GT more aligned)")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)

    # (b) cos(h, w_GT) final layer: error vs correct histogram
    for mn in ["1.5B", "3B", "7B"]:
        err_vals = cos_gt_finals[mn]["error"]
        ok_vals = cos_gt_finals[mn]["correct"]
        if err_vals:
            axes[1].hist(err_vals, bins=20, alpha=0.6, color=model_colors[mn],
                         label=f"{mn} error", edgecolor="white")
        if ok_vals:
            axes[1].hist(ok_vals, bins=20, alpha=0.3, color=model_colors[mn],
                         label=f"{mn} correct", edgecolor="white", linestyle="--")

    axes[1].set_xlabel("cos(h_final, w_GT)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("(b) Final Layer: cos(h, w_GT)\n(How aligned is hidden state with GT option)")
    axes[1].legend(loc="best", fontsize=7)
    axes[1].grid(alpha=0.3)

    # (c) Option spread: error vs correct
    data_spread = []
    labels_spread = []
    colors_spread = []
    for mn in ["1.5B", "3B", "7B"]:
        for split in ["error", "correct"]:
            vals = spreads_finals[mn][split]
            if vals:
                data_spread.append(vals)
                labels_spread.append(f"{mn}\n{split}")
                colors_spread.append(model_colors[mn])

    if data_spread:
        bp2 = axes[2].boxplot(data_spread, labels=labels_spread, patch_artist=True, widths=0.5)
        for patch, color in zip(bp2["boxes"], colors_spread):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    axes[2].set_ylabel("Std of cos(h_final, w_option) across options")
    axes[2].set_title("(c) Final Layer Option Cosine Spread\n(Lower = more 'squeezed')")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_cosine_distance_final_layer.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_generation_step_cosine(all_results: List[Dict], output_dir: str):
    """图3: 答案生成步的余弦距离分析"""
    plt = _setup_plt()
    os.makedirs(output_dir, exist_ok=True)

    model_colors = {"1.5B": "#2196F3", "3B": "#4CAF50", "7B": "#FF9800"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Generation step 0: cos_diff across layers
    for mn in ["1.5B", "3B", "7B"]:
        diff_curves = []
        for r in all_results:
            ld = r["per_model"].get(mn, {})
            layers = ld.get("gen_step0_layer_data", [])
            if not layers:
                continue
            diff_curves.append([l["cos_diff"] for l in layers])

        if not diff_curves:
            continue

        max_len = max(len(c) for c in diff_curves)
        padded = [c + [np.nan] * (max_len - len(c)) for c in diff_curves]
        padded = np.array(padded)

        x = range(max_len)
        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)

        axes[0].plot(x, mean_curve, label=mn, color=model_colors[mn], linewidth=2)
        axes[0].fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                             color=model_colors[mn], alpha=0.15)

    axes[0].axhline(y=0, color="red", linestyle="--", alpha=0.4, label="0 (no separation)")
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("cos(h_gen, w_GT) - cos(h_gen, w_Pred)")
    axes[0].set_title("(a) Answer Gen Step: GT-Pred Cosine Diff\n(At the moment of answer generation)")
    axes[0].legend(loc="best", fontsize=10)
    axes[0].grid(alpha=0.3)

    # (b) Scatter: final prefill cos_diff vs gen step cos_diff
    for mn in ["1.5B", "3B", "7B"]:
        prefill_diffs = []
        gen_diffs = []
        is_err = []
        for r in all_results:
            ld = r["per_model"].get(mn, {})
            flm = ld.get("final_layer_metrics", {})
            gen0 = ld.get("gen_step0_layer_data", [])
            if not flm or not gen0:
                continue
            prefill_diffs.append(flm.get("cos_diff_final", 0))
            # 用gen step的最后一层
            gen_diffs.append(gen0[-1]["cos_diff"])
            is_err.append(ld.get("is_error", False))

        if not prefill_diffs:
            continue

        err_p = [p for p, e in zip(prefill_diffs, is_err) if e]
        err_g = [g for g, e in zip(gen_diffs, is_err) if e]
        ok_p = [p for p, e in zip(prefill_diffs, is_err) if not e]
        ok_g = [g for g, e in zip(gen_diffs, is_err) if not e]

        if ok_p:
            axes[1].scatter(ok_p, ok_g, alpha=0.4, s=30, color=model_colors[mn],
                             label=f"{mn} correct")
        if err_p:
            axes[1].scatter(err_p, err_g, alpha=0.6, s=50, color=model_colors[mn],
                             marker="x", linewidths=2, label=f"{mn} error")

    axes[1].axhline(y=0, color="red", linestyle="--", alpha=0.4)
    axes[1].axvline(x=0, color="red", linestyle="--", alpha=0.4)
    axes[1].set_xlabel("Prefill Final Layer cos_diff")
    axes[1].set_ylabel("Gen Step 0 Final Layer cos_diff")
    axes[1].set_title("(b) Prefill vs Gen Step: cos_diff\n(Quadrant III = confused in both)")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_cosine_distance_gen_step.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def run_analysis(
    num_samples: int,
    output_dir: str = "experiment_results/option_distance",
    device: str = "cpu",
):
    """运行完整的模型内特征余弦距离分析。"""
    from transformers import AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)

    # 加载tokenizer
    model_name = MODELS["qwen1.5B"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 获取选项token ID
    option_letters = ["A", "B", "C", "D"]
    option_token_ids = get_option_token_ids(tokenizer, option_letters, "multiple_choice")
    print(f"  Option token IDs: {option_token_ids}")

    # 加载错误样本列表
    cmp_path = ANALYSIS_CONFIG["three_model_comparison"]
    if not os.path.exists(cmp_path):
        print(f"  Error: three_model_comparison.csv not found at {cmp_path}")
        return

    df_cmp = pd.read_csv(cmp_path)
    error_indices = set(df_cmp[(df_cmp["correct_7B"]) & (~df_cmp["correct_1.5B"])]["sample_idx"].astype(int).tolist())
    correct_indices = set(df_cmp[(df_cmp["correct_7B"]) & (df_cmp["correct_1.5B"])]["sample_idx"].astype(int).tolist())
    all_indices = sorted(error_indices | correct_indices)
    print(f"\n  Error samples (7B ok, 1.5B wrong): {len(error_indices)}")
    print(f"  Correct samples (both ok): {len(correct_indices)}")
    print(f"  Total to analyze: {len(all_indices)}")

    if not all_indices:
        print("  No samples to analyze.")
        return

    # 加载模型输出
    print(f"\nLoading model outputs to {device}...")
    outputs_1_5B = load_model_outputs("qwen1.5B", num_samples, map_location=device)
    outputs_3B = load_model_outputs("qwen3B", num_samples, map_location=device)
    outputs_7B = load_model_outputs("qwen7B", num_samples, map_location=device)

    # 加载LM Head权重
    print(f"\nLoading LM Head weights to {device}...")
    lm_heads = {}
    for mn, key in [("1.5B", "qwen1.5B"), ("3B", "qwen3B"), ("7B", "qwen7B")]:
        w = load_lm_head_weight(key, device=device)
        if w is not None:
            lm_heads[mn] = w

    if not lm_heads:
        print("  Error: could not load any LM Head weights.")
        return

    # ============ 核心分析 ============
    print("\n--- Feature Cosine Distance Analysis ---")
    all_results = []
    for idx in tqdm(all_indices, desc="Analyzing feature cosine"):
        outputs_map = {
            "1.5B": outputs_1_5B[idx] if idx < len(outputs_1_5B) else None,
            "3B": outputs_3B[idx] if idx < len(outputs_3B) else None,
            "7B": outputs_7B[idx] if idx < len(outputs_7B) else None,
        }
        result = analyze_sample_feature_cosine(idx, outputs_map, option_token_ids, lm_heads, device=device)
        if result is not None:
            all_results.append(result)

    error_results = [r for r in all_results if r["per_model"].get("1.5B", {}).get("is_error", False)]
    correct_results = [r for r in all_results if not r["per_model"].get("1.5B", {}).get("is_error", False)]
    print(f"  Analyzed: {len(all_results)} total ({len(error_results)} error, {len(correct_results)} correct)")

    # 释放LM Head
    del lm_heads
    import gc
    gc.collect()

    # ============ 聚合统计 ============
    print(f"\n{'=' * 80}")
    print("Step 6b: Feature Cosine Distance Analysis (Pure Hidden State, No Logit)")
    print(f"{'=' * 80}")

    summary = {}

    for mn in ["1.5B", "3B", "7B"]:
        for split_name, split_data in [("error", error_results), ("correct", correct_results), ("all", all_results)]:
            valid = [r for r in split_data
                     if mn in r.get("per_model", {})
                     and "final_layer_metrics" in r["per_model"][mn]]
            if not valid:
                continue

            flm_list = [r["per_model"][mn]["final_layer_metrics"] for r in valid]
            baseline_list = [r["per_model"][mn]["baseline_cos_gt_pred"] for r in valid]

            cos_diff_vals = [f["cos_diff_final"] for f in flm_list if "cos_diff_final" in f]
            cos_gt_vals = [f["cos_gt_final"] for f in flm_list if "cos_gt_final" in f]
            cos_pred_vals = [f["cos_pred_final"] for f in flm_list if "cos_pred_final" in f]
            spread_vals = [f["option_cos_spread_final"] for f in flm_list if "option_cos_spread_final" in f]

            key = f"{mn}_{split_name}"
            s = {
                "n": len(valid),
                "baseline_cos_gt_pred_mean": round(float(np.mean(baseline_list)), 4),
            }

            if cos_diff_vals:
                s["cos_diff_mean"] = round(float(np.mean(cos_diff_vals)), 4)
                s["cos_diff_median"] = round(float(np.median(cos_diff_vals)), 4)
                s["cos_diff_std"] = round(float(np.std(cos_diff_vals)), 4)
                s["frac_pred_more_aligned"] = round(sum(1 for v in cos_diff_vals if v < 0) / len(cos_diff_vals), 4)
            if cos_gt_vals:
                s["cos_gt_mean"] = round(float(np.mean(cos_gt_vals)), 4)
            if cos_pred_vals:
                s["cos_pred_mean"] = round(float(np.mean(cos_pred_vals)), 4)
            if spread_vals:
                s["spread_mean"] = round(float(np.mean(spread_vals)), 4)

            # 逐层cos_diff的最后几层趋势
            layer_diffs_by_sample = []
            for r in valid:
                layers = r["per_model"][mn].get("prefill_layer_data", [])
                if len(layers) >= 4:
                    # 最后4层的cos_diff
                    last_4 = [l["cos_diff"] for l in layers[-4:]]
                    layer_diffs_by_sample.append(last_4)

            if layer_diffs_by_sample:
                arr = np.array(layer_diffs_by_sample)
                s["last_4_layers_cos_diff_mean"] = [round(float(v), 4) for v in np.mean(arr, axis=0).tolist()]
                # 趋势: 最后4层cos_diff是否递减
                trends = []
                for row in layer_diffs_by_sample:
                    if len(row) >= 2:
                        trends.append(row[-1] - row[-4])
                s["cos_diff_trend_last4"] = round(float(np.mean(trends)), 4)

            summary[key] = s

    # 打印结果
    print(f"\n  --- Final Layer Cosine Distance: cos(h_final, w_GT) - cos(h_final, w_Pred) ---")
    for split in ["error", "correct"]:
        print(f"\n  [{split.upper()} samples]")
        for mn in ["1.5B", "3B", "7B"]:
            key = f"{mn}_{split}"
            if key not in summary:
                continue
            s = summary[key]
            parts = [f"{mn} (n={s['n']}):"]
            if "cos_diff_mean" in s:
                parts.append(f"cos_diff={s['cos_diff_mean']:.4f} +/- {s['cos_diff_std']:.4f}")
            if "cos_gt_mean" in s:
                parts.append(f"cos_GT={s['cos_gt_mean']:.4f}")
            if "cos_pred_mean" in s:
                parts.append(f"cos_Pred={s['cos_pred_mean']:.4f}")
            if "frac_pred_more_aligned" in s:
                parts.append(f"Pred_more_aligned={s['frac_pred_more_aligned']*100:.1f}%")
            if "spread_mean" in s:
                parts.append(f"spread={s['spread_mean']:.4f}")
            print(f"    {' | '.join(parts)}")

    # 假说检验
    print(f"\n  *** Hypothesis Test: Resolution Compression (Feature Space) ***")
    err_1_5B = summary.get("1.5B_error", {})
    err_7B = summary.get("7B_error", {})
    if err_1_5B and err_7B:
        cd_1_5B = err_1_5B.get("cos_diff_mean", 0)
        cd_7B = err_7B.get("cos_diff_mean", 0)
        sp_1_5B = err_1_5B.get("spread_mean", 0)
        sp_7B = err_7B.get("spread_mean", 0)
        tr_1_5B = err_1_5B.get("cos_diff_trend_last4", 0)
        tr_7B = err_7B.get("cos_diff_trend_last4", 0)
        fa_1_5B = err_1_5B.get("frac_pred_more_aligned", 0)
        fa_7B = err_7B.get("frac_pred_more_aligned", 0)

        print(f"    Error samples:")
        print(f"    1.5B: cos_diff={cd_1_5B:.4f}, spread={sp_1_5B:.4f}, "
              f"trend_last4={tr_1_5B:.4f}, Pred_aligned={fa_1_5B*100:.1f}%")
        print(f"    7B:   cos_diff={cd_7B:.4f}, spread={sp_7B:.4f}, "
              f"trend_last4={tr_7B:.4f}, Pred_aligned={fa_7B*100:.1f}%")

        if cd_1_5B < 0 and cd_7B > 0:
            print(f"    >>> STRONG SUPPORT: 1.5B Pred more aligned (cos_diff<0), "
                  f"7B GT more aligned (cos_diff>0)")
        elif cd_1_5B < cd_7B and (cd_7B - cd_1_5B) > 0.05:
            print(f"    >>> SUPPORTS: 1.5B has smaller GT-Pred separation in feature space")
        elif tr_1_5B < tr_7B and tr_1_5B < -0.01:
            print(f"    >>> SUPPORTS: 1.5B's separation DECREASES in last layers (trend={tr_1_5B:.4f})")
        elif sp_1_5B < sp_7B - 0.005:
            print(f"    >>> WEAK SUPPORT: 1.5B has lower option spread ({sp_1_5B:.4f} vs {sp_7B:.4f})")
        else:
            print(f"    >>> WEAK / NO SUPPORT in feature space")

    # 保存结果
    save_path = os.path.join(output_dir, "option_distance_analysis_v2.json")
    save_json({"summary": summary, "n_error": len(error_results), "n_correct": len(correct_results)}, save_path)
    print(f"\n  Results saved to: {save_path}")

    # 生成可视化
    print("\n  Generating plots...")
    plot_cosine_distance_curves(all_results, output_dir)
    plot_final_layer_comparison(all_results, output_dir)
    plot_generation_step_cosine(all_results, output_dir)

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Step 6b: Feature Cosine Distance (pure hidden state cosine analysis)"
    )
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--output_dir", type=str, default="experiment_results/option_distance")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for computation (default: cuda if available)")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    print("=" * 80)
    print("Step 6b: Feature Cosine Distance Measurement")
    print("  Measures within-model GT vs Pred separation in hidden state space")
    print("  Uses COSINE DISTANCE (not logit/projection) to avoid NLP output contamination")
    print("  Tests: does 1.5B 'squeeze' GT and Pred together in feature space?")
    print("=" * 80)

    run_analysis(
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        device=device,
    )

    print("\n" + "=" * 80)
    print("Step 6b Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
