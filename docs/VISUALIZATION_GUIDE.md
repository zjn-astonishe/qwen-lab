# 📊 可视化结果完全指南

本实验框架自动生成 **14+ 种专业可视化图表**，帮助深入理解异构模型的隐藏状态对齐、概率演化及注入效果。

## 目录

1. [概率探测可视化（Step 4）](#1-概率探测可视化-step-4)
2. [CKA 相似度可视化（Step 5）](#2-cka-相似度可视化-step-5)
3. [因果证明可视化（Steps 4b, 6b, 7b）](#3-因果证明可视化-steps-4b-6b-7b)
4. [决策性 Token 可视化（Step 6）](#4-决策性-token-可视化-step-6)
5. [注入效果可视化（Steps 8-9）](#5-注入效果可视化-steps-8-9)
6. [数据导出与自定义分析](#6-数据导出与自定义分析)
7. [可视化示例场景](#7-可视化示例场景)
8. [结果解读检查清单](#8-结果解读检查清单)

---

## 1. 概率探测可视化 (Step 4)

**自动生成时机**: 运行 `step4_probability_probing.py` 时自动生成

**存储位置**: `experiment_results/probing/`

### 1.1 聚合 P(GT) 曲线 (`probing_aggregate.png`)

**图表布局**: 1行3列的子图面板

#### 左图: 三模型平均 P(GT) 演化
- **X 轴**: 层级进度（0-100%）
- **Y 轴**: Ground Truth token 的平均概率
- **曲线**: 
  - 🔵 蓝色 - 1.5B 模型
  - 🟢 绿色 - 3B 模型  
  - 🔴 红色 - 7B 模型

**解读要点**:
```
✓ 上升趋势: 模型在推理过程中逐渐"锁定"正确答案
✗ 下降或平台: 小模型可能失去对 GT token 的追踪
△ 分叉点: 观察小模型何时开始偏离大模型
```

#### 中图: GT Token 排名演化
- **Y 轴**: GT token 在 Top-K 中的平均排名（越低越好）
- **趋势**: 理想情况下应持续下降至 1-5 名

#### 右图: P(GT) / P(pred) 比率
- **含义**: GT token 概率相对于预测 token 概率的比值
- **阈值**: > 1.0 表示模型"内心"偏向正确答案，但可能未输出

**应用场景**:
- 📊 快速概览三个模型的整体性能差异
- 🎯 识别小模型"失误"的典型阶段
- 🔄 验证模型是否存在"自我修正"机制

---

### 1.2 个体样本演化 (`probing_individual.png`)

**图表布局**: 3行4列 = 12个子图

**每个子图显示**:
- **实线**: P(GT) - Ground Truth token 概率
- **虚线**: P(pred) - 预测 token 概率
- **颜色**: 蓝(1.5B), 绿(3B), 红(7B)
- **标题**: 样本索引 + 是否正确

**典型模式识别**:

```
模式 1: 早期分歧
  1.5B: P(GT) 在 20% 就低于 P(pred)
  7B: P(GT) 始终高于 P(pred)
  → 小模型从一开始就走错方向

模式 2: 中期迷失
  1.5B: P(GT) 在 50% 之前与 7B 接近
  1.5B: 之后 P(GT) 突然下降
  → 小模型在推理中途"转向"

模式 3: 晚期崩溃
  1.5B: P(GT) 在 80% 之前接近 7B
  1.5B: 最后阶段 P(GT) < P(pred)
  → 小模型在最后决策时犹豫
```

**使用建议**:
- 挑选 2-3 个典型案例深入分析
- 结合原始问题文本理解为何小模型出错
- 寻找可能的干预点（早期/中期/晚期）

---

### 1.3 分歧点分析 (`probing_divergence.png`)

**图表布局**: 1行2列

#### 左图: 分歧点分布直方图
- **X 轴**: 分歧发生的层级进度（%）
- **Y 轴**: 样本数量
- **分歧定义**: P(GT)_large - P(GT)_small > 阈值（默认 0.1）

**峰值含义**:
```
峰值在 0-30%: 早期分歧型错误，可能是理解问题本身有误
峰值在 30-70%: 中期分歧型错误，推理路径选择问题
峰值在 70-100%: 晚期分歧型错误，最终决策失误
```

#### 右图: 最优切换点分析
- **X 轴**: 潜在切换点（层级进度%）
- **Y 轴**: 如果在此切换，能挽救的错误数量
- **峰值**: 建议的级联推理切换时机

**应用**: 指导早期退出（Early Exit）策略设计

---

### 1.4 熵演化对比 (`probing_entropy.png`)

**图表布局**: 1行2列

#### 左图: 三模型熵演化曲线
- **Y 轴**: Shannon 熵（nats）
- **理想趋势**: 熵应随层级增加而下降（模型越来越确定）

**异常模式**:
```
⚠️ 熵不下降: 模型始终不确定，可能是困难样本
⚠️ 熵先降后升: 模型曾经确定，后来又迷茫
⚠️ 小模型熵 >> 大模型熵: B 球困境的信号
```

#### 右图: 模型间熵差异
- **Y 轴**: 熵差值（7B - 1.5B, 7B - 3B）
- **正值**: 小模型比大模型更确定（可能过度自信）
- **负值**: 大模型更确定（正常情况）

---

### 1.5 早期退出决策矩阵 (`probing_early_exit.png`)

**图表布局**: 1行2列

#### 左图: P(GT) 热力图
- **X 轴**: 层级进度（%）
- **Y 轴**: 模型（1.5B, 3B, 7B）
- **颜色**: 越红 P(GT) 越高

**用途**: 可视化"何时何模型有最高 P(GT)"

#### 右图: 置信度达标比例
- **X 轴**: 置信度阈值（0.3, 0.5, 0.7）
- **Y 轴**: 各层达到阈值的样本比例
- **应用**: 确定"需要多少层才能获得足够信心"

**实际应用**:
```python
# 基于此图设计早期退出策略
if layer_progress > 60% and P(GT)_1.5B > 0.5:
    # 小模型足够自信，无需调用大模型
    return small_model_prediction
else:
    # 切换到大模型
    return large_model_prediction
```

---

## 2. CKA 相似度可视化 (Step 5)

**自动生成时机**: 运行 `step5_cka_analysis.py` 时自动生成

**存储位置**: `experiment_results/analysis/`

### 2.1 CKA 热力图 (3 个)

#### `cka_matrix_1.5Bvs7B.png`
- **行**: 1.5B 模型的 28 层
- **列**: 7B 模型的 28 层
- **值**: CKA 相似度（0-1，越红越相似）

**解读策略**:

```
对角线模式:
  对角线值高(>0.6): 对应层表征相似，容易对齐
  对角线值低(<0.4): 对应层表征差异大，投影困难

非对角线峰值:
  例如 [10, 20] 处有高值(>0.7):
  → 1.5B 的第 10 层 ≈ 7B 的第 20 层
  → 考虑用 7B-L20 指导 1.5B-L10 的投影

层级趋势:
  后层普遍高于前层: 符合预期（后层更语义化）
  前层也有高值: 可能存在早期特征对齐
```

#### `cka_matrix_7Bvs3B.png`
- 7B(28层) vs 3B(36层)
- **注意**: 3B 有更多层，矩阵非方阵

#### `cka_matrix_1.5Bvs3B.png`
- 1.5B(28层) vs 3B(36层)

**比较三个热力图**:
- 验证"缩放一致性": 相似的模式应在所有模型对中出现
- 寻找"通用层": 在所有热力图中都有高 CKA 的层

---

### 2.2 CKA 曲线图 (`cka_curves.png`)

**图表布局**: 1行2列

#### 左图: 对角线 CKA 相似度
- **X 轴**: 层索引
- **Y 轴**: 对角线 CKA 值
- **曲线**: 三个模型对的对角线趋势

**理想模式**: 后层(L20-L28)值 > 前层(L1-L10)值

#### 右图: 每层的最大 CKA 值
- **含义**: 每一层在所有可能匹配中的最佳 CKA
- **用途**: 识别"最易对齐"的层

**应用示例**:
```
如果 L25 的最大 CKA = 0.92:
  → L25 是投影训练的优选层
如果 L3 的最大 CKA = 0.35:
  → L3 不适合投影，表征差异太大
```

---

## 3. 因果证明可视化 (Steps 4b, 6b, 7b)

### 3.1 Logit 语义聚类 (Step 4b)

#### `logit_cluster_umap_*.png`

**图表内容**: 2D UMAP 降维后的聚类可视化

- **点**: 每个样本的 logit 向量
- **颜色**: 聚类标签（K-means，默认 K=5）
- **标记**: 不同标记区分两个模型
  - 圆圈: 小模型（如 1.5B）
  - 叉号: 大模型（如 7B）

**理想现象**:
```
✓ 同色区域重叠: 两个模型的同一聚类在空间上接近
✓ 聚类边界清晰: 不同语义类别分离明显
✗ 聚类混乱: 可能存在对齐问题或数据问题
```

#### `logit_cluster_similarity_heatmap_*.png`

**图表内容**: 聚类中心余弦相似度矩阵

- **行**: 小模型的聚类中心
- **列**: 大模型的聚类中心
- **值**: 余弦相似度（-1 到 1）

**匈牙利算法匹配结果**:
- 对角线（经重排）应该最亮
- 平均对角线相似度 > 0.7 表示强对齐

**因果解释**:
```
如果聚类匹配度高:
  → 两个模型在语义空间的组织结构相似
  → 支持"隐藏表示对齐"假设
  → 投影矩阵有理论基础
```

---

### 3.2 特征余弦距离 (Step 6b)

#### `feature_distance_distribution_*.png`

**图表布局**: 2行1列

#### 上图: 余弦相似度分布直方图
- **蓝色**: 训练投影后的余弦相似度
- **红色**: 随机投影基线
- **理想结果**: 蓝色分布明显右移（更高相似度）

**统计量标注**:
```
Mean (Trained): 0.87  ← 应该 > 0.80
Mean (Random): 0.12
P-value: 1.2e-45  ← 应该 < 0.001 (显著性)
```

#### 下图: 分模型对比（箱线图）
- **X 轴**: 模型对（1.5B→7B, 3B→7B, 1.5B→3B）
- **Y 轴**: 余弦相似度
- **用途**: 比较不同模型对的对齐难度

**因果解释**:
```
如果训练投影 >> 随机投影:
  → 投影矩阵确实捕捉到了模型间的对齐关系
  → 不仅仅是维度扩展，而是语义对应
  → 支持后续的状态注入实验
```

---

### 3.3 交叉解码 KL 散度 (Step 7b)

#### `cross_decode_kl_divergence_*.png`

**图表布局**: 2行1列

#### 上图: KL 散度分布
- **蓝色**: 交叉解码 KL(small_dist || cross_decoded_dist)
- **绿色**: 自身解码 KL(small_dist || small_dist) = 0 (参考)
- **橙色**: 随机基线

**理想结果**: 蓝色分布接近绿色（均值 < 1.0）

#### 下图: 累积分布函数 (CDF)
- **X 轴**: KL 散度阈值
- **Y 轴**: 小于阈值的样本比例
- **曲线**: 越陡峭越好（大部分样本 KL 低）

**因果解释**:
```
如果交叉解码 KL ≈ 自身解码:
  → 小模型隐藏状态在大模型 LM head 下产生相似分布
  → 隐藏空间真正实现了对齐
  → 最强的因果证据
```

**三阶段因果证明总结**:
```
Phase I (Logit Cluster): 语义空间结构相似 ✓
Phase II (Feature Distance): 投影后特征接近 ✓
Phase III (Cross Decode): 解码分布一致 ✓
→ 三重验证支持"隐藏状态对齐"假设
```

---

## 4. 决策性 Token 可视化 (Step 6)

#### `decisive_tokens_{model}_summary.png`

**图表布局**: 可变（基于 max_display 参数）

**每个子图显示**:
- 一个高熵样本的所有输入 token
- **柱状图**: 每个 token 的重要性得分
- **颜色**: 重要性越高越红
- **标注**: 最重要的 Top-3 token

**重要性得分定义**:
```
Score = |P(original_pred) - P(pred_without_token)|
```

**典型发现**:
```
模式 1: 问题关键词决策
  例如 "which", "how many" 等疑问词得分高
  → 模型依赖问题类型做决策

模式 2: 数值信息决策
  例如数字 token 得分高
  → 数学题依赖具体数值

模式 3: 选项标签决策  
  例如 "(A)", "(B)" 等得分高
  → 可能存在位置偏差
```

**应用价值**:
- Prompt 工程: 强调重要 token
- 数据增强: 构造相似的决策性 token
- 鲁棒性测试: 移除/替换决策性 token

---

## 5. 注入效果可视化 (Steps 8-9)

### 5.1 概率分布对比 (`case_XXX_comparison.png`)

**图表布局**: 1行2列

#### 左图: 小模型 (1.5B) 分布
- **Y 轴**: Top-30 token（按概率降序）
- **X 轴**: 概率值
- **颜色**: 珊瑚色 (coral)

#### 右图: 大模型 (7B) 分布
- **Y 轴**: Top-30 token
- **X 轴**: 概率值
- **颜色**: 天蓝色 (skyblue)

**B 球困境特征**:
```
小模型特征:
  ✗ 多个相似 token 概率接近（如 0.12, 0.11, 0.10）
  ✗ GT token 排名 10-20 名
  ✗ 概率分布"平坦"（熵高）

大模型特征:
  ✓ GT token 概率 > 0.5, 排名 1-3
  ✓ 概率分布"尖锐"（熵低）
  ✓ Top-1 与 Top-2 差距大
```

**对比分析**:
```
重叠度分析:
  Top-10 重叠 > 70%: 两模型"关注"相同候选
  但小模型无法"决断"出正确的那个

语义分析:
  小模型 Top-10: get_weather, get_climate, get_temperature
  → 都是天气相关，语义相近（类别内混淆）
```

---

### 5.2 注入前后对比 (`injection_sample_XXX_*.png`)

**图表布局**: 1行2列

#### 左图: 注入前（7B 原始）
- Top-30 概率分布
- **红色**: Ground Truth token 位置

#### 右图: 注入后
- 注入小模型知识后的分布
- **标题标注**: α 值和注入层
- **红色**: GT token 新位置

**成功注入的特征**:
```
Before:
  GT token: 第 23 名, P = 0.03
  
After (α=0.3, layer=-2):
  GT token: 第 5 名, P = 0.15
  ↑ 排名提升 18 位
  ↑ 概率提升 5 倍
```

**注入效果分类**:
```
强效果 (GT rank 1-5):
  ✓ 错误被完全修正
  ✓ α = 0.1-0.3 效果最好

中等效果 (GT rank 6-15):
  △ GT token 可见性提升
  △ 可能需要调整 α 或层

弱效果 (GT rank > 15):
  ✗ 注入未显著改变分布
  ✗ 可能是投影质量问题或样本特殊
```

---

### 5.3 实验总结面板 (`experiment_summary.png`)

**图表布局**: 2行2列 = 4 个子图

#### 子图 1: 错误类型分布（左上）
- **柱状图**: 不同错误类型的数量
- **标注**: B 球困境占错误的百分比

#### 子图 2: 注入效果 vs Alpha（右上）
- **折线图**: 不同 α 值下的平均 GT rank
- **Y 轴反转**: 越低越好
- **最优点标注**: 最佳 α 值

**典型曲线**:
```
U 型曲线:
  α=0.1: rank ≈ 15 (注入太弱)
  α=0.3: rank ≈ 8  (最优)
  α=0.8: rank ≈ 25 (注入过强，破坏原分布)
```

#### 子图 3: 注入效果 vs 层（左下）
- **柱状图**: 不同注入层的效果
- **最佳层标注**: 通常是倒数第 2-3 层

#### 子图 4: 熵分布（右下）
- **直方图**: 错误样本的熵分布
- **标注**: 中位数和平均值
- **参考线**: 熵阈值（红色虚线）

---

## 6. 数据导出与自定义分析

所有可视化的底层数据均可导出进行自定义分析：

### 6.1 概率探测数据

```python
import json
import pandas as pd

# JSON 格式 - 完整详细数据
with open("experiment_results/probing/probing_results.json") as f:
    probing_data = json.load(f)

# 访问单个样本
sample = probing_data[0]
print(f"Sample {sample['sample_idx']}")
print(f"1.5B final P(GT): {sample['1.5B_final_gt_prob']:.4f}")
print(f"7B final P(GT): {sample['7B_final_gt_prob']:.4f}")

# 访问层级详情
for layer_info in sample['1.5B_layers']:
    print(f"Layer {layer_info['layer']}: P(GT)={layer_info['gt_prob']:.4f}")

# CSV 格式 - 扁平化 per-layer 数据
df_probe = pd.read_csv("experiment_results/probing/probing_per_layer.csv")

# 分析
print("\n按模型统计 P(GT):")
print(df_probe.groupby('model')['gt_prob'].describe())

# 按层级统计
print("\n按层级统计 P(GT):")
print(df_probe.groupby('layer_idx')['gt_prob'].mean())
```

### 6.2 CKA 矩阵数据

```python
import numpy as np
import matplotlib.pyplot as plt

# 加载 CKA 矩阵
cka_matrix = np.load("experiment_results/analysis/cka_matrix_1.5Bvs7B.npy")
print(f"Shape: {cka_matrix.shape}")  # (28, 28)

# 分析对角线
diagonal = np.diag(cka_matrix)
print(f"Mean diagonal CKA: {diagonal.mean():.4f}")

# 找最佳层对
best_pair = np.unravel_index(np.argmax(cka_matrix), cka_matrix.shape)
print(f"Best layer pair: {best_pair}, CKA={cka_matrix[best_pair]:.4f}")

# 自定义可视化
plt.figure(figsize=(10, 8))
plt.imshow(cka_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
plt.colorbar(label='CKA Similarity')
plt.title('Custom CKA Heatmap')
plt.xlabel('7B Layers')
plt.ylabel('1.5B Layers')
plt.tight_layout()
plt.savefig('custom_cka.png', dpi=300)
```

### 6.3 注入结果数据

```python
# 读取注入结果
df_inj = pd.read_csv("experiment_results/analysis/injection_results_qwen1.5B_vs_qwen7B.csv")

# 分析最佳配置
best_configs = df_inj.groupby(['alpha', 'injection_layer'])['gt_rank_after_injection'].mean()
print("Top 5 configurations:")
print(best_configs.sort_values().head())

# 分析修正率
correction_rate = df_inj.groupby('alpha')['error_corrected'].mean()
print("\nCorrection rate by alpha:")
print(correction_rate)

# 按样本分析
sample_perf = df_inj.groupby('sample_idx')['gt_rank_after_injection'].min()
print(f"\nSamples with GT rank < 10: {(sample_perf < 10).sum()}")
```

### 6.4 自定义可视化示例

```python
import torch
import matplotlib.pyplot as plt

# 加载单个样本的模型输出
output = torch.load("experiment_results/model_outputs/qwen1.5B/sample_000.pt")

# 提取第一步的概率分布
probs = output["probs_per_step"][0]  # [vocab_size]

# 自定义 Top-K 可视化
top_k = 50
values, indices = torch.topk(probs, k=top_k)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
tokens = [tokenizer.decode([idx.item()]) for idx in indices]

plt.figure(figsize=(14, 10))
plt.barh(range(top_k), values.numpy()[::-1], color='steelblue', alpha=0.8)
plt.yticks(range(top_k), tokens[::-1], fontsize=8)
plt.xlabel('Probability', fontsize=12)
plt.title('Custom Top-50 Token Distribution', fontsize=14)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('custom_top50.png', dpi=300)
```

---

## 7. 可视化示例场景

### 场景 1: 发现高度对齐的层

**目标**: 为投影矩阵训练选择最佳层

**步骤**:
1. 查看 `cka_matrix_1.5Bvs7B.png`
2. 识别高 CKA 值的层对（例如 L15 ↔ L28, CKA=0.85）
3. 在 `step7_train_projection.py` 中使用该层训练投影
4. 预期: 投影质量更高，余弦相似度 > 0.90

**验证**:
- 检查 `feature_distance_distribution_*.png`
- 确认该层对的余弦相似度确实更高

---

### 场景 2: 识别 B 球困境

**目标**: 定位并理解类别内混淆错误

**步骤**:
1. 查看 `error_analysis.csv`，筛选 `is_b_ball_dilemma=True`
2. 打开对应的 `case_XXX_comparison.png`
3. 观察小模型的 Top-10:
   ```
   get_weather: 0.15
   get_temperature: 0.14
   get_climate: 0.12
   check_weather: 0.11
   weather_info: 0.10
   ```
4. 对比大模型: `get_weather: 0.68` (明确的第一名)

**结论**: 小模型知道是"天气"相关，但无法区分具体函数

---

### 场景 3: 优化注入参数

**目标**: 找到最佳的 α 值和注入层

**步骤**:
1. 查看 `experiment_summary.png` 的右上图（注入效果 vs Alpha）
2. 观察 U 型曲线的最低点，例如 α=0.3
3. 查看左下图（注入效果 vs 层），例如 layer=-2 最佳
4. 在新数据上使用 α=0.3, layer=-2 进行注入

**细化**:
- 对于不同类型的样本，最佳参数可能不同
- 高熵样本: α ↑ (需要更强注入)
- 低熵样本: α ↓ (避免过度干扰)

---

### 场景 4: 设计早期退出策略

**目标**: 优化级联推理的切换时机

**步骤**:
1. 查看 `probing_divergence.png` 的右图（最优切换点）
2. 观察峰值位置，例如在 50% 层级进度
3. 查看 `probing_early_exit.png` 的热力图，确认该层级的 P(GT) 分布
4. 设计策略：
   ```python
   if layer_progress >= 50% and P(GT) < confidence_threshold:
       # 切换到大模型
       switch_to_large_model()
   ```

**效益分析**:
- 节省计算：仅在必要时调用大模型
- 保持准确率：及时切换避免错误积累

---

### 场景 5: 验证因果假设

**目标**: 确认投影矩阵确实捕捉到模型间的对齐关系

**步骤**:
1. **Phase I**: 查看 `logit_cluster_umap_*.png`
   - 确认两模型的聚类在空间上有重叠
   - 查看相似度热力图，确认匹配度 > 0.7

2. **Phase II**: 查看 `feature_distance_distribution_*.png`
   - 确认训练投影的余弦相似度显著高于随机基线
   - P-value < 0.001 表示统计显著

3. **Phase III**: 查看 `cross_decode_kl_divergence_*.png`
   - 确认交叉解码的 KL 散度 < 1.0
   - 与自身解码接近

**结论**: 三阶段验证通过 → 投影矩阵可靠

---

## 8. 结果解读检查清单

使用以下检查清单确保实验结果的完整性和可靠性：

### ✅ 数据完整性检查

- [ ] 所有样本的模型输出文件 (.pt) 都已生成
- [ ] 每个模型都有相同数量的输出文件
- [ ] 生成文本中包含 `generated_answer_only` 字段
- [ ] 隐藏状态维度与配置中的 `hidden_dim` 一致

### ✅ 概率探测检查

- [ ] `probing_aggregate.png` 显示清晰的 P(GT) 演化趋势
- [ ] 大模型的 P(GT) 曲线普遍高于小模型
- [ ] 分歧点分析揭示模型行为差异的典型阶段
- [ ] 熵演化曲线符合预期（随层级增加而下降）
- [ ] 早期退出矩阵显示合理的切换时机建议

### ✅ CKA 分析检查

- [ ] 热力图显示合理的层级对应关系
- [ ] 对角线趋势符合预期（后层高于前层）
- [ ] 三个模型对的 CKA 模式具有一致性
- [ ] 最大 CKA 值 > 0.6（表示存在可对齐的层）
- [ ] CKA 曲线图清晰展示层级趋势

### ✅ 因果证明检查

- [ ] Logit 聚类图显示清晰的聚类结构
- [ ] 聚类相似度热力图的平均对角线值 > 0.7
- [ ] 特征距离分布：训练投影 >> 随机基线
- [ ] P-value < 0.001 表示统计显著性
- [ ] 交叉解码 KL 散度 < 1.0

### ✅ 错误分析检查

- [ ] B 球困境案例占错误的合理比例（通常 10-30%）
- [ ] 错误类型分类合理且有代表性
- [ ] 概率分布对比图清晰展示小/大模型的差异
- [ ] 详细信息（熵、Top-K 重叠度）已正确记录

### ✅ 注入实验检查

- [ ] GT token 排名在注入后有显著提升
- [ ] 存在最优的 α 值（通常在 0.1-0.5）
- [ ] 最佳注入层通常是倒数第 2-3 层
- [ ] 错误修正率 > 10%（表示注入有效）
- [ ] 注入前后对比图显示明显的概率分布变化

### ✅ 可视化质量检查

- [ ] 所有图表的分辨率 >= 300 DPI
- [ ] 坐标轴标签清晰且有单位
- [ ] 图例和颜色编码一致
- [ ] 标题描述准确且信息完整
- [ ] 无数据截断或显示异常

### ✅ 总结报告检查

- [ ] `summary_report.json` 包含所有关键指标
- [ ] 控制台输出的报告格式清晰易读
- [ ] 各部分的状态都不是 "not_found"
- [ ] 数值统计合理且在预期范围内
- [ ] 包含三模型对比和因果验证结果

---

## 9. 常见问题与解决方案

### Q1: 概率探测图显示 P(GT) 始终为 0

**可能原因**:
- GT token 不在 vocabulary 中
- Answer extraction 失败

**解决方案**:
```bash
# 检查 error_analysis.csv
grep "no_output\|gt_unavailable" error_analysis.csv

# 重新检查答案提取逻辑
python -c "from qa_utils import extract_answer; print(extract_answer('Answer: B', 'multiple_choice'))"
```

### Q2: CKA 热力图全是低值 (< 0.3)

**可能原因**:
- 使用了不匹配的生成步骤
- 样本数量太少导致估计不准

**解决方案**:
```bash
# 增加样本数量
python step5_cka_analysis.py --num_samples 500

# 尝试不同的生成步骤
python step5_cka_analysis.py --step_idx 1
```

### Q3: 注入实验没有效果

**可能原因**:
- 投影矩阵质量差
- α 值选择不当
- 注入层选择不当

**解决方案**:
```bash
# 重新训练投影矩阵，尝试多层
python step7_train_projection.py --try_multiple_layers

# 检查投影质量
python -c "
import json
with open('experiment_results/projection_matrices/layer_comparison_qwen1.5B_to_qwen7B.json') as f:
    data = json.load(f)
    for layer, info in data.items():
        print(f'{layer}: cosine={info[\"test_cosine\"]:.4f}')
"

# 扩大 α 值搜索范围
# 修改 config.py 中的 INJECTION_CONFIG['alpha_values']
```

### Q4: 可视化图表中文显示为方块

**可能原因**:
- 系统缺少中文字体

**解决方案**:
```bash
# 安装中文字体
sudo apt-get install fonts-noto-cjk

# 或在代码中指定字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
```

### Q5: 内存不足错误

**解决方案**:
```bash
# 减少样本数量
python run_experiment.py --steps 4 --max_probing_samples 50

# 在 config.py 中减少 max_new_tokens
# GENERATION_CONFIG["max_new_tokens"] = 256

# 使用 CPU 运行部分步骤
python step4b_logit_cluster_analysis.py --device cpu
```

---

## 10. 最佳实践建议

### 📝 实验报告撰写

1. **结构化呈现**: 按 README 中的 Phase 顺序组织结果
2. **关键图表**: 每个 Phase 选择 2-3 个最具代表性的图表
3. **数值佐证**: 引用 summary_report.json 中的关键指标
4. **因果链**: 强调三阶段因果验证的逻辑连贯性

### 📊 可视化定制

1. **配色方案**: 使用色盲友好的配色（如 viridis, colorblind-safe）
2. **分辨率**: 论文用图使用 600 DPI
3. **字体大小**: 确保缩小后仍然可读（建议 10-12 pt）
4. **图例位置**: 避免遮挡重要数据

### 🔬 实验设计

1. **对照实验**: 始终包含随机基线对比
2. **样本多样性**: 确保测试集涵盖不同难度和类型
3. **超参数搜索**: 使用网格搜索或贝叶斯优化
4. **可重复性**: 固定随机种子，记录所有配置

### 💾 数据管理

1. **版本控制**: 使用 git 管理代码和配置
2. **结果归档**: 每次完整实验后压缩 experiment_results/
3. **中间结果**: 保留关键步骤的中间输出以便调试
4. **文档同步**: 及时更新 README 和注释

---

## 11. 引用与参考

如果您在研究中使用了本框架或可视化方法，请考虑引用：

```bibtex
@software{qwen_lab_2026,
  title={Heterogeneous Model Hidden State Alignment Framework},
  author={Your Name},
  year={2026},
  url={https://github.com/your-repo/qwen-lab}
}
```

### 相关工作

- **CKA**: Kornblith et al. "Similarity of Neural Network Representations Revisited" (ICML 2019)
- **Early Exit**: Xin et al. "DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference" (ACL 2020)
- **Model Alignment**: Csordás et al. "The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers" (EMNLP 2021)

---

## 📞 技术支持

遇到问题？

1. **查看 README**: 基础问题通常在 README.md 的"常见问题"部分有答案
2. **检查日志**: 查看各步骤的控制台输出，定位错误来源
3. **验证配置**: 确保 config.py 中的所有路径和参数正确
4. **提交 Issue**: 在 GitHub 仓库提交详细的问题描述

**祝您实验顺利！** 🎉
