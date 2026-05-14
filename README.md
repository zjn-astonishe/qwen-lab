# 异构模型隐藏状态对齐与概率分布演化实验

本项目实现了一个完整的实验框架，用于研究不同规模语言模型（Qwen 1.5B/3B/7B）在 QA 任务上的隐藏状态对齐、概率分布演化及跨模型状态注入。

## 🎯 实验目标

1. **概率粒度分析**: 比较不同规模模型的输出层概率分布演化过程
2. **隐藏状态对齐**: 分析模型间隐藏状态在关键决策点的相似性（CKA）
3. **错误模式识别**: 识别小模型的"B球困境"现象（类别内混淆）
4. **因果验证**: 通过语义聚类、特征距离和交叉解码验证对齐假设
5. **状态注入**: 验证通过简单投影实现跨模型隐藏状态注入的可行性

## 📊 核心创新

- **三模型概率探测**: 逐层追踪 1.5B/3B/7B 模型的概率演化
- **因果证明流程**: 三阶段验证（语义聚类 → 特征距离 → 交叉解码）
- **早期退出分析**: 量化级联推理的最佳切换点
- **决策性 Token 识别**: 定位关键影响生成结果的 token

## 🔧 系统要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 支持的 GPU（推荐至少 1 张 A100 80GB）
- 至少 200GB 磁盘空间（用于模型和数据）

## 📦 安装

1. 克隆仓库并进入目录：
```bash
git clone <repository-url>
cd qwen-lab
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

依赖包括：
- `transformers>=4.36.0` - 模型推理
- `torch>=2.0.0` - 深度学习框架
- `datasets` - 数据集加载
- `matplotlib, seaborn` - 可视化
- `scikit-learn` - CKA 计算
- `pandas, numpy` - 数据处理

## 🚀 快速开始

### 方式一：一键运行完整流程

```bash
# 运行所有步骤（推荐用于首次完整实验）
python run_experiment.py --all

# 测试模式（小样本量）
python run_experiment.py --all --max_samples 10
```

### 方式二：选择性运行特定步骤

```bash
# 运行数据准备、推理和分析
python run_experiment.py --steps 1 2 3 4

# 运行因果证明流程
python run_experiment.py --steps 4b 6b 7b

# 运行投影训练和注入实验
python run_experiment.py --steps 7 8 9 10
```

### 方式三：逐步手动运行

适合调试或深入理解每个步骤。

## 📋 实验流程详解

### 🔹 Phase 1: 数据与推理

#### Step 1: 数据准备
```bash
python step1_prepare_data.py --total_samples 300
```

从 HuggingFace 加载 QA 数据集（GSM8K、ARC-Challenge、MMLU）并采样。

**输出**:
- `experiment_results/sampled_data/sampled_300.json` - 测试样本
- `experiment_results/sampled_data/alignment_100.json` - 对齐样本

**选项**:
- `--total_samples`: 测试样本数（默认: 300）
- `--num_alignment_samples`: 对齐样本数（默认: 100）
- `--random_seed`: 随机种子（默认: 42）

#### Step 2: 模型推理
```bash
# 对所有模型运行推理
python step2_run_inference.py --model all

# 或单个模型
python step2_run_inference.py --model qwen1.5B
```

对每个样本运行模型推理，保存隐藏状态、概率分布和生成文本。

**输出**:
- `experiment_results/model_outputs/{model}/sample_XXX.pt` - 每个样本的推理结果
  - `hidden_states_per_step`: 每步的隐藏状态 [num_layers, hidden_dim]
  - `probs_per_step`: 每步的概率分布（可选）
  - `top_k_info`: Top-K token 信息（节省空间）
  - `generated_text`: 完整生成文本
  - `generated_answer_only`: 纯答案文本（去除提示词）

**选项**:
- `--model`: 模型选择 (qwen1.5B/qwen3B/qwen7B/all)
- `--max_samples`: 限制样本数（用于测试）

**💡 提示**: Step 2 是最耗时的步骤（数小时），但只需运行一次。后续所有分析步骤都基于保存的 `.pt` 文件。

---

### 🔹 Phase 2: 错误分析

#### Step 3: 错误分析
```bash
python step3_error_analysis.py --num_samples 300
```

对比不同模型的预测结果，识别错误模式和 B 球困境案例。

**输出**:
- `experiment_results/analysis/error_analysis_{small}_vs_{large}.csv` - 详细错误分析
- `experiment_results/analysis/three_model_comparison.csv` - 三模型对比

**关键字段**:
- `has_error`: 小模型是否预测错误
- `is_b_ball_dilemma`: 是否为 B 球困境（类别内混淆）
- `error_type`: 错误类型分类
- `details`: 包含 Top-K 重叠度、熵等详细信息

**B 球困境判定标准**:
1. 小模型错误 且 大模型正确
2. Top-K 重叠度 > 阈值（默认 50%）
3. 熵 > 阈值（默认 2.5 nats）

---

### 🔹 Phase 3: 概率与相似度分析

#### Step 4: 概率探测分析（三模型）
```bash
python step4_probability_probing.py --max_samples 200
```

**核心功能**: 对错误样本进行逐层概率探测，追踪三个模型在每一层的概率分布演化。

**输出**:
- `experiment_results/probing/probing_results.json` - 完整探测数据
- `experiment_results/probing/probing_per_layer.csv` - 扁平化层级数据
- `experiment_results/probing/probing_*.png` - 5 种可视化图表

**可视化内容** （详见 VISUALIZATION_GUIDE.md）:
1. **聚合 P(GT) 曲线**: 三模型的平均 Ground Truth 概率演化
2. **个体样本演化**: 12 个代表性样本的详细轨迹
3. **分歧点分析**: 模型间开始产生不同预测的层级分布
4. **熵演化对比**: 模型自信度随层级的变化
5. **早期退出矩阵**: 量化级联推理的最佳切换时机

**应用场景**:
- 理解小模型"何时"和"为何"失去对正确答案的追踪
- 指导级联推理架构设计
- 优化早期退出策略

**选项**:
- `--models`: 要探测的模型列表（默认: 全部三个）
- `--max_samples`: 限制探测样本数
- `--small_model`, `--large_model`: 指定模型对

#### Step 4b: Logit 语义聚类分析（Phase I 因果证明）
```bash
python step4b_logit_cluster_analysis.py --num_samples 300 --device cuda
```

**因果假设**: 如果小模型和大模型的隐藏表示真正对齐，那么在语义空间中，它们的 logit 向量应该呈现相似的聚类结构。

**验证方法**: 
- 使用 UMAP 降维到 2D 空间
- 对两个模型的 logit 向量分别聚类（K-means）
- 计算聚类中心的余弦相似度
- 使用匈牙利算法进行最优匹配

**输出**:
- `experiment_results/analysis/logit_cluster_analysis.json` - 聚类结果
- `experiment_results/analysis/logit_cluster_umap_*.png` - UMAP 可视化
- `experiment_results/analysis/logit_cluster_similarity_heatmap_*.png` - 相似度热力图

**关键指标**:
- **聚类纯度**: 同一聚类内样本的一致性
- **跨模型匹配度**: 最优匹配后的平均余弦相似度（> 0.7 表示强对齐）

#### Step 5: CKA 相似度分析
```bash
python step5_cka_analysis.py --num_samples 200
```

计算不同模型层间的 Centered Kernel Alignment (CKA) 相似度。

**输出**:
- `experiment_results/analysis/cka_matrix_*.npy` - CKA 相似度矩阵
- `experiment_results/analysis/cka_matrix_*.png` - 热力图
- `experiment_results/analysis/cka_curves.png` - 对角线和最大值曲线

**解读**:
- 对角线高值：对应层表示相似
- 非对角线峰值：发现跨层对齐关系
- 用于指导投影矩阵训练的层选择

**选项**:
- `--num_samples`: 用于计算的样本数（默认: 200）
- `--step_idx`: 生成步骤索引（默认: 0，即首个生成 token）

#### Step 6: 决策性 Token 分析
```bash
python step6_decisive_token.py --model qwen7B --entropy_percentile 90
```

**核心目标**: 识别对模型最终预测有决策性影响的输入 token。

**方法**:
1. 按熵排序，筛选出高熵样本（模型不确定的情况）
2. 对每个 token 进行消融实验：移除该 token 后重新推理
3. 比较移除前后的预测差异，量化该 token 的重要性

**输出**:
- `experiment_results/analysis/decisive_tokens_{model}.json` - 完整分析结果
- `experiment_results/analysis/decisive_tokens_{model}_summary.png` - 可视化总结

**应用**:
- 理解哪些 token 对决策至关重要
- 指导 prompt 工程和样本构造
- 分析模型的注意力分配模式

**选项**:
- `--model`: 分析的模型（默认: qwen7B）
- `--entropy_percentile`: 熵百分位阈值（默认: 90）
- `--max_display`: 最多显示的样本数（默认: 12）

#### Step 6b: 特征余弦距离分析（Phase II 因果证明）
```bash
python step6b_feature_cosine_distance.py --num_samples 300 --device cuda
```

**因果假设**: 如果投影矩阵真正捕捉到了模型间的对齐关系，那么投影后的小模型特征应该与大模型的对应特征在余弦距离上显著接近。

**验证方法**:
- 加载已训练的投影矩阵 W
- 对每个样本: 特征距离 = cosine(W @ h_small, h_large)
- 与随机投影基线对比
- 分析距离分布的统计显著性

**输出**:
- `experiment_results/analysis/feature_cosine_distance_*.json` - 距离统计
- `experiment_results/analysis/feature_distance_distribution_*.png` - 分布对比图

**关键指标**:
- **平均余弦相似度**: > 0.8 表示强对齐
- **与随机基线的差距**: 应显著高于随机投影

---

### 🔹 Phase 4: 投影与注入

#### Step 7: 投影矩阵训练
```bash
# 自动尝试多层并选择最佳
python step7_train_projection.py --try_multiple_layers

# 或指定单层
python step7_train_projection.py --layer_idx -2
```

训练线性投影矩阵 `W: R^d_small -> R^d_large`，用于对齐小模型和大模型的隐藏空间。

**方法**: Ridge Regression (带 L2 正则化的最小二乘)

**输出**:
- `experiment_results/projection_matrices/W_up_{small}_to_{large}.pt` - 投影矩阵
- `experiment_results/projection_matrices/bias_{small}_to_{large}.pt` - 偏置向量
- `experiment_results/projection_matrices/layer_comparison_{small}_to_{large}.json` - 层级对比结果

**关键指标**:
- **测试集余弦相似度**: 越高表示投影越准确（通常 > 0.85）
- **最佳层选择**: 通常是倒数第 2-3 层

**选项**:
- `--try_multiple_layers`: 尝试多个层并选择最佳配置
- `--layer_idx`: 指定单层训练（-1 表示最后一层）
- `--num_samples`: 用于训练的样本数（默认: 100）
- `--alpha`: Ridge 正则化参数（默认: 0.01）

**💡 提示**: Step 7 会自动训练所有模型对的投影矩阵（1.5B→7B, 3B→7B, 1.5B→3B），支持 Step 6b 的因果验证。

#### Step 7b: 跨模型 LM Head 解码（Phase III 因果证明）
```bash
python step7b_cross_model_decode.py --small_model qwen1.5B --large_model qwen7B --device cuda
```

**因果假设**: 如果隐藏状态真正对齐，那么小模型的隐藏状态通过大模型的 LM head 解码后，应该产生与小模型自身解码相似的概率分布。

**验证方法**:
- 提取小模型的隐藏状态 h_small
- 通过大模型的 LM head 解码: logits_cross = LM_head_large(h_small)
- 计算与小模型原始输出的 KL 散度
- 与随机基线对比

**输出**:
- `experiment_results/analysis/cross_model_decode_*.json` - KL 散度统计
- `experiment_results/analysis/cross_decode_kl_divergence_*.png` - KL 分布图

**关键指标**:
- **平均 KL 散度**: < 1.0 表示分布相似
- **与自身解码的对比**: 交叉解码应接近自身解码的质量

#### Step 8: 状态注入实验
```bash
python step8_injection_experiment.py --max_samples 50 --small_model qwen1.5B --large_model qwen7B
```

在"B 球困境"样本上执行隐藏状态注入实验，验证投影矩阵的实用性。

**注入公式**:
```
h_injected = α * (W @ h_small + bias) + (1 - α) * h_large
```

**输出**:
- `experiment_results/analysis/injection_results_{small}_vs_{large}.csv` - 注入结果
- `experiment_results/analysis/injection_results_{small}_vs_{large}_probs.pt` - 概率数据（用于可视化）

**关键指标**:
- `error_corrected`: 注入后是否修正了错误
- `gt_rank_after_injection`: 注入后 GT token 的排名
- `correction_rate`: 总体修正率

**参数空间**:
- α 值: [0.1, 0.3, 0.5, 0.8]（控制注入强度）
- 注入层: 最后 1-4 层

**选项**:
- `--max_samples`: 限制实验样本数（默认: 50）
- `--small_model`, `--large_model`: 指定模型对

---

### 🔹 Phase 5: 可视化与总结

#### Step 9: 可视化
```bash
python step9_visualization.py --num_cases 10
```

生成概率分布对比图、注入效果图和实验总结面板。

**生成的图表**:
1. **B 球困境案例对比**: 小/大模型的 Top-30 概率分布并排展示
2. **注入效果对比**: 注入前/后的概率分布变化
3. **实验总结面板**: 4 宫格展示错误类型、注入效果、熵分布

**输出目录**:
- `experiment_results/analysis/prob_dist_plots/` - 所有可视化图表
- `experiment_results/analysis/experiment_summary.png` - 总结面板

**选项**:
- `--num_cases`: 可视化的案例数（默认: 10）
- `--skip_individual`: 跳过单个案例图表，仅生成汇总图
- `--small_model`, `--large_model`: 指定模型对

#### Step 10: 总结报告
```bash
python step10_summary.py
```

汇总所有实验结果，生成综合 JSON 报告并打印到控制台。

**输出**:
- `experiment_results/summary_report.json` - 完整 JSON 报告

**报告内容**:
- 实验配置信息
- 错误分析统计（B 球困境占比）
- 三模型对比结果
- CKA 相似度摘要
- 投影矩阵训练结果
- 注入实验效果（最佳配置、修正率）
- 概率探测摘要（早期退出分析）

---

## 📁 项目结构

```
qwen-lab/
├── config.py                          # 全局配置文件
├── utils.py                           # 工具函数（加载、保存、GPU 管理）
├── qa_utils.py                        # QA 任务专用工具（答案提取、对比）
├── requirements.txt                   # Python 依赖
├── README.md                          # 本文档
├── run_experiment.py                  # 主运行脚本（编排所有步骤）
│
├── step1_prepare_data.py             # 步骤 1: 数据准备
├── step2_run_inference.py            # 步骤 2: 模型推理
├── step3_error_analysis.py           # 步骤 3: 错误分析
├── step4_probability_probing.py      # 步骤 4: 概率探测
├── step4b_logit_cluster_analysis.py  # 步骤 4b: Logit 聚类（因果证明 I）
├── step5_cka_analysis.py             # 步骤 5: CKA 分析
├── step6_decisive_token.py           # 步骤 6: 决策性 Token
├── step6b_feature_cosine_distance.py # 步骤 6b: 特征距离（因果证明 II）
├── step7_train_projection.py         # 步骤 7: 投影矩阵训练
├── step7b_cross_model_decode.py      # 步骤 7b: 交叉解码（因果证明 III）
├── step8_injection_experiment.py     # 步骤 8: 注入实验
├── step9_visualization.py            # 步骤 9: 可视化
├── step10_summary.py                 # 步骤 10: 总结报告
│
├── docs/
│   └── VISUALIZATION_GUIDE.md        # 📊 可视化结果详细指南
│
├── experiment_results/               # 实验结果目录
│   ├── sampled_data/                # 采样数据
│   │   ├── sampled_300.json
│   │   └── alignment_100.json
│   │
│   ├── model_outputs/               # 模型推理输出
│   │   ├── qwen1.5B/
│   │   ├── qwen3B/
│   │   └── qwen7B/
│   │
│   ├── analysis/                    # 分析结果
│   │   ├── error_analysis_*.csv
│   │   ├── three_model_comparison.csv
│   │   ├── cka_matrix_*.npy
│   │   ├── cka_curves.png
│   │   ├── logit_cluster_*.json
│   │   ├── feature_cosine_distance_*.json
│   │   ├── cross_model_decode_*.json
│   │   ├── decisive_tokens_*.json
│   │   ├── injection_results_*.csv
│   │   ├── experiment_summary.png
│   │   └── prob_dist_plots/         # 概率分布图
│   │
│   ├── probing/                     # 概率探测结果
│   │   ├── probing_results.json
│   │   ├── probing_per_layer.csv
│   │   ├── probing_aggregate.png
│   │   ├── probing_individual.png
│   │   ├── probing_divergence.png
│   │   ├── probing_entropy.png
│   │   └── probing_early_exit.png
│   │
│   ├── projection_matrices/         # 投影矩阵
│   │   ├── W_up_*.pt
│   │   ├── bias_*.pt
│   │   └── layer_comparison_*.json
│   │
│   └── summary_report.json          # 综合总结报告
│
└── models/                           # 模型缓存（自动下载）
```

## 📊 可视化结果

实验会自动生成 **10+ 种可视化图表**，全面展示实验结果：

### 概率探测可视化（Step 4）
1. **聚合 P(GT) 曲线** - 三模型的平均演化趋势
2. **个体样本演化** - 12 个代表性案例的详细轨迹
3. **分歧点分析** - 模型间开始产生不同预测的层级
4. **熵演化对比** - 模型自信度的层级变化
5. **早期退出矩阵** - 级联推理的最佳切换时机

### CKA 相似度可视化（Step 5）
6. **CKA 热力图（3 个）** - 不同模型对的层级相似度矩阵
7. **CKA 曲线图** - 对角线和最大值趋势分析

### 因果证明可视化（Steps 4b, 6b, 7b）
8. **Logit UMAP 聚类图** - 语义空间中的聚类结构
9. **聚类相似度热力图** - 跨模型聚类匹配度
10. **特征距离分布图** - 投影后的余弦相似度分布
11. **交叉解码 KL 散度图** - 跨模型解码质量对比

### 注入效果可视化（Steps 8-9）
12. **概率分布对比图** - B 球困境案例的小/大模型对比
13. **注入前后对比图** - 状态注入的效果展示
14. **实验总结面板** - 4 宫格综合展示

**详细说明请参考**：[docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md) 📈

## 🎛️ 配置说明

所有配置参数在 `config.py` 中定义：

### 模型配置
```python
MODELS = {
    "qwen1.5B": {
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "hidden_dim": 1536,
        "num_layers": 28,
    },
    # ...
}
```

### 数据配置
```python
DATA_CONFIG = {
    "datasets": {
        "gsm8k": {...},
        "arc_challenge": {...},
        "mmlu": {...},
    },
    "total_samples": 300,
    "num_alignment_samples": 100,
}
```

### 分析配置
```python
ANALYSIS_CONFIG = {
    "top_k_overlap": 20,           # B 球困境的 Top-K
    "entropy_threshold": 2.5,      # 熵阈值（nats）
}
```

### 注入配置
```python
INJECTION_CONFIG = {
    "alpha_values": [0.1, 0.3, 0.5, 0.8],  # 混合系数
    "injection_layers": [1, 2, 3, 4],       # 最后 N 层
}
```

### 探测配置
```python
PROBING_CONFIG = {
    "divergence_threshold": 0.1,   # 分歧检测阈值
    "top_k_probe": 10,             # 每层记录的 Top-K
    "early_exit": {
        "gt_confidence_threshold": 0.3,  # 高置信度阈值
        "switch_confidence_gap": 0.15,   # 切换所需的最小差距
    },
}
```

## 📈 核心指标

| 指标类别 | 具体指标 | 说明 |
|---------|---------|------|
| **错误分析** | B 球困境占比 | 类别内混淆错误占总错误的比例 |
|  | 错误率 | 小模型在测试集上的错误率 |
| **概率探测** | P(GT) 演化曲线 | Ground Truth token 概率随层级的变化 |
|  | 分歧点分布 | 模型间开始产生不同预测的层级 |
|  | 早期退出阈值 | 最优级联推理的切换时机 |
| **相似度** | CKA 分数 | 层级间的表示相似度（0-1） |
|  | 对角线 CKA | 对应层的平均相似度 |
|  | 最佳层对 | 最高 CKA 值对应的层索引 |
| **因果验证** | 聚类匹配度 | 跨模型聚类中心的余弦相似度 |
|  | 投影余弦距离 | 投影后特征与目标特征的相似度 |
|  | 交叉解码 KL 散度 | 跨模型解码的分布相似度 |
| **注入效果** | GT Rank 提升 | 注入后正确答案排名的变化 |
|  | 错误修正率 | 成功修正错误的样本比例 |
|  | 最佳 α 值 | 最优混合系数 |

## 💾 数据格式说明

### 模型输出文件 (.pt)
```python
{
    "sample_idx": int,
    "input_text": str,
    "generated_text": str,
    "generated_answer_only": str,  # 纯答案文本
    "hidden_states_per_step": List[torch.Tensor],  # [num_layers, hidden_dim]
    "probs_per_step": List[torch.Tensor],  # [vocab_size] (可选)
    "top_k_info": List[Dict],  # Top-K token 信息
    "ground_truth": Dict,
    "generated_ids": List[int],
}
```

### 错误分析 CSV
```csv
sample_idx,has_error,is_b_ball_dilemma,error_type,predicted_answer,ground_truth_answer,details
0,True,True,category_confusion,B,A,"{\"top_k_overlap\": 0.75
