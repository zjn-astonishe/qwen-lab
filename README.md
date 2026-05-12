# 异构模型隐藏状态对齐与概率分布演化实验

本项目实现了一个完整的实验框架，用于研究不同规模语言模型（Qwen 1.5B/7B/14B）在Agent任务上的隐藏状态对齐与概率分布演化。

## 实验目标

1. 比较不同规模模型的输出层概率分布粒度变化
2. 分析隐藏状态在关键决策点的相似性与可迁移性
3. 识别小模型错误中的"B球困境"现象（类别内混淆）
4. 验证通过简单投影实现跨模型隐藏状态注入的可行性

## 系统要求

- Python 3.10+
- PyTorch 2.0+
- CUDA 支持的 GPU（推荐至少 1 张 A100 80GB）
- 至少 200GB 磁盘空间（用于模型和数据）

## 安装

1. 克隆仓库并进入目录：
```bash
git clone <repository-url>
cd exp
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 快速开始

### 方式一：使用主运行脚本

```bash
# 运行完整实验流程
python run_experiment.py --all

# 或者运行特定步骤
python run_experiment.py --steps 1 2 3
```

### 方式二：逐步运行

#### 步骤 1: 数据准备
```bash
python step1_prepare_data.py --use_synthetic
```

选项：
- `--use_synthetic`: 使用合成数据进行测试
- `--num_samples`: 测试样本数量（默认：200）
- `--num_alignment`: 对齐样本数量（默认：100）

#### 步骤 2: 模型推理
```bash
# 对所有模型运行推理
python step2_run_inference.py --model all

# 或对单个模型运行
python step2_run_inference.py --model qwen1.5B
```

选项：
- `--model`: 选择模型 (qwen1.5B/qwen7B/qwen3B/all)
- `--max_samples`: 限制处理的样本数量（用于测试）

#### 步骤 3: 错误分析
```bash
python step3_error_analysis.py
```

分析小模型的错误类型，识别"B球困境"案例。

#### 步骤 4: 概率探测分析
```bash
python step4_probability_probing.py --num_samples 200
```

对错误样本进行逐层概率探测，分析三个模型（1.5B/3B/7B）在每一层的概率分布演化。

选项：
- `--small_model`: 小模型选择（默认：qwen1.5B）
- `--large_model`: 大模型选择（默认：qwen7B）
- `--models`: 要探测的模型列表（默认：所有三个模型）
- `--max_samples`: 限制探测样本数量

#### 步骤 5: CKA 相似度分析
```bash
python step5_cka_analysis.py --num_samples 200
```

计算不同模型层间的 Centered Kernel Alignment (CKA) 相似度。

选项：
- `--num_samples`: 用于计算的样本数量
- `--step_idx`: 生成步骤索引（默认：0）

#### 步骤 6: 训练投影矩阵
```bash
# 尝试多层并选择最佳
python step6_train_projection.py --try_multiple_layers

# 或指定单层训练
python step6_train_projection.py --layer_idx -2
```

选项：
- `--try_multiple_layers`: 尝试多个层并选择最佳配置
- `--num_samples`: 用于训练的样本数量
- `--alpha`: Ridge 正则化参数

#### 步骤 7: 状态注入实验
```bash
python step7_injection_experiment.py --max_samples 50
```

在"B球困境"样本上执行隐藏状态注入实验。

选项：
- `--max_samples`: 限制实验样本数量
- `--small_model`: 小模型选择
- `--large_model`: 大模型选择

#### 步骤 8: 可视化
```bash
python step8_visualization.py --num_cases 10
```

生成概率分布对比图和注入效果可视化。

选项：
- `--num_cases`: 可视化案例数量
- `--skip_individual`: 跳过单个案例图表
- `--small_model`: 小模型选择
- `--large_model`: 大模型选择

#### 步骤 9: 生成总结报告
```bash
python step9_summary.py
```

汇总所有实验结果并生成综合报告。

## 项目结构

```
exp/
├── config.py                          # 配置文件
├── requirements.txt                   # Python依赖
├── README.md                          # 项目文档
├── run_experiment.py                  # 主运行脚本
│
├── step1_prepare_data.py             # 步骤1: 数据准备
├── step2_run_inference.py            # 步骤2: 模型推理
├── step3_error_analysis.py           # 步骤3: 错误分析
├── step4_probability_probing.py      # 步骤4: 概率探测分析
├── step5_cka_analysis.py             # 步骤5: CKA分析
├── step6_train_projection.py         # 步骤6: 训练投影矩阵
├── step7_injection_experiment.py     # 步骤7: 注入实验
├── step8_visualization.py            # 步骤8: 可视化
├── step9_summary.py                  # 步骤9: 总结报告
│
├── data/                             # 数据目录
│   └── bfcl_v3/                     # BFCL V3 数据集
│
├── models/                           # 模型缓存目录
│
└── experiment_results/               # 实验结果
    ├── sampled_data/                # 采样数据
    │   ├── sampled_200.json
    │   └── alignment_100.json
    │
    ├── model_outputs/               # 模型输出
    │   ├── qwen1.5B/
    │   ├── qwen3B/
    │   └── qwen7B/
    │
    ├── analysis/                    # 分析结果
    │   ├── error_analysis.csv
    │   ├── cka_matrix_*.npy
    │   ├── injection_results.csv
    │   └── prob_dist_plots/
    │
    ├── probing/                     # 概率探测结果
    │   ├── probing_results.json
    │   ├── probing_per_layer.csv
    │   └── probing_*.png
    │
    ├── projection_matrices/         # 投影矩阵
    │   ├── W_up_1.5B_to_7B.pt
    │   └── bias_1.5B_to_7B.pt
    │
    └── summary_report.json          # 总结报告
```

## 核心指标

实验将计算以下核心指标：

1. **概率分布熵**: 各模型在关键token处的top-k熵
2. **错误类型比例**: 大类错误 vs 小类错误（B球困境）
3. **CKA 相似度**: 跨模型各层的CKA值
4. **注入有效性**:
   - 正确token排名平均提升位数
   - 任务准确率变化
5. **最佳配置**: 最优混合系数α和注入层位置

## 配置说明

所有配置参数在 `config.py` 中定义：

- **MODELS**: 模型配置（名称、隐藏维度、输出目录）
- **DATA_CONFIG**: 数据配置（数据集、样本数量）
- **GENERATION_CONFIG**: 生成配置（最大token数、温度）
- **ANALYSIS_CONFIG**: 分析配置（top-k值、熵阈值）
- **INJECTION_CONFIG**: 注入实验配置（α值、注入层）
- **HARDWARE_CONFIG**: 硬件配置（设备、精度、显存）

## 实验结果

实验完成后，查看以下文件获取结果：

1. **experiment_results/summary_report.json**: 综合总结报告
2. **experiment_results/analysis/error_analysis.csv**: 详细错误分析
3. **experiment_results/analysis/injection_results.csv**: 注入实验结果
4. **experiment_results/analysis/*.png**: 可视化图表

### 📊 可视化结果

实验会自动生成以下可视化图表：

1. **CKA相似度热力图** - 展示不同模型层间的相似度矩阵
2. **CKA相似度曲线** - 层级对齐趋势分析
3. **概率分布对比图** - 小/大模型的token概率分布对比
4. **实验综合总结图** - 4宫格展示错误类型、注入效果等
5. **注入前后对比图** - 状态注入对概率分布的影响

详细说明请参考：**[docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md)** 📈

## 常见问题

### 1. 内存不足
如果遇到GPU内存不足，可以：
- 在 `config.py` 中调整 `HARDWARE_CONFIG["max_memory"]`
- 减少 `GENERATION_CONFIG["max_new_tokens"]`
- 使用 `--max_samples` 参数限制处理的样本数量

### 2. 数据集获取
BFCL V3数据集可以从以下来源获取：
- GitHub: https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- HuggingFace: `gorilla-llm/Berkeley-Function-Calling-Leaderboard`

或使用 `--use_synthetic` 标志生成合成数据进行测试。

### 3. 模型下载
首次运行时，模型会自动从HuggingFace下载到 `./models` 目录。确保有足够的磁盘空间和良好的网络连接。

## 引用

如果您使用此代码进行研究，请引用相关工作。

## 许可证

[指定许可证]

## 联系方式

如有问题或建议，请联系 [联系邮箱]
