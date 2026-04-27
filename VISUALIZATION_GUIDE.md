# 可视化结果指南

本实验框架会自动生成多种可视化图表，帮助理解和分析实验结果。

## 📊 生成的可视化图表

### 1. CKA相似度热力图 (CKA Similarity Heatmaps)

**位置**: `experiment_results/analysis/cka_matrix_*.png`

生成3个热力图，展示不同模型对之间的层级相似度：

- `cka_matrix_1.5Bvs7B.png` - 1.5B模型 vs 7B模型
- `cka_matrix_7Bvs14B.png` - 7B模型 vs 14B模型  
- `cka_matrix_1.5Bvs14B.png` - 1.5B模型 vs 14B模型

**解读**：
- 颜色越红，CKA相似度越高（0-1范围）
- 对角线表示对应层之间的相似度
- 可识别出哪些层在不同模型间表征最相似

**示例说明**：
```
如果看到1.5B的第10层与7B的第20层CKA值很高（如0.8+），
说明这两层学习到了类似的特征表示，可以作为投影注入的候选层。
```

### 2. CKA相似度曲线 (CKA Similarity Curves)

**位置**: `experiment_results/analysis/cka_curves.png`

包含2个子图：
- **左图**: 对角线CKA相似度（对应层相似度）
- **右图**: 每层的最大CKA相似度

**解读**：
- 显示模型间层级对齐的变化趋势
- 峰值位置表示最相似的层对

### 3. 概率分布对比图 (Probability Distribution Comparison)

**位置**: `experiment_results/analysis/prob_dist_plots/case_XXX_comparison.png`

对于B球困境案例，并排显示：
- **左侧**: 小模型(1.5B)的top-30 token概率分布
- **右侧**: 大模型(7B)的top-30 token概率分布

**特点**：
- 水平柱状图，按概率降序排列
- 显示token文本和对应概率值
- 可清晰看到两个模型的预测差异

**解读**：
```
示例：
- 如果小模型在多个相似token间概率平均分配（熵高）
- 而大模型集中在正确token上
- 则说明小模型存在"B球困境"（难以区分相似选项）
```

### 4. 实验综合总结图 (Experiment Summary)

**位置**: `experiment_results/analysis/experiment_summary.png`

4宫格布局，包含：

#### 子图1：错误类型分布 (左上)
- 柱状图显示不同错误类型的数量
- 包含B球困境占比标注
- 帮助理解小模型主要错在哪里

#### 子图2：注入效果 vs Alpha (右上)
- 折线图显示不同混合系数α的效果
- Y轴：Ground Truth token的平均排名
- 较低排名=更好的注入效果

#### 子图3：注入效果 vs 注入层 (左下)
- 柱状图对比不同注入层的效果
- 显示哪一层最适合进行状态注入

#### 子图4：B球困境 vs 其他错误 (右下)
- 直方图对比B球困境和其他错误的数量
- 量化"类别内混淆"现象的普遍性

### 5. 注入前后对比图 (Injection Effect)

**位置**: `experiment_results/analysis/prob_dist_plots/injection_*.png`

显示注入隐藏状态前后的概率分布变化：
- **左侧**: 注入前7B模型的原始分布
- **右侧**: 注入后7B模型的修改分布
- **红色高亮**: Ground truth token位置

**解读**：
```
成功的注入应该显示：
1. GT token排名上升（移向前列）
2. 概率分布发生明显变化
3. 错误token概率下降
```

## 🎨 可视化示例场景

### 场景1：发现高度对齐的层
```
查看 cka_matrix_1.5Bvs7B.png
→ 发现第15层(1.5B) ↔ 第28层(7B) CKA=0.85
→ 选择这对层进行投影训练
→ 期望获得更好的状态迁移效果
```

### 场景2：识别B球困境
```
查看 case_042_comparison.png
→ 左图(1.5B)：get_weather(0.15), get_temperature(0.14), get_climate(0.12)
→ 右图(7B)：get_weather(0.68), search_web(0.12), ...
→ 结论：小模型难以区分天气相关函数（类别内混淆）
```

### 场景3：优化注入参数
```
查看 experiment_summary.png 的右上图
→ α=0.3时GT排名最低（约15名）
→ α=0.8时GT排名较高（约30名）
→ 结论：使用较小的α值效果更好
```

## 📈 生成可视化的命令

```bash
# 生成所有可视化
python step7_visualization.py

# 只生成汇总图（跳过单个案例）
python step7_visualization.py --skip_individual

# 指定可视化案例数量
python step7_visualization.py --num_cases 20
```

## 🔍 深度分析建议

1. **CKA分析**：
   - 观察对角线趋势，了解层级对应关系
   - 寻找非对角线高值，发现跨层相似性
   - 对比不同模型对，验证缩放一致性

2. **概率分布分析**：
   - 计算熵值，量化不确定性
   - 比较top-k重叠度
   - 识别系统性偏差模式

3. **注入效果分析**：
   - 观察α-性能曲线的单调性
   - 识别最佳注入层的规律
   - 分析失败案例的共同特征

## 💡 进阶可视化

如需自定义可视化，可修改 `step7_visualization.py` 或使用以下代码片段：

```python
# 加载数据
import torch
import matplotlib.pyplot as plt

# 加载模型输出
output = torch.load("experiment_results/model_outputs/qwen1.5B/sample_000.pt")
probs = output["probs_per_step"][0]  # 第一步的概率

# 自定义可视化
top_k = 50
values, indices = torch.topk(probs, k=top_k)

plt.figure(figsize=(12, 8))
plt.barh(range(top_k), values.numpy())
plt.xlabel("Probability")
plt.ylabel("Token Rank")
plt.title("Custom Top-50 Probability Distribution")
plt.tight_layout()
plt.savefig("custom_plot.png", dpi=300)
```

## 📊 数据导出

所有可视化的底层数据均可导出：

```python
# CKA矩阵
import numpy as np
cka_matrix = np.load("experiment_results/analysis/cka_matrix_1.5Bvs7B.npy")
print(f"Shape: {cka_matrix.shape}")
print(f"Mean CKA: {cka_matrix.mean():.4f}")

# 错误分析
import pandas as pd
df = pd.read_csv("experiment_results/analysis/error_analysis.csv")
print(df.describe())

# 注入结果
df_inj = pd.read_csv("experiment_results/analysis/injection_results.csv")
best_configs = df_inj.groupby(['alpha', 'injection_layer'])['gt_rank_after_injection'].mean()
print(best_configs.sort_values().head())
```

## 🎯 结果解读检查清单

- [ ] CKA热力图显示合理的层级对应关系
- [ ] 概率分布图清晰展示模型差异
- [ ] B球困境案例占比在合理范围（通常10-30%）
- [ ] 注入实验显示GT排名有提升
- [ ] 最佳α值在0.1-0.5之间（通常）
- [ ] 可视化文件全部正常生成

---

**注意**：所有可视化图表会在运行 `step7_visualization.py` 或完整流程 `run_experiment.py --all` 后自动生成。
