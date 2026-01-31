# 评估指标统一标准说明

## 更新日期
2025-10-27

## 多级相关性权重设置

### MedCorpus 数据集特性
- **相关性等级**: 0/1/2 (三级)
- **0**: 不相关
- **1**: 部分相关
- **2**: 高度相关

### 统一评估指标权重

#### 1. NDCG@10 (Normalized Discounted Cumulative Gain)
- **公式**: 标准学术公式 `DCG = Σ[(2^rel - 1) / log2(i+1)]`
- **权重映射**:
  - `rel=2` → 权重 `3` (2²-1=3)
  - `rel=1` → 权重 `1` (2¹-1=1)
  - `rel=0` → 权重 `0` (2⁰-1=0)

#### 2. MAP@10 (Mean Average Precision)
- **权重设置**（统一标准）:
  - `rel=2` → 权重 `1.0`
  - `rel=1` → 权重 `0.5`
  - `rel=0` → 权重 `0.0`

- **计算公式**:
  ```
  AP = (1/R) × Σ[P(i) × weight(rel_i)] for all relevant docs
  where:
    - P(i) = precision at position i
    - weight(rel=2) = 1.0
    - weight(rel=1) = 0.5
    - R = total number of relevant documents
  ```

#### 3. P@1 (Precision at 1)
- **二元处理**: `rel > 0` 算相关
- 只关心首位文档是否相关

## 已统一的评估代码文件

### Core 评估模块
1. **`/workspace/PerMed/core/evaluate.py`**
   - 通用评估脚本
   - 支持 MedCorpus 和 LitSearch
   - ✅ MAP 权重已统一

2. **`/workspace/PerMed/core/evaluate_jina_v3.py`**
   - Jina-v3 重排专用评估
   - ✅ MAP 权重已统一

3. **`/workspace/PerMed/core/evaluate_compare_retrieval_rerank.py`**
   - 检索前 vs 重排后对比评估
   - ✅ MAP 权重已统一

### Baselines 评估模块
4. **`/workspace/PerMed/baselines/evaluate.py`**
   - 基线方法评估
   - ✅ MAP 权重已统一（新增多级相关性支持）
   - 兼容二元相关性（0/1）和多级相关性（0/1/2）

5. **`/workspace/PerMed/baselines/evaluate_with_rel_labels.py`**
   - MedCorpus 专用评估（带相关性标签）
   - ✅ MAP 权重已统一

### Evaluation 评估模块
6. **`/workspace/PerMed/evaluation/evaluate.py`**
   - 用于 LitSearch 等二元相关性数据集
   - 仅处理二元相关性，无需修改

7. **`/workspace/PerMed/evaluation/evaluate_memory_ablation_v2.py`**
   - 用于记忆消融实验
   - 仅处理二元相关性，无需修改

## 代码实现示例

### MAP@k 计算（统一实现）

```python
def average_precision(gt_rels, pred_docs, k):
    """
    计算 Average Precision@k (考虑多级相关性 0/1/2)
    
    多级相关性权重设置（统一标准）:
    - rel=2 权重: 1.0
    - rel=1 权重: 0.5
    - rel=0 权重: 0.0
    """
    score = 0.0
    num_relevant = 0
    
    for i, doc in enumerate(pred_docs[:k], 1):
        rel = gt_rels.get(doc, 0)
        if rel > 0:  # 相关文档
            num_relevant += 1
            precision_at_i = num_relevant / i
            # 按相关性等级加权: rel=2权重1.0, rel=1权重0.5
            score += precision_at_i * (rel / 2.0)
    
    # 归一化：除以总相关文档数
    total_relevant = sum(1 for r in gt_rels.values() if r > 0)
    return score / total_relevant if total_relevant > 0 else 0.0
```

## 评估结果示例

基于统一标准的 MedCorpus 评估结果：

| 方法 | NDCG@10 | MAP@10 | P@1 |
|------|---------|---------|-----|
| 检索前 (Retrieved) | 0.9044 | 0.5357 | 0.9279 |
| 重排-Profile+Query | 0.9070 | 0.5435 | 0.9203 |
| 重排-Query-only | 0.8272 | 0.5110 | 0.8154 |

### MedCorpus Memory Ablation（Profile-only, Qwen3-32B 画像, Jina reranker, 3440 queries）

| 变体 | 描述 | NDCG@10 | MAP@10 | P@1 |
|------|------|---------|--------|-----|
| no_long | 去掉 `[LONG_EXPLICIT]`（保留顺序+工作） | 0.9022 | 0.5380 | 0.9235 |
| no_sequential | 去掉顺序记忆（保留长期+工作） | 0.9129 | 0.5439 | 0.9384 |
| no_working | 去掉工作记忆（保留长期+顺序） | 0.9045 | 0.5396 | 0.9302 |

## 设计理念

### 为什么 MAP 使用线性权重（1.0, 0.5）？
1. **与 NDCG 互补**:
   - NDCG 对 rel=2 的权重是指数级（3倍于rel=1）
   - MAP 使用线性权重（2倍于rel=1）
   - 两者从不同角度评估排序质量

2. **符合实际需求**:
   - rel=2 文档确实应该获得更高权重
   - 线性权重更容易理解和解释
   - 2倍权重差异对排序优化有明确指导意义

3. **评估完整性**:
   - NDCG 强调位置和相关性的双重影响
   - MAP 强调精确率在不同位置的表现
   - P@1 强调首位结果的质量

## 参考文献

**NDCG 标准公式**:
- Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. ACM TOIS, 20(4), 422-446.

**MAP 多级相关性处理**:
- 基于标准 AP 公式扩展，考虑相关性等级加权

## 修改历史

- **2025-10-27**: 统一所有评估代码的 MAP 权重标准
  - 统一设置: rel=2权重1.0, rel=1权重0.5
  - 更新文件: core/evaluate.py, core/evaluate_jina_v3.py, core/evaluate_compare_retrieval_rerank.py
  - 更新文件: baselines/evaluate.py, baselines/evaluate_with_rel_labels.py
  - 添加详细注释说明权重标准


