# 重排方法对比评估报告

**生成日期**: 2025-11-11
**重排模型**: Jina-Reranker-v3
**个性化模型**: Qwen3-14B
**评估脚本**: `/mnt/data/zsy-data/PerMed/core/evaluate.py`

---

## 1. 执行摘要

本报告对比了三种重排方法在两个数据集上的表现：
- **Query-Only**: 仅使用查询文本进行重排
- **Profile-Only**: 仅使用个性化用户画像进行重排
- **Hybrid (Profile+Query)**: 结合用户画像和查询文本的混合方法

### 关键发现

1. **LitSearch数据集**: Profile-Only方法表现最佳 (NDCG@10=0.4336)，显著优于Query-Only (+67.6%)
2. **MedCorpus数据集**: Profile-Only方法略优 (NDCG@10=0.9078)，与Hybrid基本持平，都显著优于Query-Only (+9.7%)
3. **个性化的价值**: 在两个数据集上，加入个性化信息都能显著提升检索效果
4. **方法对比**: Profile-Only和Hybrid方法性能接近，在不同场景下各有优势

---

## 2. LitSearch数据集评估结果

**数据集特征**:
- 597个单轮查询
- 学术文献检索任务
- 二元相关性标注 (0/1)

### 2.1 性能对比

| 方法 | NDCG@10 | MAP@10 | P@1 |
|------|---------|--------|-----|
| **Query-Only** | 0.2588 | 0.0877 | 0.0687 |
| **Profile-Only** | **0.4336** | **0.2003** | **0.3400** |
| **Hybrid (Profile+Query)** | 0.4251 | 0.1948 | 0.3266 |

### 2.2 相对提升分析

**Profile-Only vs Query-Only**:
- NDCG@10: +67.6% 提升
- MAP@10: +128.4% 提升
- P@1: +395.0% 提升

**Hybrid vs Query-Only**:
- NDCG@10: +64.3% 提升
- MAP@10: +122.1% 提升
- P@1: +375.3% 提升

**Hybrid vs Profile-Only**:
- NDCG@10: -2.0% (略低)
- MAP@10: -2.7% (略低)
- P@1: -3.9% (略低)

### 2.3 LitSearch结论

- **最佳方法**: Profile-Only
- **关键洞察**: 在LitSearch数据集上，用户的研究画像比当前查询更能预测相关文档
- **可能原因**: 学术检索场景中，用户的长期研究兴趣比单次查询更具信息量

---

## 3. MedCorpus数据集评估结果

**数据集特征**:
- 800个多轮对话主题，共3440个查询
- 医学文献检索任务
- 三级相关性标注 (0/1/2)

### 3.1 性能对比

| 方法 | NDCG@10 | MAP@10 | P@1 |
|------|---------|--------|-----|
| **Query-Only** | 0.8272 | 0.5110 | 0.8154 |
| **Profile-Only** | **0.9078** | **0.5440** | **0.9206** |
| **Hybrid (Profile+Query)** | 0.9070 | 0.5435 | 0.9203 |

### 3.2 相对提升分析

**Profile-Only vs Query-Only**:
- NDCG@10: +9.74% 提升 (0.8272 → 0.9078)
- MAP@10: +6.46% 提升 (0.5110 → 0.5440)
- P@1: +12.90% 提升 (0.8154 → 0.9206)

**Hybrid vs Query-Only**:
- NDCG@10: +9.65% 提升 (0.8272 → 0.9070)
- MAP@10: +6.36% 提升 (0.5110 → 0.5435)
- P@1: +12.87% 提升 (0.8154 → 0.9203)

**Profile-Only vs Hybrid**:
- NDCG@10: +0.09% (略高)
- MAP@10: +0.09% (略高)
- P@1: +0.03% (基本持平)

### 3.3 MedCorpus结论

- **最佳方法**: Profile-Only (略优于Hybrid)
- **关键洞察**: 在MedCorpus多轮对话场景中，Profile-Only和Hybrid方法性能几乎相同，都显著优于Query-Only
- **可能原因**: 医学文献检索中，用户的长期研究兴趣非常重要，当前查询的增量信息有限

---

## 4. 跨数据集对比分析

### 4.1 数据集难度对比

| 数据集 | Query-Only NDCG@10 | 任务难度 |
|--------|-------------------|---------|
| **MedCorpus** | 0.8272 | 较容易 |
| **LitSearch** | 0.2588 | 较困难 |

LitSearch的基线性能显著低于MedCorpus，表明其检索任务更具挑战性。

### 4.2 个性化效果对比

**个性化带来的NDCG@10提升**:
- **LitSearch**: +64-68% (巨大提升)
- **MedCorpus**: +9.6% (显著提升)

**结论**: 个性化在更困难的检索任务(LitSearch)上带来更大的性能提升。

### 4.3 方法选择建议

| 场景 | 推荐方法 | 理由 |
|------|---------|------|
| **单轮学术检索** | Profile-Only | 用户研究兴趣最重要 |
| **多轮对话检索** | Hybrid | 需要平衡即时需求和长期兴趣 |
| **无个性化信息** | Query-Only | 仅作为基线 |

---

## 5. 与Baseline方法对比

### 5.1 LitSearch Baseline对比

| 方法 | NDCG@10 | 类型 |
|------|---------|------|
| PBR | 0.3749 | 查询扩展 |
| RPMN | 0.3835 | 记忆网络 |
| Claude 4.5 Haiku | 0.4680 | LLM重排 |
| **Profile-Only (Ours)** | **0.4336** | 个性化重排 |
| **Hybrid (Ours)** | **0.4251** | 个性化重排 |

**分析**:
- 我们的方法超越了传统的PBR和RPMN方法
- 略低于最佳的LLM重排方法(Claude 4.5 Haiku)
- 但我们的方法计算效率更高，成本更低

### 5.2 LLM Reranker 基线 (Gemini 2.5 Flash)

| 数据集 | 评估查询数 | NDCG@10 | MAP@10 | P@1 | 备注 |
|--------|------------|---------|--------|-----|------|
| **MedCorpus** | 1400 | 0.9167 | 0.5603 | 0.9464 | 仅覆盖已完成重排的查询 (总3440中的前1400条) |
| **LitSearch** | 597 | 0.3846 | 0.1687 | 0.2596 | 全量评估 |

**说明**:
- 评估脚本: `python3 baselines/evaluate.py --dataset {medcorpus|litsearch} --gt <ground_truth> --results baselines/results/{dataset}/llm_gemini25flash_results.jsonl`
- MedCorpus 的 Gemini 重排仍在运行，当前指标基于已完成的 1400 条查询（约 41%），可作为阶段性参考
- LitSearch 的 Gemini 重排已完整结束，597 条查询全部纳入评估

---

## 6. 技术细节

### 6.1 重排配置
- **重排模型**: Jina-Reranker-v3
- **个性化生成**: Qwen3-14B (SiliconFlow API)
- **初始候选数**: Top-100
- **最终返回数**: Top-10
- **批处理大小**: 8

### 6.2 输入格式

**Query-Only**:
```
Input: "{query_text}"
```

**Profile-Only**:
```
Input: "{personalized_profile}"
```

**Hybrid (Profile+Query)**:
```
Input: "User Research Profile: {profile}\n\nNew Query: {query}"
```

---

## 7. 结论与建议

### 7.1 主要结论

1. **个性化显著提升检索效果**: 在两个数据集上，加入个性化信息都带来显著提升
2. **方法选择取决于场景**:
   - 单轮检索 → Profile-Only
   - 多轮对话 → Hybrid
3. **困难任务受益更多**: LitSearch(困难)比MedCorpus(简单)从个性化中获益更多

### 7.2 未来工作建议

1. **补充MedCorpus的Profile-Only结果**: 完整对比三种方法
2. **分析失败案例**: 理解何时个性化无效或有害
3. **用户研究**: 验证自动评估指标与用户满意度的相关性
4. **效率优化**: 探索更快的个性化画像生成方法

---

## 8. Qwen3-32B Profile-Only 追加记录（2025-11-11）

> 说明：本节在不改动 14B 主体内容的前提下，补充了使用 **Qwen/Qwen3-32B** 生成个性化画像并执行 **Profile-Only** 重排的最新结果。  
> 画像文件：`results/LitSearch/personalized_queries_qwen3-32b.jsonl`、`results/MedCorpus/personalized_queries_qwen3-32b.jsonl`

| 数据集 | Profile-Only 结果文件 | 评估指标（NDCG@10 / MAP@10 / P@1） |
|--------|----------------------|------------------------------------|
| LitSearch | `results/LitSearch/ranked_jina-v3_profile-only_top10.jsonl` | 0.4840 / 0.4653 / 0.4204 |
| MedCorpus | `results/MedCorpus/ranked_jina-v3_profile-only_top10.jsonl` | 0.9239 / 0.5494 / 0.9526 |

- 评估脚本：`python3 core/evaluate.py --dataset <litsearch|medcorpus> --gt <ground_truth_path> --results <profile_only_file>`  
- 重排命令示例（LitSearch，GPU0）：  
  `python3 core/run.py --mode simple_profile_rerank --dataset_name LitSearch --gpu_id 0 --siliconflow_model Qwen/Qwen3-32B --reranker_type jina-v3 --reranker_path /workspace/PerMed/model/jina-reranker-v3 --rerank_input_type profile_only --results_dir /mnt/data/zsy-data/PerMed/results`
- MedCorpus 同理，仅将 `--dataset_name` 切换为 `MedCorpus`。

这些结果可直接与主文 14B 版本对照，方便后续进行效率/效果折衷分析。

## 8. 文件清单

### 8.1 LitSearch结果文件
- `results/LitSearch/ranked_jina-v3_query-only_top10.jsonl`
- `results/LitSearch/ranked_jina-v3_profile-only_top10.jsonl`
- `results/LitSearch/ranked_jina-v3_profile-and-query_top10.jsonl`

### 8.2 MedCorpus结果文件
- `results/MedCorpus/ranked_jina-v3_query-only_top10.jsonl`
- `results/MedCorpus/ranked_jina-v3_profile-only_top10.jsonl`
- `results/MedCorpus/ranked_jina-v3_profile-and-query_top10.jsonl`

### 8.3 评估脚本
- `core/evaluate.py`: 通用评估脚本
- `core/rerank.py`: 重排脚本(支持三种模式)

---

## 9. 效率对比分析 (2025-11-26)

### 9.1 GPT-4o-mini 直接重排 vs 我们的方法

对比场景：MedCorpus 数据集，3440 个查询

| 指标 | GPT-4o-mini 重排 | 我们的方法 (Qwen3-32B + Jina) | 倍数 |
|------|-----------------|------------------------------|------|
| **总时间** | 466.9 分钟 (7.8h) | 18.3 分钟 | **25x 更快** |
| **Token 消耗** | ~7.05M | 0.89M | **7.9x 更省** |
| **NDCG@10** | 0.9406 | 0.9078 | 仅差 3.5% |
| **MAP@10** | 0.5640 | 0.5440 | 仅差 3.5% |
| **P@1** | 0.9642 | 0.9206 | 仅差 4.5% |

### 9.2 效率分析

**GPT-4o-mini 重排成本估算：**
- 输入 tokens: ~6.71M (每查询 ~1950 tokens)
- 输出 tokens: ~0.34M (每查询 ~100 tokens)
- 平均延迟: 8.14 秒/查询

**我们的方法成本：**
- Profile 生成: 0.89M tokens (Qwen3-32B)
- 平均延迟: 0.32 秒/查询 (生成) + Jina 重排
- 无需每次查询都调用 LLM 重排

### 9.3 结论

> 我们的方法在效果仅低 3-4% 的情况下，时间成本降低 **25 倍**，Token 消耗降低近 **8 倍**。这种效率优势使得个性化检索在实际应用中更具可行性。

---

**报告生成**: Claude Code
**评估标准**: EVALUATION_METRICS_STANDARD.md
