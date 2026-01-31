# PerMed Experiments

This directory aggregates every evaluation workflow that sits outside the
core `core/run.py` pipeline. At the moment我们重点跟踪三组实验：

1. **记忆组件消融 (Memory Ablation, profile-only 重排)**
2. **重排输入对比 (Query vs. Profile vs. Hybrid)** — 已完成并写入
   `RERANKING_COMPARISON_REPORT.md`
3. **效率/成本对比 (我们的流水线 vs. 大模型重排)**

历史上的 **evidence-level 质检** 脚本仍放在 `experiments/evidence_eval/`，需要的时候可以单独查看。

---

## 1. Memory Ablation (profile-only)

**目标**：证明“结构化画像 + 当前查询”这一设计的必要性。我们对每个数据集运行 5 个变体：

| 变体 ID | 描述 |
|---------|------|
| `full_memory` | 原始 CLPR 画像（顺序+工作+长期） |
| `no_long` | 去掉 `[LONG_EXPLICIT]`，保留顺序 + 工作 |
| `no_sequential` | 去掉顺序记忆，只保留长期 + 工作 |
| `no_working` | 去掉工作记忆，只保留长期 + 顺序 |
| `query_only` | “画像=原始问题”，等价于 profile-only 模式下的 query-only 基线 |

**位置**：`experiments/memory_ablation/`

**关键文件**：
- `ablation_plan.yaml`：单一配置入口；已把两套数据集 (MedCorpus, LitSearch) 的 5 个变体全部登记。
- `build_metadata.py` → `cache/<Dataset>_metadata.jsonl`
- `generate_profiles.py` → 为每个变体生成 deterministic profile
- `evaluate_ablation.py` → 读取 `core/evaluate.py` 的指标 + plan 里的 subset 规则

**推荐流程**：
```bash
# 1. 预处理 metadata（仅需一次）
python experiments/memory_ablation/build_metadata.py --dataset MedCorpus
python experiments/memory_ablation/build_metadata.py --dataset LitSearch

# 2. 生成目标变体的画像
python experiments/memory_ablation/generate_profiles.py --dataset MedCorpus --variant no_long
python experiments/memory_ablation/generate_profiles.py --dataset MedCorpus --variant query_only

# 3. 调用 core/run.py 做 profile-only rerank（原命令保持不变，改传入新的画像路径）

# 4. 评估 + 分层
python experiments/memory_ablation/evaluate_ablation.py --dataset MedCorpus --k 10
```

评估 JSON 会写到 `experiments/memory_ablation/analysis/<Dataset>_ablation_eval.json`，可直接用于画
“Turn≥3”、“代词引用”、“topic shift”等细分图。

---

## 2. 重排输入对比 (Query vs. Profile vs. Hybrid)

**位置**：`RERANKING_COMPARISON_REPORT.md`

该实验已经全量完成，报告里包含两张表。为了方便引用，这里再列一次关键指标：

### LitSearch (597 queries, 二元相关性)

| 方法 | NDCG@10 | MAP@10 | P@1 |
|------|---------|--------|-----|
| Query-Only | 0.2588 | 0.0877 | 0.0687 |
| **Profile-Only (Ours)** | **0.4336** | **0.2003** | **0.3400** |
| Hybrid (Profile+Query) | 0.4251 | 0.1948 | 0.3266 |

结论：画像信号最强，较 Query-Only 的 NDCG@10 提升 +67.6%。

### MedCorpus (3440 queries, 三级相关性)

| 方法 | NDCG@10 | MAP@10 | P@1 |
|------|---------|--------|-----|
| Query-Only | 0.8272 | 0.5110 | 0.8154 |
| **Profile-Only (Ours)** | **0.9078** | **0.5440** | **0.9206** |
| Hybrid (Profile+Query) | 0.9070 | 0.5435 | 0.9203 |

结论：在多轮医学对话中，Profile-Only 和 Hybrid 表现几乎一致，但都显著优于 Query-Only
(NDCG@10 +9.7%)。

---

## 3. 效率 / 成本对比

**目标**：量化我们“画像生成 + 本地重排”与“大模型直接重排”的耗时和 token 消耗，支撑
“虽然分数略低于 LLM 重排，但性价比更高”的论点。

| 流水线 | 数据集 | 查询数 | 记录耗时 | Token / API 调用 |
|--------|--------|--------|----------|------------------|
| **Qwen3-32B 画像生成** (`results/MedCorpus/personalized_queries_qwen3-32b_stats.json`) | MedCorpus | 3440 | 1,095.5 秒 (~18.3 分钟) | 890,613 tokens, 9,275 API 调用 (成功 825) |
| **Jina Profile-Only Rerank (本地 GPU)** (`perslitrank.log` 2025-11-15 04:22 run) | MedCorpus | 3440 | 1,348.1 秒 (~22.5 分钟) | 无 API 调用 (本地推理) |
| **GPT-4o-mini LLM 重排** (`baselines/llm_gpt4omini_medcorpus.log`) | MedCorpus | 3440 | 466.9 分钟 | 全程调用 GPT-4o，速率受限 (多次 429) |
| **GPT-4o-mini LLM 重排** (`baselines/llm_gpt4omini_litsearch.log`) | LitSearch | 597 | 78.45 分钟 | 同上 |

可以看到：在 MedCorpus 上，我们的 1) 画像生成 + 2) 本地重排合计 ≈ 40 分钟；而 GPT-4o
重排一次需要 7.8 小时左右，且全部是高成本 API 调用。后续若使用 deterministic 画像生成
（`generate_profiles.py` 的 structured 模式），还可以把“画像阶段”的 API 消耗降为 0。

---

## 4. 其它实验脚本

`experiments/evidence_eval/` 依旧提供 LLM 判分脚本（`run_evidence_eval.py`、`scorer.py` 等），
可用于证据质量评价；用法与原 README 相同，此处不再赘述。

如需新增实验，只要在本 README 中补充一个小节并链接到对应目录 / 报告即可。
