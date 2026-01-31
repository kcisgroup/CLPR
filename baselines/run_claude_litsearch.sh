#!/bin/bash
# 运行Claude 4.5 Haiku Reranker - LitSearch数据集

API_URL="https://code.newcli.com/claude/aws"
API_KEY="-----"
MODEL="claude-haiku-4-5-20251001"
RPM_LIMIT=120  # Claude API 限速（提高到120）

DATASET="LitSearch"
OUTPUT="results/LitSearch/llm_claude_haiku_results.jsonl"

echo "=========================================="
echo "Claude 4.5 Haiku Reranker - LitSearch"
echo "=========================================="
echo "数据集: $DATASET"
echo "模型: $MODEL"
echo "限速: $RPM_LIMIT RPM"
echo "包含对话历史: Yes"
echo "输出: $OUTPUT"
echo "=========================================="
echo ""
echo "预计查询数: 597"
echo "预计耗时: ~5分钟"
echo ""

cd /mnt/data/zsy-data/PerMed/baselines

python3 llm_reranker.py \
  --api_type anthropic \
  --dataset_type litsearch \
  --queries data/LitSearch/queries.jsonl \
  --corpus data/LitSearch/corpus.jsonl \
  --retrieval /mnt/data/zsy-data/PerMed/results/LitSearch/retrieved.jsonl \
  --output $OUTPUT \
  --api_url $API_URL \
  --api_key $API_KEY \
  --model $MODEL \
  --top_k 10 \
  --include_history \
  --temperature 0.0 \
  --rpm_limit $RPM_LIMIT \
  --start_from 0

echo ""
echo "=========================================="
echo "LitSearch 完成!"
echo "结果保存在: $OUTPUT"
echo "=========================================="
