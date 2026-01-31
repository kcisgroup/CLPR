#!/bin/bash
# 运行LLM Reranker - LitSearch数据集

API_URL="https://api.minimaxi.com/v1"
API_KEY="-----"
MODEL="MiniMax-M2"
RPM_LIMIT=180

DATASET="LitSearch"
OUTPUT="results/LitSearch/llm_minimax_results.jsonl"

echo "=========================================="
echo "LLM Reranker - LitSearch"
echo "=========================================="
echo "数据集: $DATASET"
echo "模型: $MODEL"
echo "限速: $RPM_LIMIT RPM"
echo "包含对话历史: No (单轮查询)"
echo "输出: $OUTPUT"
echo "=========================================="
echo ""

# 首先统计查询数
NUM_QUERIES=$(wc -l < data/LitSearch/queries.jsonl)
echo "查询数: $NUM_QUERIES"
ESTIMATED_TIME=$(echo "scale=1; $NUM_QUERIES / $RPM_LIMIT" | bc)
echo "预计耗时: ~${ESTIMATED_TIME} 分钟"
echo ""

cd /mnt/data/zsy-data/PerMed/baselines

python3 llm_reranker.py \
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





