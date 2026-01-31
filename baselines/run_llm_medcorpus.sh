#!/bin/bash
# 运行LLM Reranker - MedCorpus数据集

API_URL="https://api.minimaxi.com/v1"
API_KEY="-----"
MODEL="MiniMax-M2"
RPM_LIMIT=180

DATASET="MedCorpus"
OUTPUT="results/MedCorpus/llm_minimax_results.jsonl"

echo "=========================================="
echo "LLM Reranker - MedCorpus"
echo "=========================================="
echo "数据集: $DATASET"
echo "模型: $MODEL"
echo "限速: $RPM_LIMIT RPM"
echo "包含对话历史: Yes"
echo "输出: $OUTPUT"
echo "=========================================="
echo ""
echo "预计查询数: 3440"
echo "预计耗时: ~19分钟"
echo ""

cd /mnt/data/zsy-data/PerMed/baselines

python3 llm_reranker.py \
  --dataset_type medcorpus \
  --queries data/MedCorpus/queries.jsonl \
  --corpus data/MedCorpus/corpus.jsonl \
  --retrieval /mnt/data/zsy-data/PerMed/results/MedCorpus/retrieved.jsonl \
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
echo "MedCorpus 完成!"
echo "结果保存在: $OUTPUT"
echo "=========================================="





