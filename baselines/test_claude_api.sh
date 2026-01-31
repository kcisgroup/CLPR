#!/bin/bash
# 快速测试 Claude 4.5 Haiku API 是否正常工作

echo "=========================================="
echo "测试 Claude 4.5 Haiku API"
echo "=========================================="
echo ""

cd /mnt/data/zsy-data/PerMed/baselines

# 测试前3个查询
python3 llm_reranker.py \
  --api_type anthropic \
  --dataset_type medcorpus \
  --queries data/MedCorpus/queries.jsonl \
  --corpus data/MedCorpus/corpus.jsonl \
  --retrieval /mnt/data/zsy-data/PerMed/results/MedCorpus/retrieved.jsonl \
  --output results/MedCorpus/test_claude_results.jsonl \
  --api_url "https://code.newcli.com/claude/aws" \
  --api_key "-----" \
  --model "claude-haiku-4-5-20251001" \
  --top_k 10 \
  --include_history \
  --temperature 0.0 \
  --rpm_limit 60 \
  --start_from 0

# 检查结果
if [ -f "results/MedCorpus/test_claude_results.jsonl" ]; then
    LINES=$(wc -l < results/MedCorpus/test_claude_results.jsonl)
    echo ""
    echo "=========================================="
    echo "✓ 测试成功！"
    echo "处理了 $LINES 个查询"
    echo "=========================================="
    echo ""
    echo "如果测试成功，可以运行完整任务："
    echo "  bash run_claude_medcorpus.sh > claude_medcorpus.log 2>&1 &"
    echo ""
    echo "监控进度："
    echo "  bash check_llm_progress.sh"
else
    echo ""
    echo "=========================================="
    echo "✗ 测试失败"
    echo "=========================================="
fi
