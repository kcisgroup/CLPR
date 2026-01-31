#!/bin/bash
# 检查LLM重排任务进度

echo "=========================================="
echo "LLM重排任务进度监控"
echo "=========================================="
echo ""

echo "【MedCorpus任务状态】"
echo "--- MiniMax-M2 ---"
if ps aux | grep -v grep | grep "run_llm_medcorpus.sh" > /dev/null; then
    echo "状态: 运行中 ✓"
    echo "最新进度:"
    tail -3 /mnt/data/zsy-data/PerMed/baselines/medcorpus_llm.log 2>/dev/null || echo "  日志文件未找到"
    echo ""
    # 检查结果文件
    if [ -f "results/MedCorpus/llm_minimax_results.jsonl" ]; then
        MEDCORPUS_DONE=$(wc -l < results/MedCorpus/llm_minimax_results.jsonl)
        echo "已完成: $MEDCORPUS_DONE / 3440 个查询"
    fi
else
    echo "状态: 已完成或未运行"
    if [ -f "results/MedCorpus/llm_minimax_results.jsonl" ]; then
        MEDCORPUS_DONE=$(wc -l < results/MedCorpus/llm_minimax_results.jsonl)
        echo "结果: $MEDCORPUS_DONE 个查询"
    fi
fi
echo ""

echo "--- Claude 4.5 Haiku ---"
if ps aux | grep -v grep | grep "run_claude_medcorpus.sh" > /dev/null; then
    echo "状态: 运行中 ✓"
    echo "最新进度:"
    tail -3 /mnt/data/zsy-data/PerMed/baselines/claude_medcorpus.log 2>/dev/null || echo "  日志文件未找到"
    echo ""
    # 检查结果文件
    if [ -f "results/MedCorpus/llm_claude_haiku_results.jsonl" ]; then
        CLAUDE_DONE=$(wc -l < results/MedCorpus/llm_claude_haiku_results.jsonl)
        echo "已完成: $CLAUDE_DONE / 3440 个查询"
    fi
else
    echo "状态: 已完成或未运行"
    if [ -f "results/MedCorpus/llm_claude_haiku_results.jsonl" ]; then
        CLAUDE_DONE=$(wc -l < results/MedCorpus/llm_claude_haiku_results.jsonl)
        echo "结果: $CLAUDE_DONE 个查询"
    fi
fi
echo ""

echo "【LitSearch任务状态】"
echo "--- MiniMax-M2 ---"
if ps aux | grep -v grep | grep "run_llm_litsearch.sh" > /dev/null; then
    echo "状态: 运行中 ✓"
    echo "最新进度:"
    tail -3 /mnt/data/zsy-data/PerMed/baselines/litsearch_llm.log 2>/dev/null || echo "  日志文件未找到"
    echo ""
    # 检查结果文件
    if [ -f "results/LitSearch/llm_minimax_results.jsonl" ]; then
        LITSEARCH_DONE=$(wc -l < results/LitSearch/llm_minimax_results.jsonl)
        echo "已完成: $LITSEARCH_DONE / 597 个查询"
    fi
else
    echo "状态: 已完成或未运行"
    if [ -f "results/LitSearch/llm_minimax_results.jsonl" ]; then
        LITSEARCH_DONE=$(wc -l < results/LitSearch/llm_minimax_results.jsonl)
        echo "结果: $LITSEARCH_DONE 个查询"
    fi
fi
echo ""

echo "--- Claude 4.5 Haiku ---"
if ps aux | grep -v grep | grep "run_claude_litsearch.sh" > /dev/null; then
    echo "状态: 运行中 ✓"
    echo "最新进度:"
    tail -3 /mnt/data/zsy-data/PerMed/baselines/claude_litsearch.log 2>/dev/null || echo "  日志文件未找到"
    echo ""
    # 检查结果文件
    if [ -f "results/LitSearch/llm_claude_haiku_results.jsonl" ]; then
        CLAUDE_LIT_DONE=$(wc -l < results/LitSearch/llm_claude_haiku_results.jsonl)
        echo "已完成: $CLAUDE_LIT_DONE / 597 个查询"
    fi
else
    echo "状态: 已完成或未运行"
    if [ -f "results/LitSearch/llm_claude_haiku_results.jsonl" ]; then
        CLAUDE_LIT_DONE=$(wc -l < results/LitSearch/llm_claude_haiku_results.jsonl)
        echo "结果: $CLAUDE_LIT_DONE 个查询"
    fi
fi
echo ""

echo "=========================================="
echo "查看完整日志:"
echo "  MedCorpus MiniMax:   tail -f medcorpus_llm.log"
echo "  MedCorpus Claude:    tail -f claude_medcorpus.log"
echo "  LitSearch MiniMax:   tail -f litsearch_llm.log"
echo "  LitSearch Claude:    tail -f claude_litsearch.log"
echo "=========================================="


