#!/bin/bash
# 检查baseline运行进度

echo "================================================================================"
echo "Baseline运行进度检查"
echo "================================================================================"

# 检查正在运行的进程
echo -e "\n【运行中的进程】"
ps aux | grep "python.*personarag_baseline\|python.*pbr_baseline" | grep -v grep | while read line; do
    pid=$(echo $line | awk '{print $2}')
    cmd=$(echo $line | grep -o "python.*" | cut -c1-60)
    runtime=$(ps -p $pid -o etime= | xargs 2>/dev/null || echo "N/A")
    echo "  PID: $pid | 运行时长: $runtime"
    echo "  命令: $cmd..."
done

# 检查结果文件
echo -e "\n【已完成的结果】"
find /workspace/PerMed/baselines/results -name "*.jsonl" -type f 2>/dev/null | while read file; do
    lines=$(wc -l < "$file")
    size=$(du -h "$file" | cut -f1)
    echo "  ✅ $file"
    echo "     查询数: $lines | 大小: $size"
done

# 检查最新日志
echo -e "\n【运行日志（最后10行）】"
if [ -f "/workspace/PerMed/baselines/personarag_qwen_medcorpus.log" ]; then
    echo "PersonaRAG-Qwen on MedCorpus:"
    tail -10 /workspace/PerMed/baselines/personarag_qwen_medcorpus.log | grep -E "PersonaRAG|%|✅" | tail -3
fi

echo -e "\n================================================================================"

