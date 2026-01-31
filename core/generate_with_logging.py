#!/usr/bin/env python3
"""
生成个性化文本并记录详细日志
"""
import time
import json
import sys
from personalized_generator import main as generate_main

if __name__ == "__main__":
    print("="*80)
    print("开始生成个性化文本 - Qwen3-Next-80B-A3B-Thinking")
    print("数据集: MedCorpus")
    print("="*80)

    start_time = time.time()

    # 设置命令行参数
    sys.argv = [
        'generate_with_logging.py',
        '--dataset_name', 'MedCorpus',
        '--siliconflow_model', 'Qwen/Qwen3-Next-80B-A3B-Thinking',
        '--results_dir', '/mnt/data/zsy-data/PerMed/results'
    ]

    # 运行生成
    generate_main()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("="*80)
    print(f"生成完成！")
    print(f"总耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    print("="*80)

    # 保存日志
    log_data = {
        "model": "Qwen/Qwen3-Next-80B-A3B-Thinking",
        "dataset": "MedCorpus",
        "elapsed_time_seconds": elapsed_time,
        "elapsed_time_minutes": elapsed_time / 60
    }

    with open('/tmp/qwen3-next-80b_generation_log.json', 'w') as f:
        json.dump(log_data, f, indent=2)
