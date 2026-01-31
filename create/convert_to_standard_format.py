#!/usr/bin/env python3
"""
清洗银标准数据并转换为标准格式
- 移除explanation字段,只保留relevance标签
- 转换为queries.jsonl和labels_turn.jsonl格式
- 统计标注分布情况
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

# 路径配置
SILVER_LABELS_FILE = "/workspace/PerMed/results/silver_labels_qwen3.jsonl"
OUTPUT_DIR = Path("/workspace/PerMed/data/MedCorpus_MultiTurn/qwen3_silver")
QUERIES_OUTPUT = OUTPUT_DIR / "queries.jsonl"
LABELS_OUTPUT = OUTPUT_DIR / "labels_turn.jsonl"
METADATA_OUTPUT = OUTPUT_DIR / "metadata.json"

def load_silver_labels():
    """加载原始银标准数据"""
    data = []
    with open(SILVER_LABELS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def convert_to_standard_format(silver_data):
    """转换为标准格式"""
    queries = []
    labels = []

    # 统计信息
    stats = {
        'total_conversations': len(silver_data),
        'total_turns': 0,
        'total_labels': 0,
        'relevance_distribution': Counter(),
        'relevance_by_turn': defaultdict(Counter),
        'turns_distribution': Counter()
    }

    for conv in silver_data:
        topic_id = conv['topic_id']
        target_turns = conv['target_turns']

        # 统计轮数分布
        stats['turns_distribution'][target_turns] += 1

        # 构建queries格式
        query_data = {
            'conversation_id': topic_id,
            'turns': [],
            'target_turns': target_turns
        }

        for turn in conv['turns']:
            turn_id = turn['turn_id']
            question = turn['question']

            stats['total_turns'] += 1

            # 添加到queries
            query_data['turns'].append({
                'turn_id': turn_id,
                'text': question
            })

            # 处理labels
            for label in turn['labels']:
                doc_id = label['doc_id']
                relevance = label['relevance']

                stats['total_labels'] += 1
                stats['relevance_distribution'][relevance] += 1
                stats['relevance_by_turn'][turn_id][relevance] += 1

                # 添加到labels (不包含explanation)
                labels.append({
                    'conversation_id': topic_id,
                    'turn_id': turn_id,
                    'doc_id': doc_id,
                    'rel': relevance
                })

        queries.append(query_data)

    return queries, labels, stats

def save_data(queries, labels, stats):
    """保存数据到文件"""
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存queries
    with open(QUERIES_OUTPUT, 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')

    # 保存labels
    with open(LABELS_OUTPUT, 'w', encoding='utf-8') as f:
        for label in labels:
            f.write(json.dumps(label, ensure_ascii=False) + '\n')

    # 保存metadata
    metadata = {
        'dataset_name': 'MedCorpus_MultiTurn_Qwen3_Silver',
        'description': 'Multi-turn conversational retrieval dataset with silver-standard labels',
        'retrieval_model': 'Qwen3-Embedding-0.6B',
        'annotation_model': 'DeepSeek-V3.2-Exp',
        'annotation_type': 'silver-standard (LLM-generated)',
        'relevance_scale': '0 (Not Relevant), 1 (Partially Relevant), 2 (Highly Relevant)',
        'statistics': {
            'conversations': stats['total_conversations'],
            'turns': stats['total_turns'],
            'labels': stats['total_labels'],
            'avg_turns_per_conversation': stats['total_turns'] / stats['total_conversations'],
            'turns_distribution': dict(stats['turns_distribution']),
            'relevance_distribution': dict(stats['relevance_distribution']),
            'relevance_by_turn': {k: dict(v) for k, v in stats['relevance_by_turn'].items()}
        }
    }

    with open(METADATA_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return metadata

def print_statistics(stats):
    """打印统计信息"""
    print("=" * 80)
    print("银标准数据统计")
    print("=" * 80)
    print()

    print(f"📊 数据规模:")
    print(f"  - 对话数: {stats['total_conversations']}")
    print(f"  - 总轮数: {stats['total_turns']}")
    print(f"  - 总标签数: {stats['total_labels']}")
    print(f"  - 平均每对话轮数: {stats['total_turns'] / stats['total_conversations']:.2f}")
    print()

    print(f"📈 轮数分布:")
    for turns, count in sorted(stats['turns_distribution'].items()):
        pct = count / stats['total_conversations'] * 100
        print(f"  - {turns}轮对话: {count} ({pct:.1f}%)")
    print()

    print(f"🏷️  相关性标签分布 (全局):")
    total = stats['total_labels']
    for rel in [0, 1, 2]:
        count = stats['relevance_distribution'][rel]
        pct = count / total * 100
        label_name = ['不相关', '部分相关', '高度相关'][rel]
        print(f"  - {rel} ({label_name}): {count:,} ({pct:.1f}%)")
    print()

    print(f"📊 按轮次统计相关性分布:")
    for turn_id in sorted(stats['relevance_by_turn'].keys()):
        turn_stats = stats['relevance_by_turn'][turn_id]
        turn_total = sum(turn_stats.values())
        print(f"  Turn {turn_id}:")
        for rel in [0, 1, 2]:
            count = turn_stats[rel]
            pct = count / turn_total * 100 if turn_total > 0 else 0
            print(f"    {rel}: {count} ({pct:.1f}%)")
    print()

def main():
    print("=" * 80)
    print("清洗银标准数据并转换为标准格式")
    print("=" * 80)
    print()

    # 1. 加载数据
    print("[1/4] 加载银标准数据...")
    silver_data = load_silver_labels()
    print(f"  ✓ 已加载 {len(silver_data)} 个对话")
    print()

    # 2. 转换格式
    print("[2/4] 转换为标准格式...")
    queries, labels, stats = convert_to_standard_format(silver_data)
    print(f"  ✓ queries: {len(queries)} 个对话")
    print(f"  ✓ labels: {len(labels)} 条标签")
    print()

    # 3. 保存数据
    print("[3/4] 保存数据到文件...")
    metadata = save_data(queries, labels, stats)
    print(f"  ✓ queries.jsonl: {QUERIES_OUTPUT}")
    print(f"  ✓ labels_turn.jsonl: {LABELS_OUTPUT}")
    print(f"  ✓ metadata.json: {METADATA_OUTPUT}")
    print()

    # 4. 打印统计
    print("[4/4] 统计信息:")
    print_statistics(stats)

    print("=" * 80)
    print("✅ 数据清洗和转换完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
