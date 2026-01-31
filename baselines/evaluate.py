#!/usr/bin/env python3
"""
通用评估脚本，支持：
1. MedCorpus（多轮对话，0/1/2 相关性）
2. LitSearch（单轮查询，二元相关性）

主要指标：NDCG@10, MAP@10, P@1
"""
import json
import math
from collections import defaultdict
import argparse
from pathlib import Path


def load_ground_truth_medcorpus(gt_file):
    """加载 MedCorpus ground truth (多轮，带相关性等级)"""
    gt = defaultdict(dict)
    with open(gt_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            conv_id = data['conversation_id']
            turn_id = data['turn_id']
            query_id = f"{conv_id}_turn_{turn_id}"
            doc_id = data['doc_id']
            rel = data['rel']  # 0, 1, 2
            gt[query_id][doc_id] = rel
    return gt


def load_ground_truth_litsearch(gt_file):
    """加载 LitSearch ground truth (单轮，二元相关性)"""
    gt = defaultdict(dict)
    with open(gt_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            query_id = str(data['query_id'])
            relevant_texts = data['relevant_texts']
            # 将相关文档标记为 1（二元相关性）
            for doc_id in relevant_texts:
                gt[query_id][str(doc_id)] = 1
    return gt


def load_results(results_file):
    """加载重排结果"""
    results = {}
    with open(results_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            query_id = str(data['query_id'])
            
            # 处理格式差异：将 "topic_001_1" 转换为 "topic_001_turn_1"
            if query_id.count('_') == 2 and 'turn' not in query_id:
                parts = query_id.rsplit('_', 1)
                query_id = f"{parts[0]}_turn_{parts[1]}"
            
            ranked_docs = [str(doc['text_id']) for doc in data['ranked_results']]
            results[query_id] = ranked_docs
    return results


def dcg_at_k(relevances, k):
    """计算 DCG@k (标准公式，支持多级相关性)"""
    relevances = relevances[:k]
    return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(gt_rels, pred_docs, k):
    """计算 NDCG@k"""
    # 获取预测文档的相关性分数
    relevances = [gt_rels.get(doc, 0) for doc in pred_docs[:k]]
    
    # 计算 DCG
    dcg = dcg_at_k(relevances, k)
    
    # 计算 IDCG（理想情况下的 DCG）
    ideal_rels = sorted(gt_rels.values(), reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(gt_rels, pred_docs, k):
    """计算 Precision@k"""
    relevant_docs = set([doc for doc, rel in gt_rels.items() if rel > 0])
    if k == 0:
        return 0.0
    
    retrieved_relevant = set(pred_docs[:k]) & relevant_docs
    return len(retrieved_relevant) / k


def average_precision(gt_rels, pred_docs, k):
    """
    计算 Average Precision@k (考虑多级相关性 0/1/2)
    
    多级相关性权重设置（统一标准）:
    - rel=2 权重: 1.0
    - rel=1 权重: 0.5
    - rel=0 权重: 0.0
    """
    score = 0.0
    num_relevant = 0
    
    for i, doc in enumerate(pred_docs[:k], 1):
        rel = gt_rels.get(doc, 0)
        if rel > 0:  # 相关文档
            num_relevant += 1
            precision_at_i = num_relevant / i
            # 按相关性等级加权: rel=2权重1.0, rel=1权重0.5
            score += precision_at_i * (rel / 2.0)
    
    # 归一化：除以总相关文档数
    total_relevant = sum(1 for r in gt_rels.values() if r > 0)
    return score / total_relevant if total_relevant > 0 else 0.0


def evaluate(gt_file, results_file, dataset_type='medcorpus', k=10):
    """
    评估重排结果
    
    Args:
        gt_file: ground truth 文件路径
        results_file: 重排结果文件路径
        dataset_type: 'medcorpus' 或 'litsearch'
        k: 评估的截断值
    """
    # 加载 ground truth
    if dataset_type == 'medcorpus':
        gt = load_ground_truth_medcorpus(gt_file)
    elif dataset_type == 'litsearch':
        gt = load_ground_truth_litsearch(gt_file)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    # 加载结果
    results = load_results(results_file)
    
    # 计算各项指标
    ndcg_scores = []
    map_scores = []
    p1_scores = []
    
    # 只评估有 ground truth 的查询
    common_queries = set(gt.keys()) & set(results.keys())
    
    for query_id in common_queries:
        gt_rels = gt[query_id]
        pred_docs = results[query_id]
        
        ndcg_scores.append(ndcg_at_k(gt_rels, pred_docs, k))
        map_scores.append(average_precision(gt_rels, pred_docs, k))
        p1_scores.append(precision_at_k(gt_rels, pred_docs, 1))
    
    # 计算平均值
    metrics = {
        f'NDCG@{k}': sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0,
        f'MAP@{k}': sum(map_scores) / len(map_scores) if map_scores else 0,
        'P@1': sum(p1_scores) / len(p1_scores) if p1_scores else 0,
        'num_queries': len(common_queries)
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='评估重排结果')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['medcorpus', 'litsearch'],
                       help='数据集类型')
    parser.add_argument('--gt', type=str, required=True,
                       help='Ground truth 文件路径')
    parser.add_argument('--results', type=str, nargs='+', required=True,
                       help='重排结果文件路径（可以多个）')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='结果文件的标签（用于显示）')
    parser.add_argument('--k', type=int, default=10,
                       help='评估的截断值 (default: 10)')
    
    args = parser.parse_args()
    
    # 如果没有提供标签，使用文件名
    if args.labels is None:
        args.labels = [Path(f).stem for f in args.results]
    
    if len(args.labels) != len(args.results):
        print(f"错误: 标签数量 ({len(args.labels)}) 与结果文件数量 ({len(args.results)}) 不匹配")
        return
    
    print("=" * 80)
    print(f"评估数据集: {args.dataset.upper()}")
    print(f"Ground Truth: {args.gt}")
    print(f"评估指标: NDCG@{args.k}, MAP@{args.k}, P@1")
    print("=" * 80)
    print()
    
    results_summary = {}
    
    for label, file_path in zip(args.labels, args.results):
        print(f"评估: {label}")
        print("-" * 80)
        
        metrics = evaluate(args.gt, file_path, dataset_type=args.dataset, k=args.k)
        results_summary[label] = metrics
        
        print(f"查询数量: {metrics['num_queries']}")
        print(f"NDCG@{args.k}:  {metrics[f'NDCG@{args.k}']:.4f}")
        print(f"MAP@{args.k}:   {metrics[f'MAP@{args.k}']:.4f}")
        print(f"P@1:      {metrics['P@1']:.4f}")
        print()
    
    # 如果有多个结果，输出对比表格
    if len(results_summary) >= 2:
        print("=" * 80)
        print("对比总结")
        print("=" * 80)
        print()
        
        labels_list = list(results_summary.keys())
        print(f"{'指标':<15} ", end="")
        for label in labels_list:
            print(f"{label:<20} ", end="")
        if len(labels_list) == 2:
            print("提升", end="")
        print()
        print("-" * 80)
        
        for metric in [f'NDCG@{args.k}', f'MAP@{args.k}', 'P@1']:
            print(f"{metric:<15} ", end="")
            values = []
            for label in labels_list:
                val = results_summary[label][metric]
                values.append(val)
                print(f"{val:<20.4f} ", end="")
            
            # 如果是两个结果，计算提升
            if len(values) == 2:
                diff = values[0] - values[1]
                diff_pct = (diff / values[1] * 100) if values[1] > 0 else 0
                print(f"{diff:+.4f} ({diff_pct:+.2f}%)", end="")
            print()
        
        print()


if __name__ == '__main__':
    main()

