#!/usr/bin/env python3
"""
对比检索前和重排后的评估结果
评估指标：NDCG@10, MAP@10, P@1
支持MedCorpus多级相关性 (0/1/2)
"""
import json
import math
from collections import defaultdict


def load_ground_truth(gt_file):
    """加载 ground truth (MedCorpus 格式)"""
    gt = defaultdict(dict)
    with open(gt_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            conv_id = data['conversation_id']
            turn_id = data['turn_id']
            query_id = f"{conv_id}_{turn_id}"
            doc_id = data['doc_id']
            rel = data['rel']
            gt[query_id][doc_id] = rel
    return gt


def load_retrieved_results(results_file):
    """加载检索前的结果（格式：results字段）"""
    results = {}
    with open(results_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            query_id = data['query_id']
            # 检索前的文件使用 'results' 字段
            ranked_docs = [doc['text_id'] for doc in data['results']]
            results[query_id] = ranked_docs
    return results


def load_reranked_results(results_file):
    """加载重排后的结果（格式：ranked_results字段）"""
    results = {}
    with open(results_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            query_id = data['query_id']
            # 重排后的文件使用 'ranked_results' 字段
            ranked_docs = [doc['text_id'] for doc in data['ranked_results']]
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


def evaluate(gt, results, k=10):
    """评估单个结果文件"""
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
    # 数据文件路径
    gt_file = '/workspace/PerMed/data/MedCorpus_MultiTurn/query_to_texts.jsonl'
    
    # 评估三个文件
    files = {
        '检索前 (Retrieved)': {
            'path': '/workspace/PerMed/results/MedCorpus/retrieved.jsonl',
            'type': 'retrieved'  # 使用特殊加载函数
        },
        '重排-Profile+Query': {
            'path': '/workspace/PerMed/results/MedCorpus/ranked_jina-v3_profile-and-query_top10.jsonl',
            'type': 'reranked'
        },
        '重排-Query-only': {
            'path': '/workspace/PerMed/results/MedCorpus/ranked_jina-v3_query-only_top10.jsonl',
            'type': 'reranked'
        }
    }
    
    print("=" * 100)
    print("MedCorpus 数据集 - 检索前 vs 重排后 评估对比")
    print("=" * 100)
    print()
    print("评估说明:")
    print("  - 数据集: MedCorpus (多级相关性: 0/1/2)")
    print("  - NDCG公式: 标准学术公式 DCG = Σ[(2^rel - 1) / log2(i+1)]")
    print("    * rel=2 权重: 3 (2^2-1)")
    print("    * rel=1 权重: 1 (2^1-1)")
    print("  - MAP计算: 考虑多级相关性")
    print("    * rel=2 权重: 1.0")
    print("    * rel=1 权重: 0.5")
    print("=" * 100)
    print()
    
    # 加载 ground truth
    gt = load_ground_truth(gt_file)
    
    results_summary = {}
    
    # 评估每个文件
    for name, file_info in files.items():
        print(f"评估: {name}")
        print("-" * 100)
        
        # 根据类型选择加载函数
        if file_info['type'] == 'retrieved':
            results = load_retrieved_results(file_info['path'])
        else:
            results = load_reranked_results(file_info['path'])
        
        metrics = evaluate(gt, results, k=10)
        results_summary[name] = metrics
        
        print(f"  查询数量: {metrics['num_queries']}")
        print(f"  NDCG@10:  {metrics['NDCG@10']:.4f}")
        print(f"  MAP@10:   {metrics['MAP@10']:.4f}")
        print(f"  P@1:      {metrics['P@1']:.4f}")
        print()
    
    # 对比表格
    print("=" * 100)
    print("详细对比表格")
    print("=" * 100)
    print()
    print(f"{'方法':<25} {'NDCG@10':<12} {'MAP@10':<12} {'P@1':<12}")
    print("-" * 100)
    
    for name, metrics in results_summary.items():
        print(f"{name:<25} {metrics['NDCG@10']:<12.4f} {metrics['MAP@10']:<12.4f} {metrics['P@1']:<12.4f}")
    
    print()
    
    # 计算相对于检索前的提升
    print("=" * 100)
    print("相对于检索前的提升分析")
    print("=" * 100)
    print()
    
    baseline = results_summary['检索前 (Retrieved)']
    
    for name in ['重排-Profile+Query', '重排-Query-only']:
        if name in results_summary:
            print(f"{name}:")
            print("-" * 100)
            
            metrics = results_summary[name]
            
            for metric_name in ['NDCG@10', 'MAP@10', 'P@1']:
                val_curr = metrics[metric_name]
                val_base = baseline[metric_name]
                diff = val_curr - val_base
                diff_pct = (diff / val_base * 100) if val_base > 0 else 0
                
                indicator = "🏆" if diff > 0 else "❌" if diff < 0 else "➖"
                print(f"  {metric_name:<10}: {val_curr:.4f} vs {val_base:.4f} = {diff:+.4f} ({diff_pct:+.2f}%) {indicator}")
            
            print()
    
    # 对比两种重排方法
    print("=" * 100)
    print("重排方法对比: Profile+Query vs Query-only")
    print("=" * 100)
    print()
    
    if '重排-Profile+Query' in results_summary and '重排-Query-only' in results_summary:
        profile_query = results_summary['重排-Profile+Query']
        query_only = results_summary['重排-Query-only']
        
        print(f"{'指标':<15} {'Profile+Query':<15} {'Query-only':<15} {'提升':<25}")
        print("-" * 100)
        
        for metric in ['NDCG@10', 'MAP@10', 'P@1']:
            val1 = profile_query[metric]
            val2 = query_only[metric]
            diff = val1 - val2
            diff_pct = (diff / val2 * 100) if val2 > 0 else 0
            
            indicator = "✓" if diff > 0 else "✗"
            print(f"{metric:<15} {val1:<15.4f} {val2:<15.4f} {diff:+.4f} ({diff_pct:+.2f}%) {indicator}")
        
        print()
    
    # 保存结果到文件
    output_file = '/workspace/PerMed/retrieval_vs_rerank_comparison.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("MedCorpus 数据集 - 检索前 vs 重排后 评估对比\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("评估说明:\n")
        f.write("  - 数据集: MedCorpus (多级相关性: 0/1/2)\n")
        f.write("  - NDCG公式: 标准学术公式 DCG = Σ[(2^rel - 1) / log2(i+1)]\n")
        f.write("    * rel=2 权重: 3 (2^2-1)\n")
        f.write("    * rel=1 权重: 1 (2^1-1)\n")
        f.write("  - MAP计算: 考虑多级相关性\n")
        f.write("    * rel=2 权重: 1.0\n")
        f.write("    * rel=1 权重: 0.5\n\n")
        
        f.write("-" * 100 + "\n")
        f.write(f"{'方法':<25} {'NDCG@10':<12} {'MAP@10':<12} {'P@1':<12}\n")
        f.write("-" * 100 + "\n")
        
        for name, metrics in results_summary.items():
            f.write(f"{name:<25} {metrics['NDCG@10']:<12.4f} {metrics['MAP@10']:<12.4f} {metrics['P@1']:<12.4f}\n")
        
        f.write("\n")
        f.write("相对于检索前的提升:\n")
        f.write("-" * 100 + "\n")
        
        for name in ['重排-Profile+Query', '重排-Query-only']:
            if name in results_summary:
                f.write(f"\n{name}:\n")
                metrics = results_summary[name]
                
                for metric_name in ['NDCG@10', 'MAP@10', 'P@1']:
                    val_curr = metrics[metric_name]
                    val_base = baseline[metric_name]
                    diff = val_curr - val_base
                    diff_pct = (diff / val_base * 100) if val_base > 0 else 0
                    
                    f.write(f"  {metric_name:<10}: {val_curr:.4f} vs {val_base:.4f} = {diff:+.4f} ({diff_pct:+.2f}%)\n")
    
    print(f"详细结果已保存到: {output_file}")
    print()


if __name__ == '__main__':
    main()

