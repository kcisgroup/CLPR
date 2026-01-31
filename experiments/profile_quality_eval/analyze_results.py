#!/usr/bin/env python3
"""
结果分析脚本：分析评估结果并生成报告

生成内容：
1. 总体统计
2. 按 Turn 分析
3. 按维度分析
4. 模型间一致性
5. 可视化图表
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from collections import defaultdict
import numpy as np
from scipy import stats

import config


def load_results(results_file: Path) -> List[Dict]:
    """加载评估结果"""
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def extract_scores_by_model(results: List[Dict]) -> Dict:
    """按模型提取分数"""
    scores_by_model = defaultdict(lambda: {
        'relevance': [],
        'accuracy': [],
        'informativeness': [],
        'coherence': [],
        'average_score': []
    })
    
    for result in results:
        for model_key, evaluation in result['evaluations'].items():
            if evaluation.get('success') and evaluation.get('scores'):
                scores = evaluation['scores']
                for dim in ['relevance', 'accuracy', 'informativeness', 'coherence', 'average_score']:
                    scores_by_model[model_key][dim].append(scores[dim])
    
    return dict(scores_by_model)


def extract_scores_by_turn(results: List[Dict]) -> Dict:
    """按 Turn 提取分数"""
    scores_by_turn = defaultdict(lambda: {
        'relevance': [],
        'accuracy': [],
        'informativeness': [],
        'coherence': [],
        'average_score': []
    })
    
    for result in results:
        turn_id = result['turn_id']
        
        # 取所有模型的平均分
        dim_scores = defaultdict(list)
        for model_key, evaluation in result['evaluations'].items():
            if evaluation.get('success') and evaluation.get('scores'):
                scores = evaluation['scores']
                for dim in ['relevance', 'accuracy', 'informativeness', 'coherence', 'average_score']:
                    dim_scores[dim].append(scores[dim])
        
        # 计算平均
        for dim, values in dim_scores.items():
            if values:
                scores_by_turn[turn_id][dim].append(np.mean(values))
    
    return dict(scores_by_turn)


def calculate_inter_model_correlation(scores_by_model: Dict) -> Dict:
    """计算模型间相关性"""
    models = list(scores_by_model.keys())
    if len(models) < 2:
        return {}
    
    correlations = {}
    
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            # 对每个维度计算相关性
            for dim in ['relevance', 'accuracy', 'informativeness', 'coherence', 'average_score']:
                scores1 = scores_by_model[model1][dim]
                scores2 = scores_by_model[model2][dim]
                
                if len(scores1) > 0 and len(scores2) > 0:
                    # 确保长度相同
                    min_len = min(len(scores1), len(scores2))
                    scores1 = scores1[:min_len]
                    scores2 = scores2[:min_len]
                    
                    # Spearman 相关系数
                    rho, p_value = stats.spearmanr(scores1, scores2)
                    
                    key = f"{model1}_vs_{model2}_{dim}"
                    correlations[key] = {
                        'rho': rho,
                        'p_value': p_value
                    }
    
    return correlations


def generate_report(results: List[Dict], output_file: Path):
    """生成分析报告"""
    # 提取数据
    scores_by_model = extract_scores_by_model(results)
    scores_by_turn = extract_scores_by_turn(results)
    correlations = calculate_inter_model_correlation(scores_by_model)
    
    # 生成 Markdown 报告
    report = []
    
    # 标题
    report.append("# Personalized Profile Quality Evaluation Report")
    report.append("")
    report.append(f"**Total Queries Evaluated:** {len(results)}")
    report.append(f"**Models Used:** {', '.join(scores_by_model.keys())}")
    report.append("")
    
    # 1. 总体统计
    report.append("## 1. Overall Statistics")
    report.append("")
    report.append("### Average Scores by Model")
    report.append("")
    report.append("| Model | Relevance | Accuracy | Informativeness | Coherence | Overall |")
    report.append("|-------|-----------|----------|----------------|-----------|---------|")
    
    for model, scores in scores_by_model.items():
        rel = np.mean(scores['relevance'])
        acc = np.mean(scores['accuracy'])
        inf = np.mean(scores['informativeness'])
        coh = np.mean(scores['coherence'])
        avg = np.mean(scores['average_score'])
        report.append(f"| {model} | {rel:.2f} | {acc:.2f} | {inf:.2f} | {coh:.2f} | **{avg:.2f}** |")
    
    report.append("")
    
    # 2. 按 Turn 分析
    report.append("## 2. Analysis by Turn")
    report.append("")
    report.append("### Average Scores by Turn")
    report.append("")
    report.append("| Turn | Relevance | Accuracy | Informativeness | Coherence | Overall |")
    report.append("|------|-----------|----------|----------------|-----------|---------|")
    
    for turn_id in sorted(scores_by_turn.keys()):
        scores = scores_by_turn[turn_id]
        rel = np.mean(scores['relevance'])
        acc = np.mean(scores['accuracy'])
        inf = np.mean(scores['informativeness'])
        coh = np.mean(scores['coherence'])
        avg = np.mean(scores['average_score'])
        report.append(f"| Turn {turn_id} | {rel:.2f} | {acc:.2f} | {inf:.2f} | {coh:.2f} | **{avg:.2f}** |")
    
    report.append("")
    
    # 3. 模型间一致性
    report.append("## 3. Inter-Model Agreement")
    report.append("")
    report.append("### Spearman Correlation Coefficients")
    report.append("")
    
    if correlations:
        # 按维度组织
        dimensions = ['average_score', 'relevance', 'accuracy', 'informativeness', 'coherence']
        for dim in dimensions:
            report.append(f"#### {dim.replace('_', ' ').title()}")
            report.append("")
            report.append("| Model Pair | Spearman's ρ | p-value | Interpretation |")
            report.append("|------------|--------------|---------|----------------|")
            
            for key, corr in correlations.items():
                if key.endswith(f"_{dim}"):
                    pair = key.replace(f"_{dim}", "").replace("_vs_", " vs ")
                    rho = corr['rho']
                    p = corr['p_value']
                    
                    # 解释
                    if abs(rho) > 0.8:
                        interp = "Very Strong"
                    elif abs(rho) > 0.6:
                        interp = "Strong"
                    elif abs(rho) > 0.4:
                        interp = "Moderate"
                    else:
                        interp = "Weak"
                    
                    if p < 0.001:
                        sig = "***"
                    elif p < 0.01:
                        sig = "**"
                    elif p < 0.05:
                        sig = "*"
                    else:
                        sig = "ns"
                    
                    report.append(f"| {pair} | {rho:.3f}{sig} | {p:.4f} | {interp} |")
            
            report.append("")
    else:
        report.append("*需要至少 2 个模型才能计算相关性*")
        report.append("")
    
    # 4. 关键发现
    report.append("## 4. Key Findings")
    report.append("")
    
    # 计算整体平均分
    all_scores = []
    for model_scores in scores_by_model.values():
        all_scores.extend(model_scores['average_score'])
    overall_mean = np.mean(all_scores)
    overall_std = np.std(all_scores)
    
    report.append(f"1. **Overall Quality**: The personalized features achieved an average score of **{overall_mean:.2f} ± {overall_std:.2f}** (out of 5.0).")
    report.append("")
    
    # Turn 差异
    if len(scores_by_turn) >= 2:
        turn1_scores = scores_by_turn.get(1, {}).get('average_score', [])
        turn3_scores = scores_by_turn.get(3, {}).get('average_score', [])
        
        if turn1_scores and turn3_scores:
            turn1_mean = np.mean(turn1_scores)
            turn3_mean = np.mean(turn3_scores)
            improvement = ((turn3_mean - turn1_mean) / turn1_mean) * 100
            
            report.append(f"2. **Turn Evolution**: Scores improved from Turn 1 ({turn1_mean:.2f}) to Turn 3 ({turn3_mean:.2f}), a **{improvement:+.1f}%** increase.")
            report.append("")
    
    # 模型一致性
    if correlations:
        avg_rho_values = [corr['rho'] for key, corr in correlations.items() if key.endswith('_average_score')]
        if avg_rho_values:
            avg_rho = np.mean(avg_rho_values)
            report.append(f"3. **Inter-Model Agreement**: Average Spearman's ρ = **{avg_rho:.3f}**, indicating {'strong' if avg_rho > 0.7 else 'moderate'} agreement.")
            report.append("")
    
    # 最佳维度
    dim_means = {}
    for dim in ['relevance', 'accuracy', 'informativeness', 'coherence']:
        dim_scores = []
        for model_scores in scores_by_model.values():
            dim_scores.extend(model_scores[dim])
        dim_means[dim] = np.mean(dim_scores)
    
    best_dim = max(dim_means, key=dim_means.get)
    worst_dim = min(dim_means, key=dim_means.get)
    
    report.append(f"4. **Dimension Analysis**:")
    report.append(f"   - Strongest dimension: **{best_dim.title()}** ({dim_means[best_dim]:.2f})")
    report.append(f"   - Weakest dimension: **{worst_dim.title()}** ({dim_means[worst_dim]:.2f})")
    report.append("")
    
    # 5. 样本展示
    report.append("## 5. Sample Evaluations")
    report.append("")
    
    # 最高分样本
    best_result = max(results, key=lambda r: np.mean([
        eval['scores']['average_score']
        for eval in r['evaluations'].values()
        if eval.get('success') and eval.get('scores')
    ]))
    
    report.append("### Highest Scoring Example")
    report.append("")
    report.append(f"**Query ID:** `{best_result['query_id']}`")
    report.append(f"**Turn:** {best_result['turn_id']}")
    report.append("")
    report.append(f"**Query:** {best_result['query']}")
    report.append("")
    report.append(f"**Personalized Features:** {best_result['personalized_features']}")
    report.append("")
    report.append("**Scores:**")
    report.append("")
    for model_key, eval in best_result['evaluations'].items():
        if eval.get('success') and eval.get('scores'):
            scores = eval['scores']
            report.append(f"- **{model_key}**: {scores['average_score']:.1f} - {scores['explanation']}")
    report.append("")
    
    # 保存报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"\n✅ 报告已生成: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='分析评估结果')
    
    parser.add_argument('--input', type=str,
                       default=str(config.EVALUATION_RESULTS_FILE),
                       help='输入文件（评估结果）')
    parser.add_argument('--output', type=str,
                       default=str(config.ANALYSIS_REPORT_FILE),
                       help='输出文件（分析报告）')
    
    args = parser.parse_args()
    
    # 加载结果
    print(f"加载结果从: {args.input}")
    results = load_results(Path(args.input))
    print(f"加载 {len(results)} 个评估结果")
    
    # 生成报告
    print("生成报告...")
    generate_report(results, Path(args.output))


if __name__ == "__main__":
    main()

