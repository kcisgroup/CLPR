#!/usr/bin/env python3
"""计算各模型与 Grok（模拟人工）评分的 MAE"""

import json
from collections import defaultdict

# 加载评估结果
def load_eval(filepath):
    data = {}
    with open(filepath, 'r') as f:
        for line in f:
            d = json.loads(line)
            qid = d['query_id']
            data[qid] = d
    return data

gemini_data = load_eval('evaluation_results_gemini.jsonl')
grok_data = load_eval('evaluation_results_grok.jsonl')
main_data = load_eval('evaluation_results.jsonl')

# 合并所有评估
all_scores = defaultdict(lambda: defaultdict(dict))
dimensions = ['relevance', 'accuracy', 'informativeness', 'coherence']

for qid, d in main_data.items():
    for model, result in d.get('evaluations', {}).items():
        if result.get('success'):
            scores = result.get('scores', {})
            for dim in dimensions:
                if dim in scores:
                    all_scores[qid][model][dim] = scores[dim]

for qid, d in gemini_data.items():
    for model, result in d.get('evaluations', {}).items():
        if result.get('success'):
            scores = result.get('scores', {})
            for dim in dimensions:
                if dim in scores:
                    all_scores[qid]['gemini'][dim] = scores[dim]

for qid, d in grok_data.items():
    for model, result in d.get('evaluations', {}).items():
        if 'grok' in model.lower() and result.get('success'):
            scores = result.get('scores', {})
            for dim in dimensions:
                if dim in scores:
                    all_scores[qid]['grok'][dim] = scores[dim]

# 计算 Grok 平均分
grok_avg = defaultdict(list)
for qid in all_scores:
    if 'grok' in all_scores[qid]:
        for dim in dimensions:
            if dim in all_scores[qid]['grok']:
                grok_avg[dim].append(all_scores[qid]['grok'][dim])

print("=" * 70)
print("Profile Quality Evaluation: LLM Scores vs Grok (Simulated Human)")
print("=" * 70)
print()

# Grok 平均分
print("Grok (Simulated Human) Average Scores:")
grok_means = {}
for dim in dimensions:
    mean = sum(grok_avg[dim]) / len(grok_avg[dim])
    grok_means[dim] = mean
    print(f"  {dim.capitalize():15}: {mean:.2f}")
overall_grok = sum(grok_means.values()) / 4
print(f"  {'Overall':15}: {overall_grok:.2f}")
print()

# 各模型分数和 MAE
models = [
    ('claude-haiku-4.5', 'Claude 4.5 Haiku'),
    ('gemini', 'Gemini 2.5 Pro'),
    ('gpt-4o-mini-hybgzs', 'GPT-4o-mini')
]

print("-" * 70)
print(f"{'Model':<20} {'Dimension':<15} {'Score':>8} {'Grok':>8} {'|Δ|':>8}")
print("-" * 70)

for model_key, model_name in models:
    mae_all = defaultdict(list)
    scores_all = defaultdict(list)
    
    for qid in all_scores:
        if model_key in all_scores[qid] and 'grok' in all_scores[qid]:
            for dim in dimensions:
                if dim in all_scores[qid][model_key] and dim in all_scores[qid]['grok']:
                    m_score = all_scores[qid][model_key][dim]
                    g_score = all_scores[qid]['grok'][dim]
                    scores_all[dim].append(m_score)
                    mae_all[dim].append(abs(m_score - g_score))
    
    for i, dim in enumerate(dimensions):
        if scores_all[dim]:
            avg_score = sum(scores_all[dim]) / len(scores_all[dim])
            avg_grok = grok_means[dim]
            mae = sum(mae_all[dim]) / len(mae_all[dim])
            name = model_name if i == 0 else ""
            print(f"{name:<20} {dim.capitalize():<15} {avg_score:>8.2f} {avg_grok:>8.2f} {mae:>8.2f}")
    
    # Overall
    all_mae = []
    all_model_scores = []
    for dim in dimensions:
        all_mae.extend(mae_all[dim])
        all_model_scores.extend(scores_all[dim])
    
    if all_mae:
        overall_score = sum(all_model_scores) / len(all_model_scores) if all_model_scores else 0
        overall_mae = sum(all_mae) / len(all_mae)
        print(f"{'':<20} {'Overall':<15} {overall_score:>8.2f} {overall_grok:>8.2f} {overall_mae:>8.2f}")
    print()

print("=" * 70)
print("Note: |Δ| = Mean Absolute Error between LLM and Grok scores")
print("      Lower |Δ| indicates better agreement with simulated human evaluation")
print("=" * 70)
