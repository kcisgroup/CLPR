#!/usr/bin/env python3
"""
Simplified failure case analysis for personalized reranking.

直接从现有 rerank 结果 vs ground truth 对比，筛选出典型失败案例，
输出 JSONL + Markdown 报告供人工审阅。

不需要 LLM 重新分析/重排，只做数据筛选和格式化。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import config

PROJECT_ROOT = config.PROJECT_ROOT
EXPERIMENT_DIR = config.EXPERIMENTS_DIR

# 默认结果文件路径
DEFAULT_RESULTS = {
    "litsearch": {
        "baseline": PROJECT_ROOT / "results" / "LitSearch" / "ranked_jina-v3_query-only_top10.jsonl",
        "personalized": PROJECT_ROOT / "results" / "LitSearch" / "ranked_jina-v3_profile-and-query_top10.jsonl",
        "ground_truth": PROJECT_ROOT / "data" / "LitSearch" / "query_to_texts.jsonl",
        "queries": PROJECT_ROOT / "data" / "LitSearch" / "queries.jsonl",
    },
    "medcorpus": {
        "baseline": PROJECT_ROOT / "results" / "MedCorpus" / "ranked_jina-v3_query-only_top10.jsonl",
        "personalized": PROJECT_ROOT / "results" / "MedCorpus" / "ranked_jina-v3_profile-and-query_top10.jsonl",
        "ground_truth": PROJECT_ROOT / "data" / "MedCorpus_MultiTurn" / "query_to_texts.jsonl",
        "queries": PROJECT_ROOT / "data" / "MedCorpus_MultiTurn" / "queries.jsonl",
    },
}

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_results_map(path: Path) -> Dict[str, Dict[str, Any]]:
    """加载 rerank 结果，返回 query_id -> record 映射"""
    data = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            query_id = str(record.get("query_id") or record.get("conversation_id"))
            data[query_id] = record
    return data


def load_litsearch_ground_truth(path: Path) -> Dict[str, Dict[str, int]]:
    """LitSearch: query_id -> {doc_id: rel}"""
    ground_truth = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            query_id = str(row["query_id"])
            rels = {str(doc_id): 1 for doc_id in row["relevant_texts"]}
            ground_truth[query_id] = rels
    return ground_truth


def load_medcorpus_ground_truth(path: Path) -> Dict[str, Dict[str, int]]:
    """MedCorpus: query_id -> {doc_id: rel}"""
    ground_truth = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = f"{row['conversation_id']}_{row['turn_id']}"
            ground_truth.setdefault(qid, {})
            ground_truth[qid][row["doc_id"]] = row["rel"]
    return ground_truth


def load_samples(path: Path) -> List[Dict[str, Any]]:
    """加载采样的 100 个样本"""
    return load_jsonl(path)


# ---------------------------------------------------------------------------
# 指标计算
# ---------------------------------------------------------------------------

def precision_at_1(ranked_results: List[Dict[str, Any]], rels: Dict[str, int]) -> float:
    if not ranked_results:
        return 0.0
    top_doc = ranked_results[0]
    doc_id = str(top_doc.get("text_id") or top_doc.get("doc_id"))
    rel = rels.get(doc_id, 0)
    return 1.0 if rel > 0 else 0.0


def ndcg_at_k(ranked_results: List[Dict[str, Any]], rels: Dict[str, int], k: int = 10) -> float:
    if not ranked_results or not rels:
        return 0.0
    
    dcg = 0.0
    for idx, doc in enumerate(ranked_results[:k]):
        doc_id = str(doc.get("text_id") or doc.get("doc_id"))
        rel = rels.get(doc_id, 0)
        if rel > 0:
            dcg += (2 ** rel - 1) / math.log2(idx + 2)
    
    ideal_rels = sorted(rels.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(ideal_rels) if rel > 0)
    
    return dcg / idcg if idcg > 0 else 0.0


def mrr(ranked_results: List[Dict[str, Any]], rels: Dict[str, int], k: int = 10) -> float:
    """Mean Reciprocal Rank"""
    for idx, doc in enumerate(ranked_results[:k], start=1):
        doc_id = str(doc.get("text_id") or doc.get("doc_id"))
        if rels.get(doc_id, 0) > 0:
            return 1.0 / idx
    return 0.0


def compute_metrics(ranked_results: List[Dict[str, Any]], rels: Dict[str, int]) -> Dict[str, float]:
    return {
        "p@1": precision_at_1(ranked_results, rels),
        "ndcg@10": ndcg_at_k(ranked_results, rels, k=10),
        "mrr@10": mrr(ranked_results, rels, k=10),
    }


# ---------------------------------------------------------------------------
# 失败案例筛选
# ---------------------------------------------------------------------------

def classify_failure_type(baseline_m: Dict, personalized_m: Dict, rels: Dict) -> str:
    """分类失败类型"""
    delta_p1 = personalized_m["p@1"] - baseline_m["p@1"]
    delta_ndcg = personalized_m["ndcg@10"] - baseline_m["ndcg@10"]
    
    if personalized_m["p@1"] == 0 and baseline_m["p@1"] == 1:
        return "top1_regression"  # 个性化把正确的第一位推下去了
    elif personalized_m["p@1"] == 0 and baseline_m["p@1"] == 0:
        if delta_ndcg < -0.1:
            return "both_miss_worse"  # 两者都没命中top1，但个性化更差
        else:
            return "both_miss"  # 两者都没命中top1
    elif delta_ndcg < -0.1:
        return "ndcg_regression"  # NDCG 明显下降
    elif personalized_m["mrr@10"] < baseline_m["mrr@10"]:
        return "mrr_regression"  # MRR 下降
    else:
        return "other"


def find_failure_cases(
    samples: List[Dict[str, Any]],
    baseline_results: Dict[str, Dict],
    personalized_results: Dict[str, Dict],
    ground_truth: Dict[str, Dict[str, int]],
    dataset_key: str,
) -> List[Dict[str, Any]]:
    """
    找出失败案例：个性化 rerank 比 baseline 差的样本
    """
    failures = []
    
    for sample in samples:
        if sample["dataset"].lower() != dataset_key:
            continue
        
        query_id = str(sample["query_id"])
        rels = ground_truth.get(query_id, {})
        
        baseline_entry = baseline_results.get(query_id)
        personalized_entry = personalized_results.get(query_id)
        
        if not baseline_entry or not personalized_entry:
            continue
        
        baseline_ranked = baseline_entry.get("ranked_results", [])
        personalized_ranked = personalized_entry.get("ranked_results", [])
        
        baseline_m = compute_metrics(baseline_ranked, rels)
        personalized_m = compute_metrics(personalized_ranked, rels)
        
        # 判断是否失败：个性化比baseline差
        delta_p1 = personalized_m["p@1"] - baseline_m["p@1"]
        delta_ndcg = personalized_m["ndcg@10"] - baseline_m["ndcg@10"]
        delta_mrr = personalized_m["mrr@10"] - baseline_m["mrr@10"]
        
        is_failure = (delta_p1 < 0) or (delta_ndcg < -0.05) or (delta_mrr < -0.1)
        
        if not is_failure:
            continue
        
        failure_type = classify_failure_type(baseline_m, personalized_m, rels)
        
        # 提取 top-5 文档信息
        def extract_top_docs(ranked: List, max_n: int = 5) -> List[Dict]:
            docs = []
            for i, doc in enumerate(ranked[:max_n]):
                doc_id = str(doc.get("text_id") or doc.get("doc_id"))
                rel = rels.get(doc_id, 0)
                snippet = (doc.get("text") or doc.get("title") or "")[:200]
                docs.append({
                    "rank": i + 1,
                    "doc_id": doc_id,
                    "relevance": rel,
                    "is_relevant": rel > 0,
                    "snippet": snippet.replace("\n", " ").strip(),
                })
            return docs
        
        failures.append({
            "dataset": sample["dataset"],
            "query_id": query_id,
            "turn_id": sample.get("turn_id"),
            "conversation_id": sample.get("conversation_id"),
            "query": sample.get("query"),
            "profile": sample.get("personalized_features"),
            "history": sample.get("history", []),
            "failure_type": failure_type,
            "metrics": {
                "baseline": baseline_m,
                "personalized": personalized_m,
                "delta": {
                    "p@1": delta_p1,
                    "ndcg@10": delta_ndcg,
                    "mrr@10": delta_mrr,
                },
            },
            "baseline_top5": extract_top_docs(baseline_ranked),
            "personalized_top5": extract_top_docs(personalized_ranked),
            "ground_truth_docs": [
                {"doc_id": doc_id, "rel": rel}
                for doc_id, rel in sorted(rels.items(), key=lambda x: -x[1])[:10]
            ],
            # 计算严重程度分数，用于排序
            "severity_score": abs(delta_p1) * 2 + abs(delta_ndcg) + abs(delta_mrr) * 0.5,
        })
    
    # 按严重程度排序
    failures.sort(key=lambda x: -x["severity_score"])
    return failures


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def generate_markdown_report(
    failures: List[Dict[str, Any]],
    output_path: Path,
    top_n: int = 20,
) -> None:
    """生成 Markdown 格式的失败案例报告"""
    
    lines = [
        "# Failure Case Analysis Report",
        "",
        f"**Total failure cases found:** {len(failures)}",
        f"**Top {min(top_n, len(failures))} cases shown below (sorted by severity)**",
        "",
        "## Summary by Failure Type",
        "",
    ]
    
    # 统计失败类型
    type_counts = defaultdict(int)
    for f in failures:
        type_counts[f["failure_type"]] += 1
    
    for ftype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- **{ftype}**: {count} cases")
    
    lines.extend(["", "---", ""])
    
    # 详细案例
    for i, case in enumerate(failures[:top_n], 1):
        lines.extend([
            f"## Case {i}: {case['dataset']} - Query {case['query_id']}",
            "",
            f"**Failure Type:** `{case['failure_type']}`",
            f"**Severity Score:** {case['severity_score']:.3f}",
            "",
            "### Query",
            f"> {case['query']}",
            "",
            "### Profile",
            f"> {case['profile'] or '(no profile)'}",
            "",
        ])
        
        if case["history"]:
            lines.append("### Conversation History")
            for h in case["history"]:
                lines.append(f"- {h[:100]}...")
            lines.append("")
        
        # 指标对比
        m = case["metrics"]
        lines.extend([
            "### Metrics Comparison",
            "",
            "| Metric | Baseline | Personalized | Delta |",
            "|--------|----------|--------------|-------|",
            f"| P@1 | {m['baseline']['p@1']:.2f} | {m['personalized']['p@1']:.2f} | {m['delta']['p@1']:+.2f} |",
            f"| NDCG@10 | {m['baseline']['ndcg@10']:.3f} | {m['personalized']['ndcg@10']:.3f} | {m['delta']['ndcg@10']:+.3f} |",
            f"| MRR@10 | {m['baseline']['mrr@10']:.3f} | {m['personalized']['mrr@10']:.3f} | {m['delta']['mrr@10']:+.3f} |",
            "",
        ])
        
        # Top-5 文档对比
        lines.extend([
            "### Top-5 Documents Comparison",
            "",
            "**Baseline (Query-only):**",
            "",
        ])
        for doc in case["baseline_top5"]:
            rel_mark = "✓" if doc["is_relevant"] else "✗"
            lines.append(f"{doc['rank']}. [{rel_mark}] `{doc['doc_id']}` - {doc['snippet'][:80]}...")
        
        lines.extend([
            "",
            "**Personalized (Profile+Query):**",
            "",
        ])
        for doc in case["personalized_top5"]:
            rel_mark = "✓" if doc["is_relevant"] else "✗"
            lines.append(f"{doc['rank']}. [{rel_mark}] `{doc['doc_id']}` - {doc['snippet'][:80]}...")
        
        # Ground Truth
        lines.extend([
            "",
            "### Ground Truth Relevant Documents",
            "",
        ])
        for gt in case["ground_truth_docs"][:5]:
            lines.append(f"- `{gt['doc_id']}` (rel={gt['rel']})")
        
        lines.extend(["", "---", ""])
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Markdown report saved: {output_path}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Simplified failure case analysis")
    parser.add_argument("--samples-file", type=str, 
                        default=str(config.SAMPLED_QUERIES_FILE),
                        help="Path to sampled queries JSONL")
    parser.add_argument("--output-jsonl", type=str,
                        default=str(EXPERIMENT_DIR / "failure_cases.jsonl"),
                        help="Output JSONL file for failure cases")
    parser.add_argument("--output-md", type=str,
                        default=str(EXPERIMENT_DIR / "FAILURE_CASE_REPORT.md"),
                        help="Output Markdown report")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Number of top cases to show in report")
    
    # 可选：覆盖默认结果文件
    parser.add_argument("--litsearch-baseline", type=str, default="")
    parser.add_argument("--litsearch-personalized", type=str, default="")
    parser.add_argument("--medcorpus-baseline", type=str, default="")
    parser.add_argument("--medcorpus-personalized", type=str, default="")
    
    args = parser.parse_args()
    
    # 加载采样样本
    samples_path = Path(args.samples_file)
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")
    
    samples = load_samples(samples_path)
    print(f"Loaded {len(samples)} samples")
    
    all_failures = []
    
    # 处理 LitSearch
    print("\n[LitSearch] Loading data...")
    ls_baseline = Path(args.litsearch_baseline) if args.litsearch_baseline else DEFAULT_RESULTS["litsearch"]["baseline"]
    ls_personalized = Path(args.litsearch_personalized) if args.litsearch_personalized else DEFAULT_RESULTS["litsearch"]["personalized"]
    
    if ls_baseline.exists() and ls_personalized.exists():
        ls_baseline_results = load_results_map(ls_baseline)
        ls_personalized_results = load_results_map(ls_personalized)
        ls_ground_truth = load_litsearch_ground_truth(DEFAULT_RESULTS["litsearch"]["ground_truth"])
        
        ls_failures = find_failure_cases(
            samples, ls_baseline_results, ls_personalized_results, ls_ground_truth, "litsearch"
        )
        print(f"[LitSearch] Found {len(ls_failures)} failure cases")
        all_failures.extend(ls_failures)
    else:
        print(f"[LitSearch] Skipped (files not found)")
    
    # 处理 MedCorpus
    print("\n[MedCorpus] Loading data...")
    mc_baseline = Path(args.medcorpus_baseline) if args.medcorpus_baseline else DEFAULT_RESULTS["medcorpus"]["baseline"]
    mc_personalized = Path(args.medcorpus_personalized) if args.medcorpus_personalized else DEFAULT_RESULTS["medcorpus"]["personalized"]
    
    if mc_baseline.exists() and mc_personalized.exists():
        mc_baseline_results = load_results_map(mc_baseline)
        mc_personalized_results = load_results_map(mc_personalized)
        mc_ground_truth = load_medcorpus_ground_truth(DEFAULT_RESULTS["medcorpus"]["ground_truth"])
        
        mc_failures = find_failure_cases(
            samples, mc_baseline_results, mc_personalized_results, mc_ground_truth, "medcorpus"
        )
        print(f"[MedCorpus] Found {len(mc_failures)} failure cases")
        all_failures.extend(mc_failures)
    else:
        print(f"[MedCorpus] Skipped (files not found)")
    
    # 按严重程度排序
    all_failures.sort(key=lambda x: -x["severity_score"])
    
    # 保存 JSONL
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for case in all_failures:
            json.dump(case, f, ensure_ascii=False)
            f.write("\n")
    print(f"\nJSONL saved: {output_jsonl} ({len(all_failures)} cases)")
    
    # 生成 Markdown 报告
    output_md = Path(args.output_md)
    generate_markdown_report(all_failures, output_md, top_n=args.top_n)
    
    # 打印摘要
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples analyzed: {len(samples)}")
    print(f"Total failure cases: {len(all_failures)}")
    
    type_counts = defaultdict(int)
    for f in all_failures:
        type_counts[f["failure_type"]] += 1
    
    print("\nBy failure type:")
    for ftype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  - {ftype}: {count}")


if __name__ == "__main__":
    main()
