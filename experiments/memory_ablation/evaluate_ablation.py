#!/usr/bin/env python3
"""
Evaluation + analysis pipeline for the redesigned memory ablation experiments.
Computes overall metrics as well as subset-specific performance gaps defined in
ablation_plan.yaml.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Any, Optional

from experiments.memory_ablation.utils import (
    get_dataset_config,
    load_plan,
    match_conditions,
)


def load_metadata_records(metadata_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_ground_truth(dataset_cfg: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    path = Path(dataset_cfg["ground_truth"])
    if not path.exists():
        raise FileNotFoundError(f"Ground truth not found: {path}")

    dataset_type = dataset_cfg.get("type", "multi_turn")
    ground_truth: Dict[str, Dict[str, int]] = defaultdict(dict)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if dataset_type == "multi_turn":
                conv_id = data.get("conversation_id")
                turn_id = data.get("turn_id")
                qid = f"{conv_id}_{turn_id}"
                doc_id = str(data.get("doc_id"))
                rel = int(data.get("rel", 0))
                ground_truth[qid][doc_id] = rel
            else:
                qid = str(data.get("query_id"))
                relevant = data.get("relevant_texts") or []
                for doc_id in relevant:
                    ground_truth[qid][str(doc_id)] = 1
    return ground_truth


def load_rankings(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Ranking file not found: {path}")
    rankings: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            qid = str(data.get("query_id"))
            candidates = (
                data.get("ranked_results")
                or data.get("results")
                or data.get("reranked_results")
            )
            if not candidates:
                continue
            doc_ids = []
            for cand in candidates:
                doc_id = cand.get("text_id") or cand.get("doc_id")
                if doc_id is None:
                    continue
                doc_ids.append(str(doc_id))
            rankings[qid] = doc_ids
    return rankings


def dcg_at_k(relevances: List[int], k: int) -> float:
    from math import log2

    relevances = relevances[:k]
    value = 0.0
    for idx, rel in enumerate(relevances):
        value += (2**rel - 1) / log2(idx + 2)
    return value


def average_precision(relevances: List[int], k: int) -> float:
    score = 0.0
    hits = 0
    for idx, rel in enumerate(relevances[:k]):
        if rel > 0:
            hits += 1
            weight = rel / 2.0 if rel <= 2 else 1.0
            score += (hits / (idx + 1)) * weight
    return score


def evaluate_metrics(
    ground_truth: Dict[str, Dict[str, int]],
    rankings: Dict[str, List[str]],
    query_filter: Optional[Iterable[str]],
    k: int,
) -> Dict[str, float]:
    queries = list(query_filter) if query_filter is not None else list(ground_truth.keys())
    ndcgs, maps, recalls, p1s = [], [], [], []
    covered = 0

    for qid in queries:
        gt = ground_truth.get(qid)
        pred = rankings.get(qid)
        if not gt or not pred:
            continue
        relevances = [gt.get(doc, 0) for doc in pred[:k]]
        ideal = sorted(gt.values(), reverse=True)
        idcg = dcg_at_k(ideal, k)
        ndcg = dcg_at_k(relevances, k) / idcg if idcg > 0 else 0.0
        ap = average_precision(relevances, k)

        total_relevant = sum(1 for rel in gt.values() if rel > 0)
        hits = sum(1 for rel in relevances if rel > 0)
        recall = hits / total_relevant if total_relevant else 0.0
        p1 = 1.0 if relevances and relevances[0] > 0 else 0.0

        ndcgs.append(ndcg)
        maps.append(ap)
        recalls.append(recall)
        p1s.append(p1)
        covered += 1

    def safe_mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "query_count": covered,
        f"ndcg@{k}": safe_mean(ndcgs),
        f"map@{k}": safe_mean(maps),
        f"recall@{k}": safe_mean(recalls),
        "p@1": safe_mean(p1s),
    }


def precompute_subsets(
    dataset_cfg: Dict[str, Any], metadata_records: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    subset_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for subset in dataset_cfg.get("subsets", []):
        rules_out: Dict[str, Dict[str, Any]] = {}
        for rule in subset.get("rules", []):
            label = rule.get("label")
            conditions = rule.get("conditions") or {}
            matching_qids = [
                rec["query_id"] for rec in metadata_records if match_conditions(rec, conditions)
            ]
            rules_out[label] = {
                "conditions": conditions,
                "query_ids": matching_qids,
                "count": len(matching_qids),
            }
        subset_map[subset["id"]] = {
            "description": subset.get("description", ""),
            "rules": rules_out,
        }
    return subset_map


def evaluate_dataset(
    dataset_name: str,
    dataset_cfg: Dict[str, Any],
    variants: List[Dict[str, Any]],
    k: int,
) -> Dict[str, Any]:
    metadata_records = load_metadata_records(Path(dataset_cfg["metadata_cache"]))
    subset_map = precompute_subsets(dataset_cfg, metadata_records)
    ground_truth = load_ground_truth(dataset_cfg)

    results_out: Dict[str, Any] = {
        "dataset": dataset_name,
        "k": k,
        "overall": {},
        "subsets": subset_map,
    }

    for variant in variants:
        variant_id = variant["id"]
        ranked_path = Path(variant["ranked_file"])
        if not ranked_path.exists():
            print(f"[eval] Missing ranking file for {dataset_name}/{variant_id}: {ranked_path}")
            continue
        rankings = load_rankings(ranked_path)
        overall_metrics = evaluate_metrics(ground_truth, rankings, None, k)
        results_out["overall"][variant_id] = {
            "metrics": overall_metrics,
            "ranked_file": str(ranked_path),
        }

        for subset_id, subset_data in subset_map.items():
            subset_results = results_out.setdefault("subset_results", {}).setdefault(
                subset_id, {}
            )
            for rule_label, rule_data in subset_data["rules"].items():
                qids = rule_data["query_ids"]
                metrics = evaluate_metrics(ground_truth, rankings, qids, k)
                subset_results.setdefault(rule_label, {})[variant_id] = metrics
                subset_results[rule_label]["query_count"] = len(qids)

    return results_out


def main() -> None:
    plan = load_plan()
    dataset_names = sorted(plan.get("datasets", {}).keys())

    parser = argparse.ArgumentParser(description="Evaluate memory ablation variants.")
    parser.add_argument(
        "--dataset",
        action="append",
        choices=dataset_names,
        help="Datasets to evaluate (default: all).",
    )
    parser.add_argument(
        "--variant",
        action="append",
        help="Restrict evaluation to specific variant IDs.",
    )
    parser.add_argument("--k", type=int, default=10, help="Truncation threshold.")
    parser.add_argument(
        "--output-dir",
        default="experiments/memory_ablation/analysis",
        help="Directory for saving JSON reports.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_datasets = args.dataset or dataset_names

    for dataset_name in target_datasets:
        dataset_cfg = get_dataset_config(plan, dataset_name)
        variants = dataset_cfg.get("variants", [])
        if args.variant:
            variants = [v for v in variants if v.get("id") in set(args.variant)]
        if not variants:
            print(f"[eval] No variants to evaluate for {dataset_name}")
            continue
        result = evaluate_dataset(dataset_name, dataset_cfg, variants, args.k)
        out_path = output_dir / f"{dataset_name}_ablation_eval.json"
        with open(out_path, "w", encoding="utf-8") as f_out:
            json.dump(result, f_out, ensure_ascii=False, indent=2)
        print(f"[eval] Saved {dataset_name} summary -> {out_path}")


if __name__ == "__main__":
    main()
