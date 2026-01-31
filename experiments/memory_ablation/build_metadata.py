#!/usr/bin/env python3
"""
Constructs metadata caches used by the new memory ablation pipeline.
Each metadata line contains derived attributes (turn depth, token counts,
contextual flags, history queries) so downstream scripts can slice data
without reparsing the large cognitive feature files.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any

from experiments.memory_ablation.utils import (
    TAG_PREFIXES,
    count_feature_dimensions,
    ensure_parent_dir,
    get_dataset_config,
    infer_topic_id,
    infer_turn_id,
    jaccard_similarity,
    load_plan,
    tokenize,
)


def determine_conversation_id(entry: Dict[str, Any], dataset_type: str) -> str:
    if dataset_type == "multi_turn":
        return infer_topic_id(entry)
    topic = entry.get("topic_id")
    if topic not in (None, ""):
        return str(topic)
    return str(entry.get("query_id"))


def build_metadata_for_dataset(dataset_name: str, cfg: Dict[str, Any]) -> Path:
    data_path = Path(cfg["cognitive_features"])
    if not data_path.exists():
        raise FileNotFoundError(f"Cognitive feature file missing: {data_path}")

    dataset_type = cfg.get("type", "multi_turn")
    topic_shift_threshold = cfg.get("topic_shift_threshold", 0.3)
    metadata_path = Path(cfg["metadata_cache"])
    ensure_parent_dir(metadata_path)

    entries_by_conv: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(data_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            record["_line_idx"] = idx
            conv_id = determine_conversation_id(record, dataset_type)
            entries_by_conv[conv_id].append(record)

    metadata_records: List[Dict[str, Any]] = []
    stats_flags = Counter()

    for conv_id, items in sorted(entries_by_conv.items()):
        # Sort by explicit turn id if available, otherwise preserve file order.
        items_sorted = sorted(
            items,
            key=lambda item: (
                infer_turn_id(item, dataset_type, item.get("_line_idx", 0)),
                item.get("_line_idx", 0),
            ),
        )
        history_queries: List[str] = []
        history_ids: List[str] = []

        for idx, entry in enumerate(items_sorted):
            qid = str(entry.get("query_id"))
            query = entry.get("query", "").strip()
            topic_id = infer_topic_id(entry)
            turn_id = infer_turn_id(entry, dataset_type, idx)
            tokens = tokenize(query)

            tagged_features = entry.get("tagged_memory_features") or []
            tag_counts = count_feature_dimensions(tagged_features)

            reference_info = entry.get("reference_info") or {}
            resolved_refs = entry.get("resolved_references") or {}
            has_pronoun_reference = bool(
                reference_info.get("pronouns")
                or resolved_refs.get("pronoun_resolutions")
            )
            has_connector_reference = bool(
                reference_info.get("connectors")
                or resolved_refs.get("connector_context")
            )

            topic_shift_score = 1.0
            topic_shift_flag = False
            if history_queries:
                overlaps = [
                    jaccard_similarity(tokens, tokenize(prev_q))
                    for prev_q in history_queries
                ]
                topic_shift_score = max(overlaps) if overlaps else 0.0
                topic_shift_flag = topic_shift_score < topic_shift_threshold

            metadata_record = {
                "dataset": dataset_name,
                "query_id": qid,
                "query": query,
                "conversation_id": conv_id,
                "topic_id": topic_id,
                "turn_id": turn_id,
                "history_queries": list(history_queries),
                "history_query_ids": list(history_ids),
                "history_size": len(history_queries),
                "token_count": len(tokens),
                "tagged_memory_features": tagged_features,
                "tag_counts": tag_counts,
                "has_pronoun_reference": has_pronoun_reference,
                "has_connector_reference": has_connector_reference,
                "topic_shift_score": topic_shift_score,
                "topic_shift": topic_shift_flag,
            }

            metadata_records.append(metadata_record)

            stats_flags.update(
                {
                    "total_queries": 1,
                    **{f"{dim}_count": tag_counts.get(dim, 0) for dim in TAG_PREFIXES},
                    "has_pronoun_reference": int(has_pronoun_reference),
                    "has_connector_reference": int(has_connector_reference),
                    "topic_shift": int(topic_shift_flag),
                }
            )

            history_queries.append(query)
            history_ids.append(qid)

    with open(metadata_path, "w", encoding="utf-8") as f_out:
        for record in metadata_records:
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[metadata] {dataset_name}: wrote {len(metadata_records)} records -> {metadata_path}"
    )
    print(
        f"           pronoun_ref={stats_flags['has_pronoun_reference']} "
        f"connector_ref={stats_flags['has_connector_reference']} "
        f"topic_shift={stats_flags['topic_shift']}"
    )
    return metadata_path


def main() -> None:
    plan = load_plan()
    datasets = plan.get("datasets", {})
    dataset_names_all = sorted(datasets.keys())

    parser = argparse.ArgumentParser(description="Build metadata caches for memory ablations.")
    parser.add_argument(
        "--dataset",
        choices=dataset_names_all,
        help="Process a single dataset (default: all configured datasets).",
    )
    args = parser.parse_args()

    if args.dataset:
        dataset_names = [args.dataset]
    else:
        dataset_names = dataset_names_all

    for name in dataset_names:
        cfg = get_dataset_config(plan, name)
        try:
            build_metadata_for_dataset(name, cfg)
        except Exception as exc:
            print(f"[metadata] Failed for {name}: {exc}")
            raise


if __name__ == "__main__":
    main()
