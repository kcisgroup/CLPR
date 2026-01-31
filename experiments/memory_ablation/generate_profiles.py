#!/usr/bin/env python3
"""
Profile generator for the redesigned memory ablation experiments.
Reads pre-built metadata caches and emits deterministic personalized text
according to the configuration in ablation_plan.yaml.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

from experiments.memory_ablation.utils import (
    filter_features,
    format_profile_text,
    get_dataset_config,
    load_plan,
    match_conditions,
    strip_tag_prefix,
    truncate_text,
)


def load_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata cache missing: {metadata_path}")
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


def structured_profile(record: Dict[str, Any], gen_cfg: Dict[str, Any]) -> Tuple[str, List[str]]:
    exclude_dims = gen_cfg.get("exclude_dimensions") or []
    filtered = filter_features(record.get("tagged_memory_features") or [], exclude_dims)
    segments = [strip_tag_prefix(ft) for ft in filtered]

    profile_text = format_profile_text(
        segments,
        prefix=gen_cfg.get("prefix", "Researcher focusing on: "),
    )

    if not profile_text and gen_cfg.get("include_query_fallback", True):
        profile_text = gen_cfg.get("fallback_template", "Researcher investigating: {query}").format(
            query=record.get("query", "")
        )

    if gen_cfg.get("include_history_hint") and record.get("history_queries"):
        recent = record["history_queries"][-1]
        profile_text = f"{profile_text} || Recent turn: {recent}"

    max_chars = gen_cfg.get("max_chars")
    profile_text = truncate_text(profile_text, max_chars)

    if not filtered and gen_cfg.get("replace_empty_with_query", False):
        profile_text = truncate_text(record.get("query", ""), max_chars)

    return profile_text, filtered


def concat_history_profile(record: Dict[str, Any], gen_cfg: Dict[str, Any]) -> Tuple[str, List[str]]:
    history_turns = gen_cfg.get("history_turns", 3)
    delimiter = gen_cfg.get("delimiter", " || ")
    include_query = gen_cfg.get("include_query", True)

    history = record.get("history_queries") or []
    selected_history = history[-history_turns:]
    pieces = []

    total = len(selected_history)
    for idx, text in enumerate(selected_history):
        label = total - idx
        pieces.append(f"[Prev-{label}] {text}")

    if include_query or not pieces:
        pieces.append(f"[Current] {record.get('query', '')}")

    text = delimiter.join(pieces)
    text = truncate_text(text, gen_cfg.get("max_chars"))
    return text, record.get("tagged_memory_features") or []


def query_only_profile(record: Dict[str, Any], gen_cfg: Dict[str, Any]) -> Tuple[str, List[str]]:
    max_chars = gen_cfg.get("max_chars")
    text = truncate_text(record.get("query", ""), max_chars)
    return text, []


PROFILE_BUILDERS = {
    "structured": structured_profile,
    "concat_history": concat_history_profile,
    "query_only": query_only_profile,
}


def generate_variant_profiles(
    dataset_name: str,
    metadata_records: List[Dict[str, Any]],
    variant_cfg: Dict[str, Any],
) -> None:
    generation_cfg = variant_cfg.get("generation") or {}
    gen_type = generation_cfg.get("type")

    if gen_type == "passthrough" or not gen_type:
        print(f"[profiles] {dataset_name}/{variant_cfg['id']} uses passthrough generation. Skipping.")
        return

    builder = PROFILE_BUILDERS.get(gen_type)
    if builder is None:
        raise ValueError(f"Unsupported generation type '{gen_type}' for variant {variant_cfg['id']}")

    output_path = Path(variant_cfg["profile_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subset_conditions = generation_cfg.get("metadata_filter")

    written = 0
    with open(output_path, "w", encoding="utf-8") as f_out:
        for record in metadata_records:
            if subset_conditions and not match_conditions(record, subset_conditions):
                continue
            profile_text, filtered_features = builder(record, generation_cfg)
            payload = {
                "query_id": record["query_id"],
                "query": record.get("query"),
                "personalized_features": profile_text,
                "tagged_memory_features": filtered_features,
                "topic_id": record.get("topic_id"),
                "turn_id": record.get("turn_id"),
                "profile_variant": variant_cfg["id"],
                "history_size": record.get("history_size"),
                "generation_type": gen_type,
            }
            f_out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"[profiles] {dataset_name}/{variant_cfg['id']} wrote {written} entries -> {output_path}"
    )


def main() -> None:
    plan = load_plan()
    dataset_names = sorted(plan.get("datasets", {}).keys())

    parser = argparse.ArgumentParser(description="Generate deterministic ablation profiles.")
    parser.add_argument("--dataset", required=True, choices=dataset_names)
    parser.add_argument(
        "--variant",
        action="append",
        help="Variant ID(s) to regenerate (default: all variants with generation rules).",
    )
    args = parser.parse_args()

    dataset_cfg = get_dataset_config(plan, args.dataset)
    metadata_path = Path(dataset_cfg["metadata_cache"])
    metadata_records = load_metadata(metadata_path)

    variants = dataset_cfg.get("variants", [])
    if args.variant:
        target_variants = [v for v in variants if v.get("id") in set(args.variant)]
    else:
        target_variants = [
            v for v in variants if v.get("generation", {}).get("type") not in (None, "passthrough")
        ]

    if not target_variants:
        print("[profiles] Nothing to generate.")
        return

    for variant in target_variants:
        generate_variant_profiles(args.dataset, metadata_records, variant)


if __name__ == "__main__":
    main()
