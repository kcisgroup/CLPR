"""
Utility helpers for memory ablation experiments.
These helpers centralize plan loading, feature filtering, and subset matching.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

import yaml

PLAN_PATH = Path(__file__).with_name("ablation_plan.yaml")

TAG_PREFIXES = {
    "sequential": "[SEQUENTIAL_MEMORY]",
    "working": "[WORKING_MEMORY]",
    "long": "[LONG_EXPLICIT]",
}


def load_plan(plan_path: Path = PLAN_PATH) -> Dict[str, Any]:
    """Load the YAML ablation plan."""
    with open(plan_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict) or "datasets" not in data:
        raise ValueError(f"Invalid ablation plan at {plan_path}")
    return data


def get_dataset_config(plan: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    datasets = plan.get("datasets") or {}
    if dataset_name not in datasets:
        raise KeyError(f"Dataset '{dataset_name}' not defined in ablation_plan.yaml")
    cfg = datasets[dataset_name]
    cfg["name"] = dataset_name
    return cfg


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [tok.lower() for tok in TOKEN_PATTERN.findall(text)]


def jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def strip_tag_prefix(feature: str) -> str:
    """Remove '[TAG]' prefix and trim whitespace."""
    if not feature:
        return ""
    return re.sub(r"^\[[A-Z0-9_]+\]\s*", "", feature).strip()


def classify_feature(feature: str) -> Optional[str]:
    for dim, prefix in TAG_PREFIXES.items():
        if feature.startswith(prefix):
            return dim
    return None


def count_feature_dimensions(features: Sequence[str]) -> Dict[str, int]:
    counts = {dim: 0 for dim in TAG_PREFIXES}
    for feat in features or []:
        dim = classify_feature(feat)
        if dim and dim in counts:
            counts[dim] += 1
    return counts


def filter_features(
    features: Sequence[str], exclude_dimensions: Optional[Sequence[str]] = None
) -> List[str]:
    exclude = set(exclude_dimensions or [])
    cleaned: List[str] = []
    for feat in features or []:
        dim = classify_feature(feat)
        if dim and dim in exclude:
            continue
        cleaned.append(feat)
    return cleaned


def truncate_text(text: str, max_chars: Optional[int]) -> str:
    if not text:
        return ""
    if not max_chars or len(text) <= max_chars:
        return text
    truncated = text[: max_chars - 3].rstrip()
    return truncated + "..."


def infer_topic_id(entry: Dict[str, Any]) -> str:
    topic = entry.get("topic_id")
    if topic:
        return str(topic)
    qid = str(entry.get("query_id", "")).strip()
    if not qid:
        return ""
    parts = qid.split("_")
    if len(parts) > 1:
        return "_".join(parts[:-1])
    return qid


def infer_turn_id(entry: Dict[str, Any], dataset_type: str, default_index: int) -> int:
    turn = entry.get("turn_id")
    if isinstance(turn, int):
        if dataset_type == "single_turn" and turn == 0:
            return 1
        return turn
    try:
        turn_int = int(turn)
        if dataset_type == "single_turn" and turn_int == 0:
            return 1
        return turn_int
    except (TypeError, ValueError):
        return default_index + 1 if dataset_type == "multi_turn" else 1


def match_conditions(record: Dict[str, Any], conditions: Dict[str, Any]) -> bool:
    """Check if a metadata record satisfies a condition dictionary."""
    if not conditions:
        return True

    turn_id = record.get("turn_id", 0)
    history_size = record.get("history_size", 0)
    token_count = record.get("token_count", 0)
    topic_shift_score = record.get("topic_shift_score", 1.0)
    counts = record.get("tag_counts", {})

    def satisfies_threshold(key_prefix: str, get_value):
        min_key = f"min_{key_prefix}"
        max_key = f"max_{key_prefix}"
        if min_key in conditions and get_value() < conditions[min_key]:
            return False
        if max_key in conditions and get_value() > conditions[max_key]:
            return False
        return True

    if not satisfies_threshold("turn", lambda: turn_id):
        return False
    if not satisfies_threshold("history_size", lambda: history_size):
        return False
    if not satisfies_threshold("token_count", lambda: token_count):
        return False
    if not satisfies_threshold("topic_shift_score", lambda: topic_shift_score):
        return False

    for dim in TAG_PREFIXES:
        if not satisfies_threshold(f"{dim}_count", lambda d=dim: counts.get(d, 0)):
            return False

    def flags_as_list(name: str) -> List[str]:
        if name not in conditions:
            return []
        vals = conditions[name]
        if isinstance(vals, str):
            return [vals]
        return list(vals or [])

    def get_flag(flag_name: str) -> bool:
        return bool(record.get(flag_name))

    for flag in flags_as_list("flags_true"):
        if not get_flag(flag):
            return False

    for flag in flags_as_list("flags_false"):
        if get_flag(flag):
            return False

    any_flags = flags_as_list("flags_any")
    if any_flags:
        if not any(get_flag(flag) for flag in any_flags):
            return False

    return True


def format_profile_text(segments: Sequence[str], prefix: str = "Researcher focusing on: ") -> str:
    cleaned = [seg.strip() for seg in segments if seg and seg.strip()]
    if not cleaned:
        return ""
    joined = "; ".join(cleaned)
    return f"{prefix}{joined}"
