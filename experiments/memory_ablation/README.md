# Memory Ablation Experiments (v2)

The old ad‑hoc scripts have been replaced by a config-driven pipeline that
supports MedCorpus (multi-turn) and LitSearch (single-turn) with consistent
metadata, profile generation, and evaluation logic.

## 🔧 Key Components

```
memory_ablation/
├── ablation_plan.yaml           # Single source of truth for datasets, variants, subsets
├── utils.py                     # Shared helpers (plan loading, feature filtering, subsets)
├── build_metadata.py            # Derive per-query metadata caches
├── generate_profiles.py         # Deterministic profile generator driven by the plan
├── evaluate_ablation.py         # Unified evaluator + subset analysis
├── cache/                       # Auto-generated metadata jsonl files
├── analysis/                    # Evaluation summaries (JSON / plots)
├── no_long/, no_sequential/, ...# Variant-specific assets (profiles + rerank outputs)
└── (legacy rerank outputs stay inside each variant folder)
```

## 📄 Ablation Plan (`ablation_plan.yaml`)

- Defines every dataset, where to load cognitive features / ground truth, and where
  to cache metadata.
- Lists every variant (full model, no_long, concat_history, query_only, etc.) with:
  - deterministic profile generation strategy (`structured`, `concat_history`,
    `query_only`, `passthrough`);
  - file locations for generated profiles and ranked outputs.
- Declares **subset rules** (per-turn buckets, dependency classes, topic-shift,
  query-length bins, cue-density buckets, …) so that evaluation can slice metrics
  without editing code.

Updating the plan is the only step needed to add new datasets or experimental arms.

## 🚀 Pipeline

1. **Build metadata (once per dataset)**  
   ```bash
   # All datasets
   python experiments/memory_ablation/build_metadata.py
   # or a specific one
   python experiments/memory_ablation/build_metadata.py --dataset MedCorpus
   ```
   Produces `cache/<Dataset>_metadata.jsonl` capturing turn depth, history context,
   pronoun/connector flags, topic-shift signals, token counts, and tagged feature
   statistics. These caches are reused by the profile generator + evaluator.

2. **Generate variant profiles**  
   ```bash
   # Regenerate every variant with a non-passthrough configuration
   python experiments/memory_ablation/generate_profiles.py --dataset MedCorpus

   # Only refresh the sequential-ablation text
   python experiments/memory_ablation/generate_profiles.py \
       --dataset MedCorpus --variant no_sequential
   ```
   Deterministic modes:
   - `structured`: filter tagged features (e.g., drop `[LONG_EXPLICIT]`) and
     compress them into a short “Researcher focusing on …” profile (with optional
     query fallback);
   - `concat_history`: naïve concatenation of the last *k* queries as a baseline;
   - `query_only`: treat the current query as the profile text;
   - `passthrough`: reuse an existing LLM-generated profile file (e.g., the full
     CLPR system).

3. **Run reranking**  
   Use the existing `core/run.py --mode simple_profile_rerank ...` command per
   variant, pointing `--personalized_queries_path` to the regenerated file. The
   produced rankings live inside each variant folder (see plan).

4. **Evaluate + slice metrics**  
   ```bash
   python experiments/memory_ablation/evaluate_ablation.py
   python experiments/memory_ablation/evaluate_ablation.py \
       --dataset MedCorpus --variant no_long --k 10
   ```
   The evaluator:
   - Loads the ground truth format for each dataset (graded MedCorpus vs. binary
     LitSearch) and computes NDCG@k, MAP@k, Recall@k, P@1.
   - Generates subset dashboards based on the plan (e.g., Turn≥3, pronoun/connector
     queries, topic-shift cases, LitSearch query-length tiers, cue-density buckets).
   - Writes JSON summaries under `analysis/<Dataset>_ablation_eval.json`.

## 📊 Variants Currently Defined

| Dataset   | Variant ID        | Description (from plan)                                   |
|-----------|------------------|-----------------------------------------------------------|
| MedCorpus | `full_memory`    | Original CLPR profiles (passthrough)                      |
|           | `no_long`        | Remove background cues (structured)                       |
|           | `no_sequential`  | Remove sequential cues (structured)                       |
|           | `no_working`     | Remove focus cues (structured)                            |
|           | `concat_history` | Simple concatenation baseline (planned)                   |
|           | `query_only`     | Query-only control (planned)                              |
| LitSearch | `full_memory`    | Original CLPR profiles                                    |
|           | `no_long`        | Drop background cues                                      |
|           | `no_sequential`  | Drop sequential cues (mostly no-op but kept for parity)   |
|           | `no_working`     | Drop focus cues                                           |
|           | `background_only`| Only keep `[LONG_EXPLICIT]` tags (planned)                |
|           | `working_only`   | Only keep `[WORKING_MEMORY]` cues (planned)               |
|           | `query_only`     | Query stub                                                |

Use `ablation_plan.yaml` to add more structured heuristics or control baselines.

## 📁 Variant Directories

Each variant folder retains its historical rerank outputs. Example (`no_long/MedCorpus`):

```
personalized_queries_qwen3-32b.jsonl   # structured profile text
ranked_jina_profile-only_qwen3-32b_top10.jsonl
retrieved.jsonl -> ../../results/MedCorpus/retrieved.jsonl
```

LitSearch mirrors the same layout.

## 📌 Tips

- The metadata cache stores the entire history trace per query; deterministic
  generators can therefore implement new heuristics (topic-shift-aware text,
  pronoun-clarified summaries) without re-parsing the cognitive logs.
- Subset conditions accept `min/max_turn`, `min/max_history_size`, token-length
  bounds, `flags_true/flags_false/flags_any`, and per-dimension feature counts
  (`min_long_count`, etc.). Match logic lives in `utils.match_conditions`.
- To introduce a brand-new ablation:
  1. Add it to `ablation_plan.yaml` with `generation` + output paths.
  2. Rebuild profiles (`generate_profiles.py`).
  3. Run reranking + evaluation as usual.

This structure ensures MedCorpus (multi-turn) and LitSearch (single-turn) share
one reproducible framework for future ablations, subset diagnostics, and paper-ready
tables/plots.
