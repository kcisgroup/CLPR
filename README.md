# CLPR: Conversational Literature Personalized Re-ranking

[![Under Review](https://img.shields.io/badge/status-under%20review-yellow)](https://github.com/kcisgroup/CLPR)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-arXiv-red.svg)](https://github.com/kcisgroup/CLPR)

> **State-of-the-art personalized retrieval for multi-turn academic literature search**

CLPR addresses the "information overload" versus "evidence scarcity" paradox in academic literature retrieval. By dynamically modeling user intent through cognitive-inspired profiling, CLPR delivers precise, personalized document ranking for conversational search scenarios.

---

## Overview

Academic literature retrieval faces a persistent challenge: researchers must navigate millions of papers to find specific, high-quality evidence. This problem intensifies in conversational settings where users iteratively refine their queries across multiple turns.

**CLPR** introduces a three-stage framework that unifies dense semantic retrieval with personalized user profiling:

1. **High-Recall Retrieval** - Collects candidate documents using dense embeddings
2. **Dynamic Profile Generation** - Synthesizes conversational history into a concise textual profile
3. **Profile-Guided Re-ranking** - Uses the profile as a pseudo-query for neural cross-encoder re-ranking

The framework captures three complementary dimensions of user context:
- **Sequential cues** - Cross-turn continuity and anaphora resolution
- **Focus cues** - Short-term session-level intent
- **Background cues** - Long-term research expertise and interests

### Framework Architecture

![CLPR Framework Architecture](assets/figures/framework_architecture.svg)

*Figure 1: The CLPR framework consists of three stages: (1) High-recall semantic retrieval, (2) Dynamic profile generation synthesizing sequential, focus, and background cues via LLM, and (3) Profile-guided neural re-ranking.*

---

## Performance Highlights

| Dataset | Domain | NDCG@10 | P@1 | Status |
|---------|--------|---------|-----|--------|
| **MedCorpus** | Biomedical | **0.9271** | **0.9497** | SOTA |
| **LitSearch** | Computer Science | **0.4793** | **0.4238** | SOTA |

**Efficiency Gains:**
- ⚡ **25× faster** than GPT-4o-mini reranking (0.32s vs 8.14s per query)
- 💰 **7.9× fewer tokens** (259 vs 2,050 tokens)
- 📉 Only **1.8% performance drop** in NDCG@10

---

## Repository Structure

```
CLPR/
├── core/                      # Main framework implementation
│   ├── cognitive_retrieval.py # Stage 1: Feature extraction
│   ├── personalized_generator.py  # Stage 2: Profile generation
│   ├── simple_profile_reranker.py # Stage 3: Neural re-ranking
│   ├── memory_system.py       # Cognitive memory architecture
│   └── run.py                 # Main entry point
├── baselines/                 # Baseline implementations
│   ├── pbr_baseline.py        # Personalize Before Retrieve
│   ├── rpmn.py                # Re-finding Personalized Memory Network
│   └── llm_reranker.py        # GPT-4o, Claude, Gemini rerankers
├── evaluation/                # Evaluation scripts and metrics
├── experiments/               # Ablation studies and analysis
│   ├── memory_ablation/       # Memory component ablation
│   └── profile_quality_eval/  # Profile quality diagnostics
├── data/                      # Datasets (queries and labels)
│   ├── MedCorpus_MultiTurn/   # 800 conversations, 3,440 turns
│   └── LitSearch/             # 597 CS queries
└── create/                    # Dataset construction tools
```

---

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/kcisgroup/CLPR.git
cd CLPR

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')"
```

### Dataset Setup

The repository includes query files and relevance labels. Large corpus files must be downloaded separately:

| Dataset | File | Size | Status |
|---------|------|------|--------|
| MedCorpus | `corpus.jsonl` | ~132 MB | 🔄 Organizing |
| MedCorpus | `corpus_embeddings_qwen3.npy` | ~380 MB | 🔄 Organizing |
| LitSearch | `corpus.jsonl` | ~25 MB | [🤗 Hugging Face](https://huggingface.co/datasets/UKPLab/litsearch) |
| LitSearch | `corpus_embeddings_qwen3.npy` | ~55 MB | 🔄 Organizing |

> **Note**: MedCorpus corpus files are being organized and will be released soon. For immediate access, please open an issue or contact the authors.

---

## Datasets

### MedCorpus

The first large-scale multi-turn biomedical conversational retrieval benchmark:

![MedCorpus Overview](assets/figures/medcorpus_overview.png)

*Figure 2: Overview of MedCorpus dataset statistics and coverage.*

- **800** conversations (3-6 turns each, avg 4.3)
- **3,440** total query turns
- **92,703** documents from PubMed (2018-2023)
- **34,387** silver-standard relevance labels
- **41** biomedical subfields covered
- **54.7%** Chinese-language articles (multilingual)

### LitSearch

Cross-domain validation dataset:

- **597** expert-written CS queries
- **12,600** computer science papers
- Single-turn queries for generalization testing
- Sparse judgments (~2.3 relevant docs per query)

---

## Method

### Three-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: High-Recall Initial Retrieval                          │
│  ├── Encode query and documents with Qwen3-Embedding-0.6B        │
│  └── Retrieve top-K candidates via FAISS ANN index               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Dynamic Profile Generation                             │
│  ├── Extract Sequential cues (cross-turn continuity)             │
│  ├── Extract Focus cues (session-level intent)                   │
│  ├── Extract Background cues (long-term interests)               │
│  └── Synthesize via LLM into 200-300 char profile                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Profile-Guided Re-ranking                              │
│  ├── Use profile as pseudo-query                                 │
│  ├── Score candidates with Jina-Reranker-v3 cross-encoder        │
│  └── Return top-N re-ranked documents                            │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**Profile-Only Re-ranking:** The final ranking depends solely on the generated profile rather than explicit fusion of profile and query signals. This eliminates hyperparameter tuning while preserving personalization quality.

**Cognitive Memory Architecture:** Three complementary cue types modeled after human memory systems:
- Sequential: Working memory for anaphora resolution
- Focus: Short-term activation decay mechanism
- Background: Long-term stable research identity

---

## Results

### Main Results

| Method | MedCorpus NDCG@10 | MedCorpus P@1 | LitSearch NDCG@10 |
|--------|-------------------|---------------|-------------------|
| PBR | 0.7854 | 0.9064 | 0.3749 |
| RPMN | 0.8546 | 0.9302 | 0.3835 |
| GPT-4o-mini | 0.9406 | 0.9642 | 0.4405 |
| Claude 4.5 Haiku | **0.9519** | **0.9743** | 0.4680 |
| **CLPR (Qwen3-80B)** | 0.9271 | 0.9497 | **0.4793** |
| **CLPR (Qwen3-32B)** | 0.9239 | 0.9526 | 0.4749 |

### Ablation Study

Removing each memory component on MedCorpus:

| Configuration | NDCG@10 | Δ |
|---------------|---------|---|
| Full CLPR | 0.9271 | — |
| − Background | 0.9099 | −1.72% |
| − Sequential | 0.9181 | −0.90% |
| − Focus | 0.9211 | −0.60% |

**Key finding:** Long-term background cues contribute the largest personalization gain.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MedCorpus dataset is released for research purposes only.

