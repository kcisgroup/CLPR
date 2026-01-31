# Personalized Profile Quality Evaluation Report

**Total Queries Evaluated:** 220
**Models Used:** claude-haiku-4.5, gemini-2.5-pro-hybgzs, gpt-4o-mini-hybgzs

## 1. Overall Statistics

### Average Scores by Model

| Model | Relevance | Accuracy | Informativeness | Coherence | Overall | Success Rate |
|-------|-----------|----------|----------------|-----------|---------|--------------|
| claude-haiku-4.5 | 4.52 | 3.04 | 2.75 | 3.00 | **3.33** | 100.0% |
| gemini-2.5-pro-hybgzs | 4.92 | 3.87 | 3.08 | 3.55 | **3.85** | 97.7% |
| gpt-4o-mini-hybgzs | 4.83 | 3.90 | 3.57 | 3.03 | **3.83** | 100.0% |

## 2. Analysis by Turn

### Average Scores by Turn

| Turn | Relevance | Accuracy | Informativeness | Coherence | Overall |
|------|-----------|----------|----------------|-----------|---------|
| Turn 1 | 4.50 | 3.05 | 2.86 | 3.00 | **3.35** |
| Turn 2 | 4.23 | 3.03 | 2.80 | 3.00 | **3.27** |
| Turn 3 | 4.86 | 3.04 | 2.53 | 3.00 | **3.36** |

## 3. Inter-Model Agreement

### Model Comparison

**vs Claude 4.5 Haiku (baseline):**
- **Gemini 2.5 Pro**: Overall +15.6% (3.85 vs 3.33)
  - Relevance: +8.8%, Accuracy: +27.3%, Informativeness: +12.0%, Coherence: +18.3%
- **GPT-4o-mini**: Overall +15.0% (3.83 vs 3.33)
  - Relevance: +6.9%, Accuracy: +28.3%, Informativeness: +29.8%, Coherence: +1.0%

**Gemini 2.5 Pro vs GPT-4o-mini:**
- Very similar overall scores (3.85 vs 3.83)
- Gemini stronger in: Relevance (+1.9%), Coherence (+17.2%)
- GPT-4o-mini stronger in: Accuracy (+0.8%), Informativeness (+15.9%)

## 4. Key Findings

1. **Overall Quality**:
   - Claude 4.5 Haiku: **3.33 ± 0.50** (out of 5.0)
   - Gemini 2.5 Pro: **3.85 ± 0.59** (out of 5.0)
   - GPT-4o-mini: **3.83 ± 0.31** (out of 5.0)

2. **Model Performance**: Both Gemini 2.5 Pro and GPT-4o-mini significantly outperform Claude 4.5 Haiku (~15% improvement), with complementary strengths:
   - **Gemini 2.5 Pro**: Best at Relevance (4.92) and Coherence (3.55)
   - **GPT-4o-mini**: Best at Informativeness (3.57) and competitive Accuracy (3.90)

3. **Dimension Analysis**:
   - Strongest dimension: **Relevance** (4.92 for Gemini, 4.83 for GPT-4o-mini)
   - Most improved: **Informativeness** (+29.8% for GPT-4o-mini vs Claude)
   - Weakest dimension: **Coherence** for GPT-4o-mini (3.03), **Informativeness** for Claude (2.75)

4. **Reliability**: Claude and GPT-4o-mini achieve 100% success rate, while Gemini achieves 97.7% (215/220).

## 5. Sample Evaluations

### Highest Scoring Example

**Query ID:** `7`
**Turn:** 1

**Query:** Can you direct me to research that explores methods for transforming multi-hop questions into single-hop sub-questions to leverage existing single-hop answer models?

**Personalized Features:** Research on natural language processing focusing on transforming multi-hop questions into single-hop sub-questions in context of leveraging existing single-hop answer models for improved question answering efficiency.

**Scores:**

- **claude-haiku-4.5**: 4.2 - The personalized features text excellently captures the core intent of the query. Relevance (5): It directly addresses the transformation of multi-hop questions into single-hop sub-questions and the leveraging of existing single-hop answer models, which are the exact focus points of the query. Accuracy (5): Despite no conversation history, it accurately represents the user's intent by correctly identifying the NLP domain, the specific transformation task, and the purpose (improved question answering efficiency). Informativeness (4): It adds valuable context by explicitly mentioning 'natural language processing' as the domain and 'improved question answering efficiency' as the goal, which enriches the search context beyond the original query. However, it doesn't add extensive additional context about specific methodologies or applications. Coherence (3): With no conversation history, this is the first turn, so coherence is naturally limited to internal consistency, which it maintains well. The text is logically structured and doesn't contradict itself, earning a fair score for a first-turn query.
