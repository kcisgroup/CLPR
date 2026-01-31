#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced LLM-based turn generation with question type classification.

Question Types:
- definition: Conceptual questions (What is...? How does... work?)
- methodology: Experimental design, validation, analysis methods
- comparison: Comparative analysis (How does X compare to Y?)
- clinical: Clinical application, translational research, patient outcomes
"""

from typing import List, Dict, Any, Tuple


# ==================== Turn 4: Methodology Focus ====================

def build_turn4_prompt_typed(turns_1_3: List[str]) -> List[Dict[str, Any]]:
    """
    Generate Turn 4 with **methodology** question type.

    Focus: Experimental design, statistical validation, technical details.
    """
    history = "\n".join([f"Turn {i+1}: {t}" for i, t in enumerate(turns_1_3)])

    system = (
        "You are an expert biomedical researcher conducting a literature search. "
        "Generate realistic follow-up queries that naturally continue the conversation."
    )

    user = f"""Given the conversation history below, generate Turn 4 as a **METHODOLOGY-focused** question.

**Question Type: METHODOLOGY**
This means the question should explore:
- Experimental design and protocols
- Statistical methods and validation strategies
- Technical implementation details
- Data analysis approaches
- Quality control and reproducibility

**Requirements**:
1. **Semantic Continuity**: Use anaphora (e.g., "these methods", "this approach") to reference Turn 3
2. **Specificity**: Ask about concrete methodological aspects, not vague concepts
3. **Length**: 60-100 words
4. **Academic Style**: Professional but conversational (like a researcher thinking aloud)
5. **Avoid**: Generic phrases, exact repetition from Turn 3

**Conversation History**:
{history}

**Output Format**:
Question type: methodology
Question text: [Your Turn 4 question here]

Generate the Turn 4 question:"""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ==================== Turn 5: Clinical/Comparison Focus ====================

def build_turn5_prompt_typed(turns_1_4: List[str], question_type: str = "clinical") -> List[Dict[str, Any]]:
    """
    Generate Turn 5 with **clinical** or **comparison** question type.

    Args:
        question_type: "clinical" (70%) or "comparison" (30%)
    """
    history = "\n".join([f"Turn {i+1}: {t}" for i, t in enumerate(turns_1_4)])

    if question_type == "clinical":
        type_desc = """**Question Type: CLINICAL**
This means the question should explore:
- Clinical applications and translational research
- Patient outcomes and therapeutic efficacy
- Real-world deployment challenges
- Safety, side effects, or contraindications
- Clinical trial results or evidence-based practice"""

    else:  # comparison
        type_desc = """**Question Type: COMPARISON**
This means the question should explore:
- Comparative effectiveness (X vs Y)
- Advantages and disadvantages of different approaches
- Trade-offs between methods or treatments
- Meta-analysis or systematic comparisons
- Relative performance across contexts"""

    system = (
        "You are an expert biomedical researcher conducting a literature search. "
        "Generate realistic follow-up queries that explore related aspects."
    )

    user = f"""Given the conversation history below, generate Turn 5 as a **{question_type.upper()}-focused** question.

{type_desc}

**Requirements**:
1. **Related Exploration**: Shift focus to a complementary angle while maintaining topical coherence
2. **Contextual Awareness**: Reference insights from previous turns (e.g., "Given these findings...")
3. **Length**: 60-100 words
4. **Natural Flow**: Show awareness of the dialogue's progression
5. **Avoid**: Complete topic change, redundant restating

**Conversation History**:
{history}

**Output Format**:
Question type: {question_type}
Question text: [Your Turn 5 question here]

Generate the Turn 5 question:"""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ==================== Turn 6: Comparison/Definition Focus ====================

def build_turn6_prompt_typed(turns_1_5: List[str], question_type: str = "comparison") -> List[Dict[str, Any]]:
    """
    Generate Turn 6 with **comparison** or **definition** question type.

    Args:
        question_type: "comparison" (60%) or "definition" (40%)
    """
    history = "\n".join([f"Turn {i+1}: {t}" for i, t in enumerate(turns_1_5)])

    if question_type == "comparison":
        type_desc = """**Question Type: COMPARISON**
This means the question should explore:
- Comparative evaluation of methods discussed earlier
- Limitations and strengths across different approaches
- Risk-benefit analysis or cost-effectiveness
- Future directions: how does this compare to emerging alternatives?
- Critical assessment of trade-offs"""

    else:  # definition
        type_desc = """**Question Type: DEFINITION**
This means the question should explore:
- Clarifying concepts or mechanisms mentioned earlier
- Deep-dive into theoretical foundations
- Asking "what exactly is..." or "how does... fundamentally work?"
- Defining technical terms or frameworks in context
- Conceptual understanding beyond surface-level"""

    system = (
        "You are an expert biomedical researcher conducting a literature search. "
        "Generate realistic follow-up queries that critically evaluate or clarify concepts."
    )

    user = f"""Given the conversation history below, generate Turn 6 as a **{question_type.upper()}-focused** question.

{type_desc}

**Requirements**:
1. **Critical/Analytical Tone**: Show reflective thinking (e.g., "What are the main challenges...", "How does this compare...")
2. **Long-term Perspective**: Consider future directions, limitations, or unresolved questions
3. **Length**: 60-100 words
4. **Dialogue Continuity**: Reference specific insights from the 5-turn conversation
5. **Avoid**: Vague questions, pure repetition

**Conversation History**:
{history}

**Output Format**:
Question type: {question_type}
Question text: [Your Turn 6 question here]

Generate the Turn 6 question:"""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ==================== Relevance Labeling (Unchanged) ====================

def build_relevance_labeling_prompt(query: str, doc_title: str, doc_abstract: str) -> List[Dict[str, Any]]:
    """
    LLM-based relevance judgment: rel=0/1/2.

    rel=2: Highly relevant (directly answers query, on-topic)
    rel=1: Somewhat relevant (related concepts, partial match)
    rel=0: Not relevant (off-topic, tangential)
    """
    system = (
        "You are an expert annotator for biomedical literature retrieval. "
        "Judge document relevance on a 3-level scale: 0 (not relevant), 1 (somewhat relevant), 2 (highly relevant)."
    )

    user = f"""Query:
{query}

Document:
Title: {doc_title}
Abstract: {doc_abstract[:500]}...

Rate the relevance of this document to the query:
- 2 (highly relevant): Directly addresses the query's core question, on-topic methodology/findings
- 1 (somewhat relevant): Contains related concepts or tangential information, but not a direct answer
- 0 (not relevant): Off-topic or only superficially related

Output format: A single digit (0, 1, or 2) with a brief 1-sentence justification.
Example: "2. This RCT directly evaluates the treatment mentioned in the query."

Your rating:"""

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ==================== Helper: Parse LLM Response ====================

def parse_typed_response(llm_output: str) -> Tuple[str, str]:
    """
    Parse LLM response to extract question type and text.

    Expected format:
        Question type: methodology
        Question text: What statistical methods were used to validate...

    Returns:
        (question_type, question_text)
    """
    lines = llm_output.strip().split("\n")
    q_type = "unknown"
    q_text = ""

    for line in lines:
        if line.lower().startswith("question type:"):
            q_type = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("question text:"):
            q_text = line.split(":", 1)[1].strip()

    # Fallback: if parsing fails, treat entire output as question text
    if not q_text:
        q_text = llm_output.strip()

    return q_type, q_text
