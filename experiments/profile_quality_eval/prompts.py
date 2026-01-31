"""
Prompt 模板：个性化特征质量评估
"""

EVALUATION_PROMPT_TEMPLATE = """You are an expert evaluator assessing the quality of personalized query representations for medical literature search systems.

Your task is to evaluate how well a system-generated "Personalized Features" text captures the user's search intent based on their query and conversation history.

## Input Information

**Current Query:**
{query}

**Conversation History:**
{history}

**Generated Personalized Features:**
{personalized_features}

## Evaluation Task

Please evaluate the **Personalized Features** text on the following 4 dimensions (1-5 scale):

### 1. Relevance (1-5)
*How relevant is the personalized text to the current query?*

- **1 (Very Poor)**: Completely off-topic or irrelevant
- **2 (Poor)**: Mostly irrelevant with minor connections
- **3 (Fair)**: Partially relevant but missing key aspects
- **4 (Good)**: Highly relevant with minor gaps
- **5 (Excellent)**: Perfectly captures the query's core intent

### 2. Accuracy (1-5)
*How accurately does it capture the user's intent from the conversation history?*

- **1 (Very Poor)**: Misrepresents or contradicts the user's intent
- **2 (Poor)**: Captures some intent but with significant errors
- **3 (Fair)**: Reasonably accurate but misses important context
- **4 (Good)**: Accurately captures most of the user's intent
- **5 (Excellent)**: Perfectly infers and represents the user's intent

### 3. Informativeness (1-5)
*How much useful information does it add beyond the original query?*

- **1 (Very Poor)**: Adds no new information (just repeats the query)
- **2 (Poor)**: Adds minimal context or background
- **3 (Fair)**: Adds some useful context
- **4 (Good)**: Adds substantial relevant context
- **5 (Excellent)**: Adds rich, valuable context that significantly enhances understanding

### 4. Coherence (1-5)
*How coherent is it with the conversation flow and history?*

- **1 (Very Poor)**: Contradicts previous turns or is inconsistent
- **2 (Poor)**: Weakly connected to the conversation
- **3 (Fair)**: Consistent but doesn't build on history
- **4 (Good)**: Well-integrated with conversation context
- **5 (Excellent)**: Seamlessly builds on and extends the conversation

## Special Considerations

- **Turn 1 (No History)**: If there's no conversation history, Accuracy and Coherence may naturally be lower (around 3). This is expected and should not be penalized.
- **Medical Terminology**: The personalized text should maintain appropriate medical terminology from the query and history.
- **Context Enrichment**: Good personalized features should add implicit context (e.g., research background, domain focus) that helps refine the search.

## Output Format

**CRITICAL: Respond with ONLY the JSON object. NO markdown, NO extra text.**

Format (output this exactly, with actual values):
{{
  "relevance": 4,
  "accuracy": 4,
  "informativeness": 4,
  "coherence": 4,
  "average_score": 4.0,
  "explanation": "Your justification here"
}}

Rules:
- Output ONLY the JSON (starting with {{ and ending with }})
- NO markdown code blocks
- NO text before or after the JSON
- All scores must be integers 1-5
- average_score is the mean of the 4 scores

Your JSON response:
"""


def format_history(history: list) -> str:
    """
    格式化对话历史
    
    Args:
        history: 对话历史列表
        
    Returns:
        格式化的历史字符串
    """
    if not history or len(history) == 0:
        return "(No previous conversation history)"
    
    formatted_lines = []
    for i, turn in enumerate(history, 1):
        formatted_lines.append(f"Turn {i}: {turn}")
    
    return "\n".join(formatted_lines)


def create_evaluation_prompt(query: str, history: list, personalized_features: str) -> str:
    """
    创建评估 prompt
    
    Args:
        query: 当前查询
        history: 对话历史
        personalized_features: 生成的个性化特征文本
        
    Returns:
        完整的评估 prompt
    """
    formatted_history = format_history(history)
    
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        query=query,
        history=formatted_history,
        personalized_features=personalized_features
    )
    
    return prompt


# ============================================
# 测试示例
# ============================================

if __name__ == "__main__":
    # 测试 prompt 生成
    
    # Turn 1 (无历史)
    test_query_1 = "Key tribological properties and advantages of MoS₂ as a solid lubricant"
    test_history_1 = []
    test_features_1 = "Researcher in materials science investigating key tribological properties and advantages of MoS₂ as a solid lubricant for applications in mechanical systems and high-temperature environments."
    
    prompt_1 = create_evaluation_prompt(test_query_1, test_history_1, test_features_1)
    print("=" * 80)
    print("Example 1: Turn 1 (No History)")
    print("=" * 80)
    print(prompt_1[:500] + "...\n")
    
    # Turn 2 (有历史)
    test_query_2 = "Influence of oxygen incorporation on these MoS₂ tribological properties"
    test_history_2 = ["Key tribological properties and advantages of MoS₂ as a solid lubricant"]
    test_features_2 = "Researcher in materials science investigating the influence of oxygen incorporation on MoS₂ tribological properties and its effect on lubricant performance and durability in mechanical engineering applications."
    
    prompt_2 = create_evaluation_prompt(test_query_2, test_history_2, test_features_2)
    print("=" * 80)
    print("Example 2: Turn 2 (With History)")
    print("=" * 80)
    print(prompt_2[:500] + "...\n")

