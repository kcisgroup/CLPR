# personalized_generator.py - 修复版本：更严格的长度控制
import logging
import re
import gc
import requests
import time
import math
import threading
import numpy as np
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple

try:
    from utils import logger, get_config
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('PersonalizedGenerator_Fallback')
    logger.warning("Could not import logger/get_config from utils, using fallback.")
    class DummyConfig:
        device="cpu"; llm_device="cpu";
        personalized_text_target_length=250;
        llm_api_type="siliconflow"
        siliconflow_api_key=""
        siliconflow_api_url="https://api.siliconflow.cn/v1/chat/completions"
        siliconflow_model="Qwen/Qwen3-14B"
        local_model_temperature=0.3; local_model_top_p=0.9; local_model_top_k=15;
        profile_generation_attempts=3;
        use_fixed_seed=True;
        llm_seed=42;
        def __getattr__(self, name): return None
    def get_config(): return DummyConfig()

try:
    from prompt_templates import DynamicPromptTemplates
except ImportError:
    logger.error("Could not import DynamicPromptTemplates from prompt_templates.py.")
    class DynamicPromptTemplates:
        @staticmethod
        def format_memory_features(features: List[str]) -> str:
            if not features: return "No specific memory features available for this query."
            return "\n".join(features)

class PersonalizedGenerator:
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # Only SiliconFlow API with Qwen3 models
        self.api_type = 'siliconflow'
        self._http = requests.Session()
        self.api_url = getattr(self.config, 'siliconflow_api_url', "https://api.siliconflow.cn/v1/chat/completions")
        self.api_key = getattr(self.config, 'siliconflow_api_key', "")
        self.model_name = getattr(self.config, 'siliconflow_model', "Qwen/Qwen3-14B")
        logger.info(f"PersonalizedGenerator initialized with SiliconFlow API (Model: {self.model_name})")

        self.target_length = getattr(self.config, 'personalized_text_target_length', 250)
        self.min_len, self.max_len = self._get_length_range()
        logger.info(f"Generator using target length {self.target_length}, aiming for range: {self.min_len}-{self.max_len} characters.")

        # 根据目标长度动态设置默认max_tokens
        if self.target_length <= 150:
            self.default_max_tokens = 60  # 更保守的设置
        elif self.target_length <= 250:
            self.default_max_tokens = 100
        else:
            self.default_max_tokens = 140

        # 降低温度以获得更一致的输出
        self.temperature = getattr(self.config, 'local_model_temperature', 0.3)
        self.top_p = getattr(self.config, 'local_model_top_p', 0.9)
        self.top_k = getattr(self.config, 'local_model_top_k', 15)
        # 增加生成尝试次数
        self.generation_attempts = getattr(self.config, 'profile_generation_attempts', 3)
        self.use_fixed_seed = getattr(self.config, 'use_fixed_seed', True)
        self.llm_seed = getattr(self.config, 'llm_seed', 42)

        # 并发控制参数
        self.max_concurrent = getattr(self.config, 'max_concurrent_requests', 20)  # 并发数
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent)

        # 统计信息（线程安全）
        self.api_call_count = 0
        self.successful_call_count = 0
        self.failed_call_count = 0
        self.total_api_latency = 0.0
        self.prompt_tokens_reported = 0
        self.completion_tokens_reported = 0
        self.total_tokens_reported = 0
        self.total_tokens_estimated = 0
        self.prompt_tokens_estimated = 0
        self.completion_tokens_estimated = 0
        self.stats_lock = threading.Lock()

        logger.info(f"Generator params: temp={self.temperature}, top_p={self.top_p}, top_k={self.top_k}")
        logger.info(f"Default max_tokens for target_length {self.target_length}: {self.default_max_tokens}")
        logger.info(f"Quality control: attempts={self.generation_attempts}, fixed_seed={self.use_fixed_seed}, seed={self.llm_seed}")
        logger.info(f"Concurrent requests: {self.max_concurrent}")
        self._check_api_connection()
        
    def _get_length_range(self) -> Tuple[int, int]:
        if self.target_length <= 150: return 100, 200
        elif self.target_length <= 250: return 200, 300
        else: return 300, 400

    def _check_api_connection(self):
        """Check SiliconFlow API connection"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "max_tokens": 10
            }
            response = self._http.post(self.api_url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            logger.info("SiliconFlow API connection successful.")
        except Exception as e:
            logger.error(f"Could not connect to SiliconFlow API: {e}")

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 4))


    def _llm_request(self, prompt: str, max_tokens: int = None, temperature: float = None, seed: int = None, retry: bool = True) -> str:
        if max_tokens is None: max_tokens = self.default_max_tokens
        if temperature is None: temperature = self.temperature
        if seed is None and self.use_fixed_seed: seed = self.llm_seed

        max_attempts = 3 if retry else 1

        for attempt in range(max_attempts):
            attempt_start = time.time()
            try:
                headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k
                }
                response = self._http.post(self.api_url, json=payload, headers=headers, timeout=120)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"].strip() if "choices" in data and data["choices"] else ""
                usage = data.get("usage", {}) if isinstance(data, dict) else {}
                prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
                completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
                total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
                prompt_estimate = self._estimate_tokens(prompt)
                completion_estimate = self._estimate_tokens(content)
                with self.stats_lock:
                    self.api_call_count += 1
                    self.successful_call_count += 1
                    self.total_api_latency += time.time() - attempt_start
                    if prompt_tokens is not None:
                        self.prompt_tokens_reported += prompt_tokens
                    else:
                        self.prompt_tokens_estimated += prompt_estimate
                    if completion_tokens is not None:
                        self.completion_tokens_reported += completion_tokens
                    else:
                        self.completion_tokens_estimated += completion_estimate
                    if total_tokens is not None:
                        self.total_tokens_reported += total_tokens
                    else:
                        inferred_total = (prompt_tokens if prompt_tokens is not None else prompt_estimate) + \
                                         (completion_tokens if completion_tokens is not None else completion_estimate)
                        self.total_tokens_estimated += inferred_total
                return content
            except requests.exceptions.RequestException as e:
                with self.stats_lock:
                    self.api_call_count += 1
                    self.failed_call_count += 1
                    self.total_api_latency += time.time() - attempt_start
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f"Error: API request failed - {str(e)}"
            except Exception as e:
                with self.stats_lock:
                    self.api_call_count += 1
                    self.failed_call_count += 1
                    self.total_api_latency += time.time() - attempt_start
                return f"Error: Failed to process response - {str(e)}"
        return ""

    def _evaluate_description_quality(self, description: str, memory_features: List[str], query: str) -> Tuple[float, Dict[str, float]]:
        scores = {}
        length = len(description)
        ideal_min_len, ideal_max_len = self.min_len, self.max_len

        # 长度评分更严格
        if ideal_min_len <= length <= ideal_max_len: 
            scores['length'] = 1.0
        elif length < ideal_min_len: 
            scores['length'] = max(0, 1.0 - (ideal_min_len - length) / ideal_min_len)
        else: 
            scores['length'] = max(0, 1.0 - (length - ideal_max_len) / ideal_max_len)

        feature_terms = {term for feature in memory_features for term in re.findall(r'\b\w{3,}\b', re.sub(r'\[[^\]]+\]', '', feature).lower())}
        scores['term_coverage'] = min(sum(1 for term in feature_terms if term in description.lower()) / len(feature_terms), 1.0) if feature_terms else 0.5
        
        structure_score = 0.2 if description.strip().endswith('.') else 0.0
        structure_score += 0.2 if len(re.split(r'[.!?]+', description)) >= 1 else 0.0
        scores['structure'] = min(structure_score, 0.4)
        
        content_score = 0.3 if any(verb in description.lower() for verb in ['define', 'explain', 'compare', 'analyze', 'list', 'show', 'find', 'identify']) else 0.0
        technical_indicators = ['method', 'approach', 'algorithm', 'model', 'system', 'framework', 'technique', 'analysis', 'optimization']
        content_score += min(sum(1 for ind in technical_indicators if ind in description.lower()) * 0.3, 0.6)
        scores['content'] = min(content_score, 1.0)
        
        query_terms = {term for term in re.findall(r'\b\w{3,}\b', query.lower()) if term not in {'what', 'are', 'the', 'any', 'papers', 'on'}}
        if query_terms:
            desc_terms = {term for term in re.findall(r'\b\w{3,}\b', description.lower())}
            scores['relevance'] = len(query_terms.intersection(desc_terms)) / len(query_terms.union(desc_terms)) if query_terms.union(desc_terms) else 0.0
        else: scores['relevance'] = 0.5

        # 调整权重，增加长度的重要性
        weights = {'length': 0.35, 'term_coverage': 0.2, 'structure': 0.1, 'content': 0.2, 'relevance': 0.15}
        return sum(scores.get(key, 0) * weight for key, weight in weights.items()), scores

    def get_generation_stats(self) -> Dict[str, Any]:
        with self.stats_lock:
            total_prompt_tokens = self.prompt_tokens_reported + self.prompt_tokens_estimated
            total_completion_tokens = self.completion_tokens_reported + self.completion_tokens_estimated
            total_tokens = total_prompt_tokens + total_completion_tokens
            avg_latency = (self.total_api_latency / self.api_call_count) if self.api_call_count else 0.0
            return {
                "model": self.model_name,
                "api_calls_total": self.api_call_count,
                "api_calls_successful": self.successful_call_count,
                "api_calls_failed": self.failed_call_count,
                "avg_api_latency_seconds": avg_latency,
                "prompt_tokens_reported": self.prompt_tokens_reported,
                "prompt_tokens_estimated": self.prompt_tokens_estimated,
                "completion_tokens_reported": self.completion_tokens_reported,
                "completion_tokens_estimated": self.completion_tokens_estimated,
                "total_tokens_reported": self.total_tokens_reported,
                "total_tokens_estimated": self.total_tokens_estimated,
                "aggregated_prompt_tokens": total_prompt_tokens,
                "aggregated_completion_tokens": total_completion_tokens,
                "aggregated_total_tokens": total_tokens
            }

    def _build_unified_turn1_style_prompt(self, query: str, formatted_features: str) -> str:
        """CoT策略：通过链式思考生成高质量的重排辅助文本，保留所有关键术语"""
        min_len, max_len = self.min_len, self.max_len
        
        length_instruction = f"OUTPUT LENGTH: {min_len}-{max_len} characters ONLY."
        
        prompt_lines = [
            f"CRITICAL REQUIREMENT: {length_instruction}",
            "",
            "TASK: Generate a research background description that helps a reranker identify relevant documents for this query.",
            "",
            f"Current Query: \"{query}\"",
            ""
        ]
        
        if formatted_features and "No relevant" not in formatted_features:
            prompt_lines.extend([
                "User's Research Context (extracted cognitive features):",
                "",
                "The following features represent the user's research focus extracted from conversation history.",
                "Three types of features are provided:",
                "",
                "• [SEQUENTIAL_MEMORY]: Keywords/concepts from previous turns in this conversation",
                "  → Shows what was discussed earlier (e.g., 'ocd depression' means Turn 1 discussed OCD and depression)",
                "  → Use these to resolve pronouns like 'this', 'these', 'one...other' in current query",
                "",
                "• [WORKING_MEMORY]: Current session's main focus areas",
                "  → Indicates what user is currently concentrating on",
                "",
                "• [LONG_EXPLICIT]: User's long-term research interests",
                "  → Broader research domain the user works in",
                "",
                "IMPORTANT: These features contain KEYWORDS extracted from previous queries.",
                "When you see comma-separated terms like 'ocd, depression', understand they represent",
                "specific entities mentioned in previous discussions, even if their relationship isn't explicit.",
                "",
                "Extracted Features:",
                formatted_features,
                ""
            ])
        
        prompt_lines.extend([
            "CHAIN-OF-THOUGHT REASONING (think step by step):",
            "",
            "Step 1 - IDENTIFY KEY TERMS:",
            "   First, extract ALL technical terms, entities, and method names from the query.",
            "   These MUST be preserved EXACTLY as written in the final output.",
            "   Examples: specific chemicals (MoS₂), technical terms (tribological), method names (osimertinib)",
            "",
            "Step 2 - DETERMINE RESEARCH DOMAIN:",
            "   What is the high-level research field? (e.g., materials science, oncology, neuroscience)",
            "   What is the application area? (e.g., aerospace, clinical treatment, drug development)",
            "",
            "Step 3 - INTELLIGENTLY USE EXTRACTED FEATURES:",
            "   The features contain KEYWORDS from previous conversations. Use them to:",
            "",
            "   a) RESOLVE PRONOUNS AND REFERENCES:",
            "      If current query has 'this', 'these', 'one...other', look at [SEQUENTIAL_MEMORY]",
            "      Example: Query says 'treating one condition over the other'",
            "               Features show 'ocd, depression' → understand: one=OCD, other=depression",
            "",
            "   b) UNDERSTAND IMPLICIT RELATIONSHIPS:",
            "      Comma-separated keywords imply relationships from previous discussion",
            "      Example: 'ocd, depression, comparison' → they were being compared",
            "               'mos₂, oxygen, incorporation' → oxygen incorporation into MoS₂",
            "",
            "   c) EXTRACT DOMAIN CONTEXT:",
            "      Use [WORKING_MEMORY] and [LONG_EXPLICIT] for research domain",
            "      But ONLY if relevant to current query",
            "",
            "   ✗ AVOID:",
            "      - Including features unrelated to current query",
            "      - Listing keywords without integrating them meaningfully",
            "",
            "Step 4 - CONSTRUCT OUTPUT:",
            "   Combine: [User Perspective] + [Research Domain] + [Key Terms from Query]",
            "   Format: \"Researcher in [domain] investigating [key terms] for [application/context]\"",
            "   Or: \"Researcher specializing in [domain] with focus on [key terms] in [context]\"",
            "   IMPORTANT: Use user perspective (\"Researcher...\"), NOT task perspective (\"Research on...\")!",
            "",
            "CRITICAL RULES:",
            "• PRESERVE EXACTL: Every technical term, entity name, and chemical formula from the query",
            "• NO LISTING: Do NOT use 'including X, Y, Z' or 'such as A, B, C'",
            "• NO REPHRASING: Keep original terminology (don't change 'MoS₂' to 'molybdenum disulfide')",
            "• ADD CONTEXT: Provide research domain and application area",
            "• BE CONCISE: Stay within character limit",
            "",
            f"FINAL OUTPUT REQUIREMENTS:",
            f"• Length: {min_len}-{max_len} characters",
            "• Format: Single declarative statement",
            "• NO explanation, NO quotes, ONLY the research background description",
            "",
            "Now generate the output:"
        ])
        return "\n".join(prompt_lines)

    def generate_personalized_text(self, query: str, memory_results: Dict, **kwargs) -> str:
        try:
            tagged_features = memory_results.get("tagged_memory_features", [])
            formatted_features = DynamicPromptTemplates.format_memory_features(tagged_features or [])
            prompt = self._build_unified_turn1_style_prompt(query, formatted_features)
            candidates, detailed_scores = [], []
            
            # 调整后的token设置 - 适当增加以改善流畅性
            if self.target_length <= 150:
                max_tokens_for_request = 70  # 从60提升到70
            elif self.target_length <= 250:
                max_tokens_for_request = 120  # 从100提升到120
            else:
                max_tokens_for_request = 160  # 从140提升到160

            for attempt in range(self.generation_attempts):
                # 温度变化更小，保持稳定性
                temp_variation = self.temperature + (attempt - self.generation_attempts // 2) * 0.02 if not self.use_fixed_seed and self.generation_attempts > 1 else self.temperature
                seed = (self.llm_seed + attempt) if self.use_fixed_seed else None
                
                raw_response = self._llm_request(prompt, max_tokens=max_tokens_for_request, temperature=temp_variation, seed=seed, retry=True)

                if raw_response.startswith("Error:"):
                    continue

                cleaned_text = self._clean_and_validate_description(raw_response)

                if not cleaned_text:
                    continue

                # 立即检查长度
                if len(cleaned_text) > self.max_len:
                    cleaned_text = self._smart_truncate(cleaned_text, self.max_len)
                    
                quality_score, score_details = self._evaluate_description_quality(cleaned_text, tagged_features or [], query)
                candidates.append(cleaned_text)
                detailed_scores.append({
                    'total': quality_score, 
                    'details': score_details, 
                    'attempt': attempt + 1, 
                    'seed_used': seed,
                    'length': len(cleaned_text)
                })

            if candidates:
                # 优先选择长度符合要求的候选
                valid_candidates = [(i, c) for i, c in enumerate(candidates) 
                                   if self.min_len <= len(c) <= self.max_len]
                
                if valid_candidates:
                    # 从符合长度的候选中选择得分最高的
                    valid_scores = [(i, detailed_scores[i]['total']) for i, _ in valid_candidates]
                    best_valid_idx = max(valid_scores, key=lambda x: x[1])[0]
                    best_text = candidates[best_valid_idx]
                    score_info = detailed_scores[best_valid_idx]
                else:
                    # 如果没有符合长度的，选择最接近的
                    length_diffs = []
                    for i, c in enumerate(candidates):
                        if len(c) < self.min_len:
                            diff = self.min_len - len(c)
                        elif len(c) > self.max_len:
                            diff = len(c) - self.max_len
                        else:
                            diff = 0
                        length_diffs.append((i, diff, detailed_scores[i]['total']))
                    
                    # 选择长度差异最小且得分较高的
                    length_diffs.sort(key=lambda x: (x[1], -x[2]))
                    best_idx = length_diffs[0][0]
                    best_text = candidates[best_idx]

                return best_text
                
            return self._generate_fallback_description(query, formatted_features)
            
        except Exception as e:
            logger.error(f"Error in generate_personalized_text: {e}", exc_info=True)
            return self._generate_fallback_description(query, "")

    def _smart_truncate(self, text: str, max_len: int) -> str:
        """智能截断文本到指定长度"""
        if len(text) <= max_len:
            return text
            
        # 尝试在句子边界截断
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            truncated = ""
            for sent in sentences:
                if len(truncated) + len(sent) + 1 <= max_len - 3:  # +1 for punctuation, -3 for "..."
                    truncated = truncated + sent + "." if truncated else sent + "."
                else:
                    break
            if truncated:
                return truncated.rstrip()
        
        # 否则在词边界截断
        limit = text.rfind(' ', 0, max_len - 3)
        return text[:limit] + "..." if limit != -1 else text[:max_len-3] + "..."

    def _clean_and_validate_description(self, text: str) -> str:
        """简化版本清理"""
        text = text.strip()
        if len(text) < 10:
            return ""
        return text

    def _generate_fallback_description(self, query: str, features: str) -> str:
        processed_query = re.sub(r"^(are there any|what are|how to|can you find)\s+", "", query.lower(), flags=re.IGNORECASE).strip()
        fallback = processed_query.capitalize()
        
        # 确保fallback也在长度范围内
        if len(fallback) < self.min_len:
            # 添加一些通用词来延长
            fallback = f"Research on {fallback}"
        elif len(fallback) > self.max_len:
            # 截断
            fallback = self._smart_truncate(fallback, self.max_len)

        return self._clean_and_validate_description(fallback)

    def _process_single_query(self, json_line: str, index: int) -> Tuple[int, str, float]:
        """处理单个查询，返回 (index, description, quality_score)"""
        try:
            data_item = json.loads(json_line)
            if not isinstance(data_item, dict):
                raise TypeError("Parsed JSON is not a dictionary.")
        except (json.JSONDecodeError, TypeError):
            return (index, self._generate_fallback_description("Invalid data line", ""), 0.0)

        query_text = data_item.get("query", "")
        if not query_text:
            return (index, self._generate_fallback_description("Missing query", ""), 0.0)

        mem_res = data_item.get("memory_results", {})
        desc = self.generate_personalized_text(query_text, mem_res)

        if not desc:
            desc = self._generate_fallback_description(query_text, "")

        score, _ = self._evaluate_description_quality(desc, mem_res.get("tagged_memory_features", []), query_text)
        return (index, desc, score)

    def generate_personalized_text_batch(self, queries_data: List[str], progress_callback=None) -> List[str]:
        """并发批量生成个性化描述文本"""
        total = len(queries_data)
        descriptions = [""] * total
        quality_scores = []

        # 使用 ThreadPoolExecutor 并发处理
        futures = []
        for i, json_line in enumerate(queries_data):
            future = self.executor.submit(self._process_single_query, json_line, i)
            futures.append(future)

        # 收集结果
        completed = 0
        for future in futures:
            try:
                index, desc, score = future.result()
                descriptions[index] = desc
                quality_scores.append(score)
                completed += 1

                # 回调进度
                if progress_callback:
                    progress_callback(completed)

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                completed += 1
                if progress_callback:
                    progress_callback(completed)

        if quality_scores:
            logger.info(f"Batch complete. Quality stats: Avg={np.mean(quality_scores):.3f}, Std={np.std(quality_scores):.3f}")

        return descriptions

    def generate_personalized_text_batch_old(self, queries_data: List[str]) -> List[str]:
        """批量生成个性化描述文本"""
        descriptions, quality_scores = [], []
        for i, json_line in enumerate(queries_data):
            try:
                data_item = json.loads(json_line)
                if not isinstance(data_item, dict): raise TypeError("Parsed JSON is not a dictionary.")
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Skipping invalid JSON line {i+1}: {e}. Line: '{json_line.strip()}'")
                descriptions.append(self._generate_fallback_description("Invalid data line", ""))
                continue

            query_text = data_item.get("query", "")
            if not query_text:
                logger.warning(f"Skipping item {i+1} due to missing 'query'.")
                descriptions.append(self._generate_fallback_description("Missing query", ""))
                continue

            mem_res = data_item.get("memory_results", {})
            desc = self.generate_personalized_text(query_text, mem_res)
            descriptions.append(desc if desc else self._generate_fallback_description(query_text, ""))
            
            score, _ = self._evaluate_description_quality(desc, mem_res.get("tagged_memory_features", []), query_text)
            quality_scores.append(score)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(queries_data):
                logger.info(f"Batch progress: {i+1}/{len(queries_data)}, Recent avg quality: {np.mean(quality_scores[-10:]):.3f}")
            gc.collect()

        if quality_scores:
            logger.info(f"Batch complete. Quality stats: Avg={np.mean(quality_scores):.3f}, Std={np.std(quality_scores):.3f}")
        return descriptions

if __name__ == '__main__':
    logger.info("Starting PersonalizedGenerator standalone test.")
    class TestConfig:
        personalized_text_target_length = 150
        llm_api_type = "siliconflow"
        siliconflow_api_url = "https://api.siliconflow.cn/v1/chat/completions"
        siliconflow_api_key = ""
        siliconflow_model = "Qwen/Qwen3-14B"
        local_model_temperature=0.3; local_model_top_p=0.9; local_model_top_k=15;
        profile_generation_attempts=3; use_fixed_seed=True; llm_seed=123
        def __getattr__(self, name): return None
    
    config_to_use = TestConfig()
    generator = PersonalizedGenerator(config=config_to_use)
    logger.info(f"Test generator initialized with range: {generator.min_len}-{generator.max_len}")
    
    test_queries_examples = [
        {"query_id": "1", "query": "Research on task-agnostic knowledge distillation for compressing large language models."}, 
        {"query_id": "2", "query": "Resources for Tunisian Arabic dialect translation with native speaker comments and data augmentation."}
    ]
    batch_data_str = [json.dumps({"query": item["query"], "memory_results": {}}) for item in test_queries_examples]
    
    logger.info("\n--- Testing Batch Generation with Range Logic ---")
    try:
        generated_descriptions = generator.generate_personalized_text_batch(batch_data_str)
        for i, desc in enumerate(generated_descriptions):
            logger.info(f"\nOriginal Query: {test_queries_examples[i]['query']}")
            logger.info(f"Generated Text: '{desc}' (Length: {len(desc)})")
            logger.info(f"In target range [{generator.min_len}-{generator.max_len}]: {generator.min_len <= len(desc) <= generator.max_len}")
    except Exception as e:
        logger.error(f"Error during batch generation test: {e}", exc_info=True)
    logger.info("PersonalizedGenerator standalone test finished.")
