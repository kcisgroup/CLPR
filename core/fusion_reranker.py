# fusion_reranker.py
"""
Implements an advanced reranking strategy: "Separate Scoring & Fusion".

This script calculates two separate scores for each document:
1.  Score_Query: Relevance to the current query.
2.  Score_Profile: Alignment with the user's personalized profile.

It then fuses these scores using a weighting factor 'beta':
Final_Score = beta * Score_Query + (1 - beta) * Score_Profile

This allows for fine-grained control over the personalization strength.
"""
import gc
import json
import logging
import os
import torch
import argparse
import re
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from tqdm import tqdm

# --- Boilerplate and Helpers (adapted from your rerank.py) ---

try:
    from utils import get_config, logger
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('FusionReranker_Fallback')
    logger.error(f"Failed to import from utils: {e}.")
    class DummyConfig:
        device="cpu"; reranker_path=None; dataset_name="unknown";
        reranker_type="jina"; length_suffix="";
        results_dir = "./results"
        _personalized_queries_base = "personalized_queries"
        model_suffix = "_dummy_model"
        initial_top_k = 100
        final_top_k = 10
        batch_size = 8
        reranker_max_length = 512

        @property
        def personalized_queries_path(self):
            return os.path.join(self.results_dir, self.dataset_name, f"{self._personalized_queries_base}{self.model_suffix}.jsonl")
        
        @property
        def retrieved_results_path(self):
            return os.path.join(self.results_dir, self.dataset_name, "retrieved.jsonl")
        
        def __getattr__(self, name): return None
    def get_config(): return DummyConfig()

def load_personalized_features(path: str) -> Dict[str, str]:
    """Loads personalized profile texts for each query."""
    features_data = {}
    if not os.path.exists(path):
        logger.warning(f"Personalized features file not found: {path}")
        return features_data
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    query_id = str(data.get("query_id"))
                    p_text = data.get("personalized_features", "")
                    if query_id and "Error:" not in p_text and p_text:
                        features_data[query_id] = p_text
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON in {path}")
        logger.info(f"Loaded profiles for {len(features_data)} queries from {path}")
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
    return features_data

def load_retrieved_results(path: str) -> Dict[str, Dict]:
    """Loads initially retrieved documents for each query."""
    retrieved_data = {}
    if not os.path.exists(path):
        logger.error(f"Retrieved results file not found: {path}")
        return retrieved_data
    try:    
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    query_id = str(data.get("query_id"))
                    if query_id:
                        retrieved_data[query_id] = {
                            "query": data.get("query", ""),
                            "results": data.get("results", [])
                        }
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON in {path}")
    except Exception as e:
        logger.error(f"Error loading {path}: {e}")
    return retrieved_data

class FusionRerankerPromptFormatter:
    """Formats input for the reranker model."""
    def _get_doc_content(self, doc: Dict[str, Any]) -> str:
        parts = [str(doc.get(k, '')) for k in ['title', 'text', 'full_paper'] if doc.get(k)]
        return " ".join(parts).strip().replace("\n", " ")

    def format_input_qwen3(self, query_text: str, doc: Dict[str, Any], instruction: str = None) -> str:
        """Format input for Qwen3-Reranker (CausalLM style)"""
        doc_content = self._get_doc_content(doc)
        if instruction is None:
            instruction = 'Given a scientific literature search query, retrieve relevant papers that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query_text}\n<Document>: {doc_content}"

    def format_input(self, text: str, doc: Dict[str, Any], reranker_type: str, context_label: str) -> Any:
        doc_content = self._get_doc_content(doc)
        # Only Qwen3 reranker format
        return (f"{context_label}: {text}", doc_content)

def get_model_and_tokenizer(config, reranker_type):
    """Loads the Qwen3 reranker model and tokenizer."""
    from transformers import AutoModelForCausalLM
    
    # Only Qwen3-Reranker-0.6B  
    model_path = config.reranker_path or "/workspace/PerMed/model/Qwen3-Reranker-0.6B"
    
    logger.info(f"Loading Qwen3 reranker (CausalLM) from {model_path} to {config.device}")
    dtype = torch.float16 if "cuda" in str(config.device) else torch.float32
    
    try:
        # Qwen3-Reranker 是 CausalLM，不是 SequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=dtype
        ).to(config.device).eval()
        
        # 获取 yes/no token IDs
        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        
        # 准备特殊 tokens
        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        
        # 将这些信息附加到模型对象上
        model.token_false_id = token_false_id
        model.token_true_id = token_true_id
        model.prefix_tokens = prefix_tokens
        model.suffix_tokens = suffix_tokens
        
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load Qwen3 reranker from {model_path}: {e}", exc_info=True)
        raise

# --- Core Reranking and Fusion Logic ---

def compute_scores_qwen3_reranker(
    model, tokenizer,
    query_text: str,
    docs: List[Dict[str, Any]],
    config: Any,
    max_length: int = 8192
) -> Dict[str, float]:
    """Qwen3-Reranker 专用评分函数"""
    if not docs or not query_text:
        return {doc.get("text_id", ""): 0.0 for doc in docs}
    
    formatter = FusionRerankerPromptFormatter()
    doc_scores = {}
    
    # 准备特殊 tokens
    prefix_tokens = model.prefix_tokens
    suffix_tokens = model.suffix_tokens
    token_false_id = model.token_false_id
    token_true_id = model.token_true_id
    
    with torch.no_grad():
        for doc in tqdm(docs, desc="Qwen3 Reranking", leave=False):
            try:
                # 格式化输入
                formatted_text = formatter.format_input_qwen3(query_text, doc)
                
                # Tokenize
                inputs = tokenizer(
                    [formatted_text], 
                    padding=False, 
                    truncation='longest_first',
                    return_attention_mask=False,
                    max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
                )
                
                # 添加前缀和后缀 tokens
                input_ids = prefix_tokens + inputs['input_ids'][0] + suffix_tokens
                
                # Pad 到 max_length
                if len(input_ids) < max_length:
                    input_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids)) + input_ids
                else:
                    input_ids = input_ids[:max_length]
                
                # 转为 tensor
                input_ids_tensor = torch.tensor([input_ids]).to(config.device)
                attention_mask = torch.ones_like(input_ids_tensor)
                
                # 前向传播
                outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask)
                logits = outputs.logits[0, -1, :]  # 取最后一个 token 的 logits
                
                # 计算 yes/no 分数
                true_logit = logits[token_true_id].item()
                false_logit = logits[token_false_id].item()
                
                # Log-softmax 然后取 exp
                logits_pair = torch.tensor([false_logit, true_logit])
                scores_pair = torch.nn.functional.log_softmax(logits_pair, dim=0)
                score = scores_pair[1].exp().item()  # yes 的概率
                
                doc_scores[doc["text_id"]] = float(score)
                
            except Exception as e:
                logger.error(f"Error scoring doc {doc.get('text_id', 'unknown')}: {e}")
                doc_scores[doc["text_id"]] = 0.0
    
    return doc_scores

def compute_scores_for_context(
    model, tokenizer,
    context_text: str,
    docs: List[Dict[str, Any]],
    config: Any,
    context_label: str # "Query" or "Profile"
) -> Dict[str, float]:
    """Computes reranking scores for a list of documents against a single context (query or profile)."""
    # 对 Qwen3-Reranker 使用专用函数
    return compute_scores_qwen3_reranker(model, tokenizer, context_text, docs, config)

def run_fusion_reranking(config, beta):
    """
    Main function to run the separate scoring and fusion reranking process.
    """
    # 再次确保initial_top_k的值正确
    logger.info(f"进入run_fusion_reranking，当前config.initial_top_k = {config.initial_top_k}")
    
    # --- 文件名构造：总是包含K值 ---
    beta_str = f"beta{str(beta).replace('.', 'p')}"
    k_suffix = f"_K{config.initial_top_k}"  # 总是添加K值

    # 不再使用 length_suffix 和 model_suffix，统一文件命名
    output_filename = f"ranked_fusion_{config.reranker_type}_{beta_str}{k_suffix}_top{config.final_top_k}.jsonl"
    final_output_path = os.path.join(config.results_dir, config.dataset_name, output_filename)
    
    logger.info("--- Reranking Strategy: Separate Scoring & Fusion ---")
    logger.info(f"Dataset: {config.dataset_name}, Reranker: {config.reranker_type}")
    logger.info(f"Fusion Beta (Query Weight): {beta}")
    logger.info(f"Initial Top-K: {config.initial_top_k}")  # 显示K值
    logger.info(f"Profile Source File: {config.personalized_queries_path}")
    logger.info(f"Final Output: {final_output_path}")

    # 加载数据
    retrieved_data = load_retrieved_results(config.retrieved_results_path)
    profile_data = load_personalized_features(config.personalized_queries_path)
    
    if not retrieved_data:
        logger.error("No retrieved data found. Aborting.")
        return

    try:
        model, tokenizer = get_model_and_tokenizer(config, config.reranker_type)
    except Exception:
        return

    final_results = []
    
    queries_to_process = [qid for qid in retrieved_data if qid in profile_data]
    if not queries_to_process:
        logger.error("No queries with both retrieved results and a valid profile. Aborting.")
        return

    # 应用 test_query_limit
    if config.test_query_limit is not None and config.test_query_limit > 0:
        logger.info(f"应用 test_query_limit: {config.test_query_limit}")
        if config.dataset_type == "medcorpus":
            logger.info(f"MedCorpus 测试模式: 将限制处理前 {config.test_query_limit} 个主题 (topics)。")
            
            topic_to_qids = defaultdict(list)
            for qid in queries_to_process:
                try:
                    topic_id = '_'.join(qid.split('_')[:-1]) if '_' in qid else qid
                    topic_to_qids[topic_id].append(qid)
                except Exception as e:
                    logger.warning(f"无法从查询ID '{qid}' 中提取主题ID: {e}")
                    topic_to_qids[qid].append(qid)
            
            # 按主题ID排序并取前N个
            sorted_topics = sorted(topic_to_qids.keys())
            topics_to_keep = sorted_topics[:config.test_query_limit]
            
            filtered_qids = []
            for topic in topics_to_keep:
                filtered_qids.extend(topic_to_qids[topic])
            
            # 按 query_id 排序保持顺序
            filtered_qids.sort()
            
            logger.info(f"限制到前 {config.test_query_limit} 个 topics，共 {len(filtered_qids)} 条查询 (从 {len(queries_to_process)} 总数)")
            queries_to_process = filtered_qids
        else:
            # 其他数据集，直接按查询数量限制
            if len(queries_to_process) > config.test_query_limit:
                queries_to_process = queries_to_process[:config.test_query_limit]
                logger.info(f"{config.dataset_name}: 限制到前 {config.test_query_limit} 条查询")

    logger.info(f"Processing {len(queries_to_process)} queries with both retrieval and profile data.")

    for qid in tqdm(queries_to_process, desc="Fusion Reranking Queries"):
        q_info = retrieved_data[qid]
        query_text = q_info["query"]
        profile_text = profile_data[qid]
        
        # 关键修改：根据initial_top_k截断候选文档
        candidate_docs = q_info["results"][:config.initial_top_k]
        
        if not candidate_docs:
            logger.debug(f"No candidates for query {qid} after K={config.initial_top_k} truncation")
            continue

        # ===== Prompt级别融合（适合CausalLM reranker）=====
        # 根据beta权重决定如何组合query和profile
        if beta >= 0.7:
            # 更依赖原始查询
            fused_query = f"{query_text}\n\nUser Background: {profile_text[:100]}"
        elif beta >= 0.4:
            # 均衡组合
            fused_query = f"Query: {query_text}\nUser Context: {profile_text}"
        else:
            # 更依赖profile
            fused_query = f"User Research Interest: {profile_text}\n\nCurrent Query: {query_text}"
        
        # 用融合后的query进行重排
        fused_scores = compute_scores_for_context(model, tokenizer, fused_query, candidate_docs, config, "Fused")

        fused_docs = []
        for doc in candidate_docs:
            doc_id = doc["text_id"]
            final_score = fused_scores.get(doc_id, 0.0)
            
            fused_doc = doc.copy()
            fused_doc["score"] = final_score
            fused_docs.append(fused_doc)

        fused_docs.sort(key=lambda x: x["score"], reverse=True)

        final_results.append({
            "query_id": qid,
            "query": query_text,
            "ranked_results": fused_docs[:config.final_top_k]
        })

    # 保存结果
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
    with open(final_output_path, 'w', encoding='utf-8') as f:
        for item in final_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"Fusion reranking complete. Results saved to {final_output_path}")
    logger.info(f"Processed {len(final_results)} queries.")

    # 清理
    del model, tokenizer, retrieved_data, profile_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# This main function is for standalone execution, which we avoid by using run.py
# However, it's good practice to keep it and ensure it's up-to-date.
def main():
    parser = argparse.ArgumentParser(description="Run Fusion Reranking (Strategy 1)")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/workspace/PerMed/data")
    parser.add_argument("--results_dir", type=str, default="./results")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--reranker_type", type=str, default="jina", choices=["gte", "jina", "minicpm"])
    parser.add_argument("--reranker_path", type=str, help="Explicit path to the reranker model.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--initial_top_k", type=int, default=100)
    parser.add_argument("--final_top_k", type=int, default=10)
    parser.add_argument("--reranker_max_length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for the query score in fusion. Range [0.0, 1.0].")
    parser.add_argument("--personalized_text_target_length", type=int, default=250)
    parser.add_argument("--llm_api_type", type=str, default="ollama", choices=["ollama", "siliconflow"])
    parser.add_argument("--llm_model", type=str, default="llama3:8b")
    parser.add_argument("--siliconflow_model", type=str, default="Qwen/Qwen3-14B", 
                        choices=["Qwen/Qwen3-14B", "Qwen/Qwen3-32B", "Qwen/Qwen3-Next-80B-A3B-Thinking"],
                        help="SiliconFlow model to use (qwen3-14b or qwen3-32b)")

    args = parser.parse_args()
    if not (0.0 <= args.beta <= 1.0):
        raise ValueError("Beta must be between 0.0 and 1.0.")

    config = get_config()
    config.update(args)
    run_fusion_reranking(config, args.beta)

if __name__ == "__main__":
    main()
