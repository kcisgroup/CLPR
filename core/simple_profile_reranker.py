#!/usr/bin/env python3
"""
简化的个性化重排器
直接使用profile + query的简单prompt策略，不做复杂融合
"""

import json
import logging
import torch
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoConfig

from utils import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleProfileReranker:
    """简化的个性化重排器"""
    
    def __init__(self, config=None, rerank_mode="profile_and_query"):
        """
        Args:
            config: 配置对象
            rerank_mode: 重排模式
                - "profile_only": 仅使用个性化背景
                - "query_only": 仅使用原始查询
                - "profile_and_query": 同时使用背景和查询（默认）
        """
        self.config = config or get_config()
        self.model = None
        self.rerank_mode = rerank_mode
        
    def load_model(self):
        """加载 Jina-Reranker-v3 模型"""
        logger.info(f"Loading Jina reranker from {self.config.reranker_path}")

        import sys
        import os
        model_path = str(self.config.reranker_path)
        if model_path not in sys.path:
            sys.path.insert(0, model_path)

        from modeling import JinaForRanking

        device = self.config.device if isinstance(self.config.device, torch.device) else torch.device(self.config.device)

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = JinaForRanking.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
        ).to(device).eval()

        self.reranker_type = 'jina'
        logger.info("✅ Model loaded successfully")
    
    def format_prompt(self, query: str, profile: str, doc_content: str) -> str:
        """
        构建简单清晰的重排prompt
        
        Args:
            query: 当前查询
            profile: 用户个性化背景
            doc_content: 候选文档内容
            
        Returns:
            格式化的prompt字符串
        """
        if self.rerank_mode == "profile_only":
            # 仅使用个性化背景
            instruction = (
                "Given a research topic description that captures a user's current investigation focus, "
                "determine if the following document is highly relevant to this research direction. "
                "The research description integrates the user's domain expertise with their current inquiry. "
                "Match the document against both the research domain and the specific technical focus described."
            )
            prompt = f"""<Instruct>: {instruction}

<Research Focus>: {profile}

<Document>: {doc_content}"""
        
        elif self.rerank_mode == "query_only":
            # 仅使用原始查询 - 使用官方默认极简instruction
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
            
            prompt = f"""<Instruct>: {instruction}

<Query>: {query}

<Document>: {doc_content}"""
        
        else:  # profile_and_query
            # 同时使用背景和查询
            instruction = (
                "You are an expert in personalized scientific literature recommendation. "
                "Given a researcher's background and their current search query, determine if the document "
                "is relevant. Prioritize documents that: (1) directly answer the current query, AND "
                "(2) align with the researcher's expertise and interests for deeper understanding. "
                "Focus on personalized relevance beyond simple keyword matching."
            )
            prompt = f"""<Instruct>: {instruction}

<Researcher Background & Interests>: {profile}

<Current Query>: {query}

<Document>: {doc_content}"""
        
        return prompt
    
    def rerank_documents(
        self,
        query: str,
        profile: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        对候选文档进行重排
        
        Args:
            query: 查询文本
            profile: 用户个性化背景
            documents: 候选文档列表
            
        Returns:
            重排后的文档列表
        """
        # Jina-Reranker批量重排
        doc_texts = [doc.get("text", "")[:2048] for doc in documents]

        # 根据模式构建query
        if self.rerank_mode == "profile_only":
            rerank_query = profile
        elif self.rerank_mode == "query_only":
            rerank_query = query
        else:  # profile_and_query
            # 组合Profile和Query
            rerank_query = f"{profile}\n\nCurrent Query: {query}"

        # Jina批量重排
        results = self.model.rerank(rerank_query, doc_texts)

        # 根据Jina返回的顺序重排原文档
        scored_docs = []
        for result in results:
            idx = result['index']
            doc = documents[idx].copy()
            doc["rerank_score"] = result['relevance_score']
            scored_docs.append(doc)

        return scored_docs


def run_simple_profile_reranking(config, top_k: int = 10, rerank_mode: str = "profile_and_query"):
    """
    运行简化的个性化重排
    
    Args:
        config: 配置对象
        top_k: 返回top-k结果
        rerank_mode: 重排模式 ("profile_only", "query_only", "profile_and_query")
    """
    logger.info("=" * 70)
    logger.info(f"🚀 Starting Simple Profile-based Reranking (mode: {rerank_mode})")
    logger.info("=" * 70)
    
    # 数据路径
    # MedCorpus是多轮数据集，LitSearch是单轮数据集
    if config.dataset_name == "MedCorpus":
        dataset_dir = Path(config.base_data_dir) / f"{config.dataset_name}_MultiTurn"
    else:
        dataset_dir = Path(config.base_data_dir) / config.dataset_name
    
    results_dir = Path(config.results_dir) / config.dataset_name
    
    # 加载文档库
    corpus_file = dataset_dir / "corpus.jsonl"
    logger.info(f"Loading corpus from {corpus_file}")
    
    corpus_data = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            corpus_data[data['text_id']] = {
                'title': data.get('title', ''),
                'text': data.get('text', '')
            }
    
    logger.info(f"Loaded {len(corpus_data)} documents")
    
    # 加载检索结果
    retrieved_file = results_dir / "retrieved.jsonl"
    logger.info(f"Loading retrieved results from {retrieved_file}")
    
    retrieved_data = {}
    with open(retrieved_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # 将corpus中的文档内容添加到retrieved结果中
            for doc in data['results']:
                text_id = doc['text_id']
                if text_id in corpus_data:
                    doc['text'] = corpus_data[text_id]['text']
                    if 'title' not in doc or not doc['title']:
                        doc['title'] = corpus_data[text_id]['title']
            retrieved_data[data['query_id']] = data
    
    logger.info(f"Loaded {len(retrieved_data)} queries with document content")
    
    # 加载个性化profiles和queries
    # 不再使用长度后缀，统一文件命名
    profile_file = results_dir / f"personalized_queries{config.model_suffix}.jsonl"
    logger.info(f"Loading profiles from {profile_file}")
    
    profile_data = {}
    query_data = {}
    with open(profile_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            qid = data['query_id']
            profile_data[qid] = data['personalized_features']
            query_data[qid] = data['query']
    
    logger.info(f"Loaded {len(profile_data)} profiles and queries")
    
    # 筛选查询（如果有限制）
    queries_to_process = list(retrieved_data.keys())
    
    if hasattr(config, 'test_query_limit') and config.test_query_limit and config.test_query_limit > 0:
        # 按topic筛选
        if config.dataset_name == "MedCorpus":
            topic_queries = {}
            for qid in queries_to_process:
                if qid.startswith("topic_"):
                    parts = qid.split("_")
                    if len(parts) >= 2 and parts[1].isdigit():
                        topic_num = int(parts[1])
                        if topic_num not in topic_queries:
                            topic_queries[topic_num] = []
                        topic_queries[topic_num].append(qid)
            
            # 选择前N个topics
            selected_topics = sorted(topic_queries.keys())[:config.test_query_limit]
            queries_to_process = []
            for topic in selected_topics:
                queries_to_process.extend(topic_queries[topic])
            
            logger.info(f"Limited to first {config.test_query_limit} topics ({len(queries_to_process)} queries)")
        else:
            queries_to_process = queries_to_process[:config.test_query_limit]
            logger.info(f"Limited to first {config.test_query_limit} queries")
    
    # 初始化重排器
    reranker = SimpleProfileReranker(config, rerank_mode=rerank_mode)
    reranker.load_model()
    
    # 执行重排
    final_results = []
    
    for qid in tqdm(queries_to_process, desc="Simple Profile Reranking"):
        if qid not in profile_data or qid not in query_data:
            logger.warning(f"No profile or query for {qid}, skipping")
            continue
        
        q_info = retrieved_data[qid]
        query_text = query_data[qid]
        profile_text = profile_data[qid]
        candidate_docs = q_info["results"][:config.initial_top_k]
        
        if not candidate_docs:
            logger.debug(f"No candidates for query {qid}")
            continue
        
        # 重排
        reranked_docs = reranker.rerank_documents(
            query=query_text,
            profile=profile_text,
            documents=candidate_docs
        )
        
        # 准备输出
        final_docs = []
        for doc in reranked_docs[:top_k]:
            output_doc = {
                "text_id": doc["text_id"],
                "score": float(doc.get("rerank_score", 0.0)),
                "text": doc.get("text", "")
            }
            final_docs.append(output_doc)
        
        final_results.append({
            "query_id": qid,
            "query": query_text,
            "ranked_results": final_docs
        })
    
    # 保存结果
    # 统一文件命名格式: ranked_{reranker}_{mode}_top{k}.jsonl
    reranker_name = getattr(config, 'reranker_type', 'qwen3')
    mode_name = rerank_mode.replace('_', '-')
    model_suffix = getattr(config, "model_suffix", "")
    suffix_part = model_suffix if model_suffix else ""
    output_file = results_dir / f"ranked_{reranker_name}_{mode_name}{suffix_part}_top{top_k}.jsonl"
    logger.info(f"Saving results to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in final_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info("=" * 70)
    logger.info(f"✅ Reranking completed! Processed {len(final_results)} queries")
    logger.info(f"📄 Results saved to: {output_file}")
    logger.info("=" * 70)
    
    # 清理
    if hasattr(reranker, "model"):
        del reranker.model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    config = get_config()
    run_simple_profile_reranking(config, top_k=10)
