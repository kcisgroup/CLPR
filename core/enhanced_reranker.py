#!/usr/bin/env python3
"""
增强版重排器 - 测试不同的输入组合策略
对比3种方案：
1. profile + query (baseline)
2. profile + query + memory_keywords (增强)
3. query + memory_keywords (无profile)
"""

import json
import logging
import torch
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from utils import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedReranker:
    """增强版重排器 - 支持多种输入策略"""

    def __init__(self, config, strategy="enhanced"):
        """
        Args:
            strategy: 输入策略
                - "baseline": profile + query
                - "enhanced": profile + query + memory_keywords
                - "memory_only": query + memory_keywords (无profile)
        """
        self.config = config
        self.strategy = strategy
        self.model = None

    def load_model(self):
        """加载Jina-Reranker-v3"""
        logger.info(f"Loading Jina-Reranker-v3 from {self.config.reranker_path}")

        import sys
        model_path = str(self.config.reranker_path)
        if model_path not in sys.path:
            sys.path.insert(0, model_path)

        from modeling import JinaForRanking
        from transformers import AutoConfig

        device = torch.device(self.config.device)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.model = JinaForRanking.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
        ).to(device).eval()

        logger.info("✅ Jina-Reranker-v3 loaded")

    def format_input(self, query: str, profile: str, memory_features: List[str], strategy: str = None) -> str:
        """根据策略格式化输入"""
        strategy = strategy or self.strategy

        # 提取关键记忆词（前5个最相关的）
        memory_keywords = []
        for feat in memory_features[:5]:
            # 去掉标签，只保留内容
            if ']' in feat:
                content = feat.split(']', 1)[1].strip()
                memory_keywords.append(content)

        if strategy == "baseline":
            # 方案1: 只用profile + query
            return f"{profile}\n\nCurrent Query: {query}"

        elif strategy == "enhanced":
            # 方案2: profile + query + memory_keywords
            memory_text = ""
            if memory_keywords:
                memory_text = f"\n\nKey Context: {'; '.join(memory_keywords[:3])}"
            return f"{profile}\n\nCurrent Query: {query}{memory_text}"

        elif strategy == "memory_only":
            # 方案3: query + memory_keywords (无profile)
            memory_text = ""
            if memory_keywords:
                memory_text = f"\nContext from history: {'; '.join(memory_keywords[:3])}"
            return f"Query: {query}{memory_text}"

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def rerank_documents(self, query: str, profile: str, memory_features: List[str],
                        documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重排文档"""
        if not documents:
            return []

        # 构建输入
        rerank_query = self.format_input(query, profile, memory_features)

        # 提取文档文本
        doc_texts = [doc.get("text", "")[:2048] for doc in documents]

        # Jina批量重排
        results = self.model.rerank(rerank_query, doc_texts)

        # 构建结果
        scored_docs = []
        for result in results:
            idx = result['index']
            doc = documents[idx].copy()
            doc["rerank_score"] = result['relevance_score']
            scored_docs.append(doc)

        return scored_docs


def run_enhanced_reranking(strategy: str = "enhanced"):
    """运行增强版重排"""
    # 清理GPU显存
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    config = get_config()
    config.dataset_name = "MedCorpus"
    config.reranker_path = "/mnt/data/zsy-data/PerMed/model/jina-reranker-v3"
    config.reranker_type = "jina-v3"
    config.initial_top_k = 200
    config.final_top_k = 10
    config.results_dir = "/mnt/data/zsy-data/PerMed/results"
    config.device = "cuda:0"  # 明确指定GPU

    logger.info("=" * 70)
    logger.info(f"🚀 Enhanced Reranking - Strategy: {strategy}")
    logger.info(f"   Device: {config.device}")
    logger.info("=" * 70)

    # 路径设置
    dataset_dir = Path("/mnt/data/zsy-data/PerMed/data/MedCorpus_MultiTurn")
    results_dir = Path("/mnt/data/zsy-data/PerMed/results/MedCorpus")

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
            for doc in data['results']:
                text_id = doc['text_id']
                if text_id in corpus_data:
                    doc['text'] = corpus_data[text_id]['text']
                    if 'title' not in doc or not doc['title']:
                        doc['title'] = corpus_data[text_id]['title']
            retrieved_data[data['query_id']] = data

    logger.info(f"Loaded {len(retrieved_data)} queries")

    # 加载个性化profiles
    profile_file = results_dir / "personalized_queries_qwen3-14b.jsonl"
    logger.info(f"Loading profiles from {profile_file}")

    profile_data = {}
    query_data = {}
    with open(profile_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            qid = data['query_id']
            profile_data[qid] = data['personalized_features']
            query_data[qid] = data['query']

    logger.info(f"Loaded {len(profile_data)} profiles")

    # 加载认知特征（包含原始记忆）
    cognitive_file = results_dir / "cognitive_features_detailed.jsonl"
    logger.info(f"Loading cognitive features from {cognitive_file}")

    memory_data = {}
    with open(cognitive_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            qid = data['query_id']
            memory_data[qid] = data.get('tagged_memory_features', [])

    logger.info(f"Loaded {len(memory_data)} memory features")

    # 初始化重排器
    reranker = EnhancedReranker(config, strategy=strategy)
    reranker.load_model()

    # 执行重排
    final_results = []
    queries_to_process = list(retrieved_data.keys())

    for qid in tqdm(queries_to_process, desc=f"Reranking ({strategy})"):
        if qid not in profile_data or qid not in query_data:
            continue

        q_info = retrieved_data[qid]
        query_text = query_data[qid]
        profile_text = profile_data[qid]
        memory_features = memory_data.get(qid, [])
        candidate_docs = q_info["results"][:config.initial_top_k]

        if not candidate_docs:
            continue

        # 重排
        reranked_docs = reranker.rerank_documents(
            query=query_text,
            profile=profile_text,
            memory_features=memory_features,
            documents=candidate_docs
        )

        # 准备输出
        final_docs = []
        for doc in reranked_docs[:config.final_top_k]:
            output_doc = {
                "text_id": doc["text_id"],
                "score": float(doc["rerank_score"]),  # 转换为Python float
                "text": doc.get("text", "")
            }
            final_docs.append(output_doc)

        final_results.append({
            "query_id": qid,
            "query": query_text,
            "ranked_results": final_docs
        })

    # 保存结果
    output_file = results_dir / f"ranked_jina-v3_{strategy}_top{config.final_top_k}.jsonl"
    logger.info(f"Saving results to {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in final_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    logger.info("=" * 70)
    logger.info(f"✅ Reranking completed! Processed {len(final_results)} queries")
    logger.info(f"📄 Results saved to: {output_file}")
    logger.info("=" * 70)

    # 清理
    del reranker.model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        strategy = sys.argv[1]
    else:
        strategy = "enhanced"

    if strategy not in ["baseline", "enhanced", "memory_only"]:
        print("Usage: python enhanced_reranker.py [baseline|enhanced|memory_only]")
        sys.exit(1)

    run_enhanced_reranking(strategy)
