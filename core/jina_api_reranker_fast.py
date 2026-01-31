#!/usr/bin/env python3
"""
Jina Reranker V3 API 重排器 - 并发加速版
支持 RPM=500 的并发请求，大幅加速重排过程
"""

import json
import requests
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Jina API 配置
JINA_API_KEY = "REDACTED_JINA_API_KEY"
JINA_API_URL = "https://api.jina.ai/v1/rerank"


class RateLimiter:
    """简单的速率限制器，确保不超过 RPM 限制"""
    def __init__(self, rpm: int = 450):  # 留一些余量，设置为 450 而不是 500
        self.min_interval = 60.0 / rpm  # 每个请求的最小间隔
        self.lock = threading.Lock()
        self.last_request_time = 0
    
    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request_time = time.time()


def call_jina_rerank(
    query: str,
    documents: List[str],
    top_n: int = 10,
    rate_limiter: RateLimiter = None,
    max_retries: int = 3
) -> List[Dict]:
    """调用 Jina Reranker API"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    
    payload = {
        "model": "jina-reranker-v3",
        "query": query,
        "top_n": top_n,
        "documents": documents,
        "return_documents": False
    }
    
    for attempt in range(max_retries):
        try:
            if rate_limiter:
                rate_limiter.wait()
            
            response = requests.post(JINA_API_URL, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 429:  # Rate limit exceeded
                wait_time = 2 ** attempt
                logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            
            response.raise_for_status()
            result = response.json()
            return result.get("results", [])
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
            else:
                raise
    
    return []


def process_single_query(
    qid: str,
    profile_data: Dict,
    candidate_docs: List[Dict],
    corpus: Dict,
    rerank_mode: str,
    top_k: int,
    rate_limiter: RateLimiter
) -> Tuple[str, Dict]:
    """处理单个查询的重排"""
    
    # 构建查询文本
    if rerank_mode == "profile_only":
        query_text = profile_data['profile']
    elif rerank_mode == "query_only":
        query_text = profile_data['query']
    else:  # profile_and_query
        query_text = f"Research Profile: {profile_data['profile']}\n\nCurrent Query: {profile_data['query']}"
    
    # 构建文档列表
    doc_texts = []
    doc_ids = []
    for doc in candidate_docs:
        text_id = str(doc['text_id'])
        if text_id in corpus:
            doc_info = corpus[text_id]
            title = doc_info.get('title', '')
            text = doc_info.get('text', '')
            content = f"{title}\n{text}"[:2000]
            doc_texts.append(content)
            doc_ids.append(text_id)
    
    if not doc_texts:
        return qid, None
    
    try:
        rerank_results = call_jina_rerank(query_text, doc_texts, top_n=top_k, rate_limiter=rate_limiter)
        
        ranked_docs = []
        for r in rerank_results:
            idx = r['index']
            score = r['relevance_score']
            if idx < len(doc_ids):
                ranked_docs.append({
                    "text_id": doc_ids[idx],
                    "score": float(score),
                    "text": doc_texts[idx][:500]
                })
        
        return qid, {
            "query_id": qid,
            "query": profile_data['query'],
            "ranked_results": ranked_docs
        }
    except Exception as e:
        logger.warning(f"重排 {qid} 失败: {e}")
        return qid, None


def load_corpus(corpus_file: Path, needed_ids: set = None) -> Dict[str, Dict]:
    """加载文档库"""
    corpus = {}
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text_id = str(data['text_id'])
            if needed_ids is None or text_id in needed_ids:
                corpus[text_id] = data
    return corpus


def load_retrieved_results(retrieved_file: Path) -> Dict[str, Dict]:
    """加载检索结果"""
    results = {}
    with open(retrieved_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            qid = str(data['query_id'])
            results[qid] = data
    return results


def load_profiles(profile_file: Path) -> Dict[str, Dict]:
    """加载个性化 profile"""
    profiles = {}
    with open(profile_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            qid = str(data['query_id'])
            profiles[qid] = {
                'query': data.get('query', ''),
                'profile': data.get('personalized_features', '')
            }
    return profiles


def run_jina_api_reranking_fast(
    dataset_name: str,
    model_name: str,
    rerank_mode: str = "profile_only",
    top_k: int = 10,
    results_dir: str = "/mnt/data/zsy-data/PerMed/results",
    data_dir: str = "/mnt/data/zsy-data/PerMed/data",
    rpm: int = 450,  # 保守设置，留余量
    max_workers: int = 8  # 并发线程数
):
    """
    执行 Jina API 重排（并发版）
    """
    results_path = Path(results_dir) / dataset_name
    
    # 确定文件路径
    if dataset_name == "MedCorpus":
        corpus_file = Path(data_dir) / "MedCorpus_MultiTurn" / "corpus.jsonl"
    else:
        corpus_file = Path(data_dir) / dataset_name / "corpus.jsonl"
    
    retrieved_file = results_path / "retrieved.jsonl"
    
    model_suffix = model_name.split("/")[-1].lower().replace(":", "-")
    profile_file = results_path / f"personalized_queries_{model_suffix}.jsonl"
    
    mode_name = rerank_mode.replace('_', '-')
    output_file = results_path / f"ranked_jina-v3_{mode_name}_{model_suffix}_top{top_k}.jsonl"
    
    logger.info(f"=" * 70)
    logger.info(f"Jina API Reranking (并发加速版)")
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Mode: {rerank_mode}")
    logger.info(f"  RPM limit: {rpm}")
    logger.info(f"  Workers: {max_workers}")
    logger.info(f"  Profile file: {profile_file}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"=" * 70)
    
    # 检查文件
    for f, name in [(profile_file, "Profile"), (retrieved_file, "检索结果"), (corpus_file, "Corpus")]:
        if not f.exists():
            logger.error(f"{name}文件不存在: {f}")
            return
    
    # 加载数据
    logger.info("加载数据...")
    retrieved_data = load_retrieved_results(retrieved_file)
    profiles = load_profiles(profile_file)
    
    needed_doc_ids = set()
    for qid, data in retrieved_data.items():
        for doc in data.get('results', [])[:100]:
            needed_doc_ids.add(str(doc['text_id']))
    
    corpus = load_corpus(corpus_file, needed_doc_ids)
    logger.info(f"  查询数: {len(retrieved_data)}, Profile数: {len(profiles)}, 文档数: {len(corpus)}")
    
    # 按 topic 分组（保持组内顺序）
    if dataset_name == "MedCorpus":
        topic_groups = {}
        for qid in retrieved_data.keys():
            if qid.startswith("topic_"):
                parts = qid.split("_")
                if len(parts) >= 2:
                    topic_id = f"topic_{parts[1]}"
                    if topic_id not in topic_groups:
                        topic_groups[topic_id] = []
                    topic_groups[topic_id].append(qid)
        
        # 组内按 turn_id 排序
        for topic_id in topic_groups:
            topic_groups[topic_id].sort(key=lambda x: int(x.split("_")[-1]) if x.split("_")[-1].isdigit() else 0)
        
        # 按 topic 顺序排列
        sorted_topics = sorted(topic_groups.keys(), key=lambda x: int(x.split("_")[1]))
        query_ids = []
        for topic_id in sorted_topics:
            query_ids.extend(topic_groups[topic_id])
        
        logger.info(f"  MedCorpus: {len(topic_groups)} topics, {len(query_ids)} queries")
    else:
        query_ids = list(retrieved_data.keys())
    
    # 过滤有效查询
    valid_queries = [(qid, profiles[qid], retrieved_data[qid].get('results', [])[:100]) 
                     for qid in query_ids if qid in profiles]
    
    logger.info(f"  有效查询: {len(valid_queries)}")
    
    # 创建速率限制器
    rate_limiter = RateLimiter(rpm=rpm)
    
    # 并发执行
    final_results = {}
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for qid, profile_data, candidate_docs in valid_queries:
            future = executor.submit(
                process_single_query,
                qid, profile_data, candidate_docs, corpus,
                rerank_mode, top_k, rate_limiter
            )
            futures[future] = qid
        
        with tqdm(total=len(futures), desc="Jina API Reranking") as pbar:
            for future in as_completed(futures):
                qid, result = future.result()
                if result:
                    final_results[qid] = result
                else:
                    failed_count += 1
                pbar.update(1)
    
    # 按原始顺序排列结果
    ordered_results = [final_results[qid] for qid in query_ids if qid in final_results]
    
    # 保存结果
    logger.info(f"保存结果到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in ordered_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"=" * 70)
    logger.info(f"✅ 完成！处理了 {len(ordered_results)} 个查询，失败 {failed_count} 个")
    logger.info(f"📄 结果保存到: {output_file}")
    logger.info(f"=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Jina API Reranker (Fast)")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["MedCorpus", "LitSearch"])
    parser.add_argument("--model", type=str, required=True,
                        help="如 Qwen/Qwen3-Next-80B-A3B-Thinking")
    parser.add_argument("--mode", type=str, default="profile_only",
                        choices=["profile_only", "query_only", "profile_and_query"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--results_dir", type=str, 
                        default="/mnt/data/zsy-data/PerMed/results")
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/data/zsy-data/PerMed/data")
    parser.add_argument("--rpm", type=int, default=450,
                        help="API RPM 限制（建议设为实际限制的 90%）")
    parser.add_argument("--workers", type=int, default=8,
                        help="并发线程数")
    
    args = parser.parse_args()
    
    run_jina_api_reranking_fast(
        dataset_name=args.dataset,
        model_name=args.model,
        rerank_mode=args.mode,
        top_k=args.top_k,
        results_dir=args.results_dir,
        data_dir=args.data_dir,
        rpm=args.rpm,
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
