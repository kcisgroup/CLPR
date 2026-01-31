#!/usr/bin/env python3
"""
正确实现PBR (Personalize Before Retrieve) - ACL 2025

完整包含：
1. P-PRF: LLM生成个性化查询扩展（异步批量调用）
2. P-Anchor: 基于图PageRank的记忆检索
3. 组合检索: 使用扩展查询+图中心+reasoning
"""

import json
import numpy as np
import faiss
import os
import sys
import requests
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import argparse
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class PBRRetriever:
    """
    完整的PBR实现
    
    参考: /workspace/PBR-code/src/retrieval/retrieval_PBR.py
    """
    
    def __init__(self,
                 corpus_ids: List[str],
                 corpus_texts: List[str],
                 corpus_embeddings: np.ndarray,
                 retriever_model_path: str,
                 llm_api_key: str = None,
                 device: str = "cuda"):
        
        self.corpus_ids = corpus_ids
        self.corpus_texts = corpus_texts
        self.embeddings = corpus_embeddings
        self.device = device
        
        print(f"🚀 初始化PBR Retriever (ACL 2025)")
        print(f"   文档数: {len(corpus_ids)}")
        print(f"   Embedding维度: {corpus_embeddings.shape[1]}")
        
        # 1. 加载embedding模型
        print(f"   加载embedding模型: {retriever_model_path}")
        self.retriever_model = SentenceTransformer(
            retriever_model_path,
            device=device,
            trust_remote_code=True
        )
        
        # 2. 创建FAISS索引
        print(f"   创建FAISS索引...")
        self.index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
        self.index.add(corpus_embeddings)
        print(f"   ✅ FAISS索引: {self.index.ntotal} 向量")
        
        # 3. LLM配置（用于P-PRF）
        self.llm_api_key = llm_api_key or ""
        self.llm_api_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.llm_model = "Qwen/Qwen3-14B"
        print(f"   LLM: {self.llm_model} (用于P-PRF)")
        
        # PBR参数
        self.sim_threshold = 0.75
        self.damping_factor = 0.85
    
    async def _call_llm_async(self, session: aiohttp.ClientSession, prompt: str, max_tokens: int = 2048) -> str:
        """异步调用LLM"""
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        retry = 0
        while retry < 3:
            try:
                async with session.post(
                    self.llm_api_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result["choices"][0]["message"]["content"].strip()
                    else:
                        print(f"   ⚠️ LLM调用失败 (status {response.status})")
                        retry += 1
                        await asyncio.sleep(1)
            except Exception as e:
                retry += 1
                await asyncio.sleep(1)
                if retry >= 3:
                    print(f"   ⚠️ LLM调用失败: {e}")
        return ""
    
    def _call_llm_sync(self, prompt: str, max_tokens: int = 2048) -> str:
        """同步串行调用LLM（避免限流）"""
        headers = {
            "Authorization": f"Bearer {self.llm_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        for retry in range(3):
            try:
                response = requests.post(
                    self.llm_api_url,
                    json=payload,
                    headers=headers,
                    timeout=60
                )
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"].strip()
                elif response.status_code == 429:
                    wait_time = 5 * (retry + 1)  # 递增等待时间
                    print(f"   ⚠️ 限流中，等待{wait_time}秒...")
                    time.sleep(wait_time)
                else:
                    print(f"   ⚠️ API错误 {response.status_code}")
                    time.sleep(2)
            except Exception as e:
                print(f"   ⚠️ 请求失败: {e}")
                time.sleep(2)
        
        return ""
    
    async def _batch_call_llm_mini_batch(self, prompts: List[str], max_tokens: int = 2048, 
                                          batch_size: int = 10, desc: str = "") -> List[str]:
        """小批量并行调用LLM（批次内并行，批次间串行，避免限流）"""
        all_responses = []
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        print(f"   {desc}: 共{len(prompts)}个请求，分{num_batches}批，每批{batch_size}个并发")
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"   批次 {batch_num}/{num_batches} ({len(batch)}个请求)...", end='', flush=True)
            
            # 批次内并行调用
            async with aiohttp.ClientSession() as session:
                tasks = [self._call_llm_async(session, prompt, max_tokens) for prompt in batch]
                responses = await asyncio.gather(*tasks)
            
            all_responses.extend(responses)
            print(f" 完成")
            
            # 批次间等待，避免限流
            if i + batch_size < len(prompts):
                await asyncio.sleep(3)  # 批次间隔3秒
        
        return all_responses
    
    def _build_prompts(self, query: str, history: List[str]) -> Tuple[str, str]:
        """
        构建P-PRF的两个prompts（但不立即调用）
        
        Returns:
            prompt_fake: 用于生成10个查询变体的prompt
            prompt_reason: 用于生成reasoning的prompt
        """
        history_text = "\n".join(history) if history else "No previous dialogue"
        
        prompt_fake = f"""You are to generate 10 natural candidate utterances for medical literature search, inspired by the dialogue history and the current question.

Context
------------
User dialogue history (for style imitation):  
{history_text}

Current question (to inspire the utterances):  
{query}
------------

Guidelines
1. Generate 10 fluent, natural search queries the user might plausibly say.
2. Do NOT just paraphrase; include variations in medical terminology, specificity, or context.
3. Each query > 25 words.
4. Preserve medical terms and concepts.
5. Return ONLY valid JSON in this format (no comments, no markdown):
   {{
     "candidates": [
       "query variation 1...",
       "query variation 2...",
       ...
     ]
   }}
"""
        
        prompt_reason = f"""Solve the medical literature search question step-by-step, inspired by the dialogue history.

Context
------------
User dialogue history (for context):  
{history_text}

Current question:  
{query}
------------
Output (step-by-step reasoning, 2-3 sentences):
"""
        
        return prompt_fake, prompt_reason
    
    def _parse_fake_queries(self, fake_result: str, query: str, history: List[str]) -> List[str]:
        """解析LLM生成的fake queries"""
        fake_queries = []
        try:
            import re
            match = re.search(r'\{.*?\}', fake_result, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                fake_queries = data.get('candidates', [])[:10]
        except:
            pass
        
        if not fake_queries:
            # Fallback: 使用历史作为扩展
            fake_queries = history[:3] if history else [query]
        
        return fake_queries
    
    def _build_memory_graph(self, history_texts: List[str]) -> np.ndarray:
        """
        P-Anchor模块: 构建记忆图并计算图中心
        
        基于PBR原始实现的_build_memory_graph和_mem_pagerank
        """
        if not history_texts:
            return None
        
        # 编码历史
        history_embeddings = self.retriever_model.encode(
            history_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        n = len(history_embeddings)
        if n == 1:
            return history_embeddings[0]
        
        # 构建相似度矩阵
        sim_matrix = np.dot(history_embeddings, history_embeddings.T)
        adjacency = (sim_matrix >= self.sim_threshold).astype(float)
        np.fill_diagonal(adjacency, 0)
        
        # PageRank
        out_degree = adjacency.sum(axis=1)
        out_degree[out_degree == 0] = 1
        
        pi = np.ones(n) / n
        for _ in range(50):
            pi_new = (1 - self.damping_factor) * (adjacency.T @ (pi / out_degree)) + self.damping_factor / n
            if np.abs(pi_new - pi).max() < 1e-6:
                break
            pi = pi_new
        
        pi = pi / pi.sum()
        graph_center = np.dot(pi, history_embeddings)
        
        return graph_center
    
    def retrieve_pbr(self, query: str, history: List[str], 
                     fake_queries: List[str] = None, reasoning: str = None,
                     top_k: int = 10, use_llm: bool = True) -> List[Dict]:
        """
        PBR完整检索流程
        
        Args:
            query: 当前查询
            history: 对话历史
            fake_queries: 预先生成的查询扩展（如果None，则不使用P-PRF）
            reasoning: 预先生成的推理（如果None，则不使用）
            top_k: 返回数量
            use_llm: 是否使用LLM扩展（P-PRF）
        """
        # Step 1: 编码原始查询
        q_embedding = self.retriever_model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Step 2: 构建记忆图中心（P-Anchor）
        g_embedding = self._build_memory_graph(history) if history else np.zeros_like(q_embedding)
        
        if use_llm and fake_queries is not None and reasoning is not None:
            # Step 3: 编码扩展查询（P-PRF）
            if fake_queries:
                prf_embeddings = self.retriever_model.encode(
                    fake_queries,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                prf_embd_mean = prf_embeddings.mean(axis=0)
            else:
                prf_embd_mean = np.zeros_like(q_embedding)
            
            # 编码reasoning
            if reasoning:
                reason_embd = self.retriever_model.encode(
                    reasoning,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
            else:
                reason_embd = np.zeros_like(q_embedding)
            
            # Step 4: PBR核心公式（来自原始代码）
            avg_qg = (q_embedding + g_embedding) / 2
            w1 = 1 + cosine_similarity(avg_qg[None,:], prf_embd_mean[None,:])[0,0]
            w2 = 1 + cosine_similarity(avg_qg[None,:], reason_embd[None,:])[0,0]
            
            final_query_embedding = q_embedding + g_embedding + w1 * prf_embd_mean + w2 * reason_embd
        
        else:
            # 简化版：不使用LLM扩展
            final_query_embedding = q_embedding + g_embedding
        
        # 归一化
        norm = np.linalg.norm(final_query_embedding)
        if norm > 1e-12:
            final_query_embedding = final_query_embedding / norm
        
        # Step 5: FAISS检索
        scores, indices = self.index.search(
            final_query_embedding.reshape(1, -1).astype(np.float32),
            top_k
        )
        
        # Step 6: 构造结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.corpus_ids):
                results.append({
                    "text_id": self.corpus_ids[idx],
                    "score": float(score),
                    "rank": len(results) + 1
                })
        
        return results


async def run_pbr_mini_batch(dataset: str, use_llm: bool = True):
    """运行PBR baseline - 使用小批量并行LLM调用（避免限流）"""
    
    print("="*80)
    print(f"PBR Baseline (ACL 2025) - {'完整版(P-PRF+P-Anchor) [小批量并行]' if use_llm else '简化版(P-Anchor)'}")
    print("="*80)
    
    # 路径
    data_dir = Path("/workspace/PerMed/baselines/data") / dataset
    output_dir = Path("/workspace/PerMed/baselines/results") / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 加载corpus
    print(f"\n📚 加载corpus...")
    corpus_ids, corpus_texts = [], []
    with open(data_dir / "corpus.jsonl", 'r') as f:
        for line in f:
            doc = json.loads(line.strip())
            corpus_ids.append(doc['text_id'])
            corpus_texts.append(f"{doc['title']}。{doc['text']}")
    print(f"   ✅ {len(corpus_ids)} 个文档")
    
    # 2. 加载embeddings
    print(f"   加载embeddings...")
    embeddings = np.load(data_dir / "corpus_embeddings_qwen3.npy").astype(np.float32)
    print(f"   ✅ Shape: {embeddings.shape}")
    
    # 3. 初始化PBR
    pbr = PBRRetriever(
        corpus_ids=corpus_ids,
        corpus_texts=corpus_texts,
        corpus_embeddings=embeddings,
        retriever_model_path="/workspace/PerMed/model/Qwen3-Embedding-0.6B",
        device="cuda"
    )
    
    # 4. 加载查询
    print(f"\n📖 加载查询...")
    queries = []
    with open(data_dir / "queries.jsonl", 'r') as f:
        for line in f:
            queries.append(json.loads(line.strip()))
    
    # 统计对话组（如果有的话）
    try:
        conversations = set(q.get('conversation_id', q['query_id']) for q in queries)
        print(f"   ✅ {len(queries)} 个查询 ({len(conversations)} 组对话)")
    except:
        print(f"   ✅ {len(queries)} 个查询")
    
    # 5. 批量异步生成（如果使用LLM）
    fake_queries_list = [None] * len(queries)
    reasoning_list = [None] * len(queries)
    
    if use_llm:
        print(f"\n🤖 批量异步生成 (P-PRF)...")
        print(f"   构建prompts...")
        
        # 构建所有prompts
        fake_prompts = []
        reason_prompts = []
        for query_item in queries:
            query = query_item['query']
            history = query_item['history']
            prompt_fake, prompt_reason = pbr._build_prompts(query, history)
            fake_prompts.append(prompt_fake)
            reason_prompts.append(prompt_reason)
        
        print(f"   小批量并行调用LLM...")
        fake_responses = await pbr._batch_call_llm_mini_batch(
            fake_prompts, max_tokens=2048, batch_size=20, desc="生成fake queries"
        )
        
        reason_responses = await pbr._batch_call_llm_mini_batch(
            reason_prompts, max_tokens=512, batch_size=20, desc="生成reasoning"
        )
        
        print(f"   解析结果...")
        for i, (query_item, fake_res, reason_res) in enumerate(zip(queries, fake_responses, reason_responses)):
            fake_queries_list[i] = pbr._parse_fake_queries(fake_res, query_item['query'], query_item['history'])
            reasoning_list[i] = reason_res if reason_res else query_item['query']
        
        print(f"   ✅ 批量生成完成！")
    
    # 6. 运行PBR检索
    print(f"\n🔍 PBR检索...")
    results = []
    for i, query_item in enumerate(tqdm(queries, desc="PBR检索")):
        query_id = query_item['query_id']
        query = query_item['query']
        history = query_item['history']
        
        try:
            retrieved = pbr.retrieve_pbr(
                query, 
                history, 
                fake_queries=fake_queries_list[i],
                reasoning=reasoning_list[i],
                top_k=10, 
                use_llm=use_llm
            )
            
            results.append({
                "query_id": query_id,
                "query": query,
                "results": retrieved
            })
        except Exception as e:
            print(f"\n   ⚠️ {query_id} 失败: {e}")
            continue
    
    # 7. 保存
    output_file = output_dir / ("pbr_full_results.jsonl" if use_llm else "pbr_simple_results.jsonl")
    print(f"\n💾 保存: {output_file}")
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"   ✅ {len(results)} 个结果")
    
    print("\n" + "="*80)
    print("✅ PBR完成！")
    print("="*80)


def run_pbr_correct(dataset: str, use_llm: bool = True):
    """运行PBR baseline的入口"""
    asyncio.run(run_pbr_mini_batch(dataset, use_llm))


def main():
    parser = argparse.ArgumentParser(description='PBR Baseline (Correct Implementation)')
    parser.add_argument('--dataset', type=str, default='MedCorpus',
                       choices=['MedCorpus', 'LitSearch'])
    parser.add_argument('--use_llm', action='store_true',
                       help='使用LLM进行P-PRF查询扩展（完整PBR）')
    args = parser.parse_args()
    
    run_pbr_correct(args.dataset, args.use_llm)


if __name__ == "__main__":
    main()

