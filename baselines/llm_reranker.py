#!/usr/bin/env python3
"""
LLM-based Reranker for MedCorpus
使用大语言模型对检索结果进行重排序

支持OpenAI兼容API（包括MiniMax等）和Anthropic API（Claude系列）
"""

import json
import time
import os
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from openai import OpenAI
from tqdm import tqdm
import re

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


def load_queries(queries_file):
    """加载查询数据"""
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            queries[data['query_id']] = data
    return queries


def load_corpus(corpus_file, needed_ids=None):
    """
    加载文档库
    
    Args:
        corpus_file: corpus文件路径
        needed_ids: 需要加载的文档ID集合（如果为None则加载全部）
    """
    corpus = {}
    
    # 转换needed_ids为统一的字符串类型（用于查找）
    if needed_ids is not None:
        needed_ids_str = set(str(id) for id in needed_ids)
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            text_id = data['text_id']
            
            # 如果指定了needed_ids，只加载需要的文档
            if needed_ids is None or str(text_id) in needed_ids_str:
                corpus[text_id] = data
                
                # 如果已经加载了所有需要的文档，提前退出
                if needed_ids is not None and len(corpus) >= len(needed_ids):
                    break
    
    return corpus


def load_retrieval_results(retrieval_file):
    """加载检索结果"""
    results = {}
    with open(retrieval_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            query_id = data['query_id']
            
            # 处理格式差异：将 "topic_001_1" 转换为 "topic_001_turn_1"
            if query_id.count('_') == 2 and not 'turn' in query_id:
                parts = query_id.rsplit('_', 1)
                query_id = f"{parts[0]}_turn_{parts[1]}"
            
            results[query_id] = data['results']
    return results


def create_rerank_prompt(query_info: Dict, candidates: List[Dict], corpus: Dict, include_history: bool = True, abstract_max_chars: int = 500) -> str:
    """
    创建重排prompt
    
    Args:
        query_info: 查询信息（包含query, history等）
        candidates: 候选文档列表
        corpus: 文档库
        include_history: 是否包含对话历史
    """
    prompt = """You are an expert medical literature search assistant. Your task is to rerank a list of candidate research papers based on their relevance to the user's query.

"""
    
    # 添加对话历史
    if include_history and query_info.get('history') and len(query_info['history']) > 0:
        prompt += "**Conversation History:**\n"
        for i, hist in enumerate(query_info['history'], 1):
            prompt += f"{i}. {hist}\n"
        prompt += "\n"
    
    # 当前查询
    prompt += f"**Current Query:**\n{query_info['query']}\n\n"
    
    # 候选文档
    prompt += "**Candidate Papers (in initial retrieval order):**\n\n"
    for i, doc in enumerate(candidates, 1):
        doc_info = corpus.get(doc['text_id'], {})
        title = doc_info.get('title', 'N/A')
        text = doc_info.get('text', 'N/A')
        
        # 截断文本（避免超长）
        if len(text) > abstract_max_chars:
            text = text[:abstract_max_chars] + "..."
        
        prompt += f"[{i}] ID: {doc['text_id']}\n"
        prompt += f"Title: {title}\n"
        prompt += f"Abstract: {text}\n\n"
    
    # 输出格式说明
    prompt += """**Task:**
Analyze each paper's relevance to the current query (considering the conversation history if provided). Rerank the papers from most relevant to least relevant.

**Output Format:**
Return ONLY a comma-separated list of document IDs in descending order of relevance (most relevant first).

Example output format:
permed-12345, permed-67890, permed-11111, ...

Important:
- Output ONLY the ranked list of IDs, no explanations
- Use the exact document IDs provided above
- Separate IDs with commas and spaces
- Include ALL documents in your ranking

Your ranked list:"""
    
    return prompt


def parse_llm_ranking(response: str, candidate_ids: List, dataset_type: str = 'medcorpus') -> List:
    """
    解析LLM返回的排序结果
    
    Args:
        response: LLM的响应文本
        candidate_ids: 原始候选文档ID列表
        dataset_type: 数据集类型 ('medcorpus' 或 'litsearch')
    
    Returns:
        重排后的文档ID列表
    """
    # 根据数据集类型提取文档ID
    if dataset_type == 'medcorpus':
        # 格式：permed-xxxxx
        pattern = r'permed-\d+'
        ranked_ids = re.findall(pattern, response)
    else:  # litsearch
        # 格式：纯数字
        pattern = r'\b\d{6,}\b'  # 至少6位数字
        ranked_ids_str = re.findall(pattern, response)
        ranked_ids = [int(x) for x in ranked_ids_str]
    
    # 去重但保持顺序
    seen = set()
    unique_ranked = []
    for doc_id in ranked_ids:
        if doc_id not in seen and doc_id in candidate_ids:
            seen.add(doc_id)
            unique_ranked.append(doc_id)
    
    # 将未排序的文档添加到末尾（保持原始顺序）
    for doc_id in candidate_ids:
        if doc_id not in seen:
            unique_ranked.append(doc_id)
    
    return unique_ranked


def llm_rerank(
    client,  # OpenAI 或 Anthropic 客户端
    query_info: Dict,
    candidates: List[Dict],
    corpus: Dict,
    model: str,
    dataset_type: str = 'medcorpus',
    include_history: bool = True,
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_delay: int = 2,
    api_type: str = 'openai'
) -> List:
    """
    使用LLM对候选文档进行重排

    Args:
        client: OpenAI 或 Anthropic 客户端
        api_type: 'openai' 或 'anthropic'

    Returns:
        重排后的文档ID列表
    """
    candidate_ids = [doc['text_id'] for doc in candidates]
    abstract_lengths = [500, 350, 200]

    for attempt in range(max_retries):
        current_max_chars = abstract_lengths[min(attempt, len(abstract_lengths) - 1)]
        current_include_history = include_history if attempt < max_retries - 1 else False

        prompt = create_rerank_prompt(
            query_info,
            candidates,
            corpus,
            include_history=current_include_history,
            abstract_max_chars=current_max_chars
        )

        try:
            if api_type == 'anthropic':
                # 使用 Anthropic API
                response = client.messages.create(
                    model=model,
                    max_tokens=500,
                    temperature=temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                # 检查响应是否为空
                if response.content is None or len(response.content) == 0:
                    raise Exception(
                        f"API返回空响应（可能触发内容过滤）。"
                        f"输入tokens: {response.usage.input_tokens}, 输出tokens: {response.usage.output_tokens},"
                        f" 尝试: {attempt + 1}/{max_retries}, 摘要长度: {current_max_chars}, 包含历史: {current_include_history}"
                )
                llm_output = response.content[0].text.strip()
            else:
                # 使用 OpenAI 兼容 API
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=500
                )
                llm_output = response.choices[0].message.content.strip()

            # 解析响应
            ranked_ids = parse_llm_ranking(llm_output, candidate_ids, dataset_type)

            return ranked_ids

        except Exception as e:
            error_msg = str(e)
            print(f"\n错误 (尝试 {attempt + 1}/{max_retries}): {error_msg}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # 指数退避
            else:
                # 最后一次尝试也失败了，直接抛出异常，不使用原始排序
                print(f"\n致命错误: 所有重试都失败了，停止处理")
                raise Exception(f"LLM重排失败 (尝试{max_retries}次): {error_msg}")

    # 如果所有尝试都失败，抛出异常
    raise Exception("LLM重排失败：超过最大重试次数")


def main():
    parser = argparse.ArgumentParser(description='LLM-based Reranker')
    
    # 数据路径
    parser.add_argument('--queries', type=str, 
                       default='data/MedCorpus/queries.jsonl',
                       help='查询文件路径')
    parser.add_argument('--corpus', type=str,
                       default='data/MedCorpus/corpus.jsonl',
                       help='文档库文件路径')
    parser.add_argument('--retrieval', type=str,
                       default='/mnt/data/zsy-data/PerMed/results/MedCorpus/retrieved.jsonl',
                       help='检索结果文件路径')
    parser.add_argument('--output', type=str,
                       default='results/MedCorpus/llm_rerank_results.jsonl',
                       help='输出文件路径')
    
    # LLM配置
    parser.add_argument('--api_type', type=str,
                       default='openai',
                       choices=['openai', 'anthropic'],
                       help='API类型: openai (OpenAI兼容) 或 anthropic (Claude)')
    parser.add_argument('--api_url', type=str,
                       default='https://api.minimaxi.com/v1',
                       help='API基础URL (仅用于OpenAI兼容API)')
    parser.add_argument('--api_key', type=str,
                       required=True,
                       help='API密钥')
    parser.add_argument('--model', type=str,
                       default='MiniMax-M2',
                       help='模型名称')
    
    # 重排配置
    parser.add_argument('--dataset_type', type=str, default='medcorpus',
                       choices=['medcorpus', 'litsearch'],
                       help='数据集类型')
    parser.add_argument('--top_k', type=int, default=10,
                       help='对top-k个结果进行重排')
    parser.add_argument('--include_history', action='store_true', default=True,
                       help='是否包含对话历史')
    parser.add_argument('--no_history', action='store_false', dest='include_history',
                       help='不包含对话历史')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='生成温度（0.0为确定性）')
    
    # 限速配置
    parser.add_argument('--rpm_limit', type=int, default=180,
                       help='每分钟请求数限制（RPM）')
    parser.add_argument('--start_from', type=int, default=0,
                       help='从第N个查询开始（用于断点续传）')
    
    args = parser.parse_args()

    # 初始化客户端
    if args.api_type == 'anthropic':
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("需要安装 anthropic 库: pip install anthropic")
        client = Anthropic(api_key=args.api_key, base_url=args.api_url)
    else:
        client = OpenAI(
            api_key=args.api_key,
            base_url=args.api_url
        )
    
    print(f"加载数据...")
    queries = load_queries(args.queries)
    
    # 先加载检索结果，获取所有需要的文档ID
    print(f"  加载检索结果...")
    retrieval_results = load_retrieval_results(args.retrieval)
    
    # 收集所有检索结果中的文档ID
    needed_ids = set()
    for query_results in retrieval_results.values():
        for doc in query_results[:args.top_k]:  # 只需要top-k的文档
            needed_ids.add(doc['text_id'])
    
    print(f"  需要加载 {len(needed_ids)} 个文档（共 {len(queries)} 个查询）")
    print(f"  加载corpus（仅需要的文档）...")
    corpus = load_corpus(args.corpus, needed_ids=needed_ids)
    
    print(f"总查询数: {len(queries)}")
    print(f"文档库大小: {len(corpus)}")
    print(f"重排配置:")
    print(f"  - 模型: {args.model}")
    print(f"  - Top-K: {args.top_k}")
    print(f"  - 包含历史: {args.include_history}")
    print(f"  - 限速: {args.rpm_limit} RPM")
    print()
    
    # 创建输出目录
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # 计算每个请求的延迟（以满足RPM限制）
    request_delay = 60.0 / args.rpm_limit if args.rpm_limit > 0 else 0
    
    # 处理每个查询
    results = []
    processed_query_ids = set()

    # 自动检测断点续传
    if Path(args.output).exists():
        print(f"检测到已有结果文件，加载已完成的查询...")
        with open(args.output, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line)
                results.append(result)
                processed_query_ids.add(result['query_id'])
        print(f"  已加载 {len(results)} 个已完成的结果")
        
        # 如果用户没有指定 start_from，自动设置
        if args.start_from == 0 and len(processed_query_ids) > 0:
            print(f"  自动断点续传模式：跳过已完成的 {len(processed_query_ids)} 个查询")

    query_ids = sorted(queries.keys())

    start_time = time.time()

    # 过滤出需要处理的查询
    if args.start_from > 0:
        # 如果明确指定了 start_from，则从该位置开始
        queries_to_process = query_ids[args.start_from:]
        start_idx = args.start_from
    else:
        # 否则只处理未完成的查询
        queries_to_process = [qid for qid in query_ids if qid not in processed_query_ids]
        start_idx = len(processed_query_ids)

    print(f"开始处理 {len(queries_to_process)} 个查询 (已完成: {len(processed_query_ids)}/{len(query_ids)})")
    print()

    for idx, query_id in enumerate(tqdm(queries_to_process,
                                         desc="重排进度",
                                         initial=start_idx,
                                         total=len(query_ids))):
        query_info = queries[query_id]
        candidates = retrieval_results.get(query_id, [])[:args.top_k]
        
        if not candidates:
            print(f"\n警告: {query_id} 没有检索结果")
            continue
        
        # LLM重排（如果失败会抛出异常，不会fallback到原始排序）
        try:
            ranked_ids = llm_rerank(
                client=client,
                query_info=query_info,
                candidates=candidates,
                corpus=corpus,
                model=args.model,
                dataset_type=args.dataset_type,
                include_history=args.include_history,
                temperature=args.temperature,
                api_type=args.api_type
            )
        except Exception as e:
            print(f"\n\n{'='*60}")
            print(f"错误：LLM重排失败在查询 {query_id}")
            print(f"当前进度: {len(results)}/{len(query_ids)} 个查询已完成")
            print(f"错误信息: {str(e)}")
            print(f"{'='*60}\n")
            print(f"保存 {len(results)} 个成功的结果到 {args.output}...")
            # 保存已有结果
            with open(args.output, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"结果已保存！")
            print(f"\n重新运行相同命令即可自动从断点继续处理")
            raise  # 重新抛出异常，终止程序
        
        # 构建输出（与其他方法格式一致）
        ranked_results = []
        for rank, text_id in enumerate(ranked_ids, 1):
            # 找到原始分数
            original_score = next(
                (c['score'] for c in candidates if c['text_id'] == text_id),
                0.0
            )
            ranked_results.append({
                'text_id': text_id,
                'score': original_score,  # 保留原始检索分数
                'rank': rank
            })
        
        results.append({
            'query_id': query_id,
            'ranked_results': ranked_results
        })
        
        # 定期保存（每100个查询）
        if (idx + 1) % 100 == 0:
            with open(args.output, 'w', encoding='utf-8') as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"\n已保存 {len(results)} 个结果")
        
        # 限速
        if request_delay > 0:
            time.sleep(request_delay)
    
    # 最终保存
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    elapsed_time = time.time() - start_time
    print(f"\n完成!")
    print(f"总耗时: {elapsed_time/60:.2f} 分钟")
    print(f"处理查询数: {len(results)}")
    print(f"输出文件: {args.output}")


if __name__ == '__main__':
    main()

