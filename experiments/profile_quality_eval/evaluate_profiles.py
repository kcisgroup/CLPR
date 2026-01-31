#!/usr/bin/env python3
"""
主评估脚本：使用多个 LLM 评估个性化特征质量

支持的模型：
- GPT-4o (OpenAI)
- Claude Sonnet 4.5 (Anthropic)
- Gemini 2.5 Pro (Google)
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
import hashlib

# LLM clients
import openai
from anthropic import Anthropic

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

import config
from prompts import create_evaluation_prompt


# ============================================
# LLM 评估函数
# ============================================

def call_gpt4o(prompt: str, api_key: str, model_name: str, temperature: float = 0.0, max_tokens: int = 500, api_url: str = None) -> str:
    """调用 GPT-4o 或 GPT-5"""
    # fox 端点需要特殊处理 SSE 流
    if api_url and "code.newcli.com" in api_url:
        import urllib.request
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are an expert evaluator for medical literature search systems. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        req = urllib.request.Request(
            f"{api_url}/chat/completions",
            data=json.dumps(payload).encode('utf-8'),
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            method='POST'
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode('utf-8', 'ignore')
        
        # 解析 SSE 流，提取所有 content delta
        text_parts = []
        for line in raw.splitlines():
            if line.startswith('data: '):
                try:
                    chunk = json.loads(line[6:])
                    if 'choices' in chunk and chunk['choices']:
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta and delta['content']:
                            text_parts.append(delta['content'])
                except:
                    pass
        text = ''.join(text_parts).strip()
    else:
        # 标准 OpenAI 调用
        client = openai.OpenAI(api_key=api_key, base_url=api_url) if api_url else openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are an expert evaluator for medical literature search systems. Respond ONLY with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        text = response.choices[0].message.content.strip()
    
    # 尝试提取 JSON（可能包含 markdown 代码块）
    import re
    # 先尝试提取 ```json ... ``` 或 ``` ... ```
    code_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if code_block_match:
        return code_block_match.group(1)
    
    # 否则提取第一个完整的 JSON 对象（非贪婪匹配，避免多余内容）
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
    if json_match:
        return json_match.group(0)
    
    return text


def call_claude(prompt: str, api_key: str, api_url: str, model_name: str, temperature: float = 0.0, max_tokens: int = 500) -> str:
    """调用 Claude"""
    client = Anthropic(api_key=api_key, base_url=api_url)
    
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    if not response.content or len(response.content) == 0:
        raise Exception(f"Claude returned empty response")
    
    text = response.content[0].text.strip()
    
    # 尝试提取 JSON（Claude 可能在 JSON 前后添加文本）
    import re
    json_match = re.search(r'\{[\s\S]*\}', text)
    if json_match:
        return json_match.group(0)
    
    return text


def call_gemini(prompt: str, api_key: str, model_name: str, temperature: float = 0.0, max_tokens: int = 500) -> str:
    """调用 Gemini"""
    if not GEMINI_AVAILABLE:
        raise ImportError("google.generativeai module not available")
    
    genai.configure(api_key=api_key)
    
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "response_mime_type": "application/json"
        }
    )
    
    response = model.generate_content(prompt)
    return response.text


def evaluate_with_llm(
    query_data: Dict,
    model_key: str,
    model_config: Dict,
    max_retries: int = 3,
    retry_delay: int = 2
) -> Dict:
    """
    使用指定的 LLM 评估个性化特征
    
    Args:
        query_data: 查询数据
        model_key: 模型名称（如 "gpt-4o"）
        model_config: 模型配置
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
        
    Returns:
        评估结果
    """
    # 创建 prompt
    prompt = create_evaluation_prompt(
        query=query_data['query'],
        history=query_data.get('history', []),
        personalized_features=query_data['personalized_features']
    )
    
    # 根据模型类型调用对应 API
    api_type = model_config['api_type']
    
    for attempt in range(max_retries):
        try:
            if api_type == 'openai':
                response_text = call_gpt4o(
                    prompt=prompt,
                    api_key=model_config['api_key'],
                    model_name=model_config['model_name'],
                    temperature=model_config['temperature'],
                    max_tokens=model_config['max_tokens'],
                    api_url=model_config.get('api_url')
                )
            elif api_type == 'anthropic':
                response_text = call_claude(
                    prompt=prompt,
                    api_key=model_config['api_key'],
                    api_url=model_config['api_url'],
                    model_name=model_config['model_name'],
                    temperature=model_config['temperature'],
                    max_tokens=model_config['max_tokens']
                )
            elif api_type == 'gemini':
                response_text = call_gemini(
                    prompt=prompt,
                    api_key=model_config['api_key'],
                    model_name=model_config['model_name'],
                    temperature=model_config['temperature'],
                    max_tokens=model_config['max_tokens']
                )
            else:
                raise ValueError(f"Unknown API type: {api_type}")
            
            # 解析响应
            result = json.loads(response_text)
            
            # 验证结果 - 核心维度必须存在
            required_keys = ['relevance', 'accuracy', 'informativeness', 'coherence']
            if not all(k in result for k in required_keys):
                raise ValueError(f"Missing required dimension keys in response: {result.keys()}")
            
            # 如果缺少 average_score，自动计算
            if 'average_score' not in result:
                result['average_score'] = (
                    result['relevance'] + 
                    result['accuracy'] + 
                    result['informativeness'] + 
                    result['coherence']
                ) / 4.0
            
            # 如果缺少 explanation，添加默认值
            if 'explanation' not in result:
                result['explanation'] = "No explanation provided"
            
            return {
                "success": True,
                "scores": result,
                "raw_response": response_text
            }
        
        except Exception as e:
            error_msg = str(e)
            print(f"\n⚠️ {model_key} 错误 (尝试 {attempt + 1}/{max_retries}): {error_msg}")
            
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
            else:
                return {
                    "success": False,
                    "error": error_msg,
                    "scores": None
                }
    
    return {
        "success": False,
        "error": "Max retries exceeded",
        "scores": None
    }


# ============================================
# 缓存管理
# ============================================

def get_cache_key(query_id: str, model_key: str) -> str:
    """生成缓存 key"""
    combined = f"{query_id}_{model_key}"
    return hashlib.md5(combined.encode()).hexdigest()


def load_cache(cache_file: Path) -> Dict:
    """加载缓存"""
    if not cache_file.exists():
        return {}
    
    cache = {}
    with open(cache_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            cache[data['cache_key']] = data
    
    return cache


def save_cache_entry(cache_file: Path, cache_key: str, data: Dict):
    """保存单个缓存条目"""
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "cache_key": cache_key,
        **data
    }
    
    with open(cache_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


# ============================================
# 主评估流程
# ============================================

def evaluate_all_queries(
    queries: List[Dict],
    models: List[str],
    output_file: Path,
    dry_run: bool = False,
    use_cache: bool = True
):
    """
    评估所有查询
    
    Args:
        queries: 查询列表
        models: 要使用的模型列表
        output_file: 输出文件路径
        dry_run: 是否为测试模式
        use_cache: 是否使用缓存
    """
    results = []
    cache_file = config.CACHE_DIR / "evaluation_cache.jsonl"
    
    # 加载缓存
    cache = load_cache(cache_file) if use_cache else {}
    cache_hits = 0
    cache_misses = 0
    
    print(f"\n开始评估 {len(queries)} 个查询...")
    print(f"使用模型: {', '.join(models)}")
    if use_cache:
        print(f"缓存文件: {cache_file}")
        print(f"已缓存条目: {len(cache)}")
    print()
    
    for query_data in tqdm(queries, desc="评估进度"):
        query_id = query_data['query_id']
        query_result = {
            "query_id": query_id,
            "turn_id": query_data['turn_id'],
            "query": query_data['query'],
            "personalized_features": query_data['personalized_features'],
            "evaluations": {}
        }
        
        for model_key in models:
            cache_key = get_cache_key(query_id, model_key)
            
            # 检查缓存
            if use_cache and cache_key in cache:
                cached_data = cache[cache_key]
                query_result['evaluations'][model_key] = cached_data.get('evaluation', {})
                cache_hits += 1
                continue
            
            cache_misses += 1
            
            # Dry-run 模式
            if dry_run:
                query_result['evaluations'][model_key] = {
                    "success": True,
                    "scores": {
                        "relevance": 4,
                        "accuracy": 4,
                        "informativeness": 4,
                        "coherence": 4,
                        "average_score": 4.0,
                        "explanation": "[DRY RUN] Simulated evaluation"
                    }
                }
                continue
            
            # 真实评估
            model_config = config.MODELS[model_key]
            evaluation = evaluate_with_llm(query_data, model_key, model_config)
            
            query_result['evaluations'][model_key] = evaluation
            
            # 保存到缓存
            if use_cache:
                save_cache_entry(cache_file, cache_key, {
                    "query_id": query_id,
                    "model": model_key,
                    "evaluation": evaluation
                })
            
            # 限速（避免 API 限流）
            if not dry_run:
                time.sleep(1)  # 1秒延迟
        
        results.append(query_result)
        
        # 定期保存
        if len(results) % config.SAVE_INTERVAL == 0:
            save_results(results, output_file)
    
    # 最终保存
    save_results(results, output_file)
    
    # 打印统计
    print()
    print("=" * 60)
    print("评估完成！")
    print("=" * 60)
    if use_cache:
        print(f"缓存命中: {cache_hits}")
        print(f"新评估: {cache_misses}")
    print(f"总查询数: {len(results)}")
    print(f"总评估数: {len(results) * len(models)}")
    print(f"结果保存到: {output_file}")
    print("=" * 60)


def save_results(results: List[Dict], output_file: Path):
    """保存结果"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


# ============================================
# 命令行接口
# ============================================

def main():
    parser = argparse.ArgumentParser(description='评估个性化特征质量')
    
    parser.add_argument('--input', type=str,
                       default=str(config.SAMPLED_QUERIES_FILE),
                       help='输入文件（采样的查询）')
    parser.add_argument('--output', type=str,
                       default=str(config.EVALUATION_RESULTS_FILE),
                       help='输出文件（评估结果）')
    parser.add_argument('--models', nargs='+',
                       default=['gpt-5'],
                       choices=['gpt-4o', 'gpt-5', 'claude-haiku-4.5', 'gemini-2.5-pro', 'gemini-2.5-pro-jiubanai', 'gemini-2.5-flash-jiubanai', 'gemini-2.5-pro-hybgzs', 'gpt-4o-mini-hybgzs', 'grok-4.1-thinking', 'claude-sonnet-4.5'],
                       help='要使用的模型')
    parser.add_argument('--dry-run', action='store_true',
                       help='测试模式（不调用真实 API）')
    parser.add_argument('--no-cache', action='store_true',
                       help='不使用缓存')
    
    args = parser.parse_args()
    
    # 加载查询
    print(f"加载查询从: {args.input}")
    queries = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            queries.append(json.loads(line))
    print(f"加载 {len(queries)} 个查询")
    
    # 检查样本文件
    if not Path(args.input).exists():
        print(f"❌ 样本文件不存在: {args.input}")
        return
    
    # 开始评估
    evaluate_all_queries(
        queries=queries,
        models=args.models,
        output_file=Path(args.output),
        dry_run=args.dry_run,
        use_cache=not args.no_cache
    )


if __name__ == "__main__":
    main()

