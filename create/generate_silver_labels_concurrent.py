"""
全量银标准标注脚本 - Qwen3检索 + DeepSeek评分
支持并发API调用、增量保存、断点续传
"""

import json
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==================== 配置 ====================

# 文件路径
CORPUS_FILE = "/workspace/PerMed/data/MedCorpus/corpus.jsonl"
EMBEDDINGS_FILE = "/workspace/PerMed/data/MedCorpus/corpus_embeddings_qwen3.npy"
CONVERSATIONS_FILE = "/workspace/PerMed/create/final_800_topics.jsonl"
MODEL_PATH = "/workspace/PerMed/model/Qwen3-Embedding-0.6B"

# 输出文件
OUTPUT_FILE = "/workspace/PerMed/results/silver_labels_qwen3.jsonl"
LOG_FILE = "/workspace/PerMed/results/silver_labeling.log"

# DeepSeek API
API_KEY = ""
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL = "deepseek-ai/DeepSeek-V3.2-Exp"

# 并发设置
MAX_WORKERS = 10  # 并发线程数（RPM=1000，保守使用10个并发）
BATCH_SIZE = 5    # 每批处理5个对话后保存一次

# ==================== 工具函数 ====================

def log_message(message, log_file=LOG_FILE):
    """写入日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_msg + '\n')

def call_deepseek(client, prompt, max_retries=3):
    """调用DeepSeek API，支持重试"""
    for retry in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            resp = response.choices[0].message.content.strip()

            # 提取分数
            if resp.startswith("2"):
                return 2, resp
            elif resp.startswith("1"):
                return 1, resp
            elif resp.startswith("0"):
                return 0, resp
            else:
                return -1, resp

        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(1)
            else:
                return -1, f"API Error: {str(e)}"

    return -1, "Max retries exceeded"

def annotate_single_document(client, doc_id, doc_title, doc_text, retrieval_score,
                             question, turn_idx, history):
    """标注单个文档的相关性"""

    # 构建prompt
    if history:
        hist_text = "\n".join([f"Turn {i+1}: {t}" for i, t in enumerate(history)])
        prompt = f"""You are a scientific literature relevance annotator.

Conversation History:
{hist_text}

Current Question (Turn {turn_idx}): {question}

Document:
Title: {doc_title}
Text: {doc_text}...

Task: Rate relevance (0/1/2):
- 0: Not Relevant (completely different research topic or field)
- 1: Partially Relevant (discusses related concepts, methods, materials, or belongs to the same research area)
- 2: Highly Relevant (directly addresses the specific question being asked)

Output format: Score + brief explanation.
Example: 2 - The document directly discusses the treatment methods mentioned.

Rate:"""
    else:
        prompt = f"""You are a scientific literature relevance annotator.

Question: {question}

Document:
Title: {doc_title}
Text: {doc_text}...

Task: Rate relevance (0/1/2):
- 0: Not Relevant (completely different research topic or field)
- 1: Partially Relevant (discusses related concepts, methods, materials, or belongs to the same research area)
- 2: Highly Relevant (directly addresses the specific question)

Output format: Score + brief explanation.
Example: 2 - The document directly discusses the requested topic.

Rate:"""

    relevance, explanation = call_deepseek(client, prompt)

    return {
        'rank': None,  # 将在外部设置
        'doc_id': doc_id,
        'retrieval_score': float(retrieval_score),
        'relevance': relevance,
        'explanation': explanation
    }

# ==================== 主程序 ====================

def main():
    start_time = time.time()

    log_message("=" * 80)
    log_message("银标准标注 - Qwen3检索 + DeepSeek评分")
    log_message("=" * 80)

    # 1. 加载Qwen3模型
    log_message("[1/6] 加载Qwen3模型到GPU...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_PATH, device=device)
    log_message(f"      ✓ 模型加载完成 (设备: {device}, 维度: {model.get_sentence_embedding_dimension()})")

    # 2. 加载corpus embeddings
    log_message("[2/6] 加载corpus embeddings...")
    corpus_embeddings = np.load(EMBEDDINGS_FILE)
    log_message(f"      ✓ Embeddings形状: {corpus_embeddings.shape}")

    # 3. 加载corpus文档
    log_message("[3/6] 加载corpus文档...")
    corpus_docs = []
    with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            corpus_docs.append(json.loads(line.strip()))
    log_message(f"      ✓ 文档数: {len(corpus_docs)}")

    # 4. 加载对话数据
    log_message("[4/6] 加载对话数据...")
    conversations = []
    with open(CONVERSATIONS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            conversations.append(json.loads(line.strip()))
    log_message(f"      ✓ 对话数: {len(conversations)}")

    # 统计总轮数
    total_turns = sum(len(conv['turns']) for conv in conversations)
    total_labels = total_turns * 10
    log_message(f"      ✓ 总轮数: {total_turns}, 预计标签数: {total_labels}")

    # 5. 检查已处理的对话
    log_message("[5/6] 检查断点续传...")
    processed_topics = set()
    results = []

    if Path(OUTPUT_FILE).exists():
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                processed_topics.add(data['topic_id'])
                results.append(data)
        log_message(f"      ✓ 已处理 {len(processed_topics)} 个对话，从第 {len(processed_topics)+1} 个继续")
    else:
        log_message("      ✓ 未发现已有结果，从头开始")

    # 6. 初始化DeepSeek客户端
    log_message("[6/6] 初始化DeepSeek API...")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    log_message("      ✓ API客户端初始化完成")

    log_message("")
    log_message("=" * 80)
    log_message("开始银标准标注")
    log_message("=" * 80)

    # 处理对话
    api_calls_count = 0
    batch_results = []

    for conv_idx, conversation in enumerate(conversations, 1):
        topic_id = conversation['topic_id']

        # 跳过已处理
        if topic_id in processed_topics:
            continue

        log_message(f"\n[{conv_idx}/{len(conversations)}] 处理对话: {topic_id} ({len(conversation['turns'])}轮)")

        # 存储这个对话的结果
        topic_result = {
            'topic_id': topic_id,
            'target_turns': conversation.get('target_turns', len(conversation['turns'])),
            'turns': []
        }

        # 对话历史
        history = []

        # 逐轮处理
        for turn_idx, turn in enumerate(conversation['turns'], 1):
            question = turn['text']
            turn_id = turn.get('turn_id', turn_idx)

            # 编码问题
            query_embedding = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)[0]

            # 检索top-10
            scores = np.dot(corpus_embeddings, query_embedding)
            top_indices = np.argsort(scores)[::-1][:10]

            # 并发标注top-10文档
            turn_labels = []

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = []

                for rank, idx in enumerate(top_indices, 1):
                    doc = corpus_docs[idx]
                    doc_id = doc['text_id']
                    doc_title = doc['title']
                    doc_text = doc['text'][:300]
                    retrieval_score = float(scores[idx])

                    # 提交并发任务
                    future = executor.submit(
                        annotate_single_document,
                        client, doc_id, doc_title, doc_text, retrieval_score,
                        question, turn_idx, history
                    )
                    futures.append((rank, future))

                # 收集结果
                for rank, future in futures:
                    try:
                        label = future.result(timeout=30)
                        label['rank'] = rank
                        turn_labels.append(label)
                        api_calls_count += 1
                    except Exception as e:
                        log_message(f"      ❌ Turn {turn_idx} Rank {rank} 标注失败: {e}")

            # 按rank排序
            turn_labels.sort(key=lambda x: x['rank'])

            # 保存这一轮
            topic_result['turns'].append({
                'turn_id': turn_id,
                'turn_idx': turn_idx,
                'question': question,
                'labels': turn_labels
            })

            # 统计相关性
            rel_counts = {0: 0, 1: 0, 2: 0, -1: 0}
            for label in turn_labels:
                rel_counts[label['relevance']] += 1

            log_message(f"      Turn {turn_idx}: 0={rel_counts[0]}, 1={rel_counts[1]}, 2={rel_counts[2]}, Error={rel_counts[-1]}")

            # 添加到历史
            history.append(question)

        # 保存这个对话
        batch_results.append(topic_result)

        # 每BATCH_SIZE个对话保存一次
        if len(batch_results) >= BATCH_SIZE:
            with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
                for result in batch_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

            log_message(f"  ✓ 已保存 {len(batch_results)} 个对话到文件")
            batch_results = []

        # 显示进度
        elapsed = time.time() - start_time
        processed_count = conv_idx - len([c for c in conversations[:conv_idx] if c['topic_id'] in processed_topics])
        if processed_count > 0:
            avg_time = elapsed / processed_count
            remaining = (len(conversations) - conv_idx) * avg_time
            log_message(f"  进度: {conv_idx}/{len(conversations)}, 已用时: {elapsed/3600:.1f}h, 预计剩余: {remaining/3600:.1f}h, API调用: {api_calls_count}")

    # 保存剩余结果
    if batch_results:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            for result in batch_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        log_message(f"  ✓ 已保存最后 {len(batch_results)} 个对话")

    # 统计
    total_time = time.time() - start_time

    log_message("")
    log_message("=" * 80)
    log_message("✅ 银标准标注完成!")
    log_message("=" * 80)
    log_message(f"总用时: {total_time/3600:.2f} 小时")
    log_message(f"API调用次数: {api_calls_count}")
    log_message(f"输出文件: {OUTPUT_FILE}")
    log_message(f"日志文件: {LOG_FILE}")

if __name__ == "__main__":
    main()
