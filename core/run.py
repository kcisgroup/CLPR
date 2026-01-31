# run.py - 已集成fusion_rerank模式
import os
import argparse
import logging
import time
import json
from datetime import datetime
from tqdm import tqdm
import gc
import torch
from collections import defaultdict

from utils import get_config, logger, Query, progress_logger

def parse_args():
    parser = argparse.ArgumentParser(description="运行 PersLitRank 系统")
    parser.add_argument("--mode", type=str,
                        choices=["all", "extract_cognitive_features", "generate_narratives", "retrieve", "rerank", "fusion_rerank", "simple_profile_rerank"],
                        required=True, help="处理模式")

    # 通用参数
    parser.add_argument("--dataset_name", type=str, default="MedCorpus", choices=["MedCorpus", "LitSearch", "CORAL"])
    parser.add_argument("--data_dir", type=str, default="/workspace/PerMed/data", help="基础数据目录")
    parser.add_argument("--results_dir", type=str, default="./results", help="基础结果目录")
    parser.add_argument("--gpu_id", type=int, default=0, help="当前进程的GPU ID")
    parser.add_argument("--batch_size", type=int, help="适用的批处理大小 (用于检索或重排)")
    parser.add_argument("--personalized_text_target_length", type=int, default=250,
                        help="个性化画像的目标长度 (例如 150, 250, 350)")
    
    # LLM API选择
    parser.add_argument("--llm_api_type", type=str, default="ollama",
                        choices=["ollama", "siliconflow", "openai"],  # 已经有 openai 了！
                        help="选择使用的LLM API类型")
    
    # Llama3 API 参数
    parser.add_argument("--llm_base_url", type=str, default="http://172.18.147.77:11434",
                        help="Ollama API 基础URL")
    parser.add_argument("--llm_model", type=str, default="llama3:8b",
                        help="使用的Llama3模型 (如 llama3:8b, llama3.3:72b-5k-context)")
    
    # SiliconFlow API 参数
    parser.add_argument("--siliconflow_api_key", type=str,
                        default="",
                        help="SiliconFlow API密钥")
    parser.add_argument("--siliconflow_model", type=str,
                        default="Qwen/Qwen3-Next-80B-A3B-Thinking",
                        help="使用的SiliconFlow模型")
    
    # OpenAI API 参数（新增）
    parser.add_argument("--openai_api_key", type=str,
                        default="",
                        help="OpenAI API密钥")
    parser.add_argument("--openai_base_url", type=str,
                        default="https://api.openai-proxy.org/v1",
                        help="OpenAI API基础URL")
    parser.add_argument("--openai_model", type=str,
                        default="gpt-4o-mini",
                        help="使用的OpenAI模型 (如 gpt-4o-mini, o3-mini)")
    
    # 通用LLM参数
    parser.add_argument("--temperature", type=float, help="LLM temperature")
    parser.add_argument("--top_p", type=float, help="LLM top_p")
    parser.add_argument("--top_k", type=int, help="LLM top_k")
    parser.add_argument("--local_model_max_tokens", type=int, help="LLM为画像生成的最大token数")

    # Reranker 参数
    parser.add_argument("--reranker_type", type=str, choices=["qwen3", "gte", "jina", "jina-v3", "minicpm"], 
                        default="qwen3", help="重排器类型")
    parser.add_argument("--reranker_path", type=str, help="重排器模型的显式路径")
    parser.add_argument("--reranker_max_length", type=int, help="重排器的最大序列长度")
    parser.add_argument("--rerank_input_type", type=str, default="profile_and_query",
                        choices=["profile_only", "query_only", "profile_and_query"],
                        help="选择'rerank'模式的输入类型")

    # fusion_rerank 参数
    parser.add_argument("--beta", type=float, help="融合重排中'查询分数'的权重 (0.0 to 1.0)。仅用于 'fusion_rerank' 模式。")
    parser.add_argument("--initial_top_k", type=int, help="初始检索召回的文档数")
    parser.add_argument("--final_top_k", type=int, help="重排后最终保留的文档数")
    
    # 其他参数
    parser.add_argument("--feature_extractor", type=str, help="特征提取器类型")
    parser.add_argument("--memory_type", type=str, help="内存类型")
    parser.add_argument("--memory_components", type=str, help="要使用的内存组件，逗号分隔")
    parser.add_argument("--conversational", action="store_true", help="是否为对话模式")
    parser.add_argument("--test_query_limit", type=int, default=None,
                        help="测试模式下限制处理的查询数量 (对MedCorpus是topic/组的数量)")
    parser.add_argument("--use_flash_attention", action="store_true", help="为 MiniCPM 启用 Flash Attention 2")

    return parser.parse_args()

def run_cognitive_feature_extraction_stage(config):
    logger.info(f"--- 阶段1: 认知特征提取 ---")
    logger.info(f"数据集: {config.dataset_name}")
    logger.info(f"详细认知特征输出到: {config.cognitive_features_detailed_path}")
    stage_success = False
    try:
        from cognitive_retrieval import main as cognitive_main
        cognitive_main()
        logger.info(f"--- 阶段1: 认知特征提取完成 ---")
        stage_success = True
    except ImportError:
        logger.error("无法导入 cognitive_retrieval。跳过阶段1。", exc_info=True)
    except Exception as e:
        logger.error(f"阶段1 (认知特征提取) 期间出错: {e}", exc_info=True)
    return stage_success

def run_narrative_generation_stage(config):
    api_type = getattr(config, 'llm_api_type', 'ollama')
    if api_type == 'openai':  # 新增 OpenAI 处理
        model_label = getattr(config, 'openai_model', 'unknown')
    elif api_type == 'siliconflow':
        model_label = getattr(config, 'siliconflow_model', 'unknown')
    else:
        model_label = getattr(config, 'llm_model', 'unknown')
    
    stage_success = False
    try:
        from personalized_generator import PersonalizedGenerator
    except ImportError:
        logger.error("无法导入 PersonalizedGenerator。无法运行画像生成阶段。", exc_info=True)
        return False

    if not os.path.exists(config.cognitive_features_detailed_path):
        logger.error(f"认知特征文件未找到: {config.cognitive_features_detailed_path}。请先运行 'extract_cognitive_features' 模式。")
        return False

    try:
        narrative_generator = PersonalizedGenerator(config=config)
    except Exception as e:
        logger.error(f"初始化 PersonalizedGenerator 时出错: {e}", exc_info=True)
        return False

    generated_narratives_data = []
    queries_processed_count = 0
    
    try:
        cognitive_features_lines = []
        with open(config.cognitive_features_detailed_path, 'r', encoding='utf-8') as f_in:
            cognitive_features_lines = f_in.readlines()
        
        # 按topic限制（针对MedCorpus多轮对话数据集）
        if config.test_query_limit is not None and config.test_query_limit > 0:
            if config.dataset_name == "MedCorpus":
                # 按topic分组
                topic_lines = {}
                for line in cognitive_features_lines:
                    try:
                        data = json.loads(line)
                        qid = data['query_id']
                        if qid.startswith('topic_'):
                            parts = qid.split('_')
                            if len(parts) >= 2 and parts[1].isdigit():
                                topic_num = int(parts[1])
                                if topic_num not in topic_lines:
                                    topic_lines[topic_num] = []
                                topic_lines[topic_num].append(line)
                    except:
                        pass
                
                # 选择前N个topics
                selected_topics = sorted(topic_lines.keys())[:config.test_query_limit]
                cognitive_features_lines = []
                for topic in selected_topics:
                    cognitive_features_lines.extend(topic_lines[topic])
                
                logger.info(f"Limited to first {config.test_query_limit} topics ({len(cognitive_features_lines)} queries)")
            else:
                # 其他数据集按查询数量限制
                cognitive_features_lines = cognitive_features_lines[:config.test_query_limit]
        
        total_queries = len(cognitive_features_lines)
        stage_start_time = time.time()
        start_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        print(f"[{start_stamp}] Starting narrative generation for {config.dataset_name}")
        print(f"  Model: {model_label}")
        print(f"  Queries: {total_queries}")
        print(f"  Target length: {config.personalized_text_target_length}")
        print(f"  Max concurrent: {getattr(config, 'max_concurrent_requests', 20)}")

        # 记录到日志文件
        progress_logger.info(
            f"[START] dataset={config.dataset_name} model={model_label} "
            f"queries={total_queries} target_len={config.personalized_text_target_length} "
            f"concurrent={getattr(config, 'max_concurrent_requests', 20)}"
        )

        # 创建进度条
        pbar = tqdm(total=total_queries, desc="Generating narratives", unit="query")

        def update_progress(completed):
            pbar.update(1)

        # 并发批量生成
        generated_descriptions = narrative_generator.generate_personalized_text_batch(
            cognitive_features_lines,
            progress_callback=update_progress
        )

        pbar.close()

        # 构建输出数据
        for line, narrative_text in zip(cognitive_features_lines, generated_descriptions):
            try:
                cognitive_data = json.loads(line.strip())
                query_id = cognitive_data.get("query_id")
                query_text = cognitive_data.get("query")

                if not query_id or not query_text:
                    continue

                output_entry = {
                    "query_id": query_id,
                    "query": query_text,
                    "personalized_features": narrative_text,
                    "tagged_memory_features": cognitive_data.get("tagged_memory_features", [])
                }
                for key in ["topic_id", "turn_id"]:
                    if key in cognitive_data:
                        output_entry[key] = cognitive_data[key]

                generated_narratives_data.append(output_entry)
                queries_processed_count += 1
            except (json.JSONDecodeError, Exception):
                pass
        
        output_file = config.personalized_queries_path
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for entry in generated_narratives_data:
                f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

        duration = time.time() - stage_start_time
        end_stamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        avg_time = duration / queries_processed_count if queries_processed_count else 0.0

        print(f"\n[{end_stamp}] Completed narrative generation")
        print(f"  Processed: {queries_processed_count} queries")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Avg time: {avg_time:.2f}s/query")
        print(f"  Output: {output_file}")

        stats_snapshot = {}
        if hasattr(narrative_generator, "get_generation_stats"):
            stats_snapshot = narrative_generator.get_generation_stats()
            stats_snapshot.update({
                "dataset": config.dataset_name,
                "model": model_label,
                "queries_total": total_queries,
                "queries_processed": queries_processed_count,
                "started_at": start_stamp,
                "completed_at": end_stamp,
                "duration_seconds": duration,
                "avg_seconds_per_query": avg_time,
                "output_file": output_file
            })
            stats_path = os.path.splitext(output_file)[0] + "_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as stats_out:
                json.dump(stats_snapshot, stats_out, ensure_ascii=False, indent=2)
            print(f"  Stats saved: {stats_path}")

        # 记录到日志文件
        progress_logger.info(
            f"[DONE] dataset={config.dataset_name} model={model_label} "
            f"processed={queries_processed_count} duration={duration:.1f}s "
            f"avg_time={avg_time:.3f}s/query output={output_file}"
        )

        stage_success = True
        
    except Exception as e:
        logger.error(f"在画像生成阶段发生错误: {e}", exc_info=True)
    finally:
        if 'narrative_generator' in locals(): del narrative_generator
        gc.collect(); torch.cuda.empty_cache()
    
    logger.info(f"--- 阶段2: 个性化画像生成 {'完成' if stage_success else '失败'} ---")
    return stage_success

def main():
    args = parse_args()
    config = get_config()
    config.update(args)

    # 特别处理 fusion_rerank 模式的参数
    if args.mode == "fusion_rerank":
        # 确保这些参数被正确更新到config
        if args.initial_top_k is not None:
            config.initial_top_k = args.initial_top_k
            logger.info(f"设置 initial_top_k = {config.initial_top_k}")
        if args.final_top_k is not None:
            config.final_top_k = args.final_top_k
        if args.batch_size is not None:
            config.batch_size = args.batch_size

    start_time = time.time()
    logger.info(f"--- PersLitRank 运行 ---")
    logger.info(f"模式: {args.mode}, 数据集: {config.dataset_name} (类型: {config.dataset_type})")
    logger.info(f"当前进程 GPU: {config.device}")
    
    # 模式执行
    if args.mode == "extract_cognitive_features":
        run_cognitive_feature_extraction_stage(config)

    elif args.mode == "generate_narratives":
        run_narrative_generation_stage(config)

    elif args.mode == "retrieve":
        logger.info("--- 执行: 初始文档检索 ---")
        try:
            from feature_retrieval import main as retrieval_main
            retrieval_main()
            logger.info("--- 初始文档检索完成 ---")
        except ImportError: logger.error("无法导入 feature_retrieval。", exc_info=True)
        except Exception as e: logger.error(f"检索期间出错: {e}", exc_info=True)

    elif args.mode == "rerank":
        logger.info("--- 执行: 文档重排 ---")
        try:
            from rerank import main as rerank_main
            rerank_main()
            logger.info("--- 文档重排完成 ---")
        except ImportError: logger.error("无法导入 rerank。", exc_info=True)
        except Exception as e: logger.error(f"重排期间出错: {e}", exc_info=True)

    elif args.mode == "fusion_rerank":
        logger.info("--- 执行: 融合重排 (策略一) ---")
        if args.beta is None:
            logger.error("错误: 'fusion_rerank' 模式需要 '--beta' 参数。请提供一个0.0到1.0之间的值。")
            return
        
        # 再次确认参数
        logger.info(f"确认参数: initial_top_k={config.initial_top_k}, beta={args.beta}")
        
        # 检查输入文件是否存在
        retrieved_exists = os.path.exists(config.retrieved_results_path)
        profiles_exist = os.path.exists(config.personalized_queries_path)

        if not retrieved_exists:
            logger.error(f"融合重排输入缺失: {config.retrieved_results_path}。")
        elif not profiles_exist:
            logger.error(f"用于融合重排的画像文件缺失: {config.personalized_queries_path}。请使用目标长度 {config.personalized_text_target_length} 运行 'generate_narratives'。")
        
        if retrieved_exists and profiles_exist:
            try:
                from fusion_reranker import run_fusion_reranking
                run_fusion_reranking(config, args.beta)
                logger.info("--- 融合重排完成 ---")
            except ImportError:
                logger.error("无法导入 fusion_reranker。", exc_info=True)
            except Exception as e:
                logger.error(f"融合重排期间出错: {e}", exc_info=True)
        else:
            logger.warning("由于输入缺失，跳过融合重排。")
    
    elif args.mode == "simple_profile_rerank":
        rerank_mode = args.rerank_input_type
        logger.info(f"--- 执行: 简化个性化重排 (模式: {rerank_mode}) ---")
        
        # 检查输入文件是否存在
        retrieved_exists = os.path.exists(config.retrieved_results_path)
        profiles_exist = os.path.exists(config.personalized_queries_path)

        if not retrieved_exists:
            logger.error(f"简化重排输入缺失: {config.retrieved_results_path}。")
        elif not profiles_exist:
            logger.error(f"用于简化重排的画像文件缺失: {config.personalized_queries_path}。请使用目标长度 {config.personalized_text_target_length} 运行 'generate_narratives'。")
        
        if retrieved_exists and profiles_exist:
            try:
                from simple_profile_reranker import run_simple_profile_reranking
                top_k = config.final_top_k if hasattr(config, 'final_top_k') and config.final_top_k else 10
                run_simple_profile_reranking(config, top_k=top_k, rerank_mode=rerank_mode)
                logger.info(f"--- 简化个性化重排完成 (模式: {rerank_mode}) ---")
            except ImportError:
                logger.error("无法导入 simple_profile_reranker。", exc_info=True)
            except Exception as e:
                logger.error(f"简化重排期间出错: {e}", exc_info=True)
        else:
            logger.warning("由于输入缺失，跳过简化重排。")
            
    elif args.mode == "all":
        logger.info("--- 执行所有阶段 (不包括fusion_rerank) ---")
        # ...原有代码...
        
    end_time = time.time()
    total_duration = end_time - start_time
    progress_logger.info(
        "[run] complete mode=%s dataset=%s duration_sec=%.2f",
        args.mode,
        config.dataset_name,
        total_duration,
    )
    tqdm.write(f"Run complete in {total_duration:.1f}s (mode={args.mode}, dataset={config.dataset_name})")

if __name__ == "__main__":
    main()
