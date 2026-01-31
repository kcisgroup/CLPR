#!/usr/bin/env python
# memory_ablation.py - 修改版：仅生成消融后的画像
"""
上下文特征维度消融实验 - 画像生成阶段
读取详细的认知特征，根据消融设置过滤这些特征，
使用 PersonalizedGenerator 生成消融版的个性化画像。
生成的画像将保存到文件中，供后续融合重排使用。
"""
import os
import json
import argparse
import logging
import torch
import re 
import gc
from typing import List, Dict, Any, Optional, Set, Tuple
from tqdm import tqdm
from collections import defaultdict

# 导入必要的模块
try:
    from core.utils import get_config, logger as ablation_logger
    from core.personalized_generator import PersonalizedGenerator
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ablation_logger = logging.getLogger("MemoryAblation_Fallback")
    ablation_logger.error(f"未能导入必要的模块: {e}。")
    raise

logger = ablation_logger


def build_deterministic_profile_from_features(
    filtered_features: List[str],
    query_text: str,
    max_chars: Optional[int] = None,
) -> str:
    """基于过滤后的记忆特征构造确定性的 profile 文本

    设计目标：
    - 只依赖于保留的 tagged_memory_features，不再调用 LLM；
    - 不同的消融设置（no_long / no_sequential / no_working）会直接反映到文本内容中；
    - 控制大致长度，保证与原 profile-only 重排设置兼容。
    """

    # 去掉前缀标签，例如 "[SEQUENTIAL_MEMORY] ", "[WORKING_MEMORY] " 等
    phrases: List[str] = []
    for feat in filtered_features or []:
        cleaned = re.sub(r"^\[[A-Z_]+\]\s*", "", feat).strip()
        if cleaned:
            phrases.append(cleaned)

    if phrases:
        # 只基于记忆特征来描述研究焦点
        base = "Researcher focusing on: " + "; ".join(phrases)
    else:
        # 如果某个消融模式下该查询几乎没有记忆特征，则退化为基于 query 的简短描述
        qt = (query_text or "").strip()
        if not qt:
            base = "General medical research interests"
        else:
            base = f"Researcher investigating: {qt}"

    if max_chars is not None and max_chars > 0 and len(base) > max_chars:
        # 尝试在分号或空格边界截断，保证语义完整度
        truncated = base[:max_chars]
        cut_pos = max(truncated.rfind(";"), truncated.rfind(" "))
        min_prefix = len("Researcher focusing on: ")
        if cut_pos > min_prefix:
            truncated = truncated[:cut_pos]
        base = truncated.rstrip()

    return base

def filter_memory_features_for_ablation(all_features: List[str], exclude_dimension_name: Optional[str]) -> List[str]:
    """根据消融设置过滤内存特征"""
    if exclude_dimension_name is None or exclude_dimension_name.lower() == "none" or not all_features:
        return all_features 
    
    filtered_features = []
    DIMENSION_TAG_MAP = {
        "sequential": "[SEQUENTIAL_MEMORY]",
        "working": "[WORKING_MEMORY]",
        "long": "[LONG_EXPLICIT]", 
    }
    
    tag_to_exclude_prefix = DIMENSION_TAG_MAP.get(exclude_dimension_name.lower())
    if not tag_to_exclude_prefix:
        logger.warning(f"无效的排除维度名称: '{exclude_dimension_name}'. 将返回所有特征。")
        return all_features
    
    for feature_str in all_features:
        if feature_str.startswith(tag_to_exclude_prefix):
            continue
        filtered_features.append(feature_str)
    
    logger.debug(f"特征维度消融: 排除 '{exclude_dimension_name}' 维度。保留 {len(filtered_features)}/{len(all_features)} 条标记特征。")
    return filtered_features

def load_cognitive_features(cognitive_features_path: str) -> List[Dict[str, Any]]:
    """加载认知特征数据"""
    cognitive_data = []
    
    if not os.path.exists(cognitive_features_path):
        logger.error(f"认知特征文件未找到: {cognitive_features_path}")
        return []
    
    try:
        with open(cognitive_features_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    cognitive_data.append(data)
                except json.JSONDecodeError: 
                    logger.warning(f"跳过 {cognitive_features_path} 中的无效JSON行")
    except Exception as e: 
        logger.error(f"加载 {cognitive_features_path} 时出错: {e}")
        return []
    
    logger.info(f"从 {cognitive_features_path} 加载了 {len(cognitive_data)} 条认知特征记录")
    return cognitive_data

def main():
    parser = argparse.ArgumentParser(description="上下文特征维度消融实验 - 画像生成")
    
    # 基本参数
    parser.add_argument("--dataset_name", type=str, required=True, choices=["MedCorpus", "LitSearch"])
    parser.add_argument("--exclude_dimension", type=str, default="none", 
                        choices=["sequential", "working", "long", "none"],
                        help="要排除的内存维度")
    parser.add_argument("--cognitive_features_input_path", type=str, required=True,
                        help="输入的认知特征文件路径")
    parser.add_argument("--output_path", type=str, required=True,
                        help="输出的消融画像文件路径")
    
    # 画像生成模式
    parser.add_argument(
        "--profile_generation_mode",
        type=str,
        default="llm",
        choices=["llm", "deterministic"],
        help="画像生成模式: 'llm' 使用 PersonalizedGenerator, 'deterministic' 直接由记忆特征拼接文本",
    )

    # LLM参数（仅在 profile_generation_mode='llm' 时真正生效）
    parser.add_argument("--personalized_text_target_length", type=int, default=150)
    parser.add_argument("--llm_api_type", type=str, default="ollama")
    parser.add_argument("--llm_base_url", type=str, default="http://172.18.147.77:11434")
    parser.add_argument("--llm_model", type=str, default="llama3:8b")
    
    # 其他参数
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--test_query_limit", type=int, default=None)

    args = parser.parse_args()

    # 配置设置
    config = get_config()
    config.dataset_name = args.dataset_name
    config.dataset_type = config._infer_dataset_type()
    config.personalized_text_target_length = args.personalized_text_target_length
    # 旧版本会用 personalized_text_max_length / length_suffix 控制文件名和长度；
    # 这里保持兼容，即使 deterministic 模式下不会用到 LLM。
    if hasattr(config, "personalized_text_max_length"):
        config.personalized_text_max_length = args.personalized_text_target_length
    if hasattr(config, "length_suffix"):
        config.length_suffix = f"_L{config.personalized_text_target_length}"
        if hasattr(config, "_update_text_length_constraints"):
            config._update_text_length_constraints()

    # LLM 相关配置只在 'llm' 模式下真正起作用，但保留赋值不影响 deterministic 模式
    config.llm_api_type = args.llm_api_type
    config.llm_base_url = args.llm_base_url
    config.llm_model = args.llm_model
    config.gpu_id = args.gpu_id
    config.device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    config.llm_device = config.device
    config.max_concurrent_requests = 50
    config.use_fixed_seed = False  # 消融实验必须禁用固定seed
    config.local_model_temperature = 0.7  # 提高温度以增加多样性

    logger.info("--- 上下文特征维度消融 - 画像生成 ---")
    logger.info(f"数据集: {args.dataset_name}, 排除维度: {args.exclude_dimension}")
    logger.info(f"画像生成模式: {args.profile_generation_mode}")
    if args.profile_generation_mode == "llm":
        logger.info(f"LLM: {config.llm_model}, 目标长度: {config.personalized_text_target_length}")
    logger.info(f"输入: {args.cognitive_features_input_path}")
    logger.info(f"输出: {args.output_path}")

    # 加载认知特征
    cognitive_data = load_cognitive_features(args.cognitive_features_input_path)
    if not cognitive_data:
        logger.error("未能加载认知特征数据。退出。")
        return

    # 如果设置了test_query_limit，限制处理的查询数
    if args.test_query_limit is not None and args.test_query_limit > 0:
        if args.dataset_name == "MedCorpus":
            # 对MedCorpus，按topic限制
            topic_queries = defaultdict(list)
            for item in cognitive_data:
                topic_id = item.get("topic_id", "")
                if not topic_id and "query_id" in item:
                    topic_id = '_'.join(str(item["query_id"]).split('_')[:-1])
                topic_queries[topic_id].append(item)
            
            # 只保留前N个topic的数据
            sorted_topics = sorted(topic_queries.keys())[:args.test_query_limit]
            filtered_data = []
            for topic in sorted_topics:
                filtered_data.extend(topic_queries[topic])
            cognitive_data = filtered_data
            logger.info(f"MedCorpus: 限制为前 {args.test_query_limit} 个topics，共 {len(cognitive_data)} 个查询")
        else:
            # 对其他数据集，直接限制查询数
            cognitive_data = cognitive_data[:args.test_query_limit]
            logger.info(f"限制为前 {args.test_query_limit} 个查询")

    # 初始化画像生成器（仅在 LLM 模式下）
    narrative_generator = None
    if args.profile_generation_mode == "llm":
        try:
            narrative_generator = PersonalizedGenerator(config=config)
        except Exception as e:
            logger.error(f"初始化 PersonalizedGenerator 失败: {e}。退出。", exc_info=True)
            return

    # 生成消融画像 - 按topic分组并行
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 按topic分组
    topic_groups = defaultdict(list)
    for item in cognitive_data:
        topic_id = item.get("topic_id", "")
        if not topic_id and "query_id" in item:
            topic_id = '_'.join(str(item["query_id"]).split('_')[:-1])
        topic_groups[topic_id].append(item)

    def process_topic(topic_id, items):
        """处理单个topic的所有查询（顺序执行以保持依赖）"""
        results = []
        current_profile = None
        for item in items:
            query_id = item.get("query_id")
            query_text = item.get("query", "")
            turn_id = item.get("turn_id", 0)

            if not query_id or not query_text:
                continue

            all_tagged_features = item.get("tagged_memory_features", [])
            filtered_features = filter_memory_features_for_ablation(all_tagged_features, args.exclude_dimension)
            memory_input = {"tagged_memory_features": filtered_features}

            try:
                if args.profile_generation_mode == "llm":
                    # 原始 LLM 画像生成路径
                    ablated_narrative = narrative_generator.generate_personalized_text(
                        query=query_text,
                        memory_results=memory_input,
                        previous_profile=current_profile,
                        turn_id=turn_id,
                    )
                    current_profile = ablated_narrative
                else:
                    # 确定性画像：只基于保留的记忆特征构造 profile 文本
                    max_chars = args.personalized_text_target_length if args.personalized_text_target_length else None
                    ablated_narrative = build_deterministic_profile_from_features(
                        filtered_features, query_text, max_chars=max_chars
                    )
                    current_profile = ablated_narrative

                result_entry = {
                    "query_id": query_id,
                    "query": query_text,
                    "personalized_features": ablated_narrative,
                    "tagged_memory_features": filtered_features,
                    "excluded_dimension": args.exclude_dimension
                }
                for key in ["topic_id", "turn_id"]:
                    if key in item:
                        result_entry[key] = item[key]
                results.append(result_entry)
            except Exception as e:
                logger.error(f"为查询 {query_id} 生成画像时出错: {e}")
                result_entry = {
                    "query_id": query_id,
                    "query": query_text,
                    "personalized_features": "",
                    "tagged_memory_features": filtered_features,
                    "excluded_dimension": args.exclude_dimension,
                    "error": str(e)
                }
                for key in ["topic_id", "turn_id"]:
                    if key in item:
                        result_entry[key] = item[key]
                results.append(result_entry)
        return results

    # 并行处理所有topic
    ablated_profiles = []
    with ThreadPoolExecutor(max_workers=config.max_concurrent_requests) as executor:
        futures = {executor.submit(process_topic, tid, items): tid for tid, items in topic_groups.items()}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"生成消融画像 (排除: {args.exclude_dimension})"):
            try:
                results = future.result()
                ablated_profiles.extend(results)
            except Exception as e:
                logger.error(f"处理topic时出错: {e}")

    # 保存结果
    logger.info(f"正在将 {len(ablated_profiles)} 条消融画像保存到: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    try:
        with open(args.output_path, 'w', encoding='utf-8') as f:
            for profile in ablated_profiles:
                f.write(json.dumps(profile, ensure_ascii=False) + '\n')
        logger.info("消融画像生成和保存成功完成。")
    except IOError as e: 
        logger.error(f"写入结果到 {args.output_path} 失败: {e}")

    # 清理资源
    if narrative_generator is not None:
        del narrative_generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    main()
