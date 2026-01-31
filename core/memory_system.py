# memory_system.py - 完整优化版（包含引用解析）
import json
from typing import Dict, List, Any, Optional, Tuple, Set
import re
from datetime import datetime
import numpy as np
from collections import defaultdict
import os

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from utils import FeatureExtractorRegistry, MemorySystemRegistry, logger, get_config
from sentence_transformers import SentenceTransformer
import torch
import gc

_shared_embedding_model = None
_embedding_model_device = "cpu"

SEQUENTIAL_MEMORY_TAG = "[SEQUENTIAL_MEMORY]"
WORKING_MEMORY_TAG = "[WORKING_MEMORY]"
LONG_EXPLICIT_TAG = "[LONG_EXPLICIT]"

# memory_system.py - 优化后的 KeyBERTExtractor
@FeatureExtractorRegistry.register('keybert')
class KeyBERTExtractor:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(KeyBERTExtractor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name=None, config=None):
        if self._initialized: return
        self.config = config or get_config()
        try:
            from keybert import KeyBERT
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            
            # 使用 Qwen3 模型（统一embedding模型）
            keybert_embedder_device = getattr(self.config, 'keybert_embedder_device', 'cpu')
            effective_model_name = model_name or getattr(self.config, 'keybert_model',
                '/workspace/PerMed/model/Qwen3-Embedding-0.6B')
            
            # Qwen3 模型需要 trust_remote_code
            keybert_sentence_model = SentenceTransformer(
                effective_model_name,
                device=keybert_embedder_device,
                trust_remote_code=True
            )
            self.model = KeyBERT(model=keybert_sentence_model)
            self._initialized = True
            logger.info(f"KeyBERT initialized with Qwen3 model on {keybert_embedder_device}.")
        except Exception as e:
            logger.error(f"Error initializing KeyBERT: {e}", exc_info=True)
            self._initialized = False
            raise

    def extract_terms(self, text: str, top_n=10) -> List[Tuple[str, float]]:
        """使用 Qwen3 模型优化的关键词提取"""
        if not self._initialized or not text or len(text.strip()) < 5:
            return []
        
        try:
            # GTE 模型性能更强，可以处理更复杂的 n-gram
            keywords = self.model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),  # 1-3 词的短语
                stop_words='english',
                use_mmr=True,
                diversity=0.4,  # 降低多样性以获得更相关的结果
                top_n=top_n * 4,  # 多提取一些用于精细筛选
                highlight=False,
                nr_candidates=30,  # 增加候选词数量
                use_maxsum=True  # 使用 Max Sum 策略获得更好的覆盖
            )
            
            # 使用更智能的后处理
            return self._advanced_post_process(keywords, text, top_n)
            
        except Exception as e:
            logger.error(f"KeyBERT extraction error: {e}", exc_info=True)
            return []

    def _advanced_post_process(self, keywords: List[Tuple[str, float]],
                               original_text: str, top_n: int) -> List[Tuple[str, float]]:
        """高级后处理，充分利用 GTE 模型的能力"""
        processed = []
        seen_roots = set()  # 用于去重相似概念
        
        # 预定义的科学领域常见模式
        scientific_patterns = {
            'methods': ['algorithm', 'method', 'approach', 'technique', 'framework', 'model'],
            'materials': ['material', 'compound', 'alloy', 'composite', 'polymer'],
            'processes': ['process', 'synthesis', 'fabrication', 'optimization', 'analysis']
        }
        
        for phrase, score in keywords:
            phrase = phrase.strip()
            
            # 跳过低质量的结果
            if score < 0.15:  # GTE 模型分数通常更高
                continue
            
            # 清理短语
            words = phrase.split()
            
            # 过滤规则
            if len(words) == 1:
                # 单词必须是：专有名词、技术术语或足够长
                if len(phrase) < 5 and not phrase.isupper() and phrase.lower() not in ['gnn', 'ffr', 'sinr', 'ml']:
                    continue
            
            # 检查是否是有意义的科学术语
            is_scientific = any(
                pattern in phrase.lower()
                for patterns in scientific_patterns.values()
                for pattern in patterns
            )
            
            # 去除冗余 - 检查是否已有相似概念
            is_redundant = False
            phrase_lower = phrase.lower()
            
            for seen in seen_roots:
                # 如果是子串或父串关系
                if seen in phrase_lower or phrase_lower in seen:
                    # 保留更具体的版本
                    if len(phrase) > len(seen):
                        seen_roots.remove(seen)
                        seen_roots.add(phrase_lower)
                    else:
                        is_redundant = True
                    break
            
            if not is_redundant:
                # 提升科学术语的分数
                if is_scientific:
                    score = min(score * 1.2, 1.0)
                
                processed.append((phrase, score))
                seen_roots.add(phrase_lower)
        
        # 按分数排序并返回
        processed.sort(key=lambda x: x[1], reverse=True)
        return processed[:top_n]

    def extract_concepts(self, text: str, top_n=10) -> List[Tuple[str, float]]:
        """提取概念，与 extract_terms 相同但可以有不同的参数"""
        return self.extract_terms(text, top_n)

class SequentialMemory:
    def __init__(self, capacity: int = 10, feature_extractor=None, config=None, embedding_model=None):
        self.recent_queries = []
        self.capacity = capacity
        self.term_usage = defaultdict(int)
        self.feature_extractor = feature_extractor
        self.config = config if config else get_config()
        self.reference_history = []  # 新增：存储引用历史

    def process_query(self, query: str, user_id: str, clicked_docs: List[Dict] = None) -> Dict[str, Any]:
        """处理查询，记录和提取相关概念，并识别引用关系"""
        self._update_memory(query, clicked_docs or [])
        
        # 提取引用信息
        reference_info = self._extract_references(query)
        
        # 解析引用（如果有历史查询）
        resolved_references = self._resolve_references(reference_info)
        
        # 提取与历史相关的概念
        related_concepts = self._extract_related_concepts(query)
        terminology_result = self._detect_terminology_consistency(query)
        
        return {
            "query": query,
            "related_previous_concepts": related_concepts,
            "sequential_terminology": terminology_result,
            "reference_info": reference_info,  # 新增
            "resolved_references": resolved_references  # 新增
        }

    def _update_memory(self, query: str, clicked_docs: List[Dict]) -> None:
        """更新记忆，记录查询和提取的术语"""
        self.recent_queries.append({
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "clicked_docs": clicked_docs
        })
        
        if len(self.recent_queries) > self.capacity:
            self.recent_queries.pop(0)
        
        # 更新术语使用频率
        if self.feature_extractor:
            for term, score in self.feature_extractor.extract_terms(query, top_n=8):
                # 根据分数加权
                self.term_usage[term] += score

    def _extract_references(self, query: str) -> Dict[str, Any]:
        """提取查询中的指代词和连接词"""
        references = {
            "pronouns": [],
            "connectors": [],
            "full_references": []  # 完整的引用短语
        }
        
        # 指代词模式 - 更全面
        pronoun_patterns = [
            (r'\bthese\s+([\w\s-]+?)(?:\s+(?:that|which|who)|[,.])', 'these'),
            (r'\bthis\s+([\w\s-]+?)(?:\s+(?:that|which|who)|[,.])', 'this'),
            (r'\bthose\s+([\w\s-]+?)(?:\s+(?:that|which|who)|[,.])', 'those'),
            (r'\*these\*\s+([\w\s-]+?)(?:\s+|[,.])', 'these_emphasized'),  # 处理*these*
        ]
        
        # 连接词模式
        connector_patterns = [
            r'(Building on (?:this|these)[^,]*)',
            r'(Beyond (?:these|this) [\w\s]+)',
            r'(Following (?:this|these)[^,]*)',
            r'(Based on (?:these|this) [\w\s]+)',
            r'(addressing \*?these\*? [\w\s-]+)',
        ]
        
        # 提取指代词及其上下文
        for pattern, pronoun_type in pronoun_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                context = match.group(1).strip()
                full_match = match.group(0).strip()
                references["pronouns"].append({
                    "type": pronoun_type,
                    "context": context,
                    "full_phrase": full_match,
                    "position": match.start()
                })
        
        # 提取连接词
        for pattern in connector_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                references["connectors"].append({
                    "phrase": match.group(1),
                    "position": match.start()
                })
                references["full_references"].append(match.group(1))
        
        return references

    def _resolve_references(self, reference_info: Dict[str, Any]) -> Dict[str, Any]:
        """解析引用，找到具体指代的内容"""
        resolved = {
            "pronoun_resolutions": [],
            "connector_context": []
        }
        
        if not self.recent_queries:
            return resolved
        
        # 获取上一个查询的关键概念
        last_query = self.recent_queries[-1]["query"]
        if self.feature_extractor:
            last_concepts = self.feature_extractor.extract_concepts(last_query, top_n=5)
            
            # 解析每个代词
            for pronoun_info in reference_info.get("pronouns", []):
                pronoun_type = pronoun_info["type"]
                context = pronoun_info["context"]
                
                # 根据上下文找到最可能的指代
                best_match = self._find_best_reference_match(context, last_concepts, last_query)
                if best_match:
                    resolved["pronoun_resolutions"].append({
                        "original": pronoun_info["full_phrase"],
                        "resolved": best_match,
                        "confidence": 0.8  # 可以基于相似度计算
                    })
        
        # 解析连接词的上下文
        for connector in reference_info.get("connectors", []):
            resolved["connector_context"].append({
                "connector": connector["phrase"],
                "previous_context": self._get_previous_context_summary()
            })
        
        return resolved

    def _find_best_reference_match(self, context: str, last_concepts: List[Tuple[str, float]],
                                   last_query: str) -> Optional[str]:
        """找到最匹配的引用内容"""
        context_words = set(context.lower().split())
        best_match = None
        best_score = 0
        
        # 检查概念匹配
        for concept, score in last_concepts:
            concept_words = set(concept.lower().split())
            overlap = len(context_words & concept_words)
            if overlap > best_score:
                best_score = overlap
                best_match = concept
        
        # 如果没有找到，尝试从完整查询中提取相关短语
        if not best_match and context:
            # 在上一个查询中查找相关的名词短语
            pattern = r'\b(\w+\s+)?' + re.escape(context.split()[0]) + r'(\s+\w+)*'
            matches = re.findall(pattern, last_query.lower())
            if matches:
                best_match = ' '.join(matches[0]).strip()
        
        return best_match

    def _get_previous_context_summary(self) -> str:
        """获取前面查询的上下文摘要"""
        if not self.recent_queries:
            return ""
        
        last_query = self.recent_queries[-1]["query"]
        if self.feature_extractor:
            concepts = self.feature_extractor.extract_concepts(last_query, top_n=3)
            return ", ".join([c[0] for c in concepts])
        return last_query[:50] + "..."

    def _extract_related_concepts(self, current_query: str) -> Dict[str, Any]:
        """提取与历史查询相关的概念，不做连续性判断"""
        if not self.recent_queries or not self.feature_extractor:
            return {"previous_concepts": [], "shared_concepts": []}
        
        # 获取当前查询的概念
        current_concepts = {
            concept.lower(): score
            for concept, score in self.feature_extractor.extract_concepts(current_query, top_n=5)
        }
        
        # 从最近的查询中提取相关概念
        all_previous_concepts = []
        shared_concepts = []
        
        # 分析最近的2-3个查询
        for recent in self.recent_queries[-3:]:
            if recent["query"]:
                prev_concepts = self.feature_extractor.extract_concepts(recent["query"], top_n=5)
                for concept, score in prev_concepts:
                    concept_lower = concept.lower()
                    all_previous_concepts.append((concept, score))
                    
                    # 如果在当前查询中也出现
                    if concept_lower in current_concepts:
                        shared_concepts.append((concept, min(score, current_concepts[concept_lower])))
        
        # 去重并排序
        shared_concepts = list(set(shared_concepts))
        shared_concepts.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "previous_concepts": all_previous_concepts[:5],  # 最相关的历史概念
            "shared_concepts": shared_concepts[:3]  # 共享的概念
        }

    def _detect_terminology_consistency(self, current_query: str) -> Dict[str, Any]:
        """检测术语的一致性使用"""
        if not self.feature_extractor:
            return {"detected": False, "consistent_terms": []}
        
        # 获取频繁使用的术语（根据加权分数）
        frequent_terms = [
            {"term": term, "weighted_frequency": freq}
            for term, freq in sorted(self.term_usage.items(), key=lambda x: x[1], reverse=True)
            if freq > 0.5  # 加权频率阈值
        ][:10]
        
        return {
            "detected": bool(frequent_terms),
            "consistent_terms": frequent_terms
        }

class WorkingMemory:
    def __init__(self, concept_limit: int = 20, feature_extractor=None, config=None):
        self.current_session_queries = []
        self.current_session_concepts = defaultdict(lambda: {"count": 0, "query_indices": []})
        self.concept_limit = concept_limit
        self.feature_extractor = feature_extractor
        self.config = config or get_config()
        self.session_evolution = []  # 新增：追踪session演进

    def process_query(self, query: str, sequential_memory_results: Dict[str, Any], clicked_docs: List[Dict] = None) -> Dict[str, Any]:
        if not query or not self.feature_extractor: return {"session_focus": None, "current_query_core_concepts": []}
        self._update_session_state(query, clicked_docs or [])
        current_query_concepts_data = self.feature_extractor.extract_concepts(query, top_n=5)
        current_query_core_concepts = [concept for concept, score in current_query_concepts_data if score > 0.1]
        session_focus = self._determine_session_focus()
        return {"session_focus": session_focus, "current_query_core_concepts": current_query_core_concepts}

    def _update_session_state(self, query: str, clicked_docs: List[Dict]) -> None:
        self.current_session_queries.append(query)
        query_idx = len(self.current_session_queries) - 1
        if self.feature_extractor:
            concepts = self.feature_extractor.extract_concepts(query, top_n=10)
            # 记录当前查询的主要概念用于演进追踪
            if concepts:
                main_concept = concepts[0][0] if concepts[0][1] > 0.3 else None
                if main_concept:
                    self.session_evolution.append(main_concept)
            
            for concept, _ in concepts:
                self.current_session_concepts[concept]["count"] += 1
                if query_idx not in self.current_session_concepts[concept]["query_indices"]:
                    self.current_session_concepts[concept]["query_indices"].append(query_idx)

    def _determine_session_focus(self) -> Optional[str]:
        """改进的session focus确定，体现研究演进"""
        if not self.current_session_concepts: return None
        
        # 1. 获取当前查询的主要概念
        recent_concepts = []
        if len(self.current_session_queries) > 0:
            last_query = self.current_session_queries[-1]
            if self.feature_extractor:
                last_concepts = self.feature_extractor.extract_concepts(last_query, top_n=2)
                recent_concepts = [c[0] for c in last_concepts if c[1] > 0.3]
        
        # 2. 获取历史高频概念
        sorted_concepts = sorted(
            self.current_session_concepts.items(),
            key=lambda item: (item[1]["count"], len(item[1]["query_indices"])),
            reverse=True
        )
        
        historical_concepts = []
        for concept, data in sorted_concepts[:3]:
            if data["count"] >= 2:
                historical_concepts.append(concept)
        
        # 3. 构建演进式的focus
        if len(self.session_evolution) > 1 and recent_concepts:
            # 显示演进路径
            if historical_concepts and recent_concepts[0] != historical_concepts[0]:
                return f"{recent_concepts[0]} (evolved from {historical_concepts[0]})"
            else:
                return recent_concepts[0]
        elif recent_concepts and historical_concepts:
            # 组合当前和历史概念
            if recent_concepts[0] != historical_concepts[0]:
                return f"{recent_concepts[0]} + {historical_concepts[0]}"
            else:
                return recent_concepts[0]
        elif historical_concepts:
            # 多个历史概念
            if len(historical_concepts) > 1:
                return " + ".join(historical_concepts[:2])
            else:
                return historical_concepts[0]
        elif sorted_concepts:
            return sorted_concepts[0][0]
        return None

    def new_session(self):
        self.current_session_queries = []
        self.current_session_concepts.clear()
        self.session_evolution = []
        logger.info("WorkingMemory: New session started.")

class LongTermMemory:
    def __init__(self, feature_extractor=None, vector_file=None, embedding_model_name=None, config=None):
        self.explicit_memory = {"research_topics": defaultdict(float), "methodologies": defaultdict(float)}
        self.implicit_memory = {"academic_background": {}, "technical_familiarity": defaultdict(float), "academic_level": "unknown", "level_confidence": 0.0}
        self.feature_extractor = feature_extractor
        self.config = config if config else get_config()
        self.vectors = {"topics": {}, "methods": {}}
        self.indices = {}
        self.topic_evolution = []  # 新增：追踪研究主题演变

        global _shared_embedding_model, _embedding_model_device
        try:
            if _shared_embedding_model is None:
                _embedding_model_device = self.config.device if torch.cuda.is_available() and self.config.device and self.config.device != "cpu" else "cpu"
                model_path = embedding_model_name or getattr(self.config, 'sentence_transformer_model', 'all-MiniLM-L6-v2')
                use_trust = "gte" in model_path.lower() or "modelscope" in model_path.lower()
                _shared_embedding_model = SentenceTransformer(model_path, device=_embedding_model_device, trust_remote_code=use_trust)
            self.embedding_model = _shared_embedding_model
        except Exception as e:
            logger.error(f"LTM SBERT model error: {e}", exc_info=True)
            self.embedding_model = None

    def update(self, query: str, working_memory_state: Dict[str, Any], clicked_docs: List[Dict] = None) -> None:
        """优化后的长期记忆更新，更智能的主题和方法论识别"""
        if not self.feature_extractor: return
        try:
            # 提取概念时考虑上下文权重
            concepts = self.feature_extractor.extract_concepts(query, top_n=5)
            
            # 记录主题演变
            if concepts and concepts[0][1] > 0.3:
                self.topic_evolution.append(concepts[0][0])
            
            for concept, score in concepts:
                # 1. 对研究主题的更新要考虑衰减和阈值
                if concept in self.explicit_memory["research_topics"]:
                    # 已存在的概念，增强但有衰减
                    old_score = self.explicit_memory["research_topics"][concept]
                    self.explicit_memory["research_topics"][concept] = old_score * 0.9 + score
                else:
                    # 新概念，需要较高分数才加入
                    if score > 0.25:  # 提高阈值
                        self.explicit_memory["research_topics"][concept] = score
            
            # 2. 处理点击的文档
            if clicked_docs:
                for doc in clicked_docs:
                    content = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                    if not content: continue
                    for concept, score in self.feature_extractor.extract_concepts(content, top_n=3):
                        if score > 0.3:
                            self.explicit_memory["research_topics"][concept] += score * 0.5
            
            # 3. 改进的方法论提取
            method_indicators = [
                "method", "approach", "algorithm", "technique", "analysis",
                "model", "framework", "system", "architecture", "design",
                "optimization", "learning", "network", "process", "strategy"
            ]
            
            # 方法论短语模式
            method_patterns = [
                r"(machine|deep|reinforcement) learning",
                r"neural network",
                r"(optimization|prediction|classification) (method|approach|algorithm)",
                r"(computational|experimental|theoretical) (approach|method)",
                r"graph (neural|convolutional) network"
            ]
            
            for concept, score in concepts:
                concept_lower = concept.lower()
                
                # 检查是否匹配方法论模式
                is_methodology = False
                
                # 检查指示词
                if any(indicator in concept_lower for indicator in method_indicators):
                    is_methodology = True
                
                # 检查模式匹配
                for pattern in method_patterns:
                    if re.search(pattern, concept_lower):
                        is_methodology = True
                        break
                
                # 如果是方法论且分数足够高
                if is_methodology and score > 0.3:
                    self.explicit_memory["methodologies"][concept] += score
                    
        except Exception as e:
            logger.error(f"LTM update error: {e}", exc_info=True)

    def retrieve(self, query: str, working_memory_state: Dict[str, Any]) -> Dict[str, Any]:
        # 按分数排序并限制数量，避免返回太多低分项
        persistent_research_topics = [
            topic for topic, strength in sorted(
                self.explicit_memory["research_topics"].items(),
                key=lambda x: x[1],
                reverse=True
            ) if strength > 0.1  # 只返回分数大于0.1的主题
        ][:10]  # 最多返回10个
        
        persistent_methodologies = [
            method for method, strength in sorted(
                self.explicit_memory["methodologies"].items(),
                key=lambda x: x[1],
                reverse=True
            ) if strength > 0.1
        ][:5]  # 最多返回5个方法
        
        query_relevant_ltm_topics = []
        if self.feature_extractor:
            current_query_concepts = [c for c, _ in self.feature_extractor.extract_concepts(query, top_n=3)]
            for topic in persistent_research_topics:
                # 更严格的相关性判断
                if any(
                    qc.lower() in topic.lower() or
                    topic.lower() in qc.lower() or
                    self._concept_similarity(qc, topic) > 0.7
                    for qc in current_query_concepts
                ):
                    query_relevant_ltm_topics.append(topic)

        return {
            "explicit_memory_keywords": {
                "persistent_research_topics": persistent_research_topics,
                "persistent_methodologies": persistent_methodologies,
                "query_relevant_ltm_topics": query_relevant_ltm_topics[:5]  # 限制数量
            },
            "implicit_memory_snapshot": self._retrieve_implicit_memory_snapshot()
        }

    def _concept_similarity(self, concept1: str, concept2: str) -> float:
        """计算两个概念的相似度"""
        # 简单的词重叠相似度
        words1 = set(concept1.lower().split())
        words2 = set(concept2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    def _retrieve_implicit_memory_snapshot(self) -> Dict[str, Any]:
        max_items = getattr(self.config, 'max_phrases_per_tag', 3)
        acad_bg = [{"discipline": d.replace("_", " "), "confidence": c} for d,c in sorted(self.implicit_memory["academic_background"].items(),key=lambda x:x[1],reverse=True)[:max_items]]
        top_techs = [{"technology": t, "familiarity": f} for t,f in sorted(self.implicit_memory["technical_familiarity"].items(),key=lambda x:x[1],reverse=True)[:max_items]]
        return {"academic_background_profile": acad_bg,
                "academic_level_profile": {"level": self.implicit_memory["academic_level"], "confidence": self.implicit_memory["level_confidence"]},
                "top_technologies_profile": top_techs}
    
    def new_session(self):
        self.topic_evolution = []

class CognitiveMemorySystem:
    def __init__(self, config=None):
        self.user_profiles = {}
        self.config = config if config else get_config()
        self.dataset_type = getattr(self.config, 'dataset_type', 'unknown')
        self.feature_extractor_type = getattr(self.config, 'feature_extractor', 'keybert')
        global _shared_embedding_model
        self.embedding_model_instance = _shared_embedding_model
        try:
            keybert_model_path = getattr(self.config, 'keybert_model', None)
            self.feature_extractor = FeatureExtractorRegistry.get_extractor(self.feature_extractor_type, model_name=keybert_model_path, config=self.config)
        except Exception as e:
            logger.warning(f"CMS: Failed to init {self.feature_extractor_type} extractor: {e}. Using simple.", exc_info=True)
            self.feature_extractor = self._create_simple_extractor()

    def _create_simple_extractor(self):
        class SimpleExtractor:
            def extract_terms(self, text, top_n=10):
                words=re.findall(r'\b[a-zA-Z]{3,}\b',text.lower())
                c=defaultdict(int)
                for w in words: c[w] += 1
                s={'the','and','is','in','to','of','that','for','on','with','an','are'}
                l=len(words)
                return [(w,v/(l or 1)) for w,v in sorted(c.items(),key=lambda x:x[1],reverse=True) if w not in s][:top_n]
            def extract_concepts(self, text, top_n=10): return self.extract_terms(text, top_n)
        return SimpleExtractor()

    def _get_or_initialize_user_memory(self, user_id: str):
        if user_id not in self.user_profiles:
            logger.info(f"CMS: Initializing new memory profile for user_id: {user_id}")
            self.user_profiles[user_id] = {
                "sequential_memory": SequentialMemory(feature_extractor=self.feature_extractor, config=self.config, embedding_model=self.embedding_model_instance),
                "working_memory": WorkingMemory(feature_extractor=self.feature_extractor, config=self.config),
                "long_term_memory": LongTermMemory(feature_extractor=self.feature_extractor, config=self.config),
                "current_topic_id": None
            }
        return self.user_profiles[user_id]

    def process_query(self, query_text: str, user_id: str, clicked_docs: List[Dict] = None, topic_id: Optional[str] = None) -> Dict[str, Any]:
        user_memory = self._get_or_initialize_user_memory(user_id)
        if topic_id and user_memory.get("current_topic_id") != topic_id:
            user_memory["working_memory"].new_session()
            user_memory["current_topic_id"] = topic_id
        seq_mem, work_mem, lt_mem = user_memory["sequential_memory"], user_memory["working_memory"], user_memory["long_term_memory"]
        seq_res_raw = seq_mem.process_query(query_text, user_id, clicked_docs)
        work_state_raw = work_mem.process_query(query_text, seq_res_raw, clicked_docs)
        lt_mem.update(query_text, work_state_raw, clicked_docs)
        ltm_res_raw = lt_mem.retrieve(query_text, work_state_raw)
        return {"sequential_results_raw": seq_res_raw, "working_memory_state_raw": work_state_raw, "long_term_memory_results_raw": ltm_res_raw}

    def get_tagged_features(self, memory_results_raw: Dict[str, Any],
                            active_components: List[str] = None,
                            turn_id: int = 0) -> List[str]:
        """
        【V2优化】动态优先级的特征标记生成
        
        核心改进：
        1. 所有特征候选都有优先级分数
        2. 引用信息获得最高优先级（10.0）
        3. 根据 turn_id 动态调整各模块优先级
        4. 确保高质量特征不会因为模块配额被截断
        """
        current_active_components = [c.lower() for c in (active_components or getattr(self.config, 'memory_components', []))]
        config = self.config
        max_overall_features = getattr(config, 'max_tagged_features_for_llm', 10)
        max_kws_in_str = getattr(config, 'max_phrases_per_tag', 5)

        # ===== 第一步：收集所有候选特征并评分 =====
        all_candidate_features = []  # [(feature_string, priority_score, source_module)]
        used_concepts = set()

        # Sequential Memory - 引用信息给最高优先级
        if 'sequential' in current_active_components:
            seq_raw = memory_results_raw.get("sequential_results_raw", {})
            reference_info = seq_raw.get("reference_info", {})
            resolved_refs = seq_raw.get("resolved_references", {})
            
            # 1. 引用解析 - 最高优先级（10.0）
            if reference_info.get("pronouns") or reference_info.get("connectors"):
                ref_features = []
                
                # 处理代词解析
                for resolution in resolved_refs.get("pronoun_resolutions", []):
                    ref_features.append(f"'{resolution['original']}' refers to: {resolution['resolved']}")
                
                # 处理连接词
                for connector_info in resolved_refs.get("connector_context", []):
                    ref_features.append(f"Context for '{connector_info['connector']}': {connector_info['previous_context']}")
                
                if ref_features:
                    feature_str = f"{SEQUENTIAL_MEMORY_TAG} References: {'; '.join(ref_features[:2])}"
                    priority = 10.0  # 🔥 最高优先级
                    all_candidate_features.append((feature_str, priority, "sequential"))
                    logger.debug(f"Found reference with priority {priority}")
            
            # 2. 共享概念 - 高优先级（7.5-8.5）
            related_info = seq_raw.get("related_previous_concepts", {})
            shared_concepts = related_info.get("shared_concepts", [])
            
            if shared_concepts:
                concepts_str = ", ".join([c[0] for c in shared_concepts[:max_kws_in_str] if c[0] not in used_concepts])
                if concepts_str:
                    # 优先级根据共享概念数量动态调整
                    priority = 7.5 + min(len(shared_concepts) * 0.2, 1.0)
                    feature_str = f"{SEQUENTIAL_MEMORY_TAG} Continuing exploration of: {concepts_str}"
                    all_candidate_features.append((feature_str, priority, "sequential"))
                    used_concepts.update([c[0] for c in shared_concepts[:max_kws_in_str]])
            
            # 3. 术语一致性 - 中等优先级（5.0-6.5）
            term_info = seq_raw.get("sequential_terminology", {})
            consistent_terms = [
                item['term'] for item in term_info.get("consistent_terms", [])[:max_kws_in_str]
                if item['weighted_frequency'] > 1.0 and item['term'] not in used_concepts
            ]
            
            if consistent_terms:
                priority = 5.0 + min(len(consistent_terms) * 0.3, 1.5)
                feature_str = f"{SEQUENTIAL_MEMORY_TAG} Established terminology: {', '.join(consistent_terms)}"
                all_candidate_features.append((feature_str, priority, "sequential"))
                used_concepts.update(consistent_terms)

        # Working Memory - 根据轮次动态调整优先级
        if 'working' in current_active_components:
            wm_raw = memory_results_raw.get("working_memory_state_raw", {})
            focus = wm_raw.get("session_focus")
            
            if focus:
                # 🎯 Turn 1-2: 高优先级（帮助建立上下文）
                # Turn 3+: 中等优先级（上下文已建立）
                if turn_id <= 2:
                    priority = 9.0  # 早期轮次，焦点很重要
                else:
                    priority = 6.0  # 后期轮次，降低优先级
                
                if "evolved from" in focus:
                    feature_str = f"{WORKING_MEMORY_TAG} Research evolution: {focus}"
                    priority += 0.5  # 演进信息额外加分
                elif " + " in focus:
                    feature_str = f"{WORKING_MEMORY_TAG} Current session exploring: {focus}"
                else:
                    feature_str = f"{WORKING_MEMORY_TAG} Session focus: {focus}"
                
                all_candidate_features.append((feature_str, priority, "working"))
                
                # 记录focus中的概念
                focus_concepts = re.findall(r'\b\w+\b', focus.lower())
                used_concepts.update(focus_concepts)
            
            # 当前查询核心概念 - 始终重要
            core_concepts = wm_raw.get("current_query_core_concepts", [])
            if core_concepts:
                meaningful_concepts = [
                    c for c in core_concepts[:max_kws_in_str]
                    if len(c) > 3 and c not in used_concepts
                ]
                if meaningful_concepts:
                    priority = 8.5  # 当前概念总是重要
                    feature_str = f"{WORKING_MEMORY_TAG} Query emphasizes: {', '.join(meaningful_concepts)}"
                    all_candidate_features.append((feature_str, priority, "working"))
                    used_concepts.update(meaningful_concepts)

        # Long-Term Memory - 后期轮次才重要
        if 'long' in current_active_components:
            ltm_raw = memory_results_raw.get("long_term_memory_results_raw", {})
            explicit_kws = ltm_raw.get("explicit_memory_keywords", {})
            
            # 🎯 LTM 的优先级随轮次增加而提高
            ltm_base_priority = min(3.0 + turn_id * 0.5, 7.0)
            
            # 研究主题
            persistent_topics = explicit_kws.get("persistent_research_topics", [])
            if persistent_topics:
                top_topics = [
                    t for t in persistent_topics[:max_kws_in_str]
                    if len(t) > 4 and t not in used_concepts
                ]
                if top_topics:
                    priority = ltm_base_priority
                    feature_str = f"{LONG_EXPLICIT_TAG} Established research areas: {', '.join(top_topics)}"
                    all_candidate_features.append((feature_str, priority, "long"))
                    used_concepts.update(top_topics)
            
            # 方法论专长
            persistent_methods = explicit_kws.get("persistent_methodologies", [])
            if persistent_methods:
                method_list = [m for m in persistent_methods[:max_kws_in_str] if m not in used_concepts]
                if method_list:
                    priority = ltm_base_priority + 0.5
                    feature_str = f"{LONG_EXPLICIT_TAG} Methodological expertise: {', '.join(method_list)}"
                    all_candidate_features.append((feature_str, priority, "long"))
                    used_concepts.update(method_list)
            
            # 查询相关的历史兴趣
            query_relevant_ltm = explicit_kws.get("query_relevant_ltm_topics", [])
            if query_relevant_ltm and turn_id >= 2:  # 只在第2轮后使用
                relevant_unique = [
                    t for t in query_relevant_ltm[:max_kws_in_str]
                    if t not in used_concepts
                ]
                if relevant_unique:
                    priority = ltm_base_priority + 0.3
                    feature_str = f"{LONG_EXPLICIT_TAG} Related past interests: {', '.join(relevant_unique)}"
                    all_candidate_features.append((feature_str, priority, "long"))
        
        # ===== 第二步：按优先级排序并选择 =====
        all_candidate_features.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前N个，但确保多样性（不要全来自一个模块）
        selected_features = []
        module_counts = defaultdict(int)
        max_per_module = max(max_overall_features // 2, 3)  # 任何模块最多占50%，但至少3条
        
        for feature_str, priority, source_module in all_candidate_features:
            if len(selected_features) >= max_overall_features:
                break
            
            # 避免某个模块垄断（但优先级>=9.5的总是保留）
            if priority < 9.5 and module_counts[source_module] >= max_per_module:
                continue
            
            selected_features.append(feature_str)
            module_counts[source_module] += 1
        
        logger.info(f"Turn {turn_id}: Selected {len(selected_features)} features. "
                   f"Distribution: {dict(module_counts)}")
        if all_candidate_features:
            top3 = [(f[:60] + "..." if len(f) > 60 else f, f"{p:.1f}") 
                    for f, p, _ in all_candidate_features[:3]]
            logger.debug(f"Top 3 priorities: {top3}")
        
        return selected_features

    def new_user_session(self, user_id: str, topic_id: str):
        self._get_or_initialize_user_memory(user_id)
        user_memory = self.user_profiles[user_id]
        if user_memory.get("current_topic_id") != topic_id:
            logger.info(f"CMS: New session for user '{user_id}', topic_id '{topic_id}'. Resetting Working Memory.")
            if "working_memory" in user_memory and hasattr(user_memory["working_memory"], "new_session"):
                user_memory["working_memory"].new_session()
            user_memory["current_topic_id"] = topic_id
