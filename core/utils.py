# utils.py - 支持Llama3 API和SiliconFlow API
import os
import logging
import torch
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from tqdm import tqdm

# 配置日志记录 - 控制台仅警告/错误，详细时间记录单独写入文件
logging.basicConfig(level=logging.CRITICAL)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

logger = logging.getLogger('PersLitRank')
logger.setLevel(logging.WARNING)
logger.addHandler(console_handler)
logger.propagate = False

_progress_handler = logging.FileHandler('perslitrank.log')
_progress_handler.setLevel(logging.INFO)
_progress_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

progress_logger = logging.getLogger('PersLitRankProgress')
progress_logger.setLevel(logging.INFO)
progress_logger.addHandler(_progress_handler)
progress_logger.propagate = False

# 注册表类 - memory_system.py 需要这些
class FeatureExtractorRegistry:
    _extractors = {}
    @classmethod
    def register(cls, name):
        def decorator(extractor_class):
            cls._extractors[name] = extractor_class
            return extractor_class
        return decorator
    @classmethod
    def get_extractor(cls, name, **kwargs):
        if name not in cls._extractors: raise ValueError(f"Unknown feature extractor: {name}")
        return cls._extractors[name](**kwargs)

class MemorySystemRegistry:
    _systems = {}
    @classmethod
    def register(cls, name):
        def decorator(system_class):
            cls._systems[name] = system_class
            return system_class
        return decorator
    @classmethod
    def get_system(cls, name, **kwargs):
        if name not in cls._systems: raise ValueError(f"Unknown memory system: {name}")
        return cls._systems[name](**kwargs)

@dataclass
class Document:
    text_id: str
    title: str = ""
    text: str = ""
    full_paper: Optional[str] = None
    full_text: Optional[str] = None
    score: float = 0.0

@dataclass
class Query:
    query_id: str
    query: str
    personalized_features: str = ""
    tagged_memory_features: List[str] = field(default_factory=list)
    sequential_results_raw: Optional[Dict] = field(default_factory=dict)
    working_memory_state_raw: Optional[Dict] = field(default_factory=dict)
    long_term_memory_results_raw: Optional[Dict] = field(default_factory=dict)
    topic_id: str = ""
    turn_id: int = 0
    conversation_id: Optional[str] = None  # Support for MedCorpus_MultiTurn format
    turns: Optional[List] = None  # Support for MedCorpus_MultiTurn format
    target_turns: Optional[List] = None  # Support for MedCorpus_MultiTurn format
    def __post_init__(self):
        self.query_id = str(self.query_id)
        if "_" in self.query_id:
            parts = self.query_id.split("_")
            if len(parts) >= 2 and parts[-1].isdigit():
                self.topic_id = "_".join(parts[:-1])
                try: self.turn_id = int(parts[-1])
                except ValueError: self.turn_id = 0
            else:
                self.topic_id = self.query_id; self.turn_id = 0
        else:
            self.topic_id = self.query_id; self.turn_id = 0

class Config:
    def __init__(self):
        self.dataset_name = "MedCorpus"
        self.base_data_dir = "/workspace/PerMed/data"
        self.results_dir = "./results"
        self.gpu_id = 0; self.device = None; self.llm_device = None
        self._setup_device()
        self.memory_components = ["sequential", "working", "long"]
        self.feature_extractor = "keybert"
        self.keybert_embedder_device = "cpu"
        if self.device and "cuda" in self.device: self.keybert_embedder_device = self.device
        self.keybert_model = "/workspace/PerMed/model/Qwen3-Embedding-0.6B"
        self.embedding_path = "/workspace/PerMed/model/Qwen3-Embedding-0.6B"
        self.memory_type = "vector"
        self.sentence_transformer_model = "/workspace/PerMed/model/Qwen3-Embedding-0.6B"
        self.dataset_type = self._infer_dataset_type()
        self.batch_size = 256
        self.initial_top_k = 200
        self.final_top_k = 10
        
        # --- 修改点: 移除 personalized_text_max_length ---
        self.personalized_text_target_length = 250
        # 不再使用 length_suffix，统一文件命名
        self.length_suffix = ""

        self.max_tagged_features_for_llm = 10
        self.max_features_per_memory_module = 3
        self.max_phrases_per_tag = 5
        self._cognitive_features_detailed_base = "cognitive_features_detailed"
        self._personalized_queries_base = "personalized_queries"
        self._final_results_base = "ranked"
        self._initialize_data_paths()
        
        # Reranker configuration - only Qwen3-Reranker-0.6B
        self.reranker_path = "/workspace/PerMed/model/Qwen3-Reranker-0.6B"
        self.reranker_type = "qwen3"
        self.rerank_input_type = "profile_only"
        self.reranker_max_length = 512
        
        # LLM API configuration - only SiliconFlow with Qwen3 models
        self.llm_api_type = "siliconflow"
        self.siliconflow_api_key = ""
        self.siliconflow_api_url = "https://api.siliconflow.cn/v1/chat/completions"
        # Default SiliconFlow模型改为最新80B思考版，仍可通过命令行覆盖
        self.siliconflow_model = "Qwen/Qwen3-Next-80B-A3B-Thinking"

        self.model_suffix = ""; self._update_model_suffix()
        self.profile_generation_attempts = 1
        self.use_fixed_seed = True; self.llm_seed = 42
        self.local_model_max_tokens = 400
        self.local_model_temperature = 0.4
        self.local_model_top_p = 0.9
        self.local_model_top_k = 20
        self.test_query_limit: Optional[int] = None
        self.two_pass_rerank = False
        self.use_flash_attention = False

    def _initialize_data_paths(self):
        self.queries_path = self._get_data_path("queries.jsonl")
        self.corpus_path = self._get_data_path("corpus.jsonl")
        self.corpus_embeddings_path = self._get_data_path("corpus_embeddings.npy")
        self.retrieved_results_path = self._get_results_path_nosuffix("retrieved.jsonl")
        self.cognitive_features_detailed_path = self._get_results_path_nosuffix(f"{self._cognitive_features_detailed_base}.jsonl")

    def _update_model_suffix(self):
        # Only SiliconFlow API with Qwen3 models
        model_name = self.siliconflow_model.split("/")[-1].lower()
        self.model_suffix = f"_{model_name.replace(':', '-')}"

    @property
    def personalized_queries_path(self):
        base_filename = f"{self._personalized_queries_base}{self.length_suffix}{self.model_suffix}.jsonl"
        return self._get_results_path_with_suffix(base_filename)

    @property
    def final_results_path(self):
        input_type_sfx = f"_{self.rerank_input_type.replace('_','-')}"
        reranker_sfx = f"_{self.reranker_type}"
        k_sfx = f"_top{self.final_top_k}"
        two_pass_sfx = "_2pass" if self.two_pass_rerank else ""
        model_sfx = self.model_suffix if getattr(self, "model_suffix", "") else ""
        base_filename = f"{self._final_results_base}{reranker_sfx}{input_type_sfx}{two_pass_sfx}"
        return self._get_results_path_with_suffix(f"{base_filename}{model_sfx}{k_sfx}.jsonl")

    def _setup_device(self):
        if not torch.cuda.is_available(): self.device = "cpu"; self.llm_device = "cpu"; return
        num_gpus = torch.cuda.device_count()
        if self.gpu_id >= num_gpus: self.gpu_id = 0
        self.device = f"cuda:{self.gpu_id}"; self.llm_device = self.device
        try: logger.info(f"GPU Set: Main device {self.device} ('{torch.cuda.get_device_name(self.gpu_id)}').")
        except Exception as e: logger.error(f"Could not get GPU name for ID {self.gpu_id}: {e}")

    def _infer_dataset_type(self):
        name_lower = self.dataset_name.lower()
        if "coral" in name_lower: return "coral"
        if "medcorpus" in name_lower: return "medcorpus"
        if "litsearch" in name_lower: return "litsearch"
        return "unknown"

    def _get_data_path(self, filename: str) -> str:
        # Map MedCorpus to MedCorpus_MultiTurn directory
        actual_dataset_name = "MedCorpus_MultiTurn" if self.dataset_name == "MedCorpus" else self.dataset_name
        return os.path.join(self.base_data_dir, actual_dataset_name, filename)

    def _get_results_path_with_suffix(self, filename: str) -> str:
        d = os.path.join(self.results_dir, self.dataset_name); os.makedirs(d, exist_ok=True); return os.path.join(d, filename)

    def _get_results_path_nosuffix(self, filename: str) -> str:
        d = os.path.join(self.results_dir, self.dataset_name); os.makedirs(d, exist_ok=True); return os.path.join(d, filename)

    def update(self, args):
        if hasattr(args, 'results_dir') and args.results_dir:
            self.results_dir = args.results_dir
        if hasattr(args, 'dataset_name') and args.dataset_name and self.dataset_name != args.dataset_name:
            self.dataset_name = args.dataset_name; self.dataset_type = self._infer_dataset_type()
        # Re-initialize paths after results_dir or dataset_name changes
        if hasattr(args, 'results_dir') or (hasattr(args, 'dataset_name') and args.dataset_name):
            self._initialize_data_paths()
        if hasattr(args, 'gpu_id') and args.gpu_id is not None and self.gpu_id != args.gpu_id:
            self.gpu_id = args.gpu_id; self._setup_device()
            self.keybert_embedder_device = self.device if self.device and "cuda" in self.device else "cpu"

        # --- 修改点: 移除 personalized_text_max_length 的处理 ---
        if hasattr(args, 'personalized_text_target_length') and args.personalized_text_target_length is not None:
            if self.personalized_text_target_length != args.personalized_text_target_length:
                self.personalized_text_target_length = args.personalized_text_target_length
                # 不再使用 length_suffix
                self.length_suffix = ""
                logger.info(f"Config: Target length set to {self.personalized_text_target_length}.")
        
        for attr in ['siliconflow_api_key', 'siliconflow_model',
                     'reranker_type', 'rerank_input_type', 'reranker_path']:
            if hasattr(args, attr) and getattr(args, attr): 
                setattr(self, attr, getattr(args, attr))
        
        self._update_model_suffix()

        for attr in ['final_top_k', 'test_query_limit', 'two_pass_rerank', 'intermediate_top_k_two_pass', 'use_flash_attention']:
            if hasattr(args, attr) and getattr(args, attr) is not None: setattr(self, attr, getattr(args, attr))

_config = None
def get_config():
    global _config
    if _config is None: _config = Config()
    return _config

def load_queries(config: Config) -> List[Query]:
    queries = []
    if not os.path.exists(config.queries_path): logger.error(f"Queries file not found: {config.queries_path}"); return queries
    with open(config.queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                # Handle MedCorpus_MultiTurn format
                if 'turns' in data and 'conversation_id' in data:
                    for turn in data['turns']:
                        query_id = f"{data['conversation_id']}_{turn['turn_id']}"
                        query = Query(
                            query_id=query_id,
                            query=turn['text'],
                            conversation_id=data['conversation_id'],
                            turns=data['turns'],
                            target_turns=data.get('target_turns')
                        )
                        queries.append(query)
                else:
                    # Handle standard format
                    queries.append(Query(**data))
            except Exception as e: logger.warning(f"Skipping invalid query line: {e}")
    logger.info(f"Loaded {len(queries)} original queries from {config.queries_path}")
    return queries
# ... (load_corpus 函数保持不变) ...
def load_corpus(config: Config) -> Dict[str, Document]:
    documents = {}
    if not os.path.exists(config.corpus_path):
        print(f"ERROR: 语料库文件未找到: {config.corpus_path}")
        return documents
    try:
        with open(config.corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    text_id_str = str(data['text_id'])
                    title = data.get('title','') or ""
                    text_content = data.get('text','') or ""
                    full_paper_content = data.get('full_paper','') or ""
                    text_parts = [title, text_content, full_paper_content]
                    full_text = " ".join(filter(None, text_parts)).strip()
                    documents[text_id_str] = Document(
                        text_id=text_id_str,
                        title=title,
                        text=text_content,
                        full_paper=full_paper_content if full_paper_content else None,
                        full_text=full_text
                    )
                except KeyError as e:
                    print(f"ERROR: 语料库文件 {config.corpus_path} 第 {line_num} 行缺少键 {e}")
                except Exception as e:
                    print(f"ERROR: 解析语料库文件 {config.corpus_path} 第 {line_num} 行时出错: {e}")
    except Exception as e:
        print(f"ERROR: 从 {config.corpus_path} 加载语料库失败: {e}")
    print(f"INFO: 从 {config.corpus_path} 加载了 {len(documents)} 个文档")
    return documents
