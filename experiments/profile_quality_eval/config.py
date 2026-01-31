"""
配置文件：个性化特征质量评估实验
"""
import os
from pathlib import Path

# ============================================
# 路径配置
# ============================================

PROJECT_ROOT = Path("/mnt/data/zsy-data/PerMed")
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "profile_quality_eval"

# 输入数据
PERSONALIZED_QUERIES_FILE = PROJECT_ROOT / "results" / "MedCorpus" / "personalized_queries_qwen3-14b.jsonl"
QUERIES_FILE = PROJECT_ROOT / "baselines" / "data" / "MedCorpus" / "queries.jsonl"

# 输出文件
SAMPLED_QUERIES_FILE = EXPERIMENTS_DIR / "samples_100.jsonl"
EVALUATION_RESULTS_FILE = EXPERIMENTS_DIR / "evaluation_results.jsonl"
ANALYSIS_REPORT_FILE = EXPERIMENTS_DIR / "analysis_report.md"

# 缓存目录
CACHE_DIR = EXPERIMENTS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================
# 采样配置
# ============================================

# 采样数量
TOTAL_SAMPLES = 100  # LitSearch 40 + MedCorpus 60

# 随机种子（确保可复现）
RANDOM_SEED = 42

# ============================================
# LLM 模型配置
# ============================================

# API Keys（从环境变量读取）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROK_API_KEY = os.getenv("GROK_API_KEY", "")

# 模型名称
MODELS = {
    "gpt-4o": {
        "api_type": "openai",
        "model_name": "gpt-4o-2024-11-20",
        "api_key": OPENAI_API_KEY,
        "temperature": 0.0,
        "max_tokens": 500
    },
    "gpt-5": {
        "api_type": "openai",
        "model_name": "gpt-5",
        "api_key": OPENAI_API_KEY,
        "api_url": "https://code.newcli.com/codex/v1",
        "temperature": 0.0,
        "max_tokens": 500
    },
    "claude-haiku-4.5": {
        "api_type": "anthropic",
        "model_name": "claude-haiku-4-5-20251001",
        "api_key": "-----",
        "api_url": "https://code.newcli.com/claude/aws",
        "temperature": 0.0,
        "max_tokens": 500
    },
    "claude-sonnet-4.5": {
        "api_type": "openai",  # 使用 OpenAI 兼容 API
        "model_name": "claude-sonnet-4-5-20250929",
        "api_key": "-----",
        "api_url": "https://529961.com/v1",
        "temperature": 0.0,
        "max_tokens": 500
    },
    "gemini-2.5-pro": {
        "api_type": "gemini",
        "model_name": "gemini-2.5-pro-latest",
        "api_key": GEMINI_API_KEY,
        "temperature": 0.0,
        "max_tokens": 500
    },
    "gemini-2.5-pro-jiubanai": {
        "api_type": "openai",  # 使用 OpenAI 兼容 API
        "model_name": "gemini-2.5-flash",  # 注意：该API的pro版本返回空响应，使用flash版本
        "api_key": "-----",
        "api_url": "https://gy.jiubanai.com/v1",
        "temperature": 0.0,
        "max_tokens": 2048  # 增加到2048以确保完整响应
    },
    "gemini-2.5-flash-jiubanai": {
        "api_type": "openai",  # 使用 OpenAI 兼容 API
        "model_name": "gemini-2.5-flash",
        "api_key": "-----",
        "api_url": "https://gy.jiubanai.com/v1",
        "temperature": 0.0,
        "max_tokens": 2048  # 增加到2048以确保完整响应
    },
    "gemini-2.5-pro-hybgzs": {
        "api_type": "openai",  # 使用 OpenAI 兼容 API
        "model_name": "hyb-Optimal/GCP/gemini-2.5-pro",
        "api_key": "-----",
        "api_url": "https://ai.hybgzs.com/v1",
        "temperature": 0.0,
        "max_tokens": 2048
    },
    "gpt-4o-mini-hybgzs": {
        "api_type": "openai",  # 使用 OpenAI 兼容 API
        "model_name": "openai/gpt-4o-mini",
        "api_key": "-----",
        "api_url": "https://ai.hybgzs.com/v1",
        "temperature": 0.0,
        "max_tokens": 2048
    },
    "grok-4.1-thinking": {
        "api_type": "openai",  # OpenAI 兼容协议
        "model_name": "grok-4.1-thinking",
        "api_key": "-----",
        "api_url": "https://ai.hybgzs.com/v1",
        "temperature": 0.0,
        "max_tokens": 2048
    }
}

# ============================================
# 评估配置
# ============================================

# 评估维度
EVALUATION_DIMENSIONS = [
    "relevance",
    "accuracy",
    "informativeness",
    "coherence"
]

# 评分范围
SCORE_RANGE = (1, 5)

# 重试配置
MAX_RETRIES = 3
RETRY_DELAY = 2  # 秒

# 批处理配置
BATCH_SIZE = 10  # 每批处理的查询数
SAVE_INTERVAL = 10  # 每处理多少个查询保存一次

# ============================================
# 人工标注配置
# ============================================

# 人工标注样本数（可选）
HUMAN_ANNOTATION_SAMPLES = 50  # 从100个中抽取50个人工标注

# 标注者数量
NUM_ANNOTATORS = 2

# ============================================
# 调试配置
# ============================================

# Dry-run 模式（不调用真实 API）
DRY_RUN = False

# Verbose 模式（打印详细日志）
VERBOSE = True

# ============================================
# 验证配置
# ============================================

def validate_config():
    """验证配置是否有效"""
    errors = []
    
    # 检查 API keys
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY 未设置")
    if not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY 未设置")
    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY 未设置")
    
    # 检查样本文件
    if not SAMPLED_QUERIES_FILE.exists():
        errors.append(f"样本文件不存在: {SAMPLED_QUERIES_FILE}")
    
    if errors:
        print("⚠️ 配置错误:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✅ 配置验证通过!")
    return True

if __name__ == "__main__":
    validate_config()
