"""
使用Qwen3 embeddings构建LlamaIndex索引
"""

import json
import numpy as np
from pathlib import Path
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from tqdm import tqdm

# 配置
CORPUS_FILE = "/workspace/PerMed/data/MedCorpus/corpus.jsonl"
EMBEDDINGS_FILE = "/workspace/PerMed/data/MedCorpus/corpus_embeddings_qwen3.npy"
EMBED_MODEL_PATH = "/workspace/PerMed/model/Qwen3-Embedding-0.6B"
INDEX_DIR = "/workspace/PerMed/create/llamaindex_index_qwen3"

print("=" * 80)
print("使用Qwen3 embeddings构建LlamaIndex索引")
print("=" * 80)
print(f"Corpus文件: {CORPUS_FILE}")
print(f"Embeddings文件: {EMBEDDINGS_FILE}")
print(f"索引输出目录: {INDEX_DIR}")
print()

# 创建索引目录
Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

# 加载embedding模型
print("[1] 加载Qwen3 embedding模型...")
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_PATH)
print(f"  ✓ 模型加载完成")
print(f"  模型维度: {embed_model._model.get_sentence_embedding_dimension()}")
print()

# 加载预计算的embeddings
print("[2] 加载预计算的embeddings...")
embeddings = np.load(EMBEDDINGS_FILE)
print(f"  ✓ Embeddings加载完成")
print(f"  形状: {embeddings.shape}")
print()

# 读取corpus并创建TextNode
print("[3] 创建TextNode...")
nodes = []

with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(tqdm(f, total=92703, desc="  处理文档")):
        data = json.loads(line.strip())

        # 拼接title和text
        doc_text = f"{data['title']}\n{data['text']}"

        # 创建TextNode，使用预计算的embedding
        node = TextNode(
            text=doc_text,
            metadata={
                'text_id': data['text_id'],
                'title': data['title']
            },
            id_=data['text_id'],
            embedding=embeddings[idx].tolist()  # 使用预计算的embedding
        )

        nodes.append(node)

print(f"  ✓ 创建了 {len(nodes)} 个TextNode")
print()

# 构建索引
print("[4] 构建VectorStoreIndex...")
print("  (由于使用预计算的embeddings,这个过程会很快)")

index = VectorStoreIndex(
    nodes,
    embed_model=embed_model,
    show_progress=True
)

print(f"  ✓ 索引构建完成")
print()

# 保存索引
print("[5] 保存索引到磁盘...")
index.storage_context.persist(persist_dir=INDEX_DIR)
print(f"  ✓ 索引已保存到: {INDEX_DIR}")
print()

# 保存配置信息
config = {
    'num_documents': len(nodes),
    'embedding_dim': embeddings.shape[1],
    'embed_model': EMBED_MODEL_PATH,
    'corpus_file': CORPUS_FILE,
    'embeddings_file': EMBEDDINGS_FILE
}

with open(f"{INDEX_DIR}/config.json", 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("[6] 检查索引大小...")
index_size = sum(f.stat().st_size for f in Path(INDEX_DIR).rglob('*') if f.is_file())
print(f"  ✓ 索引大小: {index_size / 1024 / 1024:.1f} MB")
print()

print("=" * 80)
print("✅ Qwen3索引构建完成!")
print("=" * 80)
