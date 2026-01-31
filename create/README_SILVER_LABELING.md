# Silver标准标注 - LlamaIndex + Gemini方案

## 📋 总体流程

```
Step 1: 构建索引 (build_llamaindex.py)
  ├─ 读取 corpus.jsonl (92,703篇论文)
  ├─ 使用 BGE-M3 生成 embeddings
  └─ 保存 LlamaIndex 向量索引

Step 2: 银标准标注 (label_with_llamaindex.py)
  ├─ 加载索引和800对话数据
  ├─ 对每个turn使用LlamaIndex检索top-10
  ├─ 用Gemini判断relevance (0/1/2)
  └─ 保存标注结果

Step 3: 质量检查 (check_labels.py)
  └─ 统计分析标注结果
```

---

## 🔧 环境配置

### 1. 安装LlamaIndex

```bash
# 安装LlamaIndex核心包
pip install llama-index-core

# 安装HuggingFace embeddings支持
pip install llama-index-embeddings-huggingface

# 如果需要,安装其他组件
pip install llama-index-llms-gemini  # Gemini集成(可选)
```

### 2. 依赖检查

```bash
# 已有的包 (应该已安装)
- sentence-transformers  # BGE-M3需要
- transformers
- torch
- tqdm
```

### 3. API配置

```bash
# Gemini API Key (标注时需要)
export GOOGLE_API_KEY="your-gemini-api-key"
```

---

## 📂 文件结构

```
/workspace/PerMed/
├── data/
│   ├── MedCorpus/
│   │   └── corpus.jsonl              # 92,703篇论文
│   ├── selected_800_topics.json      # 800个选中的topic
│   └── final_800_topics.jsonl        # 800对话 (3-6轮)
├── create/
│   ├── build_llamaindex.py           # [新] 构建索引
│   ├── label_with_llamaindex.py      # [新] 银标准标注
│   ├── check_labels.py               # [新] 质量检查
│   └── llamaindex_index/             # [新] 索引存储目录
│       ├── docstore.json
│       ├── index_store.json
│       └── vector_store.json
└── results/
    └── silver_labels/                # [新] 标注结果
        ├── labels_full.jsonl         # 完整标注结果
        ├── labels_top10.jsonl        # Top-10标注
        └── statistics.json           # 统计信息
```

---

## ⚙️ 配置说明

### corpus.jsonl 格式
```json
{
  "text_id": "permed-0001",
  "title": "可充气球囊压迫辅助下...",
  "text": "目的探讨可充气球囊压迫辅助下..."
}
```

### 索引配置
- **嵌入模型**: BAAI/bge-m3 (SOTA开源模型)
- **设备**: CUDA (GPU加速)
- **批量大小**: 32 (embedding batch)
- **分块大小**: 512 tokens
- **文档数**: 92,703篇

### 检索配置
- **Top-K**: 10 (每个问题检索top-10文档)
- **相似度**: Cosine similarity
- **模式**: condense_plus_context (自动处理对话历史)

### 标注配置
- **LLM**: Gemini-2.5-Pro
- **Temperature**: 0.0 (确保一致性)
- **Relevance**: 0/1/2 (Not/Partial/Highly Relevant)

---

## 🚀 运行步骤

### Step 1: 构建索引 (约30-60分钟)

```bash
cd /workspace/PerMed/create

# 运行索引构建
python build_llamaindex.py

# 输出:
# [步骤1] 加载语料库... 92,703篇论文
# [步骤2] 初始化BGE-M3嵌入模型...
# [步骤3] 构建向量索引... (30-60分钟)
# [步骤4] 保存索引到磁盘...
# [步骤5] 验证索引...
# ✅ 索引构建完成!
```

**预期输出**:
- `llamaindex_index/docstore.json` - 文档存储
- `llamaindex_index/index_store.json` - 索引元数据
- `llamaindex_index/vector_store.json` - 向量数据

**资源需求**:
- GPU显存: ~8-12GB (BGE-M3模型)
- 磁盘空间: ~5-10GB (索引文件)
- 时间: 30-60分钟 (取决于GPU)

### Step 2: 银标准标注 (待创建)

```bash
# 运行标注脚本 (稍后创建)
python label_with_llamaindex.py

# 输出:
# [步骤1] 加载索引...
# [步骤2] 加载800对话...
# [步骤3] 标注进度: [====>] 800/800 对话
# [步骤4] 保存标注结果...
# ✅ 标注完成! 36,000 labels
```

### Step 3: 质量检查 (待创建)

```bash
# 检查标注质量
python check_labels.py

# 输出统计信息
```

---

## 🔍 技术细节

### LlamaIndex vs 直接使用BGE-M3

**为什么用LlamaIndex?**

1. ✅ **多轮对话管理**: ChatMemoryBuffer自动处理历史
2. ✅ **照应解析**: condense_plus_context模式自动解决"它"、"这个"
3. ✅ **简化代码**: 几行代码实现复杂RAG
4. ✅ **可复现性**: 40K+ stars,文档齐全

**LlamaIndex底层使用BGE-M3**:
```python
# LlamaIndex配置使用BGE-M3
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

# LlamaIndex会:
# 1. 用BGE-M3对文档编码
# 2. 存储向量到VectorStore
# 3. 查询时用BGE-M3对query编码
# 4. 计算cosine相似度
# 5. 返回top-k结果
```

所以**本质上还是用BGE-M3检索,只是LlamaIndex提供了多轮对话的封装**。

---

## 📊 预期结果

### 索引统计
- **文档数**: 92,703
- **平均长度**: ~500 tokens/doc
- **总向量数**: ~92,703 (每文档1个向量)
- **索引大小**: ~5-10GB

### 标注统计 (预期)
- **对话数**: 800
- **问题数**: ~3,600 (平均4.5 turns/对话)
- **标注数**: ~36,000 (每问题top-10文档)
- **时间**: ~10小时
- **成本**: ~$100 (Gemini API)

---

## ⚠️ 注意事项

### 1. GPU内存

BGE-M3需要~8GB GPU显存:
```bash
# 检查GPU
nvidia-smi

# 如果显存不足,可以减小batch_size
embed_batch_size=16  # 默认32,可降低
```

### 2. 索引时间

92,703篇文档需要30-60分钟:
```bash
# 建议使用nohup后台运行
nohup python build_llamaindex.py > build_index.log 2>&1 &

# 监控进度
tail -f build_index.log
```

### 3. 索引持久化

**重要**: 索引构建完成后会保存到磁盘,之后可以直接加载,不需要重新构建:

```python
# 后续使用时,直接加载即可
from llama_index.core import load_index_from_storage, StorageContext

storage_context = StorageContext.from_defaults(
    persist_dir="/workspace/PerMed/create/llamaindex_index"
)
index = load_index_from_storage(storage_context)
```

---

## 🎯 下一步

1. ✅ 运行 `build_llamaindex.py` 构建索引
2. ⏸️ 等待索引构建完成 (30-60分钟)
3. ⏸️ 创建 `label_with_llamaindex.py` 标注脚本
4. ⏸️ 运行银标准标注
5. ⏸️ 质量检查和统计

---

## 📚 参考资料

- LlamaIndex文档: https://docs.llamaindex.ai/
- BGE-M3: https://huggingface.co/BAAI/bge-m3
- ChatEngine: https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/
- Gemini API: https://ai.google.dev/
