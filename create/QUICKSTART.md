# 🚀 MedCorpus 多轮对话数据集生成流程

## 📊 当前状态

✅ **已完成**: 抽样 800 组对话,分配轮次,重新编号
- 3 轮: 200 组 (25%)
- 4 轮: 240 组 (30%)
- 5 轮: 280 组 (35%)
- 6 轮: 80 组 (10%)
- Topic IDs: topic_001 ~ topic_800 (顺序编号)

📂 **输出文件**: `sampled_800_topics.jsonl` (800 行)

✅ **API 测试通过**: Claude 4.5 Sonnet (via https://b4u.qzz.io)
- 测试结果: 3 组对话生成成功 (topic_441-443)
- 成功率: 100% (Turn4=3, Turn5=3, Turn6=0)

---

## 📋 数据格式

### sampled_800_topics.jsonl

每行一个对话,格式:

```json
{
  "topic_id": "topic_001",
  "target_turns": 6,
  "turns": [
    {"turn_id": 1, "text": "Fear of cancer recurrence in breast cancer survivors..."},
    {"turn_id": 2, "text": "Gender differences in this fear of cancer recurrence..."},
    {"turn_id": 3, "text": "Adapting the Lee-Jones model for FCR in male survivors..."},
    {"turn_id": 4, "text": "[TO_GENERATE_TURN_4]", "question_type": "unknown"},
    {"turn_id": 5, "text": "[TO_GENERATE_TURN_5]", "question_type": "unknown"},
    {"turn_id": 6, "text": "[TO_GENERATE_TURN_6]", "question_type": "unknown"}
  ]
}
```

**说明**:
- Turn 1-3: 已有高质量问题
- Turn 4-6: 标记为 `[TO_GENERATE_TURN_X]`,待 LLM 生成

---

## 🔧 下一步: LLM 生成 Turn 4-6

### ✅ API 配置 (已测试可用)

**当前 API 配置**:
- **API Key**: `REDACTED_SILICONFLOW_API_KEY_4`
- **Base URL**: `https://b4u.qzz.io/v1`
- **Model**: `claude-4.5-sonnet` (注意不是 `claude-sonnet-4.5`)

配置已更新在:
- `batch_generate_turns.py` (第 39-41 行)
- `generate_800_multiturn_v2.py` (第 219-230 行)

### 🧪 测试结果

已成功测试 3 组对话 (topic_441-443):

```bash
python batch_generate_turns.py --start 440 --end 443 --output test_3conversations.jsonl
```

**输出示例** (topic_441, Turn 4):
```
What are the standard protocols for evaluating nutrient bioavailability from these
optimized substrates—specifically, are researchers using in vitro digestion models,
animal bioassays, or human intervention trials? I'm particularly interested in
understanding how bioactive compound stability is assessed throughout cultivation...
```

**质量特点**:
✅ 正确使用指代 ("these", "given", "building on")
✅ 问题类型准确 (Turn 4=methodology, Turn 5=clinical/comparison)
✅ 长度适中 (100-150 tokens)
✅ 学术语气自然

### 📦 批量生成方案

#### 方案 A: 一次性生成 (顺序执行,约 2-3 小时)

```bash
python batch_generate_turns.py \
  --start 0 \
  --end 800 \
  --output generated_800.jsonl \
  --delay 0.5
```

#### 方案 B: 分批并行生成 (推荐,约 30-45 分钟)

```bash
# 同时启动 4 个进程
python batch_generate_turns.py --start 0 --end 200 --output batch_001_200.jsonl &
python batch_generate_turns.py --start 200 --end 400 --output batch_201_400.jsonl &
python batch_generate_turns.py --start 400 --end 600 --output batch_401_600.jsonl &
python batch_generate_turns.py --start 600 --end 800 --output batch_601_800.jsonl &

# 等待全部完成
wait

# 合并结果
cat batch_*.jsonl > generated_800.jsonl

# 验证行数
wc -l generated_800.jsonl  # 应该是 800
```

**参数说明**:
- `--start`: 起始索引 (0-based)
- `--end`: 结束索引 (不含)
- `--output`: 输出文件名
- `--delay`: 每次 API 调用后延迟(秒),默认 0.5

**预期输出**: Turn 4-6 被替换为 LLM 生成的问题,带 `question_type` 标签

### Step 3: 标注相关性 (rel=0/1/2)

对每个 Turn 的 top-10 文档标注相关性:

```bash
python label_relevance.py --input generated_800.jsonl --output final_dataset/
```

**输出**:
- `queries.jsonl`: 800 组完整对话
- `labels_turn.jsonl`: ~35,600 条标签 (800×平均4.45轮×10文档)
- `metadata.json`: 统计信息

---

## 📁 文件清单

```
PerMed/create/
├── sample_800_topics.py              # 抽样脚本 (已执行)
├── sampled_800_topics.jsonl          # 800 组对话 (topic_001-800)
├── batch_generate_turns.py           # 批量生成脚本 (推荐使用)
├── generate_800_multiturn_v2.py      # 完整生成脚本 (包含标注)
├── llm_utils_v2.py                   # LLM 工具类 (支持 OpenAI/Gemini)
├── prompts/
│   └── llm_turn_generation_typed.py  # Prompt 模板 (带问题类型)
├── test_3conversations.jsonl         # 测试输出 (3 组)
└── QUICKSTART.md                     # 本文档

已归档 (archive/):
├── pilot_prepare_10.py               # 旧的规则生成脚本
├── generate_800_multiturn.py         # 旧版本
└── llm_utils.py                      # 旧版本
```

---

## 🎯 问题类型分布 (设计)

| 轮次 | 问题类型 | 占比 | 说明 |
|------|----------|------|------|
| Turn 1-3 | definition | 100% | 基础概念,原始问题 |
| Turn 4 | methodology | 100% | 实验方法,技术细节 |
| Turn 5 | clinical (70%) <br> comparison (30%) | 混合 | 临床应用或对比分析 |
| Turn 6 | comparison (60%) <br> definition (40%) | 混合 | 批判性评价或概念深挖 |

**实现方式** (batch_generate_turns.py):
```python
if target_turn == 4:
    expected_type = "methodology"
elif target_turn == 5:
    q_type = random.choice(["clinical"] * 7 + ["comparison"] * 3)
elif target_turn == 6:
    q_type = random.choice(["comparison"] * 6 + ["definition"] * 4)
```

---

## 📊 数据集统计 (预期)

生成完成后:

| 项目 | 数量 |
|------|------|
| 对话总数 | 800 |
| 问题总数 | 3,560 (200×3 + 240×4 + 280×5 + 80×6) |
| 标签总数 | ~35,600 (每轮 10 个文档) |
| 唯一文档数 | ~5,000 |
| definition 问题 | ~2,464 (69%) |
| methodology 问题 | 600 (17%) |
| clinical 问题 | 280 (8%) |
| comparison 问题 | 216 (6%) |

---

## 🆘 故障排查

### 问题 1: API 连接失败

检查 API 状态:
```bash
curl -X POST https://b4u.qzz.io/v1/chat/completions \
  -H "Authorization: Bearer REDACTED_SILICONFLOW_API_KEY_4" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-4.5-sonnet",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 10
  }'
```

**常见错误**:
- `503 - 无可用渠道`: 模型名错误,使用 `claude-4.5-sonnet` 而不是 `claude-sonnet-4.5`
- `429 - 速率限制`: 增加 `--delay` 参数 (如 `--delay 1.0`)

### 问题 2: 生成速度慢

原因: API rate limit (每分钟请求数限制)

解决:
- **方案 B (分批并行)**: 同时运行 4 个进程
- 增加延迟: `--delay 1.0`
- 分散时间段: 不同时间段运行不同批次

### 问题 3: Embedding 模型下载失败

使用本地模型 (已配置):
```python
model_path = "/workspace/PerMed/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(model_path)
```

### 问题 4: 生成失败 (GENERATION_FAILED_TURN_X)

原因: API 超时或返回空内容

解决:
- 检查失败条目: `grep FAILED generated_800.jsonl`
- 重新生成失败部分: 提取失败的 topic_id,单独运行

---

## ✅ 检查清单

生成前确认:

- [x] API 可用 (已测试 3 组对话)
- [x] `sampled_800_topics.jsonl` 存在 (800 行)
- [x] Topic IDs 已重新编号 (topic_001-800)
- [x] Embedding 模型路径正确 (`/workspace/PerMed/all-MiniLM-L6-v2`)
- [ ] 输出目录有写权限
- [ ] 磁盘空间充足 (至少 100MB)

生成后验证:

- [ ] 检查行数: `wc -l generated_800.jsonl` (应为 800)
- [ ] 检查失败率: `grep -c FAILED generated_800.jsonl` (应 < 1%)
- [ ] 抽查 2-3 个 6 轮对话质量
- [ ] 验证问题类型分布是否符合预期
- [ ] 检查指代使用 (these, given, building on, etc.)

---

## 🚀 快速开始

```bash
# 1. 确认文件存在
ls -lh sampled_800_topics.jsonl

# 2. 测试 API (可选,已测试过)
python batch_generate_turns.py --start 0 --end 3 --output test.jsonl

# 3. 开始批量生成 (推荐方案 B)
python batch_generate_turns.py --start 0 --end 200 --output batch_001_200.jsonl &
python batch_generate_turns.py --start 200 --end 400 --output batch_201_400.jsonl &
python batch_generate_turns.py --start 400 --end 600 --output batch_401_600.jsonl &
python batch_generate_turns.py --start 600 --end 800 --output batch_601_800.jsonl &

# 4. 监控进度 (另一个终端)
watch -n 5 'tail -n 10 batch_*.jsonl | grep topic_id'

# 5. 完成后合并
wait
cat batch_*.jsonl > generated_800.jsonl
wc -l generated_800.jsonl

# 6. 检查质量
grep -c FAILED generated_800.jsonl
head -n 3 generated_800.jsonl | jq '.'
```

---

**需要帮助?** 联系: shuaiyuzhang275@gmail.com
