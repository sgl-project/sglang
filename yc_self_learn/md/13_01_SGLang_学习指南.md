# SGLang 基础学习指南

## 什么是 SGLang？

**SGLang** 是一个用于大语言模型（LLM）和视觉语言模型（VLM）的**快速服务框架**。它通过协同设计后端运行时和前端语言，使模型交互更快、更可控。

### 核心特点

1. **快速后端运行时（Fast Backend Runtime）**
   - RadixAttention：前缀缓存技术，加速推理
   - 零开销 CPU 调度器
   - Prefill-Decode 分离（Prefill-decode disaggregation）
   - 推测解码（Speculative decoding）
   - 连续批处理（Continuous batching）
   - 分页注意力（Paged attention）
   - 多种并行策略：张量/流水线/专家/数据并行
   - 结构化输出
   - 量化支持（FP4/FP8/INT4/AWQ/GPTQ）
   - 多 LoRA 批处理

2. **灵活的前端语言（Flexible Frontend Language）**
   - 链式生成调用
   - 高级提示工程
   - 控制流
   - 多模态输入
   - 并行处理
   - 外部交互

3. **广泛的模型支持**
   - 生成模型：Llama, Qwen, DeepSeek, Kimi, GPT, Gemma, Mistral 等
   - 嵌入模型：e5-mistral, gte, mcdse
   - 奖励模型：Skywork

## 安装 SGLang

### 方法 1：使用 pip（推荐）

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[all]>=0.5.3rc0"
```

### 方法 2：从源码安装

```bash
git clone -b v0.5.3rc0 https://github.com/sgl-project/sglang.git
cd sglang
pip install --upgrade pip
pip install -e "python[all]"
```

### 方法 3：使用 Docker（推荐用于本地开发）

Docker 方式适合在本地运行和开发。详细设置请参考 **[Docker_SGLang_本地开发环境设置.md](./04_Docker_SGLang_本地开发环境设置.md)**。

**快速启动（Windows）：**

```powershell
# 1. 设置 HuggingFace token
$env:HF_TOKEN = "your-token-here"

# 2. 使用提供的脚本启动
cd docker_scripts
.\run_sglang.ps1 -Model "qwen/qwen2.5-0.5b-instruct"
```

**或使用 Docker Compose：**

```powershell
docker-compose up -d
```

**基本 Docker 命令：**

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<your-token>" \
    --ipc=host \
    lmsysorg/sglang:latest \
    python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

## 快速开始

### 1. 启动服务器

```bash
python3 -m sglang.launch_server \
    --model-path qwen/qwen2.5-0.5b-instruct \
    --host 0.0.0.0 \
    --port 30000
```

### 2. 发送请求

#### 方法 A：使用 cURL

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": "法国的首都是什么？"}]
  }'
```

#### 方法 B：使用 Python requests

```python
import requests

url = "http://localhost:30000/v1/chat/completions"

data = {
    "model": "qwen/qwen2.5-0.5b-instruct",
    "messages": [{"role": "user", "content": "法国的首都是什么？"}],
}

response = requests.post(url, json=data)
print(response.json())
```

#### 方法 C：使用 OpenAI Python 客户端（推荐）

```python
import openai

client = openai.Client(
    base_url="http://127.0.0.1:30000/v1",
    api_key="None"  # SGLang 不需要 API key
)

response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[
        {"role": "user", "content": "列出3个国家和它们的首都。"},
    ],
    temperature=0,
    max_tokens=64,
)

print(response.choices[0].message.content)
```

#### 方法 D：流式输出（Streaming）

```python
import openai

client = openai.Client(base_url="http://127.0.0.1:30000/v1", api_key="None")

response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[
        {"role": "user", "content": "列出3个国家和它们的首都。"},
    ],
    temperature=0,
    max_tokens=64,
    stream=True,  # 启用流式输出
)

# 处理流式输出
for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

## 核心概念

### 1. 后端运行时（Backend Runtime）

SGLang 的后端运行时（SRT - SGLang Runtime）负责：
- **模型加载和推理**：高效加载模型到 GPU 内存
- **批处理调度**：智能管理多个请求的批处理
- **内存管理**：使用 RadixAttention 进行前缀缓存，减少重复计算
- **并行处理**：支持多种并行策略以加速推理

### 2. 前端语言（Frontend Language）

SGLang 的前端语言提供了更高级的编程接口，允许你：
- 定义复杂的生成流程
- 使用控制流（if/else, for 循环等）
- 处理多模态输入（文本、图像等）
- 实现并行生成

### 3. API 兼容性

SGLang 提供 **OpenAI 兼容的 API**，这意味着：
- 你可以直接使用 OpenAI 的 Python 客户端
- 现有的 OpenAI 代码可以无缝迁移
- 支持标准的 Chat Completions 和 Completions API

## 常用参数说明

### Chat Completions 参数

```python
response = client.chat.completions.create(
    model="qwen/qwen2.5-0.5b-instruct",
    messages=[
        {"role": "system", "content": "你是一个知识渊博的历史学家。"},
        {"role": "user", "content": "告诉我关于古罗马的事情"},
    ],
    temperature=0.7,      # 控制随机性（0-2），越高越随机
    max_tokens=128,      # 最大生成 token 数
    top_p=0.95,          # 核采样参数
    presence_penalty=0.2,  # 存在惩罚，避免重复话题
    frequency_penalty=0.2, # 频率惩罚，避免重复词汇
    n=1,                 # 生成多少个响应
    seed=42,             # 随机种子，用于可复现性
)
```

### 原生 Generate API

除了 OpenAI 兼容的 API，SGLang 还提供原生的 `/generate` 端点：

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "法国的首都是",
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
        },
    },
)

print(response.json())
```

## 项目结构

```
sglang/
├── python/sglang/          # Python 包主目录
│   ├── lang/               # 前端语言实现
│   ├── srt/                # 后端运行时（SRT）
│   ├── api.py              # 公共 API
│   └── launch_server.py    # 服务器启动入口
├── docs/                   # 文档
│   ├── basic_usage/        # 基础使用教程
│   ├── advanced_features/  # 高级特性
│   └── get_started/        # 入门指南
├── examples/               # 示例代码
└── benchmark/             # 性能基准测试
```

## 学习路径建议

### 第一阶段：基础使用
1. ✅ 安装 SGLang
2. ✅ 启动服务器
3. ✅ 使用 OpenAI 客户端发送请求
4. ✅ 了解基本参数（temperature, max_tokens 等）

### 第二阶段：进阶功能
1. 学习流式输出
2. 了解结构化输出（JSON Schema, Regex, EBNF）
3. 探索前端语言（Frontend Language）
4. 学习多模态输入（图像、视频）

### 第三阶段：高级特性
1. 性能优化（RadixAttention, 批处理）
2. 分布式部署
3. 量化（FP8, INT4 等）
4. 自定义模型支持

## 有用的资源

- **官方文档**：https://docs.sglang.ai/
- **GitHub 仓库**：https://github.com/sgl-project/sglang
- **博客文章**：https://lmsys.org/blog/
- **Slack 社区**：https://slack.sglang.ai/

## 常见问题

### Q: 如何选择模型？
A: SGLang 支持 Hugging Face 上的大多数模型。推荐从小模型开始（如 `qwen/qwen2.5-0.5b-instruct`）进行测试。

### Q: 需要多少 GPU 内存？
A: 取决于模型大小。7B 模型通常需要约 14GB，13B 需要约 26GB。可以使用量化来减少内存需求。

### Q: 如何提高性能？
A: 
- 使用 RadixAttention（自动启用）进行前缀缓存
- 调整批处理大小
- 使用量化（FP8/INT4）
- 使用多 GPU 并行

### Q: 支持哪些推理模型？
A: SGLang 支持多种推理模型，如 DeepSeek-R1、Qwen3-Thinking 等。需要在启动服务器时指定 `--reasoning-parser` 参数。

## 下一步

1. 尝试运行示例代码
2. 阅读 `docs/basic_usage/` 目录下的教程
3. 查看 `examples/` 目录中的示例
4. 加入 Slack 社区提问

祝你学习愉快！🚀

