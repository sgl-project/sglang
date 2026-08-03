# 第 2 章 第一次启动：把服务跑起来

> 本章是动手章。目标只有一个：**亲手启动一个服务、发一个请求、看懂返回结果**。内部细节看不懂没关系，那是后面的事。

## 2.1 安装

仓库根目录是项目本体，Python 包在 `python/` 下。开发模式安装：

```bash
cd python
pip install -e ".[all]"
```

装完验证一下：

```bash
sglang version
```

能打印版本号就说明环境 OK。

## 2.2 启动服务

```bash
sglang serve --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000
```

如果你是第一次跑，会看到模型权重下载，然后出现类似这样的启动日志：

```text
Loading model ...
KV Cache is allocated. dtype: bfloat16, #tokens: 123456, KV size: 12.3 GB
The server is listening on http://0.0.0.0:30000
```

**请盯着这三行日志看一会儿**：

1. `Loading model`：模型权重在加载，占用一部分显存。
2. `KV Cache is allocated. ... #tokens: ...`：系统把剩余显存切出来给 KV Cache 用。这个数字就是"理论上最多能同时缓存的 token 数"，后面调优会反复提到。
3. `listening on ...`：HTTP 服务已就绪。

> 实验：再启动一次，换 `--mem-fraction-static 0.5`，对比 KV Cache 分配的数字。你会发现显存这块蛋糕，权重和 KV Cache 在抢。

## 2.3 发第一个请求

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
    "temperature": 0.7
  }'
```

返回的 JSON 大概长这样：

```json
{
  "id": "chatcmpl-...",
  "choices": [{"message": {"role": "assistant", "content": "我是一个大语言模型..."}}],
  "usage": {"prompt_tokens": 12, "completion_tokens": 30, "total_tokens": 42}
}
```

注意最后那个 `usage` 字段：`prompt_tokens` 是 prefill 处理的 token 数，`completion_tokens` 是 decode 吐出来的 token 数。**这两个数字直接对应第 1 章的两个阶段。**

## 2.4 用 Python 调用（离线批处理）

服务在线（HTTP）适合单条请求；批量处理（比如一次喂 100 条）更适合用进程内引擎：

```python
import sglang as sgl

engine = sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")

out = engine.generate(
    ["讲个冷笑话", "写一首五言绝句", "1+1=?"],
    sampling_params={"temperature": 0.8, "max_new_tokens": 64},
)
for item in out:
    print(item["text"])

engine.shutdown()
```

`sgl.Engine` 不启动 HTTP 服务，在当前进程里直接做推理。RL 训练框架（verl 等）就是用这个形态做大批量生成的。

## 2.5 看一眼日志，感受"请求是怎么被处理的"

用 debug 级别重启服务，再发一个请求：

```bash
sglang serve --model-path ... --port 30000 --log-level debug
```

你会看到类似：

```text
Recv requests: 1
Prefill batch. #req: 1, #token: 12
Decode batch. #req: 1, #new_token: 1
Decode batch. #req: 1, #new_token: 1
...
```

第一行是一次 prefill（处理 12 个 prompt token），后面每一行是一次 decode（每次产出 1 个 token）。**你现在看到的，就是第 1 章说的 prefill/decode 在真实系统里的样子。**

## 2.6 本仓库的目录，现在可以扫一眼了

跑通之后，仓库结构不再是无字天书：

```text
python/sglang/
├── srt/       # 服务端运行时（你刚启动的东西，核心都在这里）
├── lang/      # 前端（你刚用的 sgl.Engine 在这里）
├── benchmark/ # 压测脚本
├── cli/       # sglang 命令行
docs-mdbook/   # 你正在读的这份文档
benchmark/     # 更多端到端 benchmark
rust/          # 高性能组件（后面进阶再看）
experimental/  # 实验性组件，如 sgl-router
```

现在你只需要记住一点：**入口在 `srt`，前端在 `lang`**。其他目录遇到时再认识。

## 2.7 三种语言，各管一摊

如果你扫一眼仓库，会发现代码不止 Python：有 `rust/`，还有一堆 `.cu`、`.cpp` 和 Triton 文件。它们不是"同一个功能的三种写法"，而是**各管各的一层**：

```text
Python       = 总指挥：接请求、排队、决定 GPU 跑什么
Rust         = 替代 Python 干"高频 CPU 杂活"：HTTP 解析、文本 ↔ token
CUDA/Triton  = GPU 上真正干重活的内核：注意力、矩阵乘、采样
```

一个请求大致是这么穿过三层的：

```text
你的请求 → Rust/Python 把文本变成 token ids
         → Python 调度器决定"这一批跑什么"
         → CUDA/Triton 内核在 GPU 上算出新 token
         → Python/Rust 把 token 拼回文本返回给你
```

现在只需要记住这个分工，不用记细节。第 9 章会讲 GPU 内核，第 16 章会讲 Rust 与 Python 的边界和完整分层。

## 2.8 本章自测

1. 启动日志里 `KV Cache is allocated` 的 `#tokens` 是什么含义？
2. `usage.prompt_tokens` 和 `completion_tokens` 分别对应哪个阶段？
3. `sgl.Engine` 和 HTTP 服务有什么区别？各适合什么场景？
4. 把 `temperature` 从 0.7 改成 1.5 再试一次，输出有什么变化？为什么？

> 跑通了？下一章我们用"餐厅"这个比喻，把"你刚发出去的那个请求"在服务端到底经历了什么讲清楚。
