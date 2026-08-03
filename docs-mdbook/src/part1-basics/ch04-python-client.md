# 第 4 章 用 Python 调用服务：Engine、Runtime 和编程原语

> 本章只讲"怎么用"。看完你应该能不看文档写出批量生成、流式生成和简单的结构化输出实验。

## 4.1 三个对象怎么选

Python 前端有三个容易混淆的对象，记住一句话就能选对：

| 对象 | 是什么 | 什么时候用 |
| --- | --- | --- |
| `sgl.Engine` | 在当前进程里直接做推理，不启动 HTTP | 离线批处理、RL rollout、想少一层网络开销时 |
| `sgl.Runtime` | 帮你把 HTTP 服务起在子进程里，并返回一个客户端句柄 | 想用 Python 一条龙"启动 + 调用"时 |
| `RuntimeEndpoint` | 纯客户端，连接一个**已经跑起来**的 HTTP 服务 | 服务已经部署好，你只想发请求时 |

```python
import sglang as sgl

# Engine：进程内推理
engine = sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
engine.generate("你好")
engine.shutdown()

# Runtime：Python 里启动 + 调用
rt = sgl.Runtime(model_path="meta-llama/Llama-3.1-8B-Instruct", port=30000)
rt.generate("你好")
rt.shutdown()

# RuntimeEndpoint：连已启动的服务（在另一个终端跑 sglang serve）
from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint
client = RuntimeEndpoint("http://localhost:30000")
```

注意：`Engine` 和 `Runtime` 的参数和启动服务的命令行参数（`--model-path`、`--tp-size` 等）完全一致，只是把 `--` 换成下划线（`model_path`、`tp_size`）。

## 4.2 批量生成

`Engine.generate` 接受列表，一次处理一批：

```python
out = engine.generate(
    ["讲个冷笑话", "写一首诗", "1+1=?"],
    sampling_params={"temperature": 0.8, "max_new_tokens": 64},
)
for item in out:
    print(item["text"])
```

关键点：**这一批请求在 GPU 上是一起算的**（共享一次前向），所以数量从 1 涨到 10，耗时不会涨 10 倍。这就是批处理的价值。你可以把列表从 1 个加到 32 个，观察耗时变化，直观感受"批处理摊薄成本"。

## 4.3 流式生成

```python
engine = sgl.Engine(model_path="...")
for chunk in engine.generate("给我讲一个很长的故事", stream=True):
    print(chunk, end="", flush=True)
engine.shutdown()
```

流式模式下，你会看到文字一个 chunk 一个 chunk 地出现——第一个 chunk 出现的时间就是 **TTFT**，之后每个 chunk 的间隔就是 **TPOT** 的感觉。

## 4.4 采样参数：让模型"性格"可调

每次生成都受 `sampling_params` 控制，最常用的几个：

| 参数 | 作用 | 直观理解 |
| --- | --- | --- |
| `temperature` | 分布"尖锐"程度 | 低 → 保守稳定；高 → 天马行空 |
| `top_p` | 只在概率和达到 p 的 token 里选 | 剪掉长尾 |
| `max_new_tokens` | 最多生成多少 token | 预算 |
| `stop` / `stop_token_ids` | 遇到什么就停 | 句号、换行等 |
| `regex` / `json_schema` | 强制输出格式 | 见 4.6 |

> 实验：同一个问题，分别用 `temperature=0.1` 和 `temperature=1.5` 各生成 5 次，对比"稳定程度"。你会在第 11 章看到它在代码里是怎么实现的。

## 4.5 编程原语：gen / select（什么时候值得用）

除了"喂文本、收文本"，前端还提供一种结构化写法：把一个生成任务描述成一段程序。用 `@sgl.function` 装饰函数，里面用 `sgl.gen` / `sgl.select` 标记"这里要生成"：

```python
@sgl.function
def answer_phone(s, question):
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_new_tokens=64))

rt = sgl.Runtime(model_path="...", port=30000)
ret = answer_phone.run(question="1+1=?", backend=rt)
print(ret["answer"])
rt.shutdown()
```

这样写的好处有两个，等用到时自然体会：

1. **批量跑同一个程序**：`run_batch` 一次跑 100 个问题，每个问题都执行同一套"台词"；
2. **约束输出**：`sgl.gen("answer", regex=r"\d{11}")` 强制手机号格式；`sgl.select` 从选项里选，比如让模型做选择题。

如果只是"文本进、文本出"，直接用 `engine.generate` 就够，不用上这套。

## 4.6 结构化输出（先用起来）

让模型输出 JSON：

```python
out = engine.generate(
    "输出一个用户的姓名和年龄，用 JSON 格式",
    sampling_params={"json_schema": {"type": "object",
                                     "properties": {"name": {"type": "string"},
                                                    "age": {"type": "integer"}}}},
)
print(out[0]["text"])   # 一定是合法 JSON，且字段符合 schema
```

秘诀是服务端在**采样的每一步都只允许"合法"的 token**，而不是生成完再修补。原理在第 11 章。

## 4.7 本章自测

1. `Engine`、`Runtime`、`RuntimeEndpoint` 三者的区别是什么？各举一个使用场景。
2. 批量生成的耗时为什么不会随请求数线性增长？
3. `temperature` 影响的是什么？`max_new_tokens` 呢？
4. `sgl.select` 和普通生成的区别是什么？什么场景会用到？

> 会调用了，下一章看看"不走 Python 前端、直接用 HTTP 协议"怎么玩——包括流式、JSON Schema、工具调用和多模态。
