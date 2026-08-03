# 第 4 章 前端 sglang.lang：编程范式与 Backend 抽象

## 4.1 前端是什么

`python/sglang/lang/` 是 SGLang 的可编程前端。顶层公共 API 在 `python/sglang/__init__.py`（`__all__` 列出全部导出），具体实现分散在 `lang/api.py`、`lang/ir.py`、`lang/interpreter.py`。前端提供两类东西：

1. **高层对象**：`LLM`、`Engine`、`Runtime`、`set_default_backend()` 等，直接对应推理后端。
2. **编程原语**：`gen()`、`gen_int()`、`select()`、`chat()`、`system()/user()/assistant()` 等，用于描述"生成一个满足条件的文本"。

入口在 `python/sglang/__init__.py`（顶层导出）与 `python/sglang/lang/api.py`（原语实现）。顶层命名空间很干净：

```python
from sglang import (
    Runtime, Engine,
    gen, select, image, video,
    system, user, assistant,
    set_default_backend, flush_cache, get_server_info,
)
```

## 4.2 两个核心对象：Runtime / Engine

`lang/api.py` 里 `Runtime(...)` 与 `Engine(...)` 都是工厂函数（懒加载具体实现）：

- **`Runtime(...)`**：在独立进程中启动完整的 HTTP 服务，返回一个可调用的客户端句柄。适合"先起服务，再反复调用"的场景。
- **`Engine(...)`**：在**当前进程内**启动推理引擎，不监听 HTTP 端口，适合离线批处理、RL rollout 这类高吞吐场景。

`Runtime` 的实现类在 `lang/backend/runtime_endpoint.py`（内部把 ServerArgs 解析好后用 spawn 子进程拉起 `launch_server`，再轮询 `/health_generate` 等服务就绪）；`Engine` 的实现类在 `python/sglang/srt/entrypoints/engine.py`（第 192 行），构造参数与 `ServerArgs` 完全一致。

```python
import sglang as sgl

# 进程内引擎：批量生成，适合离线任务
engine = sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
out = engine.generate(["讲个笑话", "写一首诗"])
print(out[0]["text"])
engine.shutdown()

# HTTP 服务包装：启动服务并返回客户端句柄
rt = sgl.Runtime(model_path="meta-llama/Llama-3.1-8B-Instruct", port=30000)
print(rt.generate("你好"))
rt.shutdown()
```

## 4.3 Backend 抽象：前端与后端的解耦

`python/sglang/lang/backend/` 目录定义了 `BaseBackend` 接口（`base_backend.py`），当前主要有三个实现：

| 实现 | 路径 | 说明 |
| --- | --- | --- |
| `RuntimeEndpoint` | `lang/backend/runtime_endpoint.py` | 通过 HTTP/JSON 与 SRT 服务通信 |
| `OpenAI` | `lang/backend/openai.py` | 兼容任意 OpenAI 兼容服务 |
| `VertexAI` / 其他 | `lang/backend/vertexai.py` 等 | 云端厂商适配 |

`RuntimeEndpoint` 内部把 `GenerateReqInput` 序列化成 HTTP 请求发给 `/generate` 等端点，这与你 curl OpenAI API 是两条独立路径（后者走 `sglang.srt.entrypoints.openai` 的协议层）。`set_default_backend()` 可以随时切换目标。

## 4.4 编程原语：gen / select 与结构化生成

`gen()` 是最核心的原语，它不只是"生成文本"，还能附带约束。这些原语要写在一个 `@sgl.function` 装饰的函数里，再交给 `Runtime`/`RuntimeEndpoint` 执行（`SglFunction` 定义在 `lang/ir.py`，`run`/`run_batch` 在 160/223 行）：

```python
import sglang as sgl

@sgl.function
def program(s, question):
    s += sgl.user(question)
    # 带正则约束的生成
    s += sgl.assistant(sgl.gen("answer", regex=r"\d{11}"))

rt = sgl.Runtime(model_path="meta-llama/Llama-3.1-8B-Instruct", port=30000)
ret = program.run(question="输出一个 10 位手机号：", backend=rt)
print(ret["answer"])
rt.shutdown()
```

`select()`（`lang/api.py` 第 236 行）则是"从给定选项里选一个"，本质是带约束的受限生成。

这些约束最终会传递到 SRT 侧的 `constrained/` 模块（xgrammar/outlines 等），由服务端在采样时强制合法，而不是生成后修补。第 11 章会深入。

## 4.5 会话式编程：system / user / assistant 块

`lang/api.py` 提供成对的作用域函数（`system_begin/system_end`、`user_begin/user_end`、`assistant_begin/assistant_end` 以及简写 `system/user/assistant`），把对话模板拼装从"手工拼字符串"变成"结构化描述"：

```python
@sgl.function
def chat_program(s):
    s += sgl.system("你是数学助手")
    s += sgl.user("1+1=?")
    s += sgl.assistant(sgl.gen("answer"))
```

配合 `user_begin()/user_end()` 等成对原语，可以描述多轮、带图像、带推理过程的复杂对话。`lang/chat_template.py` 负责把这种描述渲染成模型对应的 chat template。

## 4.6 流式输出与异步

`Engine.generate(..., stream=True)`（`entrypoints/engine.py` 第 372 行）支持流式；异步批量用 `async_generate` / `async_encode`（第 441/570 行）：

```python
engine = sgl.Engine(model_path="...")
for chunk in engine.generate("讲个故事", stream=True):
    print(chunk, end="")
engine.shutdown()
```

异步场景见 `examples/runtime/engine/offline_batch_inference_async.py`。

## 4.7 与 SRT 的关系：前端只是"翻译层"

一个容易混淆的点：**前端不参与推理**。`RuntimeEndpoint.generate()` 把参数打包成 `GenerateReqInput`，POST 给 `/generate`，然后由服务端完成 tokenize → 调度 → 推理 → detokenize，再把文本流式返回。前端的所有工作都是"构造请求 + 解析响应"。

因此，学习顺序建议是：

1. 会用前端发请求（本章）；
2. 会用 OpenAI API 发请求（第 5 章）；
3. 弄懂 `/generate` 入口到 GPU 执行之间发生了什么（第 6 章起）。

## 4.8 本章小结

- 前端 = 高层对象（LLM/Engine/Runtime）+ 编程原语（gen/select/chat）。
- Backend 抽象让同一套前端代码可以对接 SRT、OpenAI 兼容服务或云厂商。
- 前端只是请求构造与响应解析层，真正的复杂度在 SRT。
- 下一章看服务端的 HTTP 层：它如何把 OpenAI 协议翻译成内部的 `GenerateReqInput`。
