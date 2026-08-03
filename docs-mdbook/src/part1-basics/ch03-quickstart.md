# 第 3 章 快速开始：安装、启动与第一个请求

> 本章命令以仓库 `python/` 目录下的包为标准。如果你在本地没有 GPU，可以把文中的模型换成 CPU 可跑的量化小模型，并加上 `--device cpu` 一类的参数（具体见 `python/sglang/srt/hardware_backend/` 与 `server_args.py` 的说明）。

## 3.1 安装

SGLang 的 Python 包放在 `python/` 目录，安装入口是 `python/pyproject.toml`。开发模式下推荐可编辑安装：

```bash
cd python
pip install -e ".[all]"
```

只装推理核心（无多模态/评测等额外依赖）：

```bash
pip install -e .
```

安装后确认 CLI 可用：

```bash
sglang version
```

## 3.2 启动服务

现代推荐用法是 `sglang serve`，它由 `python/sglang/cli/main.py` 分发到 `python/sglang/cli/serve.py`：

```bash
sglang serve --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000
```

旧入口仍然可用，代码完全一致（`python/sglang/launch_server.py`）：

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000
```

`serve.py` 里有个值得注意的设计：它会先探测模型类型（`get_is_diffusion_model`），如果是扩散模型就转到 `sglang.multimodal_gen` 的入口，否则走标准 LLM 服务。所以"同一个命令启动两种服务"是这个仓库的现状，读代码时注意分支。

启动日志里你会看到多进程被拉起的过程，例如：

```text
INFO: Launching scheduler process ...
INFO: Launching detokenizer process ...
INFO: The server is listening on http://0.0.0.0:30000
```

## 3.3 发第一个请求

### OpenAI 兼容接口（curl）

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "messages": [{"role": "user", "content": "你好，介绍一下你自己"}],
    "temperature": 0.7
  }'
```

### Python 侧调用（sglang.lang）

当前版本顶层导出的是 `Engine` 与 `Runtime` 两个核心对象（没有独立的 `LLM` 类，见 `python/sglang/__init__.py` 的 `__all__`）：

```python
import sglang as sgl

# 方式一：进程内 Engine（不启动 HTTP 服务，适合离线批处理 / RL rollout）
engine = sgl.Engine(model_path="meta-llama/Llama-3.1-8B-Instruct")
out = engine.generate(["Hello world", "SGLang 是什么"])
print(out[0]["text"])
engine.shutdown()

# 方式二：Runtime（在子进程中启动 HTTP 服务，返回客户端句柄）
rt = sgl.Runtime(model_path="meta-llama/Llama-3.1-8B-Instruct", port=30000)
print(rt.generate("你好"))
rt.shutdown()
```

对话类请求（chat）没有独立的 `chat()` 方法，直接走 OpenAI 兼容接口（见第 5 章）即可；`sglang.lang` 的编程范式（`gen`/`select`）见第 4 章。

## 3.4 仓库里的现成示例

不要自己从零写，仓库 `examples/` 提供了大量可运行脚本：

| 场景 | 路径 |
| --- | --- |
| 离线批处理 | `examples/runtime/engine/offline_batch_inference.py` |
| 异步批处理 | `examples/runtime/engine/offline_batch_inference_async.py` |
| 自定义 FastAPI 服务 | `examples/runtime/engine/custom_server.py` |
| 多模态 | `examples/runtime/engine/offline_batch_inference_vlm.py` |
| embedding | `examples/runtime/embedding.py` |
| LoRA | `examples/runtime/lora.py` |
| token-in/token-out 流式 | `examples/runtime/token_in_token_out/` |
| 推理引擎 (RL 用) | `examples/runtime/engine/launch_engine.py` |

以最基础的离线批处理为例：

```bash
cd examples/runtime/engine
python offline_batch_inference.py --model-path meta-llama/Llama-3.1-8B-Instruct
```

## 3.5 跑 benchmark：感受一下服务形态

`python/sglang/benchmark/` 下有 serving 基准：

```bash
python -m sglang.benchmark.serving --model-path meta-llama/Llama-3.1-8B-Instruct \
  --num-prompts 100 --request-rate 10
```

它走 OpenAI 协议打流式请求，输出吞吐、TTFT、TPOT 等指标。第 19 章会介绍怎么读这些指标。

## 3.6 常用调试参数速查

下面这些参数在后文会反复出现，先混个脸熟（定义都在 `python/sglang/srt/server_args.py`）：

| 参数 | 作用 |
| --- | --- |
| `--tp-size` / `--dp-size` | 张量并行 / 数据并行度 |
| `--mem-fraction-static` | KV Cache 可占用的显存比例 |
| `--chunked-prefill-size` | 超长 prefill 分块大小 |
| `--max-running-requests` | 同时运行请求上限 |
| `--schedule-policy` | 调度优先级策略（lpm/fcfs 等） |
| `--attention-backend` | 注意力实现后端（flashinfer/triton/...） |
| `--disable-cuda-graph` | 关闭 CUDA graph（排查问题用） |
| `--log-level debug` | 查看更详细的内部流转 |

## 3.7 本章小结

- 一条命令启动、一个 curl 验证、一个 Python 脚本调用，SGLang 的"hello world"就完成了。
- 服务是多进程的，启动日志里能看到 Scheduler / Detokenizer 等子进程。
- 官方示例集中在 `examples/runtime/`，benchmark 集中在 `python/sglang/benchmark/`。
- 下一章进入前端 `sglang.lang`，理解"编程式调用"与 HTTP API 之间的关系。
