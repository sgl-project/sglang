# SGLang renderer

The renderer runs as a separate service. It owns text preprocessing, token decoding,
and OpenAI chat/completion responses. It submits token IDs through the native
Rust server's existing `/generate` endpoint.

The renderer targets the existing `/generate` contract on SGLang main and must
work with an unmodified Rust server. It accepts both cumulative and incremental
streaming responses, using the engine's configured format. Additional generate
request fields or server behavior changes are deferred to separate PRs.

## Build and run

From the repository root, build the standalone renderer. Rendering and
tokenization work without an engine; generation requires a running SGLang engine.

```sh
cargo build --manifest-path rust/Cargo.toml -p sglang-renderer --release --features http --locked
```

Start the engine in one terminal.

```sh
SGLANG_RUST_SERVER=1 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 --port 30001 --skip-server-warmup
```

Keep engine tokenization enabled for stop conditions and minimum-token handling.

Start the renderer in another terminal. Match the engine's model revision,
tokenizer, context limit, and sampling defaults. Set tool and reasoning parsers
on the renderer when needed.

```sh
rust/target/release/sglang-renderer meta-llama/Llama-3.1-8B-Instruct \
  --engine-url http://127.0.0.1:30001 \
  --host 127.0.0.1 --port 30000 \
  --sampling-defaults openai --proxy-unhandled-routes
```

Send OpenAI requests to port 30000. With `--proxy-unhandled-routes`, routes such as
`/v1/models` and engine health checks are forwarded to the engine. The renderer's
own `/_sglang_renderer/ready` endpoint returns HTTP 204 with
`x-sglang-renderer: ready`; engine readiness is checked separately.

For preprocessing without an engine, omit `--engine-url`. This mode serves render
and tokenization endpoints without inference.

```sh
rust/target/release/sglang-renderer meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 --port 30000 --sampling-defaults openai
```

The CLI defaults to sampling parameters from the model's generation config.
`--sampling-defaults openai` matches SGLang's OpenAI API defaults. Use
`--help` for template, parser, and limit options. A custom Cargo target directory
or compilation target changes the executable path shown above.

## Tool-call parser support

`--tool-call-parser` uses Dynamo's parsers. See
[Dynamo's supported tool-call parsers](https://docs.nvidia.com/dynamo/dev/parsing/tool-call-parsing#supported-tool-call-parsers)
for parser names and model formats. These SGLang names need special attention:

| SGLang name | Renderer support |
| --- | --- |
| `llama3` | Accepted alias for `llama3_json` |
| `qwen` | Accepted alias for `qwen25` |
| `glm`, `glm45` | Accepted aliases for `glm47` |
| `deepseekv3` | Use `deepseek_v3` |
| `gpt-oss` | Use `harmony` |
| `step3` | Unsupported |

Reasoning parsers are configured separately with `--reasoning-parser`.

## Docker image

Build the CPU-only renderer image from the repository root (`linux/amd64` or
`linux/arm64`).

```sh
docker buildx build --load -f docker/renderer.Dockerfile \
  -t local/sglang-renderer:dev .
```

Run preprocessing without an engine.

```sh
docker run --rm -p 30000:30000 \
  -v renderer-cache:/home/sglang/.cache/huggingface \
  -e HF_TOKEN \
  local/sglang-renderer:dev meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --sampling-defaults openai
```

For inference, add `--engine-url` with a URL reachable from the container.

## Current scope

OpenAI serving supports text chat and completions. Multimodal OpenAI inputs,
`/responses`, and `/messages` are deferred. Automatic engine launch and packaged
renderer installation are also deferred; manage both processes explicitly.
The renderer does not implement API-key authentication or TLS.
