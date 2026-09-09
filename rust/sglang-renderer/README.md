# SGLang renderer

The renderer runs as a separate service. It owns text preprocessing, token decoding,
and OpenAI chat/completion responses. The native Rust server accepts token IDs;
the renderer submits them through the engine's `/generate` endpoint.

## Build and run

From the repository root, build the standalone executable separately from the
Python package. The Python environment must already have the native Rust server
extension built for this checkout.

```sh
cargo build --manifest-path rust/Cargo.toml -p sglang-renderer --release --features http --locked
```

Start the engine in one terminal, using its normal Python launcher.

```sh
SGLANG_RUST_SERVER=1 python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 --port 30001
```

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
`--sampling-defaults openai` matches the Python server's default policy. Use
`--help` for template, parser, and limit options. A custom Cargo target directory
or compilation target changes the executable path shown above.

## Current scope

OpenAI serving supports text chat and completions. Multimodal OpenAI inputs,
`/responses`, and `/messages` are deferred. Automatic Python co-launch and wheel
installation of the renderer are also deferred; manage both processes explicitly.
The renderer does not implement API-key authentication or TLS.
