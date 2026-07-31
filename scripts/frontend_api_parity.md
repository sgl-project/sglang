# Frontend API Parity Harness

`frontend_api_parity.py` records the Python frontend's observable HTTP behavior
and compares the Rust frontend against it. It compares status codes, media types,
JSON values, SSE event order, errors, token IDs, and logprobs.

The initial Stage 1 cases are in
`test/fixtures/frontend_api_parity/stage1_generation.json`. They cover OpenAI
Completions and Vertex generation. The OpenAI cases intentionally include unary,
streaming, validation-error, token-ID, and exact-logprob behavior.

## Run it

Start a Python-frontend server with the target model on port 30000, then capture
the reference:

```bash
python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --port 30000 \
  --api-key sk-123456
```

In another terminal:

```bash
python3 scripts/frontend_api_parity.py capture \
  --base-url http://127.0.0.1:30000 \
  --cases test/fixtures/frontend_api_parity/stage1_generation.json \
  --output /tmp/python-stage1.json \
  --label python \
  --revision "$(git rev-parse HEAD)" \
  --header 'Authorization=Bearer sk-123456'
```

Restart the same model and launch configuration with the Rust frontend on the
same port. Compare it with the Python reference:

```bash
SGLANG_RUST_SERVER=1 python3 -m sglang.launch_server \
  --model-path meta-llama/Llama-3.2-1B-Instruct \
  --port 30000 \
  --api-key sk-123456
```

In another terminal:

```bash
python3 scripts/frontend_api_parity.py compare \
  --base-url http://127.0.0.1:30000 \
  --cases test/fixtures/frontend_api_parity/stage1_generation.json \
  --reference /tmp/python-stage1.json \
  --write-actual /tmp/rust-stage1.json \
  --label rust \
  --revision "$(git rev-parse HEAD)" \
  --header 'Authorization=Bearer sk-123456'
```

Compare already captured snapshots without either server:

```bash
python3 scripts/frontend_api_parity.py diff \
  --reference /tmp/python-stage1.json \
  --actual /tmp/rust-stage1.json
```

Run one or more cases by repeating `--case`. A full reference snapshot can be
used for a selected comparison:

```bash
python3 scripts/frontend_api_parity.py compare \
  --base-url http://127.0.0.1:30000 \
  --cases test/fixtures/frontend_api_parity/stage1_generation.json \
  --reference /tmp/python-stage1.json \
  --label rust \
  --case openai_completion_stream
```

The exit status is `0` for a match or successful capture, `1` for a parity
difference, and `2` for a malformed fixture, bad invocation, or HTTP failure.

Run the harness's CPU-only unit tests with:

```bash
.venv/bin/python test/registered/unit/tools/test_frontend_api_parity.py
```

## Case format

Each case supplies an HTTP request and response handling rules:

```json
{
  "name": "example",
  "request": {
    "method": "POST",
    "path": "/v1/completions",
    "json": {
      "model": "default",
      "prompt": "hello",
      "temperature": 0
    }
  },
  "response_mode": "json",
  "normalize_paths": [
    "/body/id",
    "/body/created"
  ],
  "float_tolerance": 0
}
```

`response_mode` is `auto`, `json`, `sse`, or `text`. SSE data is decoded as JSON
when possible, and event order remains significant.

`normalize_paths` contains explicit JSON pointers. `*` matches every key or list
element at that level. Only genuinely unstable values such as request IDs and
timestamps should be normalized; a path that matches nothing is a harness error.

`float_tolerance` is an absolute tolerance for every numeric value in that case.
It defaults to zero, which is the required setting for zero-KL logprob parity.

Pass credentials with `--header`, not in the case file. Command-line headers are
sent to the server but are not written into snapshots.
