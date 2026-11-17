# sgl-router examples

Two engines behind one router, with cache-aware routing, for two models:
[GLM-5.3-Flash](#glm-53-flash) and [DeepSeek-V4.1-Flash](#deepseek-v41-flash). Each
example shows the engine flags the router needs, the router's tokenizer and routing
flags, and how to choose between `--affinity-mode prefer` and `balanced`.

> **Validation pending.** The commands and numbers below are filled in from real runs
> before this page leaves draft.

- [Build](#build)
- [Engine flags the router needs](#engine-flags-the-router-needs)
- [GLM-5.3-Flash](#glm-53-flash)
- [DeepSeek-V4.1-Flash](#deepseek-v41-flash)
- [Router flags](#router-flags)
- [Prefer or balanced](#prefer-or-balanced)
- [Check routing](#check-routing)
- [Gotchas](#gotchas)

## Build

```bash
cd sgl-router && cargo build --release
```

## Engine flags the router needs

Start each engine with its [cookbook](../docs/cookbook/autoregressive/) recipe and add
two flags:

- `--kv-events-config` publishes KV block events. `--policy cache_aware` builds its
  prefix tree from them.
- `--load-publish-endpoint auto` publishes load reports. Balanced affinity,
  `power_of_two` fallback and `--max-waiting-requests` read them. `auto` takes
  `2 x dp_size` ports from the KV port, so keep co-hosted engines' KV ports apart.

Check that an engine advertises both:

```bash
curl -s localhost:31000/server_info | python3 -c "import json,sys; print(json.load(sys.stdin)['kv_events'])"
```

## GLM-5.3-Flash

The cookbook [High Throughput recipe](../docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx),
split into two TP4 engines on one 8-GPU B300 node (the GB300 recipe's TP):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sglang serve --model-path zai-org/GLM-5.3-Flash --tp 4 \
  --dsa-prefill-backend trtllm --dsa-decode-backend trtllm \
  --kv-cache-dtype fp8_e4m3 --moe-runner-backend flashinfer_trtllm \
  --reasoning-parser auto --tool-call-parser auto \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv"}' \
  --load-publish-endpoint auto \
  --host 127.0.0.1 --port 31000
```

The second engine uses `CUDA_VISIBLE_DEVICES=4,5,6,7`, `tcp://*:5567`, `--port 32000`,
and its own `--nccl-port` if the defaults collide.

Router:

```bash
./target/release/sgl-router --host 0.0.0.0 --port 30000 \
  --model-id zai-org/GLM-5.3-Flash \
  --worker-urls http://127.0.0.1:31000 http://127.0.0.1:32000 \
  --tokenizer-backend fast --tokenizer-l1-cache-mb 4096 \
  --chat-routing reorg --policy cache_aware --affinity-mode prefer \
  --cache-affinity-min-match-ratio 0.2 \
  --max-inflight-requests 56 --max-waiting-requests 16 \
  --retry-max-attempts 2 --request-timeout-secs 540 --stream-idle-timeout-secs 300
```

`--model-id` must match the engines' served model name. The router fetches the
tokenizer, `config.json` and chat template from the Hub under that id. For a local
snapshot, pass `--tokenizer-path /path/to/snapshot/tokenizer.json`: the file, not the
directory. The router reads the other files from the file's directory.

The router renders GLM-5.3-Flash chats itself and forwards `input_ids`. Chats without
tools are served over `/generate`, with the router's `glm45` reasoning parser. Chats
with tools keep the engine's route (the cookbook's `glm47` tool parser), still with
router-rendered ids. Thinking is on by default; turn it off with
`"chat_template_kwargs": {"thinking": false}`.

## DeepSeek-V4.1-Flash

The cookbook [High Throughput recipe](../docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx)
is already 4 GPUs (TP4 + EP4), so one 8-GPU B300 node holds two engines:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sglang serve --model-path deepseek-ai/DeepSeek-V4.1-Flash \
  --trust-remote-code --tp 4 --ep-size 4 --max-running-requests 256 \
  --reasoning-parser auto --tool-call-parser auto \
  --kv-events-config '{"publisher":"zmq","endpoint":"tcp://*:5557","topic":"kv"}' \
  --load-publish-endpoint auto \
  --host 127.0.0.1 --port 31000
```

The second engine uses `CUDA_VISIBLE_DEVICES=4,5,6,7`, `tcp://*:5567`, `--port 32000`.

Router: the GLM command with `--model-id deepseek-ai/DeepSeek-V4.1-Flash`.

The router renders V4.1 chats with deepseek-recipe and encodes them with the
checkpoint's `tokenizer.json`, then parses `deepseek-v41` reasoning and `deepseekv41`
tool calls. Thinking is off by default; any `reasoning_effort` other than `none` turns
it on. If the engines set `SGLANG_DSV41_REASONING_EFFORT` or `SGLANG_DEFAULT_THINKING`,
set the same value on the router, or the router renders a different prompt.

## Router flags

| Flag | What it does |
| --- | --- |
| `--tokenizer-backend fast` | Encode with fastokens, decode with HF. Falls back to `hf` (the default) when fastokens cannot load the tokenizer. |
| `--tokenizer-l1-cache-mb 4096` | Reuse the tokens of an earlier prompt up to its last shared special token, so a multi-turn chat encodes only its new turns. `0` (default) disables it. |
| `--chat-routing reorg` | The routing path this page describes. Without it, the legacy path runs. |
| `--policy cache_aware` | Send a request to the engine holding its longest cached prefix, if the hit is at least `--cache-affinity-min-matched-tokens` (default 1024) and `--cache-affinity-min-match-ratio` of the prompt. Otherwise, `power_of_two`. |
| `--affinity-mode prefer` / `balanced` | What happens when the prefix owner is busy. See [below](#prefer-or-balanced). |
| `--max-inflight-requests N` | Skip an engine with `N` requests in flight from this router. Counted at dispatch, so it holds under bursts. `--max-in-flight` is an alias. |
| `--max-waiting-requests N` | Skip an engine that reports `N` waiting requests. This lags by one load report. |
| `--retry-max-attempts 2` | Retry a request that failed before any response bytes were sent, on an engine it has not tried yet. |

When every engine is at a cap, the router returns 503 right away; it has no queue of
its own. Keep client concurrency below `engines x --max-inflight-requests`.

## Prefer or balanced

Both modes keep a request on the engine holding its prefix. They differ when that
engine is busy.

- **`--affinity-mode prefer`** (default): the owner wins until it fails admission. You
  get the best cache reuse, but a hot prefix keeps its owner at the cap.
- **`--affinity-mode balanced`**: the router also samples a less-loaded alternative. It
  moves the request there when the owner's load is above both
  `alternative x --affinity-load-factor` and `alternative + --affinity-load-gap`. Some
  cache hits are given up to keep both engines below the cap.

```bash
  --affinity-mode balanced --affinity-balanced-by running-requests \
  --affinity-load-factor 1.5 --affinity-load-gap 4
```

Use `--affinity-balanced-by running-requests`. The default `prefill-tokens` (factor 2,
gap 1024 tokens) barely moves on cache hits, because a cached request adds little
pending prefill to its owner, even while the owner is overloaded with decode.

Measured: open-loop, 400 requests at about 8 req/s, 128-768 output tokens, one shared
prefix of about 3-4k tokens. The table gives peak in-flight per engine.

| Config | GLM-5.3-Flash (owner / other, 503s) | DeepSeek-V4.1-Flash (owner / other, 503s) |
| --- | --- | --- |
| `prefer` | pending | pending |
| `balanced` (defaults) | pending | pending |
| `balanced`, `running-requests` 1.5x / 4 | pending | pending |

From most cache-sticky to most balanced:

| Change | Effect |
| --- | --- |
| `--affinity-mode prefer` | The owner wins until it fails admission. |
| `--affinity-mode balanced` | Leave the owner when its pending prefill is more than 2x and 1024 tokens above the alternative's. |
| `... --affinity-balanced-by running-requests --affinity-load-factor 1.5 --affinity-load-gap 4` | Leave the owner when it runs more than 1.5x and 4+ more requests. This reacts to decode load too. |
| `--cache-affinity-min-matched-tokens 4096 --cache-affinity-min-match-ratio 0.5` | Only long hits get affinity. Short ones are load-balanced. |
| `--max-inflight-requests 32` | The owner overflows sooner, in any mode. |
| `--policy power_of_two` | No cache affinity; pure load balancing. |

## Check routing

```bash
curl -s localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/GLM-5.3-Flash","messages":[{"role":"user","content":"What is 2+2?"}]}'

curl -s localhost:30000/metrics | grep -E '^sgl_router_(worker_requests_total|worker_inflight_requests|openai_route_total)'
```

- `sgl_router_worker_requests_total`: dispatches per engine.
- `sgl_router_worker_inflight_requests`: live in-flight requests per engine.
- `sgl_router_openai_route_total{outcome="generate"}`: chats the router served over `/generate`.

Accuracy through the router should match a direct engine. Use
[sgl-eval](https://github.com/sgl-project/sgl-eval) with the cookbook sampling:

```bash
sgl-eval run gsm8k --base-url http://127.0.0.1:30000/v1 --num-examples 200 \
  --temperature 1.0 --top-p 0.95
```

| Model | Through router | Direct to engine |
| --- | --- | --- |
| GLM-5.3-Flash | pending | pending |
| DeepSeek-V4.1-Flash | pending | pending |

## Gotchas

- **Restarted router, empty prefix tree.** Engines do not re-announce blocks they
  already hold, so prefixes cached before a router restart are invisible until they
  are evicted. For a clean test, run `curl -X POST localhost:30000/flush_cache`. In
  Kubernetes, use [peer bootstrap](REFERENCE.md#peer-bootstrap-kubernetes).
- **Shutdown drain warning at startup.** `--shutdown-drain-secs` (default 30) plus
  in-flight requests must fit in the pod's grace period. Outside Kubernetes, ignore
  the warning or pass `--shutdown-drain-secs 0`. In Kubernetes, set
  `--termination-grace-secs` to the pod's `terminationGracePeriodSeconds`.
- **Load reports lag.** Balanced mode and `--max-waiting-requests` read engine reports,
  so a burst that arrives within one report interval behaves like prefer mode.
  `--max-inflight-requests` does not lag.
