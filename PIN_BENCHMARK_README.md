# PIN (Prefix Pinning) for Agentic Cache Control

KV cache prefix pinning with TTL through the Dynamo + SGLang stack.
PIN prevents eviction of cached prefix blocks during memory pressure, enabling
fast TTFT for long-running agentic conversations. Pins auto-expire via TTL
and refresh-on-hit keeps active conversations pinned while idle ones reclaim memory.

## API

```jsonc
POST /v1/chat/completions
{
  "model": "Qwen/Qwen3-14B-FP8",
  "messages": [...],
  "nvext": {
    "cache_control": {
      "type": "ephemeral",    // Anthropic-style cache control
      "ttl": "5m"             // auto-expire after 5 minutes of inactivity
    }
  }
}
```

Supported TTL formats: `"30s"`, `"5m"`, `"30m"`, `"1h"`. Default: `"5m"` (300s).
Refresh-on-hit: each cache hit resets the TTL, so active conversations stay pinned.

## Results (2x L40S 48GB, Qwen3-14B-FP8, TP=1, depth=10, 5x eviction flood)

| Metric | Baseline | Pinned (TTL=5m) |
|--------|----------|-----------------|
| **TTFT** | **1606 ms** | **315 ms** |
| Cached tokens | 0/10830 | 9664/10830 |
| Cache hit | 0% | 89% |
| **Speedup** | -- | **5.1x** |

5,770 flood requests (5x full cache eviction cycles, concurrency=128).
Pinned blocks survive on host memory via HiCache while baseline blocks
are fully evicted.

```
NO PIN:                                    WITH PIN (cache_control + TTL):

Client -> warmup 10 turns                  Client -> warmup 10 turns
  Cache stores prefix (~10.8K tokens)        Cache stores prefix + PIN nodes (ttl=300s)

Flood traffic arrives (5770 reqs)          Flood traffic arrives (5770 reqs)
  LRU evicts ALL cached blocks               LRU evicts unpinned blocks
  Prefix gone from GPU + CPU                 Pinned blocks: GPU evict -> backed up to CPU
                                             Pinned blocks SURVIVE on host memory (L2)

Next request (turn 11):                    Next request (turn 11):
  0 cached tokens                            9664/10830 cached (89%)
  Full prefill: 1606ms TTFT                  Load-back CPU->GPU + partial prefill: 315ms
```

## Branches

| Repo | Branch | Description |
|------|--------|-------------|
| **SGLang** | `idhanani/dyn-1986-poc-pin-v2` | HiRadixCache `pin_prefix`/`unpin_prefix`, `load_back` recovery |
| **Dynamo** | `idhanani/dyn-1986-pin-on-router-queue` | `--enable-agentic-cache-control`, router PIN fire-and-forget, `cache_control` endpoint |

## Prerequisites

- 2x GPUs (tested on L40S 48GB)
- Docker (for etcd, NATS)
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Datasets at `~/datasets/` (see below)

## Datasets

Both datasets are real multi-turn Claude Code conversation traces, representative
of agentic coding workloads with long system prompts and iterative back-and-forth.

### `claude_history_sonnet.jsonl` -- VIP conversation

The conversation whose prefix we pin and measure. A single 32-turn Claude Code
session with a ~3.8K token system prompt, totaling ~29.5K tokens. This simulates
a long-running agentic session where preserving the conversation prefix in cache
avoids expensive re-prefill on each follow-up turn.

Format: 1 JSON line with `{session_id, turns: [{role, text}, ...]}`.

### `long_multiturn_opus.jsonl` -- Flood traffic

Background traffic used to create cache pressure and evict unpinned blocks.
10 diverse multi-turn conversations (19-31 turns each, ~5-6K tokens per
conversation), generated synthetically via [stepfun/step-3.5-flash](https://openrouter.ai/stepfun/step-3.5-flash:free) on OpenRouter. aiperf cycles through these to fill the radix cache with
unrelated prefixes, forcing LRU eviction of the VIP conversation's blocks.

Format: 10 JSON lines, each `{session_id, turns: [{role, text}, ...]}`.

## Running

### Terminal 1: Start the Dynamo stack

```bash
./dynamo-stack.sh
```

Starts 1x Dynamo frontend with KV routing + agentic cache control and
2x SGLang workers (GPUs 0,1) with HiCache and KV event publishing enabled.
The KV router subscribes to BlockStored/BlockRemoved events over ZMQ->NATS
to build a per-worker radix tree index for cache-aware routing.
Edit the config block at the top of `dynamo-stack.sh` to change model,
HiCache ratio, GPU count, etc.

### Terminal 2: Run the benchmark

```bash
# Pinned (with TTL-based cache_control)
python3 bench-pin.py --depth 10

# Baseline (no pin)
python3 bench-pin.py --depth 10 --no-pin
```

### What the benchmark does

```
1. Warmup     Send N conversation turns with nvext.cache_control (if pinned mode)
2. Flood      aiperf blasts unrelated multi-turn requests to evict cache
              (auto-sized from GPU capacity * (1 + hicache_ratio) * 3 cycles)
3. Measure    Send turn N+1 streaming, report TTFT + cached_tokens
```

### Flood sizing

The flood is auto-calculated from the worker's `max_total_num_tokens` (parsed
from the startup log) and the `--hicache-ratio` flag:

```
total_capacity = gpu_capacity * (1 + hicache_ratio)
flood_requests = total_capacity * 3 / 150
```

Each multi-turn flood request adds ~150 unique tokens. 3 eviction cycles ensure
full LRU eviction of unpinned blocks from both GPU and CPU tiers.

## How PIN works

```
Client POST /v1/chat/completions
  with: nvext.cache_control = {"type": "ephemeral", "ttl": "5m"}
    |
    v
Dynamo Frontend (--enable-agentic-cache-control)
    |  Extracts cache_control, parses ttl_seconds (e.g. "5m" -> 300)
    |  KV router selects worker based on prefix overlap
    v
SGLang Worker
    |  Processes request normally (prefill + decode)
    |  Returns streaming response
    v
After stream completes (fire-and-forget):
    Router -> cache_control endpoint -> worker.pin_prefix(token_ids, ttl=300)
    |  Walks radix tree, sets pin_count, pin_expiry, pin_ttl on matched nodes
    |  Pinned nodes survive LRU eviction (GPU -> CPU backup, not deleted)
    v
Next request with same prefix:
    match_prefix finds pinned nodes -> load_back (CPU -> GPU) -> fast TTFT
    Refresh-on-hit: pin_expiry reset to now + pin_ttl on each cache hit
    |
    v
After TTL expires (lazy):
    evict()/evict_host() checks time.monotonic() > pin_expiry
    Expired pins cleared -> node becomes evictable normally
```

### HiCache tier model

```
L1 (GPU VRAM)     -- Active KV cache. Fast. Scarce (~20GB per GPU).
  |  LRU eviction (pinned blocks backed up to L2 first)
  v
L2 (Host Memory)  -- Pinned blocks guaranteed here until TTL expires.
  |                   2-3x GPU budget. Load-back to L1: ~5-10ms.
  |  LRU eviction (pinned blocks skip eviction until TTL expires)
  v
L3 (Storage)      -- Future: Mooncake, 3FS, NIXL. Cross-worker sharing.
```

## Files

| File | Description |
|------|-------------|
| `dynamo-stack.sh` | Start frontend + 2 workers with HiCache, PIN, and KV events |
| `bench-pin.py` | Warmup -> flood -> measure TTFT benchmark |
| `convert_claude_history.py` | Convert Claude Code session JSONL into aiperf multi_turn dataset |
| `HICACHE_KV_EVENTS_DESIGN.md` | Design doc for HiCache KV event emission (multi-worker) |

## Configuration

Key parameters in `dynamo-stack.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HICACHE_RATIO` | 1.0 | CPU pool = this multiple of GPU pool. Higher = more CPU cache headroom |
| `HICACHE_POLICY` | write_through | Copies blocks to CPU immediately on insert |
| `MEM_FRACTION` | 0.5 | Fraction of GPU memory for KV cache |
| `CONTEXT_LENGTH` | 32768 | Max context length per request |

The `--hicache-ratio` flag in `bench-pin.py` must match `HICACHE_RATIO` in
`dynamo-stack.sh` for correct flood sizing.

## Known issues

### Thinking models and prefix stability

PIN requires the chat template to be **prefix-stable** -- the tokenization of
earlier turns must not change when new turns are appended.

Qwen3's chat template conditionally injects `<think>\n\n</think>\n` into
assistant messages based on position. This causes token divergence mid-prefix
and degrades cache hits (~81% instead of ~99%).

Workarounds:
- Use `enable_thinking=False` in `chat_template_kwargs` (what bench-pin.py does)
- Use a non-thinking model (Qwen2.5, Llama, Mistral)
- For Qwen3 in production: strip thinking content from history (prefix-stable)

## Troubleshooting

**`etcd not running`**: Start docker infra (see Setup step 2).

**Workers crash on startup**: Check `/tmp/dynamo-stack/all.log`. Common causes:
- OOM: reduce `MEM_FRACTION` in `dynamo-stack.sh`
- Model not cached: `huggingface-cli download Qwen/Qwen3-14B-FP8`

**`cached_tokens=0` after warmup with pin**: Check worker logs for
`[PIN] pin_prefix` messages. PIN is fire-and-forget after stream completion.

**Flood aiperf errors**: aiperf must be cloned to `~/aiperf` and run via `uv run` (it uses its own venv). `git clone https://github.com/ai-dynamo/aiperf.git ~/aiperf && cd ~/aiperf && uv run aiperf --help`.
