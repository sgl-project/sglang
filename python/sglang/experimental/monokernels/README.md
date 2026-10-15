# Monokernels (experimental)

`--enable-monokernel` runs the decode and verify steps of a supported model on monokernels: persistent kernels
that run most of a decoder layer, including its tensor-parallel all-reduce, in one or two launches. It is off by
default. With it off, nothing changes. Design: RFC #43949.

## Supported

| Model | Devices | Parallelism | Steps |
|---|---|---|---|
| DeepSeek-V4.1-Flash | gfx950 (MI355X) | TP 2 or TP 4, one node | decode and DSpark verify steps up to 48 rows |

## Startup refusal

With the switch on, the server refuses to start, and lists every reason at once, when the deployment uses a
feature the model's monokernels do not support. Refused for every model unless its spec lists the feature:
pipeline parallelism, TP across nodes, data-parallel attention, expert parallelism and EPLB, context parallelism,
decode context parallelism, speculative decoding (DeepSeek-V4.1 accepts DSpark), uneven verify steps, two-batch
overlap, LoRA, PD disaggregation, hierarchical KV cache, memory saver, CPU offload and weight backup,
torch-compiled decode, deterministic inference, RL on-policy mode, a decode CUDA graph that is eager
(`--disable-decode-cuda-graph`, `--cuda-graph-backend-decode disabled`) or breakable (`--debug-cuda-graph`,
`--cuda-graph-backend-decode breakable`), and returning routed experts, indexer top-k or hidden states. A spec that
is not bitwise also refuses deterministic and RL on-policy modes (RFC section 10). A `--watchdog-timeout` below 10 s
is refused (every wait has a deadline of half the watchdog, at most 60 s). Online weight updates are refused while
the kernels are attached.

## Tested recipe (DeepSeek-V4.1-Flash, MI355X)

The cookbook's agentic command (`docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx`, section 3.6), with
`--enable-monokernel` added and nothing else changed: FP4 routed experts and the FP8 shared expert,
`--fp8-gemm-backend aiter`, `--enforce-shared-experts-fusion`, DSpark with block size 5,
`--enable-decoder-swa-bounded-replay`, `--cuda-graph-backend-prefill disabled`, `--cuda-graph-max-bs-decode 64`,
`--watchdog-timeout 3600`, and its environment, including the Engram host table
(`SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1`, `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank`). The cookbook's
data-parallel-attention command (concurrency 32-64) is refused. "Tested" here means it has passed
`test/manual/monokernels/validate.py` with this command, at the commit and GPU count stated in the pull request.

## How a failed step stops (fail-stop)

Every cross-workgroup and cross-GPU wait has a deadline of half the watchdog (at most 60 s). A wait that passes it
records the reason and site in this rank's stop block, flags every peer, and ends its launch; a flagged peer ends
its launch the same way. A rank finishes a step only once it has every peer's data for it, and every serving step
ends with a TP collective (the logits gather) that a stopped rank never joins, so no token of a failed step is
returned and SGLang stops the process tree (RFC section 8). There is no per-launch completion handshake.

## Span naming

The RFC names the per-span entry point `kernel(layer_id, point, key)`; this plugin's build object spells it
`span(plan, layer_id, span)`, taking the step's plan instead of a free key. The hooks `route`, `kernel`, `attach`,
`load_tensor` and `loaded` follow the RFC names.

## Test hooks

- `SGLANG_TEST_MONO_SPANS=<span>,<span>`: run only these spans (default: all). DeepSeek-V4.1's spans are `k1`
  and `ffn`. An unknown name refuses startup.
- `SGLANG_TEST_MONO_FAULT_INJECT=<rank>:<site>[:<after>]`: stall one named wait on one rank after `after` launches,
  to test that the server stops on every rank and sends no token of the failed step. A bad rank or site refuses
  startup.
- `SGLANG_TEST_MONO_CHECK_NUMERICS=log|enforce`: at attach, measure each span against the normal path on the
  loaded weights at every built width; `log` records the values, `enforce` refuses startup above the spec's bounds.
