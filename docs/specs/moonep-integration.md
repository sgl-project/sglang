# MoonEP integration for Kimi-K3 PoC

## Status

- Workflow phase: implementation, with spec/tickets backfilled after the first
  three PoC slices.
- Branch/worktree: `sg-moonep` in `/home/user/sg-moonep`.
- Local MoonEP reference: `/home/user/MoonEP` at commit `0f385f0`.
- Fork issue map: `wirybeaver/sglang#7`.
- Upstream roadmap context: `sgl-project/sglang#32607` includes the Kimi-K3
  MoonEP integration item.

Completed implementation slices:

1. `1396477f8 Recognize MoonEP MoE A2A backend`
2. `9865fc123 Add MoonEP dispatcher data contract`
3. `d47680245 Add MoonEP buffer facade`

Runtime dispatch is still intentionally guarded until MoonEP expert weight
layout and expert GEMM support exist.

## Problem

SGLang currently has DeepEP-style MoE all-to-all backends whose expert runners
consume either DeepEP normal outputs (`num_recv_tokens_per_expert`) or DeepEP
low-latency outputs (`masked_m`/`expected_m`). MoonEP has a different runtime
contract: dispatch returns a `MoonEPCommPlan`, `cu_seqlens`, and expert-grouped
`[NvS, H]` token storage, and its expert compute requires contiguous
symmetric-memory expert weights in `[E+B, H, H']` layout.

Treating MoonEP as a DeepEP alias would silently route MoonEP outputs into the
wrong runner contract. The integration must therefore introduce a distinct
MoonEP backend path.

## Goals

- Add a production-shaped `--moe-a2a-backend moonep` path for Kimi-K3 PoC work.
- Preserve fail-fast behavior until every required runtime contract is wired.
- Reuse existing SGLang MoE seams where possible: backend enum, dispatcher
  abstraction, `MaybeTboDeepEPDispatcher`, and runtime resource ownership.
- Keep implementation slices modular and independently reviewable.
- Start with BF16 inference support before Kimi-K3 quantized fast paths.

## Non-goals for the PoC

- Do not implement MoonEP training backward support initially.
- Do not make MoonEP look DeepEP-compatible by adapting `cu_seqlens` into
  DeepEP's `num_recv_tokens_per_expert`/`masked_m` contracts.
- Do not add quantized Kimi-K3 kernels before the BF16 path is functional.
- Do not remove the runtime guard until dispatch, weight layout, expert compute,
  combine, and validation are all in place.

## Design decisions

### MoonEP is a distinct A2A backend

`MoeA2ABackend.MOONEP` should stay separate from `DEEPEP`, `MOONCAKE`, `NIXL`,
and `PPLX`. SGLang can instantiate a `MoonEPDispatcher` via the same high-level
MoE dispatcher factory, but the dispatcher output format is `MOONEP`, not a
DeepEP format.

### Dispatch contract

MoonEP dispatch output carries:

- `hidden_states`: MoonEP-dispatched `[NvS, H]` tensor.
- `route_weights_nvs`: optional `[NvS]` route weights.
- `cu_seqlens`: `[E+B]` expert-row cumulative sequence lengths.
- `plan`: opaque MoonEP plan object; typed as `Any` in SGLang so importing the
  dispatcher module does not require the optional `moonep` package.

MoonEP combine input carries:

- expert output `hidden_states` in `[NvS, H]` layout.
- `route_weights_nvs`, when route-weight gather is needed.
- the saved `plan`.

### Buffer lifecycle

MoonEP buffer allocation is static by shape. `MoonEPBuffer` caches process-wide
buffers in `get_resources().buffers["moonep_ep_state"]`, keyed by:

- `S`: max dispatch tokens per rank,
- `H`: hidden size,
- `K`: router top-k,
- `E`: total routed experts in EP group,
- EP group identity and rank count,
- `B`: prefetch slots,
- token padding,
- communication SM count.

Environment knobs:

- `SGLANG_MOONEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` defaults to `128`.
- `SGLANG_MOONEP_NUM_PREFETCH_SLOTS` defaults to `-1`, meaning `E / EP`.
- `SGLANG_MOONEP_TOKEN_PADDING` defaults to `128`.
- `SGLANG_MOONEP_NUM_SMS` defaults to `32`.

The next runtime slice must decide how SGLang chooses `S` across prefill,
decode, and CUDA graph capture. Until then, the backend remains guarded.

### Weight layout

Each routed expert projection needs one contiguous symmetric-memory tensor with
shape `[E+B, H, H']`:

- rows `[0, E)`: all experts in global expert-id order;
- rows `[E, E+B)`: local prefetch slots filled by `buffer.prefetch_weight`.

The layout must preserve SGLang checkpoint loading semantics while exposing the
MoonEP contiguous row contract to the expert GEMM. Prefer a small adapter around
existing expert weight ownership before introducing broad abstractions.

### Expert runner

The first runnable path should be BF16 inference:

1. Dispatch hidden states and route weights with MoonEP.
2. Prefetch selected remote expert weights using the saved plan.
3. Run a MoonEP-compatible grouped expert GEMM over `[NvS, H]` and
   `cu_seqlens[E+B]`.
4. Combine expert outputs with the saved plan.

Quantized Kimi-K3 variants should come after this path is correct.

## Phased tickets

The GitHub issue tracker lives on `wirybeaver/sglang`; completed PoC issues are
later cloned or summarized upstream to `sgl-project/sglang` before upstream PRs.

| Phase | Status | Deliverable |
| --- | --- | --- |
| 1 | Done | Research DeepEP and MoonEP integration seam (`wirybeaver/sglang#8`). |
| 2 | Done | Recognize `moonep` backend and fail fast (`wirybeaver/sglang#9`). |
| 3 | Done | Add MoonEP dispatch/combine data contract (`wirybeaver/sglang#10`). |
| 4 | Done | Add process-wide MoonEP buffer facade (`wirybeaver/sglang#11`). |
| 5 | Next | Decide static token-capacity policy (`wirybeaver/sglang#12`). |
| 6 | Pending | Add MoonEP contiguous symmetric weight layout (`wirybeaver/sglang#13`). |
| 7 | Pending | Add BF16 MoonEP expert runner consuming `cu_seqlens` (`wirybeaver/sglang#14`). |
| 8 | Pending | Wire runtime dispatcher dispatch/prefetch/compute/combine (`wirybeaver/sglang#15`). |
| 9 | Pending | Add Kimi-K3 recipe, validation, and upstream handoff notes (`wirybeaver/sglang#16`). |

Issue links are tracked in `.scratch/moonep/tickets.md`.

## Validation plan

Local/unit validation:

- Backend enum/CLI recognizes `moonep`.
- Server args fail fast while runtime support is incomplete.
- Dispatch/combine formats type-check against the dispatcher protocol.
- Buffer facade keying, env defaults, and destruction work with a fake `moonep`
  module.
- `python3 -m compileall` and `git diff --check` pass.

Runtime validation once executable:

- Multi-GPU/NVLink test against local MoonEP examples for BF16 dispatch,
  prefetch, expert compute, and combine.
- Compare correctness against the existing DeepEP/standard MoE path for a small
  Kimi-style MoE model.
- Validate Kimi-K3 launch recipe and document required env vars.

Known local environment blockers at time of backfill:

- `orjson` is missing, so Python unit tests fail during SGLang import.
- `pytest` is not installed.
- `cargo` is not installed, so Rust `cargo check` cannot run.
