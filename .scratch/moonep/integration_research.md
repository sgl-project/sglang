# MoonEP Integration Research Notes

## Context

- Upstream issue `sgl-project/sglang#32607` tracks Kimi-K3 work and includes an
  open roadmap item to integrate MoonEP for better all-to-all performance.
- Local MoonEP checkout used for these notes: `/home/user/MoonEP` at commit
  `0f385f0`.
- SGLang already has several MoE A2A backends selected by `--moe-a2a-backend`;
  DeepEP-style backends are routed through `MaybeTboDeepEPDispatcher` and the
  common dispatcher abstraction
  (`python/sglang/srt/layers/moe/fused_moe_triton/layer.py:109-150`,
  `python/sglang/srt/batch_overlap/two_batch_overlap.py:1074-1100`).

## Primary-source findings

### Existing SGLang DeepEP seam

- The MoE A2A enum and helpers live in
  `python/sglang/srt/layers/moe/utils.py`. DeepEP, Mooncake, MoRI, NIXL, PPLX,
  FlashInfer, MegaMoE, and Ascend backends are represented there
  (`python/sglang/srt/layers/moe/utils.py:28-92`).
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` creates the
  dispatcher. DeepEP-class backends are wrapped by
  `MaybeTboDeepEPDispatcher`, which can allocate one or two inner dispatchers
  depending on two-batch overlap
  (`python/sglang/srt/layers/moe/fused_moe_triton/layer.py:109-150`,
  `python/sglang/srt/batch_overlap/two_batch_overlap.py:1074-1100`).
- `python/sglang/srt/layers/moe/token_dispatcher/deepep.py` exposes two
  dispatch output contracts:
  - normal mode: dispatched hidden states, top-k ids/weights, and
    `num_recv_tokens_per_expert`;
  - low-latency mode: packed hidden states, top-k ids/weights, `masked_m`, and
    `expected_m`
    (`python/sglang/srt/layers/moe/token_dispatcher/deepep.py:96-154`,
    `python/sglang/srt/layers/moe/token_dispatcher/deepep.py:496-846`).
- Current expert runners consume those DeepEP contracts through
  `DeepEPNormalCombineInput` / `DeepEPLLCombineInput`; they do not consume a
  MoonEP communication plan or MoonEP `cu_seqlens`
  (`python/sglang/srt/layers/moe/ep_moe/layer.py:270-341`).

### MoonEP contract

- MoonEP's public API is `moonep.Buffer` in `/home/user/MoonEP/moonep/api.py`.
  It is constructed with static shape parameters `S`, `H`, `K`, `E`, EP rank
  count, and optional `B` prefetch slots
  (`/home/user/MoonEP/moonep/api.py:431-499`).
- `Buffer.dispatch(...)` returns
  `(hidden_nvsh, route_weights_nvs, cu_seqlens, plan)` and requires
  `topk_experts_sk` as `int32` plus a local `tokens_per_expert` vector when
  creating a fresh plan (`/home/user/MoonEP/moonep/api.py:685-814`).
- `Buffer.combine(...)` requires the saved `MoonEPCommPlan` and returns
  token-major output plus optional gathered route weights
  (`/home/user/MoonEP/moonep/api.py:881-1007`).
- `Buffer.prefetch_weight(...)` requires the saved plan plus three contiguous
  BF16 full expert-weight tensors for gate/up/down projections with shape
  `[E+B, H, H']` (`/home/user/MoonEP/moonep/api.py:816-879`).
- MoonEP's README states that framework integration requires one contiguous
  symmetric-memory expert weight tensor per projection. This is a stricter
  contract than SGLang's existing local-expert DeepEP runner path
  (`/home/user/MoonEP/README.md:43-81`).

## Design implication

MoonEP should be a distinct `moonep` backend, not an alias for `deepep`.
Pretending it is DeepEP-compatible would be incorrect because SGLang's current
DeepEP runners expect `num_recv_tokens_per_expert` or `masked_m`, while MoonEP
produces `cu_seqlens` and a reusable `MoonEPCommPlan`.

## First implementation slice

The initial patch should:

1. add `moonep` to the CLI/backend enum and EP-size resolution path;
2. add an explicit `MoonEPDispatcher` placeholder so backend selection is
   centralized and future implementation has a clear seam;
3. fail fast with a clear runtime message rather than falling into DeepEP or
   standard-dispatch assumptions.

## Follow-up tickets

1. **MoonEP dispatch output contract** — done in
   `9865fc123 Add MoonEP dispatcher data contract`
   - Add `DispatchOutputFormat.MOONEP` and `CombineInputFormat.MOONEP`.
   - Carry `hidden_nvsh`, `route_weights_nvs`, `cu_seqlens`, and
     `MoonEPCommPlan` through the MoE core.

2. **MoonEP buffer lifecycle** — initial facade done in
   `MoonEPBuffer`; runtime sizing policy still needs validation
   - Add a process-wide MoonEP buffer facade similar to `DeepEPBuffer`.
   - Key the buffer by static `S`, `H`, `K`, `E`, EP group, and `B`.
   - Decide how SGLang should choose `S` for prefill/decode/cuda-graph capture.

3. **MoonEP weight layout**
   - Add contiguous symmetric-memory `[E+B, H, H']` weight buffers for each
     routed expert projection.
   - Map existing checkpoint-loaded local expert weights into rows `[0, E)`.
   - Allocate/pool rows `[E, E+B)` for prefetch slots.

4. **MoonEP expert runner**
   - Implement a grouped expert GEMM that consumes MoonEP `cu_seqlens`.
   - Wire `prefetch_weight(plan, ...)` between dispatch and expert GEMM.
   - Start with BF16 inference before quantized Kimi-K3 paths.

5. **Kimi-K3 integration and validation**
   - Add a Kimi-K3 launch recipe once runtime execution is implemented.
   - Validate on the multi-GPU/NVLink MoonEP test matrix; local unit tests can
     only cover backend selection and fail-fast behavior.
