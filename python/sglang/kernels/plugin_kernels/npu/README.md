# NPU megakernel providers

An external Python distribution can expose a callable in the existing
`sglang.srt.plugins` entry-point group. Its registration function associates each
host architecture class name with a lazy `module:attribute` spec.

```python
def register():
    from sglang.kernels.plugin_kernels.npu.spec import register as register_spec

    register_spec(architecture="GlmMoeDsaForCausalLM", path="my_provider.glm:TARGET")
    register_spec(architecture="GlmMoeDsaForCausalLMNextN", path="my_provider.glm:NEXTN")
```

Each attribute is a `MegakernelSpec` with the role `target` or `nextn`. Its
`prepare` callback inspects the loaded model, allocated KV pool and resolved
configuration. It returns a binding or declines by returning `None`. Resource
preflight must complete without TP collectives. SGLang compares decisions
across the group before invoking the binding's `build` method.

Enable installation with `--enable-npu-megakernel`. The target binding votes across TP ranks before entering a collective, computes decode or verify logits and returns outputs owned by SGLang.

## EAGLE lifecycle

1. After weights, pools and attention backends are ready, `init_eagle_draft_runtime` selects the NextN strategy on the draft worker. Bootstrap and the native mode use the ordinary EAGLE implementation.
2. After verify, `plan_fused_extend` checks model eligibility, executor coverage and outstanding ownership across TP ranks before metadata preparation or speculative writes. An admitted extend selects row `accept_lens - 1`. The host copies the borrowed tokens and selected hidden state and publishes the candidate chain on the request.
3. At the next draft, `propose_chain` consumes that chain once, validating the request slot and position. In decode-loop mode it instead takes the independent extend seed and local write slots and calls the suffix executor once. Both paths feed the existing EAGLE verify input. Verification, target sampling, bonus and request cleanup remain SGLang operations.
4. An executor returns the complete linear candidate tensor of shape `[batch, N]`. Column zero in decode-loop mode is the seed. Failure after dispatch propagates to the worker.

## Graph runner integration

Today the external executor owns capture, replay, cache eviction and static input addresses. SGLang prefill/decode graph backends are disabled for this provider configuration. A runner can install phase-specific executors once at worker startup using `NpuEagleDraftRuntime.bind_executors(decode=..., extend=...)`. Both implement `DraftExecutor`: `covers(phase, batch)` is a local, side-effect-free capability query; `execute_decode(DraftLoopIO)` and `execute_extend(DraftExtendIO)` receive inputs after TP admission. The executor deals only with this round. EAGLE owns the cross-round chain and the lifetime of borrowed results.

A future SGLang graph backend still needs platform registration, capture/bucket selection and a source of stable bindings. Its setup can select the target binding and install its EAGLE executors before serving; all ranks must choose the same owner. Change the provider's graph-ownership gate when that backend is available. Do not nest a BlockRT capture inside a host graph capture.
