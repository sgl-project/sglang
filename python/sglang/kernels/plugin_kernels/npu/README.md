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
configuration. It returns a binding, or a `Decline(reason, expected)` when it
refuses. `expected=True` is an intentional off switch and is logged at info;
any other refusal is a warning. Return `None` only when there is no reason to
report. Do not store the reason on the prepare function. Resource preflight
must complete without TP collectives. SGLang compares decisions across the
group before invoking the binding's `build` method.

Enable installation with `--enable-npu-megakernel`. The target binding votes across TP ranks before entering a collective, computes decode or verify logits and returns outputs owned by SGLang.

## Graph and draft lifecycle

An enabled binding can return a phase-specific NPU backend from
`graph_backend(runner, enable_memory_saver)`. Returning None keeps the platform
backend. No discovery or device import occurs here on other platforms.
The provider may opt in to receiving the capture batch with `uses_capture_batch`.
A runner can then build both a native fallback graph and a separate MK artifact
on the actual KV pool. Do not nest a BlockRT capture inside a device graph.

The normal runner's graph eligibility check still applies. Before native target
replay the optional backend may return owned decode/verify logits or decline.
Draft extend and prefill always retain their original metadata and forward.
Only draft decode may fuse: after independent extend, the graph runner asks the
strategy to admit the seed, PD transfer generation and physical slots across TP.
A single replay returns the seed plus N−1 recurrent tokens. It is copied before
reuse and handed to the ordinary verify path. There is no cross-round chain.
Decline before writing; do not retry a different forward after execution.
