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

Enable installation with `--enable-npu-megakernel`. During execution, the
provider must agree on admission across TP ranks before it writes KV or starts
a collective. A target binding owns its model compute and returns a normal
logits-processor result. A NextN binding can return one owned tensor containing
the complete linear candidate chain for the existing EAGLE verify path.
Column zero is the seed from draft extend. The host retains sampling, acceptance,
allocator lifetime and scheduling. A provider managing its own captured graphs
must return outputs that remain valid for the overlap consumer.
