# Expert executor providers

`executor_provider.py` supplies an optional construction-time selection seam
for FusedMoE-compatible expert modules. Gemma4 and Kimi-K3 use it for
`self.experts`; their outer MoE blocks, routers and model-specific forward paths
remain unchanged. This is separate from MoE runner/kernel backend registration.

## Register from a platform plugin

Use the existing `sglang.srt.plugins` entry point to call
`register_moe_provider`. The key must equal the active platform's
`device_name` (for example, `"cuda"`, `"rocm"`, or `"your-platform"`).
Do not use the operator dispatch key: in-tree platforms can share `"native"`.

```python
from sglang.srt.layers.moe.executor_provider import (
    MoeExecutorSelection,
    register_moe_provider,
)


def select_experts(request):
    args = request.expert_args
    if args.get("activation", "silu") != "gelu":
        return None
    if args.get("quant_config") is not None:
        return None
    if request.tp_size != 1 or request.ep_size != 1:
        return None
    if not request.a2a_backend.is_none() or request.is_draft:
        return None
    # Also validate the implementation's dtype, layout and runtime-mode limits.
    # Read resolved runtime configuration, not capture/per-forward flags, when
    # deciding whether a planned graph, LoRA or other feature is supported.

    from your_platform.moe import ExpertExecutor
    from sglang.srt.layers.moe.topk import TopKOutputFormat

    return MoeExecutorSelection(ExpertExecutor, TopKOutputFormat.STANDARD)


def register():
    register_moe_provider(
        "your-platform.gated-gelu",
        platform_key="your-platform",
        selector=select_experts,
    )
```

Replace `your_platform` and `your-platform` with your plugin's module and
platform device name. Import hardware implementations only after accepting a
request. Inactive-platform selectors are not invoked. Model identity can be an
additional compatibility guard, but there is no upstream model-to-provider map.

## Request and selection agreement

The request contains a read-only shallow snapshot of the same arguments used
to construct the expert module, the effective parameter dtype, active MoE
backends, TP/EP widths, draft scope, and optional model identity. A quantization
object is passed by reference and must not be mutated. In K3, `hidden_size`
is the expert/latent width and `quant_config` is the expert-specific converted
configuration, not necessarily the outer model's original configuration.

A selector returns `None` only to decline. An accepted factory must support the
caller's existing constructor, parameter names/loading callbacks, backend and
routing attributes, and execution paths for that configuration. This includes
activation/layout/bias, routing-weight placement and scaling, output shape,
mutation/aliasing rules, and reduction ownership. Advanced paths can directly
use expert parameters, split dispatch/combine, or deferred finalization; decline
configurations you cannot support. This API does not standardize those methods.

An optional `required_topk_format` declares a hard routing requirement.
`selection.topk_output_format(model_requirement)` preserves the model's choice
when the provider imposes none, and rejects incompatible hard requirements.
Apply the result before TopK's existing `bind_runner_backend` call. A stock
selection imposes no new format, preserving automatic backend binding.

## Fallback and lifecycle

`resolve_moe_executor` completes general plugin loading before selection. Do not
construct models or resolve executors inside a plugin registration callback.
All providers must register before the first resolution; the metadata registry
then freezes for the process. Requests and selections are evaluated separately
for each layer/target/draft construction and are not cached by model identity.

No match delegates to the existing `get_moe_impl_class`, including its MoRI EPv2
and DeepEP-family decisions. This also permits a legacy global factory hook to
coexist during migration. Fallback is not a promise of support on new hardware.

Names are unique within a platform. Duplicate/late registration errors remain
fatal for that platform even if the general plugin loader logs and catches the
exception. Multiple accepting providers, invalid selector results, and selector
errors fail explicitly. Factory, loading and execution failures never trigger
an automatic retry with the stock executor.

This upstream API does not implement a platform's executor or remove its hooks.
After syncing the API into an OOT package, migrate its selection logic to
providers and remove legacy hooks only after replacement coverage is validated.
