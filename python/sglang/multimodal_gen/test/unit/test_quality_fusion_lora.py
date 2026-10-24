"""Mounting a quality fusion must not drop an active LoRA adapter's output.

A fused path that multiplies a linear's weight itself sees only the base weight
of a LoRA wrapper. Every fusion in ``_QUALITY_FUSION_HANDLERS`` is either run
below on a site whose linears carry a dynamic adapter, or listed with the
reason its sites cannot drop one, so a new fusion has to pick a side.
"""

import pytest
import torch
import torch.nn as nn

from sglang.kernels.ops.diffusion import (
    FLUX2_NVFP4_SWIGLU_QUANT,
    FUSED_GATE_RMSNORM,
    FUSED_LINEAR_GELU,
    FUSED_LN_MODULATE,
    HELIOS_GATED_RESIDUAL,
    HUNYUAN_QKNORM,
    LINGBOT_VIDEO_GATED_RESIDUAL,
    LINGBOT_VIDEO_RMSNORM,
    LTX2_QKNORM_SPLIT_ROPE,
    LTX2_RMS_NORM_MODULATE,
    MINIMAX_H3_NORM_MODULATE,
    NVFP4_BIAS_GELU,
    QWEN_IMAGE_ADDED_QKV,
    SANA_VIDEO_LINEAR_ATTENTION,
    can_use_linear_gelu,
)
from sglang.kernels.ops.diffusion.sites.fused_linear_gelu_site import (
    can_use_linear_gelu_static,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.lora.linear import (
    BaseLayerWithLoRA,
    wrap_with_lora_layer,
)
from sglang.multimodal_gen.runtime.models.dits.qwen_image import (
    QwenImageCrossAttention,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import _WanGELUMLP
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    _QUALITY_FUSION_HANDLERS,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)

DIM = 64
LORA_RANK = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused path runs on CUDA only"
)

# fusion -> why mounting it cannot drop an adapter's output
_ADAPTER_SAFE_FUSIONS = {
    FLUX2_NVFP4_SWIGLU_QUANT: "fuses only a linear_in that NVFP4 loading marked "
    "ready; a LoRA wrapper carries no such mark",
    NVFP4_BIAS_GELU: "rejects an fc_in without the ModelOpt FP4 method, "
    "which a LoRA wrapper does not expose",
    FUSED_LN_MODULATE: "norm and modulation, no linear",
    LTX2_QKNORM_SPLIT_ROPE: "runs on the projections' outputs",
    LTX2_RMS_NORM_MODULATE: "norm and modulation, no linear",
    FUSED_GATE_RMSNORM: "norm, gate and residual, no linear",
    HUNYUAN_QKNORM: "runs on the projections' outputs",
    LINGBOT_VIDEO_RMSNORM: "norm, no linear",
    LINGBOT_VIDEO_GATED_RESIDUAL: "residual update, no linear",
    HELIOS_GATED_RESIDUAL: "residual update, no linear",
    MINIMAX_H3_NORM_MODULATE: "norm and modulation, no linear",
    SANA_VIDEO_LINEAR_ATTENTION: "runs on the projections' outputs",
}


def _ensure_single_process_parallel_runtime() -> None:
    if model_parallel_is_initialized():
        return
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def _wan_gelu_mlp():
    site = _WanGELUMLP(DIM, 4 * DIM, prefix="", quant_config=None)
    return site, ("fc_in",), lambda site, x: site(x)


def _qwen_image_added_qkv():
    # only the state _get_added_qkv_projections reads
    site = object.__new__(QwenImageCrossAttention)
    nn.Module.__init__(site)
    site.use_fused_added_qkv = True
    site._unquantized_added_qkv_is_packed = True
    site.to_added_qkv = MergedColumnParallelLinear(DIM, [DIM] * 3, bias=True)
    QWEN_IMAGE_ADDED_QKV.mark(site)
    return (
        site,
        ("to_added_qkv",),
        lambda site, x: torch.cat(site._get_added_qkv_projections(x), dim=-1),
    )


# a site builder returns (site, names of its LoRA-wrappable linears, call)
_LORA_SITES = [
    pytest.param(
        FUSED_LINEAR_GELU, _wan_gelu_mlp, id="linear_gelu", marks=requires_cuda
    ),
    pytest.param(QWEN_IMAGE_ADDED_QKV, _qwen_image_added_qkv, id="qwen_added_qkv"),
]


def _attach_dynamic_lora(site: nn.Module, name: str) -> BaseLayerWithLoRA:
    linear = site.get_submodule(name)
    out_features, in_features = linear.weight.shape
    wrapper = wrap_with_lora_layer(linear, lora_rank=LORA_RANK, lora_alpha=LORA_RANK)
    wrapper.set_lora_weights(
        torch.randn(LORA_RANK, in_features, device=DEVICE, dtype=torch.bfloat16),
        torch.randn(out_features, LORA_RANK, device=DEVICE, dtype=torch.bfloat16) * 0.1,
        merge_weights=False,
    )
    site.register_module(name, wrapper)
    return wrapper


def _site_with_dynamic_lora(make_site):
    _ensure_single_process_parallel_runtime()
    torch.manual_seed(0)
    site, linear_names, run = make_site()
    site = site.to(device=DEVICE, dtype=torch.bfloat16)
    for parameter in site.parameters():
        parameter.normal_(mean=0.0, std=0.02)
    wrappers = [_attach_dynamic_lora(site, name) for name in linear_names]
    x = torch.randn(2, 129, DIM, device=DEVICE, dtype=torch.bfloat16)
    return site, wrappers, run, x


def test_every_quality_fusion_has_a_lora_site_or_a_reason():
    with_site = {param.values[0] for param in _LORA_SITES}
    assert with_site.isdisjoint(_ADAPTER_SAFE_FUSIONS)
    unclassified = {fusion for _, fusion in _QUALITY_FUSION_HANDLERS} - (
        with_site | set(_ADAPTER_SAFE_FUSIONS)
    )
    assert not unclassified, (
        "add a LoRA site for these fusions, or say why they cannot drop an "
        f"adapter's output: {sorted(fusion.name for fusion in unclassified)}"
    )


@pytest.mark.parametrize("fusion, make_site", _LORA_SITES)
@torch.no_grad()
def test_mounted_fusion_keeps_the_dynamic_lora_delta(fusion, make_site):
    site, wrappers, run, x = _site_with_dynamic_lora(make_site)
    reference = run(site, x)
    for wrapper in wrappers:
        wrapper.disable_lora = True
    base_only = run(site, x)
    for wrapper in wrappers:
        wrapper.disable_lora = False
    assert not torch.allclose(reference, base_only, rtol=1e-2, atol=1e-2)

    fusion.mount(site)
    try:
        assert torch.equal(run(site, x), reference)
    finally:
        fusion.unmount(site)


@requires_cuda
@torch.no_grad()
def test_linear_gelu_guard_follows_the_adapter_state():
    site, (wrapper,), _, x = _site_with_dynamic_lora(_wan_gelu_mlp)
    fusable = torch.cuda.get_device_capability() < (12, 0)
    assert FUSED_LINEAR_GELU.mount(site)
    try:
        assert not can_use_linear_gelu(wrapper, x)
        wrapper.disable_lora = True
        assert can_use_linear_gelu(wrapper, x) == fusable
        wrapper.disable_lora = False
        # merged: the base weight the epilogue reads already holds the update
        wrapper.merge_lora_weights()
        assert can_use_linear_gelu(wrapper, x) == fusable
    finally:
        FUSED_LINEAR_GELU.unmount(site)


def test_linear_gelu_static_guard_reads_the_wrapped_layer():
    _ensure_single_process_parallel_runtime()
    linear = ColumnParallelLinear(
        DIM, 4 * DIM, bias=True, gather_output=False, params_dtype=torch.bfloat16
    )
    wrapper = wrap_with_lora_layer(linear, lora_rank=LORA_RANK, lora_alpha=LORA_RANK)
    assert can_use_linear_gelu_static(wrapper)

    # the per-shard epilogue would skip the cross-rank gather
    linear.gather_output, linear.tp_size = True, 2
    assert not can_use_linear_gelu_static(wrapper)
    linear.gather_output, linear.tp_size = False, 1

    # the epilogue adds the bias the layer hands back to its caller
    linear.skip_bias_add = True
    assert not can_use_linear_gelu_static(wrapper)
