"""Independent prepared NVFP4 expert views for prefill and decode runners.

Clone canonical checkpoint tensors before either post-load transformation,
adapt the W13 half order, and prepare independent layouts for each runner.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import replace

import torch

from sglang.srt.runtime_context import get_exec, get_flags


def has_phase_moe_backends() -> bool:
    moe = get_exec().moe
    return (
        not get_flags().moe.in_speculative_scope
        and not get_flags().moe.speculative_context
        and moe.moe_prefill_runner_backend is not None
        and moe.moe_decode_runner_backend is not None
        and moe.moe_prefill_runner_backend != moe.moe_decode_runner_backend
    )


@contextmanager
def moe_backend_scope(backend):
    from sglang.srt.layers.moe.utils import MoeA2ABackend, resolve_moe_runner_backend

    backend = resolve_moe_runner_backend(backend)
    a2a = get_exec().moe.moe_a2a_backend
    if backend.is_flashinfer_megamoe():
        a2a = "flashinfer_megamoe"
    elif a2a == "flashinfer_megamoe":
        # The ordinary runner uses a dispatcher while MegaMoE owns communication.
        a2a = "flashinfer"
    with get_flags().moe.override(
        runner_backend=backend, a2a_backend=MoeA2ABackend(a2a)
    ):
        yield


def configure_phase_moe_layer(layer) -> None:
    if layer.is_draft or not has_phase_moe_backends():
        return
    from sglang.srt.layers.quantization.modelopt_quant import (
        ModelOptNvFp4FusedMoEMethod,
    )

    # Quantization-excluded experts retain their existing fallback backend.
    if not isinstance(layer.quant_method, ModelOptNvFp4FusedMoEMethod):
        return
    moe = get_exec().moe
    layer._phase_moe_backends = (
        moe.moe_prefill_runner_backend,
        moe.moe_decode_runner_backend,
    )
    layer._moe_forward_phase = "prefill"
    layer._derived_weight_cache_error = (
        "Online weight updates are not supported with different prefill/decode "
        "MoE runners: captured CUDA graphs retain independently prepared expert "
        "weights. Restart the worker to prepare both layouts."
    )
    # Model blocks cache this capability before the decode view exists.
    layer.supports_deferred_finalize = False


def _copy_checkpoint_layer(layer):
    if layer._modules:
        raise ValueError(
            "Phase-specific MoE runners do not support expert layers with child modules."
        )
    view = copy.copy(layer)
    view._parameters = {}
    for name, param in layer._parameters.items():
        if param is None:
            view._parameters[name] = None
            continue
        cloned = torch.nn.Parameter(param.detach().clone(), requires_grad=False)
        cloned.__dict__.update(param.__dict__)
        view._parameters[name] = cloned
    view._buffers = {
        name: value.clone() if value is not None else None
        for name, value in layer._buffers.items()
    }
    view._modules = {}
    view._phase_moe_backends = None
    view._moe_forward_phase = "decode"
    view._phase_moe_preparing = True
    view.moe_runner_config = replace(layer.moe_runner_config, layer=view)
    return view


def prepare_phase_moe_weights(method, layer) -> bool:
    backends = getattr(layer, "_phase_moe_backends", None)
    if not backends or getattr(layer, "_phase_moe_preparing", False):
        return False
    if hasattr(layer, "_decode_moe_view"):
        raise ValueError(
            "Reloading weights with different prefill/decode MoE runners is not "
            "supported yet. Restart the worker to prepare both layouts."
        )
    if getattr(layer, "inference_moe_w13_interleaved", False):
        raise ValueError(
            "Phase-specific MoE runners do not support interleaved checkpoint W13 layouts."
        )
    from sglang.srt.layers.moe.fused_moe_triton.layer import create_moe_dispatcher

    prefill, decode = backends
    with moe_backend_scope(prefill):
        load_up_first = method.load_up_proj_weight_first
    view = _copy_checkpoint_layer(layer)
    with moe_backend_scope(decode):
        # Reuse quantization's factory: nvfp4_online methods also require the
        # layer prefix, and model-specific methods can carry constructor state.
        view.quant_method = layer.quant_config.get_quant_method(view, layer.layer_name)
        if type(view.quant_method) is not type(method):
            raise ValueError(
                "Phase-specific MoE runners require the same NVFP4 quantization "
                "method for both phases."
            )
        view.quant_method.create_moe_runner(view, view.moe_runner_config)
        if view.quant_method.load_up_proj_weight_first != load_up_first:
            # The checkpoint loader follows the primary's physical W13 order.
            # MegaMoE consumes [Gate, Up], while CuteDSL/TRTLLM use [Up, Gate].
            # Per-tensor scales are always logical [Gate, Up] and stay unchanged.
            for name in ("w13_weight", "w13_weight_scale"):
                weight = getattr(view, name)
                first, second = weight.data.chunk(2, dim=1)
                weight.data = torch.cat((second, first), dim=1).contiguous()
        view.runner_backend = view.quant_method.get_moe_runner_backend(view)
        view.moe_runner_config.runner_backend = view.runner_backend
        view._set_runner_flags()
        if view.use_flashinfer_trtllm_moe:
            view.moe_runner_config.inplace = False
        view.runner = view.quant_method.runner
        view.dispatcher = create_moe_dispatcher(
            view.moe_runner_config, view.quant_method
        )
        view.quant_method.process_weights_after_loading(view)
        # The secondary method skips create_weights, so its post-load padding
        # can differ from the primary allocation (e.g. non-gated TRTLLM MoE).
        view.moe_runner_config.intermediate_size_per_partition = (
            view.intermediate_size_per_partition
        )
    layer._phase_moe_preparing = True
    try:
        with moe_backend_scope(prefill):
            method.process_weights_after_loading(layer)
    finally:
        layer._phase_moe_preparing = False
    # Keep the view outside module traversal: it is already processed, and has
    # the same layer id as the owner used by piecewise CUDA graph custom ops.
    # Register its tensors on the owner so memory-saver/accounting sees both.
    object.__setattr__(layer, "_decode_moe_view", view)
    for name, tensor in view.named_parameters():
        layer.register_parameter(f"_decode_{name}", tensor)
    for name, tensor in view.named_buffers():
        layer.register_buffer(f"_decode_{name}", tensor, persistent=False)
    return True


def phase_moe_layer(layer):
    from sglang.srt.layers.moe.phase import get_moe_forward_phase

    phase = get_moe_forward_phase()
    if phase is None:
        raise RuntimeError("Phase-specific MoE runners require a forward context.")
    if phase == "decode":
        return getattr(layer, "_decode_moe_view", layer)
    return layer
