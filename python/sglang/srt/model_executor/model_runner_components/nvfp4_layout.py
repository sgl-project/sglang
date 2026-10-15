"""SM120 serial Nemotron NVFP4 layout changes outside CUDA graphs."""

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel, get_schedule

logger = logging.getLogger(__name__)


def initialize_nvfp4_layout(model, *, is_draft_worker):
    if not envs.SGLANG_SM120_NVFP4_PHASE_LAYOUT.get() or is_draft_worker:
        return None

    from sglang.kernels.ops.gemm.sm120_nvfp4_layout import Nvfp4LayoutCycle
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
        _sm120_mtp_mma_supported,
    )
    from sglang.srt.models.nano_nemotron_vl import NemotronH_Omni_Reasoning_V3
    from sglang.srt.models.nemotron_h import NemotronHForCausalLM

    language_model = (
        model.language_model
        if isinstance(model, NemotronH_Omni_Reasoning_V3)
        else model
    )

    if (
        not isinstance(language_model, NemotronHForCausalLM)
        or get_parallel().tp_size != 1
        or get_parallel().ep_size != 1
        or get_parallel().pp_size != 1
        or get_schedule().max_running_requests != 1
        or not _sm120_mtp_mma_supported()
        or torch.cuda.get_device_capability() != (12, 0)
    ):
        raise ValueError(
            "SGLANG_SM120_NVFP4_PHASE_LAYOUT requires serial Nemotron NVFP4 "
            "on SM120 with TP/EP/PP=1 and the native NVFP4 MoE path"
        )
    experts = [
        module
        for module in language_model.modules()
        if getattr(module, "w2_weight", None) is not None
        and module.w2_weight.shape == (512, 1024, 1344)
        and module.w2_weight.dtype == torch.uint8
        and getattr(module, "w13_weight", None) is not None
        and module.w13_weight.shape == (512, 2688, 512)
    ]
    if len(experts) != 40:
        raise ValueError("Expected all 40 native Nemotron NVFP4 MoE layers")
    if not all(
        getattr(module.quant_method, "enable_flashinfer_cutlass_moe", False)
        for module in experts
    ):
        raise ValueError("The layout cycle requires FlashInfer CUTLASS MoE")
    cycle = Nvfp4LayoutCycle([module.w2_weight for module in experts])
    for index, module in enumerate(experts):
        module._sglang_nvfp4_layout_slot = (cycle, index)
    # Graph warmup calls the inner language model directly, whereas replay
    # preparation uses the outer model. Both must transition the same cycle.
    language_model._sglang_nvfp4_layout_cycle = cycle
    model._sglang_nvfp4_layout_cycle = cycle
    logger.info(
        "Prepared reversible NVFP4 FC2 layout cycle: %d layers, %d extra bytes",
        len(experts),
        cycle.workspace_bytes,
    )
    return cycle


def prepare_nvfp4_layout(model, forward_batch):
    cycle = getattr(model, "_sglang_nvfp4_layout_cycle", None)
    if cycle is None:
        return
    from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
        _flashinfer_autotune_active,
    )

    mode = forward_batch.forward_mode
    pair = (
        (mode.is_target_verify() or mode.is_decode_or_idle())
        and forward_batch.input_ids.numel() <= 4
        and not _flashinfer_autotune_active()
    )
    cycle.ensure(pair)


def pair_weight_for_layer(layer):
    slot = getattr(layer, "_sglang_nvfp4_layout_slot", None)
    if slot is None:
        return None
    cycle, index = slot
    return cycle.pair[index] if cycle.is_pair else None
