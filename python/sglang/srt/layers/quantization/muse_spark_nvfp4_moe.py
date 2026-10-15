"""Muse-specific NVFP4 expert preparation for the TRT-LLM and Marlin MoE kernels."""

from __future__ import annotations

from typing import cast

import torch

from sglang.srt.environ import envs
from sglang.srt.layers import muse_spark_expert_tp as expert_tp
from sglang.srt.layers.quantization import marlin_utils_fp4
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4FusedMoEMethod,
    _compute_gemm1_alphas,
)
from sglang.srt.layers.utils import copy_or_rebind_param

_DROPPED_CHECKPOINT_LAYOUT_ERROR = (
    "Muse Spark NVFP4 Marlin experts released their checkpoint-layout tensors "
    "after repacking, so their weights cannot be updated or checked. Restart "
    "with SGLANG_ENABLE_MUSE_SPARK_NVFP4_WEIGHT_UPDATE=1 to keep them."
)


def _align_fp4_moe_weights_for_flashinfer_trtllm(layer) -> None:
    """Align Muse's ``[w1, w3]`` checkpoint layout in-place.

    The kernel permutation interleaves gated rows but does not exchange the
    checkpoint's logical gate/up halves. Muse therefore performs the explicit
    ``[w1, w3] -> [w3, w1]`` reorder before the standard shuffle. The aligned
    tensors replace the canonical parameters so the model keeps only one copy.
    """
    from sglang.srt.layers.quantization.utils import (
        prepare_static_weights_for_trtllm_fp4_moe,
        reorder_w1w3_to_w3w1,
    )
    from sglang.srt.layers.utils import copy_or_rebind_param

    w13_weight = cast(torch.Tensor, layer.gate_up_proj)
    w2_weight = cast(torch.Tensor, layer.down_proj)
    w13_weight_scale = cast(torch.Tensor, layer.gate_up_proj_weight_scale)
    w2_weight_scale = cast(torch.Tensor, layer.down_proj_weight_scale)

    if layer.moe_runner_config.is_gated:
        w13_weight, w13_weight_scale = reorder_w1w3_to_w3w1(
            w13_weight, w13_weight_scale
        )

    (
        gemm1_weights_fp4_shuffled,
        gemm1_scales_fp4_shuffled,
        gemm2_weights_fp4_shuffled,
        gemm2_scales_fp4_shuffled,
    ) = prepare_static_weights_for_trtllm_fp4_moe(
        w13_weight,
        w2_weight,
        w13_weight_scale,
        w2_weight_scale,
        w2_weight.size(-2),
        w13_weight.size(-2) // 2,
        w13_weight.size(0),
        is_gated=layer.moe_runner_config.is_gated,
    )

    copy_or_rebind_param(layer, "gate_up_proj", gemm1_weights_fp4_shuffled.contiguous())
    copy_or_rebind_param(layer, "down_proj", gemm2_weights_fp4_shuffled.contiguous())
    copy_or_rebind_param(
        layer, "gate_up_proj_weight_scale", gemm1_scales_fp4_shuffled.contiguous()
    )
    copy_or_rebind_param(
        layer, "down_proj_weight_scale", gemm2_scales_fp4_shuffled.contiguous()
    )

    # The up-projection (w3), not the gate (w1), supplies TRT-LLM's extra
    # GEMM1-output dequantization factor before GEMM2 requantization.
    w2_input_scale_quant = cast(torch.Tensor, layer.w2_input_scale_quant)
    w13_weight_scale_2 = cast(torch.Tensor, layer.gate_up_proj_weight_scale_2)
    if (
        layer.moe_runner_config.is_gated
        and w13_weight_scale_2.dim() == 2
        and w13_weight_scale_2.shape[1] >= 2
    ):
        w13_input_scale = (1.0 / layer.w13_input_scale_quant).to(torch.float32)
        g1_scale_c = (
            w13_input_scale * w13_weight_scale_2[:, 1] * w2_input_scale_quant
        ).to(torch.float32)
    else:
        g1_alphas = cast(torch.Tensor, layer.g1_alphas)
        g1_scale_c = (w2_input_scale_quant * g1_alphas).to(torch.float32)
    copy_or_rebind_param(layer, "g1_scale_c", g1_scale_c.contiguous())


@torch.no_grad()
def _repack_marlin_expert_(layer, expert_id: int) -> None:
    """Re-derive one expert's Marlin tensors from its checkpoint layout, in place.

    Per-expert form of prepare_moe_nvfp4_layer_for_marlin for a gated layer,
    including its collapse of the gate/up global scales to the gate column.
    """
    expert = slice(expert_id, expert_id + 1)
    intermediate = layer.intermediate_size_per_partition
    hidden = layer.gate_up_proj.shape[-1] * 2
    dtype = layer.params_dtype
    perm = torch.empty(0, dtype=torch.int, device=layer.gate_up_proj.device)
    for marlin_weight, marlin_scale, weight, scale, size_n, size_k in (
        (
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.gate_up_proj,
            layer.gate_up_proj_weight_scale,
            2 * intermediate,
            hidden,
        ),
        (
            layer.w2_weight,
            layer.w2_weight_scale,
            layer.down_proj,
            layer.down_proj_weight_scale,
            hidden,
            intermediate,
        ),
    ):
        marlin_weight.data[expert].copy_(
            marlin_utils_fp4._repack_moe_fp4_weight_for_marlin(
                weight.data[expert],
                num_experts=1,
                size_n=size_n,
                size_k=size_k,
                perm=perm,
            )
        )
        marlin_scale.data[expert].copy_(
            marlin_utils_fp4._permute_moe_fp4_scales_for_marlin(
                scale.data[expert].to(dtype),
                num_experts=1,
                size_n=size_n,
                size_k=size_k,
                group_size=16,
                process_scales=marlin_utils_fp4.nvfp4_marlin_process_scales,
            )
        )
    layer.w13_weight_scale_2.data[expert].copy_(
        marlin_utils_fp4.nvfp4_marlin_process_global_scale(
            layer.gate_up_proj_weight_scale_2.data[expert, 0].to(dtype)
        )
    )
    layer.w2_weight_scale_2.data[expert].copy_(
        marlin_utils_fp4.nvfp4_marlin_process_global_scale(
            layer.down_proj_weight_scale_2.data[expert].to(dtype)
        )
    )


class MuseSparkNvFp4FusedMoEMethod(ModelOptNvFp4FusedMoEMethod):
    """Prepare Muse FP4 experts without process-global mutation.

    The Marlin backend repacks the experts into a kernel layout that loaders
    cannot write. With SGLANG_ENABLE_MUSE_SPARK_NVFP4_WEIGHT_UPDATE it keeps the
    checkpoint-layout tensors as the updatable source of truth and re-derives
    the repack of every rewritten expert in place; otherwise it frees them.
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.keep_checkpoint_layout = (
            envs.SGLANG_ENABLE_MUSE_SPARK_NVFP4_WEIGHT_UPDATE.get()
        )

    def uses_marlin(self) -> bool:
        backend = getattr(self, "_moe_runner_backend", None)
        return backend is not None and backend.is_marlin()

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        """Allocate the serialized HF expert tensors without renaming them."""
        n, h, i = num_experts, hidden_size, intermediate_size_per_partition
        group = self.quant_config.group_size
        for name, shape, dtype in (
            ("gate_up_proj", (n, 2 * i, h // 2), torch.uint8),
            ("down_proj", (n, h, i // 2), torch.uint8),
            ("gate_up_proj_weight_scale", (n, 2 * i, h // group), torch.float8_e4m3fn),
            ("down_proj_weight_scale", (n, h, i // group), torch.float8_e4m3fn),
            ("gate_up_proj_weight_scale_2", (n, 2), torch.float32),
            ("down_proj_weight_scale_2", (n,), torch.float32),
            ("gate_up_proj_input_scale", (n, 2), torch.float32),
            ("down_proj_input_scale", (n,), torch.float32),
        ):
            layer.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.empty(shape, dtype=dtype), requires_grad=False
                ),
            )
        layer.nvfp4_weights_prepared = False
        # Local expert ids whose checkpoint-layout tensors changed after the
        # Marlin repack was derived from them.
        layer.nvfp4_stale_experts = set()
        layer._derived_weight_cache_error = None

    # The base method's Marlin branch, and prepare_moe_nvfp4_layer_for_marlin
    # underneath it, address the experts by their pre-rename names.
    _MARLIN_PARAM_ALIASES = (
        ("w13_weight", "gate_up_proj"),
        ("w2_weight", "down_proj"),
        ("w13_weight_scale", "gate_up_proj_weight_scale"),
        ("w2_weight_scale", "down_proj_weight_scale"),
        ("w13_weight_scale_2", "gate_up_proj_weight_scale_2"),
        ("w2_weight_scale_2", "down_proj_weight_scale_2"),
    )

    def _process_weights_for_marlin(self, layer):
        if layer.nvfp4_weights_prepared:
            # A disk reload or the end of a weight-update session post-processes
            # again; the repack stays where CUDA graphs captured it.
            self.refresh_marlin_weights(layer)
            return
        for alias, native in self._MARLIN_PARAM_ALIASES:
            setattr(layer, alias, getattr(layer, native))
        if self.keep_checkpoint_layout:
            # The base Marlin branch collapses w13_weight_scale_2 to its gate
            # column by rebinding the parameter's data; spare the native copy.
            layer.w13_weight_scale_2 = torch.nn.Parameter(
                layer.gate_up_proj_weight_scale_2.data.clone(), requires_grad=False
            )
        super().process_weights_after_loading(layer)
        # Marlin W4A16 never reads activation scales, and no update sends them.
        layer.gate_up_proj_input_scale._skip_weight_check = True
        layer.down_proj_input_scale._skip_weight_check = True
        if not self.keep_checkpoint_layout:
            # Marlin repacks into freshly allocated parameters. Drop the native
            # names so the unpacked originals are freed instead of being kept
            # alive at full size alongside the repacked copies.
            for _, native in self._MARLIN_PARAM_ALIASES:
                if native in layer._parameters:
                    del layer._parameters[native]
                elif hasattr(layer, native):
                    delattr(layer, native)
            layer._derived_weight_cache_error = _DROPPED_CHECKPOINT_LAYOUT_ERROR
        layer.nvfp4_stale_experts.clear()
        layer.nvfp4_weights_prepared = True

    def refresh_marlin_weights(self, layer) -> None:
        """Re-derive the Marlin repack of every stale expert in place."""
        if not layer.nvfp4_weights_prepared:
            return
        for expert_id in sorted(layer.nvfp4_stale_experts):
            _repack_marlin_expert_(layer, expert_id)
        layer.nvfp4_stale_experts.clear()

    def process_weights_after_loading(self, layer):
        """Prepare the TRT-LLM layout from native HF parameters in place."""
        if self.uses_marlin():
            return self._process_weights_for_marlin(layer)
        input_scale = layer.gate_up_proj_input_scale.max().float()
        output_scale = layer.down_proj_input_scale.max().float()
        if self.quant_config.use_per_token_activation:
            input_scale = torch.ones_like(input_scale)
            output_scale = torch.ones_like(output_scale)
        gate_alpha, up_alpha = _compute_gemm1_alphas(
            layer.gate_up_proj_weight_scale_2, input_scale, True
        )
        copy_or_rebind_param(layer, "g1_alphas", gate_alpha)
        copy_or_rebind_param(layer, "g1_alphas_up", up_alpha)
        copy_or_rebind_param(
            layer, "g2_alphas", output_scale * layer.down_proj_weight_scale_2
        )
        copy_or_rebind_param(layer, "w13_input_scale_quant", input_scale.reciprocal())
        copy_or_rebind_param(layer, "w2_input_scale_quant", output_scale.reciprocal())
        for weight, scale in (
            (layer.gate_up_proj, layer.gate_up_proj_weight_scale),
            (layer.down_proj, layer.down_proj_weight_scale),
        ):
            if (
                scale.dtype != torch.float8_e4m3fn
                or scale.shape[-1] != weight.shape[-1] * 2 // 16
            ):
                raise ValueError(
                    "Muse NVFP4 requires one FP8 scale per 16 weight elements"
                )
        self._align_fp4_moe_weights_for_flashinfer_trtllm(layer)
        layer.nvfp4_weights_prepared = True

    def _align_fp4_moe_weights_for_flashinfer_trtllm(self, layer) -> None:
        if getattr(layer, "use_megamoe", False):
            from sglang.srt.layers.moe.muse_spark_megamoe import (
                build_nvfp4_megamoe_weights,
            )

            # Capture scale metadata while gate/up order is still canonical.
            build_nvfp4_megamoe_weights(layer)
        _align_fp4_moe_weights_for_flashinfer_trtllm(layer)
        if getattr(layer, "use_megamoe", False):
            from sglang.srt.layers.moe.muse_spark_megamoe import (
                bind_megamoe_shared_weights,
            )

            bind_megamoe_shared_weights(layer, "nvfp4xnvfp4")
        elif expert_tp.helion_verify_enabled():
            layer.bind_helion_verify_bank()


__all__ = [
    "MuseSparkNvFp4FusedMoEMethod",
    "_align_fp4_moe_weights_for_flashinfer_trtllm",
]
