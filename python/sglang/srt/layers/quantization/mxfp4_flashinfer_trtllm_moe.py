from __future__ import annotations

import logging
from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

import torch
from torch.nn import Module
from torch.nn.parameter import Parameter

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.layers.utils.common import copy_or_rebind_param
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_platform,
)
from sglang.srt.utils import (
    is_flashinfer_available,
    log_info_on_rank0,
    set_weight_attrs,
)
from sglang.srt.utils.common import next_power_of_2, print_warning_once

_MXFP8_QUANTIZE_BACKEND = "cute-dsl" if get_platform().is_sm100 else "cuda"

if is_flashinfer_available():
    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )
    from flashinfer.utils import (
        get_shuffle_matrix_a_row_indices,
        get_shuffle_matrix_sf_a_row_indices,
    )

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

from sglang.srt.utils.common import get_bool_env_var

_USE_OFFICIAL_SHUFFLE = get_bool_env_var(
    "SGLANG_MXFP4_USE_OFFICIAL_SHUFFLE", default="true"
)


def _pad_intermediate_size(layer: Module, intermediate_size: int) -> None:
    padded_size = (intermediate_size + 127) // 128 * 128
    if padded_size == intermediate_size:
        return

    if layer.w13_weight.shape[1] == 2 * padded_size:
        # Partial checkpoint loads leave the padding untouched. After a kernel
        # row permutation those slots can contain data, so clear them on reload.
        for name, fill_value in (("w13_weight", 0), ("w13_weight_scale_inv", 1)):
            param = getattr(layer, name).data
            param[:, intermediate_size:padded_size].fill_(fill_value)
            param[:, padded_size + intermediate_size :].fill_(fill_value)
        layer.w2_weight.data[:, :, intermediate_size // 2 :].zero_()
        layer.w2_weight_scale_inv.data[:, :, intermediate_size // 32 :].fill_(1)
        return

    # Gate and up occupy separate halves; each needs its own zero tail.
    for name, fill_value in (
        ("w13_weight", 0),
        ("w13_weight_scale_inv", 1),
    ):
        param = getattr(layer, name)
        num_experts, _, width = param.shape
        padded = torch.full(
            (num_experts, 2 * padded_size, width),
            fill_value,
            dtype=param.dtype,
            device=param.device,
        )
        padded[:, :intermediate_size] = param[:, :intermediate_size]
        padded[:, padded_size : padded_size + intermediate_size] = param[
            :, intermediate_size:
        ]
        param.data = padded

    for name, elements_per_column, fill_value in (
        ("w2_weight", 2, 0),
        ("w2_weight_scale_inv", 32, 1),
    ):
        param = getattr(layer, name)
        num_experts, hidden_size, width = param.shape
        padded = torch.full(
            (num_experts, hidden_size, padded_size // elements_per_column),
            fill_value,
            dtype=param.dtype,
            device=param.device,
        )
        padded[:, :, :width] = param
        param.data = padded

    layer.intermediate_size_per_partition = padded_size
    print_warning_once(
        f"flashinfer_mxfp4 MoE padded the local intermediate size from "
        f"{intermediate_size} to {padded_size} for 128-element kernel alignment "
        "after TP weight loading. Padding adds unused channels and may waste "
        "compute and memory. Use this TP MoE configuration with caution and "
        "benchmark it against a TP/EP configuration that avoids padding."
    )


def routed_hidden_size(layer: Module) -> int:
    """Hidden size the routed GEMM1 expects (packed bytes hold two FP4 values)."""
    w13 = layer.w13_weight
    return w13.shape[2] * 2 if w13.dtype in (torch.int8, torch.uint8) else w13.shape[2]


class Mxfp8RoutedInputPreQuant(NamedTuple):
    """MXFP8 linear-layout quant of the routed MoE input. ``ready`` is recorded on
    the producing stream; the consumer must wait on it before the routed MoE op."""

    x_q: torch.Tensor
    x_sf: torch.Tensor
    ready: Optional[torch.cuda.Event]


def _shuffled_scale_name(name: str) -> str:
    """Name of the kernel-layout copy kept next to a checkpoint-layout scale."""
    return f"{name}_shuffled"


class Mxfp4FlashinferTrtllmMoEMethod(QuantizeMethodBase):
    fuse_routed_scaling_factor_in_topk = True

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix
        # precision=fp8 is an SM90 knob (Humming W4A8); this SM100 trtllm path
        # already runs MXFP8 activations, so the flag is inert here rather than
        # an error -- one config can move across hardware.
        self.flashinfer_mxfp4_moe_precision = (
            get_exec().moe.flashinfer_mxfp4_moe_precision
        )

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config
        # Applies flashinfer trtllm directly instead of going through a
        # MoeRunner; FusedMoE still reads `.runner`, and this class is not a
        # FusedMoEMethodBase subclass so it inherits no default.
        self.runner = None

        swiglu_limit = moe_runner_config.swiglu_limit
        clamp_limit_tensor = (
            torch.full(
                (layer.num_local_experts,),
                swiglu_limit,
                dtype=torch.float32,
                device=layer.w13_weight.device,
            )
            if swiglu_limit is not None
            else None
        )
        layer.register_buffer(
            "_gemm1_clamp_limit_tensor",
            clamp_limit_tensor,
            persistent=False,
        )

    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype,
        **extra_weight_attrs,
    ):
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

        fp4_block_k = 32
        self._intermediate_size_unpadded = intermediate_size_per_partition
        self._weights_in_kernel_layout = False
        layer.register_buffer("_mxfp4_w13_restore_rows", None, persistent=False)
        layer.register_buffer("_mxfp4_w2_restore_rows", None, persistent=False)

        w13_weight = Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        w2_weight = Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // fp4_block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // fp4_block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w13_weight_scale.format_ue8m0 = False
        w2_weight_scale.format_ue8m0 = False
        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.BLOCK.value
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, scale_attrs)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, scale_attrs)

    def restore_weights_before_loading(self, layer: Module) -> None:
        """Restore checkpoint row order without replacing captured weight storage."""
        if not self._weights_in_kernel_layout:
            return
        for weight, rows in (
            (layer.w13_weight, layer._mxfp4_w13_restore_rows),
            (layer.w2_weight, layer._mxfp4_w2_restore_rows),
        ):
            for expert in weight.data:
                expert.copy_(expert[rows])
        self._weights_in_kernel_layout = False

    def process_weights_after_loading(self, layer: Module) -> None:
        """Turn the freshly loaded checkpoint layout into what the kernel reads.

        The kernel wants each expert's rows permuted, with ``w13`` reordered
        from ``[w1, w3]`` to ``[w3, w1]`` first. For the weights that is a pure
        row permutation of a tensor with the same shape and dtype, so it is
        applied in place, one expert at a time; the parameters keep their
        identity and storage, which is what ``load_weights`` (via
        ``weight_loader``) and a captured CUDA graph both depend on. The
        scales additionally change dtype, interleaving and extent, so their
        kernel layout lives in separate ``*_shuffled`` parameters rebuilt from
        the checkpoint-layout scales here; the checkpoint-layout scales are
        never modified and stay loadable.
        """
        if self._weights_in_kernel_layout:
            return
        self._fp8.process_weights_after_loading(layer)

        if getattr(layer, "_mega_moe_weights_built", False):
            return

        _pad_intermediate_size(layer, self._intermediate_size_unpadded)
        log_info_on_rank0(
            logger,
            f"Shuffling FP4 expert weights for TRT-LLM MxFP4 kernel "
            f"(layer: {self.prefix})...",
        )

        w13 = layer.w13_weight.data
        w2 = layer.w2_weight.data
        w13_scale = layer.w13_weight_scale_inv.data
        w2_scale = layer.w2_weight_scale_inv.data
        num_experts = w13.shape[0]
        device = w13.device

        if w13_scale.dtype == torch.float32:
            w13_scale = w13_scale.to(torch.float8_e8m0fnu)
            w2_scale = w2_scale.to(torch.float8_e8m0fnu)

        w13_u8 = w13.view(torch.uint8)
        w13_s_u8 = w13_scale.view(torch.uint8)
        w2_u8 = w2.view(torch.uint8)
        w2_s_u8 = w2_scale.view(torch.uint8)

        epilogue_tile_m = 128
        if _USE_OFFICIAL_SHUFFLE:
            cache: dict = {}
            w13_rows = _maybe_get_cached_w3_w1_permute_indices(
                cache, w13_u8[0], epilogue_tile_m
            )
            w13_sf_rows = _maybe_get_cached_w3_w1_permute_indices(
                cache, w13_s_u8[0], epilogue_tile_m, num_elts_per_sf=16
            )
            w2_rows = get_w2_permute_indices_with_cache(
                cache, w2_u8[0], epilogue_tile_m
            )
            w2_sf_rows = get_w2_permute_indices_with_cache(
                cache, w2_s_u8[0], epilogue_tile_m, num_elts_per_sf=16
            )
        else:
            w13_rows = get_shuffle_matrix_a_row_indices(w13_u8[0], epilogue_tile_m)
            w13_sf_rows = get_shuffle_matrix_sf_a_row_indices(
                w13_s_u8[0], epilogue_tile_m
            )
            w2_rows = get_shuffle_matrix_a_row_indices(w2_u8[0], epilogue_tile_m)
            w2_sf_rows = get_shuffle_matrix_sf_a_row_indices(
                w2_s_u8[0], epilogue_tile_m
            )
        # [w1, w3] -> [w3, w1] is a row permutation as well; fold it into the
        # gather so each expert is rewritten exactly once.
        half = w13.shape[1] // 2
        w3_w1_rows = torch.cat(
            [
                torch.arange(half, 2 * half, device=device),
                torch.arange(0, half, device=device),
            ]
        )
        w13_rows = w3_w1_rows[w13_rows.to(device)]
        w13_sf_rows = w3_w1_rows[w13_sf_rows.to(device)]
        w2_rows = w2_rows.to(device)
        w2_sf_rows = w2_sf_rows.to(device)
        first_build = layer._mxfp4_w13_restore_rows is None
        if first_build:
            layer._mxfp4_w13_restore_rows = torch.argsort(w13_rows)
            layer._mxfp4_w2_restore_rows = torch.argsort(w2_rows)

        g1_s, g2_s = [], []
        for i in range(num_experts):
            w13_u8[i].copy_(w13_u8[i][w13_rows])
            w2_u8[i].copy_(w2_u8[i][w2_rows])
            g1_s.append(block_scale_interleave(w13_s_u8[i][w13_sf_rows].contiguous()))
            g2_s.append(block_scale_interleave(w2_s_u8[i][w2_sf_rows].contiguous()))

        # Keep scales as Parameters: EPLB moves expert-indexed parameters, not buffers.
        # FP32 load scales stay resident alongside shuffled scales (1/4 of packed weight bytes).
        copy_or_rebind_param(
            layer,
            _shuffled_scale_name("w13_weight_scale_inv"),
            torch.stack(g1_s).reshape(num_experts, w13.shape[1], -1),
        )
        copy_or_rebind_param(
            layer,
            _shuffled_scale_name("w2_weight_scale_inv"),
            torch.stack(g2_s).reshape(num_experts, w2.shape[1], -1),
        )

        self._register_static_scale_ones(layer)
        self._weights_in_kernel_layout = True
        if first_build:
            torch.cuda.empty_cache()

    def _register_static_scale_ones(self, layer: Module) -> None:
        # Constant across reloads; created once so their addresses stay valid
        # for a captured CUDA graph.
        device = layer.w13_weight.device
        for name in (
            "output1_scale_scalar",
            "output1_scale_gate_scalar",
            "output2_scale_scalar",
        ):
            if getattr(layer, name, None) is not None:
                continue
            layer.register_buffer(
                name,
                torch.ones(layer.num_local_experts, device=device, dtype=torch.float32),
                persistent=False,
            )

    def quantize_routed_input(
        self, hidden_states: torch.Tensor, hidden_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MXFP8 quant of the routed input, with the scale in the linear
        [tokens, hidden // 32] layout the routed MoE op requires."""
        from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

        x_quant, x_scale = flashinfer_mxfp8_quantize(
            hidden_states,
            False,
            alignment=hidden_size,
            backend=_MXFP8_QUANTIZE_BACKEND,
        )
        x_scale = x_scale.view(torch.float8_e4m3fn).reshape(
            *hidden_states.shape[:-1], -1
        )
        return x_quant, x_scale

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.token_dispatcher import (
            DispatchOutputChecker,
            StandardCombineInput,
        )
        from sglang.srt.layers.moe.topk import TopKOutputChecker

        hidden_states = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        pre_quant = getattr(dispatch_output, "hidden_states_pre_quant", None)
        is_flashinfer_dispatch = DispatchOutputChecker.format_is_flashinfer(
            dispatch_output
        )
        if is_flashinfer_dispatch and (
            hidden_states.dtype != torch.bfloat16
            or dispatch_output.hidden_states_scale is not None
        ):
            raise ValueError(
                "FlashInfer A2A + TRT-LLM MXFP4 MoE requires an unquantized BF16 "
                "dispatch payload; activations are quantized locally before GEMM."
            )

        # Loaders keep signed checkpoint bytes; FlashInfer identifies packed FP4
        # and its logical dimensions through unsigned, zero-copy byte views.
        w13 = layer.w13_weight.view(torch.uint8)
        w2 = layer.w2_weight.view(torch.uint8)
        w13_scale = getattr(layer, _shuffled_scale_name("w13_weight_scale_inv")).view(
            torch.float8_e4m3fn
        )
        w2_scale = getattr(layer, _shuffled_scale_name("w2_weight_scale_inv")).view(
            torch.float8_e4m3fn
        )

        intermediate_size = w2.shape[2] * 2
        hidden_size = routed_hidden_size(layer)

        num_local_experts = layer.num_local_experts
        if w13_scale.dim() == 2:
            w13_scale = w13_scale.reshape(num_local_experts, 2 * intermediate_size, -1)
        if w2_scale.dim() == 2:
            w2_scale = w2_scale.reshape(num_local_experts, hidden_size, -1)

        if TopKOutputChecker.format_is_bypassed(topk_output):
            raise NotImplementedError(
                "the old code in this branch is WRONG. e.g. it does not consider HashTopK, and may miss args"
            )
        if not TopKOutputChecker.format_is_standard(topk_output):
            raise ValueError(f"Unsupported topk output format: {topk_output.format}")

        topk_ids = topk_output.topk_ids
        topk_weights = topk_output.topk_weights

        precision = self.flashinfer_mxfp4_moe_precision
        input_ready: Optional[torch.cuda.Event] = None
        if precision == "bf16":
            assert hidden_states.dtype == torch.bfloat16
            x_quant = hidden_states
            x_scale = None
            origin_dim = x_quant.shape[-1]
            if hidden_size != origin_dim:
                x_quant = torch.nn.functional.pad(
                    x_quant,
                    (0, hidden_size - origin_dim),
                    mode="constant",
                    value=0.0,
                )
        elif precision == "default":
            if isinstance(pre_quant, Mxfp8RoutedInputPreQuant):
                assert pre_quant.x_q.shape[0] == hidden_states.shape[0]
                x_quant, x_scale, input_ready = pre_quant
            else:
                x_quant, x_scale = self.quantize_routed_input(
                    hidden_states, hidden_size
                )
        else:
            raise NotImplementedError(f"Unsupported mxfp4 moe precision: {precision}")

        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            _make_deferred_finalize_output,
            is_deferred_finalize_enabled,
            trtllm_moe_enable_pdl,
        )

        num_tokens = x_quant.shape[0]
        # Deferred finalize returns the permuted GEMM2 output plus the routing
        # triple instead of the finalized [T, hidden] tensor.
        defer_finalize = is_deferred_finalize_enabled()
        symm_output = None
        # Preserve the ordinary output shape in the medium-batch autotuner
        # cache key. The deferred ABI ignores this allocation and returns the
        # expanded GEMM output for the separate fused finalize epilogue.
        if not defer_finalize or 96 < num_tokens <= 384:
            with use_symmetric_memory(
                get_parallel().tp_group, disabled=not is_allocation_symmetric()
            ):
                out_hidden_size = (
                    x_quant.shape[-1] * 2
                    if x_quant.dtype == torch.uint8
                    else x_quant.shape[-1]
                )
                symm_output = torch.empty(
                    num_tokens,
                    out_hidden_size,
                    dtype=torch.bfloat16,
                    device=x_quant.device,
                )

        if input_ready is not None:
            # The op launches the routing kernel, so the join must precede it.
            torch.cuda.current_stream().wait_event(input_ready)

        result = trtllm_fp4_block_scale_routed_moe(
            topk_ids=(topk_ids, topk_weights),
            routing_bias=None,
            hidden_states=x_quant,
            hidden_states_scale=x_scale,
            gemm1_weights=w13,
            gemm1_weights_scale=w13_scale,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=layer._gemm1_clamp_limit_tensor,
            gemm2_weights=w2,
            gemm2_weights_scale=w2_scale,
            gemm2_bias=None,
            output1_scale_scalar=layer.output1_scale_scalar,
            output1_scale_gate_scalar=layer.output1_scale_gate_scalar,
            output2_scale_scalar=layer.output2_scale_scalar,
            num_experts=layer.num_experts,
            top_k=topk_ids.shape[1],
            n_group=1,
            topk_group=1,
            intermediate_size=intermediate_size,
            local_expert_offset=layer.moe_ep_rank * layer.num_local_experts,
            local_num_experts=num_local_experts,
            routed_scaling_factor=1.0,
            routing_method_type=int(RoutingMethodType.TopK),
            do_finalize=not defer_finalize,
            tune_max_num_tokens=next_power_of_2(num_tokens),
            output=symm_output,
            enable_pdl=trtllm_moe_enable_pdl(num_tokens),
        )
        if defer_finalize:
            output = _make_deferred_finalize_output(result, top_k=topk_ids.shape[1])
        else:
            output = result[0]

        if is_flashinfer_dispatch:
            from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
                FlashinferCombineInput,
            )

            return FlashinferCombineInput(hidden_states=output)
        # Prefill uses all-gather dispatch and reduce-scatter combine rather than
        # the decode A2A workspace, even when FlashInfer A2A is selected globally.
        return StandardCombineInput(hidden_states=output)


def maybe_fuse_routed_scale_and_shared_add(
    experts,
    routed: torch.Tensor,
    shared: torch.Tensor | None,
    routed_scaling_factor: float,
) -> torch.Tensor:
    # When MxFP4 fusion is on, the upstream `routed *= scale` is skipped and
    # the scaling is folded into the shared-add via `shared.add_(routed,
    # alpha=scale)`. With no shared output, the missing scale is applied
    # in-place. Otherwise `routed` is already scale-final and we just add
    # `shared` (or pass through if there is none).
    from sglang.srt.layers.quantization.expert_pack import ExpertPackMoEMethod
    from sglang.srt.layers.quantization.mxfp4_flashinfer_cutlass_moe import (
        Mxfp4FlashinferCutlassMoEMethod,
    )
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import (
        Mxfp4MarlinMoEMethod,
    )

    fused = isinstance(
        experts.quant_method,
        (
            Mxfp4FlashinferTrtllmMoEMethod,
            Mxfp4FlashinferCutlassMoEMethod,
            Mxfp4MarlinMoEMethod,
            ExpertPackMoEMethod,
        ),
    )
    if fused:
        already_scaled = experts.should_fuse_routed_scaling_factor_in_topk
        if shared is not None:
            alpha = 1.0 if already_scaled else routed_scaling_factor
            return shared.add_(routed, alpha=alpha)
        if already_scaled:
            return routed
        return routed.mul_(routed_scaling_factor)
    if shared is not None:
        routed += shared
    return routed


# Fused finalize + shared add + TP all-reduce
_fused_finalize_all_reduce_world_size: Optional[int] = None
_fused_finalize_all_reduce_probed = False
_fused_finalize_all_reduce_comm = None


def _fused_finalize_all_reduce_comm_world_size() -> Optional[int]:
    """Reserve a separate push plane for up to 384 rows of fused MoE output."""
    global _fused_finalize_all_reduce_world_size, _fused_finalize_all_reduce_probed
    global _fused_finalize_all_reduce_comm
    if not _fused_finalize_all_reduce_probed:
        # The eager warmup must initialize peer workspaces before capture.
        if torch.cuda.is_current_stream_capturing():
            return None
        _fused_finalize_all_reduce_probed = True
        from sglang.kernels.ops.communication import all_reduce_fusion
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            CustomAllReduceV2,
        )

        ca_comm = get_parallel().tp_group.ca_comm
        if isinstance(ca_comm, CustomAllReduceV2) and not ca_comm.disabled:
            fused_comm = ca_comm
            if ca_comm.world_size == 4:
                from sglang.kernels.ops.communication.mp import register_comm_cleanup

                fused_comm = CustomAllReduceV2(
                    ca_comm.group,
                    ca_comm.device,
                    max_pull_size=0,
                    max_pull_blocks=0,
                    max_push_size=4 * 1024 * 1024,
                    max_push_blocks=512,
                )
                register_comm_cleanup(fused_comm)
            if fused_comm.disabled:
                return None
            _fused_finalize_all_reduce_comm = fused_comm
            all_reduce_fusion.register_comm(fused_comm.obj)
            _fused_finalize_all_reduce_world_size = fused_comm.world_size
        else:
            log_info_on_rank0(
                logger,
                "Fused MoE finalize: TP group has no "
                "CustomAllReduceV2 push plane; keeping the unfused finalize path",
            )
    return _fused_finalize_all_reduce_world_size


def should_use_fuse_finalize_all_reduce(
    experts, num_tokens: int, hidden_dim: int
) -> bool:
    """Capability only; the batch-size policy cap lives at the call site. The
    kernel never rescales, so the expert weights must carry the routed scaling."""
    if not isinstance(experts.quant_method, Mxfp4FlashinferTrtllmMoEMethod):
        return False
    if experts.quant_method.flashinfer_mxfp4_moe_precision != "default":
        return False
    if not experts.should_fuse_routed_scaling_factor_in_topk:
        return False
    if num_tokens <= 0:
        return False
    if num_tokens > 96:
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

        if is_batch_invariant_mode_enabled():
            return False
    from sglang.kernels.ops.communication import all_reduce_fusion

    if not all_reduce_fusion.valid_cluster_sizes(hidden_dim):
        return False
    tp_group = get_parallel().tp_group
    if _fused_finalize_all_reduce_comm_world_size() != tp_group.world_size:
        return False
    comm = _fused_finalize_all_reduce_comm
    # Each token row owns a push phase counter on the epilogue's plane.
    if num_tokens > comm.config.num_push_blocks:
        return False
    return all_reduce_fusion.fits_push_slot(comm.max_push_size, num_tokens, hidden_dim)
