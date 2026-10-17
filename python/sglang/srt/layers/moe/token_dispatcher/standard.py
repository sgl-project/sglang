from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

import torch

from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    get_local_dp_buffer,
    is_allocation_symmetric,
    is_dp_max_padding,
    mask_dp_pad_moe_topk_ids,
    is_dp_attention_enabled,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutput, TopKOutputChecker
from sglang.srt.layers.moe.utils import (
    DispatcherOutputDtype,
    get_ascend_dispatcher_output_dtype,
    get_moe_a2a_backend,
    get_moe_runner_backend,
    should_use_flashinfer_cutlass_moe_fp4_allgather,
    should_use_npu_moe_allgather,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils.common import (
    get_bool_env_var,
    get_device,
    is_hip,
    is_npu,
)

_is_hip = is_hip()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_npu:
    import torch_npu

from sglang.srt.environ import envs as _envs

_MASK_DP_PAD_MOE = _envs.SGLANG_OPT_MASK_DP_PAD_MOE.get()

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


try:
    from flashinfer import (
        nvfp4_block_scale_interleave as nvfp4_block_scale_interleave_flashinfer,
    )

    from sglang.srt.layers.quantization.modelopt_quant import (
        fp4_quantize as fp4_quantize_flashinfer,
    )
except ImportError:
    fp4_quantize_flashinfer = None
    nvfp4_block_scale_interleave_flashinfer = None


class StandardDispatchOutput(NamedTuple):
    """Standard dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_output: TopKOutput
    # Pre-quantized activation for ``hidden_states``, which itself stays bf16:
    # either a (q, scale) pair (per-token-group-128 fp8, q rows padded to a
    # multiple of 4) or an ``Mxfp8RoutedInputPreQuant``.
    hidden_states_pre_quant: Optional[Tuple] = None

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.STANDARD


assert isinstance(StandardDispatchOutput, DispatchOutput)


class StandardCombineInput(NamedTuple):
    """Standard combine input."""

    hidden_states: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.STANDARD


assert isinstance(StandardCombineInput, CombineInput)


def _map_topk_ids_to_local_experts(
    topk_ids: torch.Tensor, local_expert_mapping: torch.Tensor
) -> torch.Tensor:
    """Map global expert ids while preserving the ``-1`` drop sentinel."""
    return torch.where(
        topk_ids < 0,
        -1,
        local_expert_mapping[topk_ids.clamp_min(0)],
    )


class StandardDispatcher(BaseDispatcher):
    def __init__(self, moe_runner_config: MoeRunnerConfig):
        super().__init__()
        self.moe_ep_size = get_parallel().moe_ep_size
        backend = get_moe_runner_backend()
        self.enable_flashinfer_cutlass_moe = backend.is_flashinfer_cutlass()
        self.enable_flashinfer_mxfp4_moe = backend.is_flashinfer_mxfp4()
        self.enable_flashinfer_trtllm_routed_moe = backend.is_flashinfer_trtllm_routed()
        # AITER fast paths can be on while the MoE runner stays Triton; only the
        # AITER runner keeps global expert IDs, so Triton must remap to local range.
        self.use_aiter_moe_runner = backend.is_aiter() or (
            backend.is_auto() and _use_aiter and get_moe_a2a_backend().supports_aiter()
        )
        # Skip local expert mapping when the backend handles EP with global expert IDs:
        # - cutlass / cutedsl / trtllm_routed handle EP internally
        # - mxfp4 dispatcher mapping is already global
        # - hpc_ops consumes global ids together with rank_ep / num_expert_total
        # - flashinfer_megamoe routes by global expert ID inside the mega kernel
        self.skip_local_expert_mapping = (
            backend.is_flashinfer_cutlass()
            or backend.is_flashinfer_cutedsl()
            or backend.is_flashinfer_trtllm()
            or backend.is_experimental_sgl_trtllm()
            or backend.is_flashinfer_trtllm_routed()
            or backend.is_hpc_ops()
            or backend.is_flashinfer_megamoe()
            or self.enable_flashinfer_mxfp4_moe
        )
        self.num_experts = moe_runner_config.num_experts
        self.num_local_experts = moe_runner_config.num_local_experts
        self.num_local_shared_experts = moe_runner_config.num_fused_shared_experts
        self.num_local_routed_experts = (
            self.num_local_experts - self.num_local_shared_experts
        )
        self.moe_ep_rank = get_parallel().moe_ep_rank
        self.local_expert_mapping = None
        self.expert_mask_gpu = None
        self._allgather_sizes: Optional[list[int]] = None
        self._is_dp: bool = True

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> StandardDispatchOutput:

        if should_use_flashinfer_cutlass_moe_fp4_allgather():
            # all-gather fp4 hidden states
            if (
                fp4_quantize_flashinfer is None
                or nvfp4_block_scale_interleave_flashinfer is None
            ):
                raise RuntimeError(
                    "FlashInfer fp4_quantize and nvfp4_block_scale_interleave "
                    "are required for the flashinfer_cutlass FP4 all-gather "
                    "path."
                )
            global_scale = self.quant_config.get("input_global_scale", None)
            assert global_scale is not None, "input_global_scale is not set"
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

            # Quantize before comm, swizzle after.
            with use_symmetric_memory(
                get_parallel().tp_group, disabled=not is_allocation_symmetric()
            ):
                if hidden_states.shape[0] > 0:
                    x, x_sf = fp4_quantize_flashinfer(
                        hidden_states, global_scale, is_sf_swizzled_layout=False
                    )
                else:
                    x_col = hidden_states.shape[1]
                    x = torch.zeros(
                        0, x_col // 2, dtype=torch.uint8, device=hidden_states.device
                    )
                    x_sf = torch.zeros(
                        0, x_col // 16, dtype=torch.uint8, device=hidden_states.device
                    )
            topk_weights, topk_ids, x, x_sf = get_parallel().tp_group.all_gatherv(
                [topk_weights, topk_ids, x, x_sf], sizes=get_dp_global_num_tokens()
            )
            # TODO: fuse into cutlass moe
            x_sf = nvfp4_block_scale_interleave_flashinfer(x_sf)

            hidden_states = x
            hidden_states_scale = x_sf
            topk_output = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=topk_output.router_logits,  # never tested
            )
        elif should_use_npu_moe_allgather():
            # NPU all-gather (DP) / all-reduce (non-DP) MoE path.
            topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
            dispatcher_dtype = get_ascend_dispatcher_output_dtype(self)
            self._is_dp = is_dp_attention_enabled()

            # Pre-quantize to MXFP8 before any communication to halve
            # bandwidth.  Per-token quant is safe locally: each rank
            # quantizes its own tokens and the gather (if any) just
            # concatenates rows without mixing them.
            if dispatcher_dtype is DispatcherOutputDtype.MXFP8:
                if hidden_states.shape[0] > 0:
                    hidden_states, hidden_states_scale = (
                        torch_npu.npu_dynamic_mx_quant(
                            hidden_states, dst_type=torch.float8_e4m3fn
                        )
                    )
                else:
                    hidden_size = hidden_states.shape[1]
                    hidden_states = torch.empty(
                        0, hidden_size,
                        dtype=torch.float8_e4m3fn, device=hidden_states.device,
                    )
                    hidden_states_scale = torch.empty(
                        0, hidden_size // 64, 2,
                        dtype=torch.uint8, device=hidden_states.device,
                    )
            else:
                hidden_states_scale = None

            if self._is_dp:
                # DP path: all-gather different tokens from DP ranks, then
                # reduce-scatter partial outputs back.  The e8m0 scale stays
                # uint8 (as returned by npu_dynamic_mx_quant) for HCCL
                # compatibility; it is viewed to float8_e8m0fnu only when
                # the kernel consumes it downstream.
                self._allgather_sizes = get_dp_global_num_tokens()
                gather_list = [hidden_states, topk_weights, topk_ids]
                if hidden_states_scale is not None:
                    gather_list.append(hidden_states_scale)
                gathered = get_parallel().tp_group.all_gatherv(
                    gather_list, sizes=self._allgather_sizes,
                )
                if hidden_states_scale is not None:
                    hidden_states, topk_weights, topk_ids, hidden_states_scale = gathered
                else:
                    hidden_states, topk_weights, topk_ids = gathered
            else:
                # Non-DP path: every rank already has the full token set.
                # No all-gather needed; combine will all-reduce instead.
                self._allgather_sizes = None

            topk_output = StandardTopKOutput(
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                router_logits=topk_output.router_logits,
            )
        else:
            hidden_states = hidden_states
            hidden_states_scale = None

        if (
            self.moe_ep_size > 1
            and not self.skip_local_expert_mapping
            and TopKOutputChecker.format_is_standard(topk_output)
        ):
            if self.local_expert_mapping is None:
                device = get_device()
                self.local_expert_mapping = torch.full(
                    (self.num_experts,), -1, dtype=torch.int32, device=device
                )
                self.local_expert_mapping[
                    self.moe_ep_rank * self.num_local_routed_experts : (
                        self.moe_ep_rank + 1
                    )
                    * self.num_local_routed_experts
                ] = torch.arange(
                    0, self.num_local_routed_experts, dtype=torch.int32, device=device
                )

                if self.num_local_shared_experts > 0:
                    self.local_expert_mapping[-self.num_local_shared_experts :] = (
                        torch.arange(
                            self.num_local_routed_experts,
                            self.num_local_routed_experts
                            + self.num_local_shared_experts,
                            dtype=torch.int32,
                            device="cpu",
                        )
                    )

        if self.local_expert_mapping is not None and not self.skip_local_expert_mapping:
            if self.use_aiter_moe_runner and self.expert_mask_gpu is None:
                self.expert_mask_gpu = (
                    (
                        (self.local_expert_mapping >= 0)
                        & (self.local_expert_mapping < self.num_local_experts)
                    )
                    .to(torch.int32)
                    .to(device="cuda")
                )
            elif not self.use_aiter_moe_runner:
                if TopKOutputChecker.format_is_standard(topk_output):
                    topk_ids_local = _map_topk_ids_to_local_experts(
                        topk_output.topk_ids, self.local_expert_mapping
                    )
                    # Drop dp-attention MAX_LEN pad rows from the dispatch:
                    # pad rows carry stale hidden through the router and
                    # their expert outputs are discarded downstream — pure
                    # wasted compute (and a masked-grouped-GEMM workspace
                    # blow-up when they collide on the same top-k). -1 is the
                    # drop sentinel both the triton and deep_gemm runners honor.
                    if _MASK_DP_PAD_MOE and is_dp_max_padding():
                        mask_dp_pad_moe_topk_ids(topk_ids_local)
                    # Mask topk_weights for non-local experts (mapped to
                    # -1) so that unpermute zeroes their contributions.
                    # Each rank only computes its local experts; the
                    # combine reduce-scatter / all-reduce sums the partial
                    # results across ranks.
                    if should_use_npu_moe_allgather():
                        mask = (topk_ids_local >= 0).to(
                            topk_output.topk_weights.dtype
                        )
                        topk_weights_masked = topk_output.topk_weights * mask
                        topk_output = topk_output._replace(
                            topk_ids=topk_ids_local,
                            topk_weights=topk_weights_masked,
                        )
                    else:
                        topk_output = topk_output._replace(topk_ids=topk_ids_local)
                elif TopKOutputChecker.format_is_triton_kernels(topk_output):
                    raise NotImplementedError()

        return StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            topk_output=topk_output,
        )

    def combine(self, combine_input: StandardCombineInput) -> torch.Tensor:
        (hidden_states,) = combine_input
        if should_use_flashinfer_cutlass_moe_fp4_allgather():
            hidden_states, global_hidden_states = (
                get_local_dp_buffer(get_parallel().tp_group),
                hidden_states,
            )
            get_parallel().tp_group.reduce_scatterv(
                global_hidden_states,
                output=hidden_states,
                sizes=get_dp_global_num_tokens(),
            )
        elif should_use_npu_moe_allgather():
            if self._is_dp:
                # DP path: reduce-scatter partial outputs back to each
                # DP rank's local token shard.
                local_output, global_hidden_states = (
                    get_local_dp_buffer(get_parallel().tp_group),
                    hidden_states,
                )
                get_parallel().tp_group.reduce_scatterv(
                    global_hidden_states,
                    output=local_output,
                    sizes=self._allgather_sizes,
                )
                hidden_states = local_output
            else:
                # Non-DP path: every rank has the same tokens, so just
                # all-reduce to sum the per-rank partial expert outputs.
                hidden_states = get_parallel().moe_ep_group.all_reduce(
                    hidden_states
                )
        return hidden_states
