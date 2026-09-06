from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Optional, Tuple

import torch

from sglang.srt.distributed import (
    get_tp_group,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    get_local_dp_buffer_len,
    is_allocation_symmetric,
    is_dp_max_padding,
    mask_dp_pad_moe_topk_ids,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    CombineInput,
    CombineInputFormat,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput, TopKOutputChecker
from sglang.srt.layers.moe.utils import (
    get_moe_a2a_backend,
    get_moe_runner_backend,
    should_use_flashinfer_moe_fp4_allgather,
)
from sglang.srt.runtime_context import get_exec, get_parallel
from sglang.srt.utils.common import (
    get_bool_env_var,
    get_device,
    is_hip,
)

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

from sglang.srt.environ import envs as _envs

_MASK_DP_PAD_MOE = _envs.SGLANG_OPT_MASK_DP_PAD_MOE.get()

if TYPE_CHECKING:
    from sglang.srt.layers.moe.topk import TopKOutput


try:
    from flashinfer import (
        nvfp4_block_scale_interleave as nvfp4_block_scale_interleave_flashinfer,
    )

    from sglang.srt.layers.quantization.fp4_utils import (
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
    # SGLANG_OPT_MOE_QUANT_ONCE: optional pre-quantized (q, scale) pair for
    # ``hidden_states`` (per-token-group-128 fp8, q rows possibly padded to a
    # multiple of 4). Consumed by the standard->triton fused runner so it can
    # skip its own activation quant; ``hidden_states`` itself stays bf16.
    hidden_states_pre_quant: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    # FP32 row scales accompanying per-token NVFP4 activation dispatch.
    hidden_states_per_token_scale: Optional[torch.Tensor] = None

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
        self.skip_local_expert_mapping = (
            backend.is_flashinfer_cutlass()
            or backend.is_flashinfer_cutedsl()
            or backend.is_flashinfer_trtllm()
            or backend.is_experimental_sgl_trtllm()
            or backend.is_flashinfer_trtllm_routed()
            or backend.is_hpc_ops()
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

    def dispatch(
        self, hidden_states: torch.Tensor, topk_output: TopKOutput
    ) -> StandardDispatchOutput:

        hidden_states_per_token_scale = None
        if should_use_flashinfer_moe_fp4_allgather():
            (
                hidden_states,
                hidden_states_scale,
                hidden_states_per_token_scale,
                topk_output,
            ) = self._dispatch_allgather(hidden_states, topk_output)
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
                    topk_ids_local = self.local_expert_mapping[topk_output.topk_ids]
                    # Drop dp-attention MAX_LEN pad rows from the dispatch:
                    # pad rows carry stale hidden through the router and
                    # their expert outputs are discarded downstream — pure
                    # wasted compute (and a masked-grouped-GEMM workspace
                    # blow-up when they collide on the same top-k).  Must
                    # run POST-translation (a pre-translation -1 aliases to
                    # the mapping table's last entry); -1 is the drop
                    # sentinel both the triton and deep_gemm runners honor.
                    if _MASK_DP_PAD_MOE and is_dp_max_padding():
                        mask_dp_pad_moe_topk_ids(topk_ids_local)
                    topk_output = topk_output._replace(topk_ids=topk_ids_local)
                elif TopKOutputChecker.format_is_triton_kernels(topk_output):
                    raise NotImplementedError()

        return StandardDispatchOutput(
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            topk_output=topk_output,
            hidden_states_per_token_scale=hidden_states_per_token_scale,
        )

    def _quantize_fp4(self, hidden_states: torch.Tensor, global_scale: torch.Tensor):
        per_token_activation = self.quant_config.get("use_per_token_activation", False)
        per_token_scale = None
        num_tokens, hidden_size = hidden_states.shape

        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            if num_tokens == 0:
                x = hidden_states.new_empty((0, hidden_size // 2), dtype=torch.uint8)
                x_sf = hidden_states.new_empty(
                    (0, hidden_size // 16), dtype=torch.uint8
                )
                if per_token_activation:
                    per_token_scale = hidden_states.new_empty((0,), dtype=torch.float32)
            elif per_token_activation:
                from flashinfer import SfLayout, nvfp4_quantize

                x, x_sf, per_token_scale = nvfp4_quantize(
                    hidden_states,
                    global_scale,
                    sfLayout=SfLayout.layout_linear,
                    per_token_activation=True,
                    backend="cute-dsl",
                )
            else:
                if fp4_quantize_flashinfer is None:
                    raise RuntimeError(
                        "FlashInfer fp4_quantize is required for FP4 all-gather"
                    )
                x, x_sf = fp4_quantize_flashinfer(
                    hidden_states, global_scale, is_sf_swizzled_layout=False
                )

        return (
            x.reshape(num_tokens, hidden_size // 2),
            x_sf.view(torch.uint8).reshape(num_tokens, hidden_size // 16),
            per_token_scale,
        )

    def _dispatch_allgather(self, hidden_states: torch.Tensor, topk_output: TopKOutput):
        x, x_sf, per_token_scale = hidden_states, None, None
        global_scale = self.quant_config.get("input_global_scale")
        if global_scale is not None:
            x, x_sf, per_token_scale = self._quantize_fp4(hidden_states, global_scale)
        # Layers excluded from NVFP4 (e.g. SGLANG_FP4_IGNORED_LAYERS) retain
        # their input precision, with the same token gather/combine geometry.
        payloads = [x]
        if x_sf is not None:
            payloads.append(x_sf)
        if per_token_scale is not None:
            payloads.append(per_token_scale)
        routing_offset = len(payloads)
        if TopKOutputChecker.format_is_bypassed(topk_output):
            # An empty DP rank has no router invocation to supply the logits
            # width or dtype. Use logical routed experts and FP32 on every
            # rank; converting BF16/FP16 logits to FP32 preserves their values.
            if hidden_states.shape[0] == 0:
                num_routed_experts = (
                    self.num_experts
                    - self.num_local_shared_experts
                    - get_exec().moe.ep_num_redundant_experts
                )
                router_logits = hidden_states.new_empty(
                    (0, num_routed_experts), dtype=torch.float32
                )
            else:
                router_logits = topk_output.router_logits.float()
            topk_output = topk_output._replace(router_logits=router_logits)
            routing_fields = ["router_logits"]
        elif TopKOutputChecker.format_is_packed(topk_output):
            routing_fields = ["packed_topk_ids"]
        else:
            assert TopKOutputChecker.format_is_standard(topk_output)
            routing_fields = ["topk_weights", "topk_ids"]
            if hasattr(topk_output, "packed_topk_ids"):
                routing_fields.append("packed_topk_ids")
        payloads.extend(getattr(topk_output, name) for name in routing_fields)
        gathered = get_tp_group().all_gatherv(
            payloads, sizes=get_dp_global_num_tokens()
        )
        x = gathered[0]
        if x_sf is not None:
            x_sf = gathered[1]
        if per_token_scale is not None:
            per_token_scale = gathered[2]
        routing = dict(zip(routing_fields, gathered[routing_offset:]))
        if TopKOutputChecker.format_is_bypassed(topk_output):
            # Keep routing inside the non-routed TRT-LLM kernel. Its logits
            # must have the same global token order as the packed activations.
            topk_output = topk_output._replace(
                **routing, hidden_states=x, num_token_non_padded=None
            )
        else:
            topk_output = topk_output._replace(**routing, router_logits=None)
        # Communicate linear block scales; only CUTLASS needs a swizzle.
        if x_sf is not None:
            if self.enable_flashinfer_cutlass_moe:
                x_sf = nvfp4_block_scale_interleave_flashinfer(x_sf)
            else:
                x_sf = x_sf.view(torch.float8_e4m3fn)
        return x, x_sf, per_token_scale, topk_output

    def combine(self, combine_input: StandardCombineInput) -> torch.Tensor:
        (hidden_states,) = combine_input
        if should_use_flashinfer_moe_fp4_allgather():
            global_hidden_states = hidden_states
            group = get_tp_group()
            # Latent MoE outputs can be narrower than the model's DP buffer.
            with use_symmetric_memory(group, disabled=not is_dp_max_padding()):
                hidden_states = global_hidden_states.new_empty(
                    (get_local_dp_buffer_len(), *global_hidden_states.shape[1:])
                )
            group.reduce_scatterv(
                global_hidden_states,
                output=hidden_states,
                sizes=get_dp_global_num_tokens(),
            )
        return hidden_states
