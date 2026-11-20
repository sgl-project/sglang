"""FlashInfer RDMA dispatch/combine for expert parallel decode.

ModelOpt NVFP4 experts consume received packed activations directly. Native
MXFP4 experts retain their checkpoint format. They accept BF16/NVFP4 wire
activations or prequantized MXFP8 data/scales through the same CUTLASS runner.
Dispatch precision is independent of combine and weight precision. Eager
prefill uses BF16 gather/compute/reduce-scatter to avoid
reserving an expert-major RDMA workspace for the much larger prefill batch.
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    get_dp_global_num_tokens,
    get_is_extend_in_batch,
)
from sglang.srt.layers.moe.token_dispatcher.base import BaseDispatcher
from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
    FlashinferDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker
from sglang.srt.runtime_context import get_parallel, get_resources, get_spec

_BUFFER_KEY = "flashinfer_rdma_alltoall"
logger = logging.getLogger(__name__)


def _validate_parallel_topology(
    parallel, *, group_ranks, group_rank, tp_ranks, ep_ranks
):
    """Each rank must own distinct tokens and a distinct expert partition."""
    ranks = tuple(group_ranks)
    ep = len(ranks)
    if ep < 2 or len(set(ranks)) != ep:
        raise ValueError("FlashInfer RDMA requires at least two distinct ranks")
    expected_widths = {
        "tp_size": ep,
        "attn_dp_size": ep,
        "attn_tp_size": 1,
        "attn_cp_size": 1,
        "attn_dcp_size": 1,
        "moe_ep_size": ep,
        "moe_tp_size": 1,
        "moe_dp_size": 1,
    }
    for name, expected in expected_widths.items():
        actual = getattr(parallel, name)
        if actual != expected:
            raise ValueError(
                f"FlashInfer RDMA requires TP=attention DP=EP={ep}, "
                f"attention TP/CP/DCP and MoE TP/DP=1; "
                f"{name}={actual}, expected {expected}"
            )
    # Prefill gathers in TP order, while expert weights are partitioned by EP.
    if tuple(tp_ranks) != ranks or tuple(ep_ranks) != ranks:
        raise ValueError(
            "FlashInfer RDMA, TP and EP groups must have identical ordered membership"
        )
    if not 0 <= group_rank < ep:
        raise ValueError("Invalid FlashInfer RDMA group rank")
    for name in ("tp_rank", "attn_dp_rank", "moe_ep_rank"):
        if getattr(parallel, name) != group_rank:
            raise ValueError(
                f"FlashInfer RDMA token and expert ownership disagree: {name}"
            )


@dataclass
class _RdmaState:
    buffer: Any
    outputs: tuple[torch.Tensor, ...]
    packed: torch.Tensor
    packed_sf: torch.Tensor
    ids: torch.Tensor
    weights: torch.Tensor
    slot_map: torch.Tensor
    expert_output: torch.Tensor
    combine_scales: torch.Tensor
    compact_output: torch.Tensor
    owner: object | None = None


def _shared_state(
    group, *, hidden, num_experts, topk, capacity, dispatch_dtype, combine_dtype
):
    key = (
        _BUFFER_KEY,
        id(group),
        hidden,
        num_experts,
        topk,
        capacity,
        dispatch_dtype,
        combine_dtype,
    )
    resources = get_resources().buffers
    if key in resources:
        return resources[key]
    try:
        from flashinfer.comm.rdma_moe_alltoall import (
            RdmaMoeAlltoAll,
            RdmaMoeAlltoAllConfig,
        )
    except ImportError as exc:
        raise ImportError(
            "FlashInfer RDMA all-to-all is unavailable. Install FlashInfer with "
            "RdmaMoeAlltoAll support (flashinfer-ai/flashinfer#9449) and its "
            "documented NCCL/GDAKI dependencies."
        ) from exc

    ep = group.size()
    local_experts = num_experts // ep
    cap = ep * capacity
    rows = cap * topk
    width = hidden // 2 if dispatch_dtype == "nvfp4" else hidden
    sf_width = {"nvfp4": hidden // 16, "mxfp8": hidden // 32, "bf16": 0}[dispatch_dtype]
    dtype = torch.bfloat16 if dispatch_dtype == "bf16" else torch.uint8
    config_kwargs = {"communication_dtype": combine_dtype}
    if dispatch_dtype == "mxfp8":
        from flashinfer.comm.rdma_moe_alltoall import RdmaDispatchLayout

        config_kwargs["dispatch_layout"] = RdmaDispatchLayout(hidden, hidden // 32)
    elif dispatch_dtype != combine_dtype:
        config_kwargs["dispatch_dtype"] = dispatch_dtype
    device = torch.device("cuda", torch.cuda.current_device())
    buffer = RdmaMoeAlltoAll(
        group,
        num_max_dispatch_tokens_per_rank=capacity,
        hidden=hidden,
        num_experts=num_experts,
        config=RdmaMoeAlltoAllConfig(**config_kwargs),
    )
    state = _RdmaState(
        buffer=buffer,
        outputs=(
            torch.empty((local_experts, cap, width), dtype=dtype, device=device),
            torch.empty(
                (local_experts, cap, sf_width), dtype=torch.uint8, device=device
            ),
            torch.empty(local_experts, dtype=torch.int32, device=device),
            torch.empty((local_experts, cap), dtype=torch.int32, device=device),
            torch.empty((local_experts, ep), dtype=torch.int64, device=device),
        ),
        # Invalid compact rows retain payload; their expert IDs mask them out.
        packed=torch.zeros((rows, width), dtype=dtype, device=device),
        packed_sf=torch.zeros((rows, sf_width), dtype=torch.uint8, device=device),
        ids=torch.empty((rows, 1), dtype=torch.int32, device=device),
        weights=torch.empty((rows, 1), dtype=torch.float32, device=device),
        slot_map=torch.empty(rows, dtype=torch.int32, device=device),
        expert_output=torch.empty(
            (local_experts, cap, hidden), dtype=torch.bfloat16, device=device
        ),
        combine_scales=torch.ones(
            (local_experts, cap, 1), dtype=torch.float32, device=device
        ),
        compact_output=torch.empty((rows, hidden), dtype=torch.bfloat16, device=device),
    )
    resources[key] = state
    return state


class FlashinferRdmaDispatcher(BaseDispatcher):
    """Serial RDMA dispatch/combine with full decode CUDA graphs.

    ``max_tokens_per_rank`` counts physical token rows, including all verify
    tokens when speculative decoding is enabled. Layers share one workspace
    per geometry; their dispatch/compute/combine pairs must not interleave.
    """

    def __init__(
        self,
        group,
        router_topk,
        num_experts,
        num_local_experts,
        hidden_size,
        moe_runner_config,
        max_tokens_per_rank,
        params_dtype=None,
    ):
        super().__init__()
        self.native_mxfp4 = moe_runner_config.runner_backend.is_flashinfer_mxfp4()
        if not (
            self.native_mxfp4
            or moe_runner_config.runner_backend.is_flashinfer_cutlass()
        ):
            raise ValueError("FlashInfer RDMA requires a supported CUTLASS MoE runner")
        if moe_runner_config.apply_router_weight_on_input:
            raise ValueError(
                "FlashInfer RDMA applies routing weights after expert compute"
            )
        parallel = get_parallel()
        _validate_parallel_topology(
            parallel,
            group_ranks=dist.get_process_group_ranks(group),
            group_rank=group.rank(),
            tp_ranks=dist.get_process_group_ranks(parallel.tp_group.device_group),
            ep_ranks=dist.get_process_group_ranks(parallel.moe_ep_group.device_group),
        )
        if num_experts % group.size() or not 1 <= router_topk <= 8:
            raise ValueError(
                "FlashInfer RDMA requires divisible expert count and top-k in 1..8"
            )
        if num_local_experts != num_experts // group.size():
            raise ValueError("FlashInfer RDMA requires separate shared expert compute")
        if max_tokens_per_rank < 1:
            raise ValueError("FlashInfer RDMA token capacity must be positive")
        self.group = group
        self.tp_group = parallel.tp_group
        self.ep_rank = group.rank()
        self.num_experts = num_experts
        self.topk = router_topk
        self.hidden = hidden_size
        self.capacity = max_tokens_per_rank
        # Published DP sizes need not bound per-rank verify widths in these modes.
        self.rows_follow_batch = (
            envs.SGLANG_RAGGED_VERIFY_MODE.get() == "static"
            and not get_spec().speculative_adaptive
        )
        self.wire_dtype = envs.SGLANG_FLASHINFER_RDMA_WIRE_DTYPE.get()
        if self.wire_dtype not in ("nvfp4", "bf16"):
            raise ValueError("FlashInfer RDMA wire dtype must be nvfp4 or bf16")
        self.dispatch_dtype = (
            envs.SGLANG_FLASHINFER_RDMA_DISPATCH_DTYPE.get() or self.wire_dtype
        )
        self.combine_dtype = self.wire_dtype
        if self.dispatch_dtype not in ("nvfp4", "bf16", "mxfp8"):
            raise ValueError(
                "FlashInfer RDMA dispatch dtype must be nvfp4, bf16 or mxfp8"
            )
        if not self.native_mxfp4 and self.dispatch_dtype != "nvfp4":
            raise ValueError("ModelOpt NVFP4 experts require NVFP4 dispatch")
        if self.dispatch_dtype == "mxfp8" and hidden_size % 32:
            raise ValueError("MXFP8 dispatch requires hidden divisible by 32")
        self.wire_global_scale = None
        if self.native_mxfp4:
            if self.dispatch_dtype == "nvfp4":
                scale = envs.SGLANG_FLASHINFER_RDMA_NVFP4_GLOBAL_SCALE.get()
                if scale is None or not math.isfinite(scale) or scale <= 0:
                    raise ValueError(
                        "Native MXFP4 + NVFP4 wire requires an explicit positive finite "
                        "SGLANG_FLASHINFER_RDMA_NVFP4_GLOBAL_SCALE"
                    )
                try:
                    self.wire_global_scale = struct.unpack(
                        "f", struct.pack("f", scale)
                    )[0]
                except OverflowError as exc:
                    raise ValueError(
                        "RDMA wire global scale overflows float32"
                    ) from exc
                if (
                    not math.isfinite(self.wire_global_scale)
                    or self.wire_global_scale <= 0
                ):
                    raise ValueError(
                        "RDMA wire global scale must be positive finite float32"
                    )
            # Fail before creating differently sized NCCL windows if ranks disagree.
            contract = torch.tensor(
                [
                    {"bf16": 0, "nvfp4": 1, "mxfp8": 2}[self.dispatch_dtype],
                    int(self.combine_dtype == "nvfp4"),
                    self.wire_global_scale or 0.0,
                ],
                dtype=torch.float32,
                device="cuda",
            )
            low, high = contract.clone(), contract.clone()
            dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
            dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
            if not torch.equal(low, high):
                raise ValueError(
                    "FlashInfer RDMA wire dtype/global scale disagree across ranks"
                )
            # The native kernel receives float32, and dequantization must use
            # exactly that same rounded scalar.
            self.wire_global_scale = (
                float(low[2].item()) if self.dispatch_dtype == "nvfp4" else None
            )
            if self.wire_global_scale is not None and self.wire_global_scale == 0:
                raise ValueError("FlashInfer RDMA wire global scale underflows float32")
            if self.ep_rank == 0:
                logger.info(
                    "Native MXFP4 RDMA: dispatch=%s, combine=%s, NVFP4 global_scale=%s, "
                    "NVFP4 nonzero block-scale underflow at block_max<=%s, "
                    "scale saturation above block_max=%s; expert input=%s, "
                    "checkpoint MXFP4 weights unchanged",
                    self.dispatch_dtype,
                    self.combine_dtype,
                    self.wire_global_scale,
                    None
                    if self.wire_global_scale is None
                    else 3 / (512 * self.wire_global_scale),
                    None
                    if self.wire_global_scale is None
                    else 2688 / self.wire_global_scale,
                    "MXFP8" if self.dispatch_dtype == "mxfp8" else "BF16->MXFP8",
                )
        self.state = _shared_state(
            group,
            hidden=hidden_size,
            num_experts=num_experts,
            topk=router_topk,
            capacity=max_tokens_per_rank,
            dispatch_dtype=self.dispatch_dtype,
            combine_dtype=self.combine_dtype,
        )
        self.packed_global_scales = (
            torch.full(
                (self.state.packed.shape[0], 1),
                self.wire_global_scale,
                dtype=torch.float32,
                device=self.state.packed.device,
            )
            if self.native_mxfp4 and self.dispatch_dtype == "nvfp4"
            else None
        )
        self._pending = None

    def set_quant_config(self, quant_config):
        super().set_quant_config(quant_config)
        if self.native_mxfp4:
            return
        scale = quant_config.get("input_global_scale")
        if (
            not isinstance(scale, torch.Tensor)
            or scale.numel() != 1
            or scale.dtype != torch.float32
        ):
            raise ValueError(
                "FlashInfer RDMA requires one static FP32 NVFP4 activation scale"
            )
        # The wire carries block scales, but not the checkpoint's global scale.
        # Check its cross-rank contract once, before CUDA graph capture.
        extrema = scale.detach().reshape(1).repeat(2)
        dist.all_reduce(extrema[:1], op=dist.ReduceOp.MIN, group=self.group)
        dist.all_reduce(extrema[1:], op=dist.ReduceOp.MAX, group=self.group)
        minimum, maximum = extrema.cpu().tolist()
        if not (0 < minimum < float("inf")) or minimum != maximum:
            raise ValueError(
                f"FlashInfer RDMA activation scales disagree: min={minimum}, max={maximum}"
            )

    def set_overlap_args(self, combine_overlap_args, meta_overlap_args):
        raise ValueError("FlashInfer RDMA requires serial dispatch/compute/combine")

    def _live_rows(self, tokens):
        """Bound compact compute rows by this forward's published DP sizes.

        Each source row can reach at most top-k experts. CUDA graph capture
        publishes padded sizes for the bucket, keeping this bound replay-safe.
        Keep the full workspace if the metadata does not describe this forward.
        """
        full = self.state.packed.shape[0]
        sizes = get_dp_global_num_tokens()
        if (
            not self.rows_follow_batch
            or sizes is None
            or len(sizes) != self.group.size()
            or sizes[self.ep_rank] != tokens
        ):
            return full
        return min(max(int(sum(sizes)), 1) * self.topk, full)

    def dispatch(self, hidden_states, topk_output):
        if self._pending is not None or self.state.owner is not None:
            raise RuntimeError(
                "FlashInfer RDMA dispatch/combine pairs must not interleave"
            )
        if (
            hidden_states.ndim != 2
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.shape[1] != self.hidden
        ):
            raise ValueError("FlashInfer RDMA expects BF16 [tokens, hidden] inputs")
        if TopKOutputChecker.format_is_bypassed(topk_output):
            topk_output = topk_output.to_standard()
        ids = topk_output.topk_ids.to(torch.int32).contiguous()
        weights = topk_output.topk_weights.to(torch.float32).contiguous()
        if (
            ids.shape != (hidden_states.shape[0], self.topk)
            or weights.shape != ids.shape
        ):
            raise ValueError("Unexpected FlashInfer RDMA router top-k shapes")
        hidden_states = hidden_states.contiguous()
        if get_is_extend_in_batch():
            return self._prefill_dispatch(hidden_states, ids, weights)
        if hidden_states.shape[0] > self.capacity:
            raise ValueError(
                "Decode/verify token rows exceed --flashinfer-rdma-max-tokens-per-rank"
            )
        from sglang.srt.layers.moe.token_dispatcher.flashinfer_rdma_kernels import (
            pack_expert_rows,
        )

        state = self.state
        if self.dispatch_dtype == "mxfp8":
            from flashinfer import mxfp8_quantize

            if hidden_states.shape[0]:
                x, sf = mxfp8_quantize(
                    hidden_states, is_sf_swizzled_layout=False, alignment=32
                )
                x = x.view(torch.uint8)
                sf = sf.view(torch.uint8).reshape(
                    hidden_states.shape[0], self.hidden // 32
                )
            else:
                x = state.packed[:0]
                sf = state.packed_sf[:0]
            recv, recv_sf, counts, handle, _ = state.buffer.dispatch(
                x,
                ids,
                x_scales=sf,
                input_dtype="raw",
                out=state.outputs,
                return_recv_hook=False,
            )
        elif self.native_mxfp4:
            recv, recv_sf, counts, handle, _ = state.buffer.dispatch(
                hidden_states,
                ids,
                input_dtype="bf16",
                nvfp4_global_scale=self.wire_global_scale or 1.0,
                out=state.outputs,
                return_recv_hook=False,
            )
        else:
            from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

            if hidden_states.shape[0]:
                x, sf = fp4_quantize(
                    hidden_states,
                    self.quant_config["input_global_scale"],
                    is_sf_swizzled_layout=False,
                )
            else:
                x = torch.empty(
                    (0, self.hidden // 2),
                    dtype=torch.uint8,
                    device=hidden_states.device,
                )
                sf = torch.empty(
                    (0, self.hidden // 16),
                    dtype=torch.uint8,
                    device=hidden_states.device,
                )
            recv, recv_sf, counts, handle, _ = state.buffer.dispatch(
                x,
                ids,
                x_scales=sf,
                input_dtype="prequantized",
                out=state.outputs,
                return_recv_hook=False,
            )
        rows = self._live_rows(hidden_states.shape[0])
        packed = state.packed[:rows]
        packed_sf = state.packed_sf[:rows]
        compact_ids = state.ids[:rows]
        compact_weights = state.weights[:rows]
        pack_expert_rows(
            recv,
            recv_sf,
            counts,
            packed,
            packed_sf,
            compact_ids,
            compact_weights,
            state.slot_map[:rows],
            ep_rank=self.ep_rank,
            num_experts=self.num_experts,
        )
        if self.dispatch_dtype == "mxfp8":
            from flashinfer import nvfp4_block_scale_interleave

            packed = packed.view(torch.float8_e4m3fn)
            # MXFP8 UE8M0 bytes use the same 128x4 scale permutation as NVFP4.
            packed_sf = nvfp4_block_scale_interleave(packed_sf)
        elif self.native_mxfp4:
            if self.dispatch_dtype == "nvfp4":
                packed = state.buffer.dequantize_nvfp4_to_bf16(
                    packed, self.packed_global_scales[:rows], packed_sf
                )
            packed_sf = None
        else:
            from flashinfer import nvfp4_block_scale_interleave

            packed_sf = nvfp4_block_scale_interleave(packed_sf)
        self._pending = ("decode", ids, weights, handle, hidden_states.shape[0], counts)
        state.owner = self
        # Compact rows use unit weights. Original routing weights are applied
        # exactly once, by combine after expert computation.
        return FlashinferDispatchOutput(
            packed,
            packed_sf,
            StandardTopKOutput(compact_weights, compact_ids, None),
            state.compact_output[:rows],
            torch.bfloat16,
        )

    def _prefill_dispatch(self, hidden, ids, weights):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "FlashInfer RDMA requires eager prefill; disable prefill CUDA graphs"
            )
        sizes = get_dp_global_num_tokens()
        if (
            sizes is None
            or len(sizes) != self.group.size()
            or sizes[self.ep_rank] != hidden.shape[0]
        ):
            raise ValueError(
                "Prefill DP token counts do not match FlashInfer RDMA ownership"
            )
        x, global_ids, global_weights = self.tp_group.all_gatherv(
            [hidden, ids, weights], sizes=sizes
        )
        # The CUTLASS A2A finalize skips rows without local experts. Those rows
        # must contribute zeros to the partial-output reduction.
        output = torch.zeros(
            (x.shape[0], self.hidden), dtype=torch.bfloat16, device=x.device
        )
        self._pending = ("prefill", list(sizes))
        self.state.owner = self
        return FlashinferDispatchOutput(
            x,
            None,
            StandardTopKOutput(global_weights, global_ids, None),
            output,
            torch.bfloat16,
        )

    def combine(self, combine_input):
        if self._pending is None or self.state.owner is not self:
            raise RuntimeError("FlashInfer RDMA combine requires a preceding dispatch")
        pending = self._pending
        if pending[0] == "prefill":
            output = self.tp_group.reduce_scatterv(
                combine_input.hidden_states, sizes=pending[1]
            )
        else:
            from sglang.srt.layers.moe.token_dispatcher.flashinfer_rdma_kernels import (
                unpack_expert_rows,
            )

            _, ids, weights, handle, tokens, counts = pending
            state = self.state
            unpack_expert_rows(
                combine_input.hidden_states,
                state.slot_map,
                counts,
                state.expert_output,
                state.combine_scales,
            )
            output = torch.empty(
                (tokens, self.hidden), dtype=torch.bfloat16, device=ids.device
            )
            output, _ = state.buffer.combine(
                state.expert_output,
                handle,
                topk_idx=ids,
                topk_weights=weights,
                global_scale=state.combine_scales
                if self.combine_dtype == "nvfp4"
                else None,
                out=output,
                return_recv_hook=False,
            )
        self._pending = None
        self.state.owner = None
        return output


def close_flashinfer_rdma_buffers():
    """Collectively release workspaces after retiring graphs, before groups."""
    resources = get_resources().buffers
    keys = [
        key for key in resources if isinstance(key, tuple) and key[0] == _BUFFER_KEY
    ]
    if not keys:
        return
    torch.cuda.synchronize()
    for key in keys:
        state = resources[key]
        if state.owner is not None:
            raise RuntimeError("Cannot close FlashInfer RDMA during dispatch/combine")
        state.buffer.close()
        del resources[key]
