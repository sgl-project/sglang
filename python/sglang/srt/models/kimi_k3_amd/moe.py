"""Kimi-K3's latent MoE on ROCm.

Shared helpers are imported inside methods to avoid a cycle with ``kimi_k3``,
which imports this mixin. ROCm keeps the multicast all-reduce fusion disabled.
"""

from __future__ import annotations

from functools import cache, cached_property
from typing import TYPE_CHECKING, Callable, Optional

import msgspec
import torch

from sglang.srt.distributed import tensor_model_parallel_all_reduce
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.environ import envs
from sglang.srt.layers.activation import SituAndMul
from sglang.srt.layers.communication import k3_ar_fusion
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.layer_boundary.residual.attn_bank import AttnBank
from sglang.srt.layers.layer_boundary.residual.attn_bank_hip import row_shard
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.moe.topk import TopKOutputChecker
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.srt.utils.common import cdiv, get_bool_env_var, print_info_once

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.k3_ll_comm_hip import LLComm

_is_gfx95 = is_gfx95_supported()
_use_aiter = is_hip() and get_bool_env_var("SGLANG_USE_AITER")
if _is_gfx95:
    from sglang.kernels.ops.attention import k3_attn_res_read_hip
    from sglang.kernels.ops.gemm import kimi_k3_latent_tail_hip
    from sglang.srt.layers import attn_residual_hip
# The all-gather read's row limit; up to it the column tail's split-K partial GEMM
# beats the replicated tail on MI350X.
_COLUMN_TAIL_MAX_TOKENS = 64
# From this many tokens each rank computes its 1/TP routed up_proj columns into the
# shared-expert all-reduce.
_UP_PROJ_COLUMN_SPLIT_MIN_TOKENS = 2048
_ROW_SHARDED_TAIL_MIN_TOKENS = 4096
# Crossover vs AITER's A8W4 fused_moe on one MI350X: even at 1K tokens, faster from 2K.
_A8W4_MOE_MIN_TOKENS = 2048


class NextAttnRead(msgspec.Struct, frozen=True):
    """The next layer's attention read of the stream a latent MoE writes, which
    the column-parallel tail's all-gather runs ahead (gfx950 TP8)."""

    bank: AttnBank
    score_proj: ReplicatedLinear
    score_norm: RMSNorm
    norm: RMSNorm
    writes_block: bool
    comm: LLComm


@cache
def _hip_moe_side_stream(device: torch.device) -> torch.cuda.Stream:
    # One per device, shared by every MoE layer: layers run in order, and each
    # joins the side stream back before its tail.
    return torch.cuda.Stream(device=device)


def _run_on_side_stream(
    side_stream: Optional[torch.cuda.Stream], collective: Callable[[], None]
) -> None:
    """collective() on the MoE side stream after the current stream's work, or
    in place without one (graph capture); the caller joins the side stream back."""
    if side_stream is None:
        collective()
        return
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        collective()


def _eager_side_stream() -> Optional[torch.cuda.Stream]:
    """The MoE side stream in eager batches; None under (breakable) graph capture."""
    if torch.cuda.is_current_stream_capturing() or is_in_breakable_cuda_graph():
        return None
    return _hip_moe_side_stream(torch.cuda.current_stream().device)


def _vector_aligned(t: torch.Tensor) -> bool:
    """Rows the SiTU launch's cast can read as 16-byte vectors."""
    return (
        t.dim() == 2
        and t.stride(1) == 1
        and t.data_ptr() % 16 == 0
        and t.stride(0) * t.element_size() % 16 == 0
    )


class KimiK3MoEHipMixin:
    """KimiK3MoE's ROCm paths."""

    # Set post-load by _prepare_hip_paths; the defaults keep every path off.
    # This rank's columns of the replicated routed up_proj, used for batches of at
    # least _UP_PROJ_COLUMN_SPLIT_MIN_TOKENS tokens.
    _prefill_up_proj_cols: Optional[slice] = None
    _defer_routed_cast_eligible = False
    _fused_latent_tail_eligible = False
    # Set by link_column_tails_hip when the next layer's attention read can ride
    # the column-parallel tail's all-gather.
    next_attn_read: Optional[NextAttnRead] = None
    # Whether shard_front runs this layer's front on row shards; the full shared-expert
    # [gate; up] and down weights it uses (gathered post-load; None keeps the shared
    # experts TP-sharded); and what shard_front gathered for the forward after it.
    _row_front_on = False
    _row_front_shared: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    _row_front_out: Optional[tuple] = None
    # Set by link_row_sharded_stream_hip when every later layer reads a stream
    # left on row shards (SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM, PP1).
    _row_stream = False

    def _prepare_hip_paths(self) -> None:
        """Post-load, after _merge_front_weights: the ROCm paths this layer takes."""
        self._select_prefill_up_proj_columns()
        self._build_row_sharded_front()
        # fp32 front with a SiTU shared expert: the routed input's bf16 cast can
        # ride the shared activation's launch instead of its own.
        shared = self.shared_experts
        self._defer_routed_cast_eligible = (
            self._front_fp32
            and shared is not None
            and isinstance(shared.act_fn, SituAndMul)
        )
        self._fused_latent_tail_eligible = (
            _is_gfx95
            and self.routed_expert_norm is not None
            and self.routed_expert_up_proj is not None
            and self.routed_expert_up_proj.weight.dtype == torch.bfloat16
        )

    def _select_prefill_up_proj_columns(self) -> None:
        """Prefill-sized batches compute only this rank's 1/tp columns of the
        routed up_proj; the shared-expert all-reduce carries them."""
        weight = self.routed_expert_up_proj.weight if self.use_latent_moe else None
        if not (
            self.tp_size > 1
            and self._eligible_for_fused_front
            and isinstance(weight, torch.Tensor)
            and weight.dtype == torch.bfloat16
            and weight.shape[0] % self.tp_size == 0
        ):
            return
        rows = weight.shape[0] // self.tp_size
        rank = get_parallel().tp_rank
        self._prefill_up_proj_cols = slice(rank * rows, (rank + 1) * rows)

    def _forward_fused_hip(
        self, hidden_states: torch.Tensor, *, prefix_sum: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """KimiK3MoE._forward_fused on ROCm: the merged front, both TP-partial sums
        in one flat [latent | shared] buffer, then the tail the batch takes."""
        from sglang.srt.models.kimi_k3 import _add3

        num_tokens, hidden_size = hidden_states.shape
        # The row-sharded tail also shards the up_proj (by rows), so it replaces the
        # column split where both apply.
        row_sharded = self._row_sharded_tail_eligible(
            num_tokens, prefix_sum is not None
        )
        up_proj_cols = (
            self._prefill_up_proj_cols
            if num_tokens >= _UP_PROJ_COLUMN_SPLIT_MIN_TOKENS and not row_sharded
            else None
        )
        # fp32 front, single-collective tail: the shared SiTU launch also casts the
        # routed input to bf16 (one launch a MoE layer fewer)
        defer_cast = self._defer_routed_cast_eligible and up_proj_cols is None
        gate_up, router_logits, routed_input = self._fused_front(
            hidden_states, cast_routed=not defer_cast
        )
        if defer_cast and not _vector_aligned(routed_input):
            defer_cast = False
            routed_input = routed_input.to(hidden_states.dtype)

        # The row-sharded tail pads the rows to a multiple of TP for equal
        # reduce-scatter shards.
        rows = (
            cdiv(num_tokens, self.tp_size) * self.tp_size if row_sharded else num_tokens
        )
        if row_sharded:
            buf = hidden_states.new_empty(rows * (self.moe_hidden_size + hidden_size))
        else:
            with use_symmetric_memory(
                get_parallel().tp_group, disabled=not is_allocation_symmetric()
            ):
                buf = hidden_states.new_empty(
                    rows * (self.moe_hidden_size + hidden_size)
                )
        latent_numel = rows * self.moe_hidden_size
        latent_rows = buf[:latent_numel].view(rows, self.moe_hidden_size)
        shared_rows = buf[latent_numel:].view(rows, hidden_size)
        latent, shared_output = latent_rows[:num_tokens], shared_rows[:num_tokens]

        if up_proj_cols is not None:
            # the shared experts run on a side stream under the routed ones
            current_stream = torch.cuda.current_stream()
            side_stream = _hip_moe_side_stream(current_stream.device)
            side_stream.wait_stream(current_stream)
            with torch.cuda.stream(side_stream):
                self._forward_shared(gate_up, shared_output)
            self._forward_routed(hidden_states, router_logits, routed_input, latent)
            current_stream.wait_stream(side_stream)
            return self._column_split_tail(
                latent, shared_output, prefix_sum, cols=up_proj_cols
            )
        if defer_cast:
            routed_input = self._forward_shared_casting_routed(
                gate_up, shared_output, routed_input
            )
        else:
            self._forward_shared(gate_up, shared_output)
        self._forward_routed(hidden_states, router_logits, routed_input, latent)
        if row_sharded:
            return self._row_sharded_tail(
                latent_rows, shared_rows, num_tokens, prefix_sum
            )
        buf = self._reduce_latent_and_shared(buf)

        latent = buf[:latent_numel].view(num_tokens, self.moe_hidden_size)
        shared_output = buf[latent_numel:].view(num_tokens, hidden_size)
        if self.next_attn_read is not None:
            stream = self._column_tail_and_next_read(latent, shared_output, prefix_sum)
            if stream is not None:
                return stream
        if (
            self._fused_latent_tail_eligible
            and kimi_k3_latent_tail_hip.supports_latent_tail(
                latent,
                self.routed_expert_up_proj.weight,
                shared_output,
                prefix_sum,
            )
        ):
            return kimi_k3_latent_tail_hip.k3_latent_tail(
                latent,
                self.routed_expert_norm.weight,
                self.routed_expert_norm.variance_epsilon,
                self.routed_expert_up_proj.weight,
                shared_output,
                prefix_sum,
            )
        out, _ = self.routed_expert_up_proj(self._latent_norm(latent))

        return _add3(out, shared_output, prefix_sum, prefetch_bc=True)

    def _forward_shared_casting_routed(
        self,
        gate_up: torch.Tensor,
        shared_output: torch.Tensor,
        routed_input: torch.Tensor,
    ) -> torch.Tensor:
        """KimiK3MoE._forward_shared with the fp32 routed input's bf16 cast riding
        the SiTU launch; returns the cast routed input."""
        from sglang.kernels.ops.activation import situ_and_mul
        from sglang.srt.models.kimi_k3 import _k3_bf16_gemm

        routed_bf16 = torch.empty(
            routed_input.shape, dtype=shared_output.dtype, device=routed_input.device
        )
        shared = self.shared_experts
        act = situ_and_mul(
            gate_up,
            None,
            shared.act_fn.beta,
            shared.act_fn.linear_beta,
            (routed_input, routed_bf16),
        )
        _k3_bf16_gemm(act, shared.down_proj.weight, out=shared_output)
        return routed_bf16

    def _column_split_tail(
        self,
        latent: torch.Tensor,
        shared_output: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
        cols: slice,
    ) -> torch.Tensor:
        """Latent all-reduce and norm, this rank's up_proj columns added into its
        columns of the TP-partial shared sum, then one all-reduce carries both."""
        from sglang.srt.models.kimi_k3 import _k3_bf16_gemm

        latent = self._latent_norm(tensor_model_parallel_all_reduce(latent))
        shared_output[:, cols] += _k3_bf16_gemm(
            latent, self.routed_expert_up_proj.weight[cols]
        )
        out = tensor_model_parallel_all_reduce(shared_output)
        return out if prefix_sum is None else out + prefix_sum

    def _row_sharded_tail_eligible(self, num_tokens: int, has_prefix: bool) -> bool:
        """Whether the tail runs on this rank's 1/TP of the rows (gfx950 prefill
        batches; see _row_sharded_tail)."""
        return (
            _is_gfx95
            and num_tokens >= _ROW_SHARDED_TAIL_MIN_TOKENS
            and self.tp_size > 1
            and has_prefix
            and not k3_ar_fusion.enabled()
            and self.routed_expert_up_proj is not None
        )

    def takes_row_sharded_tail(self, num_tokens: int) -> bool:
        """Whether a batch of num_tokens runs _row_sharded_tail, which then
        also takes the written residual on this rank's padded rows."""
        return self._eligible_for_fused_front and self._row_sharded_tail_eligible(
            num_tokens, True
        )

    def _row_sharded_tail(
        self,
        latent_rows: torch.Tensor,
        shared_rows: torch.Tensor,
        num_tokens: int,
        prefix_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce-scatter the [latent | shared] partials by rows, run the latent
        norm, up-projection and shared + residual add on this rank's 1/TP of
        the rows, and all-gather the stream. latent_rows / shared_rows hold
        num_tokens rows padded to a multiple of TP; prefix_sum is the residual
        on every row or already on this rank's padded rows."""
        tp_group = get_parallel().tp_group
        padded_rows, hidden_size = shared_rows.shape
        rows = padded_rows // self.tp_size
        if padded_rows > num_tokens:
            latent_rows[num_tokens:].zero_()
            shared_rows[num_tokens:].zero_()
        latent = latent_rows.new_empty(rows, latent_rows.shape[1])
        shared = shared_rows.new_empty(rows, hidden_size)
        tp_group.reduce_scatter_tensor(latent, latent_rows)
        tp_group.reduce_scatter_tensor(shared, shared_rows)
        return self._row_tail(latent, shared, num_tokens, prefix_sum)

    def _row_tail(
        self,
        latent: torch.Tensor,
        shared: torch.Tensor,
        num_tokens: int,
        prefix_sum: torch.Tensor,
    ) -> torch.Tensor:
        """The latent norm, up-projection and shared + residual add on this
        rank's rows of the complete latent and shared sums, then the all-gather
        of the stream, or the stream left on this rank's rows."""
        from sglang.srt.models.kimi_k3 import _add3

        rows, hidden_size = shared.shape
        padded_rows = rows * self.tp_size
        tp_rank = get_parallel().tp_rank
        start = tp_rank * rows
        if prefix_sum.shape[0] == num_tokens:
            prefix = prefix_sum[start : min(start + rows, num_tokens)]
        else:
            assert prefix_sum.shape[0] == rows
            prefix = prefix_sum
        if prefix.shape[0] < rows:
            prefix = torch.cat(
                [prefix, prefix.new_zeros(rows - prefix.shape[0], hidden_size)]
            )
        out, _ = self.routed_expert_up_proj(self._latent_norm(latent))
        stream_rows = _add3(out, shared, prefix)
        if self._row_stream:
            # the next read takes this rank's rows (attn_bank._bank_rows)
            return stream_rows[: row_shard(num_tokens, self.tp_size, tp_rank)[1]]
        stream = stream_rows.new_empty(padded_rows, hidden_size)
        get_parallel().tp_group.all_gather_into_tensor(stream, stream_rows)
        return stream[:num_tokens]

    def _build_row_sharded_front(self) -> None:
        """Post-load on gfx950: turn on shard_front and, for the full front, have
        every rank gather the full shared-expert weights from the TP shards."""
        full = envs.SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_FRONT.get()
        if not (
            (full or envs.SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_ROUTED_FRONT.get())
            and _is_gfx95
            and self.tp_size > 1
            and self._eligible_for_fused_front
            # the merged front is [gate_up | router | latent] (not the EP pair)
            and len(self._front_sizes) == 3
        ):
            return
        self._row_front_on = True
        if not full:
            return
        group = get_parallel().tp_group.device_group
        gate_up = self.shared_experts.gate_up_proj.weight  # this rank's [gate; up]
        down = self.shared_experts.down_proj.weight  # [H, this rank's columns]
        inter, hidden_size = gate_up.shape[0] // 2, gate_up.shape[1]
        full_gate_up = gate_up.new_empty(2 * self.tp_size * inter, hidden_size)
        for i, half in enumerate(gate_up.split(inter)):
            torch.distributed.all_gather_into_tensor(
                full_gate_up[i * self.tp_size * inter : (i + 1) * self.tp_size * inter],
                half.contiguous(),
                group=group,
            )
        full_down_t = down.new_empty(self.tp_size * down.shape[1], down.shape[0])
        torch.distributed.all_gather_into_tensor(
            full_down_t, down.t().contiguous(), group=group
        )
        self._row_front_shared = (full_gate_up, full_down_t.t().contiguous())
        del full_down_t
        torch.cuda.empty_cache()

    def shard_front(self, normed: torch.Tensor, num_tokens: int) -> bool:
        """The row-sharded batch's front on this rank's padded rows of the
        normed FFN input: router top-k and latent from the merged front weight.
        Gathers the routed experts' MXFP8 input (rows quantize the same as after
        a gather) and the top-k, and with TP-sharded shared experts the normed
        input as well (on the side stream under the front GEMM); forward() then
        runs the experts and the tail. False leaves the batch to the regular path."""
        from sglang.kernels.ops.moe.k3_a8w4_gfx950 import quant_mxfp8
        from sglang.srt.models.kimi_k3 import _k3_bf16_gemm

        min_tokens = self._a8w4_moe_min_tokens
        if (
            not self._row_front_on
            # the A8W4 kernels and the latent reduce write bf16
            or normed.dtype != torch.bfloat16
            or min_tokens is None
            or not min_tokens <= num_tokens <= self._a8w4_moe_max_tokens
        ):
            return False
        tp_group = get_parallel().tp_group
        side_stream = _eager_side_stream()
        normed_all = None
        if self._row_front_shared is None:
            # the TP-sharded shared experts read every row; gathered under the front GEMM
            normed_all = normed.new_empty(
                self.tp_size * normed.shape[0], normed.shape[1]
            )
            _run_on_side_stream(
                side_stream,
                lambda: tp_group.all_gather_into_tensor(
                    normed_all, normed.contiguous()
                ),
            )
        fused = _k3_bf16_gemm(
            normed, self._front_w[self._front_sizes[0] :], out_dtype=torch.float32
        )
        router_logits, routed_input = torch.split(fused, self._front_sizes[1:], dim=-1)
        topk_output = self.topk(normed, router_logits)
        top_k = topk_output.topk_ids.shape[1]
        # ids ride bit-cast beside the weights
        topk = torch.cat(
            [
                topk_output.topk_ids.to(torch.int32).view(torch.float32),
                topk_output.topk_weights.to(torch.float32),
            ],
            dim=-1,
        )
        a, a_scale = quant_mxfp8(routed_input.to(normed.dtype))
        # one gather for the routed input, its scales and the top-k; bytes travel
        # as bf16 pairs
        topk_bytes = topk.view(torch.uint8)
        widths = (a.shape[1], a_scale.shape[1], topk_bytes.shape[1])
        packed = torch.cat([a, a_scale, topk_bytes], dim=-1)
        packed_all = packed.new_empty(self.tp_size * packed.shape[0], packed.shape[1])
        # behind the normed gather on the side stream, if any
        _run_on_side_stream(
            side_stream if normed_all is not None else None,
            lambda: tp_group.all_gather_into_tensor(
                packed_all.view(torch.bfloat16), packed.view(torch.bfloat16)
            ),
        )
        if side_stream is not None and normed_all is not None:
            torch.cuda.current_stream().wait_stream(side_stream)
        a_all, scale_all, topk_all = packed_all[:num_tokens].split(widths, dim=-1)
        topk_all = topk_all.contiguous().view(torch.float32)
        self._row_front_out = (
            (a_all.contiguous(), scale_all.contiguous()),
            topk_all[:, :top_k].contiguous().view(torch.int32),
            topk_all[:, top_k:].contiguous(),
            normed,
            normed_all,
        )
        return True

    def _forward_row_front(self, prefix_sum: torch.Tensor) -> torch.Tensor:
        """The routed experts on what shard_front gathered and the latent
        reduce-scatter (on the side stream in eager batches), then the row-sharded
        tail. The shared experts run complete on this rank's rows from their full
        weights under that reduce-scatter, or TP-sharded on every gathered row
        first, their reduce-scatter on the side stream under the routed experts."""
        from sglang.srt.models.kimi_k3 import _k3_bf16_gemm

        a_mx, topk_ids, topk_weights, normed, normed_all = self._row_front_out
        self._row_front_out = None
        num_tokens = a_mx[0].shape[0]
        rows = normed.shape[0]
        tp_group = get_parallel().tp_group
        side_stream = _eager_side_stream()
        shared = None
        if normed_all is not None:
            # fp32 gate_up into the SiTU, as the fused front feeds it
            gate_up = _k3_bf16_gemm(
                normed_all,
                self.shared_experts.gate_up_proj.weight,
                out_dtype=torch.float32,
            )
            shared_rows = _k3_bf16_gemm(
                self.shared_experts.act_fn(gate_up),
                self.shared_experts.down_proj.weight,
            )
            shared = shared_rows.new_empty(rows, shared_rows.shape[1])
            _run_on_side_stream(
                side_stream, lambda: tp_group.reduce_scatter_tensor(shared, shared_rows)
            )
        latent_rows = normed.new_empty(rows * self.tp_size, self.moe_hidden_size)
        # bf16 rows of the latent's shape; the experts read a_mx instead
        latent_shape = normed.new_empty(1, 1).expand(num_tokens, self.moe_hidden_size)
        self._run_a8w4(
            latent_shape, topk_ids, topk_weights, latent_rows[:num_tokens], a_mx=a_mx
        )
        latent_rows[num_tokens:].zero_()
        latent = latent_rows.new_empty(rows, self.moe_hidden_size)
        _run_on_side_stream(
            side_stream, lambda: tp_group.reduce_scatter_tensor(latent, latent_rows)
        )
        if shared is None:
            gate_up_w, down_w = self._row_front_shared
            gate_up = _k3_bf16_gemm(normed, gate_up_w, out_dtype=torch.float32)
            shared = _k3_bf16_gemm(self.shared_experts.act_fn(gate_up), down_w)
        if side_stream is not None:
            torch.cuda.current_stream().wait_stream(side_stream)
        return self._row_tail(latent, shared, num_tokens, prefix_sum)

    def _column_tail_and_next_read(
        self,
        latent: torch.Tensor,
        shared_output: torch.Tensor,
        prefix_sum: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """The latent tail's GEMM on this rank's 1/TP of the up_proj rows (1/TP
        of the replicated weight), finished by the all-gather of the stream,
        which also stages the next layer's attention read. None, before
        anything runs, when the kernels do not take the batch."""
        read = self.next_attn_read
        assert read is not None
        num_tokens = latent.shape[0]
        slice_width = shared_output.shape[1] // self.tp_size
        bank = read.bank.require()
        if not (
            self._fused_latent_tail_eligible
            and num_tokens <= _COLUMN_TAIL_MAX_TOKENS
            and latent.shape[1] == k3_attn_res_read_hip.TAIL_LATENT
            and latent.stride(-1) == 1
            and k3_attn_res_read_hip.tail_covered(shared_output, prefix_sum)
            and attn_residual_hip.gather_covers(
                bank, num_tokens, slice_width, read.writes_block
            )
        ):
            return None
        rank = get_parallel().tp_rank
        up_weight = self.routed_expert_up_proj.weight[
            rank * slice_width : (rank + 1) * slice_width
        ]
        part, sq = kimi_k3_latent_tail_hip.k3_latent_tail_partials(
            latent,
            self.routed_expert_norm.weight,
            up_weight,
        )
        return attn_residual_hip.gather_tail_read(
            bank,
            part,
            sq,
            shared_output,
            prefix_sum,
            self.routed_expert_norm.variance_epsilon,
            read.score_proj,
            read.score_norm,
            read.norm,
            read.writes_block,
            read.comm,
        )

    @cached_property
    def _a8w4_moe_min_tokens(self) -> Optional[int]:
        """Token count from which the routed experts run sglang's gfx950 A8W4
        grouped GEMMs on AITER's preshuffled weights; None keeps AITER."""
        if not _use_aiter:
            return None
        from sglang.srt.layers.quantization.mxfp4 import (
            Mxfp4MoEMethod,
            _aiter_situ_uses_a8w4,
        )

        experts = self.experts
        method = experts.quant_method
        if not (
            _aiter_situ_uses_a8w4()
            and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
            and isinstance(method, Mxfp4MoEMethod)
            and experts.moe_runner_config.activation == "situ"
            and experts.num_local_experts == experts.num_experts
            and method.hidden_pad == 0
            and method.intermediate_pad == 0
            and self.routed_scaling_factor == 1.0
            # [gate | up] rows: the down kernel is built for intermediate sizes 384 and 512
            and experts.w13_weight.shape[1] in (768, 1024)
            and self.topk.topk_config.top_k == 16
            and experts.num_experts <= 1024
            and experts.moe_runner_config.gemm1_clamp_limit is not None
        ):
            print_info_once(
                "Kimi-K3 A8W4 MoE: layers not eligible, AITER fused_moe kept"
            )
            return None
        return _A8W4_MOE_MIN_TOKENS

    @cached_property
    def _a8w4_moe_max_tokens(self) -> int:
        """The down GEMM's output holds up to T * top_k + E * block_m sorted rows and
        must stay under 4 GB."""
        experts = self.experts
        max_rows = ((1 << 32) - 1) // (2 * experts.w2_weight.shape[1])
        return (max_rows - experts.num_experts * 64) // 16

    def _forward_routed_a8w4(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        routed_input: torch.Tensor,
        latent: torch.Tensor,
    ) -> bool:
        """KimiK3MoE._forward_routed on sglang's gfx950 A8W4 grouped GEMMs, when
        they take the batch; False leaves it to the MoE runner."""
        min_tokens = self._a8w4_moe_min_tokens
        if not (
            min_tokens is not None
            and min_tokens <= routed_input.shape[0] <= self._a8w4_moe_max_tokens
            and routed_input.dtype == torch.bfloat16
            and routed_input.is_contiguous()
        ):
            return False
        topk_output = self.topk(hidden_states, router_logits)
        assert TopKOutputChecker.format_is_standard(topk_output), topk_output.format
        self._run_a8w4(
            routed_input,
            topk_output.topk_ids.to(torch.int32),
            topk_output.topk_weights.to(torch.float32),
            latent,
        )
        return True

    def _run_a8w4(
        self,
        routed_input: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        latent: torch.Tensor,
        a_mx: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        """The gfx950 A8W4 grouped GEMMs into latent; a_mx is routed_input
        already through quant_mxfp8."""
        from sglang.kernels.ops.moe.k3_a8w4_gfx950 import k3_a8w4_moe

        experts = self.experts
        k3_a8w4_moe(
            routed_input,
            topk_ids,
            topk_weights,
            experts.w13_weight,
            experts.w13_weight_scale,
            experts.w2_weight,
            experts.w2_weight_scale,
            beta=experts.moe_runner_config.gemm1_alpha,
            linear_beta=experts.moe_runner_config.gemm1_clamp_limit,
            out=latent,
            a_mx=a_mx,
        )


def ffn_row_shards_hip(layer) -> Optional[Callable[[int], bool]]:
    """gfx950: the batch-size predicate under which a KimiK3DecoderLayer's FFN read
    completes the attention output's sum on this rank's rows and leaves the written
    residual there for the row-sharded latent-MoE tail, or None."""
    if layer._sp_moe or layer.all_reduce_fusion:
        return None
    if not (
        _is_gfx95
        and layer._ffn_writes_stream
        and isinstance(layer.mlp, KimiK3MoEHipMixin)
    ):
        return None
    if get_parallel().attn_tp_size != get_parallel().tp_size:
        return None
    return layer.mlp.takes_row_sharded_tail


def link_row_sharded_stream_hip(model) -> None:
    """Let a layer's row-sharded tail leave the stream on this rank's rows only
    when every later layer reads it there (its FFN takes the row-sharded read)
    and no pipeline stage boundary follows: a dense layer or a PP send would
    read every row of the stream and of the bank, which the shard left stale."""
    if not envs.SGLANG_ENABLE_ROCM_K3_ROW_SHARDED_STREAM.get():
        return
    if get_parallel().pp_size > 1:
        return
    later_on_shards = True
    for i in reversed(range(model.start_layer, model.end_layer)):
        layer = model.layers[i]
        if isinstance(layer.mlp, KimiK3MoEHipMixin):
            layer.mlp._row_stream = later_on_shards
        later_on_shards = later_on_shards and layer._row_shards is not None


def link_column_tails_hip(model) -> None:
    """Hand each stream-writing latent MoE of a KimiK3LinearModel the next layer's
    attention read, which its column-parallel tail's all-gather then runs (gfx950
    TP8)."""
    if model.attn_bank is None:
        return
    layers = model.layers
    pairs = [
        (layers[i], layers[i + 1])
        for i in range(model.start_layer, model.end_layer - 1)
        if isinstance(layers[i].mlp, KimiK3MoEHipMixin)
        and layers[i]._ffn_writes_stream
        and not layers[i]._sp_moe
        and not layers[i + 1]._sp_moe
    ]
    # Every TP rank builds the same layers, so they all skip the collective below.
    if not pairs:
        return
    from sglang.srt.layers.communication import k3_ll_comm_hip

    comm = k3_ll_comm_hip.get_comm()
    if comm is None:
        return
    for layer, next_layer in pairs:
        layer.mlp.next_attn_read = NextAttnRead(
            bank=model.attn_bank,
            score_proj=next_layer.self_attention_res_proj,
            score_norm=next_layer.self_attention_res_norm,
            norm=next_layer.input_layernorm,
            writes_block=next_layer.is_block_write_layer,
            comm=comm,
        )
