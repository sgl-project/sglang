"""DeepSeek-V4.1 fused low-latency decode chain (SM100, small batches).

One straight chain takes the attention sublayer from the query to the next
sublayer's normalized input, replacing the mega attention path's tail:

    decode_attention    V4.1 paged fp8 SWA keys plus the fp8 / fp4 indexed keys,
                        inverse RoPE folded into the epilogue, emitted in FlashMLA
                        MegaAttention's fp8 format -- replaces flash_mla_mega_decode
    wo_a_fp8            that activation x wo_a, emitted as the (e4m3, UE8M0 128x4
                        image) pair FlashInfer's mm_mxfp8 takes -- replaces
                        deep_gemm.fp8_einsum plus the activation quant
    wo_b + all_reduce   wo_b pushing its rows straight into the TP push plane,
                        drained together with the mHC post, the next combine and
                        its norm -- replaces wo_b + all_reduce_mhc_post_combine_norm

:meth:`LowLatencyChain.attention` runs the first half in the attention sublayer and
:meth:`LowLatencyChain.post` the second half in the mHC post, because the post is
where the mHC coefficients and the next norm live. Nothing else takes a hand: the
wo_a rows travel between them as the attention's ``mhc.AttnOutput``, where their
e4m3 dtype says wo_b has not run yet.

Only the token count is dynamic. Every other precondition -- platform, model
geometry, weight layouts, parallel layout, the registered push plane -- is fixed
for a layer's lifetime and resolved once by :func:`build_chain` after the weights
load; a layer whose chain is None never takes this path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.runtime_context import get_parallel, get_platform

if TYPE_CHECKING:
    from sglang.kernels.ops.communication.all_reduce import Communicator
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# The kernels are built for exactly this geometry; see the static_asserts in
# csrc/deepseek_v4/low_latency/*.cuh.
MAX_TOKENS = 64
NUM_HEADS = 16
NUM_GROUPS = 2
O_LORA_RANK = 1024
HEAD_DIM = 512
HIDDEN = 5120
# The push plane world sizes wo_b and all_reduce_mhc are instantiated for.
WORLD_SIZES = (4,)

# An e4m3 attention contribution is this chain's wo_a rows, with wo_b still to run;
# everything else is wo_b's bf16 output.
ROWS_DTYPE = torch.float8_e4m3fn


class LowLatencyChain(msgspec.Struct, frozen=True):
    """One layer's fused low-latency material, fixed once the weights have loaded.

    ``wo_a_*`` are ``setup_mega_attention``'s quantized wo_a -- the operands
    DeepGEMM's ``fp8_einsum`` takes -- and ``wo_b_*`` the plain [5120, 2048] e4m3
    weight with the FlashInfer-swizzled scales its mm_mxfp8 takes.
    """

    wo_a_weight: torch.Tensor
    wo_a_scale: torch.Tensor
    wo_b_weight: torch.Tensor
    wo_b_scale: torch.Tensor
    # wo_a's UE8M0 scales, a 128x4 image whose size does not depend on the token
    # count, so the rows crossing to the post carry their scales implicitly.
    x_scale: torch.Tensor
    # The layer's freqs_cis as [max_pos, 64] fp32 interleaved (cos, sin) pairs, the
    # table the attention's fused inverse RoPE reads (not the mega kernel's
    # cos | sin halves).
    rope_table: torch.Tensor
    comm: Communicator
    # One multimem store per 16 B instead of one per rank; needs the push plane's
    # multicast mapping.
    multicast: bool
    # The GEMMs' own ceiling, narrowed to what the push plane can stage.
    max_tokens: int

    def is_active(self, forward_batch: ForwardBatch, num_tokens: int) -> bool:
        """The per-forward half of the gate: the token count the GEMMs and the push
        plane are built for, on a batch of single-query rows."""
        mode = forward_batch.forward_mode
        return (
            0 < num_tokens <= self.max_tokens
            and (mode.is_decode() or mode.is_target_verify())
            # The prefill piecewise graph breaks out to eager around the attention
            # and hands back a fixed bf16 buffer, which this chain does not produce.
            and not is_in_breakable_cuda_graph()
        )

    def attention(
        self,
        *,
        q: torch.Tensor,
        attn_sink: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        attn_layer,
        compress_ratio: int,
    ) -> torch.Tensor:
        """The sparse decode attention and wo_a; returns the [T, 2048] e4m3 rows
        whose scales this chain holds in ``x_scale``."""
        from sglang.kernels.ops.attention.dsv4.low_latency.wo_a import wo_a_fp8

        data, scale = attn_backend.forward(
            q=q,
            k=q,
            v=q,
            layer=attn_layer,
            forward_batch=forward_batch,
            compress_ratio=compress_ratio,
            attn_sink=attn_sink,
            save_kv_cache=False,
            cos_sin_cache=self.rope_table,
            low_latency=True,
        )
        rows, _ = wo_a_fp8(
            data,
            scale,
            self.wo_a_weight,
            self.wo_a_scale,
            out=(
                torch.empty(
                    data.shape[0],
                    NUM_GROUPS * O_LORA_RANK,
                    dtype=ROWS_DTYPE,
                    device=data.device,
                ),
                self.x_scale,
            ),
        )
        return rows

    def wo_b(self, rows: torch.Tensor) -> torch.Tensor:
        """wo_b alone, to bf16 [T, 5120]: for a post with no next norm to fold, which
        all-reduces the rows itself."""
        from sglang.kernels.ops.attention.dsv4.low_latency.wo_b import wo_b_fp8

        return wo_b_fp8(rows, self.x_scale, self.wo_b_weight, self.wo_b_scale)

    def post(
        self,
        rows: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb: torch.Tensor,
        pre: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """wo_b into the push plane, then the all-reduce, the mHC post, the next
        combine and its norm; returns ``(mhc_out, normalized)``.

        Both kernels read the plane's phase counters -- and the consumer also the
        mHC coefficients -- before their PDL wait, so two of these must have a
        kernel that waits before it triggers between them. The next layer's
        attention is that kernel; nothing may consume two posts back to back.
        """
        from sglang.kernels.ops.attention.dsv4.low_latency.wo_b import (
            wo_b_fp8_all_reduce_mhc,
        )

        return wo_b_fp8_all_reduce_mhc(
            rows,
            self.x_scale,
            self.wo_b_weight,
            self.wo_b_scale,
            self.comm,
            residual,
            post_mix,
            comb,
            pre,
            norm_weight,
            eps,
            multicast=self.multicast,
        )


def build_chain(layer, *, fused_mhc_post: bool) -> Optional[LowLatencyChain]:
    """This attention's chain, or None when any of its fixed preconditions fail.

    Runs after the weights load and after the TP group's push plane is registered
    (``precompile_kernels_after_loading``), so every check here reads a value that
    no longer changes.

    ``fused_mhc_post`` is the decoder layer's static half of ``defers_sum``: the
    chain hands the boundary wo_a rows, which only the fused post knows how to
    finish, so it must not run where the boundary would complete the sum itself.
    Its dynamic half is the push-plane capacity, which ``max_tokens`` carries.

    The checks below also subsume every term of ``_project_wo_b``'s own deferral
    predicate (CUDA, attn_tp > 1, and the DP / CP layouts that publish
    ``mlp_reduce_scatter``), so the caller's gate needs only ``defer_all_reduce``.
    """
    from sglang.kernels.ops.communication.all_reduce_fusion import get_registered_comm
    from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
        CustomAllReduceV2,
    )

    parallel = get_parallel()
    if not (
        envs.SGLANG_OPT_DSV41_LOW_LATENCY_DECODE.get()
        and fused_mhc_post
        and get_platform().is_sm100
        and layer.is_dsv41
        # The mega attention path owns the V4.1 fp8 KV layout, the fp8 wo_a and the
        # quantized wo_a operands this chain's wo_a reads.
        and layer.use_mega_attention
        and layer.mega_wo_a is not None
        # 16 local heads keep wq_b's rows unpermuted, which is the per-head query
        # layout the attention kernel reads.
        and not layer.mega_wq_b_permuted
        and layer.n_local_heads == NUM_HEADS
        and layer.n_local_groups == NUM_GROUPS
        and layer.o_lora_rank == O_LORA_RANK
        and layer.head_dim == HEAD_DIM
        and layer.hidden_size == HIDDEN
        # The push plane carries the attention sum, so the attention TP group must
        # be the whole TP group, with no attention DP or CP beside it.
        and parallel.attn_dp_size == 1
        and parallel.attn_cp_size == 1
        and layer.attn_tp_size == parallel.tp_size
        and layer.attn_tp_size in WORLD_SIZES
    ):
        return None

    from sglang.kernels.jit.utils.arch import get_jit_cuda_version

    # The attention kernel's scaled e4m3 -> bf16 conversion (cvt.rn.scaled) needs nvcc >= 13.2.
    if get_jit_cuda_version() < (13, 2):
        return None
    wo_b_weight, wo_b_scale = _flashinfer_mxfp8_wo_b_weights(layer)
    if wo_b_weight is None:
        return None
    # The fusion kernels key on world size alone, so this is the same plane
    # register_fused_all_reduce_comm() handed the MoE finalize.
    comm = get_registered_comm(layer.attn_tp_size)
    ca_comm = parallel.tp_group.ca_comm
    if comm is None or comm.push is None:
        return None
    if not isinstance(ca_comm, CustomAllReduceV2) or ca_comm.obj is not comm:
        return None
    max_tokens = min(
        MAX_TOKENS,
        # One phase counter per row, and one slot for the whole [T, 5120] bf16 push.
        ca_comm.config.num_push_blocks,
        ca_comm.max_push_size // (HIDDEN * 2),
    )
    if max_tokens <= 0:
        return None

    return LowLatencyChain(
        wo_a_weight=layer.mega_wo_a[0],
        wo_a_scale=layer.mega_wo_a[1],
        wo_b_weight=wo_b_weight,
        wo_b_scale=wo_b_scale,
        x_scale=torch.empty(
            128 * NUM_GROUPS * O_LORA_RANK // 32,
            dtype=torch.uint8,
            device=wo_b_weight.device,
        ),
        rope_table=torch.view_as_real(layer.freqs_cis).flatten(1),
        comm=comm,
        multicast=ca_comm.has_multicast,
        max_tokens=max_tokens,
    )


def _flashinfer_mxfp8_wo_b_weights(layer):
    """wo_b's ``(weight, swizzled scales)`` when it runs the FlashInfer MXFP8 GEMM
    on the shape the kernel is built for, else ``(None, None)``.

    ``use_flashinfer_mxfp8_wo_b`` is what guarantees ``weight_scale_inv_swizzled``
    exists: both arms it accepts go through ``_process_mxfp8_linear_weight_scale``.
    """
    if not layer.use_flashinfer_mxfp8_wo_b:
        return None, None
    weight = layer.wo_b.weight.data
    if (
        weight.shape != (HIDDEN, NUM_GROUPS * O_LORA_RANK)
        or weight.dtype != torch.float8_e4m3fn
        or not weight.is_contiguous()
    ):
        return None, None
    scale = layer.wo_b.weight_scale_inv_swizzled.data
    return weight, scale.contiguous().view(torch.uint8).view(-1)
