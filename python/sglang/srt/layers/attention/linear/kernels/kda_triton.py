import logging
from collections import OrderedDict
from typing import Optional

import torch

from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)
from sglang.srt.utils import is_cpu, is_npu, is_xpu

logger = logging.getLogger(__name__)

if not is_cpu():
    from sglang.kernels.ops.attention.fla.fused_recurrent import (
        fused_recurrent_kda_packed_decode,
    )
    from sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm import (
        fused_recurrent_linear_replayssm_decode,
    )
    from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.kernels.ops.attention.fla.kda import chunk_kda


class TritonKDAKernel(LinearAttnKernelBase):
    """Triton-based kernel for KDA (Kimi Delta Attention) linear attention."""

    # XPU has no tvm_ffi CUDA JIT kernel for KDA packed decode; route XPU to the
    # non-packed Triton decode() path (fused_sigmoid_gating_delta_rule_update),
    # the same fallback CPU/NPU use. Batched decode is handled via query_start_loc.
    supports_packed_decode: bool = not is_cpu() and not is_npu() and not is_xpu()
    supports_fused_chain_verify: bool = not is_cpu() and not is_npu()

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Packed decode fast path: feed the conv-1d output ``mixed_qkv``
        straight into a single fused Triton kernel that does Q/K/V extraction,
        gate/beta computation, l2-norm, and the recurrent state update.

        Returns output tensor of shape [1, B, HV, V] to match the existing
        decode kernel output layout.
        """
        B = mixed_qkv.shape[0]
        out = mixed_qkv.new_empty(B, 1, num_v_heads, head_v_dim)

        # KDA ReplaySSM buffered decode: drop-in for the packed decode, same
        # args plus the three per-layer ring caches + the per-row write cursor
        # (and optional radix-track force-flush). Uses the gate-generic kernel
        # with is_kda=True (per-K gate); g_cache is [num_slots, HV, L, K].
        # When any ring tensor / cursor is None (flag off) we fall through to
        # the byte-identical legacy path below.
        replayssm_d = kwargs.get("replayssm_d")
        replayssm_k = kwargs.get("replayssm_k")
        replayssm_g = kwargs.get("replayssm_g")
        replayssm_write_pos = kwargs.get("replayssm_write_pos")
        replayssm_force_flush = kwargs.get("replayssm_force_flush")
        if (
            lower_bound is None
            and replayssm_d is not None
            and replayssm_k is not None
            and replayssm_g is not None
            and replayssm_write_pos is not None
        ):
            if lower_bound is not None:
                raise NotImplementedError(
                    "KDA safe gate (lower_bound) is not implemented in the "
                    "ReplaySSM decode kernel; disable --enable-linear-replayssm."
                )
            K = ssm_states.shape[-1]  # ssm_states: [num_slots, HV, V, K]
            fused_recurrent_linear_replayssm_decode(
                mixed_qkv=mixed_qkv,
                a=a.reshape(B, num_v_heads, K).contiguous(),
                b=b.reshape(B, num_v_heads).contiguous(),
                A_log=A_log.reshape(-1),
                dt_bias=dt_bias.reshape(num_v_heads, K).contiguous(),
                scale=scale,
                initial_state=ssm_states,
                d_cache=replayssm_d,
                k_cache=replayssm_k,
                g_cache=replayssm_g,
                out=out,
                ssm_state_indices=cache_indices,
                write_pos=replayssm_write_pos,
                force_flush=replayssm_force_flush,
                use_qk_l2norm_in_kernel=True,
                is_kda=True,
            )
            return out.transpose(0, 1)

        # a may come in as [B, HV, K] (or [B, 1, HV*K]); b may come in as
        # [B, 1, HV]. Flatten both to the 2D shapes the kernel expects.
        if a.dim() != 2:
            a = a.reshape(B, -1)
        if b.dim() != 2:
            b = b.reshape(B, -1)
        fused_recurrent_kda_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log.reshape(-1),
            dt_bias=dt_bias.reshape(-1),
            scale=scale,
            initial_state=ssm_states,
            out=out,
            ssm_state_indices=cache_indices,
            use_qk_l2norm_in_kernel=True,
            lower_bound=lower_bound,
        )
        # [B, 1, HV, V] -> [1, B, HV, V] view to match existing decode layout.
        return out.transpose(0, 1)

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
            lower_bound=lower_bound,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        intermediate_states_buffer: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_steps: int,
        retrieve_parent_token: Optional[torch.Tensor],
        lower_bound: Optional[float] = None,
        # fused ReplaySSM ring-write (dense verify only; off elsewhere).
        cache_ring: bool = False,
        replayssm_rawv: Optional[torch.Tensor] = None,
        replayssm_rawk: Optional[torch.Tensor] = None,
        replayssm_g: Optional[torch.Tensor] = None,
        replayssm_beta: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # KDA MTP / speculative-decode verify via the fused KDA kernel (IS_KDA=True),
        # mirroring the GDN triton verify path. Reads the committed state, writes
        # per-draft-token intermediate states to the scratch buffer, does NOT mutate
        # the committed pool (disable_state_update=True), and handles chain + tree
        # (retrieve_parent_token). The verify kernel for the Triton / CuTe DSL KDA
        # decode backends, and the reference the KDA correctness tests assert against.
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=True,
            disable_state_update=True,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            lower_bound=lower_bound,
            cache_ring=cache_ring,
            replayssm_rawv=replayssm_rawv,
            replayssm_rawk=replayssm_rawk,
            replayssm_g=replayssm_g,
            replayssm_beta=replayssm_beta,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        common = dict(
            use_qk_l2norm_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
            lower_bound=lower_bound,
            beta_is_raw=beta_is_raw,
            output_intermediate_states=return_intermediate_states,
        )
        block = _extend_block_tokens()
        num_tokens = q.shape[1]
        seq_lens = None
        if 0 < block < num_tokens and not (
            q.is_cuda and torch.cuda.is_current_stream_capturing()
        ):
            seq_lens = _validated_extend_seq_lens(
                kwargs.get("extend_seq_lens_cpu"),
                num_seqs=cache_indices.numel(),
                num_starts=query_start_loc.numel(),
                num_tokens=num_tokens,
            )
        if seq_lens is None:
            return chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                cu_seqlens=query_start_loc,
                initial_state=ssm_states,
                initial_state_indices=cache_indices,
                **common,
            )
        return _extend_blocked(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            seq_lens=seq_lens,
            block=block,
            cu_dtype=query_start_loc.dtype,
            return_intermediate_states=return_intermediate_states,
            common=common,
        )


# chunk_kda's chunk size (``chunk_kda_fwd``: ``chunk_size = 64``). Gate cumsums
# are chunk-local and intermediate states are per chunk, so a block boundary
# inside a sequence must fall on a chunk boundary.
_KDA_CHUNK_TOKENS = 64
_metadata_fallback_logged = False

# cu_seqlens tensors of the blocked path, keyed by (device, stream, dtype,
# boundaries). chunk_kda's prepare_chunk_indices (fla/index.py) caches its
# derived tensors by input identity over its four most recent inputs and
# otherwise syncs on ``.tolist()``, so handing it one tensor per layout avoids
# the H2D copy and the sync for a layout still among those four; a step that
# cycles through more layouts than that syncs on the evicted ones. Entries are
# per stream: fla's derived tensors are enqueued on the stream of first use
# and carry no event, so another stream must not pick them up through a shared
# identity. At the bound the least recently used layout is evicted and rebuilt
# on its next use. The tensors are tiny.
_CU_SEQLENS_CACHE_SIZE = 64
_cu_seqlens_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()


def _extend_block_tokens() -> int:
    """SGLANG_KDA_EXTEND_BLOCK_TOKENS as a chunk-aligned token count.

    0 (the default) or any value below the 64-token chunk disables blocking;
    larger values are rounded down to a multiple of 64.
    """
    from sglang.srt.environ import envs

    block = envs.SGLANG_KDA_EXTEND_BLOCK_TOKENS.get()
    block = int(block) if block is not None else 0
    if block < _KDA_CHUNK_TOKENS:
        return 0
    return block - block % _KDA_CHUNK_TOKENS


def _cached_cu_seqlens(cu_list: list[int], dtype: torch.dtype, device) -> torch.Tensor:
    dev = torch.device(device)
    stream = torch.cuda.current_stream(dev).cuda_stream if dev.type == "cuda" else None
    key = (dev, stream, dtype, tuple(cu_list))
    cu = _cu_seqlens_cache.get(key)
    if cu is None:
        cu = torch.tensor(cu_list, dtype=dtype, device=device)
        if len(_cu_seqlens_cache) >= _CU_SEQLENS_CACHE_SIZE:
            _cu_seqlens_cache.popitem(last=False)
        _cu_seqlens_cache[key] = cu
    else:
        _cu_seqlens_cache.move_to_end(key)
    return cu


def _validated_extend_seq_lens(seq_lens, *, num_seqs, num_starts, num_tokens):
    """Per-sequence token counts for the blocked path, or None when the CPU
    metadata cannot be used (missing, or not matching the device metadata in
    entry count or total), in which case the caller takes the single-call path.

    The boundary values are not compared with ``query_start_loc`` (that would
    need a device sync). They agree by construction: the extend
    ``query_start_loc`` is the device cumsum of the same host list that
    ``forward_batch.extend_seq_lens_cpu`` is (ForwardBatch.init_new ->
    compute_position -> HybridLinearAttnBackend._forward_metadata); a batch
    built from device tensors only leaves the CPU list None and falls back.
    """
    global _metadata_fallback_logged
    reason = None
    if seq_lens is None:
        reason = "extend_seq_lens_cpu is missing"
    else:
        if isinstance(seq_lens, torch.Tensor):
            seq_lens = seq_lens.tolist()
        seq_lens = [int(n) for n in seq_lens]
        if len(seq_lens) != num_seqs or len(seq_lens) + 1 != num_starts:
            reason = (
                f"{len(seq_lens)} extend_seq_lens_cpu entries for {num_seqs} "
                f"cache indices and {num_starts} query_start_loc entries"
            )
        elif any(n < 0 for n in seq_lens) or sum(seq_lens) != num_tokens:
            reason = f"extend_seq_lens_cpu sums to {sum(seq_lens)}, not {num_tokens}"
    if reason is None:
        return seq_lens
    if not _metadata_fallback_logged:
        _metadata_fallback_logged = True
        logger.debug(
            "KDA blocked extend: %s; using a single chunk_kda call instead", reason
        )
    return None


def _extend_blocked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    seq_lens: list[int],
    block: int,
    cu_dtype: torch.dtype,
    return_intermediate_states: bool,
    common: dict,
):
    """Run chunk_kda over token blocks of at most ``block`` tokens.

    The chunk kernels' workspace scales with the tokens in one call, so this
    bounds it by the block. Consecutive sequences that fit within the block
    together share one call (one ``cu_seqlens``); a sequence longer than the
    block is split on its own, one call per block. Invariants relied on:

    - ``block`` is a multiple of the 64-token chunk (``_extend_block_tokens``),
      so split points inside a sequence are chunk boundaries: the chunk-local
      gate cumsum and the per-chunk ``h`` entries are the same as in the
      single call. The intra-chunk kernel variant is chosen per call from
      ``B * num_chunks * H <= 256`` (fla/kda.py), so a block whose grid falls
      on the other side of that threshold than the single call runs the other
      variant, with rounding-level differences; otherwise results are
      identical.
    - The fwd_h kernel loads a sequence's initial state from
      ``initial_state[index]`` and stores its final state to the same slot
      (INPLACE_UPDATE), skipping both for the padded-row sentinel ``-1``.
      Consecutive blocks of one sequence therefore chain through a per-sequence
      fp32 scratch slot: gathered from the pool, or zero for ``-1`` (the single
      call's initial state for that row), and written back once, masked for
      ``-1``. The kernel accumulates in fp32 and stores in the slot's dtype, so
      the one final cast back to a bf16/fp16 pool is the single call's one
      final rounding. Whether a row is padded is only known on device, so the
      scratch is used for every split sequence (one state gather and scatter
      per split sequence, exact for fp32 pools).
    - ``chunk_gla_fwd_o_gk`` writes the output into ``v`` (``o=v``). ``v`` is
      made contiguous once (chunk_kda would copy a strided ``v`` anyway), so
      block slices are views and the in-place writes assemble the output.
    - Per call, ``chunk_kda`` builds ``prepare_chunk_indices(cu_seqlens)``
      (fla/index.py), cached by tensor identity and otherwise syncing on
      ``.tolist()``; ``_cached_cu_seqlens`` hands it one tensor per layout.
    - Intermediate states: chunk_kda packs one state per 64-token chunk per
      sequence in call order (``len(prepare_chunk_indices(cu_seqlens, 64))``).
      Blocks are chunk-aligned, so the per-call counts sum to the single
      call's total in the same order; ``h`` is assembled into one
      preallocated tensor, releasing each block's ``h`` after the copy.
    """
    device = q.device
    v = v.contiguous()
    h_out = None
    h_pos = 0
    num_chunks = sum(-(-n // _KDA_CHUNK_TOKENS) for n in seq_lens)

    def run(s, e, cu_list, initial_state, indices):
        nonlocal h_out, h_pos
        out = chunk_kda(
            q=q[:, s:e],
            k=k[:, s:e],
            v=v[:, s:e],
            g=g[:, s:e],
            beta=beta[:, s:e],
            cu_seqlens=_cached_cu_seqlens(cu_list, cu_dtype, device),
            initial_state=initial_state,
            initial_state_indices=indices,
            **common,
        )
        if return_intermediate_states:
            h = out[1]
            if h_out is None:
                h_out = h.new_empty(h.shape[0], num_chunks, *h.shape[2:])
            h_out[:, h_pos : h_pos + h.shape[1]].copy_(h)
            h_pos += h.shape[1]

    scratch_index = None
    n = len(seq_lens)
    i = 0
    start = 0
    while i < n:
        length = seq_lens[i]
        if length > block:
            # Split sequence: one call per block, the state chained through a
            # zero-or-gathered fp32 scratch slot and written back once.
            idx = cache_indices[i : i + 1]
            # Out of place: for an int64 index ``to`` returns the caller's
            # tensor itself, and a padded -1 must stay -1 there.
            safe_idx = idx.to(torch.long).clamp(min=0)
            valid = (idx >= 0).view(-1, 1, 1, 1)
            scratch = torch.where(valid, ssm_states[safe_idx].to(torch.float32), 0.0)
            if scratch_index is None:
                scratch_index = torch.zeros(1, dtype=cache_indices.dtype, device=device)
            for b0 in range(0, length, block):
                b1 = min(b0 + block, length)
                run(start + b0, start + b1, [0, b1 - b0], scratch, scratch_index)
            ssm_states[safe_idx] = torch.where(
                valid, scratch.to(ssm_states.dtype), ssm_states[safe_idx]
            )
            start += length
            i += 1
        else:
            # Group of whole sequences: as many consecutive ones as fit.
            j, total = i, 0
            while j < n and total + seq_lens[j] <= block:
                total += seq_lens[j]
                j += 1
            if total > 0:
                cu_list = [0]
                for m in seq_lens[i:j]:
                    cu_list.append(cu_list[-1] + m)
                run(start, start + total, cu_list, ssm_states, cache_indices[i:j])
            start += total
            i = j
    if return_intermediate_states:
        assert h_pos == num_chunks, (h_pos, num_chunks)
        return v, h_out
    return v
