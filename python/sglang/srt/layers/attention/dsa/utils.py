from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import triton

from sglang.srt.environ import envs
from sglang.srt.layers.attention.mqa_logits_utils import (
    mqa_logits_needs_budget_check,
    mqa_logits_static_budget_bytes,
)
from sglang.srt.layers.dp_attention import DpPaddingMode, dp_slot_in
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_memory,
    get_parallel,
    process_model_config,
)
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hip, is_musa
from sglang.srt.utils.common import ceil_div


@lru_cache(maxsize=1)
def aiter_can_use_preshuffle_paged_mqa() -> bool:
    """Whether aiter's preshuffle paged MQA / cache kernels can be used on this runtime.

    aiter's ``deepgemm_fp8_paged_mqa_logits`` only supports ``KVBlockSize > 1`` and
    ``Preshuffle=True`` on its gluon kernel path. The gluon path is enabled when
    Triton >= 3.5.0, OR when ``AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`` is set
    (which additionally requires that the AOT gluon kernel artifacts ship inside
    the aiter wheel/image). Otherwise aiter asserts ``KVBlockSize == 1`` and
    refuses ``Preshuffle=True``.

    sglang's DSA indexer uses this single decision to pick:
      * ``page_size``: 64 (preshuffle) vs 1 (legacy) on ROCm
      * ``Preshuffle`` / ``preshuffle`` flags on the aiter MQA + cache kernels
      * ``get_page_table_64`` vs ``get_page_table_1`` on the metadata
      * whether ``GetKAndS.execute`` uses the aiter or the triton implementation

    The result is cached so the cost is paid once per process.

    Set ``SGLANG_DSA_HIP_DISABLE_PRESHUFFLE=1`` to force the legacy path even when
    the gluon kernel would otherwise be available (useful for CI bisection).
    """
    if not is_hip():
        return False
    if not get_bool_env_var("SGLANG_USE_AITER"):
        return False
    if envs.SGLANG_DSA_HIP_DISABLE_PRESHUFFLE.get():
        return False
    if get_bool_env_var("AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS"):
        return True
    try:
        from packaging.version import Version

        return Version(Version(triton.__version__).base_version) >= Version("3.5.0")
    except Exception:
        return False


# Tile size for the indexer FP8 K-cache preshuffle layout. Store and gather
# kernels reorganize each page into (tile x tile) blocks so the aiter preshuffle
# paged-MQA gather can consume the cache directly.
INDEXER_K_CACHE_PRESHUFFLE_TILE = 16


if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def compute_dsa_seqlens(original_seq_lens, dsa_index_topk: int, index_kpool: int = 1):
    if index_kpool <= 1:
        return original_seq_lens.clamp(max=dsa_index_topk)

    # Clamp only complete pools; the unfinished tail must remain selectable
    # outside the pooled top-k budget.
    full_pool_tokens = (
        torch.div(original_seq_lens, index_kpool, rounding_mode="floor") * index_kpool
    )
    selected_history_tokens = full_pool_tokens.clamp(max=dsa_index_topk)
    tail_tokens = original_seq_lens - full_pool_tokens
    return selected_history_tokens + tail_tokens


def should_remap_pd_dsa_seed_to_local_slots() -> bool:
    """Whether a PD seed should enter the allocator-local fused TopK domain."""
    return (
        (is_cuda() or is_hip())
        and envs.SGLANG_DSA_FUSE_TOPK.get()
        and get_disagg().disaggregation_mode == "decode"
        and not get_memory().enable_hisparse
        and not get_parallel().dcp_enabled
    )


def should_use_dsa_fused_topk(seed_dsa_topk_from_draft_extend: bool) -> bool:
    """Select fused TopK for PD IndexShare.

    PD Prefill worker:
    - Target prefill: fused TopK enabled.
    - Draft extend: fused TopK disabled.

    PD Decode worker:
    - Draft decode / target verify / draft extend: fused TopK enabled.
    """
    pd_index_share_seed = (
        get_disagg().disaggregation_mode != "null" and seed_dsa_topk_from_draft_extend
    )
    return envs.SGLANG_DSA_FUSE_TOPK.get() and (
        not pd_index_share_seed or should_remap_pd_dsa_seed_to_local_slots()
    )


def is_dsa_enable_prefill_cp():
    if is_hip() or is_musa():
        return False

    # Generic prefill CP derives activation from the runtime topology and model
    # architecture.
    if get_parallel().attn_cp_size <= 1:
        return False
    from sglang.srt.configs.model_config import is_deepseek_dsa, is_deepseek_v4

    hf_config = process_model_config().hf_config
    return is_deepseek_dsa(hf_config) or is_deepseek_v4(hf_config)


def is_dsa_prefill_cp_interleave():
    return is_dsa_enable_prefill_cp() and get_parallel().cp_strategy == "interleave"


# Retain the name imported by the unchanged HIP radix attention backend.
is_dsa_prefill_cp_round_robin_split = is_dsa_prefill_cp_interleave


# Structural surface where the graph DSA split-op dispatch (DSA indexer) and the
# MLA BMM-into-attention fusion apply: a non-speculative extend (prefill) running
# inside a piecewise/breakable CUDA graph. Both fusions are now on by default on
# this surface (no feature flag); each adds its own extra carve-outs at its call
# site (e.g. the indexer also excludes DSA prefill context parallelism).
def is_graph_dsa_split_op_surface(forward_batch: "ForwardBatch") -> bool:
    return (
        is_cuda()
        and (is_in_tc_piecewise_cuda_graph() or is_in_breakable_cuda_graph())
        and forward_batch.forward_mode.is_extend_without_speculative()
    )


def can_dsa_prefill_cp_interleave(forward_batch: "ForwardBatch"):
    if not forward_batch.forward_mode.is_context_parallel_extend():
        return False
    cp_size = get_parallel().attn_cp_size
    seq_len = sum(forward_batch.extend_seq_lens_cpu)
    return (
        is_dsa_prefill_cp_interleave()
        and seq_len > 0
        and seq_len >= cp_size
        and cp_size > 1
    )


def cal_padded_tokens(forward_batch: "ForwardBatch"):
    # Consistent with the padding calculation logic in ForwardBatch.prepare_mlp_sync_batch,
    # calculate the actual token length after padding when attn_tp_size > 1 or in the MAX_LEN padding mode.
    from sglang.srt.layers.cp.utils import is_cp_active

    # CP-v2 already pads each rank-local shard to its physical size
    if is_cp_active(forward_batch):
        return forward_batch.attn_cp_metadata.per_rank_actual_token[
            get_parallel().attn_cp_rank
        ]

    global_num_tokens = forward_batch.global_num_tokens_cpu.copy()
    attn_cp_size = get_parallel().attn_cp_size
    # Non-CP forwards (including speculative forwards) use attention-TP padding
    # only, matching ForwardBatch.prepare_mlp_sync_batch.
    # Reuse the mode selected when the DP buffer was prepared.
    dp_padding_mode = forward_batch.dp_padding_mode
    if dp_padding_mode is None:
        dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
            forward_batch.is_extend_in_batch, global_num_tokens
        )
    if dp_padding_mode.is_max_len():
        tokens = max(global_num_tokens)
    else:
        tokens = global_num_tokens[dp_slot_in(global_num_tokens)]
    if can_dsa_prefill_cp_interleave(forward_batch):
        tokens = ceil_div(tokens, attn_cp_size)
    return tokens


def pad_dsa_cache_seqlens(forward_batch: "ForwardBatch", dsa_cache_seqlens):
    attn_cp_size = get_parallel().attn_cp_size
    needs_cp_pad = attn_cp_size > 1 and can_dsa_prefill_cp_interleave(forward_batch)
    needs_dp_pad = forward_batch.global_num_tokens_cpu is not None
    if not needs_cp_pad and not needs_dp_pad:
        return dsa_cache_seqlens
    tokens = cal_padded_tokens(forward_batch)
    pad_len = tokens - dsa_cache_seqlens.shape[0]
    if pad_len > 0:
        dsa_cache_seqlens = torch.cat(
            [
                dsa_cache_seqlens,
                dsa_cache_seqlens.new_zeros(pad_len, *dsa_cache_seqlens.shape[1:]),
            ]
        )
    return dsa_cache_seqlens


def dsa_use_prefill_cp(forward_batch, dsa_enable_prefill_cp=None):
    if dsa_enable_prefill_cp is None:
        dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
    if (
        forward_batch.attn_cp_metadata is not None
        and dsa_enable_prefill_cp
        and forward_batch.forward_mode.is_context_parallel_extend()
    ):
        return True
    else:
        return False


def fp8_mqa_logits_ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


def fp8_mqa_logits_make_fused_kv(
    kv_fp8: torch.Tensor,
    kv_scales: torch.Tensor,
    block_kv: int,
    head_dim: int,
) -> torch.Tensor:
    num_phys_blocks = kv_fp8.shape[0]
    per_token_size = head_dim + 4
    block_bytes = block_kv * per_token_size
    scale_offset = block_kv * head_dim

    fused = torch.zeros(
        num_phys_blocks, block_bytes, dtype=torch.uint8, device=kv_fp8.device
    )
    for blk in range(num_phys_blocks):
        fused[blk, :scale_offset] = kv_fp8[blk].view(torch.uint8).reshape(-1)
        fused[blk, scale_offset:] = (
            kv_scales[blk].float().contiguous().view(torch.uint8).reshape(-1)
        )
    return fused.view(num_phys_blocks, block_kv, 1, per_token_size)


def _use_torch_mqa_logits() -> bool:
    return envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get()


@lru_cache(maxsize=1)
def resolve_num_sms() -> int:
    """SM count used to size the paged-MQA-logits schedule.

    DeepGEMM exposes this as ``get_num_sms()``, but on the torch fallback path
    DeepGEMM may not be installed at all, so read it off the device instead.
    Both consumers only use it to shape the schedule metadata buffer, so the
    raw device SM count is an acceptable stand-in.
    """
    if not _use_torch_mqa_logits():
        import deep_gemm

        return deep_gemm.get_num_sms()
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


@lru_cache(maxsize=1)
def resolve_paged_mqa_logits_metadata_fn():
    if _use_torch_mqa_logits():
        from sglang.kernels.ops.attention.dsv4 import get_paged_mqa_logits_metadata

        return get_paged_mqa_logits_metadata
    import deep_gemm

    return deep_gemm.get_paged_mqa_logits_metadata


def _fp8_paged_mqa_logits_torch(
    q_fp8,
    kv_cache_fp8,
    weights,
    context_lens,
    block_table,
    schedule_metadata,
    max_seq_len,
    clean_logits=False,
    indices=None,
):
    from sglang.srt.layers.attention.dsv4.indexer import fp8_paged_mqa_logits_torch

    assert indices is None, "torch paged MQA logits does not take an index tensor"
    if context_lens.dim() == 2:
        assert context_lens.shape[1] == 1, (
            "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH does not support next_n > 1 "
            f"(got {context_lens.shape[1]}); disable speculative decoding."
        )
        context_lens = context_lens[:, 0]
    batch_size = q_fp8.shape[0]
    block_size = kv_cache_fp8.shape[1]
    head_dim = q_fp8.shape[-1]
    padded_seq_len = block_table.shape[1] * block_size
    per_token_work_bytes = 2 * (head_dim + 4) + 2 * head_dim + 4
    per_row_bytes = padded_seq_len * per_token_work_bytes + max_seq_len * 4
    if (
        q_fp8.is_cuda
        and per_row_bytes > 0
        and mqa_logits_needs_budget_check(
            num_rows=batch_size, num_cols=max(max_seq_len, padded_seq_len)
        )
    ):
        device_index = q_fp8.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        budget_bytes = mqa_logits_static_budget_bytes(device_index=device_index)
        rows_per_chunk = max(budget_bytes // per_row_bytes, 1)
    else:
        rows_per_chunk = batch_size
    if rows_per_chunk < batch_size:
        logits_chunks = []
        for start in range(0, batch_size, rows_per_chunk):
            end = min(start + rows_per_chunk, batch_size)
            logits_chunks.append(
                fp8_paged_mqa_logits_torch(
                    q_fp8[start:end],
                    kv_cache_fp8,
                    weights[start:end],
                    context_lens[start:end],
                    block_table[start:end],
                    schedule_metadata,
                    max_seq_len,
                    clean_logits=clean_logits,
                )
            )
        return torch.cat(logits_chunks, dim=0)
    return fp8_paged_mqa_logits_torch(
        q_fp8,
        kv_cache_fp8,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_seq_len,
        clean_logits=clean_logits,
    )


@lru_cache(maxsize=1)
def resolve_fp8_paged_mqa_logits_fn():
    if _use_torch_mqa_logits():
        return _fp8_paged_mqa_logits_torch
    import deep_gemm

    return deep_gemm.fp8_paged_mqa_logits


def _fp8_mqa_logits_torch(
    q_fp8, kv, weights, ks, ke, clean_logits=False, max_seqlen_k=0
):
    """Pure-torch prefill indexer fallback (no DeepGEMM).

    Accumulates per-head relu(q·k) weighted by gate, then applies fp8 scale.
    Iterates over heads to keep the [num_q, num_kv] intermediate small.
    """
    assert not clean_logits, "torch fp8_mqa_logits only implements clean_logits=False"
    k_fp8, k_scale = kv
    num_q, num_heads, head_dim = q_fp8.shape
    num_kv = k_fp8.shape[0]

    q = q_fp8.to(torch.bfloat16)
    k_t = k_fp8.reshape(num_kv, head_dim).to(torch.bfloat16).t()

    logits = torch.zeros(num_q, num_kv, dtype=torch.float32, device=q_fp8.device)
    w = weights.float()
    for h in range(num_heads):
        scores = torch.mm(q[:, h], k_t).float()
        logits.addcmul_(torch.relu(scores), w[:, h : h + 1])
    logits *= k_scale.reshape(1, num_kv).float()

    positions = torch.arange(num_kv, device=logits.device).unsqueeze(0)
    valid = (positions >= ks.unsqueeze(1)) & (positions < ke.unsqueeze(1))
    return logits.masked_fill_(~valid, 0.0)


@lru_cache(maxsize=1)
def resolve_fp8_mqa_logits_fn():
    if _use_torch_mqa_logits():
        return _fp8_mqa_logits_torch
    import deep_gemm

    return deep_gemm.fp8_mqa_logits
