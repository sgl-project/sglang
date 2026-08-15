from functools import lru_cache
from typing import TYPE_CHECKING

import torch
import triton

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import DpPaddingMode
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import (
    get_parallel,
    process_model_config,
)
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hip


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
    ``SGLANG_NSA_HIP_DISABLE_PRESHUFFLE`` is a deprecated alias.
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
    from sglang.srt.server_args import ServerArgs


def compute_dsa_seqlens(original_seq_lens, dsa_index_topk: int):
    return original_seq_lens.clamp(max=dsa_index_topk)


def should_remap_pd_dsa_seed_to_local_slots(server_args: "ServerArgs") -> bool:
    """Whether a PD seed should enter the allocator-local fused TopK domain."""
    return (
        is_cuda()
        and envs.SGLANG_DSA_FUSE_TOPK.get()
        and server_args.disaggregation_mode == "decode"
        and not server_args.enable_hisparse
        and not get_parallel().dcp_enabled
    )


def should_use_dsa_fused_topk(
    server_args: "ServerArgs", seed_dsa_topk_from_draft_extend: bool
) -> bool:
    """Select fused TopK for PD IndexShare.

    PD Prefill worker:
    - Target prefill: fused TopK enabled.
    - Draft extend: fused TopK disabled.

    PD Decode worker:
    - Draft decode / target verify / draft extend: fused TopK enabled.
    """
    pd_index_share_seed = (
        server_args.disaggregation_mode != "null" and seed_dsa_topk_from_draft_extend
    )
    return envs.SGLANG_DSA_FUSE_TOPK.get() and (
        not pd_index_share_seed or should_remap_pd_dsa_seed_to_local_slots(server_args)
    )


def is_dsa_cp_enabled() -> bool:
    # DSA prefill CP is active when the CP group is on for a DeepSeek Sparse
    # Attention model.
    if get_parallel().attn_cp_size <= 1:
        return False
    from sglang.srt.configs.model_config import is_deepseek_dsa, is_deepseek_v4

    hf_config = process_model_config().hf_config
    return is_deepseek_dsa(hf_config) or is_deepseek_v4(hf_config)


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


def cal_padded_tokens(forward_batch: "ForwardBatch"):
    # Consistent with the padding calculation logic in ForwardBatch.prepare_mlp_sync_batch,
    # calculate the actual token length after padding when attn_tp_size > 1 or in the MAX_LEN padding mode.
    from sglang.srt.layers.cp.utils import is_cp_active

    # CP already pads each rank-local shard to its physical size.
    if is_cp_active(forward_batch):
        return forward_batch.attn_cp_metadata.per_rank_actual_token[
            get_parallel().attn_cp_rank
        ]

    global_num_tokens = forward_batch.global_num_tokens_cpu.copy()
    # Reuse the mode selected when the DP buffer was prepared.
    dp_padding_mode = forward_batch.dp_padding_mode
    if dp_padding_mode is None:
        dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
            forward_batch.is_extend_in_batch, global_num_tokens
        )
    if dp_padding_mode.is_max_len():
        tokens = max(global_num_tokens)
    elif len(global_num_tokens) > 1:
        tokens = global_num_tokens[get_parallel().attn_dp_rank]
    else:
        tokens = global_num_tokens[0]
    return tokens


def pad_dsa_cache_seqlens(forward_batch: "ForwardBatch", dsa_cache_seqlens):
    from sglang.srt.layers.cp.utils import is_cp_active

    needs_cp_pad = is_cp_active(forward_batch)
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


def is_dsa_cp_active(forward_batch) -> bool:
    from sglang.srt.layers.cp.utils import is_cp_active

    return is_dsa_cp_enabled() and is_cp_active(forward_batch)


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
