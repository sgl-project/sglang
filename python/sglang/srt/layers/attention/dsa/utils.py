from functools import lru_cache
from typing import TYPE_CHECKING, List, Tuple, Union

import torch
import triton

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.runtime_context import get_parallel, get_server_args
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hip
from sglang.srt.utils.common import ceil_align, ceil_div


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


if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.server_args import ServerArgs


def compute_dsa_seqlens(original_seq_lens, dsa_index_topk: int):
    return original_seq_lens.clamp(max=dsa_index_topk)


def should_remap_pd_dsa_seed_to_local_slots(server_args: "ServerArgs") -> bool:
    """Whether a PD seed should enter the allocator-local fused TopK domain."""
    return (
        envs.SGLANG_DSA_FUSE_TOPK.get()
        and server_args.disaggregation_mode == "decode"
        and not server_args.enable_hisparse
        and server_args.dcp_size == 1
    )


def should_use_dsa_fused_topk(
    server_args: "ServerArgs", seed_dsa_topk_from_draft_extend: bool
) -> bool:
    pd_index_share_seed = (
        server_args.disaggregation_mode != "null" and seed_dsa_topk_from_draft_extend
    )
    return envs.SGLANG_DSA_FUSE_TOPK.get() and (
        not pd_index_share_seed or should_remap_pd_dsa_seed_to_local_slots(server_args)
    )


def is_dsa_enable_prefill_cp():
    return get_server_args().enable_dsa_prefill_context_parallel


def is_dsa_prefill_cp_in_seq_split():
    return (
        is_dsa_enable_prefill_cp()
        and get_server_args().dsa_prefill_cp_mode == "in-seq-split"
    )


def is_dsa_prefill_cp_round_robin_split():
    return (
        is_dsa_enable_prefill_cp()
        and get_server_args().dsa_prefill_cp_mode == "round-robin-split"
    )


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


def can_dsa_prefill_cp_round_robin_split(forward_batch: "ForwardBatch"):
    if not forward_batch.forward_mode.is_context_parallel_extend():
        return False
    cp_size = get_parallel().attn_cp_size
    seq_len = sum(forward_batch.extend_seq_lens_cpu)
    return (
        is_dsa_prefill_cp_round_robin_split()
        and seq_len > 0
        and seq_len >= cp_size
        and cp_size > 1
    )


def dsa_cp_round_robin_split_data(input_: Union[torch.Tensor, List]):
    """
    # for round-robin-split, split the tokens evenly according to the rule of token_idx % cp_size.
    |   +-----------before split------------+|
    | token0, token1, token2, token3, token4, token5, token6, token7, ...
    |
    |   +--------------result-------------------+
    | dp_atten_tp0: token0, token4, token8, token12, token16, ... |
    | dp_atten_tp1: token1, token5, token9, token13, token17, ... |
    | dp_atten_tp2: token2, token6, token10, token14, token18, ... |
    | dp_atten_tp3: token3, token7, token11, token15, token19, ... |
    |   +-------------------------+
    """
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    if isinstance(input_, (tuple, list)):
        indices = range(cp_rank, len(input_), cp_size)
        return input_[indices]

    tokens = len(input_)
    if tokens % cp_size != 0:
        cur_len = tokens // cp_size + (tokens % cp_size > cp_rank)
        if cur_len == 0:
            return input_.new_empty(0, *input_.shape[1:])
        indices = torch.arange(cp_rank, tokens, cp_size, device=input_.device)
        return input_[indices]

    # for torch device tensor
    return input_.view(-1, cp_size, *input_.shape[1:])[:, cp_rank].contiguous()


def cal_padded_tokens(forward_batch: "ForwardBatch"):
    # Consistent with the padding calculation logic in ForwardBatch.prepare_mlp_sync_batch,
    # calculate the actual token length after padding when attn_tp_size > 1 or in the MAX_LEN padding mode.
    from sglang.srt.layers.utils.cp_utils import get_cp_padding_align_size

    global_num_tokens = forward_batch.global_num_tokens_cpu.copy()
    sync_group_size = len(global_num_tokens)
    attn_cp_size = get_parallel().attn_cp_size
    # Must match the CP padding in ForwardBatch.prepare_mlp_sync_batch.
    cp_align_size = get_cp_padding_align_size()
    for i in range(sync_group_size):
        global_num_tokens[i] = ceil_align(global_num_tokens[i], cp_align_size)
    dp_padding_mode = DpPaddingMode.get_dp_padding_mode(
        forward_batch.is_extend_in_batch, global_num_tokens
    )
    if dp_padding_mode.is_max_len():
        tokens = max(global_num_tokens)
    elif len(global_num_tokens) > 1:
        tokens = global_num_tokens[get_parallel().attn_dp_rank]
    else:
        tokens = global_num_tokens[0]
    if can_dsa_prefill_cp_round_robin_split(forward_batch):
        tokens = ceil_div(tokens, attn_cp_size)
    return tokens


def pad_dsa_cache_seqlens(forward_batch: "ForwardBatch", dsa_cache_seqlens):
    attn_cp_size = get_parallel().attn_cp_size
    needs_cp_pad = attn_cp_size > 1 and can_dsa_prefill_cp_round_robin_split(
        forward_batch
    )
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


def can_dsa_cp_split(seq_len: int, cp_size: int, use_dsa: bool, forward_batch):
    if is_dsa_prefill_cp_round_robin_split():
        cur_cp_seq_len = seq_len // cp_size
        assert (
            seq_len % cp_size == 0
        ), f"seq_len {seq_len} is not divisible by cp_size {cp_size} when dsa_prefill_cp_mode is round-robin-split"
    else:
        # TODO current just support prefill batch=1 and len(input_ids) > self.cp_size * 2
        # Note: (self.cp_size * 2) To achieve load balancing for seq computation,
        # the seq data needs to be divided and recombined at twice the size of cp_size.
        cur_cp_seq_len = seq_len // (cp_size * 2)
    if (
        cur_cp_seq_len != 0
        and cp_size > 1
        and use_dsa
        and forward_batch.forward_mode.is_context_parallel_extend()
        and is_dsa_enable_prefill_cp()
        and sum(forward_batch.extend_seq_lens_cpu) >= cp_size
    ):
        return True
    else:
        return False


from sglang.kernels.ops.attention.dsa.cp_split import (
    dsa_cp_round_robin_split_q_seqs_kernel,
)


def dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs):
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    extra_seq = 0
    q_seqs = []
    for bs, cur_len in enumerate(extend_seqs):
        cur_len += extra_seq
        cur_seq = cur_len // cp_size + int(cur_len % cp_size > cp_rank)
        q_seqs.append(cur_seq)
        extra_seq = cur_len - cur_seq * cp_size
    bs_idx = list([i for i, x in enumerate(q_seqs) if x > 0])
    q_seqs = [q_len for q_len in q_seqs if q_len > 0]
    return q_seqs, bs_idx


def dsa_cp_round_robin_split_q_seqs(
    extend_seqs_cpu, extend_seqs
) -> Tuple[List, torch.Tensor, List, torch.Tensor]:
    """
    round-robin-split distributes tokens across ranks based on token_idx % cp_size.

    Return:
    ret_q_lens_cpu(List) and ret_q_lens(torch.Tensor): the partitioned length (excluding zeros) on the current cp rank
        for each sequence after distribution across cp ranks.
    bs_idx_cpu(List) and bs_idx(torch.Tensor): marks which sequences are ultimately selected,
        i.e., those with a partitioned length greater than zero.
    """
    cp_size = get_parallel().attn_cp_size
    cp_rank = get_parallel().attn_cp_rank
    # len(ret_q_lens_cpu) == len(bs_idx_cpu)
    ret_q_lens_cpu, bs_idx_cpu = dsa_cp_round_robin_split_q_seqs_cpu(extend_seqs_cpu)
    ret_q_lens = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=extend_seqs.dtype
    )
    bs_idx = torch.empty(
        (len(bs_idx_cpu),), device=extend_seqs.device, dtype=torch.int32
    )
    grid = (1,)
    dsa_cp_round_robin_split_q_seqs_kernel[grid](
        extend_seqs, ret_q_lens, bs_idx, len(extend_seqs), cp_size, cp_rank
    )
    return ret_q_lens_cpu, ret_q_lens, bs_idx_cpu, bs_idx


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
