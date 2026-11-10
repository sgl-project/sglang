"""GLM DSpark compact coordination across attention-DP groups.

Dense drafts run independently on attention TP. Only target verification
joins the full TP/EP group, including ranks without local requests. Exchange
host sizes once at that boundary; never broadcast one DP group's budget to
another. No NPU runtime import is needed by these contracts.
"""

import logging

import msgspec
import torch

from sglang.srt.distributed.utils import all_gather_single
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout, round_up_grid

logger = logging.getLogger(__name__)


def compact_dp_graph_tokens(contributions, *, attn_tp_size, runner):
    """Return a common token width, or None for a collective eager fallback.

    Contributions are (local requests, local padded tokens), repeated on each
    attention-TP peer. Maxima deliberately do not double count those peers.
    A busy rank without a compact layout contributes -1; all ranks must then
    use the existing uniform eager verify path.
    """
    max_reqs = max((row[0] for row in contributions), default=0)
    if max_reqs == 0 or any(row[1] < 0 for row in contributions):
        return None
    tokens = max(max_reqs, max(row[1] for row in contributions))
    tokens = (tokens + attn_tp_size - 1) // attn_tp_size * attn_tp_size
    if runner is not None and runner.ragged_verify_mode:
        if max_reqs > runner.max_bs or tokens > runner.capture_num_tokens[-1]:
            return None
        tokens = round_up_grid(tokens, runner.capture_num_tokens)
        if max_reqs > runner._ragged_capture_slots(tokens):
            return None
    return tokens


def coordinate_compact_dp_layout(*, batch, layout, model_runner):
    """Called exactly once by busy AND idle ranks, after local dense draft."""
    parallel = get_parallel()
    bs = batch.batch_size()
    contribution = [
        bs,
        layout.graph_num_tokens if layout is not None else (-1 if bs else 0),
    ]
    cpu_group = parallel.tp_group.cpu_group
    local = torch.tensor(contribution, dtype=torch.int64)
    gathered = torch.empty(
        (torch.distributed.get_world_size(group=cpu_group) * 2,), dtype=torch.int64
    )
    all_gather_single(gathered, local, group=cpu_group)
    contributions = gathered.view(-1, 2).tolist()
    tokens = compact_dp_graph_tokens(
        contributions,
        attn_tp_size=parallel.attn_tp_size,
        runner=model_runner.decode_cuda_graph_runner,
    )
    if envs.SGLANG_DSPARK_DEBUG_CONFIDENCE_PREFIX_SCHEDULER.get():
        if parallel.attn_tp_rank == 0:
            logger.info(
                "[DSPARK-DP] dp_rank=%d local_reqs=%d local_tokens=%d target_tokens=%s",
                parallel.attn_dp_rank,
                bs,
                contribution[1],
                tokens,
            )
    if tokens is None:
        return None
    if bs:
        return msgspec.structs.replace(layout, graph_num_tokens=tokens)
    # Empty device-only rows describe no real request. All padded queries go
    # to the GLM metadata's separate sink slot, never request-pool row zero.
    return RaggedVerifyLayout.from_verify_lens_device(
        verify_lens=torch.zeros(1, dtype=torch.int32, device=model_runner.device),
        graph_num_tokens=tokens,
    )


def is_glm_compact_dp_batch(forward_batch, model_runner):
    return (
        get_parallel().attn_dp_enabled
        and getattr(model_runner.attn_backend, "_dspark_dsa_ragged_verify", False)
        and forward_batch.spec_info is not None
        and getattr(forward_batch.spec_info, "ragged_verify_layout", None) is not None
    )


def prepare_compact_dp_forward_batch(forward_batch, model_runner):
    """Publish packed TOKEN counts instead of request-count * fixed window.

    Both eager and graph forwards use the agreed width. Request tensors keep
    their local row count: a packed token width cannot be divided by gamma+1
    to recover a request count.
    """
    if not is_glm_compact_dp_batch(forward_batch, model_runner):
        return
    tokens = forward_batch.spec_info.ragged_verify_layout.graph_num_tokens
    forward_batch.npu_dspark_compact_dp = True
    counts = [tokens] * get_parallel().num_dp_ranks
    forward_batch.global_num_tokens_cpu = counts
    forward_batch.global_num_tokens_for_logprob_cpu = counts.copy()
    for name in ("global_num_tokens_gpu", "global_num_tokens_for_logprob_gpu"):
        tensor = getattr(forward_batch, name)
        if tensor is None or tensor.numel() != len(counts):
            tensor = torch.empty(
                len(counts), dtype=torch.int64, device=model_runner.device
            )
            setattr(forward_batch, name, tensor)
        tensor.fill_(tokens)
    # Padding is valid model input routed only to sink KV. Acceptance and
    # output publication still use the local real requests and verify_lens.
    if forward_batch.global_num_token_non_padded is not None:
        forward_batch.global_num_token_non_padded.fill_(tokens)
    forward_batch.global_num_token_non_padded_cpu = tokens
