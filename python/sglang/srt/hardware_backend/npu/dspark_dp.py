"""NPU hooks for GLM DSpark budget synchronization and packed DP inputs.

The shared DSpark planner owns cross-DP tier gathering and layout selection.
"""

import logging
from functools import partial

import torch

from sglang.srt.layers.dp_attention import get_dp_tp_group
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dspark_components.dspark_planner import DSparkVerifyHooks
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)

logger = logging.getLogger(__name__)


def make_dspark_verify_hooks(model_runner) -> DSparkVerifyHooks | None:
    # Worker construction precedes attention-backend initialization.
    from sglang.srt.configs.model_config import is_glm_moe_dsa

    config = getattr(model_runner, "model_config", None)
    if config is None or not is_glm_moe_dsa(config.hf_config):
        return None
    coordinate_dp_verify = (
        get_parallel().attn_dp_enabled
        and read_ragged_verify_mode() is RaggedVerifyMode.COMPACT
    )
    if coordinate_dp_verify and get_parallel().tp_rank == 0:
        logger.info(
            "GLM DSpark compact attention DP enabled (attn_dp=%d, attn_tp=%d, "
            "budget_sync=attention-tp, target_tier_sync=tp-cpu).",
            get_parallel().attn_dp_size,
            get_parallel().attn_tp_size,
        )
    return DSparkVerifyHooks(
        sync_budget=partial(sync_verify_budget, model_runner=model_runner),
        prepare_forward_batch=(
            partial(prepare_compact_dp_forward_batch, model_runner=model_runner)
            if coordinate_dp_verify
            else None
        ),
        dynamic_tiers_with_dp=True,
        coordinate_dp_verify=coordinate_dp_verify,
    )


def sync_verify_budget(budget, *, model_runner):
    # Resolve the capability late. A missing confidence-ring snapshot is also
    # a decision: broadcast None within the group owning the same requests.
    if getattr(
        getattr(model_runner, "attn_backend", None),
        "_dspark_dsa_ragged_verify",
        False,
    ):
        return get_dp_tp_group().broadcast_object(budget, src=0)
    return budget


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
    forward_batch.dp_verify_uses_packed_tokens = True
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
