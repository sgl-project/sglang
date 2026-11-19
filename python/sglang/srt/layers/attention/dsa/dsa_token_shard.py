# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Runtime gate and tensor plumbing for DSA token sharding.

Only the query path is sharded: after ``q_b_proj`` and the ``w_kc`` absorb a
rank swaps "my heads for every token" for "every head for my tokens", and the
swap is undone after attention, before ``w_vc``. The KV write and the indexer
stay full width.

Composes with the indexer's query sharding: the indexer derives its rows
from this plan, so both pick the same rows.
"""

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_token_shard_layout import (
    DsaTokenShardPlan,
    cumulative,
    plan_dsa_token_shard,
)
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.layer_boundary import get_attn_tp_context
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_npu, print_info_once

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def dsa_token_shard_enabled() -> bool:
    """Asked at model build AND at forward, so both agree.

    ``is_npu()``: ``deepseek_v2.py`` is shared by every backend, and a true
    answer builds an extra full-head ``RadixAttention`` only NPU ever uses.
    """
    return is_npu() and get_parallel().attn_tp_size > 1


# Distinguishes "no plan cached yet" from "cached, and it is None".
_MISSING = object()


def get_dsa_token_shard_plan(
    forward_batch: "ForwardBatch",
) -> Optional[DsaTokenShardPlan]:
    """This forward's token slice for this rank, or None. Cached on the batch.

    Extend only: decode is already fast under graph capture. Every refusal is
    logged once, because a silent refusal looks the same as the feature running.
    """
    cached = getattr(forward_batch, "npu_dsa_token_shard_plan", _MISSING)
    if cached is not _MISSING:
        return cached
    plan = _build_dsa_token_shard_plan(forward_batch)
    forward_batch.npu_dsa_token_shard_plan = plan
    return plan


def _build_dsa_token_shard_plan(forward_batch) -> Optional[DsaTokenShardPlan]:
    parallel = get_parallel()
    if parallel.attn_tp_size <= 1:
        print_info_once("DSA token-shard is off: attention TP size is 1")
        return None

    if parallel.dcp_enabled:
        # DCP's DSA path all-gathers the query across the DCP group, which
        # expects every head-sharded query row, not this rank's token slice.
        print_info_once("DSA token-shard is off: DCP all-gathers the attention query")
        return None

    if forward_batch.attn_cp_metadata is not None and is_dsa_enable_prefill_cp():
        # Prefill CP already splits tokens across the CP group and hands
        # attention its own prev/next query layout, which knows nothing of
        # this slice.
        print_info_once("DSA token-shard is off: DSA prefill CP splits the tokens")
        return None

    # The same modes the indexer query shard admits.
    if forward_batch.forward_mode not in (ForwardMode.EXTEND, ForwardMode.MIXED):
        return None

    if get_attn_tp_context().input_scattered:
        # A scattered input is already sliced; slicing it again cuts twice.
        print_info_once(
            "DSA token-shard is off: the attention input is scattered across the "
            "TP group"
        )
        return None

    extend_lens = forward_batch.extend_seq_lens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if not extend_lens or prefix_lens is None:
        print_info_once(
            "DSA token-shard is off: this extend carries no CPU length metadata"
        )
        return None

    plan = plan_dsa_token_shard(
        extend_lens,
        [p + e for p, e in zip(prefix_lens, extend_lens)],
        parallel.attn_tp_size,
        parallel.attn_tp_rank,
    )
    if plan.num_tokens < parallel.attn_tp_size:
        print_info_once(
            f"DSA token-shard is off for batches under {parallel.attn_tp_size} "
            f"tokens (this one has {plan.num_tokens}): the slice would be mostly "
            "padding"
        )
        return None

    # Attention-DP MAX_LEN pads a rank to the widest DP rank, past the
    # attn_tp_size alignment the plan covers. Refused here, before the indexer
    # reads the plan, so neither it nor attention ever takes a token slice.
    num_rows = forward_batch.input_ids.shape[0]
    if num_rows > plan.num_tokens_pad:
        print_info_once(
            f"DSA token-shard is off for batches padded past the plan ({num_rows} "
            f"rows for {plan.num_tokens} tokens): attention-DP MAX_LEN padding"
        )
        return None

    # Logged on engage too: an unset flag returns before every refusal, so the
    # absence of a refusal proves nothing.
    print_info_once(
        f"DSA token-shard is ON: the attention query is sharded across "
        f"{parallel.attn_tp_size} ranks at extend"
    )
    return plan


def dsa_token_shard_cumulative_query_lens(
    forward_batch: "ForwardBatch", plan: DsaTokenShardPlan, device: torch.device
) -> torch.Tensor:
    """Cumulative query lengths as an int32 device tensor, built once per
    forward rather than per layer: each build is a blocking H2D copy."""
    cached = getattr(forward_batch, "npu_dsa_token_shard_cu_query_lens", None)
    if cached is None:
        cached = torch.tensor(
            cumulative(plan.query_lens), dtype=torch.int32, device=device
        )
        forward_batch.npu_dsa_token_shard_cu_query_lens = cached
    return cached


def dsa_token_shard_key_lens(
    forward_batch: "ForwardBatch", plan: DsaTokenShardPlan, device: torch.device
) -> torch.Tensor:
    """One absolute KV length per request, the same shape as seq_lens.

    Paged SFA sparse_mode=3 right-aligns each request to this length.
    A running sum would only match when the forward has one request.
    """
    cached = getattr(forward_batch, "npu_dsa_token_shard_key_lens", None)
    if cached is None:
        cached = torch.tensor(list(plan.key_lens), dtype=torch.int32, device=device)
        forward_batch.npu_dsa_token_shard_key_lens = cached
    return cached


def dsa_token_shard_slice(x: torch.Tensor, plan: DsaTokenShardPlan) -> torch.Tensor:
    """This rank's ``plan.rows`` rows, zero-padded at the tail to the row count
    the query has after the all-to-all."""
    sliced = x[plan.local_start : plan.local_end]
    missing = plan.rows - sliced.shape[0]
    if missing <= 0:
        return sliced
    return torch.cat([sliced, sliced.new_zeros((missing, *x.shape[1:]))], dim=0)


def dsa_token_shard_redistribute_heads(
    x: torch.Tensor, plan: DsaTokenShardPlan
) -> torch.Tensor:
    """``[num_tokens, h, d]`` -> ``[rows, h * tp_size, d]``, by one all-to-all.

    Heads come out in global order: chunk ``s`` comes from rank ``s``, which a
    ColumnParallelLinear gives heads ``[s*h, (s+1)*h)``.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    h, d = x.shape[1], x.shape[2]
    assert x.shape[0] <= plan.num_tokens_pad, (
        f"DSA token-shard was handed {x.shape[0]} rows but planned for at most "
        f"{plan.num_tokens_pad}. The batch is padded by a wider rule than "
        "attn_tp_size; the plan must be built from the padded width."
    )
    missing = plan.num_tokens_pad - x.shape[0]
    if missing > 0:
        x = torch.cat([x, x.new_zeros((missing, h, d))], dim=0)
    # reshape, not view: q_nope_out arrives non-contiguous from
    # npu_transpose_batchmatmul.
    send = x.reshape(tp, plan.rows, h, d).contiguous()
    recv = torch.empty_like(send)
    parallel.attn_tp_group.all_to_all_single(recv, send)
    return recv.permute(1, 0, 2, 3).reshape(plan.rows, tp * h, d)


def dsa_token_shard_restore_tokens(
    x: torch.Tensor, plan: DsaTokenShardPlan, num_rows: int
) -> torch.Tensor:
    """``[rows, h * tp_size, d]`` -> ``[num_rows, h, d]``, the inverse.

    ``num_rows`` must be the width handed to the redistribute: a width that
    arrived padded must leave padded, or the next layer's KV write indexes a
    narrower tensor than its slot map. Rows past ``num_tokens`` are zeroed -- the operator never wrote
    them, and untouched device memory can be NaN.
    """
    parallel = get_parallel()
    tp = parallel.attn_tp_size
    h, d = x.shape[1] // tp, x.shape[2]
    assert x.shape[1] == h * tp, (
        f"DSA token-shard restore expects a full head set, got {x.shape[1]} for "
        f"tp_size {tp}"
    )
    assert plan.num_tokens <= num_rows <= plan.num_tokens_pad, (
        f"DSA token-shard restore asked for {num_rows} rows, outside "
        f"[{plan.num_tokens}, {plan.num_tokens_pad}]"
    )
    send = x.reshape(plan.rows, tp, h, d).permute(1, 0, 2, 3).contiguous()
    recv = torch.empty_like(send)
    parallel.attn_tp_group.all_to_all_single(recv, send)
    out = recv.reshape(plan.num_tokens_pad, h, d)[:num_rows]
    if num_rows > plan.num_tokens:
        out[plan.num_tokens :].zero_()
    return out


def dsa_cp_stream_o_proj() -> bool:
    return envs.SGLANG_NPU_STREAM_O_PROJ.get()


def _validate_dsa_cp_token_sharded(model, spec_algorithm):
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

    parallel = get_parallel()
    if not dsa_token_shard_enabled():
        raise ValueError("DSA-CP token sharding requires NPU and attention TP > 1")
    if not parallel.dcp_enabled:
        raise ValueError("DSA-CP token sharding requires DCP")
    if parallel.dcp_replicate_q_proj:
        raise ValueError(
            "DSA-CP token sharding and --dcp-replicate-q-proj use different "
            "groups for the same cached weights; enable only one"
        )
    if not get_moe_a2a_backend().is_none():
        raise ValueError("DSA-CP token sharding requires the tensor-parallel MoE path")
    if spec_algorithm is not None and not spec_algorithm.is_none():
        raise ValueError("DSA-CP token sharding does not support speculative decoding")
    if is_dp_attention_enabled():
        raise ValueError("DSA-CP token sharding does not support DP attention")
    if parallel.enable_prefill_cp:
        raise ValueError("DSA-CP token sharding does not support prefill CP")
    if envs.SGLANG_NPU_USE_MLAPO.get():
        raise ValueError("DSA-CP token sharding does not support MLAPO")
    if envs.SGLANG_SHARED_EXPERT_TP1.get():
        raise ValueError("DSA-CP token sharding does not support TP1 shared experts")

    layers = [
        module
        for module in model.modules()
        if isinstance(module, DeepseekV2AttentionMLA) and module.use_dsa
    ]
    if not layers:
        raise ValueError("DSA-CP token sharding requires a model with DSA layers")
    if any(module.w_kc is None or not module.has_q_b_proj for module in layers):
        raise ValueError(
            "DSA-CP token sharding requires MLA q_b/kv_b weights on every DSA layer"
        )
    return layers


def _dequantized_linear_weight(layer, input_size, device, dtype) -> torch.Tensor:
    output = layer(torch.eye(input_size, device=device, dtype=dtype))[0]
    return output.t().contiguous()


def prepare_dsa_cp_token_sharded(*, model, device, dtype, spec_algorithm) -> None:
    """Prepare full-head attention weights for the opt-in DCP residual rebuild."""
    if not envs.SGLANG_NPU_DSA_CP_TOKEN_SHARDED.get():
        return
    layers = _validate_dsa_cp_token_sharded(model, spec_algorithm)
    tp_group = get_parallel().attn_tp_group
    stream_o_proj = dsa_cp_stream_o_proj()
    for module in layers:
        q_b = _dequantized_linear_weight(
            module.q_b_proj, module.q_lora_rank, device, dtype
        )
        module.q_b_proj_qrep_weight = tp_group.all_gather(q_b, dim=0)
        module.w_kc_qrep = tp_group.all_gather(module.w_kc.contiguous(), dim=0)
        module.w_vc_qrep = tp_group.all_gather(module.w_vc.contiguous(), dim=0)
        o_proj = _dequantized_linear_weight(
            module.o_proj, module.o_proj.input_size_per_partition, device, dtype
        )
        if stream_o_proj:
            module.o_proj_stream_shard = o_proj
        else:
            module.o_proj_qrep_weight = tp_group.all_gather(o_proj, dim=1)


def select_dsa_cp_token_sharded(forward_batch) -> bool:
    """Decide once, after the DCP gather plan and before the model forward."""
    metadata = forward_batch.attn_dcp_metadata
    if (
        not envs.SGLANG_NPU_DSA_CP_TOKEN_SHARDED.get()
        or metadata.extend_gather_plan is None
        or get_dsa_token_shard_plan(forward_batch) is None
    ):
        return False

    from sglang.srt.layers.logits_processor import extend_reads_input_logprobs

    # Input logprobs consume hidden states at positions outside the logits tail.
    if extend_reads_input_logprobs(forward_batch):
        return False

    from sglang.srt.model_executor.runner import get_is_capture_mode

    if get_is_capture_mode():
        raise RuntimeError(
            "DSA-CP token sharding cannot capture an extend graph; "
            "its collectives depend on the batch token grid"
        )
    return True


def dsa_cp_gather_token_rows(
    x: torch.Tensor, forward_batch: "ForwardBatch"
) -> torch.Tensor:
    """Restore full token rows from this rank's contiguous attention-TP slice."""
    plan = get_dsa_token_shard_plan(forward_batch)
    if plan is None:
        raise RuntimeError("DSA-CP token sharding lost its forward plan")
    full = x.new_empty((plan.num_tokens_pad,) + x.shape[1:])
    get_parallel().attn_tp_group.all_gather_into_tensor(full, x)
    return full[: plan.num_tokens]
