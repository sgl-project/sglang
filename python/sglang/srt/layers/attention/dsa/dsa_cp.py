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

"""Runtime gate for DSA-CP token sharding. The arithmetic is in dsa_cp_layout.

Split from the layout module on purpose: everything here needs the runtime
topology and the forward batch, while the layout is plain integers and can be
replayed against a brute-force reference with no torch at all. Keep it that way
-- the clamp arithmetic is where a ragged batch goes wrong, and being able to
check it locally is worth the extra file.
"""

from typing import TYPE_CHECKING, Optional

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.dsa_cp_layout import DsaCpPlan, plan_dsa_cp_shard
from sglang.srt.layers.communicator import ScatterMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import print_info_once

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Read once, at import. This is not merely a fast path: the flag decides whether
# q_b_proj and kv_b_proj are built full-width or TP-sharded, which happens when
# the model is constructed, so flipping it later would leave the weights
# disagreeing with the forward.
_enable_dsa_cp = envs.SGLANG_NPU_ENABLE_DSA_CP.get()

if _enable_dsa_cp and envs.SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING.get():
    # Loud at startup, because quiet it would be a 16x accuracy cliff rather
    # than a crash. Both flags slice the indexer's queries across attention-TP;
    # with DSA-CP the slice has already happened by the time the indexer runs,
    # so the older flag would slice it a second time and each rank would score
    # 1/256 of the queries while the top-k gather reassembled the wrong rows.
    raise ValueError(
        "SGLANG_NPU_ENABLE_DSA_CP supersedes "
        "SGLANG_NPU_ENABLE_DSA_INDEXER_QUERY_SHARDING and the two cannot both "
        "be set: DSA-CP shards the whole attention block's tokens, which "
        "includes the indexer's. Unset the indexer-only flag."
    )

if _enable_dsa_cp and envs.SGLANG_NPU_USE_MLAPO.get():
    # vLLM-Ascend refuses the same pair ("Fused preprocessing does not support
    # DSA-CP", sfa_cp.py:309). Their fused preprocess writes the KV cache from
    # inside the operator, at the batch's own slot mapping, which under DSA-CP
    # is the sliced one -- so the rows the slice does not own never get
    # written. Untested here either way; refuse rather than find out in an
    # accuracy run.
    raise ValueError(
        "SGLANG_NPU_ENABLE_DSA_CP does not compose with SGLANG_NPU_USE_MLAPO. "
        "The fused MLA preprocess writes the KV cache itself, at a slot mapping "
        "DSA-CP has already sliced."
    )


class _Missing:
    """Distinguishes "no plan cached yet" from "cached, and it is None"."""


_MISSING = _Missing()


def dsa_cp_enabled() -> bool:
    """Whether DSA-CP is on for this process. Read at model build AND forward.

    Both sides must agree, so both ask here rather than reading the env var
    twice.
    """
    return _enable_dsa_cp and get_parallel().attn_tp_size > 1


def get_dsa_cp_plan(
    forward_batch: "ForwardBatch",
    layer_scatter_modes=None,
) -> Optional[DsaCpPlan]:
    """This forward's token slice for this rank, or None if DSA-CP is off here.

    Built once per forward and cached on the batch: the split is by token
    position and does not vary by layer.

    **Extend only, deliberately, for the first version.** The measured 34% is
    entirely in the tail prefill, and decode is already fast from graph capture,
    so leaving decode on the unsharded path keeps the capture untouched and
    removes most of the risk. vLLM-Ascend does slice at decode too
    (``sfa_cp.py``), and composes with DCP by MRO; that is a later step, not a
    prerequisite.

    Every refusal is logged once. The indexer-sharding bug this supersedes cost
    four weeks precisely because its refusal was silent.
    """
    if not _enable_dsa_cp:
        return None

    cached = getattr(forward_batch, "npu_dsa_cp_plan", _MISSING)
    if cached is not _MISSING:
        return cached

    plan = _build_dsa_cp_plan(forward_batch, layer_scatter_modes)
    forward_batch.npu_dsa_cp_plan = plan
    return plan


def _build_dsa_cp_plan(forward_batch, layer_scatter_modes) -> Optional[DsaCpPlan]:
    parallel = get_parallel()
    if parallel.attn_tp_size <= 1:
        print_info_once("DSA-CP is off: attention TP size is 1, nothing to shard")
        return None

    mode = forward_batch.forward_mode
    if not mode.is_extend():
        # Decode and the speculative modes keep the unsharded path. Not a
        # defect: see the docstring.
        return None
    if mode.is_draft_extend_v2() or mode.is_target_verify():
        print_info_once(
            f"DSA-CP is off for {mode}: the draft and verify paths carry their "
            "own per-step sequence-length tables, which this does not build yet"
        )
        return None

    if (
        layer_scatter_modes is not None
        and layer_scatter_modes.attn_mode != ScatterMode.TP_ATTN_FULL
    ):
        # The slice assumes this rank was handed the whole batch, which is what
        # TP_ATTN_FULL means. Under any other mode it already holds a piece and
        # this would cut it twice.
        print_info_once(
            "DSA-CP is off: attention scatter mode is "
            f"{layer_scatter_modes.attn_mode}, not TP_ATTN_FULL"
        )
        return None

    extend_lens = forward_batch.extend_seq_lens_cpu
    prefix_lens = forward_batch.extend_prefix_lens_cpu
    if not extend_lens or prefix_lens is None:
        print_info_once("DSA-CP is off: this extend carries no CPU length metadata")
        return None

    plan = plan_dsa_cp_shard(
        extend_lens,
        [p + e for p, e in zip(prefix_lens, extend_lens)],
        parallel.attn_tp_size,
        parallel.attn_tp_rank,
    )
    if plan.num_tokens < parallel.attn_tp_size:
        # Fewer tokens than ranks: most ranks would hold nothing but padding and
        # the all-gathers would move more than the attention saves.
        print_info_once(
            f"DSA-CP is off for batches under {parallel.attn_tp_size} tokens "
            f"(this one has {plan.num_tokens}); the slice would be mostly padding"
        )
        return None
    return plan
