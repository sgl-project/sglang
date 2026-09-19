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

"""Pure index math for DSA-CP: sharding an extend batch's TOKENS inside the
attention-TP group.

Attention-TP splits heads. DSA-CP splits tokens, over the same ranks and
without consuming any, so both are on at once: a rank computes *every* head for
*its slice* of the batch instead of *its heads* for *every* token. The compute
is identical either way -- 4 heads x 13,855 tokens and 64 heads x 866 tokens are
the same multiply -- but the sparse operator is bound by the top-k KV it reads,
and in MLA every head of a query reads the same latent row. So the KV traffic
follows the token count, not the head count, and slicing tokens divides it by
``tp_size`` where slicing heads does not divide it at all.

This module is only the layout: which tokens a rank owns, and what per-request
query and key lengths the operator must be told about them. It holds no tensors
and imports nothing device-specific, so it can be tested on CPU -- which is the
point, because the clamp arithmetic below is where a ragged batch goes wrong and
the failure is silent (attention reads the wrong span and still returns
plausible text).

Named for vLLM-Ascend's ``enable_dsa_cp``
(``vllm_ascend/attention/context_parallel/sfa_cp.py``), whose scheme this is.
"""

from typing import List, NamedTuple, Sequence


class DsaCpPlan(NamedTuple):
    """One attention-TP rank's token slice of an extend batch.

    ``rows`` is the slice width and is the same on every rank, which is what
    makes the all-gathers that rebuild the KV regular. It is
    ``ceil(num_tokens / tp_size)``, so the last rank's slice can run past the
    real tokens; ``local_end`` stops at the real ones and ``local_end_with_pad``
    does not.

    ``query_lens[i]`` is how many of request i's tokens fall in this slice, and
    ``key_lens[i]`` is the KV length the LAST of them sees. The operator infers
    the earlier ones by walking back from it (the right-down causal alignment
    ``sparse_mode=3`` applies), so one length per request is enough and it has
    to be the last token's, not the first's.
    """

    num_tokens: int
    num_tokens_pad: int
    rows: int
    local_start: int
    local_end: int
    local_end_with_pad: int
    query_lens: List[int]
    key_lens: List[int]

    @property
    def num_local_tokens(self) -> int:
        """Real tokens in this slice; ``rows`` minus whatever padding it holds."""
        return max(0, self.local_end - self.local_start)

    def is_empty(self) -> bool:
        return self.num_local_tokens == 0


def plan_dsa_cp_shard(
    extend_seq_lens: Sequence[int],
    seq_lens: Sequence[int],
    tp_size: int,
    tp_rank: int,
) -> DsaCpPlan:
    """Plan one rank's token slice of an extend batch.

    ``extend_seq_lens[i]`` is how many new tokens request i contributes to this
    forward and ``seq_lens[i]`` is its total KV length once they are written --
    prefix plus extend. The batch's tokens are laid out request-major, exactly
    as ``hidden_states`` arrives, and cut into ``tp_size`` equal slices.

    **The cut is by position, not by request.** A request can straddle a slice
    boundary, and at a 13,855-token tail over 16 ranks it usually does. So the
    per-request lengths have to be recomputed against the slice:

    - ``query_lens[i]`` is request i's tokens inside ``[local_start,
      local_end_with_pad)``, clamped at both ends and floored at zero for the
      requests that miss the slice entirely.
    - ``key_lens[i]`` is what the slice's last token of request i can see.
      Request i's tokens end at global position ``end``; the slice truncates
      them at ``req_local_end``; so the slice's last token is ``end -
      req_local_end`` positions earlier than the batch's last token for that
      request, and sees that many fewer keys: ``seq_lens[i] - (end -
      req_local_end)``.

    **Padding is never counted in ``query_lens``, and it cannot be.** The padded
    positions are ``[num_tokens, num_tokens_pad)``, and every request ends at or
    before ``num_tokens``, so padding always lands past the last request rather
    than inside one. The rank holding it is handed ``rows`` query rows while its
    ``query_lens`` sum to fewer; the operator reads the cumulative lengths to
    find where requests end and never looks at the rest. That is also how
    vLLM-Ascend leaves it (``sfa_cp.py:243-256``).

    Returns a plan whose ``query_lens`` sum to at most ``rows``.
    """
    assert tp_size >= 1, f"tp_size must be positive, got {tp_size}"
    assert 0 <= tp_rank < tp_size, f"tp_rank {tp_rank} outside [0, {tp_size})"

    extend_seq_lens = [int(x) for x in extend_seq_lens]
    seq_lens = [int(x) for x in seq_lens]
    assert len(extend_seq_lens) == len(seq_lens), (
        f"{len(extend_seq_lens)} extend lengths against {len(seq_lens)} sequence "
        "lengths; they index the same requests"
    )

    num_tokens = sum(extend_seq_lens)
    rows = -(-num_tokens // tp_size)  # ceil
    num_tokens_pad = rows * tp_size
    local_start = tp_rank * rows
    local_end_with_pad = local_start + rows
    local_end = min(local_end_with_pad, num_tokens)

    query_lens: List[int] = []
    key_lens: List[int] = []
    start = 0
    for extend_len, seq_len in zip(extend_seq_lens, seq_lens):
        end = start + extend_len
        req_local_start = max(start, local_start)
        req_local_end = min(end, local_end_with_pad)
        n = max(0, req_local_end - req_local_start)
        query_lens.append(n)
        # Zero, not seq_len, for a request with no token here: a length without
        # a query is a span the operator would read for nothing.
        key_lens.append(max(0, seq_len - (end - req_local_end)) if n else 0)
        start = end

    return DsaCpPlan(
        num_tokens=num_tokens,
        num_tokens_pad=num_tokens_pad,
        rows=rows,
        local_start=local_start,
        local_end=local_end,
        local_end_with_pad=local_end_with_pad,
        query_lens=query_lens,
        key_lens=key_lens,
    )


def cumulative(lens: Sequence[int]) -> List[int]:
    """Running sum, which is how the NPU sparse operator wants query lengths.

    ``actual_seq_lengths_query`` is cumulative and ``actual_seq_lengths_kv`` is
    cumulative too under the non-paged TND layout this backend uses at DCP
    extend -- measured, not inferred, by
    ``glm5.2_testing/p6_prefill_nonpaged_sfa_probe.py``: of the four
    combinations of {relative, absolute} indices and {per-batch, cumulative}
    lengths, exactly one reproduced a float64 reference and the other three ran
    and returned plausible garbage. The plan keeps per-request lengths because
    that is what the arithmetic is natural in; the call site applies this.
    """
    out: List[int] = []
    total = 0
    for n in lens:
        total += int(n)
        out.append(total)
    return out
