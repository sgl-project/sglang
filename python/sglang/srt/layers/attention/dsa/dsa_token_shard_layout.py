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

"""Pure index math for splitting an extend batch's tokens across the
attention-TP group: each rank takes one contiguous, equal-sized slice of the
batch's positions. The NPU DSA indexer uses it to score only its slice of the
queries.

No tensors and nothing device-specific, so the clamp arithmetic -- where a
ragged batch goes wrong silently -- is testable on CPU. The scheme is
vLLM-Ascend's ``enable_dsa_token_shard`` (``sfa_cp.py``).
"""

from itertools import accumulate
from typing import List, NamedTuple, Sequence


class DsaTokenShardPlan(NamedTuple):
    """One attention-TP rank's token slice of an extend batch.

    ``rows`` is ``ceil(num_tokens / tp_size)`` on every rank, which keeps the
    all-gather regular; ``local_end`` stops at the real tokens and
    ``local_end_with_pad`` does not.

    ``key_lens[i]`` is the KV length the LAST of request i's tokens in this slice
    sees: the operator walks back from it (``sparse_mode=3``'s right-down
    alignment), so it must be the last token's, not the first's.
    """

    num_tokens: int
    num_tokens_pad: int
    rows: int
    local_start: int
    local_end: int
    local_end_with_pad: int
    query_lens: List[int]
    key_lens: List[int]


def plan_dsa_token_shard(
    extend_seq_lens: Sequence[int],
    seq_lens: Sequence[int],
    tp_size: int,
    tp_rank: int,
) -> DsaTokenShardPlan:
    """Plan one rank's slice. ``seq_lens[i]`` is request i's prefix + extend.

    The cut is by POSITION, not by request, so a request can straddle a slice
    boundary and its lengths are recomputed against it. Padding is never in
    ``query_lens``: it lands past the last request, and the operator reads only
    up to the cumulative query lengths (as vLLM-Ascend, ``sfa_cp.py:243-256``).
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
    rows = -(-num_tokens // tp_size)
    num_tokens_pad = rows * tp_size
    local_start = tp_rank * rows
    local_end_with_pad = local_start + rows
    local_end = min(local_end_with_pad, num_tokens)

    query_lens: List[int] = []
    key_lens: List[int] = []
    start = 0
    for extend_len, seq_len in zip(extend_seq_lens, seq_lens):
        end = start + extend_len
        req_local_end = min(end, local_end_with_pad)
        n = max(0, req_local_end - max(start, local_start))
        query_lens.append(n)
        # Zero for a request with no token here: a length without a query is a
        # span the operator would read for nothing.
        key_lens.append(max(0, seq_len - (end - req_local_end)) if n else 0)
        start = end

    return DsaTokenShardPlan(
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
    """Running sum: the TND query layout takes cumulative query lengths."""
    return list(accumulate(int(n) for n in lens))
