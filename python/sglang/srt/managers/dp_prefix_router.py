# Copyright 2023-2024 SGLang Team
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
"""Prefix-aware DP rank selection for the in-process DataParallelController.

Each DP rank has its own radix cache, so a request only hits the cache when it
lands on the rank that served its prefix. ``round_robin`` and the load-based
methods ignore that: successive turns of a multi-turn session land on
arbitrary ranks, and the hit rate drops towards ``1 / dp_size`` of what a
single cache would give.

``PrefixAffinityIndex`` is an approximate, per-rank copy of what each rank's
radix cache holds, built from the prompts the controller has routed there. A
prompt is cut into fixed-size token blocks, and each block is keyed by a hash
chained over every block before it, so one dict lookup answers "does this rank
hold the first ``i`` blocks of this prompt". Each rank keeps an LRU of its
block keys with a token budget, so prefixes the rank has likely evicted stop
attracting requests.

``PrefixAwareSelector`` picks the rank with the longest matched prefix, with
two load guards borrowed from sgl-model-gateway's cache-aware policy:

- a rank whose load exceeds the least-loaded rank by more than
  ``balance_abs_threshold`` requests *and* by more than
  ``balance_rel_threshold`` times is not a candidate;
- a match shorter than ``cache_threshold`` of the prompt goes to the
  least-loaded candidate instead, unless it ends exactly where a prompt
  previously routed to that rank ended.

The ratio guard keeps a system prompt shared by every session from pulling all
new sessions to the first rank that cached it, as long as the shared part is
under ``cache_threshold`` of the prompt. A longer shared part passes it: new
sessions then pack onto the ranks holding it until the overload guard trips,
and each one that spills elsewhere copies the shared part to another rank.
Once every rank holds it, equal matches tie-break on load.

The exemption keeps a session on its rank when the turn adds more than the
history the index knows about. The index records routed prompts only, never
generated tokens, so a turn that appends a long reply or tool result matches
just the previous prompt, though the rank's radix cache holds the reply too.

TODO: the router can't tell a prefix every session shares (cheap to replicate,
worth spreading) from one session's history (worth keeping on its rank). A
per-key successor count would tell them apart; a score of queue delay plus
uncached prefill would replace both thresholds.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

MIN_PREFIX_BLOCK_SIZE = 64


def prefix_block_size(page_size: int | None) -> int:
    """Block size for the prefix index: whole radix pages, at least 64 tokens.

    Many backends default ``page_size`` to 1; hashing every token and keeping
    one dict entry per token would bog down the single-threaded DP controller
    and blow up index memory on large KV pools. Rounding 64 up to a multiple
    of ``page_size`` bounds controller work when ``page_size`` is small while
    matching whole radix pages when ``page_size >= 64``.
    """
    ps = max(1, int(page_size or 1))
    return ((MIN_PREFIX_BLOCK_SIZE + ps - 1) // ps) * ps


class PrefixAffinityIndex:
    """Approximate per-rank prefix index over chained token-block hashes."""

    def __init__(self, num_ranks: int, block_size: int, capacity_blocks: int):
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if capacity_blocks <= 0:
            raise ValueError(f"capacity_blocks must be positive, got {capacity_blocks}")
        self.block_size = block_size
        self.capacity_blocks = capacity_blocks
        # Per rank: block key -> whether a routed prompt ended there, oldest
        # first.
        self._blocks: list[OrderedDict[int, bool]] = [
            OrderedDict() for _ in range(num_ranks)
        ]

    def block_keys(self, input_ids: Sequence[int], seed: int = 0) -> list[int]:
        """Chained keys of the prompt's full blocks; a partial tail is ignored."""
        bs = self.block_size
        keys = []
        key = seed
        for start in range(0, len(input_ids) - bs + 1, bs):
            key = hash((key, tuple(input_ids[start : start + bs])))
            keys.append(key)
        return keys

    def match(self, keys: Sequence[int]) -> list[int]:
        """Number of leading blocks of ``keys`` each rank holds."""
        matched = [0] * len(self._blocks)
        alive = list(range(len(self._blocks)))
        for i, key in enumerate(keys):
            alive = [r for r in alive if key in self._blocks[r]]
            if not alive:
                break
            for r in alive:
                matched[r] = i + 1
        return matched

    def insert(self, rank: int, keys: Sequence[int]) -> None:
        """Record that ``rank`` now caches the prompt behind ``keys``.

        Keys are touched from the last block to the first, so a prefix is
        always more recent than its extensions and LRU eviction trims leaves
        first, as a radix cache does. ``match`` relies on that: a rank that
        holds block ``i`` also holds blocks ``0..i-1``.
        """
        blocks = self._blocks[rank]
        last = len(keys) - 1
        for i in range(last, -1, -1):
            key = keys[i]
            blocks[key] = blocks.pop(key, False) or i == last
        while len(blocks) > self.capacity_blocks:
            blocks.popitem(last=False)

    def ends_prompt(self, rank: int, keys: Sequence[int], num_matched: int) -> bool:
        """Whether a prompt routed to ``rank`` ended at matched block ``num_matched``."""
        return num_matched > 0 and self._blocks[rank].get(keys[num_matched - 1], False)

    def num_blocks(self, rank: int) -> int:
        return len(self._blocks[rank])


@dataclass
class PrefixRouteStats:
    """Counters for the periodic routing log line."""

    requests: int = 0
    by_prefix: int = 0
    prompt_tokens: int = 0
    matched_tokens: int = 0

    def reset(self) -> None:
        self.requests = self.by_prefix = 0
        self.prompt_tokens = self.matched_tokens = 0


class PrefixAwareSelector:
    """Choose a DP rank by longest cached prefix, guarded by load."""

    def __init__(
        self,
        num_ranks: int,
        block_size: int,
        capacity_blocks: int,
        cache_threshold: float,
        balance_abs_threshold: int,
        balance_rel_threshold: float,
        log_interval: int = 0,
    ):
        self.index = PrefixAffinityIndex(num_ranks, block_size, capacity_blocks)
        self.cache_threshold = cache_threshold
        self.balance_abs_threshold = balance_abs_threshold
        self.balance_rel_threshold = balance_rel_threshold
        self.log_interval = log_interval
        self.stats = PrefixRouteStats()

    def _overloaded(self, load: int, min_load: int) -> bool:
        return (
            load - min_load > self.balance_abs_threshold
            and load > min_load * self.balance_rel_threshold
        )

    def select(
        self,
        input_ids: Sequence[int],
        loads: Sequence[int],
        active: Sequence[int],
        seed: int = 0,
    ) -> int:
        """Return the rank for this prompt and record it in the index.

        ``loads`` is indexed by rank (e.g. running + waiting requests);
        ``active`` lists the ranks that may receive requests; ``seed``
        namespaces block keys by e.g. ``(lora_id, extra_key, cache_salt)``.
        """
        if not active:
            raise RuntimeError("No active DP workers are available for routing.")
        keys = self.index.block_keys(input_ids, seed=seed)
        matched = self.index.match(keys)
        min_load = min(loads[r] for r in active)
        candidates = [r for r in active if not self._overloaded(loads[r], min_load)]

        best = max(candidates, key=lambda r: (matched[r], -loads[r]))
        matched_tokens = matched[best] * self.index.block_size
        # A match that ends where an earlier prompt to this rank ended is that
        # session's history, which the rank caches together with the reply the
        # index never saw; the ratio only guards matches ending mid-prompt.
        by_prefix = matched[best] > 0 and (
            matched_tokens >= self.cache_threshold * len(input_ids)
            or self.index.ends_prompt(best, keys, matched[best])
        )
        if not by_prefix:
            best = min(candidates, key=lambda r: (loads[r], -matched[r]))
            matched_tokens = matched[best] * self.index.block_size

        self.index.insert(best, keys)
        self._record(len(input_ids), matched_tokens, by_prefix, loads, active)
        return best

    def _record(self, prompt_tokens, matched_tokens, by_prefix, loads, active):
        if self.log_interval <= 0:
            return
        s = self.stats
        s.requests += 1
        s.by_prefix += int(by_prefix)
        s.prompt_tokens += prompt_tokens
        s.matched_tokens += matched_tokens
        if s.requests >= self.log_interval:
            logger.info(
                "Prefix-aware DP routing: %d reqs, %.1f%% routed by prefix, "
                "%.1f%% of prompt tokens matched on the chosen rank, "
                "loads=%s, indexed blocks=%s",
                s.requests,
                100.0 * s.by_prefix / s.requests,
                100.0 * s.matched_tokens / max(s.prompt_tokens, 1),
                [loads[r] for r in active],
                [self.index.num_blocks(r) for r in active],
            )
            s.reset()
