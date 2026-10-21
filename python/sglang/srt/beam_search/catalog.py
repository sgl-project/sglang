# Copyright 2026 SGLang Team
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
"""Immutable, fixed-depth semantic item catalogs for native GR beam search.

CPU validation and item lookup are separate from device selection. Device storage
is O(edges), with O(live beams * vocabulary) masking workspace; there is no dense
node-by-vocabulary table or per-step host trie walk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ItemCatalog:
    vocab_size: int
    depth: int
    root_width: int
    edges: tuple[tuple[int, int], ...]
    items: dict[tuple[int, ...], str | int]

    @classmethod
    def from_records(cls, records, vocab_size: int):
        if vocab_size < 1:
            raise ValueError("Catalog vocabulary size must be positive.")
        children = [{}]
        items, ids, depths = {}, set(), set()
        for row, record in enumerate(records, 1):
            if not isinstance(record, dict):
                raise ValueError(f"Catalog row {row} must be a JSON object.")
            item = record.get("item_id")
            tokens = record.get("token_ids")
            if type(item) not in (str, int):
                raise ValueError(
                    f"Catalog row {row}: item_id must be a string or integer."
                )
            if (
                not isinstance(tokens, list)
                or not tokens
                or any(type(t) is not int or not 0 <= t < vocab_size for t in tokens)
            ):
                raise ValueError(
                    f"Catalog row {row}: token_ids must be nonempty vocabulary-bounded integers."
                )
            path = tuple(tokens)
            if item in ids:
                raise ValueError(f"Duplicate item_id at catalog row {row}.")
            if path in items:
                raise ValueError(f"Duplicate token path at catalog row {row}.")
            items[path] = item
            ids.add(item)
            depths.add(len(path))
            node = 0
            for token in path:
                if token not in children[node]:
                    children[node][token] = len(children)
                    children.append({})
                node = children[node][token]
        if not items:
            raise ValueError("Catalog must contain at least one item.")
        if len(depths) != 1:
            raise ValueError("GR catalogs currently require fixed-depth token paths.")
        edges = tuple(
            (node * vocab_size + token, child)
            for node, arcs in enumerate(children)
            for token, child in sorted(arcs.items())
        )
        return cls(vocab_size, depths.pop(), len(children[0]), edges, items)

    def validate_budget(self, max_new_tokens: int) -> None:
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= self.depth:
            raise ValueError(
                f"gr_catalog requires max_new_tokens in [1, {self.depth}]."
            )

    def resolve(self, tokens) -> dict[str, Any]:
        item = self.items.get(tuple(tokens))
        complete = tuple(tokens) in self.items
        return {
            "item_id": item,
            "item_ids": [item] if complete else [],
            "is_complete": complete,
        }


@lru_cache(maxsize=1)
def load_catalogs(manifest: str, vocab_size: int) -> dict[str, ItemCatalog]:
    """Load once per process; the configured manifest and files are immutable."""
    path = Path(manifest)
    entries = json.loads(path.read_text())
    if not isinstance(entries, dict) or not entries:
        raise ValueError(
            "GR catalog manifest must be a nonempty name-to-JSONL-path object."
        )
    result = {}
    for name, filename in entries.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(filename, str)
            or not filename
        ):
            raise ValueError(
                "GR catalog manifest names and paths must be nonempty strings."
            )
        catalog_path = path.parent / filename
        records = [
            json.loads(line)
            for line in catalog_path.read_text().splitlines()
            if line.strip()
        ]
        result[name] = ItemCatalog.from_records(records, vocab_size)
    return result


def get_catalog(manifest: str | None, name: str, vocab_size: int) -> ItemCatalog:
    if not manifest:
        raise ValueError("gr_catalog requires --gr-catalog-manifest.")
    try:
        catalogs = load_catalogs(manifest, vocab_size)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load GR catalogs: {exc}") from exc
    if name not in catalogs:
        raise ValueError(f"Unknown gr_catalog: {name!r}.")
    return catalogs[name]


class DeviceItemCatalog:
    def __init__(self, catalog: ItemCatalog, device):
        import torch

        self.catalog = catalog
        self.keys = torch.tensor(
            [e[0] for e in catalog.edges], dtype=torch.int64, device=device
        )
        self.children = torch.tensor(
            [e[1] for e in catalog.edges], dtype=torch.int64, device=device
        )
        self.vocabulary = torch.arange(catalog.vocab_size, device=device)

    def initial_nodes(self):
        return self.keys.new_zeros(1)

    def allowed(self, nodes):
        import torch

        queries = nodes[:, None] * self.catalog.vocab_size + self.vocabulary
        positions = torch.searchsorted(self.keys, queries)
        return self.keys[positions.clamp_max(self.keys.numel() - 1)] == queries

    def advance(self, nodes, parents, tokens):
        import torch

        keys = nodes[parents] * self.catalog.vocab_size + tokens
        positions = torch.searchsorted(self.keys, keys)
        # Selected tokens are legal by construction. Clamp keeps exceptional
        # non-finite model logits from causing an out-of-bounds device access;
        # the deferred commit rejects their selection before returning output.
        return self.children[positions.clamp_max(self.keys.numel() - 1)]

    def topk(self, logits, nodes, width: int, *, prefill: bool):
        import torch

        masked = logits.float().masked_fill(~self.allowed(nodes), -torch.inf)
        if prefill:
            # SID initial selection: normalize the whole masked vocabulary,
            # then select once. Preserve its tie order in the coordinator.
            return torch.topk(torch.log_softmax(masked, dim=-1), width, dim=-1)
        # SID batched decode: local top-K, subtract legal-child logsumexp;
        # the coordinator adds parent scores and performs the global top-K.
        values, tokens = torch.topk(masked, width, dim=-1)
        return values - torch.logsumexp(masked, dim=-1, keepdim=True), tokens
