"""KV head groups of a Muse Spark pipeline stage on ROCm.

A stage runs TP1 and holds every KV head of its layers. SGLANG_ROCM_MUSE_PP_KV_GROUPS=G keeps them as G pool
layers of heads / G heads each, group g of a layer under ``kv_group_layer_id(layer, g)``: the buffer the
attention-TP rank g of a TP-G decode holds for that layer, page for page. A PD prefill stage then sends each
decode rank the whole pages of its group.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence, Tuple

from sglang.srt.environ import envs
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

KV_GROUP_STRIDE = 1 << 10


def kv_head_groups() -> int:
    return max(1, envs.SGLANG_ROCM_MUSE_PP_KV_GROUPS.get())


def kv_group_layer_id(layer_id: int, group: int) -> int:
    return layer_id + group * KV_GROUP_STRIDE


def grouped_layer_ids(layer_ids, groups: int) -> list[int]:
    """Group-major, each group in the order of ``layer_ids``."""
    return [kv_group_layer_id(layer, g) for g in range(groups) for layer in layer_ids]


class HeadGroupedMHATokenToKVPool(MHATokenToKVPool):
    """An MHA pool that stores each layer as ``head_groups`` layers of head_num heads, group-major. For KV
    that is written here and read by a PD peer: the draft KV of a prefill stage."""

    def __init__(self, *args, head_groups: int, **kwargs):
        self.head_groups = head_groups
        super().__init__(*args, **kwargs)

    def _store_kv_layer(self, layer_idx, loc, cache_k, cache_v):
        layers = self.layer_num // self.head_groups
        rows = cache_k.shape[0]
        cache_k = cache_k.reshape(rows, self.head_groups, self.head_num, -1)
        cache_v = cache_v.reshape(rows, self.head_groups, self.head_num, -1)
        for group in range(self.head_groups):
            super()._store_kv_layer(
                group * layers + layer_idx,
                loc,
                cache_k[:, group].contiguous(),
                cache_v[:, group].contiguous(),
            )

    def set_kv_buffer_prefix_valid(self, *args, **kwargs):
        raise NotImplementedError(
            "prefix-valid commit is unsupported for KV kept as head groups"
        )


@dataclasses.dataclass(frozen=True)
class KVHeadGroups:
    """PD transfer entries of a stage with grouped KV heads. The decode registers no layer ids for a hybrid
    full / sliding-window pool, so the layers of its entries come from the model config."""

    count: int
    # (model layer, group) of every KV entry and of every window-state entry, in registration order.
    kv_entries: Tuple[Tuple[int, int], ...]
    window_entries: Tuple[Tuple[int, int], ...]
    # Model layers of the decode's target KV entries and of its window-state entries.
    peer_kv_layers: Tuple[int, ...]
    peer_window_layers: Tuple[int, ...]
    # Draft KV entries follow the target's, under layer ids from this one on.
    draft_layer_base: int
    # A draft kept on the window ids is the window state after the target's instead: its entries
    # and the layers of the decode's. Empty on a stage without the draft.
    draft_window_entries: Tuple[Tuple[int, int], ...] = ()
    peer_draft_window_layers: Tuple[int, ...] = ()

    def window_state(self, ordinal: int):
        """Entries and decode layers of the ``ordinal``-th window state: the target's, then the draft's."""
        if ordinal == 0:
            return self.window_entries, self.peer_window_layers
        return self.draft_window_entries, self.peer_draft_window_layers

    @staticmethod
    def select(
        entries: Sequence[Tuple[int, int]], group: int
    ) -> Tuple[List[int], List[int]]:
        """Indices of the group's entries and their model layers."""
        indices = [i for i, (_, g) in enumerate(entries) if g == group]
        return indices, [entries[i][0] for i in indices]

    def peer_kv_layer_ids(self, num_entries: int) -> List[int]:
        num_draft, odd = divmod(num_entries - len(self.peer_kv_layers), 2)
        if num_draft < 0 or odd:
            raise RuntimeError(
                f"The decode peer registers {num_entries} KV entries; expected K and V of "
                f"{len(self.peer_kv_layers) // 2} full-attention layers and of its draft layers"
            )
        draft = range(self.draft_layer_base, self.draft_layer_base + num_draft)
        return [*self.peer_kv_layers, *draft, *draft]


def build_kv_head_groups(
    kv_args, *, token_to_kv_pool, draft_kv_pool, draft_window_pool, model_config
) -> Optional[KVHeadGroups]:
    """``draft_kv_pool`` is a draft sent as KV entries, ``draft_window_pool`` one kept on the window
    ids and sent as the window state after the target's; a stage holds at most one of them."""
    from sglang.srt.disaggregation.base.conn import StateType

    count = kv_head_groups()
    if count == 1:
        return None

    def entries(pool):
        # K then V, each in the pool's layer order.
        return [
            divmod(layer_id, KV_GROUP_STRIDE)[::-1] for layer_id in pool.model_layer_ids
        ] * 2

    num_layers = model_config.num_hidden_layers

    def draft_entries(pool):
        if getattr(pool, "head_groups", 1) != count:
            raise NotImplementedError(
                f"The draft KV pool ({type(pool).__name__}) does not keep its heads as "
                f"{count} groups"
            )
        draft_layers = pool.layer_num // count
        return [
            (num_layers + layer, group)
            for group in range(count)
            for layer in range(draft_layers)
        ] * 2

    kv = entries(token_to_kv_pool.full_kv_pool)
    if draft_kv_pool is not None:
        kv += draft_entries(draft_kv_pool)
    window = entries(token_to_kv_pool.swa_kv_pool)
    draft_window = (
        draft_entries(draft_window_pool) if draft_window_pool is not None else []
    )
    window_states = [
        ptrs
        for state_type, ptrs in zip(kv_args.state_types, kv_args.state_data_ptrs)
        if state_type == StateType.SWA
    ]
    assert len(kv) == len(kv_args.kv_data_ptrs), (len(kv), len(kv_args.kv_data_ptrs))
    assert len(window) == len(window_states[0]), (len(window), len(window_states[0]))
    if draft_window:
        assert len(window_states) == 2 and len(draft_window) == len(window_states[1]), (
            len(draft_window),
            [len(ptrs) for ptrs in window_states],
        )
    return KVHeadGroups(
        count=count,
        kv_entries=tuple(kv),
        window_entries=tuple(window),
        peer_kv_layers=tuple(model_config.full_attention_layer_ids) * 2,
        peer_window_layers=tuple(model_config.swa_attention_layer_ids) * 2,
        draft_layer_base=num_layers,
        draft_window_entries=tuple(draft_window),
        peer_draft_window_layers=tuple(
            range(num_layers, num_layers + len(draft_window) // (2 * count))
        )
        * 2,
    )
