"""SWAKVPool must hand its sub-pools both layer ids.

A quantized sub-pool indexes its own buffers by the sub-pool-local layer index but
reads its per-layer global KV scales by the model's global layer id. SWAKVPool is the
only place that knows both, so if it drops one the scales silently come from the wrong
layer -- no error, just wrong numbers.
"""

import types

import torch

from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool


class _RecordingPool:
    """Stands in for an MHA sub-pool; records what the dispatch handed it."""

    def __init__(self):
        self.calls = []

    def set_kv_buffer(
        self,
        layer,
        loc,
        cache_k,
        cache_v,
        k_scale=None,
        v_scale=None,
        layer_id_override=None,
    ):
        self.calls.append(
            (
                None if layer is None else layer.layer_id,
                layer_id_override,
            )
        )


def _pool(full_layer_ids, swa_layer_ids):
    full, swa = _RecordingPool(), _RecordingPool()
    p = types.SimpleNamespace(full_kv_pool=full, swa_kv_pool=swa, layers_mapping={})
    for local, gid in enumerate(full_layer_ids):
        p.layers_mapping[gid] = (local, False)
    for local, gid in enumerate(swa_layer_ids):
        p.layers_mapping[gid] = (local, True)
    p.set_kv_buffer = types.MethodType(SWAKVPool.set_kv_buffer, p)
    return p, full, swa


def test_interleaved_layers_get_the_global_id_and_the_local_index():
    """The layout that makes the bug visible: global ids interleave, so a local index
    collides with a different layer's global id in the other sub-pool."""
    # full = 0, 2, 4 ; swa = 1, 3, 5
    p, full, swa = _pool([0, 2, 4], [1, 3, 5])
    loc = KVWriteLoc(loc=torch.arange(4), swa_loc=torch.arange(4), physical=True)
    for gid in range(6):
        layer = types.SimpleNamespace(layer_id=gid)
        p.set_kv_buffer(layer, loc, torch.zeros(4), torch.zeros(4))

    assert full.calls == [(0, 0), (2, 1), (4, 2)]
    assert swa.calls == [(1, 0), (3, 1), (5, 2)]
    # The point: for every call the two differ except at layer 0, so a sub-pool that
    # saw only the local index would read the wrong per-layer scale.
    for global_id, local_index in full.calls[1:] + swa.calls:
        assert global_id != local_index or global_id == 0


def test_every_global_layer_reaches_exactly_one_sub_pool():
    p, full, swa = _pool([0, 1, 2], [3, 4, 5])
    loc = KVWriteLoc(loc=torch.arange(2), swa_loc=torch.arange(2), physical=True)
    for gid in range(6):
        p.set_kv_buffer(
            types.SimpleNamespace(layer_id=gid), loc, torch.zeros(2), torch.zeros(2)
        )
    seen = [c[0] for c in full.calls] + [c[0] for c in swa.calls]
    assert sorted(seen) == list(range(6))
