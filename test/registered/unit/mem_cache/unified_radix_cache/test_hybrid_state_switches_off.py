"""With the hybrid-model state switches off, the unified radix cache evicts as
it did before they existed.

A seeded multi-session workload (inserts with and without a Mamba state,
matches, locks and their release, state and token evictions) runs through a
full + Mamba tree, and one digest covers the tree and the free KV after every
step. The literals were recorded on main before the switches were added and
are compared as data: change them only for an intended change of the default
eviction order, and say why.
"""

import hashlib
import random
import unittest
from array import array
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.static_kv_storage import acquire_rows
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA

# The digest of 200 steps for each seed, recorded on main.
_DIGESTS = {
    0: "b1b6a367306f3c4b",
    1: "4660c5302f9af1a5",
}


def _snapshot(core) -> tuple:
    rows = []
    stack = [(core.root_node, ())]
    while stack:
        node, path = stack.pop()
        for child in node.children.values():
            tokens = path + tuple(child.key.raw_token_ids())
            rows.append(
                (
                    tokens,
                    child.component_data[FULL].value is not None,
                    child.component_data[MAMBA].value is not None,
                    child.component_data[FULL].lock_ref,
                )
            )
            stack.append((child, tokens))
    return tuple(sorted(rows))


def _trace_digest(seed: int, steps: int = 200) -> str:
    rng = random.Random(seed)
    cfg = CacheConfig(
        components=(FULL, MAMBA), kv_size=160, mamba_cache_size=48, max_num_reqs=1024
    )
    with patch(
        "sglang.test.kits.unified_radix_cache.fixtures.get_device", return_value="cpu"
    ):
        cache, allocator, pool = build_fixture(cfg)
    core = cache.tree_core
    sessions = [[s * 1000] for s in range(6)]
    locks = []
    digest = hashlib.sha256()
    for step in range(steps):
        op = rng.random()
        if op < 0.6:
            i = rng.randrange(len(sessions))
            if rng.random() < 0.15:
                other = sessions[rng.randrange(len(sessions))]
                sessions[i] = other[: rng.randint(1, len(other))]
            sessions[i] = sessions[i] + [
                rng.randrange(1, 900) for _ in range(rng.randint(1, 6))
            ]
            if len(sessions[i]) > 48:
                sessions[i] = [i * 1000 + step]
            need = len(sessions[i])
            if allocator.available_size() < need:
                cache.evict(EvictParams(num_tokens=need))
            if allocator.available_size() >= need:
                if cache._pool_available(MAMBA) < 1:
                    cache.evict(EvictParams(mamba_num=1))
                params = InsertParams(
                    key=RadixKey(array("q", sessions[i])), value=allocator.alloc(need)
                )
                if rng.random() < 0.85:
                    req = Req(
                        rid=f"switches-off-{step}",
                        origin_input_text="",
                        origin_input_ids=array("q"),
                        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
                    )
                    acquire_rows(pool, pool.state, [req])
                    params.mamba_value = req.kv.mamba_pool_idx.unsqueeze(0)
                else:
                    params.allow_missing_mamba = True
                cache.insert(params)
        elif op < 0.7:
            tokens = sessions[rng.randrange(len(sessions))]
            node_id = cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", tokens)))
            ).last_device_node
            if node_id != core.root_node.id:
                locks.append((node_id, core.inc_lock_ref(node_id)))
        elif op < 0.78 and locks:
            node_id, receipt = locks.pop(rng.randrange(len(locks)))
            core.dec_lock_ref(node_id, receipt.to_dec_params())
        elif op < 0.9:
            cache.evict(EvictParams(mamba_num=rng.randint(1, 3)))
        else:
            cache.evict(EvictParams(num_tokens=rng.randint(1, 8)))
        digest.update(repr((_snapshot(core), allocator.available_size())).encode())
    return digest.hexdigest()[:16]


class TestHybridStateSwitchesOff(CustomTestCase):
    def test_the_default_eviction_order_is_unchanged(self):
        for seed, expected in _DIGESTS.items():
            with self.subTest(seed=seed):
                self.assertEqual(_trace_digest(seed), expected)


if __name__ == "__main__":
    unittest.main()
