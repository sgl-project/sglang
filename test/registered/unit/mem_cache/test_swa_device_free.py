"""Differential test for SGLANG_OPT_SWA_DEVICE_FREE.

Drives identical request lifecycles (alloc_extend, alloc_decode, rolling
eviction free_swa, composite free, free_group) against a legacy and a
device-free SWATokenToKVPoolAllocator in lockstep, checking after every op:

- available_size() (full and swa) identical across the two implementations
- per-request mapping liveness pattern identical (global slot patterns
  legitimately diverge: legacy free list is LIFO, the device ring is FIFO)
- ring conservation on both pools: no duplicate pages, free set plus live
  pages covers the whole pool

Also proves the composite free path emits no CPU-GPU sync under
torch.cuda.set_sync_debug_mode("error").
"""

import random
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator.paged import DeviceFreeListPagedAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

DEVICE = "cuda"
PAGE_SIZE = 16
NUM_FULL_PAGES = 128
NUM_SWA_PAGES = 64


class _MockSWAPool(BaseSWAKVPool):
    def __init__(self):
        self.full_kv_pool = None
        self.swa_kv_pool = None
        self.full_to_swa_index_mapping = None

    def register_mapping(self, m):
        self.full_to_swa_index_mapping = m

    def translate_loc_from_full_to_swa(self, kv_indices):
        return self.full_to_swa_index_mapping[kv_indices]

    def get_state_buf_infos(self):
        return [], [], []

    def get_kv_size_bytes(self):
        return 0

    def get_key_buffer(self, layer_id):
        raise NotImplementedError

    def get_value_buffer(self, layer_id):
        raise NotImplementedError

    def get_kv_buffer(self, layer_id):
        raise NotImplementedError

    def set_kv_buffer(self, *a, **k):
        raise NotImplementedError


def _build(device_free: bool):
    with envs.SGLANG_OPT_SWA_DEVICE_FREE.override(device_free):
        alloc = SWATokenToKVPoolAllocator(
            NUM_FULL_PAGES * PAGE_SIZE,
            NUM_SWA_PAGES * PAGE_SIZE,
            PAGE_SIZE,
            torch.float16,
            DEVICE,
            _MockSWAPool(),
            need_sort=False,
        )
    if device_free:
        assert isinstance(alloc.swa_attn_allocator, DeviceFreeListPagedAllocator)
    return alloc


class _Driver:
    """Drives one composite through request lifecycles, tracking host truth."""

    def __init__(self, alloc):
        self.a = alloc
        self.reqs = {}

    def extend(self, rid, seq_len):
        pre = torch.zeros(1, dtype=torch.int64, device=DEVICE)
        pre_cpu = torch.zeros(1, dtype=torch.int64)
        sl = torch.tensor([seq_len], dtype=torch.int64, device=DEVICE)
        sl_cpu = torch.tensor([seq_len], dtype=torch.int64)
        last = torch.tensor([-1], dtype=torch.int64, device=DEVICE)
        out = self.a.alloc_extend(pre, pre_cpu, sl, sl_cpu, last, seq_len)
        assert out is not None
        self.reqs[rid] = {"idx": out, "seq_len": seq_len, "evicted": 0}

    def decode(self, rid, steps):
        r = self.reqs[rid]
        for _ in range(steps):
            new_len = r["seq_len"] + 1
            sl = torch.tensor([new_len], dtype=torch.int64, device=DEVICE)
            sl_cpu = torch.tensor([new_len], dtype=torch.int64)
            out = self.a.alloc_decode(sl, sl_cpu, r["idx"][-1:])
            assert out is not None
            r["idx"] = torch.cat([r["idx"], out])
            r["seq_len"] = new_len

    def evict_swa(self, rid, upto_pages):
        r = self.reqs[rid]
        upto = min(upto_pages * PAGE_SIZE, r["seq_len"] // PAGE_SIZE * PAGE_SIZE)
        if upto <= r["evicted"]:
            return
        self.a.free_swa(r["idx"][r["evicted"] : upto])
        r["evicted"] = upto

    def finish(self, rid, group=False):
        r = self.reqs.pop(rid)
        if group:
            self.a.free_group_begin()
        self.a.free(r["idx"])
        if group:
            self.a.free_group_end()


class TestSWADeviceFree(CustomTestCase):
    def _check(self, legacy, devfree):
        self.assertEqual(
            legacy.a.full_available_size(), devfree.a.full_available_size()
        )
        self.assertEqual(legacy.a.swa_available_size(), devfree.a.swa_available_size())
        lp = legacy.a.full_to_swa_index_mapping > 0
        dp = devfree.a.full_to_swa_index_mapping > 0
        self.assertEqual(int(lp.sum()), int(dp.sum()))
        for rid in devfree.reqs:
            ml = legacy.a.full_to_swa_index_mapping[legacy.reqs[rid]["idx"]] > 0
            md = devfree.a.full_to_swa_index_mapping[devfree.reqs[rid]["idx"]] > 0
            self.assertTrue(torch.equal(ml, md), f"req {rid} mapping diverged")
        # ring conservation on both device-backed pools
        for sub, num_pages, live_pages in (
            (
                devfree.a.swa_attn_allocator,
                NUM_SWA_PAGES,
                set(
                    torch.unique(
                        devfree.a.full_to_swa_index_mapping[:-1][
                            devfree.a.full_to_swa_index_mapping[:-1] > 0
                        ]
                        // PAGE_SIZE
                    ).tolist()
                ),
            ),
            (
                devfree.a.full_attn_allocator,
                NUM_FULL_PAGES,
                {
                    p
                    for r in devfree.reqs.values()
                    for p in torch.unique(r["idx"] // PAGE_SIZE).tolist()
                },
            ),
        ):
            occ = int(sub._tail - sub._head)
            idx = (sub._head + torch.arange(occ, device=DEVICE)) % sub._cap
            ring = sub._buf[idx]
            self.assertEqual(len(torch.unique(ring)), occ, "duplicate pages in ring")
            got = set(ring.tolist()) | live_pages
            self.assertEqual(got, set(range(1, num_pages + 1)), "leak or alien page")

    def test_lockstep_differential(self):
        random.seed(0)
        torch.manual_seed(0)
        legacy = _Driver(_build(False))
        devfree = _Driver(_build(True))

        def both(fn):
            fn(legacy)
            fn(devfree)

        # structured lifecycle: unaligned tails, boundary decodes, partial
        # eviction, grouped finish
        both(lambda d: d.extend("a", 5 * PAGE_SIZE))
        both(lambda d: d.extend("b", 3 * PAGE_SIZE + 7))
        both(lambda d: d.decode("b", PAGE_SIZE))
        both(lambda d: d.evict_swa("a", 2))
        both(lambda d: d.finish("a"))
        both(lambda d: d.finish("b", group=True))
        self._check(legacy, devfree)

        # randomized churn; draw randomness OUTSIDE the lambda so both
        # drivers see identical ops
        live = []
        for i in range(120):
            op = random.random()
            if op < 0.45 or not live:
                rid = f"r{i}"
                need = random.randint(1, 6) * PAGE_SIZE + random.randint(
                    0, PAGE_SIZE - 1
                )
                if legacy.a.available_size() < need + PAGE_SIZE:
                    continue
                both(lambda d: d.extend(rid, need))
                live.append(rid)
            elif op < 0.65:
                rid = random.choice(live)
                steps = random.randint(1, PAGE_SIZE)
                both(lambda d: d.decode(rid, steps))
            elif op < 0.8:
                rid = random.choice(live)
                upto = random.randint(1, 4)
                both(lambda d: d.evict_swa(rid, upto))
            else:
                rid = live.pop(random.randrange(len(live)))
                grouped = random.random() < 0.5
                both(lambda d: d.finish(rid, group=grouped))
            self._check(legacy, devfree)
        for rid in list(live):
            both(lambda d: d.finish(rid))
        self._check(legacy, devfree)

    def test_free_path_has_no_sync(self):
        devfree = _Driver(_build(True))
        devfree.extend("warm", 6 * PAGE_SIZE)
        devfree.finish("warm")  # warm the jit kernel before the probe
        torch.cuda.synchronize()
        devfree.extend("x", 6 * PAGE_SIZE)
        idx = devfree.reqs["x"]["idx"]
        torch.cuda.set_sync_debug_mode("error")
        try:
            devfree.a.free(idx)
        finally:
            torch.cuda.set_sync_debug_mode("default")


if __name__ == "__main__":
    unittest.main()
