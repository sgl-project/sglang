import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    DevicePoolEntry,
    DevicePoolGroup,
)
from sglang.srt.mem_cache.storage.local_store import local_direct_linker
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, enter_override

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

PAGE = 4


def _kv(keys, first_page=None):
    indices = None
    if first_page is not None:
        indices = torch.arange(first_page * PAGE, (first_page + len(keys)) * PAGE)
    return [PoolTransfer(name=PoolName.KV, device_indices=indices, keys=list(keys))]


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestLocalDirectLinker(CustomTestCase):
    def _linker(self, num_slots):
        # Token-row KV with a draft buffer packed onto layer 0, and page-row indexer.
        self.kv = [
            torch.zeros(64, 1, 16, dtype=torch.bfloat16, device="cuda")
            for _ in range(3)
        ]
        self.indexer = [
            torch.zeros(16, 40, dtype=torch.uint8, device="cuda") for _ in range(2)
        ]

        def entry(name, buffers, mapping, rows_are_pages):
            return DevicePoolEntry(
                name=name,
                indices_from_pool=PoolName.KV,
                device_pool=None,
                components=[buffers],
                layer_mapping=mapping,
                page_size=PAGE,
                rows_are_pages=rows_are_pages,
            )

        group = DevicePoolGroup(
            [
                entry(PoolName.KV, self.kv, {0: (0, 2), 1: 1}, False),
                entry(PoolName.INDEXER, self.indexer, {0: 0, 1: 1}, True),
            ],
            num_layers=2,
            page_size=PAGE,
            rank_replicated=True,
        )
        page_bytes = 3 * PAGE * 32 + 2 * 40
        config = {"capacity_gb": (num_slots + 0.5) * page_bytes / 1024**3}
        enter_override(
            self,
            get_context().override_server_args(
                hicache_storage_backend_extra_config=json.dumps(config)
            ),
        )
        params = SimpleNamespace(
            page_size=PAGE,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
            attn_tp_cache_group=None,
            tp_cache_group=None,
        )
        with patch.object(
            local_direct_linker, "resolve_hybrid_device_pool_group", return_value=group
        ):
            linker = local_direct_linker.LocalDirectLinker(None, params, components=())
        self.addCleanup(linker.close)
        return linker

    def _offload(self, linker, keys, first_page):
        for buffer in self.kv:
            buffer[first_page * PAGE : (first_page + len(keys)) * PAGE].normal_()
        for buffer in self.indexer:
            buffer[first_page : first_page + len(keys)].random_(0, 256)
        linker.offload(_kv(keys, first_page))
        torch.cuda.synchronize()
        self.assertEqual(linker.num_completed_offloads(), 1)
        return linker.pop_completed_offload()

    def test_offload_then_load_restores_every_layer(self):
        linker = self._linker(num_slots=4)
        self.assertTrue(self._offload(linker, ["p0", "p1"], first_page=1))
        self.assertEqual(linker.lookup("rid", _kv(["p0", "p1", "px"])), [1, 2])

        self.assertTrue(linker.load("rid", _kv(["p1"], first_page=9)))
        index = linker.start_layer_wise_loading()
        linker.layer_done_counter.set_consumer(index)
        linker.layer_done_counter.wait_until(1)
        self.assertTrue(linker.finish_layer_wise_loading(index))

        for buffer in self.kv:
            self.assertTrue(torch.equal(buffer[36:40], buffer[8:12]))
        for buffer in self.indexer:
            self.assertTrue(torch.equal(buffer[9], buffer[2]))

    def test_full_store_fails_offload_until_lookup_releases(self):
        linker = self._linker(num_slots=1)
        self._offload(linker, ["p0"], first_page=1)
        linker.lookup("rid", _kv(["p0"]))
        self.assertFalse(self._offload(linker, ["p1"], first_page=2))
        linker.release_lookup("rid")
        self.assertTrue(self._offload(linker, ["p1"], first_page=2))


if __name__ == "__main__":
    unittest.main()
