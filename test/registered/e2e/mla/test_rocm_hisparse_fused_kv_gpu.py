"""Model-free AITER KV-write regression for MI35x, including graph replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")


class TestRocmHiSparseFusedKVKernel(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.version.hip or not torch.cuda.is_available():
            raise AssertionError("The MI35x regression requires ROCm and an AMD GPU")
        arch = torch.cuda.get_device_properties(0).gcnArchName
        if not arch.startswith("gfx95"):
            raise AssertionError("The MI35x regression requires gfx95 hardware")

        from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseDSATokenToKVPool
        from sglang.srt.models.deepseek_common.attention_forward_methods import (
            forward_mla_rocm,
        )

        if not forward_mla_rocm._use_aiter_gfx95:
            raise AssertionError(
                "This gfx95 regression must run with the fused MLA path; "
                "set SGLANG_USE_AITER=1 before starting Python"
            )

        cls.forward = forward_mla_rocm
        cls.pool_type = HiSparseDSATokenToKVPool

    def setUp(self):
        super().setUp()
        # The owned guard catches out-of-range writes without corrupting other tensors.
        self.backing = torch.full(
            (24, 1, 576), -99.0, dtype=torch.bfloat16, device="cuda"
        )
        mapping = torch.zeros(65, dtype=torch.int64, device="cuda")
        mapping[17], mapping[18] = 3, 5
        mapping[-1] = -1
        self.pool = self.pool_type.__new__(self.pool_type)
        self.pool.register_mapping(mapping)
        self.pool.kv_buffer = [self.backing[:8]]
        self.pool.start_layer = 7
        self.pool.layer_transfer_counter = None
        self.pool.dtype = self.pool.store_dtype = torch.bfloat16
        self.attn = SimpleNamespace(
            kv_cache_dtype="bfloat16",
            current_attention_backend="dsa",
            attn_mqa=SimpleNamespace(layer_id=7, k_scale=torch.ones(1, device="cuda")),
            rotary_emb=SimpleNamespace(
                cos_cache=torch.ones((8, 64), dtype=torch.bfloat16, device="cuda"),
                sin_cache=torch.zeros((8, 64), dtype=torch.bfloat16, device="cuda"),
                is_neox_style=False,
            ),
        )
        self.qn = torch.ones((3, 8, 512), dtype=torch.bfloat16, device="cuda")
        self.qr = torch.ones((3, 8, 64), dtype=torch.bfloat16, device="cuda")
        self.kn = (
            torch.arange(1, 4, dtype=torch.bfloat16, device="cuda")[:, None, None]
            .expand(3, 1, 512)
            .contiguous()
        )
        self.kr = torch.full((3, 1, 64), 2.5, dtype=torch.bfloat16, device="cuda")
        self.positions = torch.zeros(3, dtype=torch.int64, device="cuda")
        self.locations = torch.tensor([17, 18, -1], device="cuda")
        self.expected_rows = torch.cat((self.kn, self.kr), dim=-1)

    def write(self):
        return self.forward._fused_rope_cat_and_cache(
            self.attn,
            self.qn,
            self.qr,
            self.kn,
            self.kr,
            self.positions,
            self.locations,
        )

    def check_cache(self, row_to_slot):
        torch.cuda.synchronize()
        expected = torch.full_like(self.backing, -99.0)
        for row, slot in row_to_slot.items():
            expected[slot] = self.expected_rows[row]
        torch.testing.assert_close(self.backing, expected, rtol=0, atol=0)

    def test_eager_and_graph_writes_use_physical_slots(self):
        with patch.object(self.forward, "get_token_to_kv_pool", return_value=self.pool):
            # The helper calls the real AITER kernel with the pool's device buffer.
            self.write()
            with self.subTest(mode="eager"):
                self.check_cache({0: 3, 1: 5})
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self.write()
            self.backing.fill_(-99.0)
            graph.replay()
            with self.subTest(mode="graph"):
                self.check_cache({0: 3, 1: 5})

            # Inputs are mutable between graph replays, including padding.
            self.locations.copy_(torch.tensor([18, -1, 17], device="cuda"))
            self.backing.fill_(-99.0)
            graph.replay()
            with self.subTest(mode="graph-updated-inputs"):
                self.check_cache({0: 5, 2: 3})

            self.locations.copy_(torch.tensor([19, -1, 17], device="cuda"))
            self.backing.fill_(-99.0)
            graph.replay()
            with self.subTest(mode="graph-unmapped-dummy-slot"):
                self.check_cache({0: 0, 2: 3})

    def test_resident_strided_locations(self):
        """AITER must write selected slots rather than interleaved storage values."""
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

        resident = DSATokenToKVPool.__new__(DSATokenToKVPool)
        resident.__dict__.update(self.pool.__dict__)
        self.pool = resident
        self.locations = torch.tensor([3, 21, 5, 22, -1, 23], device="cuda")[::2]
        with patch.object(self.forward, "get_token_to_kv_pool", return_value=self.pool):
            self.write()
        with self.subTest(layout="strided"):
            self.check_cache({0: 3, 1: 5})


if __name__ == "__main__":
    unittest.main()
