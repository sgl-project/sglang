"""Model-free AITER KV-write regression for MI35x, including graph replay."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
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

        from sglang.srt.layers.attention.dsa_backend import (
            DeepseekSparseAttnBackend,
            DSAMetadata,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        cls.backend_type = DeepseekSparseAttnBackend
        cls.metadata_type = DSAMetadata
        cls.forward_mode = ForwardMode
        cls.forward = forward_mla_rocm
        cls.pool_type = HiSparseDSATokenToKVPool

    def setUp(self):
        super().setUp()
        # The owned guard catches out-of-range writes without corrupting other tensors.
        self.backing = torch.full(
            (24, 1, 576), -99.0, dtype=torch.bfloat16, device="cuda"
        )
        self.mapping = torch.zeros(65, dtype=torch.int64, device="cuda")
        self.mapping[17], self.mapping[18] = 3, 5
        self.mapping[-1] = -1
        self.backend = self.backend_type.__new__(self.backend_type)
        self.backend.token_to_kv_pool = self.pool_type.__new__(self.pool_type)
        self.pool.register_mapping(self.mapping)
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
        self.backend.forward_metadata = self.make_metadata()

    @property
    def pool(self):
        return self.backend.token_to_kv_pool

    def make_metadata(self):
        lengths = torch.ones(3, dtype=torch.int32, device="cuda")
        offsets = torch.arange(4, dtype=torch.int32, device="cuda")
        page_table = torch.zeros((3, 1), dtype=torch.int32, device="cuda")
        return self.metadata_type(
            page_size=1,
            cache_seqlens_int32=lengths,
            max_seq_len_q=1,
            max_seq_len_k=1,
            cu_seqlens_q=offsets,
            cu_seqlens_k=offsets,
            page_table_1=page_table,
            real_page_table=page_table,
            dsa_cache_seqlens_int32=lengths,
            dsa_cu_seqlens_q=offsets,
            dsa_cu_seqlens_k=offsets,
            dsa_extend_seq_lens_list=[1, 1, 1],
            dsa_seqlens_expanded=lengths,
        )

    def write(self, forward_batch=None):
        if forward_batch is None:
            forward_batch = SimpleNamespace(
                out_cache_loc=self.locations, forward_mode=self.forward_mode.EXTEND
            )
        with forward_context(ForwardContext(attn_backend=self.backend)):
            return self.forward._fused_rope_cat_and_cache(
                self.attn,
                self.qn,
                self.qr,
                self.kn,
                self.kr,
                self.positions,
                forward_batch,
            )

    def test_hoisted_two_layer_write_and_mutable_graph(self):
        second_cache = torch.full_like(self.backing, -99.0)
        self.pool.kv_buffer = [self.backing[:8], second_cache[:8]]
        batch = SimpleNamespace(
            out_cache_loc=self.locations, forward_mode=self.forward_mode.DECODE
        )

        def layers(*args):
            results = []
            for layer in range(2):
                self.attn.attn_mqa.layer_id = self.pool.start_layer + layer
                results.append(self.write(batch))
            return results

        def forward():
            # Mirrors decode capture: preparation is recorded before any writer.
            self.backend.init_forward_metadata_in_graph(batch)
            return layers()

        def check(expected):
            self.check_cache(expected)
            torch.testing.assert_close(second_cache, self.backing, rtol=0, atol=0)

        mapping = self.pool.full_to_hisparse_device_index_mapping
        with patch.object(
            self.pool,
            "translate_loc_to_hisparse_device",
            wraps=self.pool.translate_loc_to_hisparse_device,
        ) as translate:
            for dtype in (torch.int32, torch.int64):
                with self.subTest(dtype=dtype):
                    mapping[17], mapping[18] = 3, 5
                    self.locations = torch.tensor(
                        [17, 18, -1], dtype=dtype, device="cuda"
                    )
                    batch.out_cache_loc = self.locations
                    self.backend.forward_metadata = self.make_metadata()
                    self.backing.fill_(-99.0)
                    second_cache.fill_(-99.0)
                    before = translate.call_count
                    forward()
                    self.assertEqual(translate.call_count, before + 1)
                    check({0: 3, 1: 5})
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        forward()
                    self.assertEqual(translate.call_count, before + 2)
                    self.backing.fill_(-99.0)
                    second_cache.fill_(-99.0)
                    graph.replay()
                    check({0: 3, 1: 5})
                    # Both values change while their underlying objects stay live.
                    mapping[17], mapping[18] = 4, 6
                    self.locations.copy_(
                        torch.tensor([18, -1, 17], dtype=dtype, device="cuda")
                    )
                    self.backing.fill_(-99.0)
                    second_cache.fill_(-99.0)
                    # Out-of-graph metadata may be replaced for replay. The
                    # recorded translate and writers retain their captured addresses.
                    self.backend.forward_metadata = self.make_metadata()
                    graph.replay()
                    check({0: 6, 2: 4})
                    torch.testing.assert_close(
                        self.locations,
                        torch.tensor([18, -1, 17], dtype=dtype, device="cuda"),
                    )

    def check_cache(self, row_to_slot):
        torch.cuda.synchronize()
        expected = torch.full_like(self.backing, -99.0)
        for row, slot in row_to_slot.items():
            expected[slot] = self.expected_rows[row]
        torch.testing.assert_close(self.backing, expected, rtol=0, atol=0)

    def test_eager_and_graph_writes_use_physical_slots(self):
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
        self.backend.token_to_kv_pool = resident
        self.locations = torch.tensor([3, 21, 5, 22, -1, 23], device="cuda")[::2]
        self.write()
        with self.subTest(layout="strided"):
            self.check_cache({0: 3, 1: 5})


if __name__ == "__main__":
    unittest.main()
