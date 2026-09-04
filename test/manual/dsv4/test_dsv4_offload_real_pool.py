"""DSV4 host offload against a real pool built from the model config.

The CPU unit tests (``test/registered/mem_cache/test_dsv4_cpu_copy.py``) cover
the region wiring with a stub fixture. This one builds the actual
``DeepSeekV4TokenToKVPool`` + ``SWATokenToKVPoolAllocator`` from
DeepSeek-V4-Flash's ``config.json`` -- real 44-layer ``compress_ratios`` with
dense (ratio 0) layers, real fp8 row bytes, real ring sizes, and the real
``full_to_swa_index_mapping`` produced by ``alloc_extend_swa_tail`` -- then runs
the save/restore that PD decode retraction performs.

Needs one GPU and the model *config* only (no weights, ~250MB of pools).

    SGLANG_TEST_DSV4_MODEL=<path> python -m pytest \
        test/manual/dsv4/test_dsv4_offload_real_pool.py -v
"""

import os
import unittest

import torch

DEFAULT_MODEL = (
    "/root/paddlejob/share-storage/gpfs/system-public/inference_models/"
    "DeepSeek-V4/DeepSeek-V4-Flash"
)
MODEL = os.environ.get("SGLANG_TEST_DSV4_MODEL", DEFAULT_MODEL)
PAGE_SIZE = 256
FULL_PAGES = 32
NUM_REQ_SLOTS = 8
SEQ_LEN = 600


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
@unittest.skipUnless(os.path.isdir(MODEL), f"needs a DSV4 model config at {MODEL}")
class TestDSV4OffloadRealPool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt import runtime_context as rc
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4TokenToKVPool,
        )
        from sglang.srt.server_args import ServerArgs

        server_args = ServerArgs(
            model_path=MODEL, trust_remote_code=True, page_size=PAGE_SIZE
        )
        rc.publish(server_args, role="test-dsv4-offload")
        cls.model_config = ModelConfig.from_server_args(server_args)

        ratios = cls.model_config.compress_ratios
        full_size = PAGE_SIZE * FULL_PAGES
        cls.pool = DeepSeekV4TokenToKVPool(
            max_num_reqs=NUM_REQ_SLOTS,
            num_req_slots=NUM_REQ_SLOTS,
            swa_size=full_size,
            c4_size=full_size // 4,
            c128_size=full_size // 128,
            c4_state_pool_size=FULL_PAGES * 16,
            c128_state_pool_size=NUM_REQ_SLOTS * 256,
            page_size=PAGE_SIZE,
            swa_page_size=PAGE_SIZE,
            sliding_window=cls.model_config.window_size,
            dtype=torch.float8_e4m3fn,
            c4_state_dtype=torch.float32,
            c128_state_dtype=torch.float32,
            qk_nope_head_dim=cls.model_config.qk_nope_head_dim,
            qk_rope_head_dim=cls.model_config.qk_rope_head_dim,
            indexer_head_dim=cls.model_config.index_head_dim,
            layer_num=len(ratios),
            device="cuda",
            enable_memory_saver=False,
            compression_ratios=ratios,
        )
        cls.allocator = SWATokenToKVPoolAllocator(
            full_size,
            full_size,
            PAGE_SIZE,
            torch.float8_e4m3fn,
            "cuda",
            cls.pool,
            need_sort=False,
        )

    @classmethod
    def tearDownClass(cls):
        del cls.allocator, cls.pool
        torch.cuda.empty_cache()

    def _alloc(self, seq_len: int) -> torch.Tensor:
        """Mirror ``DecodePreallocQueue``: the SWA window tail is page-aligned."""
        window_start = max(0, seq_len - self.model_config.window_size)
        window_start = (window_start // PAGE_SIZE) * PAGE_SIZE
        return self.allocator.alloc_extend_swa_tail(
            prefix_lens=torch.tensor([0], dtype=torch.int64, device="cuda"),
            prefix_lens_cpu=torch.tensor([0], dtype=torch.int64),
            seq_lens=torch.tensor([seq_len], dtype=torch.int64, device="cuda"),
            seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int64),
            last_loc=torch.tensor([-1], dtype=torch.int64, device="cuda"),
            extend_num_tokens=seq_len,
            swa_tail_len=seq_len - window_start,
        )

    def _fill(self, seed: int) -> None:
        generator = torch.Generator(device="cuda").manual_seed(seed)
        for region in self.pool.iter_kv_regions():
            for tensor in region.tensors:
                if tensor.dtype == torch.uint8:
                    tensor.copy_(
                        torch.randint(
                            1,
                            255,
                            tensor.shape,
                            generator=generator,
                            dtype=torch.uint8,
                            device="cuda",
                        )
                    )
                else:
                    tensor.copy_(
                        torch.randn(
                            tensor.shape, generator=generator, device="cuda"
                        ).to(tensor.dtype)
                    )

    def _snapshot(self, indices: torch.Tensor, req_pool_idx: int) -> dict:
        from sglang.srt.mem_cache.kv_region import RequestCtx

        ctx = RequestCtx(token_indices=indices, req_pool_idx=req_pool_idx)
        return {
            region.name: [
                tensor[region.addressing.save_plan(ctx)[0]].clone()
                for tensor in region.tensors
            ]
            for region in self.pool.iter_kv_regions()
        }

    def test_retract_round_trip_on_real_geometry(self):
        save_indices = self._alloc(SEQ_LEN)
        self._fill(seed=0)
        expected = self._snapshot(save_indices, req_pool_idx=1)

        host = self.allocator.get_cpu_copy(
            save_indices, mamba_indices=None, req_pool_idx=1
        )

        # Whatever runs after the retract reuses those device rows.
        self._fill(seed=1)
        load_indices = self._alloc(SEQ_LEN)
        self.assertNotEqual(save_indices[0].item(), load_indices[0].item())

        self.allocator.load_cpu_copy(
            host, load_indices, mamba_indices=None, req_pool_idx=4
        )

        landed = self._snapshot(load_indices, req_pool_idx=4)
        for name, layers in expected.items():
            for layer_id, want in enumerate(layers):
                with self.subTest(region=name, layer=layer_id):
                    got = landed[name][layer_id]
                    # -inf sentinels in compress state make == unusable directly.
                    self.assertTrue(
                        torch.equal(
                            got.nan_to_num(neginf=-1e30), want.nan_to_num(neginf=-1e30)
                        )
                    )

    def test_region_geometry_matches_the_model(self):
        indices = self._alloc(SEQ_LEN)
        host = self.allocator.get_cpu_copy(indices, mamba_indices=None, req_pool_idx=1)
        ratios = self.model_config.compress_ratios
        # One row per 256 logical tokens for every compressed region: c4 folds 4
        # tokens into a 64-entry page, c128 folds 128 into a 2-entry page.
        pages = -(-SEQ_LEN // PAGE_SIZE)
        for name in ("c4_kv", "c128_kv", "c4_indexer_kv"):
            self.assertEqual(host[name][0][0].shape[0], pages, name)
        # Dense (ratio 0) layers own no compressed KV but do own SWA latent KV.
        self.assertEqual(len(host["swa_kv"][0]), len(ratios))
        self.assertEqual(len(host["c4_kv"][0]), sum(1 for r in ratios if r == 4))
        self.assertEqual(len(host["c128_kv"][0]), sum(1 for r in ratios if r == 128))
        # C4 state is a full ring block per SWA page; offline C128 state is one
        # 128-row block of the request's ring.
        swa_pages = host["swa_kv"][0][0].shape[0]
        self.assertEqual(
            host["c4_state"][0][0].shape[0], swa_pages * self.pool.get_ring_size(4)
        )
        self.assertEqual(host["c128_state"][0][0].shape[0], 128)


if __name__ == "__main__":
    unittest.main()
