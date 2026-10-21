import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.muse_spark import muse_spark_megamoe as megamoe
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_ENV = megamoe._TILES_ENV
_TILES = megamoe._BF16_PREFILL_TILES


def _sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return False
    try:
        import deep_gemm  # noqa: F401
    except ImportError:
        return False
    return True


def _buf(mma_type="bf16xbf16", ranks=8):
    return SimpleNamespace(
        mma_type=mma_type,
        num_experts=256,
        num_max_tokens_per_rank=9600,
        num_topk=8,
        group=SimpleNamespace(size=lambda: ranks),
    )


class TestMuseSparkMegaMoETiles(CustomTestCase):
    def setUp(self):
        self._saved = os.environ.pop(_ENV, None)
        megamoe._heuristic_block_m.cache_clear()

    def tearDown(self):
        os.environ.pop(_ENV, None)
        if self._saved is not None:
            os.environ[_ENV] = self._saved
        megamoe._heuristic_block_m.cache_clear()

    def test_only_heuristic_192_is_replaced(self):
        for block_m in (8, 16, 32, 64, 96, 128, 192, 240):
            with mock.patch.object(megamoe, "_heuristic_block_m", return_value=block_m):
                expected = _TILES if block_m == 192 else None
                self.assertEqual(megamoe._tile_override(_buf(), 1024), expected)

    def test_nvfp4_and_user_override_are_untouched(self):
        with mock.patch.object(megamoe, "_heuristic_block_m", return_value=192):
            self.assertIsNone(megamoe._tile_override(_buf("nvfp4xnvfp4"), 2048))
            os.environ[_ENV] = "2,128,32,128,2"
            self.assertIsNone(megamoe._tile_override(_buf(), 1024))
            # Empty means "DeepGEMM heuristic", which this adapter may refine.
            os.environ[_ENV] = ""
            self.assertEqual(megamoe._tile_override(_buf(), 1024), _TILES)

    def test_tiles_are_scoped_to_the_launch(self):
        with megamoe._megamoe_tiles(_TILES):
            self.assertEqual(os.environ[_ENV], _TILES)
        self.assertEqual(os.environ[_ENV], "")
        with self.assertRaises(RuntimeError):
            with megamoe._megamoe_tiles(_TILES):
                raise RuntimeError("launch failed")
        self.assertEqual(os.environ[_ENV], "")
        os.environ[_ENV] = "2,128,32,128,2"
        with megamoe._megamoe_tiles(None):
            self.assertEqual(os.environ[_ENV], "2,128,32,128,2")
        self.assertEqual(os.environ[_ENV], "2,128,32,128,2")

    @unittest.skipUnless(_sm100(), "DeepGEMM MegaMoE requires SM100")
    def test_installed_heuristic(self):
        import deep_gemm

        def block_m(n, ranks=8):
            megamoe._heuristic_block_m.cache_clear()
            return megamoe._heuristic_block_m(ranks, 256, 9600, n, 8, "bf16xbf16")

        # Muse BF16 8K prefill shards (EP8 and EP4) select 192; decode-sized keep small.
        self.assertEqual(block_m(1024), 192)
        self.assertEqual(block_m(2048, ranks=4), 192)
        for n in (1, 8, 64):
            self.assertLess(block_m(n), 192)
        os.environ[_ENV] = ""
        self.assertEqual(block_m(1024), 192)
        os.environ[_ENV] = _TILES
        self.assertEqual(
            deep_gemm.get_block_m_for_mega_moe(8, 256, 9600, 1024, 8, "bf16xbf16"), 240
        )

    @unittest.skipUnless(_sm100(), "DeepGEMM MegaMoE requires SM100")
    def test_tile_invariant_output(self):
        # BLOCK_M only partitions rows: K order, epilogue and combine are unchanged.
        import deep_gemm

        hidden, experts, topk, n = 512, 8, 8, 513
        group = SimpleNamespace(size=lambda: 1, rank=lambda: 0, barrier=lambda: None)
        buf = deep_gemm.get_symm_buffer_for_mega_moe(
            group, experts, 1024, topk, hidden, hidden, mma_type="bf16xbf16"
        )
        self.assertEqual(megamoe._tile_override(buf, n), _TILES)
        g = torch.Generator(device="cuda").manual_seed(0)
        w1 = (
            torch.randn(experts, 2 * hidden, hidden, device="cuda", generator=g) / 16
        ).bfloat16()
        w2 = (
            torch.randn(experts, hidden, hidden, device="cuda", generator=g) / 16
        ).bfloat16()
        gain = torch.rand(hidden, device="cuda", generator=g) + 0.5
        buf.x[:n].copy_(torch.randn(n, hidden, device="cuda", generator=g))
        buf.topk_idx[:n].copy_(
            torch.rand(n, experts, device="cuda", generator=g).topk(topk).indices
        )
        buf.topk_weights[:n].copy_(
            torch.rand(n, topk, device="cuda", generator=g).softmax(-1)
        )
        buf.topk_idx[n:].fill_(-1)
        buf.topk_weights[n:].zero_()
        outputs = []
        for tiles in (None, _TILES):
            y = torch.empty(n, hidden, dtype=torch.bfloat16, device="cuda")
            with megamoe._megamoe_tiles(tiles):
                deep_gemm.bf16_mega_moe(
                    y,
                    w1,
                    w2,
                    buf,
                    activation="swiglu",
                    weight_layout="trtllm",
                    fast_math=True,
                    post_expert_norm=True,
                    post_expert_norm_weight=gain,
                    post_expert_norm_eps=1e-6,
                    post_expert_norm_weight_first=True,
                )
            outputs.append(y)
        torch.cuda.synchronize()
        self.assertTrue(bool(torch.isfinite(outputs[0].float()).all()))
        self.assertTrue(
            torch.equal(outputs[0].view(torch.int16), outputs[1].view(torch.int16))
        )
        buf.destroy()


if __name__ == "__main__":
    unittest.main()
