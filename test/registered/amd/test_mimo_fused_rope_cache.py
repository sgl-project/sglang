import os
import types
import unittest

import torch

from sglang.kernels.ops.attention.mimo_fused_rope_cache import (
    try_mimo_fused_rope_cache,
)
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=45, suite="stage-b-test-1-gpu-small-amd-mi35x")


def _cos_sin_cache(length=128):
    angles = torch.randn(length, 32, device="cuda", dtype=torch.float32)
    return torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)


def _reference(q, k, v, positions, cos_sin, v_scale):
    q_expected = q.clone()
    k_expected = k.clone()
    v_expected = (v.float() * v_scale).bfloat16()
    selected = cos_sin.index_select(0, positions)
    cos, sin = selected.chunk(2, dim=-1)
    for tensor in (q_expected, k_expected):
        view = tensor.view(tensor.shape[0], -1, 192)
        first = view[..., :32].float()
        second = view[..., 32:64].float()
        view[..., :32] = (first * cos[:, None] - second * sin[:, None]).bfloat16()
        view[..., 32:64] = (
            second * cos[:, None] + first * sin[:, None]
        ).bfloat16()
    return q_expected, k_expected, v_expected


@unittest.skipUnless(is_gfx95_supported(), "gfx950 only")
class TestMiMoFusedRopeCache(CustomTestCase):
    def setUp(self):
        os.environ["SGLANG_ROCM_MIMO_FUSED_FRONT"] = "1"
        torch.manual_seed(0)

    def _case(self, m, shuffle_cache=False):
        q = torch.randn(m, 16, 192, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(m, 1, 192, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(m, 1, 128, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(7, 7 + m, device="cuda", dtype=torch.int64)
        slots = torch.arange(3, 3 + m, device="cuda", dtype=torch.int64)
        cache_size = max(32, 3 + m)
        cos_sin = _cos_sin_cache(7 + m)
        if shuffle_cache:
            key_cache = torch.zeros(
                cache_size, 1, 6, 1, 32, device="cuda", dtype=torch.bfloat16
            )
            value_cache = torch.zeros(
                cache_size, 1, 4, 1, 32, device="cuda", dtype=torch.bfloat16
            )
        else:
            key_cache = torch.zeros(
                cache_size, 1, 1, 192, device="cuda", dtype=torch.bfloat16
            )
            value_cache = torch.zeros(
                cache_size, 1, 1, 128, device="cuda", dtype=torch.bfloat16
            )
        rotary = types.SimpleNamespace(
            head_size=192,
            rotary_dim=64,
            is_neox_style=True,
            cos_sin_cache=cos_sin,
        )
        args = {
            "k_scale": torch.ones((), device="cuda"),
            "v_scale": torch.ones((), device="cuda"),
            "key_cache": key_cache,
            "value_cache": value_cache,
            "slot_mapping": slots,
            "swa_slot_mapping": None,
        }
        return q, k, v, positions, rotary, args

    @staticmethod
    def _cached(cache, slots):
        if cache.ndim == 5:
            return cache[slots, 0, :, 0, :].reshape(slots.numel(), 1, -1)
        return cache[slots, 0]

    def test_matches_reference(self):
        for m in (1, 4, 16, 32, 120, 128):
            for shuffle_cache in (False, True):
                with self.subTest(m=m, shuffle_cache=shuffle_cache):
                    q, k, v, positions, rotary, args = self._case(
                        m, shuffle_cache
                    )
                    expected = _reference(
                        q, k, v, positions, rotary.cos_sin_cache, 0.612
                    )
                    self.assertTrue(
                        try_mimo_fused_rope_cache(
                            q, k, v, positions, rotary, 0.612, args
                        )
                    )
                    torch.testing.assert_close(
                        q, expected[0], atol=2e-2, rtol=2e-2
                    )
                    torch.testing.assert_close(
                        k, expected[1], atol=2e-2, rtol=2e-2
                    )
                    self.assertTrue(torch.equal(v, expected[2]))
                    slots = args["slot_mapping"]
                    self.assertTrue(
                        torch.equal(self._cached(args["key_cache"], slots), k)
                    )
                    self.assertTrue(
                        torch.equal(self._cached(args["value_cache"], slots), v)
                    )

    def test_graph_replay(self):
        q, k, v, positions, rotary, args = self._case(120, shuffle_cache=True)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            accepted = try_mimo_fused_rope_cache(
                q, k, v, positions, rotary, 0.612, args
            )
        self.assertTrue(accepted)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(q).all())
        self.assertTrue(torch.isfinite(k).all())
        self.assertTrue(torch.isfinite(v).all())
        self.assertTrue(
            torch.equal(self._cached(args["key_cache"], args["slot_mapping"]), k)
        )
        self.assertTrue(
            torch.equal(self._cached(args["value_cache"], args["slot_mapping"]), v)
        )


if __name__ == "__main__":
    unittest.main()
