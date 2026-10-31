import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.rotary_embedding import base
from sglang.srt.models import utils as model_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRdna4RopeFallback(CustomTestCase):
    def test_gfx1201_cpu_first_constructor_and_forward(self):
        """Native dispatch must allow CPU cache construction before a device move."""
        with (
            torch.device("cpu"),
            mock.patch.object(base, "_is_cpu", True),
            mock.patch.object(base, "is_gfx1201_supported", return_value=True),
            mock.patch.object(torch, "compile", side_effect=lambda **kw: lambda f: f),
        ):
            rope = base.RotaryEmbedding(4, 4, 8, 10000, True, torch.float32)
            self.assertEqual(rope.cos_sin_cache.device, torch.device("cpu"))
            rope = rope.to("cpu")
            positions = torch.tensor([0, 1])
            query = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).repeat(2, 1)
            key = query + 4
            actual_q, actual_k = rope(positions, query, key)

            angles = positions[:, None] * torch.tensor([1.0, 0.01])
            cos = angles.cos().repeat(1, 2)
            sin = angles.sin().repeat(1, 2)
            for actual, original in ((actual_q, query), (actual_k, key)):
                rotated = torch.cat((-original[:, 2:], original[:, :2]), dim=-1)
                torch.testing.assert_close(actual, original * cos + rotated * sin)

    def test_native_rope_disables_fused_kv_cache(self):
        """The RoPE producer and attention cache-write decision must agree on HIP."""
        with (
            mock.patch.object(base, "publish_role", return_value=None),
            mock.patch.object(model_utils, "_is_cuda", False),
            mock.patch.object(model_utils, "_is_hip", True),
            mock.patch.object(
                model_utils,
                "get_token_to_kv_pool",
                return_value=SimpleNamespace(dtype=torch.bfloat16),
            ),
            mock.patch.object(
                model_utils, "is_prefill_context_parallel_enabled", return_value=False
            ),
        ):
            for gfx1201 in (True, False):
                with (
                    self.subTest(gfx1201=gfx1201),
                    mock.patch.object(
                        base, "is_gfx1201_supported", return_value=gfx1201
                    ),
                ):
                    self.assertEqual(
                        model_utils.enable_fused_set_kv_buffer(SimpleNamespace()),
                        not gfx1201,
                    )


if __name__ == "__main__":
    unittest.main()
