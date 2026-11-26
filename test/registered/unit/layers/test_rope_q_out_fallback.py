import unittest
from unittest import mock

import torch

from sglang.srt.layers.rotary_embedding import base
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRopeQOutFallback(CustomTestCase):
    """``q_out`` holds the rotated query cast to its dtype on the paths the JIT
    kernel does not cover. torch.compile switches RoPE to forward_native while
    the attention backend still asks for an fp8 query."""

    def _rope(self):
        patches = (
            mock.patch.object(base, "_is_cpu", True),
            mock.patch.object(base, "publish_role", lambda: None),
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return base.RotaryEmbedding(
            head_size=64,
            rotary_dim=64,
            max_position_embeddings=256,
            base=10000,
            is_neox_style=True,
            dtype=torch.bfloat16,
        )

    def _inputs(self):
        torch.manual_seed(0)
        positions = torch.arange(5)
        query = torch.randn(5, 4 * 64, dtype=torch.bfloat16)
        key = torch.randn(5, 2 * 64, dtype=torch.bfloat16)
        return positions, query, key

    def test_native_writes_the_cast_query(self):
        rope = self._rope()
        positions, query, key = self._inputs()
        q_ref, k_ref = rope.forward_native(positions, query.clone(), key.clone())

        q_out = torch.empty_like(query, dtype=torch.float8_e4m3fn)
        _, k_new = rope.forward_native(
            positions, query.clone(), key.clone(), q_out=q_out
        )
        expected = q_ref.to(torch.float8_e4m3fn)
        self.assertTrue(
            torch.equal(q_out.view(torch.uint8), expected.view(torch.uint8))
        )
        self.assertTrue(torch.equal(k_new, k_ref))

    def test_fallback_kernel_writes_the_cast_query(self):
        rope = self._rope()
        rope.use_fallback_kernel = True

        def rotate_in_place(positions, query, key, *args):
            query.mul_(2)

        rope.fallback_rotary_embedding = mock.Mock(side_effect=rotate_in_place)
        positions, query, key = self._inputs()
        q_out = torch.empty_like(query, dtype=torch.float8_e4m3fn)
        with mock.patch.object(base, "_is_hip", False):
            rope.forward_cuda(positions, query, key, q_out=q_out)
        # query was rotated in place; q_out is its cast
        expected = query.to(torch.float8_e4m3fn)
        self.assertTrue(
            torch.equal(q_out.view(torch.uint8), expected.view(torch.uint8))
        )


if __name__ == "__main__":
    unittest.main()
