"""Numerics + bounds regression tests for the fused MRoPE triton kernel.

Regression context: with ``mrope_interleaved=True`` and ``head_size >
rotary_dim`` (e.g. Qwen3.5: head_dim=256, partial_rotary_factor=0.25 ->
rotary_dim=64), the interleaved masks (``t_mask``/``h_mask``/``w_mask``) were
not bounded by ``half_rd``, so lanes in ``[half_rd, pad_hd // 2)`` issued
cos/sin loads past the end of ``cos_sin_cache`` — an out-of-bounds read (IMA
risk) for positions near ``max_position``. Verified with compute-sanitizer:

    PYTORCH_NO_CUDA_MEMORY_CACHING=1 compute-sanitizer --tool memcheck \
        python -m pytest test/registered/rotary/test_mrope_interleaved_partial.py

The tests below pin the triton kernel against ``forward_native`` for the
interleaved / non-interleaved / GLM layouts at max-position boundaries, so the
added ``rd_mask`` bound cannot silently change numerics.

Usage:
    python -m pytest test/registered/rotary/test_mrope_interleaved_partial.py -v
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

torch.manual_seed(0)

# (head_size, partial_rotary_factor, mrope_section, interleaved, glm)
_CASES = [
    # Qwen3.5: head 256, rotary 64, interleaved
    (256, 0.25, [11, 11, 10], True, False),
    # Qwen3-VL-style: full rotary, interleaved
    (128, 1.0, [24, 20, 20], True, False),
    # Qwen2-VL-style: full rotary, sectioned (non-interleaved)
    (128, 1.0, [16, 24, 24], False, False),
    # GLM-V-style: interleaved via axis_map
    (128, 1.0, [8, 12, 12], True, True),
    # partial rotary + non-interleaved
    (256, 0.5, [22, 21, 21], False, False),
]

_MAX_POS = 2048
_NUM_TOKENS = 16
_N_QH, _N_KH = 4, 2


def _build_rope(head_size, partial, section, interleaved, glm):
    rope_scaling = {
        "rope_type": "default",
        "mrope_section": section,
        "mrope_interleaved": interleaved,
    }
    if glm:
        rope_scaling["mrope_interleaved_glm"] = True
    rope = get_rope(
        head_size=head_size,
        rotary_dim=head_size,
        max_position=_MAX_POS,
        base=10000000,
        is_neox_style=True,
        rope_scaling=rope_scaling,
        dtype=torch.bfloat16,
        partial_rotary_factor=partial,
    )
    assert isinstance(rope, MRotaryEmbedding), type(rope)
    return rope


class TestMRopeTritonVsNative(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda:0"

    def _run_case(self, head_size, partial, section, interleaved, glm, positions):
        rope = _build_rope(head_size, partial, section, interleaved, glm).to(
            self.device
        )
        rope.cos_sin_cache = rope.cos_sin_cache.to(self.device, dtype=torch.bfloat16)
        if rope.axis_map is not None:
            rope.axis_map = rope.axis_map.to(self.device)

        q = torch.randn(
            _NUM_TOKENS, _N_QH * head_size, dtype=torch.bfloat16, device=self.device
        )
        k = torch.randn(
            _NUM_TOKENS, _N_KH * head_size, dtype=torch.bfloat16, device=self.device
        )

        q_ref, k_ref = rope.forward_native(positions, q.clone(), k.clone())
        q_tri, k_tri = rope.forward_triton(positions, q.clone(), k.clone())
        torch.cuda.synchronize()

        # bf16 rotate-add tolerance
        torch.testing.assert_close(q_tri, q_ref, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(k_tri, k_ref, atol=2e-2, rtol=2e-2)

    def test_random_positions(self):
        for case in _CASES:
            with self.subTest(case=case):
                positions = torch.randint(
                    0, _MAX_POS, (3, _NUM_TOKENS), device=self.device
                )
                self._run_case(*case, positions)

    def test_boundary_positions(self):
        """Positions at max_position - 1: the pre-fix interleaved kernel read
        past cos_sin_cache here (caught by compute-sanitizer)."""
        for case in _CASES:
            with self.subTest(case=case):
                positions = torch.full(
                    (3, _NUM_TOKENS),
                    _MAX_POS - 1,
                    dtype=torch.int64,
                    device=self.device,
                )
                self._run_case(*case, positions)


if __name__ == "__main__":
    unittest.main()
