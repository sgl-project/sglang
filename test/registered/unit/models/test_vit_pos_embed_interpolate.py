"""Bit-exact unit test for the vectorized ViT position-embedding interpolation.

The vectorized path (``fast_pos_embed_interpolate_vectorized``) removes the
per-image Python loop / CPU<->GPU sync of the legacy implementations. It is meant
to be a pure speedup, so it must be numerically *identical* (bit-exact, rtol=0
atol=0) to the loop version it replaces -- for single images, many images, video
(t>1), and mixed-size batches, in both bf16 and fp32.

The interpolation is a sequence of embedding lookups + arithmetic, so it runs and
is bit-exact on CPU; the test exercises CUDA too when available. It calls the real
model methods on a lightweight stub holding a real ``nn.Embedding`` (no model
weights / distributed init needed).

    python -m pytest test/registered/models/test_vit_pos_embed_interpolate.py -v
"""

import unittest
from types import MethodType, SimpleNamespace

import torch
import torch.nn as nn

from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")
register_amd_ci(est_time=20, stage="stage-a", runner_config="1-gpu-small-amd")

NUM_POS = 2304  # Qwen3-VL num_position_embeddings -> 48x48 grid
HIDDEN = 64  # small hidden dim keeps the unit test fast
MERGE = 2

# t, h, w grids (h, w are multiples of MERGE). Covers single / large-upsample /
# multi-mixed / video / video+image / many-duplicate.
GRID_CASES = {
    "single": [[1, 16, 16]],
    "single_large": [[1, 64, 98]],  # h, w may exceed grid side (upsample)
    "multi_mixed": [[1, 16, 24], [1, 32, 12], [1, 8, 40]],
    "video": [[4, 16, 20]],
    "video_plus_image": [[3, 12, 16], [1, 20, 28], [2, 8, 8]],
    "many": [[1, 24, 24]] * 8,
}


def _devices():
    devs = [torch.device("cpu")]
    if torch.cuda.is_available():
        devs.append(torch.device("cuda"))
    return devs


class TestViTPosEmbedInterpolate(CustomTestCase):
    def _check(self, stub, legacy_fn, vectorized_fn, grid, label):
        ref = legacy_fn(stub, grid)
        out = vectorized_fn(stub, grid)
        self.assertEqual(ref.shape, out.shape, f"{label}: shape mismatch")
        self.assertTrue(
            torch.equal(ref, out),
            f"{label}: not bit-exact, max|diff|="
            f"{(ref.float() - out.float()).abs().max().item():.3e}",
        )

    def test_qwen3_vl_vectorized_matches_loop(self):
        try:
            from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel as M
        except Exception as e:  # heavy optional deps (flashinfer, ...) unavailable
            self.skipTest(f"cannot import Qwen3VLMoeVisionModel: {e}")

        for device in _devices():
            for dtype in (torch.bfloat16, torch.float32):
                stub = SimpleNamespace(
                    num_grid_per_side=int(NUM_POS**0.5),
                    spatial_merge_size=MERGE,
                    num_position_embeddings=NUM_POS,
                    pos_embed=nn.Embedding(NUM_POS, HIDDEN).to(
                        device=device, dtype=dtype
                    ),
                    dtype=dtype,
                    device=device,
                )
                for name, grid in GRID_CASES.items():
                    self._check(
                        stub,
                        M.fast_pos_embed_interpolate_from_list,
                        M.fast_pos_embed_interpolate_vectorized,
                        grid,
                        f"qwen3_vl/{name}/{dtype}/{device.type}",
                    )

    def test_qwen3_vl_interpolation_accumulates_in_fp32(self):
        from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel as M

        for device in _devices():
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(device=device, dtype=dtype):
                    embedding = nn.Embedding(4, 1, device=device, dtype=dtype)
                    embedding.weight.data.copy_(
                        torch.tensor(
                            [[1.0], [1.5], [2.0], [3.0]], device=device, dtype=dtype
                        )
                    )
                    stub = SimpleNamespace(
                        num_grid_per_side=2,
                        spatial_merge_size=2,
                        pos_embed=embedding,
                        dtype=dtype,
                        device=device,
                    )
                    expected = (
                        torch.nn.functional.interpolate(
                            embedding.weight.float().reshape(1, 1, 2, 2),
                            size=(4, 4),
                            mode="bilinear",
                            align_corners=True,
                        )
                        .reshape(2, 2, 2, 2)
                        .permute(0, 2, 1, 3)
                        .reshape(-1, 1)
                        .to(dtype)
                    )
                    for method in (
                        M.fast_pos_embed_interpolate_from_list,
                        M.fast_pos_embed_interpolate_vectorized,
                    ):
                        actual = method(stub, [[1, 4, 4]])
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    stub.align_corners = True
                    for name in (
                        "_get_interpolation_indices",
                        "_calculate_indices_and_weights",
                        "_get_position_embedding",
                    ):
                        setattr(stub, name, MethodType(getattr(M, name), stub))
                    actual = M.fast_pos_embed_interpolate(
                        stub, torch.tensor([[1, 4, 4]])
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_moss_vl_vectorized_matches_loop(self):
        try:
            from sglang.srt.models.moss_vl import MossVLVisionModel as M
        except Exception as e:
            self.skipTest(f"cannot import MossVLVisionModel: {e}")

        for device in _devices():
            for dtype in (torch.bfloat16, torch.float32):
                stub = SimpleNamespace(
                    spatial_merge_size=MERGE,
                    num_position_embeddings=NUM_POS,
                    pos_embed=nn.Embedding(NUM_POS, HIDDEN).to(
                        device=device, dtype=dtype
                    ),
                )
                for name, grid in GRID_CASES.items():
                    # the legacy moss method consumes a [num_images, 3] tensor
                    grid_t = torch.tensor(grid, device=device)
                    self._check(
                        stub,
                        M.fast_pos_embed_interpolate,
                        M.fast_pos_embed_interpolate_vectorized,
                        grid_t,
                        f"moss_vl/{name}/{dtype}/{device.type}",
                    )


if __name__ == "__main__":
    unittest.main()
