"""Unit tests for the ViT position-embedding interpolation.

Qwen3-VL resizes its learned position-embedding grid with ``F.interpolate``
(``interpolate_pos_embed``). It is checked against an independent fp32
4-neighbour bilinear reference, in both coordinate modes (corner-aligned and
half-pixel), for single images, many images, video (t>1) and mixed-size
batches: tight in fp32, within bf16 rounding in bf16.

The MOSS-VL vectorized path must stay bit-exact (rtol=0, atol=0) with its
per-image loop version.

Both run on CPU; the test exercises CUDA too when available. It calls the real
model methods on a lightweight stub holding a real ``nn.Embedding`` (no model
weights / distributed init needed).

    python -m pytest test/registered/unit/models/test_vit_pos_embed_interpolate.py -v
"""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.test.ci.ci_register import register_amd_ci, register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")
register_amd_ci(est_time=20, stage="base-a", runner_config="1-gpu-small-amd")

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


def _source_coords(size, side, align_corners, device):
    if align_corners:
        return torch.linspace(0, side - 1, size, device=device)
    coords = (torch.arange(size, device=device) + 0.5) * (side / size) - 0.5
    return coords.clamp(0, side - 1)


def _reference_pos_embed(weight, side, grid, align_corners):
    """fp32 bilinear from the 4 neighbours, merge-block order, repeated over t."""
    table = weight.view(side, side, -1)
    outputs = []
    for t, h, w in grid:
        hs = _source_coords(h, side, align_corners, weight.device)
        ws = _source_coords(w, side, align_corners, weight.device)
        h0, w0 = hs.floor().long(), ws.floor().long()
        h1, w1 = (h0 + 1).clamp(max=side - 1), (w0 + 1).clamp(max=side - 1)
        dh, dw = (hs - h0)[:, None, None], (ws - w0)[None, :, None]
        emb = (
            table[h0][:, w0] * (1 - dh) * (1 - dw)
            + table[h0][:, w1] * (1 - dh) * dw
            + table[h1][:, w0] * dh * (1 - dw)
            + table[h1][:, w1] * dh * dw
        )
        emb = emb.reshape(h // MERGE, MERGE, w // MERGE, MERGE, -1).permute(
            0, 2, 1, 3, 4
        )
        outputs.append(emb.reshape(h * w, -1).repeat(t, 1))
    return torch.cat(outputs)


class TestViTPosEmbedInterpolate(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _check(self, stub, legacy_fn, vectorized_fn, grid, label):
        ref = legacy_fn(stub, grid)
        out = vectorized_fn(stub, grid)
        self.assertEqual(ref.shape, out.shape, f"{label}: shape mismatch")
        self.assertTrue(
            torch.equal(ref, out),
            f"{label}: not bit-exact, max|diff|="
            f"{(ref.float() - out.float()).abs().max().item():.3e}",
        )

    def test_qwen3_vl_interpolate_matches_reference(self):
        try:
            from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel as M
        except Exception as e:  # heavy optional deps (flashinfer, ...) unavailable
            self.skipTest(f"cannot import Qwen3VLMoeVisionModel: {e}")

        side = int(NUM_POS**0.5)
        for device in _devices():
            for dtype in (torch.bfloat16, torch.float32):
                stub = SimpleNamespace(
                    num_grid_per_side=side,
                    spatial_merge_size=MERGE,
                    pos_embed=nn.Embedding(NUM_POS, HIDDEN).to(
                        device=device, dtype=dtype
                    ),
                )
                for align_corners in (True, False):
                    for name, grid in GRID_CASES.items():
                        label = (
                            f"qwen3_vl/{name}/align_corners={align_corners}"
                            f"/{dtype}/{device.type}"
                        )
                        out = M.interpolate_pos_embed(
                            stub, grid, align_corners=align_corners
                        )
                        ref = _reference_pos_embed(
                            stub.pos_embed.weight.float(), side, grid, align_corners
                        )
                        self.assertEqual(out.shape, ref.shape, f"{label}: shape")
                        self.assertEqual(out.dtype, dtype, f"{label}: dtype")
                        # fp32: source coordinates up to side-1 carry ~1e-7 relative
                        # rounding, scaled by neighbour differences; bf16: the output
                        # rounds once from an fp32-accurate blend.
                        scale = ref.abs().max().item()
                        atol = (1e-4 if dtype == torch.float32 else 2**-7) * scale
                        err = (out.float() - ref).abs().max().item()
                        self.assertLessEqual(err, atol, f"{label}: max|diff|={err:.3e}")

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
