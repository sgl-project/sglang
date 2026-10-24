"""Real-device MRoPE semantics for non-interleaved [16, 24, 24], BF16/head128.

Exercise forward_npu without replacing its dispatch or vendor operations. The
multiaxis path may use the native implementation; this does not require the
vendor ordinary-RoPE operator to support multidimensional positions.
"""

import importlib.util
import unittest

import torch

from sglang.test.ci.ci_register import register_npu_ci

if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu  # noqa: F401 - register the NPU device backend

register_npu_ci(est_time=5, suite="base-b-test-1-npu-a3")


def reference_rotation(value, cache, positions):
    """Independent CPU lane lookup and NeoX half pairing, with exact integers."""
    assert value.device.type == cache.device.type == positions.device.type == "cpu"
    # Explicit ownership, independent of the implementation's axis map/splits.
    owners = torch.tensor([0] * 16 + [1] * 24 + [2] * 24, device="cpu")
    lanes = torch.arange(64, device="cpu")
    rows = positions[:, None] if positions.ndim == 1 else positions[owners].T
    cosine = cache.float()[rows, lanes]
    sine = cache.float()[rows, lanes + 64]
    heads = value.float().reshape(4, -1, 128)
    left, right = heads[..., :64], heads[..., 64:]
    cosine, sine = cosine[:, None, :], sine[:, None, :]
    return (
        torch.cat((left * cosine - right * sine, right * cosine + left * sine), dim=-1)
        .reshape(value.shape)
        .to(torch.bfloat16)
    )


@unittest.skipUnless(
    hasattr(torch, "npu") and torch.npu.is_available(), "requires an Ascend NPU"
)
class TestMRopeMultiaxis(unittest.TestCase):
    def setUp(self):
        # Keep heavy model/kernel imports out of CPU-only test collection.
        from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

        self.device = torch.device("npu", torch.npu.current_device())
        self.rope = MRotaryEmbedding(
            head_size=128,
            rotary_dim=128,
            max_position_embeddings=4,
            base=10000,
            is_neox_style=True,
            dtype=torch.bfloat16,
            mrope_section=[16, 24, 24],
            mrope_interleaved=False,
            mrope_interleaved_glm=False,
        ).to(self.device)
        # Four exact quarter-turn rotations. No trig/rounding tolerance masks
        # selecting the wrong temporal/height/width row.
        self.cache = torch.cat(
            (
                torch.tensor([1, 0, -1, 0], device="cpu")[:, None].repeat(1, 64),
                torch.tensor([0, 1, 0, -1], device="cpu")[:, None].repeat(1, 64),
            ),
            dim=1,
        ).to(torch.bfloat16)
        self.rope.cos_sin_cache = self.cache.to(self.device)
        self.packed = (
            (torch.arange(4 * 2560, device="cpu").reshape(4, 2560) % 7) + 1
        ).to(torch.bfloat16)

    def check_positions(self, positions):
        self.assertEqual(positions.device.type, "cpu")
        self.assertEqual(self.packed.device.type, "cpu")
        self.assertEqual(self.cache.device.type, "cpu")
        device_packed = self.packed.to(self.device)
        query, key, _ = device_packed.split([2048, 256, 256], dim=-1)
        for value in (query, key):
            self.assertEqual(value.stride(), (2560, 1))
            self.assertFalse(value.is_contiguous())

        inputs = (self.packed[:, :2048], self.packed[:, 2048:2304])
        expected = [
            reference_rotation(value, self.cache, positions) for value in inputs
        ]
        if positions.ndim == 2:
            # All temporal positions are zero: the old ordinary-RoPE dispatch
            # returned the input unchanged, which must fail this fixture.
            for value, prediction in zip(inputs, expected):
                self.assertFalse(torch.equal(value, prediction))

        actual = self.rope.forward_npu(positions.to(self.device), query, key)
        torch.npu.synchronize(self.device)
        self.assertEqual(len(actual), 2)
        for role, value, prediction in zip(("query", "key"), actual, expected):
            with self.subTest(role=role, position_dimensions=positions.ndim):
                self.assertEqual(value.device, self.device)
                self.assertEqual(value.dtype, torch.bfloat16)
                self.assertEqual(value.shape, prediction.shape)
                cpu = value.cpu()
                self.assertTrue(torch.isfinite(cpu).all().item())
                self.assertTrue(
                    torch.equal(
                        cpu.contiguous().view(torch.uint16),
                        prediction.contiguous().view(torch.uint16),
                    ),
                    f"{role} differs from independent exact rotation",
                )

        self.assertTrue(
            torch.equal(
                device_packed.cpu().view(torch.uint16), self.packed.view(torch.uint16)
            ),
            "MRoPE modified the packed QKV input",
        )
        self.assertTrue(
            torch.equal(
                self.rope.cos_sin_cache.cpu().view(torch.uint16),
                self.cache.view(torch.uint16),
            ),
            "MRoPE modified the cosine/sine cache",
        )

    def test_multiaxis_positions(self):
        self.check_positions(
            torch.tensor([[0, 0, 0, 0], [0, 1, 2, 3], [3, 2, 1, 0]], device="cpu")
        )

    def test_ordinary_positions(self):
        self.check_positions(torch.arange(4, device="cpu"))


if __name__ == "__main__":
    unittest.main()
