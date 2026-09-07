"""CPU format conformance, independent of CUDA and the serving memory pools."""

import unittest

import torch

from sglang.srt.mem_cache.kvbit_dsv4_codec import (
    DSV4_INT4_ALIGNED_LAYOUT,
    DSV4_INT4_LAYOUT,
    DSV4KVBitLayout,
    decode_dsv4_int4_reference,
    encode_dsv4_int4_reference,
    layout_for_row_bytes,
    repack_dsv4_int4_reference,
    validate_dsv4_int4_attention,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSV4INT4Codec(unittest.TestCase):
    def test_layout_payload_and_stride(self):
        for layout, stride in (
            (DSV4_INT4_LAYOUT, 368),
            (DSV4_INT4_ALIGNED_LAYOUT, 384),
        ):
            with self.subTest(stride=stride):
                self.assertEqual(layout.row_bytes, stride)
                self.assertEqual(layout.payload_bytes, 368)
                self.assertEqual(layout.offsets()["rope"], (240, 368))
                self.assertEqual(layout_for_row_bytes(stride), layout)
        self.assertEqual(DSV4_INT4_ALIGNED_LAYOUT.offsets()["padding"], (368, 384))
        with self.assertRaisesRegex(ValueError, "Unsupported DSV4"):
            DSV4KVBitLayout("typo")
        with self.assertRaisesRegex(ValueError, "368 or 384"):
            layout_for_row_bytes(400)

    def test_layout_repack_is_lossless_without_requantization(self):
        generator = torch.Generator().manual_seed(42)
        storage = torch.randn(2, 3, 640, generator=generator, dtype=torch.bfloat16)
        kv = storage[..., 128:]
        before = storage.clone()
        compact = encode_dsv4_int4_reference(kv)
        aligned = encode_dsv4_int4_reference(kv, layout=DSV4_INT4_ALIGNED_LAYOUT)
        self.assertTrue(torch.equal(compact, aligned[..., :368]))
        self.assertEqual(torch.count_nonzero(aligned[..., 368:]).item(), 0)
        self.assertEqual(torch.count_nonzero(aligned[..., 238:240]).item(), 0)
        self.assertTrue(torch.equal(storage, before))
        converted = repack_dsv4_int4_reference(compact, layout=DSV4_INT4_ALIGNED_LAYOUT)
        self.assertTrue(torch.equal(aligned, converted))
        self.assertTrue(
            torch.equal(
                repack_dsv4_int4_reference(converted, layout=DSV4_INT4_LAYOUT), compact
            )
        )
        aligned[..., 368:] = 255
        self.assertTrue(
            torch.equal(
                decode_dsv4_int4_reference(aligned),
                decode_dsv4_int4_reference(compact),
            )
        )
        self.assertTrue(
            torch.equal(decode_dsv4_int4_reference(aligned)[..., 448:], kv[..., 448:])
        )

    def test_signed_nibbles_nearest_scale_and_round_to_even(self):
        kv = torch.zeros(2, 512)
        kv[0, :10] = torch.tensor([-7, -1, 0, 1, 7, -6.5, -5.5, 0.5, 1.5, 2.5])
        kv[0, 32:64] = 9
        packed = encode_dsv4_int4_reference(kv)
        self.assertEqual(packed[0, :5].tolist(), [0xF9, 0x10, 0xA7, 0x0A, 0x22])
        self.assertEqual(packed[0, 225].item(), 0x3A)
        self.assertEqual(torch.count_nonzero(packed[1]).item(), 0)
        nibbles = torch.stack((packed[..., :224] & 15, packed[..., :224] >> 4))
        self.assertFalse(torch.any(nibbles == 8).item())

    def test_scale_underflow_overflow_and_bf16_rope(self):
        kv = torch.zeros(1, 512)
        kv[0, :32] = 1e-5
        kv[0, 32:64] = 1e30
        kv[0, 64:96] = -1e30
        kv[0, 448:] = torch.linspace(-1, 1, 64)
        decoded = decode_dsv4_int4_reference(encode_dsv4_int4_reference(kv))
        self.assertTrue(torch.equal(decoded[0, :32], torch.zeros(32)))
        self.assertTrue(torch.equal(decoded[0, 32:64], torch.full((32,), 3136)))
        self.assertTrue(torch.equal(decoded[0, 64:96], torch.full((32,), -3136)))
        self.assertTrue(torch.equal(decoded[0, 448:], kv[0, 448:].bfloat16()))

    def test_empty_and_single_row_shapes(self):
        for shape in ((512,), (0, 512), (2, 0, 512)):
            for layout in (DSV4_INT4_LAYOUT, DSV4_INT4_ALIGNED_LAYOUT):
                with self.subTest(shape=shape, layout=layout.layout_id):
                    encoded = encode_dsv4_int4_reference(
                        torch.zeros(shape), layout=layout
                    )
                    self.assertEqual(encoded.shape, (*shape[:-1], layout.row_bytes))
                    decoded = decode_dsv4_int4_reference(encoded)
                    self.assertEqual(decoded.shape, shape)
                    self.assertEqual(torch.count_nonzero(decoded).item(), 0)

    def test_invalid_codec_inputs(self):
        with self.assertRaisesRegex(ValueError, "last dimension"):
            encode_dsv4_int4_reference(torch.zeros(511))
        with self.assertRaisesRegex(TypeError, "floating point"):
            encode_dsv4_int4_reference(torch.zeros(512, dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "CPU"):
            encode_dsv4_int4_reference(torch.empty(512, device="meta"))
        with self.assertRaisesRegex(ValueError, "row dimension"):
            decode_dsv4_int4_reference(torch.tensor(1, dtype=torch.uint8))
        with self.assertRaisesRegex(TypeError, "uint8"):
            decode_dsv4_int4_reference(torch.zeros(368))
        with self.assertRaisesRegex(ValueError, "uint8"):
            repack_dsv4_int4_reference(torch.zeros(368), layout=DSV4_INT4_LAYOUT)
        with self.assertRaisesRegex(ValueError, "368 or 384"):
            decode_dsv4_int4_reference(torch.zeros(367, dtype=torch.uint8))

    def test_bf16_mantissa_insertion_matches_scalar_for_all_codes_and_scales(self):
        # Algebraic check of the proposed CUDA BF16x2 path, not a GPU execution.
        nibbles = torch.arange(16, dtype=torch.int32)
        signed = ((nibbles ^ 8) - 8).float()
        inserted = ((nibbles ^ 8) | 0x4300).to(torch.int16).view(torch.bfloat16)
        recovered = inserted - torch.tensor(136, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(recovered.float(), signed))
        scales = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        expected = (signed[:, None] * scales[None, :]).bfloat16()
        paired = recovered[:, None] * scales[None, :].bfloat16()
        self.assertTrue(
            torch.equal(paired.view(torch.int16), expected.view(torch.int16))
        )


class TestDSV4INT4AttentionContract(unittest.TestCase):
    def test_supported_geometry_and_rejections(self):
        args = dict(
            num_attention_heads=512,
            attn_tp_size=8,
            nope_dim=448,
            rope_dim=64,
            head_dim_v=512,
            kv_heads=1,
            sparse_width=512,
        )
        validate_dsv4_int4_attention(**args)
        validate_dsv4_int4_attention(
            **{**args, "num_attention_heads": 64, "attn_tp_size": 1}
        )
        cases = (
            ({"attn_tp_size": 1}, "64 local query heads"),
            ({"attn_tp_size": 0}, "64 local query heads"),
            ({"attn_tp_size": 3}, "64 local query heads"),
            ({"num_attention_heads": 513}, "64 local query heads"),
            ({"nope_dim": 512}, "448-nope/64-rope"),
            ({"rope_dim": 32}, "448-nope/64-rope"),
            ({"head_dim_v": 256}, "MQA"),
            ({"kv_heads": 8}, "MQA"),
            ({"sparse_width": 0}, "sparse width"),
            ({"sparse_width": 65}, "sparse width"),
        )
        for changes, error in cases:
            with (
                self.subTest(changes=changes),
                self.assertRaisesRegex(ValueError, error),
            ):
                validate_dsv4_int4_attention(**{**args, **changes})


if __name__ == "__main__":
    unittest.main()
