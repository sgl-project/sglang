import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers.models.gemma3.modeling_gemma3 import get_block_sequence_ids_for_mask

from sglang.srt.layers.attention import flashattention_backend
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMetadata,
)
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import gemma3_mm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

Gemma3 = gemma3_mm.Gemma3ForConditionalGeneration


class _OwnKernelBuild(FlashAttentionBackend):
    pass


REQUESTS = [
    dict(prefix=0, length=30, images=[(4, 11), (15, 24)]),
    dict(prefix=8, length=16, images=[(2, 5), (10, 19)]),
    dict(prefix=0, length=12, images=None),
    dict(prefix=6, length=10, images=[(3, 9)]),
]


def _image_item(spans):
    return SimpleNamespace(is_image=lambda: True, offsets=spans)


def _forward_batch():
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(REQUESTS),
        extend_seq_lens_cpu=[r["length"] for r in REQUESTS],
        extend_prefix_lens_cpu=[r["prefix"] for r in REQUESTS],
        mm_inputs=[
            None
            if r["images"] is None
            else SimpleNamespace(mm_items=[_image_item(r["images"])])
            for r in REQUESTS
        ],
    )


def _hf_allowed(total, images):
    token_type_ids = torch.zeros(1, total, dtype=torch.long)
    for begin, end in images or []:
        token_type_ids[0, begin : end + 1] = 1
    block = get_block_sequence_ids_for_mask(token_type_ids)[0]
    q = torch.arange(total)[:, None]
    k = torch.arange(total)[None, :]
    same_image = (block[:, None] >= 0) & (block[:, None] == block[None, :])
    return (k <= q) | same_image


def _ranges_allowed(ranges, prefix, length):
    q = torch.arange(prefix, prefix + length)[:, None]
    k = torch.arange(prefix + length)[None, :]
    begin, end = ranges[:, 0:1], ranges[:, 1:2]
    in_image = (begin >= 0) & (k >= begin) & (k <= end)
    return (k <= q) | in_image


class TestGemma3ImageTokenRanges(CustomTestCase):
    def test_ranges_reproduce_the_reference_image_mask(self):
        ranges = Gemma3._build_image_token_ranges(
            forward_batch=_forward_batch(), device="cpu"
        )
        self.assertEqual(ranges.shape, (sum(r["length"] for r in REQUESTS), 2))
        offset = 0
        for r in REQUESTS:
            rows = ranges[offset : offset + r["length"]]
            offset += r["length"]
            ours = _ranges_allowed(rows, r["prefix"], r["length"])
            ref = _hf_allowed(r["prefix"] + r["length"], r["images"])[r["prefix"] :]
            split = [s for s in r["images"] or [] if s[0] < r["prefix"] <= s[1]]
            if split:
                torch.testing.assert_close(rows, torch.full_like(rows, -1))
                self.assertFalse(torch.equal(ours, ref))
                continue
            self.assertTrue(torch.equal(ours, ref), r)

    def test_flash_attention_prefill_gets_the_image_ranges(self):
        cases = (
            (FlashAttentionBackend, False, torch.bfloat16, (9, 0), True),
            (FlashAttentionBackend, True, torch.bfloat16, (9, 0), True),
            (FlashAttentionBackend, False, torch.bfloat16, (8, 0), False),
            (FlashAttentionBackend, False, torch.float8_e4m3fn, (9, 0), False),
            (_OwnKernelBuild, False, torch.bfloat16, (9, 0), False),
        )
        for cls, wrap, kv_dtype, capability, expects_ranges in cases:
            with self.subTest(
                cls=cls.__name__, hybrid=wrap, kv_dtype=kv_dtype, capability=capability
            ):
                fa = cls.__new__(cls)
                fa.forward_metadata = FlashAttentionMetadata()
                fa.fa_impl_ver = 3
                fa.fa3_upgrades_to_fa4 = False
                fa.use_mla = False
                fa.kv_cache_dtype = kv_dtype
                backend = fa
                if wrap:
                    backend = HybridAttnBackend.__new__(HybridAttnBackend)
                    backend.prefill_backend = fa
                with (
                    patch.object(gemma3_mm, "get_attn_backend", return_value=backend),
                    patch.object(gemma3_mm, "is_cuda", return_value=True),
                    patch.object(
                        flashattention_backend,
                        "get_device_capability",
                        return_value=capability,
                    ),
                ):
                    Gemma3.prepare_attn_masks(
                        Gemma3.__new__(Gemma3),
                        forward_batch=_forward_batch(),
                        input_ids=torch.zeros(1, dtype=torch.long),
                        mask_dtype=torch.bool,
                    )
                if not expects_ranges:
                    self.assertIsNone(fa.forward_metadata.image_token_ranges)
                    continue
                self.assertIsNotNone(fa.forward_metadata.image_token_ranges)
                expected = Gemma3._build_image_token_ranges(
                    forward_batch=_forward_batch(), device="cpu"
                )
                torch.testing.assert_close(
                    fa.forward_metadata.image_token_ranges, expected
                )


if __name__ == "__main__":
    unittest.main()
