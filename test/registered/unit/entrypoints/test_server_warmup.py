"""Unit tests for model-specific server warmup inputs."""

import asyncio
import base64
import struct
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.srt.entrypoints.http_server import (
    KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64,
    KIMI_VLM_WARMUP_PNG_PICTURE_BASE64,
    MINIMUM_PNG_PICTURE_BASE64,
    _get_vlm_warmup_image_base64,
)
from sglang.srt.entrypoints.warmup import prefill_shapes, voice_chat
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestVlmWarmupImage(CustomTestCase):
    def test_kimi_k2_uses_representative_vision_image(self):
        image_base64 = _get_vlm_warmup_image_base64(
            {"architectures": ["KimiK25ForConditionalGeneration"]}
        )
        self.assertEqual(image_base64, KIMI_VLM_WARMUP_PNG_PICTURE_BASE64)

        png = base64.b64decode(KIMI_VLM_WARMUP_PNG_PICTURE_BASE64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (512, 512))

    def test_kimi_k3_uses_native_patch_grid_image(self):
        for model_info in (
            {"architectures": ["KimiK3ForConditionalGeneration"]},
            {"architectures": None, "model_type": "kimi_k3"},
        ):
            with self.subTest(model_info=model_info):
                self.assertEqual(
                    _get_vlm_warmup_image_base64(model_info),
                    KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64,
                )

        png = base64.b64decode(KIMI_K3_VLM_WARMUP_PNG_PICTURE_BASE64)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (448, 448))

    def test_other_vlms_keep_minimal_startup_image(self):
        self.assertEqual(
            _get_vlm_warmup_image_base64(
                {"architectures": ["Qwen3VLForConditionalGeneration"]}
            ),
            MINIMUM_PNG_PICTURE_BASE64,
        )
        self.assertEqual(
            _get_vlm_warmup_image_base64({"architectures": None}),
            MINIMUM_PNG_PICTURE_BASE64,
        )


class TestWarmupTokenVocabulary(CustomTestCase):
    def test_synthetic_warmups_respect_small_and_large_vocabularies(self):
        # Pin the largest possible random token so a fixed 65536-token range
        # cannot happen to pass for a smaller model's vocabulary.
        for warmup in (voice_chat, prefill_shapes):
            for vocab_size in (16, 100000):
                with self.subTest(warmup=warmup.__name__, vocab_size=vocab_size):
                    requests = []

                    async def generate(request, _):
                        requests.append(request)
                        yield {}

                    manager = SimpleNamespace(
                        model_config=SimpleNamespace(
                            vocab_size=vocab_size,
                            hf_text_config=SimpleNamespace(vocab_size=vocab_size),
                        ),
                        generate_request=generate,
                    )
                    with (
                        patch(
                            "sglang.srt.entrypoints.warmup.np.random.randint",
                            side_effect=lambda high, size: np.full(size, high - 1),
                        ),
                        patch(
                            "sglang.srt.entrypoints.warmup.tqdm.trange",
                            return_value=[1],
                        ),
                        patch(
                            "sglang.srt.entrypoints.warmup.tqdm.tqdm",
                            side_effect=lambda sizes, **kwargs: sizes[:1],
                        ),
                    ):
                        asyncio.run(warmup("null", manager))
                    self.assertEqual(len(requests), 1)
                    tokens = requests[0].input_ids
                    self.assertTrue(tokens)
                    self.assertEqual(set(tokens), {min(65536, vocab_size) - 1})


if __name__ == "__main__":
    unittest.main()
