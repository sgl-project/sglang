"""Test SGLANG_MM_AVOID_RETOKENIZE on the pre-tokenized (list[int]) VLM path.

When a multimodal request arrives as input_ids, `load_mm_data` decodes them back
to text and the HF processor re-tokenizes them. If the original ids were
non-canonical (decode -> re-encode is not identity), that re-tokenization drifts
away from what the user sent. With SGLANG_MM_AVOID_RETOKENIZE ON (default), the
user's original text tokens are kept verbatim and only the image placeholder is
expanded.

This drives the REAL Qwen2.5-VL / Kimi-K2.5 processors (no stubbing) with a
predefined non-canonical prompt: the word "Describe" is split into "D"+"escribe",
which decodes to the same text but re-encodes to the single merged token (a real
drift). After processing, the non-image tokens must match the input verbatim.
"""

import asyncio
import unittest
from types import SimpleNamespace

from PIL import Image
from transformers import AutoConfig, AutoProcessor

from sglang.srt.environ import envs
from sglang.srt.multimodal.processors.kimi_k25 import KimiK2_5VLImageProcessor
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor
from sglang.srt.server_args import set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Real image preprocessing (esp. Kimi's GPU path) needs a GPU.
register_cuda_ci(est_time=180, suite="base-b-test-1-gpu-small")

QWEN_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
KIMI_MODEL = "moonshotai/Kimi-K2.5"


def _server_args():
    return SimpleNamespace(
        mm_process_config={},
        disable_fast_image_processor=False,
        keep_mm_feature_on_device=True,
        skip_tokenizer_init=False,
        tokenizer_worker_num=1,
    )


class TestMmAvoidRetokenize(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # process_mm_data reads the global server args for device placement.
        set_global_server_args_for_scheduler(
            SimpleNamespace(rl_on_policy_target=None, base_gpu_id=0)
        )

    def _check(self, model, processor_cls):
        hf_config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        hf_processor = AutoProcessor.from_pretrained(
            model, trust_remote_code=True, use_fast=True
        )
        proc = processor_cls(
            hf_config, _server_args(), hf_processor, transport_mode=None
        )
        tok = hf_processor.tokenizer
        image_token_id = proc.mm_tokens.image_token_id

        def enc(text):
            return tok.encode(text, add_special_tokens=False)

        # Predefined drift: "Describe" -> "D" + "escribe". Splitting a word at an
        # interior boundary is non-canonical: it decodes to the same text but
        # re-encodes to the single merged token. One image placeholder follows.
        input_ids = (
            enc("D")
            + enc("escribe")
            + enc(" the picture: ")
            + enc(proc.mm_tokens.image_token)
        )

        # Confirm the prompt really would drift if it were re-tokenized.
        self.assertNotEqual(
            tok.encode(tok.decode(input_ids), add_special_tokens=False),
            input_ids,
            f"{model}: prompt is canonical; would not exercise retokenize drift",
        )
        self.assertEqual(input_ids.count(image_token_id), 1)

        async def _go():
            return await proc.process_mm_data_async(
                image_data=[Image.new("RGB", (64, 64), (128, 128, 128))],
                input_text=list(input_ids),
                request_obj=SimpleNamespace(
                    video_data=None, audio_data=None, rid="test-rid"
                ),
            )

        def run(avoid_retokenize):
            with envs.SGLANG_MM_AVOID_RETOKENIZE.override(avoid_retokenize):
                return list(asyncio.run(_go()).input_ids)

        def non_image(ids):
            return [t for t in ids if t != image_token_id]

        on_ids = run(True)
        off_ids = run(False)

        # ON: the non-expanded (non-image) tokens are preserved verbatim.
        self.assertEqual(non_image(on_ids), non_image(input_ids))
        # The single image placeholder was expanded.
        self.assertGreater(on_ids.count(image_token_id), 1)

        # OFF: re-tokenization drifts, so the non-image tokens no longer match.
        self.assertNotEqual(
            non_image(off_ids),
            non_image(input_ids),
            f"{model}: with avoid-retokenize OFF the prompt should drift",
        )

    def test_qwen2_5_vl(self):
        self._check(QWEN_MODEL, QwenVLImageProcessor)

    def test_kimi_k2_5(self):
        self._check(KIMI_MODEL, KimiK2_5VLImageProcessor)


if __name__ == "__main__":
    unittest.main()
