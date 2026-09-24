"""Unit tests for Voxtral prompt id construction — no server, no model loading."""

import os
import shutil
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

from sglang.srt.multimodal.processors.voxtral import (
    AUDIO_TOKEN_ID,
    BEGIN_AUDIO_TOKEN_ID,
    INST_TOKEN_ID,
    VoxtralMultimodalProcessor,
)
from sglang.srt.utils.hf_transformers.tokenizer import get_tokenizer

VOXTRAL_REPO = "mistralai/Voxtral-Mini-3B-2507"


class TestVoxtralInputIds(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from huggingface_hub import hf_hub_download

        tekken = hf_hub_download(VOXTRAL_REPO, "tekken.json")
        cls._tmpdir = tempfile.mkdtemp()
        shutil.copy(tekken, os.path.join(cls._tmpdir, "tekken.json"))
        cls.tokenizer = get_tokenizer(cls._tmpdir)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_audio_tokens_inserted_after_last_inst(self):
        processor = object.__new__(VoxtralMultimodalProcessor)
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Transcribe this audio."}],
            tokenize=False,
        )

        input_ids = processor._build_input_ids_with_audio(self.tokenizer, prompt, [3])

        self.assertIsInstance(input_ids, list)
        self.assertTrue(all(isinstance(t, int) for t in input_ids))
        inst_pos = max(i for i, t in enumerate(input_ids) if t == INST_TOKEN_ID)
        self.assertEqual(
            input_ids[inst_pos + 1 : inst_pos + 5],
            [BEGIN_AUDIO_TOKEN_ID] + [AUDIO_TOKEN_ID] * 3,
        )
        self.assertEqual(
            processor._find_audio_offsets(input_ids, AUDIO_TOKEN_ID),
            [(inst_pos + 2, inst_pos + 4)],
        )


if __name__ == "__main__":
    unittest.main()
