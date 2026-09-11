import base64
import io
import os
import re
import unittest
from unittest.mock import patch

import requests
from PIL import Image

from sglang.srt.utils.hf_transformers import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

register_cuda_ci(est_time=400, stage="extra-b", runner_config="8-gpu-h200")


class TestStep3p7Flash(DefaultServerBase):
    """Real-model regression coverage for batched Step3.7 image features."""

    model = "stepfun-ai/Step-3.7-Flash"
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3
    # Reuse the Step3.5 Flash E2E launch configuration. Keep these short
    # requests in one prefill batch so all image items reach the encoder.
    other_args = [
        "--tp",
        "8",
        "--trust-remote-code",
        "--attention-backend",
        "fa3",
        "--mem-fraction-static",
        "0.75",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "8192",
        "--max-running-requests",
        "8",
        "--disable-radix-cache",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
    ]

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = get_tokenizer(cls.model, trust_remote_code=True)
        # Hold scheduler input until the entire /generate batch has been
        # preprocessed. Concurrent HTTP calls alone can run as singletons.
        with patch.dict(os.environ, {"SGLANG_ENABLE_COLOCATED_BATCH_GEN": "1"}):
            super().setUpClass()

    def _make_request(self, images):
        image_data = []
        for color, size in images:
            with io.BytesIO() as buffer:
                Image.new("RGB", size, color).save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            image_data.append(f"data:image/png;base64,{encoded}")

        prompt = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": "<im_patch>\n"
                    * len(images)
                    + "Name the solid background color of each image in image order. "
                    "Reply only with the color names separated by commas.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        # This checkpoint's template opens <think> unconditionally. Close it
        # in the prefill so a simple color probe does not spend its token
        # budget on reasoning (enable_thinking=False is not supported).
        if prompt.endswith("<think>\n"):
            prompt += "</think>\n"
        return prompt, image_data

    def _generate(self, cases, batched):
        inputs = [self._make_request(case) for case in cases]
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": [text for text, _ in inputs] if batched else inputs[0][0],
                "image_data": (
                    [images for _, images in inputs] if batched else inputs[0][1]
                ),
                "sampling_params": {"temperature": 0, "max_new_tokens": 128},
            },
            timeout=180,
        )
        response.raise_for_status()
        outputs = response.json() if batched else [response.json()]
        self.assertEqual(len(outputs), len(cases))
        for case, output in zip(cases, outputs):
            with self.subTest(images=case, batched=batched):
                self.assertEqual(output["meta_info"]["finish_reason"]["type"], "stop")
                answer = output["text"].rsplit("</think>", 1)[-1].lower()
                colors = re.findall(r"\b(?:red|green|blue)\b", answer)
                self.assertEqual(colors, [color for color, _ in case], output["text"])

    def test_batched_image_features(self):
        # Fresh images: no earlier request can warm the embedding cache and
        # bypass get_image_feature. Small squares have no local patches;
        # large rectangles exercise patch + thumbnail assembly.
        self._generate(
            [
                [("red", (224, 224))],
                [("green", (1008, 504))],
                [("blue", (1008, 504)), ("red", (280, 280))],
                [("red", (336, 336)), ("blue", (504, 1008))],
            ],
            batched=True,
        )

    def test_single_image_features(self):
        # Distinct sizes keep these inputs out of the batched test's cache.
        for case in [[("green", (256, 256))], [("blue", (1008, 1008))]]:
            self._generate([case], batched=False)


if __name__ == "__main__":
    unittest.main()
