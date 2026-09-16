import base64
import io
import re
import unittest
from types import SimpleNamespace

import requests
from PIL import Image

from sglang.srt.utils.hf_transformers import get_tokenizer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.default_fixture import (
    DefaultServerBase,
    openai_api_env,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    is_in_ci,
    write_github_step_summary,
)

register_cuda_ci(est_time=1200, stage="extra-b", runner_config="8-gpu-h200")


class TestStep3p7Flash(DefaultServerBase):
    """Real-model coverage for batched Step3.7 images and GSM8K accuracy."""

    model = "stepfun-ai/Step-3.7-Flash"
    timeout = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3
    # Hold scheduler input until the entire /generate batch is preprocessed.
    server_env = {"SGLANG_ENABLE_COLOCATED_BATCH_GEN": "1"}
    # Reuse the Step3.5 Flash E2E launch configuration.
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
        # so a short image probe does not spend its budget on reasoning.
        if prompt.endswith("<think>\n"):
            prompt += "</think>\n"
        return prompt, image_data

    def _generate(self, inputs, batched):
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
        self.assertEqual(len(outputs), len(inputs))
        return outputs

    @staticmethod
    def _colors(output):
        answer = output["text"].rsplit("</think>", 1)[-1].lower()
        return re.findall(r"\b(?:red|green|blue)\b", answer)

    def test_batched_image_features(self):
        # Eight short cases exercise patch-free and patch + thumbnail paths.
        cases = [
            [("red", (224, 224))],
            [("green", (1008, 504))],
            [("blue", (1008, 504)), ("red", (280, 280))],
            [("red", (336, 336)), ("blue", (504, 1008))],
            [("green", (256, 256))],
            [("blue", (1008, 1008))],
            [("red", (504, 1008)), ("green", (336, 336))],
            [("green", (1120, 560)), ("blue", (280, 280))],
        ]
        inputs = [self._make_request(case) for case in cases]

        # Run each four-request batch before the single-request baseline so
        # fresh images reach get_image_feature() on the changed batched path.
        batched_outputs = []
        for start in range(0, len(cases), 4):
            batched_outputs.extend(self._generate(inputs[start : start + 4], True))
        serial_outputs = [self._generate([item], False)[0] for item in inputs]

        scores = {}
        for mode, outputs in [("batch1", serial_outputs), ("batch4", batched_outputs)]:
            correct = 0
            for case, output in zip(cases, outputs):
                with self.subTest(mode=mode, images=case):
                    self.assertTrue(output["text"].strip(), output)
                predicted = self._colors(output)
                expected = [color for color, _ in case]
                correct += predicted == expected
                print(f"Step3.7 {mode}: expected={expected}, predicted={predicted}")
            scores[mode] = correct / len(cases)

        summary = (
            f"Step3.7 short-image accuracy (8 cases): "
            f"batch1={scores['batch1']:.3f}, batch4={scores['batch4']:.3f}, "
            f"drop={scores['batch1'] - scores['batch4']:.3f}"
        )
        print(summary)
        if is_in_ci():
            write_github_step_summary(summary + "\n")
        # One case is the smallest nonzero difference on this eight-case set.
        # Gate only a larger regression until a Step3.7 accuracy baseline exists.
        self.assertGreaterEqual(scores["batch4"], scores["batch1"] - 1 / 8, summary)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=4096,
            num_examples=500,
            num_threads=128,
        )
        with openai_api_env("EMPTY"):
            metrics = run_eval(args)
        summary = f"Step3.7 GSM8K (500 samples): score={metrics['score']:.4f}"
        print(summary)
        if is_in_ci():
            write_github_step_summary(summary + "\n")
        # Initial floor from test_step3p5_flash_chain_mtp.py; this has
        # not yet been calibrated against a measured Step3.7 baseline.
        self.assertGreater(metrics["score"], 0.83, summary)


if __name__ == "__main__":
    unittest.main()
