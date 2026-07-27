import os
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.vlm_utils import IMAGE_MAN_IRONING_URL

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


class TestPreciseEmbeddingInterpolation(CustomTestCase):
    """Testcase: verify --enable-precise-embedding-interpolation changes ViT
    position-embedding interpolation on Qwen3-VL, producing different outputs
    for the same image at temperature=0

    [Test Category] Parameter
    [Test Target] --enable-precise-embedding-interpolation
    """

    model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH

    server_args = [
        "--trust-remote-code",
        "--enable-multimodal",
        "--attention-backend",
        "ascend",
        "--mm-attention-backend",
        "ascend_attn",
        "--mem-fraction-static",
        "0.8",
    ]

    def _launch_server(self, base_url, extra_args=None):
        env = os.environ.copy()
        # The parameter is only read inside the ViT cuda-graph path
        # (_prepare_graph_inputs → fast_pos_embed_interpolate).
        # Default eager path hardcodes torch.linspace and ignores the flag.
        env["SGLANG_VIT_ENABLE_CUDA_GRAPH"] = "true"

        args = list(self.server_args)
        if extra_args:
            args.extend(extra_args)

        self.process = popen_launch_server(
            self.model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args,
            env=env,
        )

    def _cleanup(self):
        if hasattr(self, "process") and self.process is not None:
            kill_process_tree(self.process.pid)
            self.process = None

    def _image_request(self, base_url):
        return requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": IMAGE_MAN_IRONING_URL},
                            },
                            {
                                "type": "text",
                                "text": "Describe this image in a sentence.",
                            },
                        ],
                    },
                ],
                "temperature": 0,
                # Limit output to keep inference predictable;
                # a single-sentence description is well under 128 tokens.
                "max_tokens": 128,
            },
        )

    def test_precise_embedding_interpolation_contrastive(self):
        # Port for the second launch (first uses DEFAULT_URL_FOR_TEST).
        # Sequential launch/teardown on the same port risks TIME_WAIT.
        alt_url = "http://127.0.0.1:23001"

        # ---- Launch WITH --enable-precise-embedding-interpolation ----
        self._launch_server(
            base_url=DEFAULT_URL_FOR_TEST,
            extra_args=["--enable-precise-embedding-interpolation"],
        )
        try:
            resp_enabled = self._image_request(DEFAULT_URL_FOR_TEST)
            self.assertEqual(
                resp_enabled.status_code,
                200,
                f"Image request failed (flag=enabled): HTTP {resp_enabled.status_code} "
                f"— check whether {IMAGE_MAN_IRONING_URL} is reachable",
            )
            text_enabled = resp_enabled.json()["choices"][0]["message"]["content"]
        finally:
            self._cleanup()

        # Brief pause so the OS releases the first port before binding
        # the second one.  Together with using a different port this
        # eliminates any TIME_WAIT race.
        time.sleep(2)

        # ---- Launch WITHOUT the flag (default: False) ----
        self._launch_server(base_url=alt_url)
        try:
            resp_default = self._image_request(alt_url)
            self.assertEqual(
                resp_default.status_code,
                200,
                f"Image request failed (flag=default): HTTP {resp_default.status_code} "
                f"— check whether {IMAGE_MAN_IRONING_URL} is reachable",
            )
            text_default = resp_default.json()["choices"][0]["message"]["content"]
        finally:
            self._cleanup()

        # Both outputs should describe the same image
        for text in (text_enabled, text_default):
            text_lower = text.lower()
            self.assertTrue(
                any(w in text_lower for w in ("man", "person", "driver", "holding")),
                f"Expected person-related word in: {text}",
            )
            self.assertTrue(
                any(w in text_lower for w in ("car", "vehicle", "suv", "cab", "taxi")),
                f"Expected vehicle-related word in: {text}",
            )

        # Core assertion: outputs differ, proving the flag changes interpolation.
        # If this fails with identical outputs it may indicate the ViT graph path
        # is not active (silent fallback to eager, which hardcodes align_corners=True
        # regardless of the flag) or the flag is no longer read at model init.
        self.assertNotEqual(
            text_enabled,
            text_default,
            "Outputs should differ because --enable-precise-embedding-interpolation "
            "changes _get_interpolation_indices (align_corners=True vs False). "
            "Identical outputs may mean the ViT graph path is not active "
            "(SGLANG_VIT_ENABLE_CUDA_GRAPH fallback to eager) or the flag is "
            "not being read at model init time.",
        )


if __name__ == "__main__":
    unittest.main()
