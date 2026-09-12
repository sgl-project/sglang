"""Image appends must generate the same tokens as independent full-history replay."""

import base64
import io
import unittest
import uuid

import requests
from PIL import Image

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-small")


class TestSessionMrope(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-streaming-session",
                "--context-length",
                "4096",
                "--mem-fraction-static",
                "0.5",
                "--max-running-requests",
                "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def post(self, path, **payload):
        response = requests.post(self.base_url + path, json=payload, timeout=180)
        response.raise_for_status()
        return response.json() if response.content else None

    def test_image_append(self):
        images = []
        for color in ("red", "blue"):
            buffer = io.BytesIO()
            Image.new("RGB", (112, 112), color).save(buffer, format="PNG")
            images.append(
                "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
            )
        prompt = (
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            "What color is this image?<|im_end|>\n<|im_start|>assistant\n"
        )
        sampling = dict(
            temperature=0,
            max_new_tokens=8,
            ignore_eos=True,
            skip_special_tokens=False,
            no_stop_trim=True,
        )
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                salt = uuid.uuid4().hex
                sid = self.post(
                    "/open_session", capacity_of_str_len=4096, streaming=streaming
                )
                try:
                    first = self.post(
                        "/generate",
                        text=prompt,
                        image_data=images[:1],
                        session_params={"id": sid},
                        cache_salt=salt,
                        sampling_params=sampling,
                    )
                    suffix = "<|im_end|>\n" + prompt
                    # Regular mode also branches from the first turn after an append.
                    for image in images[1:] + (images[:1] if not streaming else []):
                        # A separate cache namespace prevents the reference from warming the session.
                        reference = self.post(
                            "/generate",
                            text=prompt + first["text"] + suffix,
                            image_data=[images[0], image],
                            cache_salt=uuid.uuid4().hex,
                            sampling_params=sampling,
                        )
                        appended = self.post(
                            "/generate",
                            text=suffix,
                            image_data=[image],
                            session_params={"id": sid, "rid": first["meta_info"]["id"]},
                            cache_salt=salt,
                            sampling_params=sampling,
                        )
                        self.assertEqual(
                            appended["output_ids"], reference["output_ids"]
                        )
                finally:
                    self.post("/close_session", session_id=sid)
                self.assertEqual(
                    requests.get(self.base_url + "/health", timeout=10).status_code, 200
                )


if __name__ == "__main__":
    unittest.main()
