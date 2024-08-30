import base64
import io
import json
import os
import unittest

import numpy as np
import openai
import requests
from decord import VideoReader, cpu
from PIL import Image

from sglang.srt.utils import kill_child_process
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestOpenAIVisionServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--chat-template",
                "chatml-llava",
                # "--log-requests",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_child_process(cls.process.pid)

    def test_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in a very short sentence.",
                        },
                    ],
                },
            ],
            temperature=0,
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)
        assert "man" in text or "cab" in text, text
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_multi_turn_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in a very short sentence.",
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "There is a man at the back of a yellow cab ironing his clothes.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Repeat your previous answer."}
                    ],
                },
            ],
            temperature=0,
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)
        assert "man" in text or "cab" in text, text
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_mult_images_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/test/lang/example_image.png"
                            },
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png"
                            },
                        },
                        {
                            "type": "text",
                            "text": "I have two very different images. They are not related at all. "
                            "Please describe the first image in one sentence, and then describe the second image in another sentence.",
                        },
                    ],
                },
            ],
            temperature=0,
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)
        print(text)
        assert "man" in text and "taxi" in text, text
        assert "logo" in text, text
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def prepare_video_messages(self, video_path):
        max_frames_num = 32
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frame_num = len(vr)
        uniform_sampled_frames = np.linspace(
            0, total_frame_num - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()

        base64_frames = []
        for frame in frames:
            pil_img = Image.fromarray(frame)
            buff = io.BytesIO()
            pil_img.save(buff, format="JPEG")
            base64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
            base64_frames.append(base64_str)

        messages = [{"role": "user", "content": []}]
        frame_format = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,{}"},
        }

        for base64_frame in base64_frames:
            frame_format["image_url"]["url"] = "data:image/jpeg;base64,{}".format(
                base64_frame
            )
            messages[0]["content"].append(frame_format.copy())

        prompt = {"type": "text", "text": "Please describe the video in detail."}
        messages[0]["content"].append(prompt)

        return messages

    def test_video_chat_completion(self):
        url = "https://raw.githubusercontent.com/EvolvingLMMs-Lab/sglang/dev/onevision_local/assets/jobs.mp4"
        cache_dir = os.path.expanduser("~/.cache")
        file_path = os.path.join(cache_dir, "jobs.mp4")
        os.makedirs(cache_dir, exist_ok=True)

        if not os.path.exists(file_path):
            response = requests.get(url)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        messages = self.prepare_video_messages(file_path)

        video_request = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=1024,
            stream=True,
        )

        print("-" * 30)
        video_response = ""
        for chunk in video_request:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                video_response += content
                print(content, end="", flush=True)
        print("-" * 30)

        # Add assertions to validate the video response
        self.assertIsNotNone(video_response)
        self.assertGreater(len(video_response), 0)

    def test_regex(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        regex = (
            r"""\{\n"""
            + r"""   "color": "[\w]+",\n"""
            + r"""   "number_of_cars": [\d]+\n"""
            + r"""\}"""
        )

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://github.com/sgl-project/sglang/blob/main/test/lang/example_image.png?raw=true"
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in the JSON format.",
                        },
                    ],
                },
            ],
            temperature=0,
            extra_body={"regex": regex},
        )
        text = response.choices[0].message.content

        try:
            js_obj = json.loads(text)
        except (TypeError, json.decoder.JSONDecodeError):
            print("JSONDecodeError", text)
            raise
        assert isinstance(js_obj["color"], str)
        assert isinstance(js_obj["number_of_cars"], int)


if __name__ == "__main__":
    unittest.main()
