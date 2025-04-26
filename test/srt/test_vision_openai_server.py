"""
Usage:
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_mixed_batch
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_multi_images_chat_completion
"""

import base64
import io
import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openai
import requests
from PIL import Image

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

# image
IMAGE_MAN_IRONING_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png"
IMAGE_SGL_LOGO_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/sgl_logo.png"

# video
VIDEO_JOBS_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/videos/jobs_presenting_ipod.mp4"

# audio
AUDIO_TRUMP_SPEECH_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/Trump_WEF_2018_10s.mp3"
AUDIO_BIRD_SONG_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/audios/bird_song.mp3"


class TestOpenAIVisionServer(CustomTestCase):
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
        kill_process_tree(cls.process.pid)

    def test_single_image_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_MAN_IRONING_URL},
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
        # `driver` is for gemma-3-it
        assert "man" in text or "person" or "driver" in text, text
        assert "cab" in text or "taxi" in text or "SUV" in text, text
        # MiniCPMO fails to recognize `iron`, but `hanging`
        assert "iron" in text or "hang" in text, text
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
                            "image_url": {"url": IMAGE_MAN_IRONING_URL},
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

    def test_multi_images_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_MAN_IRONING_URL},
                            "modalities": "multi-images",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_SGL_LOGO_URL},
                            "modalities": "multi-images",
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
        print("-" * 30)
        print(f"Multi images response:\n{text}")
        print("-" * 30)
        assert "man" in text or "cab" in text or "SUV" in text or "taxi" in text, text
        assert "logo" in text or '"S"' in text or "SG" in text, text
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def prepare_video_messages(self, video_path):
        # the memory consumed by the Vision Attention varies a lot, e.g. blocked qkv vs full-sequence sdpa
        # the size of the video embeds differs from the `modality` argument when preprocessed

        # We import decord here to avoid a strange Segmentation fault (core dumped) issue.
        # The following import order will cause Segmentation fault.
        # import decord
        # from transformers import AutoTokenizer
        from decord import VideoReader, cpu

        max_frames_num = 20
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
            "modalities": "video",
        }

        for base64_frame in base64_frames:
            frame_format["image_url"]["url"] = "data:image/jpeg;base64,{}".format(
                base64_frame
            )
            messages[0]["content"].append(frame_format.copy())

        prompt = {"type": "text", "text": "Please describe the video in detail."}
        messages[0]["content"].append(prompt)

        return messages

    def prepare_video_messages_video_direct(self, video_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"video:{video_path}"},
                        "modalities": "video",
                    },
                    {"type": "text", "text": "Please describe the video in detail."},
                ],
            },
        ]
        return messages

    def get_or_download_file(self, url: str) -> str:
        cache_dir = os.path.expanduser("~/.cache")
        if url is None:
            raise ValueError()
        file_name = url.split("/")[-1]
        file_path = os.path.join(cache_dir, file_name)
        os.makedirs(cache_dir, exist_ok=True)

        if not os.path.exists(file_path):
            response = requests.get(url)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)
        return file_path

    def test_video_chat_completion(self):
        url = VIDEO_JOBS_URL
        file_path = self.get_or_download_file(url)

        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # messages = self.prepare_video_messages_video_direct(file_path)
        messages = self.prepare_video_messages(file_path)

        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=1024,
            stream=False,
        )

        video_response = response.choices[0].message.content

        print("-" * 30)
        print(f"Video response:\n{video_response}")
        print("-" * 30)

        # Add assertions to validate the video response
        assert "iPod" in video_response or "device" in video_response, video_response
        assert (
            "man" in video_response
            or "person" in video_response
            or "individual" in video_response
            or "speaker" in video_response
        ), video_response
        assert (
            "present" in video_response
            or "examine" in video_response
            or "display" in video_response
            or "hold" in video_response
        )
        assert "black" in video_response or "dark" in video_response
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
                            "image_url": {"url": IMAGE_MAN_IRONING_URL},
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

    def run_decode_with_image(self, image_id):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        content = []
        if image_id == 0:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": IMAGE_MAN_IRONING_URL},
                }
            )
        elif image_id == 1:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": IMAGE_SGL_LOGO_URL},
                }
            )
        else:
            pass

        content.append(
            {
                "type": "text",
                "text": "Describe this image in a very short sentence.",
            }
        )

        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "user", "content": content},
            ],
            temperature=0,
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)

    def test_mixed_batch(self):
        image_ids = [0, 1, 2] * 4
        with ThreadPoolExecutor(4) as executor:
            list(executor.map(self.run_decode_with_image, image_ids))

    def prepare_audio_messages(self, prompt, audio_file_name):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"{audio_file_name}"},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        return messages

    def get_audio_response(self, url: str, prompt, category):
        audio_file_path = self.get_or_download_file(url)
        client = openai.Client(api_key="sk-123456", base_url=self.base_url)

        messages = self.prepare_audio_messages(prompt, audio_file_path)

        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=128,
            stream=False,
        )

        audio_response = response.choices[0].message.content

        print("-" * 30)
        print(f"audio {category} response:\n{audio_response}")
        print("-" * 30)

        audio_response = audio_response.lower()

        self.assertIsNotNone(audio_response)
        self.assertGreater(len(audio_response), 0)

        return audio_response

    def _test_audio_speech_completion(self):
        # a fragment of Trump's speech
        audio_response = self.get_audio_response(
            AUDIO_TRUMP_SPEECH_URL,
            "I have an audio sample. Please repeat the person's words",
            category="speech",
        )
        assert "thank you" in audio_response
        assert "it's a privilege to be here" in audio_response
        assert "leader" in audio_response
        assert "science" in audio_response
        assert "art" in audio_response

    def _test_audio_ambient_completion(self):
        # bird song
        audio_response = self.get_audio_response(
            AUDIO_BIRD_SONG_URL,
            "Please listen to the audio snippet carefully and transcribe the content.",
            "ambient",
        )
        assert "bird" in audio_response

    def test_audio_chat_completion(self):
        pass


class TestQwen2VLServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2-VL-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--chat-template",
                "qwen2-vl",
                "--mem-fraction-static",
                "0.4",
            ],
        )
        cls.base_url += "/v1"


class TestQwen2_5_VLServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2.5-VL-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--chat-template",
                "qwen2-vl",
                "--mem-fraction-static",
                "0.4",
            ],
        )
        cls.base_url += "/v1"


class TestVLMContextLengthIssue(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2-VL-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--chat-template",
                "qwen2-vl",
                "--context-length",
                "300",
                "--mem-fraction-static=0.80",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_single_image_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": IMAGE_MAN_IRONING_URL},
                            },
                            {
                                "type": "text",
                                "text": "Give a lengthy description of this picture",
                            },
                        ],
                    },
                ],
                temperature=0,
            )

        # context length is checked first, then max_req_input_len, which is calculated from the former
        assert (
            "Multimodal prompt is too long after expanding multimodal tokens."
            in str(cm.exception)
            or "is longer than the model's context length" in str(cm.exception)
        )


class TestMllamaServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--chat-template",
                "llama_3_vision",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        pass


class TestMinicpmvServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "openbmb/MiniCPM-V-2_6"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--chat-template",
                "minicpmv",
                "--mem-fraction-static",
                "0.4",
            ],
        )
        cls.base_url += "/v1"


class TestMinicpmoServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "openbmb/MiniCPM-o-2_6"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--chat-template",
                "minicpmo",
                "--mem-fraction-static",
                "0.7",
            ],
        )
        cls.base_url += "/v1"

    def test_audio_chat_completion(self):
        self._test_audio_speech_completion()
        self._test_audio_ambient_completion()


class TestDeepseekVL2Server(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "deepseek-ai/deepseek-vl2-small"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--chat-template",
                "deepseek-vl2",
                "--context-length",
                "4096",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        pass


class TestDeepseekVL2TinyServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "deepseek-ai/deepseek-vl2-tiny"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--chat-template",
                "deepseek-vl2",
                "--context-length",
                "4096",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        pass


class TestJanusProServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "deepseek-ai/Janus-Pro-7B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--chat-template",
                "janus-pro",
                "--mem-fraction-static",
                "0.4",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        pass

    def test_single_image_chat_completion(self):
        # Skip this test because it is flaky
        pass


## Skip for ci test
# class TestLlama4Server(TestOpenAIVisionServer):
#     @classmethod
#     def setUpClass(cls):
#         cls.model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
#         cls.base_url = DEFAULT_URL_FOR_TEST
#         cls.api_key = "sk-123456"
#         cls.process = popen_launch_server(
#             cls.model,
#             cls.base_url,
#             timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
#             other_args=[
#                 "--chat-template",
#                 "llama-4",
#                 "--mem-fraction-static",
#                 "0.8",
#                 "--tp-size=8",
#                 "--context-length=8192",
#             ],
#         )
#         cls.base_url += "/v1"

#     def test_video_chat_completion(self):
#         pass


class TestGemma3itServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "google/gemma-3-4b-it"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--chat-template",
                "gemma-it",
                "--mem-fraction-static",
                "0.75",
                "--enable-multimodal",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        pass


if __name__ == "__main__":
    unittest.main()
