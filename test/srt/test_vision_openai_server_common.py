import io
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import openai
import pybase64
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


class TestOpenAIMLLMServerBase(CustomTestCase):
    model: str
    extra_args: list = []
    fixed_args: list = ["--trust-remote-code", "--enable-multimodal"]
    trust_remote_code: bool = True

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=(
                cls.extra_args + cls.fixed_args + ["--trust-remote-code"]
                if cls.trust_remote_code
                else []
            ),
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def get_vision_request_kwargs(self):
        return self.get_request_kwargs()

    def get_request_kwargs(self):
        return {}

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


class AudioOpenAITestMixin(TestOpenAIMLLMServerBase):
    def verify_speech_recognition_response(self, text):
        check_list = [
            "thank you",
            "it's a privilege to be here",
            "leader",
            "science",
            "art",
        ]
        for check_word in check_list:
            assert (
                check_word in text.lower()
            ), f"audio_response: ｜{text}｜ should contain ｜{check_word}｜"

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

    def get_audio_request_kwargs(self):
        return self.get_request_kwargs()

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
            **(self.get_audio_request_kwargs()),
        )

        audio_response = response.choices[0].message.content

        print("-" * 30)
        print(f"audio {category} response:\n{audio_response}")
        print("-" * 30)

        audio_response = audio_response.lower()

        self.assertIsNotNone(audio_response)
        self.assertGreater(len(audio_response), 0)

        return audio_response.lower()

    def test_audio_speech_completion(self):
        # a fragment of Trump's speech
        audio_response = self.get_audio_response(
            AUDIO_TRUMP_SPEECH_URL,
            "Listen to this audio and write down the audio transcription in English.",
            category="speech",
        )
        self.verify_speech_recognition_response(audio_response)

    def test_audio_ambient_completion(self):
        # bird song
        audio_response = self.get_audio_response(
            AUDIO_BIRD_SONG_URL,
            "Please listen to the audio snippet carefully and transcribe the content in English.",
            "ambient",
        )
        assert "bird" in audio_response


class ImageOpenAITestMixin(TestOpenAIMLLMServerBase):
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
                "text": "Describe this image in a sentence.",
            }
        )

        response = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "user", "content": content},
            ],
            temperature=0,
            **(self.get_vision_request_kwargs()),
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)

    def test_mixed_batch(self):
        image_ids = [0, 1, 2] * 4
        with ThreadPoolExecutor(4) as executor:
            list(executor.map(self.run_decode_with_image, image_ids))

    def verify_single_image_response(self, response):
        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)

        # `driver` is for gemma-3-it
        assert (
            "man" in text or "person" or "driver" in text
        ), f"text: {text}, should contain man, person or driver"
        assert (
            "cab" in text
            or "taxi" in text
            or "SUV" in text
            or "vehicle" in text
            or "car" in text
        ), f"text: {text}, should contain cab, taxi, SUV, vehicle or car"
        # MiniCPMO fails to recognize `iron`, but `hanging`
        assert (
            "iron" in text or "hang" in text or "cloth" in text or "holding" in text
        ), f"text: {text}, should contain iron, hang, cloth or holding"
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

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
                            "text": "Describe this image in a sentence.",
                        },
                    ],
                },
            ],
            temperature=0,
            **(self.get_vision_request_kwargs()),
        )

        print("-" * 30)
        print(f"Single image response:\n{response.choices[0].message.content}")
        print("-" * 30)

        self.verify_single_image_response(response)

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
                            "text": "Describe this image in a sentence.",
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
            **(self.get_vision_request_kwargs()),
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)
        assert (
            "man" in text or "cab" in text
        ), f"text: {text}, should contain man or cab"
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
                            "text": "I have two very different images. Please describe them.",
                        },
                    ],
                },
            ],
            temperature=0,
            **(self.get_vision_request_kwargs()),
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)
        print("-" * 30)
        print(f"Multi images response:\n{text}")
        print("-" * 30)
        assert (
            "man" in text
            or "cab" in text
            or "SUV" in text
            or "taxi" in text
            or "car" in text
        ), f"text: {text}, should contain man, cab, SUV, taxi or car"
        assert (
            "logo" in text or '"S"' in text or "SG" in text or "graphic" in text
        ), f"text: {text}, should contain logo, S or SG or graphic"
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def prepare_video_images_messages(self, video_path):
        # the memory consumed by the Vision Attention varies a lot, e.g. blocked qkv vs full-sequence sdpa
        # the size of the video embeds differs from the `modality` argument when preprocessed

        # We import decord here to avoid a strange Segmentation fault (core dumped) issue.
        # The following import order will cause Segmentation fault.
        # import decord
        # from transformers import AutoTokenizer
        from decord import VideoReader, cpu

        max_frames_num = 10
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
            base64_str = pybase64.b64encode(buff.getvalue()).decode("utf-8")
            base64_frames.append(base64_str)

        messages = [{"role": "user", "content": []}]
        frame_format = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,{}"},
            "modalities": "image",
        }

        for base64_frame in base64_frames:
            frame_format["image_url"]["url"] = "data:image/jpeg;base64,{}".format(
                base64_frame
            )
            messages[0]["content"].append(frame_format.copy())

        prompt = {"type": "text", "text": "Please describe the video in detail."}
        messages[0]["content"].append(prompt)

        return messages

    def test_video_images_chat_completion(self):
        url = VIDEO_JOBS_URL
        file_path = self.get_or_download_file(url)

        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        messages = self.prepare_video_images_messages(file_path)

        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=1024,
            stream=False,
        )

        video_response = response.choices[0].message.content

        print("-" * 30)
        print(f"Video images response:\n{video_response}")
        print("-" * 30)

        # Add assertions to validate the video response
        assert (
            "iPod" in video_response
            or "device" in video_response
            or "microphone" in video_response
        ), f"""
        ====================== video_images response =====================
        {video_response}
        ===========================================================
        should contain 'iPod' or 'device' or 'microphone'
        """
        assert (
            "man" in video_response
            or "person" in video_response
            or "individual" in video_response
            or "speaker" in video_response
            or "presenter" in video_response
            or "Steve" in video_response
            or "hand" in video_response
        ), f"""
        ====================== video_images response =====================
        {video_response}
        ===========================================================
        should contain 'man' or 'person' or 'individual' or 'speaker' or 'presenter' or 'Steve' or 'hand'
        """
        assert (
            "present" in video_response
            or "examine" in video_response
            or "display" in video_response
            or "hold" in video_response
        ), f"""
        ====================== video_images response =====================
        {video_response}
        ===========================================================
        should contain 'present' or 'examine' or 'display' or 'hold'
        """
        self.assertIsNotNone(video_response)
        self.assertGreater(len(video_response), 0)


class VideoOpenAITestMixin(TestOpenAIMLLMServerBase):
    def prepare_video_messages(self, video_path):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"{video_path}"},
                    },
                    {"type": "text", "text": "Please describe the video in detail."},
                ],
            },
        ]
        return messages

    def test_video_chat_completion(self):
        url = VIDEO_JOBS_URL
        file_path = self.get_or_download_file(url)

        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        messages = self.prepare_video_messages(file_path)

        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=1024,
            stream=False,
            **(self.get_vision_request_kwargs()),
        )

        video_response = response.choices[0].message.content.lower()

        print("-" * 30)
        print(f"Video response:\n{video_response}")
        print("-" * 30)

        # Add assertions to validate the video response
        assert (
            "ipod" in video_response
            or "device" in video_response
            or "microphone" in video_response
            or "phone" in video_response
        ), f"video_response: {video_response}, should contain 'iPod' or 'device'"
        assert (
            "man" in video_response
            or "person" in video_response
            or "individual" in video_response
            or "speaker" in video_response
            or "presenter" in video_response
            or "hand" in video_response
        ), f"video_response: {video_response}, should either have 'man' in video_response, or 'person' in video_response, or 'individual' in video_response or 'speaker' in video_response or 'presenter' or 'hand' in video_response"
        assert (
            "present" in video_response
            or "examine" in video_response
            or "display" in video_response
            or "hold" in video_response
        ), f"video_response: {video_response}, should contain 'present', 'examine', 'display', or 'hold'"
        assert (
            "black" in video_response or "dark" in video_response
        ), f"video_response: {video_response}, should contain 'black' or 'dark'"
        self.assertIsNotNone(video_response)
        self.assertGreater(len(video_response), 0)


class OmniOpenAITestMixin(
    ImageOpenAITestMixin, VideoOpenAITestMixin, AudioOpenAITestMixin
):
    def test_mixed_modality_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE_MAN_IRONING_URL},
                    },
                    {
                        "type": "audio_url",
                        "audio_url": {"url": AUDIO_TRUMP_SPEECH_URL},
                    },
                    {
                        "type": "text",
                        "text": "I have an image and audio, which are not related at all. Please:  1. Describe the image in a sentence, 2. Repeat the exact words from the audio I provided. Be exact",
                    },
                ],
            },
        ]
        response = client.chat.completions.create(
            model="default",
            messages=messages,
            temperature=0,
            max_tokens=128,
            stream=False,
        )

        text = response.choices[0].message.content

        print("-" * 30)
        print(f"Mixed modality response:\n{text}")
        print("-" * 30)

        self.verify_single_image_response(response=response)
        self.verify_speech_recognition_response(text=text)
