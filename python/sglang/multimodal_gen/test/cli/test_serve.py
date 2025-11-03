# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import base64
import subprocess
import time
import unittest
from pathlib import Path

from openai import OpenAI

from sglang.multimodal_gen.runtime.utils.common import kill_process_tree
from sglang.multimodal_gen.test.test_utils import is_mp4, is_png, wait_for_port


def wait_for_video_completion(client, video_id, timeout=300, check_interval=3):
    start = time.time()
    video = client.videos.retrieve(video_id)

    while video.status not in ("completed", "failed"):
        time.sleep(check_interval)
        video = client.videos.retrieve(video_id)
        assert time.time() - start < timeout, "video generate timeout"

    return video


class TestVideoHttpServer(unittest.TestCase):
    model_name = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    timeout = 120
    extra_args = []

    def _create_wait_and_download(
        self, client: OpenAI, prompt: str, size: str
    ) -> bytes:

        video = client.videos.create(prompt=prompt, size=size)
        video_id = video.id
        self.assertEqual(video.status, "queued")

        video = wait_for_video_completion(client, video_id, timeout=self.timeout)
        self.assertEqual(video.status, "completed", "video generate failed")

        response = client.videos.download_content(
            video_id=video_id,
        )
        content = response.read()
        return content

    @classmethod
    def setUpClass(cls):
        cls.base_command = [
            "sglang",
            "serve",
            "--model-path",
            f"{cls.model_name}",
            "--port",
            "30010",
        ]

        process = subprocess.Popen(
            cls.base_command + cls.extra_args,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        cls.pid = process.pid
        wait_for_port(host="127.0.0.1", port=30010)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.pid)

    def test_http_server_basic(self):
        client = OpenAI(
            api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1"
        )
        content = self._create_wait_and_download(
            client, "A calico cat playing a piano on stage", "832x480"
        )
        self.assertTrue(is_mp4(content))

    def test_concurrent_requests(self):
        client = OpenAI(
            api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1"
        )

        num_requests = 2

        async def generate_and_check_video(prompt, size):
            content = await asyncio.to_thread(
                self._create_wait_and_download, client, prompt, size
            )
            self.assertTrue(is_mp4(content))

        async def send_concurrent_requests():
            tasks = [
                generate_and_check_video(
                    "A dog playing a piano on stage",
                    "832x480",
                )
                for _ in range(num_requests)
            ]
            await asyncio.gather(*tasks)

        asyncio.run(send_concurrent_requests())


class TestFastWan2_1HttpServer(TestVideoHttpServer):
    model_name = "FastVideo/FastWan2.1-T2V-1.3B-Diffusers"


class TestFastWan2_2HttpServer(TestVideoHttpServer):
    model_name = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"


class TestImage2VideoHttpServer(unittest.TestCase):
    model_name = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    timeout = 1200
    extra_args = []

    def _create_wait_and_download(
        self, client: OpenAI, prompt: str, size: str
    ) -> bytes:

        image_path = "https://github.com/Wan-Video/Wan2.2/blob/990af50de458c19590c245151197326e208d7191/examples/i2v_input.JPG?raw=true"
        image_path = Path(image_path)
        video = client.videos.create(
            prompt=prompt,
            input_reference=image_path,
            size=size,
            seconds=10,
            extra_body={"fps": 16, "num_frames": 125},
        )
        # TODO: Some combinations of num_frames and fps may cause errors and need further investigation.
        video_id = video.id
        self.assertEqual(video.status, "queued")

        video = wait_for_video_completion(client, video_id, timeout=self.timeout)
        self.assertEqual(video.status, "completed", "video generate failed")

        response = client.videos.download_content(
            video_id=video_id,
        )
        content = response.read()
        return content

    @classmethod
    def setUpClass(cls):
        cls.base_command = [
            "sgl-diffusion",
            "serve",
            "--model-path",
            f"{cls.model_name}",
            "--port",
            "30010",
        ]

        process = subprocess.Popen(
            cls.base_command + cls.extra_args,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        cls.pid = process.pid
        wait_for_port(host="127.0.0.1", port=30010)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.pid)

    def test_http_server_basic(self):
        client = OpenAI(
            api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1"
        )
        content = self._create_wait_and_download(
            client, "A girl is fighting a monster.", "832x480"
        )
        self.assertTrue(is_mp4(content))

    def test_concurrent_requests(self):
        client = OpenAI(
            api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1"
        )

        num_requests = 2

        async def generate_and_check_video(prompt, size):
            content = await asyncio.to_thread(
                self._create_wait_and_download, client, prompt, size
            )
            self.assertTrue(is_mp4(content))

        async def send_concurrent_requests():
            tasks = [
                generate_and_check_video(
                    "A dog playing a piano on stage",
                    "832x480",
                )
                for _ in range(num_requests)
            ]
            await asyncio.gather(*tasks)

        asyncio.run(send_concurrent_requests())


class TestImageHttpServer(unittest.TestCase):

    def _create_wait_and_download(
        self, client: OpenAI, prompt: str, size: str
    ) -> bytes:
        img = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size=size,
            response_format="b64_json",
            output_format="png",
        )
        image_bytes = base64.b64decode(img.data[0].b64_json)
        return image_bytes

    @classmethod
    def setUpClass(cls):
        cls.base_command = [
            "sglang",
            "serve",
            "--model-path",
            "Qwen/Qwen-Image",
            "--port",
            "30020",
        ]

        process = subprocess.Popen(
            cls.base_command,
            # stdout=subprocess.PIPE,
            # stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        cls.pid = process.pid
        wait_for_port(host="127.0.0.1", port=30020)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.pid)

    def test_http_server_basic(self):
        client = OpenAI(
            api_key="sk-proj-1234567890", base_url="http://localhost:30020/v1"
        )
        content = self._create_wait_and_download(
            client, "A calico cat playing a piano on stage", "832x480"
        )
        self.assertTrue(is_png(content))

    def test_concurrent_requests(self):
        client = OpenAI(
            api_key="sk-proj-1234567890", base_url="http://localhost:30020/v1"
        )

        num_requests = 2

        async def generate_and_check_image(prompt, size):
            content = await asyncio.to_thread(
                self._create_wait_and_download, client, prompt, size
            )
            self.assertTrue(is_png(content))

        async def send_concurrent_requests():
            tasks = [
                generate_and_check_image(
                    "A dog playing a piano on stage",
                    "832x480",
                )
                for _ in range(num_requests)
            ]
            await asyncio.gather(*tasks)

        asyncio.run(send_concurrent_requests())


if __name__ == "__main__":
    # del TestPerform·anceBase
    unittest.main()
