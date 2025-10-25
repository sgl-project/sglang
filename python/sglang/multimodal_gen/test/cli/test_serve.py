import asyncio
import base64
import subprocess
import time
import unittest

from openai import OpenAI

from sglang.multimodal_gen.runtime.utils.common import kill_process_tree
from sglang.multimodal_gen.test.test_utils import is_mp4, is_png, wait_for_port


class TestVideoHttpServer(unittest.TestCase):

    def _create_wait_and_download(
        self, client: OpenAI, prompt: str, size: str
    ) -> bytes:

        video = client.videos.create(prompt=prompt, size=size)
        video_id = video.id
        self.assertEqual(video.status, "queued")

        video = client.videos.retrieve(video_id)

        while video.status != "completed":
            time.sleep(3)
            video = client.videos.retrieve(video_id)

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
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "--port",
            "30010",
        ]

        process = subprocess.Popen(
            cls.base_command,
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
