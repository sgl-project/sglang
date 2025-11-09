# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

import asyncio
import base64
import subprocess
import tempfile
import time
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

from openai import OpenAI

from sglang.multimodal_gen.runtime.utils.common import kill_process_tree
from sglang.multimodal_gen.test.test_utils import is_mp4, is_png, wait_for_port


@contextmanager
def downloaded_temp_file(url: str, prefix: str = "i2v_input_", suffix: str = ".jpg"):
    tmp_path = Path(tempfile.gettempdir()) / f"{prefix}{uuid.uuid4().hex}{suffix}"
    with urlopen(url) as resp:
        tmp_path.write_bytes(resp.read())
    try:
        yield tmp_path
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


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
    timeout = 500
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
            client, "A plane is taking off.", "832x480"
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
                    "A ship is beside the port.",
                    "832x480",
                )
                for _ in range(num_requests)
            ]
            await asyncio.gather(*tasks)

        asyncio.run(send_concurrent_requests())


class TestImage2VideoHttpServer(unittest.TestCase):
    model_name = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    timeout = 1200
    extra_args = []

    def _create_wait_and_download(
        self, client: OpenAI, prompt: str, size: str
    ) -> bytes:

        image_url = "https://github.com/Wan-Video/Wan2.2/blob/990af50de458c19590c245151197326e208d7191/examples/i2v_input.JPG?raw=true"
        with downloaded_temp_file(
            image_url, prefix="i2v_input_", suffix=".jpg"
        ) as tmp_path:
            video = client.videos.create(
                prompt=prompt,
                input_reference=tmp_path,
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
            "sglang",
            "serve",
            "--model-path",
            f"{cls.model_name}",
            "--num-gpus",
            "4",
            "--ulysses-degree",
            "4",
            "--port",
            "30010",
        ]

        process = subprocess.Popen(
            cls.base_command + cls.extra_args,
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
            client, "A cat surfing on the sea.", "832x480"
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
                    "A cat surfing on the sea.",
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
