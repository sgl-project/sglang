"""
Usage:
python3 -m unittest test_vision_chunked_prefill.TestVisionChunkedPrefill.test_chunked_prefill
"""

import base64
import io
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import numpy as np
import requests
from PIL import Image

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestVisionChunkedPrefill(CustomTestCase):

    def prepare_video_messages(self, video_path, max_frames_num=8):
        # We import decord here to avoid a strange Segmentation fault (core dumped) issue.
        # The following import order will cause Segmentation fault.
        # import decord
        # from transformers import AutoTokenizer
        from decord import VideoReader, cpu

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

        prompt = {"type": "text", "text": "Please describe the video briefly."}
        messages[0]["content"].append(prompt)

        return messages

    def get_prompt_from_messages(self, messages):
        text = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
        )
        image_data = []
        for content in messages[0]["content"]:
            if content["type"] == "image_url":
                text += "<image>\n"
                image_data.append(content["image_url"]["url"])
        text += "Please describe the video briefly.<|im_end|>\n<|im_start|>assistant\n"
        return text, image_data

    def generate(self, text, image_data):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": text,
                "image_data": image_data,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "no_stop_trim": True,
                    "skip_special_tokens": False,
                },
                "modalities": ["multi-images"],
            },
        ).json()
        return response["text"]

    def generate_for_video(self, batch, num_frame) -> Union[str, list[str]]:
        # prepare the video input about Steven introducing ipod nano
        url = "https://raw.githubusercontent.com/evolvinglmms-lab/sglang/dev/onevision_local/assets/jobs.mp4"
        cache_dir = os.path.expanduser("~/.cache")
        file_path = os.path.join(cache_dir, "jobs.mp4")
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.exists(file_path):
            response = requests.get(url)
            response.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(response.content)

        if not batch:
            assert isinstance(num_frame, int)
            messages = self.prepare_video_messages(file_path, max_frames_num=num_frame)
            text, image_data = self.get_prompt_from_messages(messages)
            return self.generate(text, image_data)
        else:
            assert isinstance(num_frame, list)
            func_args = []
            for max_frames_num in num_frame:
                messages = self.prepare_video_messages(
                    file_path,
                    max_frames_num=max_frames_num,
                )
                text, image_data = self.get_prompt_from_messages(messages)
                func_args.append((text, image_data))

            with ThreadPoolExecutor(max_workers=10) as executor:
                responses = list(executor.map(lambda p: self.generate(*p), func_args))

            return responses

    def launch_server(self, chunked_prefill_size) -> int:
        # launch server
        model = "lmms-lab/llava-onevision-qwen2-7b-ov"
        # model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        self.base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--chunked-prefill-size",
                f"{chunked_prefill_size}",
            ],
        )
        return process.pid

    def _test_chunked_prefill(self, batches, num_frames):
        # Chunked
        chunked_server_pid = self.launch_server(chunked_prefill_size=1024)
        try:
            outputs_chunked = []
            for batch, num_frame in zip(batches, num_frames):
                output_chunked = self.generate_for_video(
                    batch=batch, num_frame=num_frame
                )
                outputs_chunked += [output_chunked]
        finally:
            kill_process_tree(chunked_server_pid)

        # None-chunked
        try:
            no_chunked_server_pid = self.launch_server(chunked_prefill_size=-1)
            outputs_no_chunked = []
            for batch, num_frame in zip(batches, num_frames):
                output_no_chunked = self.generate_for_video(
                    batch=batch, num_frame=num_frame
                )
                outputs_no_chunked += [output_no_chunked]

        finally:
            kill_process_tree(no_chunked_server_pid)

        for output_chunked, output_no_chunked in zip(
            outputs_chunked, outputs_no_chunked
        ):
            print("output with chunked prefill:")
            print(output_chunked)
            print("output without chunked prefill:")
            print(output_no_chunked)
            assert output_chunked == output_no_chunked

    def test_chunked_prefill(self):
        self._test_chunked_prefill(batches=[False, True], num_frames=[1, [2, 6, 8, 10]])


if __name__ == "__main__":
    unittest.main()
