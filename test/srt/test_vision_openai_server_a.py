"""
Usage:
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_mixed_batch
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_multi_images_chat_completion
"""

import base64
import logging
import os
import time
import unittest
from io import BytesIO

import requests
from openai import OpenAI
from PIL import Image
from test_vision_openai_server_common import *

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


class TestLlava(ImageOpenAITestMixin):
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
        )
        cls.base_url += "/v1"


class TestQwen2VLServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
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
                "--mem-fraction-static",
                "0.35",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestQwen2_5_VLServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
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
                "--mem-fraction-static",
                "0.35",
                "--cuda-graph-max-bs",
                "4",
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
                "--context-length",
                "300",
                "--mem-fraction-static=0.75",
                "--cuda-graph-max-bs",
                "4",
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


# Note(Xinyuan): mllama is not stable for now, skip for CI
# class TestMllamaServer(TestOpenAIVisionServer):
#     @classmethod
#     def setUpClass(cls):
#         cls.model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
#         cls.base_url = DEFAULT_URL_FOR_TEST
#         cls.api_key = "sk-123456"
#         cls.process = popen_launch_server(
#             cls.model,
#             cls.base_url,
#             timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
#             api_key=cls.api_key,
#         )
#         cls.base_url += "/v1"


class TestMinicpmvServer(ImageOpenAITestMixin):
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
                "--mem-fraction-static",
                "0.35",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestInternVL2_5Server(ImageOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "OpenGVLab/InternVL2_5-2B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestMinicpmoServer(ImageOpenAITestMixin, AudioOpenAITestMixin):
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
                "--mem-fraction-static",
                "0.65",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestMimoVLServer(ImageOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "XiaomiMiMo/MiMo-VL-7B-RL"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.6",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"



class TestQwen2_5_AI_AGENT(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2.5-VL-7B-Instruct"
        cls.url_1 = f"http://127.0.0.1:8000"
        cls.url_2 = f"http://127.0.0.1:8080"

        cls.api_key = "sk-123456"

        env_1 = os.environ.copy()
        env_2 = os.environ.copy()

        env_1["CUDA_VISIBLE_DEVICES"] = "0"
        env_2["CUDA_VISIBLE_DEVICES"] = "1"
        env_2["SGL_CACHE_MM_IMAGE"] = "1"

        logging.info("launch server without mm_item_cache")
        cls.process = popen_launch_server(
            cls.model,
            cls.url_1,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--mem-fraction-static",
                "0.7",
                "--cuda-graph-max-bs",
                "16",
                "--tensor-parallel-size",
                "1",
                "--mm-attention-backend",
                "fa3",
                "--attention-backend",
                "flashinfer",
            ],
            env=env_1,
        )

        logging.info("launch server with mm_item_cache")
        cls.process_cache = popen_launch_server(
            cls.model,
            cls.url_2,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--mem-fraction-static",
                "0.7",
                "--cuda-graph-max-bs",
                "16",
                "--tensor-parallel-size",
                "1",
                "--mm-attention-backend",
                "fa3",
                "--attention-backend",
                "flashinfer",
            ],
            env=env_2,
        )
        cls.url_1 += "/v1"
        cls.url_2 += "/v1"

        cls.base64_strs = []
        cls.get_agent_images_base64()

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        kill_process_tree(cls.process_cache.pid)

    @classmethod
    def get_agent_images_base64(cls):
        def image_to_base64(img, format="PNG"):
            buffer = BytesIO()
            img.save(buffer, format=format)
            img_bytes = buffer.getvalue()
            base64_str = base64.b64encode(img_bytes).decode("utf-8")
            return base64_str

        target_size = (1920, 1080)
        response = requests.get(IMAGE_MAN_IRONING_URL)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))

        img_1080p = img.resize(target_size, Image.LANCZOS)
        flipped_vertical = img_1080p.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_horizontal = img_1080p.transpose(Image.FLIP_LEFT_RIGHT)
        rotated_clockwise_45 = img_1080p.rotate(-45, expand=True)
        rotated_counterclockwise_45 = img_1080p.rotate(45, expand=True)
        from PIL import ImageFilter

        img_blur = img_1080p.filter(ImageFilter.GaussianBlur(radius=5))

        for img_item in [
            img_1080p,
            flipped_vertical,
            flipped_horizontal,
            rotated_clockwise_45,
            rotated_counterclockwise_45,
            img_blur,
        ]:
            cls.base64_strs.append(image_to_base64(img_item))

    def test_ai_agent_opt(self):
        print("serving agent chat wo cache...")
        answer_normal, time_normal = self._test_AIAgent_chat_completion()
        print("serving agent chat with cache")
        answer_cache, time_cache = self._test_AIAgent_chat_completion(with_cache=True)
        print(answer_cache, " <--> ", answer_normal)
        assert answer_normal == answer_cache

        print(time_cache, " <--> ", time_normal)
        assert time_cache < time_normal

    def _test_AIAgent_chat_completion(self, with_cache=False):

        assert len(self.base64_strs) >= 6, "test image data was not ready"
        if with_cache:
            client = OpenAI(
                api_key=TestQwen2_5_AI_AGENT.api_key,
                base_url=TestQwen2_5_AI_AGENT.url_2,
            )
        else:
            client = OpenAI(
                api_key=TestQwen2_5_AI_AGENT.api_key,
                base_url=TestQwen2_5_AI_AGENT.url_1,
            )

        R1_AIagent_messages = [
            {
                "role": "system",
                "content": '## character \n you are GUI Agent, familiar with Windows、Linux os \n please finish user\'s task according to user\'s input, screen-shot and history actions \n you should finish the task step by step ,output One Action once, please follow following format \n\n## output format\nAction_Summary: ...\nAction: ...\n\n use strictly"Action_Summary: "prefix和"Action: "prefix。\n use English in Action_Summary use function call inAction。\n\n## Actionformat\n### click(start_box="<bbox>left_x top_y right_x bottom_y</bbox>")\n### left_double_click(start_box="<bbox>left_x top_y right_x bottom_y</bbox>")\n### right_click(start_box="<bbox>left_x top_y right_x bottom_y</bbox>")\n### drag(start_box="<bbox>left_x top_y right_x bottom_y</bbox>", end_box="<bbox>left_x top_y right_x bottom_y</bbox>")\n### type(content="content") // If you want to submit your input, next action use hotkey(key="enter")\n### hotkey(key="key")\n### scroll(direction:Enum[up,down,left,right]="direction",start_box="<bbox>left_x top_y right_x bottom_y</bbox>")\n### wait()\n### finished()\n### call_user() // Submit the task and call the user when the task is unsolvable, or when you need the user"s help.\n### save_memory(content="content") // 当用户明确表示“记住……”或类似表述时，自动调用`save_memory`保存记忆。next action use finished\n### output(content="content") // It is only used when the user specifies to use output, and after output is executed, it cannot be executed again.\n',
            },
            {"role": "user", "content": "please booking a hotel for me\n\n"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/any;base64,{TestQwen2_5_AI_AGENT.base64_strs[0]}"
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": 'Action_Summary: yes, I understand. firstly, I need to open the browser  \nAction: type(content="bowel")',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/any;base64,{TestQwen2_5_AI_AGENT.base64_strs[1]}"
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": 'Action_Summary: yes, I understand. firstly, I need to open the browser  \nAction: click(start_box="<bbox>570 446 625 478</bbox>")',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/any;base64,{TestQwen2_5_AI_AGENT.base64_strs[2]}"
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": 'Action_Summary: yes, I understand. firstly, I need to open the browser  \nAction: click(start_box="<bbox>497 667 517 690</bbox>")',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/any;base64,{TestQwen2_5_AI_AGENT.base64_strs[3]}"
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": 'Action_Summary: yes, I understand. firstly, I need to open the browser  \nAction: click(start_box="<bbox>367 711 386 733</bbox>")',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/any;base64,{TestQwen2_5_AI_AGENT.base64_strs[4]}"
                        },
                    }
                ],
            },
        ]

        R2_AIagent_messages = [
            {
                "role": "system",
                "content": '## character \n you are GUI Agent, familiar with Windows、Linux os \n please finish user\'s task according to user\'s input, screen-shot and history actions \n you should finish the task step by step ,output One Action once, please follow following format \n\n## output format\nAction_Summary: ...\nAction: ...\n\n use strictly"Action_Summary: "prefix和"Action: "prefix。\n use English in Action_Summary use function call inAction。\n\n## Actionformat\n### click(start_box="<bbox>left_x top_y right_x bottom_y</bbox>")\n### left_double_click(start_box="<bbox>left_x top_y right_x bottom_y</bbox>")\n### right_click(start_box="<bbox>left_x top_y right_x bottom_y</bbox>")\n### drag(start_box="<bbox>left_x top_y right_x bottom_y</bbox>", end_box="<bbox>left_x top_y right_x bottom_y</bbox>")\n### type(content="content") // If you want to submit your input, next action use hotkey(key="enter")\n### hotkey(key="key")\n### scroll(direction:Enum[up,down,left,right]="direction",start_box="<bbox>left_x top_y right_x bottom_y</bbox>")\n### wait()\n### finished()\n### call_user() // Submit the task and call the user when the task is unsolvable, or when you need the user"s help.\n### save_memory(content="content") // 当用户明确表示“记住……”或类似表述时，自动调用`save_memory`保存记忆。next action use finished\n### output(content="content") // It is only used when the user specifies to use output, and after output is executed, it cannot be executed again.\n',
            },
            {"role": "user", "content": "please booking a hotel for me\n\n"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/any;base64,{TestQwen2_5_AI_AGENT.base64_strs[1]}"
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": 'Action_Summary: yes, I understand. firstly, I need to open the browser  \nAction: type(content="bowel")',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/any;base64,{TestQwen2_5_AI_AGENT.base64_strs[2]}"
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": 'Action_Summary: yes, I understand. firstly, I need to open the browser  \nAction: click(start_box="<bbox>570 446 625 478</bbox>")',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/any;base64,{TestQwen2_5_AI_AGENT.base64_strs[3]}"
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": 'Action_Summary: yes, I understand. firstly, I need to open the browser  \nAction: click(start_box="<bbox>497 667 517 690</bbox>")',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/any;base64,{TestQwen2_5_AI_AGENT.base64_strs[4]}"
                        },
                    }
                ],
            },
            {
                "role": "assistant",
                "content": 'Action_Summary: yes, I understand. firstly, I need to open the browser  \nAction: click(start_box="<bbox>367 711 386 733</bbox>")',
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/any;base64,{TestQwen2_5_AI_AGENT.base64_strs[5]}"
                        },
                    }
                ],
            },
        ]

        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=R1_AIagent_messages,
            temperature=0,
            max_tokens=8,
        )

        start_time = time.time()
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=R2_AIagent_messages,
            temperature=0,
            max_tokens=8,
        )
        end_time = time.time()

        text = response.choices[0].message.content
        cost_time = (end_time - start_time) * 1000

        return (text, cost_time)

class TestVILAServer(ImageOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "Efficient-Large-Model/NVILA-Lite-2B-hf-0626"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.revision = "6bde1de5964b40e61c802b375fff419edc867506"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--trust-remote-code",
                "--context-length=65536",
                f"--revision={cls.revision}",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestPhi4MMServer(ImageOpenAITestMixin, AudioOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        # Manually download LoRA adapter_config.json as it's not downloaded by the model loader by default.
        from huggingface_hub import constants, snapshot_download

        snapshot_download(
            "microsoft/Phi-4-multimodal-instruct",
            allow_patterns=["**/adapter_config.json"],
        )

        cls.model = "microsoft/Phi-4-multimodal-instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        revision = "33e62acdd07cd7d6635badd529aa0a3467bb9c6a"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.70",
                "--disable-radix-cache",
                "--max-loras-per-batch",
                "2",
                "--revision",
                revision,
                "--lora-paths",
                f"vision={constants.HF_HUB_CACHE}/models--microsoft--Phi-4-multimodal-instruct/snapshots/{revision}/vision-lora",
                f"speech={constants.HF_HUB_CACHE}/models--microsoft--Phi-4-multimodal-instruct/snapshots/{revision}/speech-lora",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"

    def get_vision_request_kwargs(self):
        return {
            "extra_body": {
                "lora_path": "vision",
                "top_k": 1,
                "top_p": 1.0,
            }
        }

    def get_audio_request_kwargs(self):
        return {
            "extra_body": {
                "lora_path": "speech",
                "top_k": 1,
                "top_p": 1.0,
            }
        }

    # This _test_audio_ambient_completion test is way too complicated to pass for a small LLM
    def test_audio_ambient_completion(self):
        pass



if __name__ == "__main__":
    del (
        TestOpenAIOmniServerBase,
        ImageOpenAITestMixin,
        VideoOpenAITestMixin,
        AudioOpenAITestMixin,
    )
    unittest.main()
