import os
import threading
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.mmmu_vlm_kit import MMMUMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)
from sglang.test.vlm_utils import (
    IMAGE_MAN_IRONING_URL,
    IMAGE_SGL_LOGO_URL,
)

# Omni model for local testing; override via env var EPD_OMNI_MODEL
KIMI_VL_MODEL = "moonshotai/Kimi-VL-A3B-Instruct"


register_cuda_ci(est_time=242, stage="extra-b", runner_config="4-gpu-h100")


class TestEPDDisaggregationKimiVL(PDDisaggregationServerBase):
    """Regression test for Kimi-VL two-dimensional image grids in E/PD mode."""

    model = KIMI_VL_MODEL
    model_args = [
        "--context-length=8192",
        "--dtype=bfloat16",
        "--mem-fraction-static=0.40",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.process_encode = None
        cls.encode_port = f"{int(cls.lb_port) + 307}"
        cls.encode_url = f"http://{cls.base_host}:{cls.encode_port}"

        cls.start_encode()
        prefill_thread = threading.Thread(target=cls.start_prefill)
        decode_thread = threading.Thread(target=cls.start_decode)
        prefill_thread.start()
        decode_thread.start()
        prefill_thread.join()
        decode_thread.join()

        cls.wait_server_ready(cls.encode_url + "/health", process=cls.process_encode)
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)
        cls.launch_lb()

    @classmethod
    def start_encode(cls):
        encode_args = [
            "--trust-remote-code",
            "--encoder-only",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--tp",
            "1",
            "--base-gpu-id",
            "0",
            "--port",
            cls.encode_port,
            *cls.model_args,
        ]
        cls.process_encode = popen_launch_server(
            cls.model,
            base_url=cls.encode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
        )

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--language-only",
            "--encoder-urls",
            cls.encode_url,
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
            "--port",
            cls.prefill_port,
            *cls.model_args,
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices_for(range(1, 2))
        cls.process_prefill = popen_launch_server(
            cls.model,
            base_url=cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--base-gpu-id",
            "2",
            "--port",
            cls.decode_port,
            *cls.model_args,
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices_for(range(2, 3))
        cls.process_decode = popen_launch_server(
            cls.model,
            base_url=cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            if cls.process_encode:
                try:
                    kill_process_tree(cls.process_encode.pid)
                except Exception as e:
                    print(f"Error killing encode process: {e}")

    def test_multi_image_chat_completion(self):
        client = openai.Client(api_key="sk-123456", base_url=f"{self.lb_url}/v1")
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
                            "text": "Describe each image separately.",
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=128,
        )

        self.assertEqual(response.choices[0].message.role, "assistant")
        text = response.choices[0].message.content
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

        text_lower = text.lower()
        self.assertTrue(
            any(
                word in text_lower
                for word in ("man", "person", "car", "vehicle", "suv", "iron")
            ),
            f"First image was not described correctly: {text}",
        )
        self.assertTrue(
            any(
                word in text_lower
                for word in ("logo", "sglang", "graphic", "stylized", "letter")
            ),
            f"Second image was not described correctly: {text}",
        )

        for name, process in (
            ("encoder", self.process_encode),
            ("prefill", self.process_prefill),
            ("decode", self.process_decode),
            ("router", self.process_lb),
        ):
            self.assertIsNone(
                process.poll(),
                f"{name} process exited with code {process.returncode}",
            )


class TestEPDDisaggregationMultiEncoders(MMMUMixin, PDDisaggregationServerBase):
    """
    Test EPD disaggregation with multiple encode servers for load balancing.
    Both encode servers run on GPU 0 (different ports) for testing load distribution.
    """

    # Qwen2.5-VL-3B-Instruct scores ~0.40 on the 50-sample MMMU subset.
    accuracy = 0.40
    mmmu_args = ["--limit", "50"]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST
        cls.base_url = cls.lb_url  # MMMUMixin reads this for OPENAI_API_BASE
        cls.encode_port1 = f"{int(cls.lb_port) + 300}"
        cls.encode_port2 = f"{int(cls.lb_port) + 301}"
        cls.encode_url1 = f"http://{cls.base_host}:{cls.encode_port1}"
        cls.encode_url2 = f"http://{cls.base_host}:{cls.encode_port2}"

        print(
            f"Setting up EPD (multiple encoders): encode1={cls.encode_port1}, "
            f"encode2={cls.encode_port2}, prefill={cls.prefill_port}, decode={cls.decode_port}"
        )

        # Start two encode servers on GPU 0/1
        encode1_thread = threading.Thread(
            target=cls.start_encode_server, args=(cls.encode_port1, 0)
        )
        encode2_thread = threading.Thread(
            target=cls.start_encode_server, args=(cls.encode_port2, 1)
        )
        encode1_thread.start()
        encode2_thread.start()
        encode1_thread.join()
        encode2_thread.join()

        prefill_thread = threading.Thread(target=cls.start_prefill)
        decode_thread = threading.Thread(target=cls.start_decode)
        prefill_thread.start()
        decode_thread.start()
        prefill_thread.join()
        decode_thread.join()

        cls.wait_server_ready(cls.encode_url1 + "/health", process=cls.process_encode1)
        cls.wait_server_ready(cls.encode_url2 + "/health", process=cls.process_encode2)
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        cls.launch_lb()

        # Set OpenAI API key and base URL environment variables. Needed for lmms-eval to work.
        cls.api_key = "sk-123456"
        os.environ["OPENAI_API_KEY"] = cls.api_key
        os.environ["OPENAI_API_BASE"] = f"{cls.lb_url}/v1"

    @classmethod
    def start_encode_server(cls, port, gpu_id):
        """Start an encode server on specific port and GPU"""
        encode_args = [
            "--trust-remote-code",
            "--encoder-only",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--tp",
            "1",
            "--port",
            port,
            "--enable-prefix-mm-cache",
        ]
        # Only set base-gpu-id if not using GPU 0
        if gpu_id != 0:
            encode_args.extend(["--base-gpu-id", str(gpu_id)])

        process = popen_launch_server(
            cls.model,
            base_url=f"http://{cls.base_host}:{port}",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
        )
        if port == cls.encode_port1:
            cls.process_encode1 = process
        else:
            cls.process_encode2 = process

    @classmethod
    def start_prefill(cls):
        """Start prefill server with multiple encode URLs"""
        prefill_args = [
            "--trust-remote-code",
            "--language-only",
            "--encoder-urls",
            cls.encode_url1,
            cls.encode_url2,
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--base-gpu-id",
            "2",
            "--port",
            cls.prefill_port,
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices_for(range(2, 3))
        cls.process_prefill = popen_launch_server(
            cls.model,
            base_url=cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        """Start decode server"""
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--base-gpu-id",
            "3",
            "--port",
            cls.decode_port,
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices_for(range(3, 4))
        cls.process_decode = popen_launch_server(
            cls.model,
            base_url=cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up all processes"""
        for process in [
            cls.process_lb,
            cls.process_decode,
            cls.process_prefill,
            cls.process_encode1,
            cls.process_encode2,
        ]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process: {e}")


if __name__ == "__main__":
    unittest.main()
