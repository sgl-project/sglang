import io
import os
import re
import subprocess
import threading
import time
import unittest

import grpc
import openai
import zmq
from grpc_health.v1 import health_pb2, health_pb2_grpc

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import get_zmq_socket_on_host
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.mmmu_vlm_kit import _run_lmms_eval_with_retry
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    is_in_ci,
    popen_launch_server,
)
from sglang.test.vlm_utils import (
    AUDIO_TRUMP_SPEECH_URL,
    IMAGE_MAN_IRONING_URL,
    IMAGE_SGL_LOGO_URL,
    VIDEO_JOBS_URL,
)

# Omni model for local testing; override via env var EPD_OMNI_MODEL
DEFAULT_OMNI_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
QWEN35_27B_MODEL = "Qwen/Qwen3.5-27B"


register_cuda_ci(est_time=96, suite="stage-c-test-4-gpu-h100")


@unittest.skipIf(
    is_in_ci(),
    "Omni model EPD test with image, video, and audio modalities, running locally only",
)
class TestEPDDisaggregationOmni(PDDisaggregationServerBase):
    """
    EPD disaggregation test for omni models (e.g. Qwen3-Omni). Covers image, video,
    and audio when server_type=http (encoder_transfer_backend: mooncake/zmq_to_scheduler/zmq_to_tokenizer).
    When server_type=grpc, only image is tested (gRPC encode is image-only).
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = os.environ.get("EPD_OMNI_MODEL", DEFAULT_OMNI_MODEL)
        cls.server_type = os.environ.get("EPD_ENCODE_SERVER_TYPE", "http")
        assert cls.server_type in (
            "grpc",
            "http",
        ), f"Invalid EPD_ENCODE_SERVER_TYPE: {cls.server_type}"
        cls.encoder_transfer_backend = os.environ.get(
            "EPD_ENCODER_TRANSFER_BACKEND", "zmq_to_scheduler"
        )
        assert cls.encoder_transfer_backend in (
            "mooncake",
            "zmq_to_scheduler",
            "zmq_to_tokenizer",
        ), f"Invalid EPD_ENCODER_TRANSFER_BACKEND: {cls.encoder_transfer_backend}"
        cls.enable_global_cache = (
            os.environ.get("MOONCAKE_MASTER") is not None
            or os.environ.get("MOONCAKE_CLIENT") is not None
        )
        if cls.server_type == "grpc":
            cls.encode_port = f"{int(cls.lb_port) + 305}"
            cls.encode_url = f"grpc://{cls.base_host}:{cls.encode_port}"
        else:
            cls.encode_port = f"{int(cls.lb_port) + 300}"
            cls.encode_url = f"http://{cls.base_host}:{cls.encode_port}"

        cls.image_man_ironing = IMAGE_MAN_IRONING_URL
        cls.image_sgl_logo = IMAGE_SGL_LOGO_URL
        cls.video_jobs = VIDEO_JOBS_URL
        cls.audio_trump = AUDIO_TRUMP_SPEECH_URL

        print(
            f"Setting up EPD Omni: model={cls.model}, encode={cls.encode_port}, "
            f"prefill={cls.prefill_port}, decode={cls.decode_port}, "
            f"server_type={cls.server_type}, backend={cls.encoder_transfer_backend}, "
            f"global_cache={cls.enable_global_cache}"
        )
        print(f"Data URLs: image={cls.image_man_ironing}, audio={cls.audio_trump}")

        cls.start_encode()
        prefill_thread = threading.Thread(target=cls.start_prefill)
        decode_thread = threading.Thread(target=cls.start_decode)
        prefill_thread.start()
        decode_thread.start()
        prefill_thread.join()
        decode_thread.join()

        if cls.server_type == "grpc":
            cls._wait_grpc_ready(cls.base_host, cls.encode_port, cls.process_encode)
        else:
            cls.wait_server_ready(
                cls.encode_url + "/health", process=cls.process_encode
            )
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        cls.launch_lb()

        cls.api_key = "sk-123456"
        os.environ["OPENAI_API_KEY"] = cls.api_key
        os.environ["OPENAI_API_BASE"] = f"{cls.lb_url}/v1"

    @classmethod
    def start_encode(cls):
        if cls.server_type == "grpc":
            cls.encode_stdout = io.StringIO()
            cls.encode_stderr = io.StringIO()
            cls.process_encode = subprocess.Popen(
                [
                    "python3",
                    "-m",
                    "sglang.launch_server",
                    "--model-path",
                    cls.model,
                    "--host",
                    cls.base_host,
                    "--port",
                    cls.encode_port,
                    "--trust-remote-code",
                    "--encoder-only",
                    "--grpc-mode",
                    "--encoder-transfer-backend",
                    "zmq_to_scheduler",
                    "--tp",
                    "1",
                ]
            )
        else:
            encode_args = [
                "--trust-remote-code",
                "--encoder-only",
                "--encoder-transfer-backend",
                cls.encoder_transfer_backend,
                "--tp",
                "1",
                "--port",
                cls.encode_port,
            ]
            if cls.enable_global_cache:
                encode_args.append("--enable-mm-global-cache")
            cls.encode_stdout = io.StringIO()
            cls.encode_stderr = io.StringIO()
            cls.process_encode = popen_launch_server(
                cls.model,
                base_url=cls.encode_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=encode_args,
                return_stdout_stderr=(cls.encode_stdout, cls.encode_stderr),
            )

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--language-only",
            "--encoder-urls",
            cls.encode_url,
            "--encoder-transfer-backend",
            (
                "zmq_to_scheduler"
                if cls.server_type == "grpc"
                else cls.encoder_transfer_backend
            ),
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
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        prefill_env = os.environ.copy()
        if cls.server_type == "grpc":
            prefill_env["SGLANG_ENCODER_MM_RECEIVER_MODE"] = "grpc"
        cls.process_prefill = popen_launch_server(
            cls.model,
            base_url=cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=prefill_env,
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
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_server(
            cls.model,
            base_url=cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    @classmethod
    def tearDownClass(cls):
        for process in [
            cls.process_lb,
            cls.process_decode,
            cls.process_prefill,
            cls.process_encode,
        ]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process: {e}")

    @staticmethod
    def _wait_grpc_ready(
        host, port, process, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    ):
        deadline = time.time() + timeout
        channel = grpc.insecure_channel(f"{host}:{port}")
        stub = health_pb2_grpc.HealthStub(channel)
        try:
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"gRPC encoder exited with code {process.returncode}"
                    )
                try:
                    response = stub.Check(
                        health_pb2.HealthCheckRequest(service=""), timeout=2
                    )
                    if response.status == health_pb2.HealthCheckResponse.SERVING:
                        return
                except grpc.RpcError:
                    pass
                time.sleep(1)
        finally:
            channel.close()
        raise RuntimeError(f"gRPC encoder not ready at {host}:{port} within {timeout}s")

    # ---- helpers ----

    def _client(self):
        return openai.Client(api_key=self.api_key, base_url=f"{self.lb_url}/v1")

    def _skip_if_grpc(self, msg="gRPC encode is image-only"):
        """Skip this test when encode server is gRPC (image-only)."""
        if self.server_type == "grpc":
            self.skipTest(msg)

    def _parse_cache_log(self):
        """Parse encode server logs and return list of (local_hits, global_hits, misses)
        tuples from '=== Multi-Level Cache Check ===' lines."""
        log = self.encode_stdout.getvalue() + self.encode_stderr.getvalue()
        pattern = re.compile(
            r"Multi-Level Cache Check.*?"
            r"Local Hits:\s*(\d+).*?"
            r"Global Hits:\s*(\d+).*?"
            r"Misses.*?:\s*(\d+)"
        )
        return [(int(m[1]), int(m[2]), int(m[3])) for m in pattern.finditer(log)]

    # ---- image ----
    def test_image(self):
        client = self._client()
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self.image_man_ironing},
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in a sentence.",
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=256,
        )
        text = response.choices[0].message.content
        print(f"[Omni EPD] Image response:\n{text}")
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        text_lower = text.lower()
        self.assertTrue(
            any(w in text_lower for w in ("man", "person", "driver")),
            f"Image response should mention a person: {text}",
        )
        self.assertTrue(
            any(w in text_lower for w in ("iron", "cloth", "hang", "holding")),
            f"Image response should mention ironing/clothes: {text}",
        )

    def test_image_cache_hit(self):
        """Send the same image twice; the second request should hit the global-mm-cache."""
        self._skip_if_grpc("gRPC encode is image-only; cache test uses HTTP path")
        if not self.enable_global_cache:
            self.skipTest("global-mm-cache not enabled (MOONCAKE_MASTER not set)")
        client = self._client()
        baseline = len(self._parse_cache_log())
        for i in range(2):
            response = client.chat.completions.create(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": self.image_sgl_logo},
                            },
                            {
                                "type": "text",
                                "text": "What is shown in this image?",
                            },
                        ],
                    },
                ],
                temperature=0,
                max_tokens=128,
            )
            text = response.choices[0].message.content
            print(f"[Omni EPD] Image cache-hit round {i}: {text}")
            self.assertIsNotNone(text)
            self.assertGreater(len(text), 0)
            time.sleep(1)

        entries = self._parse_cache_log()[baseline:]
        print(f"[Omni EPD] Image cache log entries: {entries}")
        self.assertGreaterEqual(
            len(entries), 2, "Expected at least 2 cache-check log entries"
        )
        local_hits, global_hits, _ = entries[-1]
        self.assertGreater(
            local_hits + global_hits,
            0,
            f"Second image request should have cache hits, got: {entries[-1]}",
        )

    # ---- video ----
    def test_video(self):
        self._skip_if_grpc()
        client = self._client()
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the video."},
                        {
                            "type": "video_url",
                            "video_url": {"url": self.video_jobs},
                        },
                    ],
                },
            ],
            max_tokens=8192,
            stream=False,
        )
        text = response.choices[0].message.content
        print(f"[Omni EPD] Video response:\n{text}")
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        text_lower = text.lower()
        self.assertTrue(
            any(
                w in text_lower
                for w in ("ipod", "device", "microphone", "smartphone", "phone")
            ),
            f"Video response should mention a device: {text}",
        )
        self.assertTrue(
            any(
                w in text_lower
                for w in (
                    "man",
                    "person",
                    "individual",
                    "speaker",
                    "presenter",
                    "steve",
                    "hand",
                    "hands",
                )
            ),
            f"Video response should mention a person: {text}",
        )
        self.assertTrue(
            any(
                w in text_lower
                for w in (
                    "present",
                    "presenting",
                    "examine",
                    "examining",
                    "display",
                    "displaying",
                    "hold",
                    "holding",
                    "gestur",
                    "speak",
                    "speaking",
                )
            ),
            f"Video response should mention an action: {text}",
        )

    def test_video_cache_hit(self):
        """Send the same video twice; the second request should hit the global-mm-cache."""
        self._skip_if_grpc()
        if not self.enable_global_cache:
            self.skipTest("global-mm-cache not enabled (MOONCAKE_MASTER not set)")
        client = self._client()
        baseline = len(self._parse_cache_log())
        for i in range(2):
            response = client.chat.completions.create(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the video."},
                            {
                                "type": "video_url",
                                "video_url": {"url": self.video_jobs},
                            },
                        ],
                    },
                ],
                max_tokens=256,
                stream=False,
            )
            text = response.choices[0].message.content
            print(f"[Omni EPD] Video cache-hit round {i}: {text}")
            self.assertIsNotNone(text)
            self.assertGreater(len(text), 0)
            time.sleep(1)

        entries = self._parse_cache_log()[baseline:]
        print(f"[Omni EPD] Video cache log entries: {entries}")
        self.assertGreaterEqual(
            len(entries), 2, "Expected at least 2 cache-check log entries"
        )
        local_hits, global_hits, _ = entries[-1]
        self.assertGreater(
            local_hits + global_hits,
            0,
            f"Second video request should have cache hits, got: {entries[-1]}",
        )

    # ---- audio ----

    def test_audio(self):
        self._skip_if_grpc()
        client = self._client()
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {"url": self.audio_trump},
                        },
                        {
                            "type": "text",
                            "text": "Listen to this audio and write down the audio transcription in English.",
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=256,
            stream=False,
        )
        text = response.choices[0].message.content
        print(f"[Omni EPD] Audio response:\n{text}")
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        text_lower = text.lower()
        for keyword in ("thank you", "leader"):
            self.assertIn(
                keyword,
                text_lower,
                f"Audio response should contain '{keyword}': {text}",
            )

    def test_audio_cache_hit(self):
        """Send the same audio twice; the second request should hit the global-mm-cache."""
        self._skip_if_grpc()
        if not self.enable_global_cache:
            self.skipTest("global-mm-cache not enabled (MOONCAKE_MASTER not set)")
        client = self._client()
        baseline = len(self._parse_cache_log())
        for i in range(2):
            response = client.chat.completions.create(
                model="default",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {"url": self.audio_trump},
                            },
                            {
                                "type": "text",
                                "text": "What is this audio about?",
                            },
                        ],
                    },
                ],
                temperature=0,
                max_tokens=128,
                stream=False,
            )
            text = response.choices[0].message.content
            print(f"[Omni EPD] Audio cache-hit round {i}: {text}")
            self.assertIsNotNone(text)
            self.assertGreater(len(text), 0)
            time.sleep(1)

        entries = self._parse_cache_log()[baseline:]
        print(f"[Omni EPD] Audio cache log entries: {entries}")
        self.assertGreaterEqual(
            len(entries), 2, "Expected at least 2 cache-check log entries"
        )
        local_hits, global_hits, _ = entries[-1]
        self.assertGreater(
            local_hits + global_hits,
            0,
            f"Second audio request should have cache hits, got: {entries[-1]}",
        )

    # ---- mixed modality ----

    def test_mixed_image_audio_video(self):
        """Image + audio + video in one request to test multi-modal routing."""
        self._skip_if_grpc()
        client = self._client()
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self.image_man_ironing},
                        },
                        {
                            "type": "audio_url",
                            "audio_url": {"url": self.audio_trump},
                        },
                        {
                            "type": "video_url",
                            "video_url": {"url": self.video_jobs},
                        },
                        {
                            "type": "text",
                            "text": (
                                "I have an image, an audio clip, and a video, which are not related at all. "
                                "Please: 1. Describe the image in a sentence, "
                                "2. Summarize the audio content briefly, "
                                "3. Describe what happens in the video."
                            ),
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=512,
            stream=False,
        )
        text = response.choices[0].message.content
        print(f"[Omni EPD] Mixed image+audio+video response:\n{text}")
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        text_lower = text.lower()
        self.assertTrue(
            any(w in text_lower for w in ("man", "person", "iron", "cloth")),
            f"Mixed response should describe the image: {text}",
        )


@unittest.skipIf(is_in_ci(), "Skipping in CI to reduce multi-GPU runtime")
class TestEPDDisaggregationOneEncoder(PDDisaggregationServerBase):
    """Test EPD disaggregation with single encode server"""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST
        cls.encode_port = f"{int(cls.lb_port) + 300}"
        cls.encode_url = f"http://{cls.base_host}:{cls.encode_port}"

        print(
            f"Setting up EPD (one encoder): encode={cls.encode_port}, "
            f"prefill={cls.prefill_port}, decode={cls.decode_port}"
        )

        # Start servers in order: encode -> prefill/decode
        cls.start_encode()
        prefill_thread = threading.Thread(target=cls.start_prefill)
        decode_thread = threading.Thread(target=cls.start_decode)
        prefill_thread.start()
        decode_thread.start()
        prefill_thread.join()
        decode_thread.join()

        # Wait for all servers to be ready
        cls.wait_server_ready(cls.encode_url + "/health", process=cls.process_encode)
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        cls.launch_lb()

        # Set OpenAI API key and base URL environment variables. Needed for lmms-eval to work.
        cls.api_key = "sk-123456"
        os.environ["OPENAI_API_KEY"] = cls.api_key
        os.environ["OPENAI_API_BASE"] = f"{cls.lb_url}/v1"

    @classmethod
    def start_encode(cls):
        """Start encode server for multimodal processing"""
        encode_args = [
            "--trust-remote-code",
            "--encoder-only",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--tp",
            "1",
            "--port",
            cls.encode_port,
            "--enable-prefix-mm-cache",
        ]
        cls.process_encode = popen_launch_server(
            cls.model,
            base_url=cls.encode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
        )

    @classmethod
    def start_prefill(cls):
        """Start prefill server with language model only"""
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
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
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
            "2",
            "--port",
            cls.decode_port,
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
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
            cls.process_encode,
        ]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process: {e}")

    def run_mmmu_eval(self, model_version: str, output_path: str, limit: str = "50"):
        """
        Evaluate a VLM on the MMMU validation set with lmms-eval.
        Reference: test_vlm_models.py

        Args:
            model_version: Model version/checkpoint to evaluate
            output_path: Path to save evaluation results
            limit: Number of samples to evaluate (default: "50" for CI time constraints)
        """
        model = "openai_compatible"
        tp = 1
        tasks = "mmmu_val"
        batch_size = 32
        log_suffix = "openai_compatible"
        os.makedirs(output_path, exist_ok=True)

        model_args = f'model_version="{model_version}",tp={tp}'

        cmd = [
            "python3",
            "-m",
            "lmms_eval",
            "--model",
            model,
            "--model_args",
            model_args,
            "--tasks",
            tasks,
            "--batch_size",
            str(batch_size),
            "--log_samples",
            "--log_samples_suffix",
            log_suffix,
            "--output_path",
            str(output_path),
            "--limit",
            limit,
        ]

        _run_lmms_eval_with_retry(cmd, timeout=3600)

    def test_mmmu(self):
        """Test MMMU evaluation with EPD disaggregation"""
        import glob
        import json

        output_path = "./logs/epd_one_encoder_mmmu"
        self.run_mmmu_eval(self.model, output_path)

        # Get the result file
        result_files = glob.glob(f"{output_path}/**/*.json", recursive=True)
        if not result_files:
            result_files = glob.glob(f"{output_path}/*.json")

        if not result_files:
            self.fail(f"No JSON result files found in {output_path}")

        result_file_path = result_files[0]
        with open(result_file_path, "r") as f:
            result = json.load(f)
            print(f"MMMU result: {result}")

        mmmu_accuracy = result["results"]["mmmu_val"]["mmmu_acc,none"]
        print(f"MMMU accuracy: {mmmu_accuracy:.4f}")

        # for qwen2.5-vl-3b-instruct, the accuracy is 0.40
        self.assertGreater(mmmu_accuracy, 0.40)


@unittest.skipIf(
    is_in_ci(),
    "Qwen3.5 EPD image/video test runs locally only",
)
class TestEPDDisaggregationQwen35(PDDisaggregationServerBase):
    """EPD disaggregation test for Qwen3.5 image and video requests."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.process_encode = None
        cls.model = QWEN35_27B_MODEL
        cls.encode_port = f"{int(cls.lb_port) + 300}"
        cls.encode_url = f"http://{cls.base_host}:{cls.encode_port}"
        cls.language_url = cls.prefill_url
        cls.image_man_ironing = IMAGE_MAN_IRONING_URL
        cls.video_jobs = VIDEO_JOBS_URL

        print(
            f"Setting up Qwen3.5 encoder disaggregation: model={cls.model}, "
            f"encode={cls.encode_port}, language={cls.prefill_port}"
        )

        cls.start_encode()
        cls.start_prefill()

        cls.wait_server_ready(cls.encode_url + "/health", process=cls.process_encode)
        cls.wait_server_ready(cls.language_url + "/health", process=cls.process_prefill)

        cls.api_key = "sk-123456"

    @classmethod
    def start_encode(cls):
        encode_args = [
            "--trust-remote-code",
            "--encoder-only",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--tp",
            "1",
            "--port",
            cls.encode_port,
            "--reasoning-parser",
            "qwen3",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true,"num_threads": 64}',
        ]
        cls.process_encode = popen_launch_server(
            cls.model,
            base_url=cls.encode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
        )

    @classmethod
    def start_prefill(cls):
        language_args = [
            "--trust-remote-code",
            "--language-only",
            "--encoder-urls",
            cls.encode_url,
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
            "--port",
            cls.prefill_port,
            "--reasoning-parser",
            "qwen3",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true,"num_threads": 64}',
        ]
        cls.process_prefill = popen_launch_server(
            cls.model,
            base_url=cls.language_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=language_args,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process_lb:
            kill_process_tree(cls.process_lb.pid)
        if cls.process_decode:
            kill_process_tree(cls.process_decode.pid)
        if cls.process_prefill:
            kill_process_tree(cls.process_prefill.pid)
        if cls.process_encode:
            kill_process_tree(cls.process_encode.pid)

    def _client(self):
        return openai.Client(api_key=self.api_key, base_url=f"{self.language_url}/v1")

    def test_image(self):
        client = self._client()
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": self.image_man_ironing},
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in a sentence.",
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=256,
            extra_body={"reasoning_effort": "none"},
        )
        text = response.choices[0].message.content
        print(f"[Qwen3.5 EPD] Image response:\n{text}")
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        text_lower = text.lower()
        self.assertTrue(
            any(w in text_lower for w in ("man", "person", "driver")),
            f"Image response should mention a person: {text}",
        )
        self.assertTrue(
            any(w in text_lower for w in ("iron", "cloth", "hang", "holding")),
            f"Image response should mention ironing/clothes: {text}",
        )

    def test_video(self):
        client = self._client()
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe the video."},
                        {
                            "type": "video_url",
                            "video_url": {"url": self.video_jobs},
                        },
                    ],
                },
            ],
            max_tokens=1024,
            stream=False,
        )
        text = response.choices[0].message.content
        print(f"[Qwen3.5 EPD] Video response:\n{text}")
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        text_lower = text.lower()
        self.assertTrue(
            any(
                w in text_lower
                for w in ("ipod", "device", "microphone", "smartphone", "phone")
            ),
            f"Video response should mention a device: {text}",
        )
        self.assertTrue(
            any(
                w in text_lower
                for w in (
                    "man",
                    "person",
                    "individual",
                    "speaker",
                    "presenter",
                    "steve",
                    "hand",
                    "hands",
                )
            ),
            f"Video response should mention a person: {text}",
        )


class TestEPDDisaggregationMultiEncoders(PDDisaggregationServerBase):
    """
    Test EPD disaggregation with multiple encode servers for load balancing.
    Both encode servers run on GPU 0 (different ports) for testing load distribution.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST
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
        prefill_args += cls.transfer_backend + cls.rdma_devices
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
        decode_args += cls.transfer_backend + cls.rdma_devices
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

    def run_mmmu_eval(self, model_version: str, output_path: str, limit: str = "50"):
        """
        Evaluate a VLM on the MMMU validation set with lmms-eval.
        Reference: test_vlm_models.py

        Args:
            model_version: Model version/checkpoint to evaluate
            output_path: Path to save evaluation results
            limit: Number of samples to evaluate (default: "50" for CI time constraints)
        """
        model = "openai_compatible"
        tp = 1
        tasks = "mmmu_val"
        batch_size = 32
        log_suffix = "openai_compatible"
        os.makedirs(output_path, exist_ok=True)

        model_args = f'model_version="{model_version}",tp={tp}'

        cmd = [
            "python3",
            "-m",
            "lmms_eval",
            "--model",
            model,
            "--model_args",
            model_args,
            "--tasks",
            tasks,
            "--batch_size",
            str(batch_size),
            "--log_samples",
            "--log_samples_suffix",
            log_suffix,
            "--output_path",
            str(output_path),
            "--limit",
            limit,
        ]

        _run_lmms_eval_with_retry(cmd, timeout=3600)

    def test_mmmu(self):
        """Test MMMU evaluation with EPD disaggregation (multiple encoders)"""
        import glob
        import json

        output_path = "./logs/epd_multi_encoder_mmmu"
        self.run_mmmu_eval(self.model, output_path)

        # Get the result file
        result_files = glob.glob(f"{output_path}/**/*.json", recursive=True)
        if not result_files:
            result_files = glob.glob(f"{output_path}/*.json")

        if not result_files:
            self.fail(f"No JSON result files found in {output_path}")

        result_file_path = result_files[0]
        with open(result_file_path, "r") as f:
            result = json.load(f)
            print(f"MMMU result (multi encoder): {result}")

        mmmu_accuracy = result["results"]["mmmu_val"]["mmmu_acc,none"]
        print(f"MMMU accuracy (multi encoder): {mmmu_accuracy:.4f}")
        # for qwen2.5-vl-3b-instruct, the accuracy is 0.40
        self.assertGreater(mmmu_accuracy, 0.40)


@unittest.skipIf(is_in_ci(), "Skipping in CI to reduce multi-GPU runtime")
class TestEPDDisaggregationGrpcEncoderMMMU(PDDisaggregationServerBase):
    """Test MMMU evaluation with gRPC encoder in EPD mode."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST
        cls.encode_port = f"{int(cls.lb_port) + 304}"
        cls.encode_url = f"grpc://{cls.base_host}:{cls.encode_port}"

        print(
            f"Setting up gRPC EPD (one encoder): encode={cls.encode_port}, "
            f"prefill={cls.prefill_port}, decode={cls.decode_port}"
        )

        cls.start_encode()
        prefill_thread = threading.Thread(target=cls.start_prefill)
        decode_thread = threading.Thread(target=cls.start_decode)
        prefill_thread.start()
        decode_thread.start()
        prefill_thread.join()
        decode_thread.join()

        cls.wait_grpc_ready(cls.base_host, cls.encode_port, cls.process_encode)
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

        cls.launch_lb()

        cls.api_key = "sk-123456"
        os.environ["OPENAI_API_KEY"] = cls.api_key
        os.environ["OPENAI_API_BASE"] = f"{cls.lb_url}/v1"

    @classmethod
    def start_encode(cls):
        encode_command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            cls.model,
            "--host",
            cls.base_host,
            "--port",
            cls.encode_port,
            "--trust-remote-code",
            "--encoder-only",
            "--grpc-mode",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--tp",
            "1",
            "--base-gpu-id",
            "0",
            "--enable-prefix-mm-cache",
        ]
        cls.process_encode = subprocess.Popen(encode_command)

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
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        prefill_env = os.environ.copy()
        prefill_env["SGLANG_ENCODER_MM_RECEIVER_MODE"] = "grpc"
        cls.process_prefill = popen_launch_server(
            cls.model,
            base_url=cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=prefill_env,
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
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_server(
            cls.model,
            base_url=cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    @staticmethod
    def wait_grpc_ready(host, port, process, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH):
        deadline = time.time() + timeout
        channel = grpc.insecure_channel(f"{host}:{port}")
        stub = health_pb2_grpc.HealthStub(channel)
        try:
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"gRPC encoder server exited with code {process.returncode}"
                    )
                try:
                    response = stub.Check(
                        health_pb2.HealthCheckRequest(service=""), timeout=2
                    )
                    if response.status == health_pb2.HealthCheckResponse.SERVING:
                        return
                except grpc.RpcError:
                    pass
                time.sleep(1)
        finally:
            channel.close()

        raise RuntimeError(
            f"gRPC encoder server not ready at {host}:{port} within {timeout}s"
        )

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("SGLANG_ENCODER_MM_RECEIVER_MODE", None)
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("OPENAI_API_BASE", None)
        for process in [
            cls.process_lb,
            cls.process_decode,
            cls.process_prefill,
            cls.process_encode,
        ]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process: {e}")

    def run_mmmu_eval(self, model_version: str, output_path: str, limit: str = "50"):
        model = "openai_compatible"
        tp = 1
        tasks = "mmmu_val"
        batch_size = 32
        log_suffix = "openai_compatible"
        os.makedirs(output_path, exist_ok=True)

        model_args = f'model_version="{model_version}",tp={tp}'

        cmd = [
            "python3",
            "-m",
            "lmms_eval",
            "--model",
            model,
            "--model_args",
            model_args,
            "--tasks",
            tasks,
            "--batch_size",
            str(batch_size),
            "--log_samples",
            "--log_samples_suffix",
            log_suffix,
            "--output_path",
            str(output_path),
            "--limit",
            limit,
        ]

        _run_lmms_eval_with_retry(cmd, timeout=3600)

    def test_mmmu(self):
        import glob
        import json

        output_path = "./logs/epd_grpc_encoder_mmmu"
        self.run_mmmu_eval(self.model, output_path)

        result_files = glob.glob(f"{output_path}/**/*.json", recursive=True)
        if not result_files:
            result_files = glob.glob(f"{output_path}/*.json")

        if not result_files:
            self.fail(f"No JSON result files found in {output_path}")

        result_file_path = result_files[0]
        with open(result_file_path, "r") as f:
            result = json.load(f)
            print(f"MMMU result (grpc encoder): {result}")

        mmmu_accuracy = result["results"]["mmmu_val"]["mmmu_acc,none"]
        print(f"MMMU accuracy (grpc encoder): {mmmu_accuracy:.4f}")
        # for qwen2.5-vl-3b-instruct, the accuracy is 0.40
        self.assertGreater(mmmu_accuracy, 0.40)


@unittest.skipIf(is_in_ci(), "Skipping in CI to reduce multi-GPU runtime")
class TestEPDDisaggregationGrpcEncoderOnly(PDDisaggregationServerBase):
    """Test gRPC encoder server integration with zmq_to_scheduler transfers."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ["SGLANG_ENCODER_MM_RECEIVER_MODE"] = "grpc"
        cls.model = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST
        cls.encode_port = f"{int(cls.lb_port) + 302}"

        print(f"Setting up gRPC EPD encoder: encode={cls.encode_port}")

        cls.start_encode()
        cls.wait_grpc_ready(cls.base_host, cls.encode_port, cls.process_encode)

    @classmethod
    def start_encode(cls):
        encode_command = [
            "python3",
            "-m",
            "sglang.launch_server",
            "--model-path",
            cls.model,
            "--host",
            cls.base_host,
            "--port",
            cls.encode_port,
            "--trust-remote-code",
            "--encoder-only",
            "--grpc-mode",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--tp",
            "1",
            "--base-gpu-id",
            "0",
            "--enable-prefix-mm-cache",
        ]
        cls.process_encode = subprocess.Popen(encode_command)

    @staticmethod
    def wait_grpc_ready(host, port, process, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH):
        deadline = time.time() + timeout
        channel = grpc.insecure_channel(f"{host}:{port}")
        stub = health_pb2_grpc.HealthStub(channel)
        try:
            while time.time() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"gRPC encoder server exited with code {process.returncode}"
                    )
                try:
                    response = stub.Check(
                        health_pb2.HealthCheckRequest(service=""), timeout=2
                    )
                    if response.status == health_pb2.HealthCheckResponse.SERVING:
                        return
                except grpc.RpcError:
                    pass
                time.sleep(1)
        finally:
            channel.close()

        raise RuntimeError(
            f"gRPC encoder server not ready at {host}:{port} within {timeout}s"
        )

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("SGLANG_ENCODER_MM_RECEIVER_MODE", None)
        if cls.process_encode:
            try:
                kill_process_tree(cls.process_encode.pid)
            except Exception as e:
                print(f"Error killing process: {e}")
        super().tearDownClass()

    def test_grpc_encoder_zmq_to_scheduler(self):
        from smg_grpc_proto import sglang_encoder_pb2, sglang_encoder_pb2_grpc

        context = zmq.Context()
        recv_port, recv_socket = get_zmq_socket_on_host(
            context, zmq.PULL, host=self.base_host
        )
        channel = grpc.insecure_channel(f"{self.base_host}:{self.encode_port}")
        stub = sglang_encoder_pb2_grpc.SglangEncoderStub(channel)
        req_id = f"grpc-epd-{int(time.time() * 1000)}"
        image_path = os.path.abspath("examples/assets/example_image.png")

        try:
            stub.SchedulerReceiveUrl(
                sglang_encoder_pb2.SchedulerReceiveUrlRequest(
                    req_id=req_id,
                    receive_url=f"{self.base_host}:{recv_port}",
                    receive_count=1,
                ),
                timeout=60,
            )
            stub.Encode(
                sglang_encoder_pb2.EncodeRequest(
                    mm_items=[image_path],
                    req_id=req_id,
                    num_parts=1,
                    part_idx=0,
                ),
                timeout=300,
            )

            poller = zmq.Poller()
            poller.register(recv_socket, zmq.POLLIN)
            socks = dict(poller.poll(60000))
            self.assertIn(
                recv_socket,
                socks,
                "No embedding payload received from gRPC encoder server",
            )
            parts = recv_socket.recv_multipart()
            self.assertTrue(parts, "Empty embedding payload from gRPC encoder server")
        finally:
            recv_socket.close()
            context.term()
            channel.close()


if __name__ == "__main__":
    unittest.main()
