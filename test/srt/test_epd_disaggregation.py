import os
import subprocess
import threading
import time
import unittest

import grpc
import zmq
from grpc_health.v1 import health_pb2, health_pb2_grpc

from sglang.srt.grpc import sglang_encoder_pb2, sglang_encoder_pb2_grpc
from sglang.srt.utils import get_zmq_socket_on_host, kill_process_tree
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
        cls.wait_server_ready(cls.encode_url + "/health")
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

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

        model_args = f'model_version="{model_version}",' f"tp={tp}"

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

        cls.wait_server_ready(cls.encode_url1 + "/health")
        cls.wait_server_ready(cls.encode_url2 + "/health")
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")

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

        model_args = f'model_version="{model_version}",' f"tp={tp}"

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
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
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

        model_args = f'model_version="{model_version}",' f"tp={tp}"

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
class TestEPDDisaggregationGrpcEncoder(PDDisaggregationServerBase):
    """Test gRPC encoder server integration with zmq_to_scheduler transfers."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST
        cls.encode_port = f"{int(cls.lb_port) + 302}"

        print(
            f"Setting up gRPC EPD encoder: encode={cls.encode_port}"
        )

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
        if cls.process_encode:
            try:
                kill_process_tree(cls.process_encode.pid)
            except Exception as e:
                print(f"Error killing process: {e}")
        super().tearDownClass()

    def test_grpc_encoder_zmq_to_scheduler(self):
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
