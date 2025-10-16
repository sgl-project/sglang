import os
import time
import warnings
from urllib.parse import urlparse

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_with_error_check,
)


class TestDisaggregationBase(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed_url.hostname
        base_port = str(parsed_url.port)
        cls.lb_port = base_port
        cls.prefill_port = f"{int(base_port) + 100}"
        cls.decode_port = f"{int(base_port) + 200}"
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        print(f"{cls.base_host=} {cls.lb_port=} {cls.prefill_port=} {cls.decode_port=}")
        cls.process_lb, cls.process_decode, cls.process_prefill = None, None, None

        # config transfer backend and rdma devices
        if is_in_ci():
            cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
            cls.rdma_devices = ["--disaggregation-ib-device", get_rdma_devices_args()]
        else:
            cls.transfer_backend = [
                "--disaggregation-transfer-backend",
                envs.SGLANG_TEST_PD_DISAGG_BACKEND.get(),
            ]
            cls.rdma_devices = [
                "--disaggregation-ib-device",
                envs.SGLANG_TEST_PD_DISAGG_DEVICES.get(),
            ]
            if cls.rdma_devices[1] is None:
                cls.rdma_devices = []
                msg = "No RDMA devices specified for disaggregation test, using default settings."
                warnings.warn(msg)

    @classmethod
    def launch_lb(cls):
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",  # FIXME: remove this
            "--prefill",
            cls.prefill_url,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        print("Starting load balancer:", " ".join(lb_command))
        cls.process_lb = popen_with_error_check(lb_command)
        cls.wait_server_ready(cls.lb_url + "/health")

    @classmethod
    def wait_server_ready(cls, url, timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH):
        start_time = time.perf_counter()
        while True:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    print(f"Server {url} is ready")
                    return
            except Exception:
                pass

            if time.perf_counter() - start_time > timeout:
                raise RuntimeError(f"Server {url} failed to start in {timeout}s")
            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        for process in [cls.process_lb, cls.process_decode, cls.process_prefill]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")

        # wait for 5 seconds
        time.sleep(5)


def get_rdma_devices_args():
    # 1. Get visible GPU indices
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    if not cuda_visible_devices:
        warnings.warn("CUDA_VISIBLE_DEVICES is not set. Using default RDMA devices.")
        return "mlx5_roce0,mlx5_roce4"

    try:
        # Convert to list of integers (handling possible spaces and empty strings)
        gpu_indices = [
            int(idx.strip()) for idx in cuda_visible_devices.split(",") if idx.strip()
        ]
        if not gpu_indices or len(gpu_indices) > 4:
            return "mlx5_roce0,mlx5_roce4"
    except ValueError:
        warnings.warn(f"Invalid CUDA_VISIBLE_DEVICES format: {cuda_visible_devices}")
        return "mlx5_roce0,mlx5_roce4"

    # 2. Calculate base RDMA index group (each group of 4 GPUs uses consecutive devices)
    base_rdma_group = min(gpu_indices) // 4 * 4

    # 3. Generate RDMA device names
    rdma_devices = []
    for gpu_idx in gpu_indices:
        # Validate GPU index within expected range
        if gpu_idx < base_rdma_group or gpu_idx >= base_rdma_group + 4:
            warnings.warn(
                f"GPU index {gpu_idx} is outside expected group {base_rdma_group}-{base_rdma_group+3}"
            )
            continue

        # Map GPU index to RDMA device index
        rdma_index = base_rdma_group // 4 * 4 + (gpu_idx % 4)
        rdma_devices.append(f"mlx5_roce{rdma_index}")

    if not rdma_devices:
        return "mlx5_roce0,mlx5_roce4"

    return ",".join(rdma_devices)
