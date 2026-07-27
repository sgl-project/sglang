import os
import threading
import unittest
from urllib.parse import urlparse

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.mmmu_vlm_kit import MMMUMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    popen_with_error_check,
)

NPU_COMMON_ARGS = [
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.5",
]

NPU_ENV = {
    **os.environ,
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    "SGLANG_MM_SKIP_COMPUTE_HASH": "True",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_BUFFSIZE": "200",
    "TRANSFORMERS_VERBOSITY": os.getenv("TRANSFORMERS_VERBOSITY", "error"),
}

os.environ["SGLANG_MM_SKIP_COMPUTE_HASH"] = "True"

DEFAULT_NPU_ENCODER_TRANSFER_BACKEND = "zmq_to_scheduler"

DEFAULT_NPU_TP_SIZE = "4"

register_npu_ci(est_time=400, suite="full-16-npu-a3", nightly=True)


class TestNpuEPDDisaggregationMultiEncoders(MMMUMixin, PDDisaggregationServerBase):
    """
    EPD test with multiple encode servers for load balancing.
    Uses 16 NPUs: 2 encoders (TP=4 each) + prefill (TP=4) + decode (TP=4).
    """

    model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
    encoder_transfer_backend = DEFAULT_NPU_ENCODER_TRANSFER_BACKEND
    tp_size = DEFAULT_NPU_TP_SIZE
    accuracy = 0.4
    mmmu_args = [
        "--limit",
        "50",
        "--batch_size",
        "4",
        "--gen_kwargs",
        "max_new_tokens=512",
        "--system_instruction",
        "You are a helpful assistant. For multiple choice questions, first think through the problem step by step to work out the solution, then provide your final answer as a single capital letter corresponding to the correct option (A, B, C, or D). Make sure your last line or final output is just the answer letter.",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        cls.rdma_devices = []
        parsed = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed.hostname
        bp = int(parsed.port)
        cls.lb_port = str(bp)
        cls.prefill_port = str(bp + 100)
        cls.decode_port = str(bp + 200)
        cls.bootstrap_port = str(bp + 500)
        cls.encode_port1 = str(bp + 300)
        cls.encode_port2 = str(bp + 301)
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        cls.base_url = cls.lb_url
        cls.encode_url1 = f"http://{cls.base_host}:{cls.encode_port1}"
        cls.encode_url2 = f"http://{cls.base_host}:{cls.encode_port2}"
        cls.api_key = "sk-123456"
        os.environ["OPENAI_API_KEY"] = cls.api_key
        os.environ["OPENAI_API_BASE"] = f"{cls.lb_url}/v1"
        print(
            f"Setting up NPU EPD (multiple encoders): "
            f"encode1={cls.encode_port1}, encode2={cls.encode_port2}, "
            f"prefill={cls.prefill_port}, decode={cls.decode_port}"
        )

        t1 = threading.Thread(target=cls._start_encode1, args=(cls.encode_port1, 0))
        t2 = threading.Thread(target=cls._start_encode2, args=(cls.encode_port2, 4))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        tp = threading.Thread(target=cls.start_prefill)
        td = threading.Thread(target=cls.start_decode)
        tp.start()
        td.start()
        tp.join()
        td.join()

        cls.wait_server_ready(cls.encode_url1 + "/health", process=cls.process_encode1)
        cls.wait_server_ready(cls.encode_url2 + "/health", process=cls.process_encode2)
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)
        cls.launch_lb()

    @classmethod
    def _start_encode1(cls, port, base_gpu_id):
        encode_args = [
            "--encoder-only",
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--tp-size",
            cls.tp_size,
            "--port",
            port,
            "--enable-prefix-mm-cache",
            "--base-gpu-id",
            str(base_gpu_id),
        ]
        encode_args += NPU_COMMON_ARGS
        cls.process_encode1 = popen_launch_server(
            cls.model,
            base_url=f"http://{cls.base_host}:{port}",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
            env=NPU_ENV,
        )

    @classmethod
    def _start_encode2(cls, port, base_gpu_id):
        encode_args = [
            "--encoder-only",
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--tp-size",
            cls.tp_size,
            "--port",
            port,
            "--enable-prefix-mm-cache",
            "--base-gpu-id",
            str(base_gpu_id),
        ]
        encode_args += NPU_COMMON_ARGS
        cls.process_encode2 = popen_launch_server(
            cls.model,
            base_url=f"http://{cls.base_host}:{port}",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
            env=NPU_ENV,
        )

    @classmethod
    def start_prefill(cls, encoder_urls=None):
        prefill_args = [
            "--language-only",
            "--encoder-urls",
            cls.encode_url1,
            cls.encode_url2,
            "--encoder-transfer-backend",
            cls.encoder_transfer_backend,
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            cls.tp_size,
            "--base-gpu-id",
            "8",
            "--port",
            cls.prefill_port,
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        prefill_args += NPU_COMMON_ARGS
        cls.process_prefill = popen_launch_server(
            cls.model,
            base_url=cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
            env=NPU_ENV,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp-size",
            cls.tp_size,
            "--base-gpu-id",
            "12",
            "--port",
            cls.decode_port,
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        decode_args += NPU_COMMON_ARGS
        cls.process_decode = popen_launch_server(
            cls.model,
            base_url=cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env=NPU_ENV,
        )

    @classmethod
    def launch_lb(cls):
        import shlex

        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--prefill",
            cls.prefill_url,
            cls.bootstrap_port,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        print("Starting load balancer:", shlex.join(lb_command))
        cls.process_lb = popen_with_error_check(lb_command)
        cls.wait_server_ready(cls.lb_url + "/health", process=cls.process_lb)

    @classmethod
    def tearDownClass(cls):
        for process in [
            getattr(cls, "process_lb", None),
            getattr(cls, "process_decode", None),
            getattr(cls, "process_prefill", None),
            getattr(cls, "process_encode1", None),
            getattr(cls, "process_encode2", None),
        ]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process: {e}")


if __name__ == "__main__":
    unittest.main()
