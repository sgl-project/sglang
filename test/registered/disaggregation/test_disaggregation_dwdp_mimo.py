import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_cuda_ci(est_time=900, suite="nightly-8-gpu-b200", nightly=True)

MIMO_V2_MODEL_PATH = "XiaomiMiMo/MiMo-V2.5"
GSM8K_BASELINE_ACCURACY = 0.93


class TestDisaggregationDWDPMiMo(PDDisaggregationServerBase):
    """PD disagg with DWDP prefill (4 GPUs) and DP-attention decode (4 GPUs)."""

    NUM_PREFILL_GPUS = 4
    NUM_DECODE_GPUS = 4

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = MIMO_V2_MODEL_PATH

        cls.start_prefill()
        cls.start_decode()

        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)

        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            str(cls.NUM_PREFILL_GPUS),
            "--dwdp-size",
            str(cls.NUM_PREFILL_GPUS),
            "--mm-enable-dp-encoder",
            "--attention-backend",
            "fa4",
            "--mem-fraction-static",
            "0.78",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
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
            str(cls.NUM_DECODE_GPUS),
            "--dp",
            str(cls.NUM_DECODE_GPUS),
            "--enable-dp-attention",
            "--moe-dense-tp-size",
            "1",
            "--ep-size",
            str(cls.NUM_DECODE_GPUS),
            "--attention-backend",
            "fa4",
            "--mem-fraction-static",
            "0.78",
            "--base-gpu-id",
            str(cls.NUM_PREFILL_GPUS),
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_gsm8k(self):
        metrics = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                api="chat",
                num_shots=5,
                num_examples=200,
                max_tokens=4096,
                num_threads=8,
                repeat=1,
                temperature=0.0,
                top_p=1.0,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
        )
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["score"], GSM8K_BASELINE_ACCURACY)


if __name__ == "__main__":
    unittest.main()
