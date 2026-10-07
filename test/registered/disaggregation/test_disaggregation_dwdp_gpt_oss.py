import unittest
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.sgl_eval_utils import run_sgl_eval

register_cuda_ci(est_time=170, stage="extra-b", runner_config="4-gpu-b200")

GPT_OSS_MODEL_PATH = "openai/gpt-oss-120b"
GSM8K_BASELINE_ACCURACY = 0.88


class TestDisaggregationDWDPGptOss(PDDisaggregationServerBase):
    """PD disagg with DWDP prefill (2 GPUs) and DP-attention decode (2 GPUs)."""

    NUM_PREFILL_GPUS = 2
    NUM_DECODE_GPUS = 2

    # Drive the base fixture's launchers rather than reimplementing them, so
    # both sides get the pinned --nccl-port. Deriving it from get_free_port()
    # on each side races onto the same port and dies at init_process_group
    # with EADDRINUSE.
    prefill_tp_size = NUM_PREFILL_GPUS
    decode_tp_size = NUM_DECODE_GPUS
    decode_base_gpu_id = NUM_PREFILL_GPUS

    extra_prefill_args = [
        "--dwdp-size",
        str(NUM_PREFILL_GPUS),
        "--disable-flashinfer-autotune",
        "--mem-fraction-static",
        "0.85",
    ]
    extra_decode_args = [
        "--dp",
        str(NUM_DECODE_GPUS),
        "--enable-dp-attention",
        "--disable-flashinfer-autotune",
        "--mem-fraction-static",
        "0.85",
    ]

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = GPT_OSS_MODEL_PATH
        cls.launch_all()

    def test_gsm8k(self):
        metrics = run_sgl_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                num_examples=100,
                max_tokens=4096,
                num_threads=8,
                repeat=1,
                temperature=0.0,
                top_p=1.0,
                host="127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
        )
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["score"], GSM8K_BASELINE_ACCURACY)


if __name__ == "__main__":
    unittest.main()
