"""MI35x Qwen3.5 radix-cache + speculative-decoding (mamba extra_buffer) regression test (8-GPU).

Validates `--mamba-scheduler-strategy extra_buffer` running radix prefix cache **and** NEXTN speculative decoding
*together* on ROCm/MI35x for the hybrid Gated-DeltaNet + MoE model Qwen3.5-397B-A17B — the AMD counterpart of the CUDA
gate `test/registered/4-gpu-models/test_qwen35_fp4_mtp_v2.py`, added alongside the ROCm enablement of `extra_buffer`
(`server_args._handle_mamba_radix_cache`: `is_cuda() or is_musa() or is_npu() or is_hip()`).

What it checks:
  - The radix+spec+extra_buffer config serves without crashing. NOTE the explicit cuda-graph / concurrency / prefill
    caps below are required: with server defaults (cuda_graph_max_bs=512, no caps) the GDN speculative path can SIGABRT
    under eval load.
  - **Speculative decoding is genuinely active with radix on**: `avg_spec_accept_length > 2.0` (measures ~3.0). This is
    the primary, robust signal that the feature this PR enables actually works.
  - **GSM8K accuracy >= 0.93** (5-shot, greedy, thinking disabled — the 5-shot exemplars are the CoT scaffold). The eval
    runs at **num_threads=1 (sequential)**, which is the decisive choice for a non-flaky gate: this ROCm MoE/aiter stack
    is non-deterministic at the logit level (batch-composition-dependent reductions; `--enable-deterministic-inference`
    does not yet cover this path). A controlled sweep on one fixed 397B-FP8 server, varying ONLY num_threads, showed:
    threads=1 -> 0.970/0.970 (stable, bit-identical across runs); threads=8 -> 0.970; threads=32 -> 0.89/0.88/0.93
    (the high-concurrency batch noise, an ~8-pt swing). Sequential eval removes that noise; radix prefix reuse is still
    exercised because the 5-shot prefix is shared across all examples (high cache-hit). The 0.93 gate sits ~3 pts under
    the stable 0.96-0.97 floor (CUDA gate is 0.95; relaxed for the non-deterministic AMD path). (Thinking-ON + temp 0.6
    was noisier — the model over-reasons ~5.8k tok/q and truncates; thinking-off + few-shot CoT is higher and stable.
    Cache-fidelity is also validated separately: KL cache-hit-vs-fresh, radix_cache token-match.)

Locally validated on Qwen3.5-397B-A17B-FP8, TP2/EP2, MI350x (gfx950): GSM8K **0.96-0.97** sequential (5-shot greedy,
five stable runs at threads=1: run_eval 0.97/0.97/0.97 + per-question inspect 0.96/0.98/0.97), ~93-99% radix cache hit,
NEXTN accept ~3.1, no crash with the caps below.

Registry: nightly-amd-accuracy-8-gpu-mi35x-qwen35 suite (joins the existing Qwen3.5 MI35x accuracy lane).
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=3600, suite="nightly-amd-accuracy-8-gpu-mi35x-qwen35", nightly=True
)

QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B"
SERVER_LAUNCH_TIMEOUT = 3600
TP_SIZE = 8
GSM8K_THRESHOLD = 0.93  # AMD-relaxed from CUDA's 0.95; sequential greedy is stable at 0.96-0.97 here -> ~3-pt margin
SPEC_ACCEPT_THRESHOLD = 2.0  # primary check; NEXTN 3/1/4 measures ~3.0


class TestQwen35ExtraBufferMTPMI35x(CustomTestCase):
    """radix prefix cache + NEXTN speculative decoding via mamba extra_buffer on AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        envs.SGLANG_ENABLE_SPEC_V2.set(True)
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--tp",
                str(TP_SIZE),
                "--trust-remote-code",
                "--attention-backend",
                "triton",
                "--mamba-scheduler-strategy",
                "extra_buffer",
                "--page-size",
                "64",
                "--speculative-algorithm",
                "NEXTN",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--reasoning-parser",
                "qwen3",
                # explicit caps: required so the GDN spec path does not SIGABRT under load (defaults bs=512, no caps)
                "--cuda-graph-max-bs",
                "64",
                "--max-running-requests",
                "64",
                "--max-queued-requests",
                "256",
                "--chunked-prefill-size",
                "32768",
                "--max-prefill-tokens",
                "32768",
                "--mem-fraction-static",
                "0.85",
                "--watchdog-timeout",
                "1800",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true}',
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        envs.SGLANG_ENABLE_SPEC_V2.set(False)
        kill_process_tree(cls.process.pid)

    def test_gsm8k_radix_spec(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            model=self.model,
            eval_name="gsm8k",
            num_shots=5,
            num_examples=200,
            max_tokens=4096,  # headroom for the few long-CoT questions (2048 truncates ~2/run -> spurious fails)
            num_threads=1,  # SEQUENTIAL: the only setting that is stable on this non-deterministic ROCm MoE stack
            # (threads=1->0.97/0.97, threads=8->0.97, threads=32->0.89/0.88/0.93 on a fixed server; see module docstring)
            repeat=1,
            temperature=0.0,  # greedy: reproducible, avoids over-reasoning truncation
            top_p=1.0,
            top_k=None,
            chat_template_kwargs={
                "enable_thinking": False
            },  # 5-shot supplies CoT; concise answers, no truncation
            base_url=self.base_url,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        if is_in_ci():
            write_github_step_summary(
                "### test_gsm8k_radix_spec (qwen3.5 extra_buffer: radix + NEXTN)\n"
                f'{metrics["score"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )

        # primary: speculative decoding is genuinely active with radix on
        self.assertGreater(avg_spec_accept_length, SPEC_ACCEPT_THRESHOLD)
        # GSM8K accuracy gate (AMD-relaxed from CUDA's 0.95; sequential greedy is stable at 0.96-0.97 -> ~3-pt margin)
        self.assertGreaterEqual(metrics["score"], GSM8K_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
