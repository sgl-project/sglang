"""MI35x DeepSeek-V4-Pro-DSpark unified_kv GSM8K accuracy test (8-GPU).

Runs the production AMD DSpark static configuration with the HIP dsv4 backend and
SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton. The test uses the full GSM8K set
to catch regressions in unified-KV target-hidden injection, verify metadata, and
DSpark acceptance.

Registry: nightly-amd-8-gpu-mi35x-deepseek-v4-pro-dspark suite
"""

import os
import unittest
from types import SimpleNamespace

import requests
import torch

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime
from sglang.srt.speculative.dspark_components.kernels import dspark_verify_window
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=7200, suite="nightly-amd-8-gpu-mi35x-deepseek-v4-pro-dspark", nightly=True
)

DEEPSEEK_V4_DSPARK_MODEL_PATH = os.environ.get(
    "DEEPSEEK_V4_DSPARK_MODEL_PATH", "deepseek-ai/DeepSeek-V4-Pro-DSpark"
)
SERVER_LAUNCH_TIMEOUT = 5400
GSM8K_ACCURACY_THRESHOLD = 0.92
AVG_SPEC_ACCEPT_LENGTH_THRESHOLD = 3.0
DEVICE = torch.device("cuda")


class TestDSparkUnifiedKVKernelsAMD(CustomTestCase):
    def test_build_unified_commit_inject_layout(self):
        stride, ring_stride = 7, 128
        req_pool_indices = torch.tensor([3, 0, 5, 1], device=DEVICE, dtype=torch.int32)
        prefix_lens = torch.tensor(
            [10, 127, 128, 255], device=DEVICE, dtype=torch.int64
        )
        block_pos_offsets = torch.arange(stride, device=DEVICE, dtype=torch.int64)
        commit_lens = torch.tensor([0, 3, stride, 5], device=DEVICE, dtype=torch.int32)

        got = dspark_verify_window.build_unified_commit_inject_layout(
            req_pool_indices=req_pool_indices,
            prefix_lens=prefix_lens,
            block_pos_offsets=block_pos_offsets,
            commit_lens=commit_lens,
            stride=stride,
            ring_stride=ring_stride,
        )

        positions_2d = prefix_lens.view(-1, 1) + block_pos_offsets[:stride]
        loc_2d = req_pool_indices.to(torch.int64).view(-1, 1) * ring_stride
        loc_2d = loc_2d + positions_2d % ring_stride
        col = torch.arange(stride, device=DEVICE).view(1, -1)
        committed = col < commit_lens.to(torch.long).view(-1, 1)
        ref_loc = torch.where(committed, loc_2d, torch.full_like(loc_2d, -1)).to(
            torch.int32
        )

        self.assertTrue(torch.equal(got.positions, positions_2d.reshape(-1)))
        self.assertTrue(torch.equal(got.swa_loc, ref_loc.reshape(-1)))

    def test_scatter_bf16_into_unified(self):
        torch.manual_seed(20)
        n_rows, dim, n_pages = 8, 16, 32
        kv = torch.randn(n_rows, dim, device=DEVICE).to(torch.bfloat16).contiguous()
        loc = torch.tensor(
            [3, -1, 5, 7, 0, -1, 9, 11], device=DEVICE, dtype=torch.int32
        )
        unified = torch.zeros(n_pages, dim, device=DEVICE, dtype=torch.bfloat16)
        expected = unified.clone()
        keep = loc >= 0
        expected[loc[keep].long()] = kv[keep]

        runtime.scatter_bf16_into_unified(kv=kv, loc=loc, unified_kv=unified)
        self.assertTrue(torch.equal(unified, expected))

        with self.assertRaises(AssertionError):
            runtime.scatter_bf16_into_unified(kv=kv, loc=loc, unified_kv=unified.t())


class TestDeepseekV4DSparkUnifiedKVGSM8K(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V4_DSPARK_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        env = os.environ.copy()
        env.update(
            {
                "SGLANG_DEFAULT_THINKING": "1",
                "SGLANG_DSV4_REASONING_EFFORT": "max",
                "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "false",
                "SGLANG_USE_AITER": "1",
                "SGLANG_USE_ROCM700A": "0",
                "SGLANG_OPT_USE_FUSED_COMPRESS": "true",
                "SGLANG_HACK_FLASHMLA_BACKEND": "unified_kv_triton",
                "SGLANG_RAGGED_VERIFY_MODE": "static",
                "SGLANG_DSPARK_ENABLE_SPS_ONLINE_PROFILE": "0",
                "SGLANG_OPT_FP8_WO_A_GEMM": "false",
                "SGLANG_OPT_USE_JIT_INDEXER_METADATA": "false",
                "SGLANG_OPT_USE_TOPK_V2": "false",
                "SGLANG_OPT_USE_AITER_INDEXER": "true",
                "SGLANG_OPT_USE_TILELANG_INDEXER": "false",
                "SGLANG_OPT_USE_TILELANG_MHC_PRE": "false",
                "SGLANG_OPT_USE_TILELANG_MHC_POST": "false",
                "SGLANG_FP8_PAGED_MQA_LOGITS_TORCH": "1",
                "SGLANG_OPT_USE_FUSED_COMPRESS_TRITON": "true",
                "SGLANG_OPT_USE_MULTI_STREAM_OVERLAP": "false",
                "SGLANG_ROCM_USE_MULTI_STREAM": "false",
                "AITER_BF16_FP8_MOE_BOUND": "0",
                "SGLANG_EAGER_INPUT_NO_COPY": "true",
                "SGLANG_SHARED_EXPERT_TP1": "1",
                "SGLANG_DP_SHARED_EXPERT_LOCAL": "1",
                "SGLANG_DP_USE_GATHERV": "1",
                "SGLANG_DP_USE_REDUCE_SCATTER": "1",
                "GPU_MAX_HW_QUEUES": "5",
            }
        )
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--dp",
            "8",
            "--enable-dp-attention",
            "--enable-dp-lm-head",
            "--enable-prefill-delayer",
            "--disable-radix-cache",
            "--attention-backend",
            "dsv4",
            "--page-size",
            "256",
            "--mem-fraction-static",
            "0.9",
            "--swa-full-tokens-ratio",
            "0.15",
            "--disable-shared-experts-fusion",
            "--tool-call-parser",
            "deepseekv4",
            "--reasoning-parser",
            "deepseek-v4",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--chunked-prefill-size",
            "65536",
            "--cuda-graph-max-bs",
            "512",
            "--max-running-requests",
            "512",
            "--speculative-algorithm",
            "DSPARK",
            "--speculative-dspark-block-size",
            "5",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)

    def test_full_gsm8k_unified_kv_dspark_static(self):
        requests.get(self.base_url + "/flush_cache")
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=1319,
            parallel=512,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        if is_in_ci():
            write_github_step_summary(
                "### test_gsm8k (deepseek-v4-pro-dspark unified_kv static MI35x)\n"
                f"accuracy={metrics['accuracy']:.3f}\n"
                f"avg_spec_accept_length={avg_spec_accept_length:.2f}\n"
            )
        self.assertGreater(metrics["accuracy"], GSM8K_ACCURACY_THRESHOLD)
        self.assertGreater(avg_spec_accept_length, AVG_SPEC_ACCEPT_LENGTH_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
