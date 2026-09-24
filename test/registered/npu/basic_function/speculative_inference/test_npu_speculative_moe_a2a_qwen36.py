"""Exercise draft MoE communication on two Ascend NPUs with Qwen3.6 MTP.

Example (from the repository root):
    ASCEND_RT_VISIBLE_DEVICES=14,15 \
    SGLANG_TEST_MODEL_PATH=/home/weights/Qwen3.6-35B-A3B \
    python test/registered/npu/basic_function/speculative_inference/test_npu_speculative_moe_a2a_qwen36.py -v

SGLANG_TEST_TARGET_A2A defaults to deepep; SGLANG_TEST_DRAFT_A2A defaults to none.
The default compares inherited DeepEP with the draft's local expert dispatch at
EP=2, including speculative verification and deterministic output parity.
Set SGLANG_TEST_DRAFT_A2A=deepep to check explicit DeepEP selection instead.
SGLANG_TEST_LOG_DIR preserves the server logs and response/metric artifacts.
Uses the launch/cleanup pattern in python/sglang/test/ascend.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.srt.utils.network import get_open_port
from sglang.test.ascend.test_ascend_utils import MODEL_WEIGHTS_DIR
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase, popen_launch_server

register_npu_ci(est_time=600, suite="full-2-npu-a3", nightly=True)


@unittest.skipUnless(is_npu(), "Requires Ascend NPU")
class TestNpuSpeculativeMoeA2AQwen36(CustomTestCase):
    def setUp(self):
        self.assertGreaterEqual(torch.npu.device_count(), 2)
        self.model = os.environ.get(
            "SGLANG_TEST_MODEL_PATH",
            os.path.join(MODEL_WEIGHTS_DIR, "Qwen/Qwen3.6-35B-A3B"),
        )
        self.target_backend = os.environ.get("SGLANG_TEST_TARGET_A2A", "deepep")
        self.draft_backend = os.environ.get("SGLANG_TEST_DRAFT_A2A", "none")
        log_dir = os.environ.get("SGLANG_TEST_LOG_DIR")
        if log_dir is None:
            temp_dir = tempfile.TemporaryDirectory(prefix="sglang-npu-spec-a2a-")
            self.addCleanup(temp_dir.cleanup)
            log_dir = temp_dir.name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        self.answers = ["42", "56", "81", "15"]
        self.prompts = [
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": f"Calculate {expression}. Reply with only the integer answer.",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for expression in ["19 + 23", "7 * 8", "9 * 9", "45 / 3"]
        ]

    def _run_backend(self, label, draft_backend):
        base_url = f"http://127.0.0.1:{get_open_port()}"
        args = [
            "--device",
            "npu",
            "--attention-backend",
            "ascend",
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--mamba-ssm-dtype",
            "bfloat16",
            "--tp-size",
            "2",
            "--ep-size",
            "2",
            "--moe-a2a-backend",
            self.target_backend,
            "--mem-fraction-static",
            "0.75",
            "--context-length",
            "2048",
            "--max-total-tokens",
            "4096",
            "--max-running-requests",
            "4",
            "--max-mamba-cache-size",
            "8",
            "--chunked-prefill-size",
            "1024",
            "--disable-radix-cache",
            "--disable-cuda-graph",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "1",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "2",
            "--speculative-draft-model-quantization",
            "unquant",
            "--speculative-draft-attention-backend",
            "ascend",
            "--speculative-moe-runner-backend",
            "auto",
            "--random-seed",
            "0",
        ]
        if draft_backend is not None:
            args += ["--speculative-moe-a2a-backend", draft_backend]
        if "ascend_fuseep" in (self.target_backend, draft_backend):
            # DeepEP FuseEP mode 2 accepts INT8 weights only. Probe the BF16
            # checkpoint with mode 1, which exposes a BF16 kernel interface.
            args += ["--fuseep-mode", "1"]
        env = {
            **os.environ,
            # Qwen3.6 has 128 local experts at EP=2. DeepEP normal-mode
            # metadata alone needs more than the usual 200 MB test setting.
            "HCCL_BUFFSIZE": "512",
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
            # The A3 low-latency window supports at most 512. This test has
            # only four concurrent requests and two tokens per verify pass.
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
            "TOKENIZERS_PARALLELISM": "false",
        }
        log_path = self.log_dir / f"{label}.log"
        with log_path.open("w", encoding="utf-8") as log_file:
            process = None
            try:
                process = popen_launch_server(
                    self.model,
                    base_url,
                    timeout=900,
                    other_args=args,
                    env=env,
                    return_stdout_stderr=(log_file, log_file),
                )
                info_response = self.session.get(f"{base_url}/server_info", timeout=30)
                info_response.raise_for_status()
                info = info_response.json()
                self.assertEqual(info["speculative_algorithm"], "EAGLE")
                self.assertEqual(info["moe_a2a_backend"], self.target_backend)
                self.assertEqual(info["speculative_moe_a2a_backend"], draft_backend)
                outputs = []
                # Cover a single request and a batch with distinct lengths/content.
                for prompts, answers in [
                    (self.prompts[:1], self.answers[:1]),
                    (self.prompts, self.answers),
                ]:
                    response = self.session.post(
                        f"{base_url}/generate",
                        json={
                            "text": prompts,
                            "sampling_params": {"temperature": 0, "max_new_tokens": 64},
                        },
                        timeout=180,
                    )
                    response.raise_for_status()
                    results = response.json()
                    self.assertEqual(len(results), len(prompts))
                    outputs.extend(results)
                    # Save evidence before assertions, including failing output.
                    (self.log_dir / f"{label}.json").write_text(
                        json.dumps({"server_info": info, "outputs": outputs}, indent=2),
                        encoding="utf-8",
                    )
                    for result, answer in zip(results, answers):
                        self.assertEqual(result["text"].strip(), answer, result)
                        meta = result["meta_info"]
                        self.assertGreater(meta["completion_tokens"], 0)
                        self.assertGreater(meta.get("spec_verify_ct", 0), 0, meta)
                        self.assertGreater(
                            meta.get("spec_num_proposed_drafts", 0), 0, meta
                        )
                return outputs
            except Exception:
                log_file.flush()
                print(log_path.read_text(encoding="utf-8", errors="replace")[-16000:])
                raise
            finally:
                if process is not None:
                    kill_process_tree(process.pid)
                    process.wait(timeout=30)

    def test_draft_backend_override(self):
        inherited = self._run_backend("inherited", None)
        overridden = self._run_backend(
            f"draft_{self.draft_backend}", self.draft_backend
        )
        self.assertEqual(
            [output["text"] for output in inherited],
            [output["text"] for output in overridden],
        )


if __name__ == "__main__":
    unittest.main()
