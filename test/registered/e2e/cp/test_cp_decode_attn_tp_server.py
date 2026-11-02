"""Real CP8/TP-decode regression for GLM-5.3 on eight Blackwell GPUs.

The one-device scale test cannot catch an omitted/double boundary reduction or
an incorrectly padded six-token verify graph. Exercise both fusion settings on
the same weights, rescore generated tokens through CP prefill, and require
actual target-verify graph replay at the requested batch sizes.
"""

import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import requests
import torch
from transformers import AutoTokenizer

import sglang
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
    try_cached_model,
)

# Two full FP8 model loads need the eight-GPU extra lane, not the scale test's
# four-GPU lane. No simulated ranks, dummy weights, or reduced layer count.
register_cuda_ci(est_time=1200, stage="extra-b", runner_config="8-gpu-b300")

TP_SIZE = 8
OUTPUT_LEN = 32
VERIFY_WIDTH = 6
GRAPH_REPLAY = re.compile(
    r"Decode graph replay: worker=target key_size=(\d+) \(bs\) "
    r"mode=TARGET_VERIFY raw_bs=(\d+)"
)


class TestCpDecodeAttnTpServer(CustomTestCase):
    model = "zai-org/GLM-5.3"
    # FP8 GEMMs and fused BF16 reductions need not produce bitwise logprobs.
    # Keep an absolute bound as well as exact greedy token equality; do not
    # discard positions after a divergence or normalize away a scale error.
    logprob_atol = 0.02

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if torch.cuda.device_count() < TP_SIZE:
            raise RuntimeError("This registered regression requires eight GPUs")
        if any(torch.cuda.get_device_capability(i)[0] != 10 for i in range(TP_SIZE)):
            raise RuntimeError("This regression requires SM100/SM103 Blackwell GPUs")
        cls.model = try_cached_model(cls.model)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, trust_remote_code=True)
        cls.artifacts = tempfile.TemporaryDirectory(prefix="cp-decode-attn-tp-")
        cls.addClassCleanup(cls.artifacts.cleanup)
        cls.process = None
        # The source fingerprint stays useful when the test is run from a
        # transferred checkout without .git. Do not attribute an older run to
        # a newer PR head merely because the command line was unchanged.
        root = Path(__file__).resolve().parents[4]
        cls.python_root = root / "python"
        if Path(sglang.__file__).resolve().parent != cls.python_root / "sglang":
            raise RuntimeError("Run this test with SGLang imported from this checkout")
        source_paths = [
            "python/sglang/srt/layers/cp/cp_decode_attn_tp.py",
            "python/sglang/srt/layers/flashinfer_comm_fusion.py",
            "python/sglang/srt/layers/layer_boundary/factories.py",
            "python/sglang/srt/layers/layer_boundary/exit.py",
            "python/sglang/srt/layers/layer_boundary/fusions/allreduce.py",
            "python/sglang/srt/layers/layernorm.py",
            "python/sglang/srt/model_executor/runner/base_cuda_graph_runner.py",
            "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py",
            "python/sglang/srt/models/deepseek_v2.py",
            "python/sglang/srt/utils/common.py",
            str(Path(__file__).resolve().relative_to(root)),
        ]
        print(
            json.dumps(
                {
                    "model": cls.model,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "source_sha256": {
                        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                        for path in source_paths
                    },
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)
            cls.process = None

    def _post(self, path, payload):
        response = requests.post(self.base_url + path, json=payload, timeout=600)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def _prompt(self, length, index):
        if length == 1:
            # Below the CP threshold: ordinary EXTEND must agree with VERIFY.
            return [self.tokenizer.encode("Hello", add_special_tokens=False)[0]]
        prefix = self.tokenizer.encode(
            f"Document {index}. Read the following background notes.\n",
            add_special_tokens=False,
        )
        filler = self.tokenizer.encode(
            "The archive contains background notes about an ordinary day. ",
            add_special_tokens=False,
        )
        suffix = self.tokenizer.encode(
            "\nCount upwards from one: 1, 2, 3, 4, 5,", add_special_tokens=False
        )
        remaining = length - len(prefix) - len(suffix)
        self.assertGreater(remaining, 0)
        return prefix + (filler * (remaining // len(filler) + 1))[:remaining] + suffix

    def _assert_logprobs(self, actual, expected, label):
        self.assertEqual(len(actual), len(expected), label)
        self.assertEqual(
            [row[1] for row in actual], [row[1] for row in expected], label
        )
        actual_values = np.asarray([row[0] for row in actual], dtype=np.float64)
        expected_values = np.asarray([row[0] for row in expected], dtype=np.float64)
        self.assertTrue(np.isfinite(actual_values).all(), label)
        self.assertTrue(np.isfinite(expected_values).all(), label)
        np.testing.assert_allclose(
            actual_values,
            expected_values,
            rtol=0,
            atol=self.logprob_atol,
            err_msg=label,
        )

    def _assert_prefill_rescore(self, prompts, outputs):
        # Independently complete the attention output on full CP-prefill rows.
        # Both fusion settings could otherwise agree on the same wrong decode
        # reduction. Teacher forcing keeps every compared prefix identical.
        token_lists = [
            [row[1] for row in output["meta_info"]["output_token_logprobs"]]
            for output in outputs
        ]
        scores = self._post(
            "/generate",
            {
                "input_ids": [p + t for p, t in zip(prompts, token_lists, strict=True)],
                "sampling_params": {"temperature": 0, "max_new_tokens": 0},
                "return_logprob": True,
                "logprob_start_len": 0,
            },
        )
        self.assertEqual(len(scores), len(outputs))
        for prompt, output, score in zip(prompts, outputs, scores, strict=True):
            self._assert_logprobs(
                output["meta_info"]["output_token_logprobs"],
                score["meta_info"]["input_token_logprobs"][len(prompt) :],
                "VERIFY versus teacher-forced CP prefill",
            )

    def _exercise(self, log_path):
        results = {}
        # Non-divisible lengths exercise CP padding. The 8193-token case spans
        # three 4096-token chunks, including a one-token final chunk.
        cases = [(1, [1]), (1, [8193])]
        cases += [(bs, [257 + 8 * i for i in range(bs)]) for bs in (1, 4, 8)]
        for batch_size, lengths in cases:
            prompts = [self._prompt(length, i) for i, length in enumerate(lengths)]
            previous = None
            for repeat in range(2):
                log_offset = log_path.stat().st_size
                outputs = self._post(
                    "/generate",
                    {
                        "input_ids": prompts,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": OUTPUT_LEN,
                            "ignore_eos": True,
                        },
                        "return_logprob": True,
                    },
                )
                self.assertEqual(len(outputs), batch_size)
                for length, output in zip(lengths, outputs, strict=True):
                    meta = output["meta_info"]
                    self.assertEqual(meta["prompt_tokens"], length)
                    self.assertEqual(meta["completion_tokens"], OUTPUT_LEN)
                    self.assertEqual(len(meta["output_token_logprobs"]), OUTPUT_LEN)
                    self.assertEqual(meta["cached_tokens"], 0)
                    self.assertGreaterEqual(meta["spec_verify_ct"], 2)
                    self.assertEqual(meta.get("retraction_count", 0), 0)
                with log_path.open("rb") as log:
                    log.seek(log_offset)
                    replayed = [
                        tuple(map(int, match))
                        for match in GRAPH_REPLAY.findall(log.read().decode())
                    ]
                self.assertIn(
                    (batch_size, batch_size),
                    replayed,
                    "The target must replay the requested bucket, not run eagerly "
                    "or pad a six-token single-request verify to CP8",
                )
                self.assertTrue(all(bucket == raw_bs for bucket, raw_bs in replayed))
                current = [o["meta_info"]["output_token_logprobs"] for o in outputs]
                if previous is not None:
                    for actual, expected in zip(current, previous, strict=True):
                        self._assert_logprobs(actual, expected, "repeated graph replay")
                previous = current
                results[(tuple(lengths), repeat)] = current
                # Score once per shape, after the replay assertions so scoring
                # requests cannot supply the graph evidence for generation.
                if repeat == 1:
                    self._assert_prefill_rescore(prompts, outputs)
        return results

    def test_prefill_verify_replay_and_fusion_parity(self):
        common_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--attn-cp-size",
            "8",
            "--enable-prefill-cp",
            "--cp-strategy",
            "interleave",
            "--enable-cp-decode-attn-tp",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "5",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            str(VERIFY_WIDTH),
            "--cuda-graph-bs-decode",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "--disable-prefill-cuda-graph",
            "--max-running-requests",
            "8",
            "--chunked-prefill-size",
            "4096",
            "--context-length",
            "16384",
            "--max-total-tokens",
            "32768",
            "--mem-fraction-static",
            "0.8",
            "--dsa-topk-backend",
            "sgl-kernel",
            "--speculative-dsa-topk-backend",
            "sgl-kernel",
            "--disable-radix-cache",
            "--random-seed",
            "260126",
        ]
        reference = None
        for fused in (False, True):
            label = "fused" if fused else "fallback"
            log_path = Path(self.artifacts.name) / f"{label}.log"
            args = common_args + (
                ["--flashinfer-allreduce-fusion-backend", "mnnvl"]
                if fused
                else ["--enforce-disable-flashinfer-allreduce-fusion"]
            )
            try:
                with (
                    log_path.open("w") as log,
                    envs.SGLANG_LOG_DECODE_GRAPH_KEY.override(True),
                ):
                    type(self).process = popen_launch_server(
                        self.model,
                        self.base_url,
                        timeout=max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 1800),
                        other_args=args,
                        # The CLI child must also use the fingerprinted source,
                        # even if its executable belongs to another installation.
                        env={
                            "PYTHONPATH": os.pathsep.join(
                                filter(
                                    None,
                                    [str(self.python_root), os.getenv("PYTHONPATH")],
                                )
                            )
                        },
                        return_stdout_stderr=(log, log),
                    )
                    response = requests.get(self.base_url + "/server_info", timeout=30)
                    response.raise_for_status()
                    info = response.json()
                    for key, expected in {
                        "tp_size": TP_SIZE,
                        "attn_cp_size": TP_SIZE,
                        "enable_cp_decode_attn_tp": True,
                        "enable_prefill_cp": True,
                        "cp_strategy": "interleave",
                        "speculative_algorithm": "EAGLE",
                        "speculative_num_steps": 5,
                        "speculative_eagle_topk": 1,
                        "speculative_num_draft_tokens": VERIFY_WIDTH,
                        "flashinfer_allreduce_fusion_backend": "mnnvl"
                        if fused
                        else None,
                    }.items():
                        self.assertEqual(info[key], expected, key)
                    if fused:
                        log_text = log_path.read_text()
                        ranks = re.findall(
                            r"FlashInfer AllReduce Fusion enabled and workspace initialized: "
                            r"backend=mnnvl, rank=(\d+), world_size=8,",
                            log_text,
                        )
                        self.assertEqual(
                            set(map(int, ranks)),
                            set(range(TP_SIZE)),
                            "The fused arm must initialize the real eight-rank workspace",
                        )
                        self.assertNotIn(
                            "CP decode all-reduce/norm fusion declined", log_text
                        )
                    results = self._exercise(log_path)
                    if fused:
                        self.assertNotIn(
                            "CP decode all-reduce/norm fusion declined",
                            log_path.read_text(),
                        )
                    if reference is None:
                        reference = results
                    else:
                        self.assertEqual(results.keys(), reference.keys())
                        for case, outputs in results.items():
                            for actual, expected in zip(
                                outputs, reference[case], strict=True
                            ):
                                self._assert_logprobs(
                                    actual, expected, f"fusion parity {case}"
                                )
                    self.assertIsNone(self.process.poll())
                    print(
                        json.dumps({"mode": label, "shapes_and_repeats": len(results)})
                    )
            finally:
                if self.process is not None:
                    terminate_and_kill_process_tree(self.process)
                    type(self).process = None
                # Preserve diagnostic evidence in CI stdout even on launch or
                # parity failure; the temporary files are cleaned up at teardown.
                print(f"===== {label} server log =====\n{log_path.read_text()}")


if __name__ == "__main__":
    unittest.main()
