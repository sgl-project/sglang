"""Diagnostic for test_cp_decode_attn_tp_server.py; reports, never asserts gaps.

That test requires VERIFY logprobs to match a teacher-forced prefill rescore
within 0.02 and fails by up to ~0.9 with identical greedy tokens. This measures
the same gap on a plain TP8 server and on the CP8 server, plus CP prefill
against TP prefill on identical token sequences, so the gap can be attributed
to FP8 noise (TP8 shows it too) or to the CP path (only CP8 shows it).
"""

import json
import unittest

import numpy as np
import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

# Run only on demand via /rerun-test; never part of a scheduled suite.
register_cuda_ci(
    est_time=2100,
    stage="extra-b",
    runner_config="8-gpu-b300",
    disabled="diagnostic for test_cp_decode_attn_tp_server.py; run with /rerun-test",
)

OUTPUT_LEN = 32
MODEL = "zai-org/GLM-5.3"
# Same as test_cp_decode_attn_tp_server.py, fusion disabled (its failing arm).
COMMON_ARGS = [
    "--trust-remote-code",
    "--tp",
    "8",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "5",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "6",
    "--cuda-graph-bs-decode",
    *[str(bs) for bs in range(1, 9)],
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
    "--enforce-disable-flashinfer-allreduce-fusion",
]
CP_ARGS = [
    "--attn-cp-size",
    "8",
    "--enable-prefill-cp",
    "--cp-strategy",
    "interleave",
    "--enable-cp-decode-attn-tp",
]
# Control: attention data parallel gives each rank all heads, as CP does, but
# without CP prefill. A CP-vs-TP gap this server shares is layout noise.
DP_ATTN_ARGS = ["--attn-dp-size", "8"]
CASES = [[1], [257], [8193], [257 + 8 * i for i in range(4)]]
# Prefixes of one 2100-token sequence, scored by prefill alone. Position p has
# the same context in every prefix, so a CP-vs-TP gap that tracks p points at
# short key lists, and one that tracks the prefix length points at short
# sequences (padding, per-rank split). 8 is the smallest length that runs as CP.
SWEEP_LENGTHS = [8, 9, 16, 17, 33, 64, 65, 128, 257, 2049, 2100]


def _stats(diff):
    return {
        "pos0": round(float(diff[0]), 4),
        "mean": round(float(diff.mean()), 4),
        "median": round(float(np.median(diff)), 4),
        "max": round(float(diff.max()), 4),
        "per_position": [round(float(x), 4) for x in diff],
    }


class TestCpDecodeAttnTpLogprobGapDiag(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, trust_remote_code=True)

    def _prompt(self, length, index):
        if length == 1:
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
        return prefix + (filler * (remaining // len(filler) + 1))[:remaining] + suffix

    def _generate(self, input_ids, max_new_tokens):
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
        }
        if max_new_tokens == 0:
            # Teacher-forced rescore, as in test_cp_decode_attn_tp_server.py.
            payload["logprob_start_len"] = 0
        response = requests.post(self.base_url + "/generate", json=payload, timeout=600)
        response.raise_for_status()
        return response.json()

    def _run_server(self, other_args, foreign_tokens=None, run_cases=True):
        """Generate and rescore every case; optionally rescore foreign tokens."""
        process = popen_launch_server(
            self.model,
            self.base_url,
            timeout=max(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, 1800),
            other_args=other_args,
        )
        try:
            out = {}
            for lengths in CASES if run_cases else []:
                key = str(lengths)
                prompts = [self._prompt(n, i) for i, n in enumerate(lengths)]
                gens = self._generate(prompts, OUTPUT_LEN)
                tokens = [
                    [row[1] for row in g["meta_info"]["output_token_logprobs"]]
                    for g in gens
                ]
                scores = self._generate(
                    [p + t for p, t in zip(prompts, tokens, strict=True)], 0
                )
                case = {
                    "prompts": prompts,
                    "tokens": tokens,
                    "verify": [],
                    "rescore": [],
                }
                for prompt, g, s in zip(prompts, gens, scores, strict=True):
                    case["verify"].append(
                        [row[0] for row in g["meta_info"]["output_token_logprobs"]]
                    )
                    case["rescore"].append(
                        [
                            row[0]
                            for row in s["meta_info"]["input_token_logprobs"][
                                len(prompt) :
                            ]
                        ]
                    )
                if foreign_tokens is not None:
                    foreign = self._generate(
                        [
                            p + t
                            for p, t in zip(
                                prompts, foreign_tokens[key]["tokens"], strict=True
                            )
                        ],
                        0,
                    )
                    case["rescore_foreign"] = [
                        [
                            row[0]
                            for row in f["meta_info"]["input_token_logprobs"][
                                len(prompt) :
                            ]
                        ]
                        for prompt, f in zip(prompts, foreign, strict=True)
                    ]
                out[key] = case
            base = self._prompt(max(SWEEP_LENGTHS), 0)
            out["sweep"] = {}
            for length in SWEEP_LENGTHS:
                (score,) = self._generate([base[:length]], 0)
                # Entry 0 has no logprob; entry p + 1 is scored from position p.
                out["sweep"][length] = [
                    row[0] for row in score["meta_info"]["input_token_logprobs"][1:]
                ]
            return out
        finally:
            kill_process_tree(process.pid)

    def test_report_logprob_gaps(self):
        tp = self._run_server(COMMON_ARGS)
        cp = self._run_server(COMMON_ARGS + CP_ARGS, foreign_tokens=tp)
        servers = {"tp8": tp, "cp8": cp}
        try:
            servers["dpattn8"] = self._run_server(
                COMMON_ARGS + DP_ATTN_ARGS, run_cases=False
            )
        except Exception as e:
            print(f"[sweep] dpattn8 server failed, control skipped: {e!r}")
        sweep = {}
        for a, b in (("cp8", "tp8"), ("dpattn8", "tp8"), ("cp8", "dpattn8")):
            if a not in servers or b not in servers:
                continue
            for length in SWEEP_LENGTHS:
                diff = np.abs(
                    np.array(servers[a]["sweep"][length])
                    - np.array(servers[b]["sweep"][length])
                )
                row = {
                    "mean": round(float(diff.mean()), 4),
                    "max": round(float(diff.max()), 4),
                    "argmax": int(diff.argmax()),
                    "first_16": [round(float(x), 4) for x in diff[:16]],
                }
                sweep[f"{a}_vs_{b}_{length}"] = row
                print(
                    f"[sweep] {a} vs {b} len={length} mean={row['mean']} "
                    f"max={row['max']} at pos={row['argmax']} "
                    f"first_16={row['first_16']}"
                )
        # Raw logprobs too, so the direction of each gap is visible.
        raw = {
            name: {length: s["sweep"][length][:64] for length in (33, 257, 2100)}
            for name, s in servers.items()
        }
        print("CP_PREFILL_SWEEP_REPORT " + json.dumps(sweep))
        print("CP_PREFILL_SWEEP_RAW " + json.dumps(raw))
        report = {}
        for key in tp:
            if key == "sweep":
                continue
            rows = []
            for i, prompt in enumerate(tp[key]["prompts"]):
                n = len(prompt)
                tp_verify = np.array(tp[key]["verify"][i])
                tp_rescore = np.array(tp[key]["rescore"][i])
                cp_verify = np.array(cp[key]["verify"][i])
                cp_rescore = np.array(cp[key]["rescore"][i])
                # CP prefill of the exact token sequence TP8 generated and scored.
                cp_on_tp_tokens = np.array(cp[key]["rescore_foreign"][i])
                rows.append(
                    {
                        "prompt_len": n,
                        "tokens_equal_tp_vs_cp": tp[key]["tokens"][i]
                        == cp[key]["tokens"][i],
                        "tp8_verify_vs_prefill": _stats(np.abs(tp_verify - tp_rescore)),
                        "cp8_verify_vs_prefill": _stats(np.abs(cp_verify - cp_rescore)),
                        "cp8_prefill_vs_tp8_prefill_same_tokens": _stats(
                            np.abs(cp_on_tp_tokens - tp_rescore)
                        ),
                    }
                )
            report[key] = rows
        print("CP_DECODE_ATTN_TP_GAP_REPORT " + json.dumps(report))
        for key, rows in report.items():
            for row in rows:
                print(
                    f"[gap] case={key} prompt_len={row['prompt_len']} "
                    f"tokens_equal={row['tokens_equal_tp_vs_cp']} "
                    + " ".join(
                        f"{name}: pos0={row[name]['pos0']} mean={row[name]['mean']} "
                        f"max={row[name]['max']}"
                        for name in (
                            "tp8_verify_vs_prefill",
                            "cp8_verify_vs_prefill",
                            "cp8_prefill_vs_tp8_prefill_same_tokens",
                        )
                    )
                )


if __name__ == "__main__":
    unittest.main()
