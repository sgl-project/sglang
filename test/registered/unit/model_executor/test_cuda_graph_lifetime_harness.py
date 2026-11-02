# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Core fail-closed contracts of the optional graph-check harness, without GPUs."""

import hashlib
import json
import os
import tempfile
import unittest
from copy import deepcopy

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.cuda_graph_lifetime import harness

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NEW_TOKENS = 9
RUN_ID = "run-1"
NONCE = "run-1-flush"
LABELS = [f"DecodeCudaGraphRunner size=2 variant_label={v}" for v in ("lora", "nolora")]
RESOLVED = {
    "tp_size": "2",
    "pp_size": "1",
    "dp_size": "1",
    "attn_dp_size": "1",
    "nnodes": "1",
    "enable_lora": "True",
    "lora_backend": "triton_v2",
    "cuda_graph_backend_prefill": "disabled",
}


def _outputs(*, changed_row=None, token=11, score=-0.5):
    outputs = {}
    for row in range(2):
        chosen = token if row == changed_row else 11 + row
        logprob = score if row == changed_row else -0.5
        top = [[logprob, chosen, None], [logprob - 0.4, 1000, None]]
        top += [[-9.0 - i, 2000 + i, None] for i in range(3)]
        outputs[f"bs2-adapter-r0-{row}"] = {
            "meta_info": {
                "input_token_logprobs": [[None, 1, None], [-1.0, 2, None]],
                "output_token_logprobs": [
                    [logprob, chosen, None] for _ in range(NEW_TOKENS)
                ],
                "output_top_logprobs": [deepcopy(top) for _ in range(NEW_TOKENS)],
            }
        }
    return outputs


def _report(rank):
    return {
        "tracer_version": harness.TRACER_VERSION,
        "run_id": RUN_ID,
        "pid": 100 + rank,
        "hostname": "node",
        "parallel": {
            "tp_rank": rank,
            "pp_rank": 0,
            "tp_size": 2,
            "pp_size": 1,
            "dp_size": 1,
        },
        "device": f"cuda:{rank}",
        "device_uuid": f"GPU-{rank}",
        "status": "clean",
        "reason": "flush-request",
        "flush_nonce": NONCE,
        "counts": {"captures": 2, "replays": 4, "checks": 1},
        "findings": {"status": "clean", "graphs": 2, "pointers": 8},
        "label_captures": dict.fromkeys(LABELS, 1),
        "label_replays": dict.fromkeys(LABELS, 2),
    }


def _archive(directory, outputs, reports):
    required, _ = harness.required_coverage(
        RESOLVED, {"SGLANG_ENABLE_LORA_DUAL_GRAPH": "1"}, []
    )
    ledger = {
        "run_id": RUN_ID,
        "max_new_tokens": NEW_TOKENS,
        "topology": harness.topology(RESOLVED),
        "expected_workers": 2,
        "coverage_declaration": {"required": required},
        "evidence": {"outputs": {}},
        "verdicts": {},
        "phases": {},
    }
    for phase, responses in outputs.items():
        path = os.path.join(directory, f"{phase}.outputs.json")
        harness._write_outputs(path, RUN_ID, phase, responses)
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        ledger["evidence"]["outputs"][phase] = {
            "path": os.path.basename(path),
            "sha256": digest,
            "requests": len(responses),
        }
        ledger["phases"][phase] = {
            "requests": len(responses),
            "resolved_args": RESOLVED,
        }
    ledger["phases"]["instrumented"].update(worker_reports=reports, flush_nonce=NONCE)
    return ledger


class TestGraphCheckCore(unittest.TestCase):
    def test_current_evidence_is_accepted_and_incomplete_or_bad_evidence_is_not(self):
        cases = (
            ("complete", None, "PASS", None),
            ("drift", "numerical", "FAIL", "drift"),
            ("truncated", "numerical", "FAIL", "malformed"),
            ("missing worker", "lifetime", "FAIL", None),
            ("stale flush", "lifetime", "FAIL", None),
            ("missing variant", "coverage", "INCOMPLETE", None),
            ("crash with saved outputs", "numerical", "FAIL", "crash"),
            ("digest mismatch", "numerical", "INCONCLUSIVE", "evidence-mismatch"),
            ("control of another request", "numerical", "FAIL", "drift"),
            ("unmeasured control", "numerical", "FAIL", "drift"),
            ("measured control", "numerical", "INCONCLUSIVE", "batch-sensitivity"),
            ("missing control file", "numerical", "FAIL", "drift"),
        )
        for case, target, expected, kind in cases:
            with self.subTest(case), tempfile.TemporaryDirectory() as directory:
                candidate = (
                    _outputs(changed_row=0, score=-1.8)
                    if "control" in case or case == "drift"
                    else _outputs()
                )
                if case == "truncated":
                    candidate["bs2-adapter-r0-0"]["meta_info"][
                        "output_token_logprobs"
                    ].pop()
                outputs = {
                    "reference": _outputs(),
                    "ordinary": candidate,
                    "instrumented": deepcopy(candidate),
                }
                if "control" in case:
                    if case == "control of another request":
                        outputs["reference2"] = _outputs(
                            changed_row=1, token=12, score=-1.8
                        )
                    elif case == "unmeasured control":
                        outputs["reference2"] = _outputs(changed_row=0, token=99)
                    else:
                        outputs["reference2"] = deepcopy(candidate)
                reports = [_report(0), _report(1)]
                if case == "missing worker":
                    reports.pop()
                elif case == "stale flush":
                    reports[0]["flush_nonce"] = "an earlier flush"
                elif case == "missing variant":
                    reports[0]["label_replays"].pop(LABELS[1])
                ledger = _archive(directory, outputs, reports)
                if case == "crash with saved outputs":
                    ledger["phases"]["ordinary"]["error"] = "server crashed"
                elif case == "digest mismatch":
                    with open(
                        os.path.join(directory, "ordinary.outputs.json"), "w"
                    ) as f:
                        f.write("{}")
                elif case == "missing control file":
                    os.remove(os.path.join(directory, "reference2.outputs.json"))
                ledger_path = os.path.join(directory, "ledger.json")
                with open(ledger_path, "w") as f:
                    json.dump(ledger, f)
                with open(ledger_path) as f:
                    before = f.read()
                verdicts = harness.recompute(directory)["verdicts"]
                with open(ledger_path) as f:
                    self.assertEqual(f.read(), before)
                if target is None:
                    self.assertEqual(
                        {name: v["verdict"] for name, v in verdicts.items()},
                        dict.fromkeys(("lifetime", "numerical", "coverage"), "PASS"),
                    )
                else:
                    self.assertEqual(
                        verdicts[target]["verdict"], expected, verdicts[target]
                    )
                    if kind is not None:
                        self.assertEqual(verdicts[target]["kind"], kind)
                if case == "missing control file":
                    self.assertEqual(
                        verdicts["numerical"]["batch_sensitivity_control"]["verdict"],
                        "UNUSABLE",
                    )


if __name__ == "__main__":
    unittest.main()
