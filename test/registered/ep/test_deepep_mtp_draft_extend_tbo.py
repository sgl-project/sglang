"""Two-batch overlap on the *draft-extend* forward (issue #7892), end to end.

`TestMTPWithTBO` in `test_deepep_small.py` already covers MTP + TBO with the
draft-extend forward running **monolithic** -- only the target's prefill and
target-verify forwards split. This file runs the same model, parallelism and
speculative config with `SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH=1`, which is
what opens the draft worker's TBO gate and makes the NextN draft-extend forward
split into two micro-batches as well. The two files are an A/B pair over one env
var, and `TestMTPWithTBO` is the baseline -- which is why no second server is
launched here.

WHAT THIS FILE PROVES, AND HOW
------------------------------
Coverage is not asserted by prose, it is **measured**: `SGLANG_TBO_DEBUG=1`
makes `TboForwardBatchPreparer.prepare_raw` -- the function reached only after
both TBO gates pass -- log one line per split batch carrying its forward mode.
`test_tbo_split_forward_mode_census` parses every one of those lines, prints the
full census, and requires the modes this feature spans. So a run either shows
you which phases really split, or fails; it never silently degrades into
testing the monolithic path.

WHAT THIS FILE DOES *NOT* COVER (measure these elsewhere)
--------------------------------------------------------
1. **The DSA seed-topk channel** (`op_capture_dsa_seed_topk`,
   `dsa_seed_topk_capture` / `dsa_seed_topk_select` slicing). `is_deepseek_dsa`
   additionally requires `index_topk` in the HF config, which plain DeepSeek-V3
   does not have. On a non-DSA model the seed buffer is None and that op is a
   no-op. Only the CPU unit tests exercise it, on synthetic tensors.
2. **The attn-TP padding tail** (the `end_token_index_b` clamp and the relaxed
   `extend_num_tokens` identity). Here `attn_tp_size = tp_size // dp_size = 1`,
   so `input_ids` never carries a padding tail.
3. **The idle-rank arm of the gate.** It needs a DP rank to actually go idle
   while others run draft-extend; steady traffic does not guarantee that. The
   gate's positive and negative branches are pinned deterministically by
   `test/registered/unit/batch_overlap/test_tbo_draft_extend_split.py`
   (`TestDraftExtendChildrenGate`) instead.

Admission notes (`.claude/rules/unit-test-admission.md`): the census case is a
**completeness contract** -- the gate is a six-term AND over configuration-static
facts, and if any term degrades (e.g. `speculative_moe_a2a_backend` stops
inheriting the target's `deepep` default) the feature turns off while gsm8k
still passes, because monolithic draft-extend is exactly what main does today.
The accept-length floor is the **derived property**: mis-slice `hidden_states`,
`select_index` or `kv_indptr` and the target still verifies correctly -- accuracy
survives -- while draft quality collapses and accept length falls toward 1.0.
Accuracy alone cannot see that; the accept-length floor is the guard for the
split math.
"""

import os
import re
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import requests

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=420, stage="base-c", runner_config="deepep-4-gpu-h100")

STDOUT_FILENAME = "stdout_deepep_mtp_draft_extend_tbo.txt"
STDERR_FILENAME = "stderr_deepep_mtp_draft_extend_tbo.txt"

# Emitted once per launch by EagleDraftWorker.__init__ when the gate opens.
GATE_BANNER = "Draft-extend TBO enabled"

# One line per *split* batch, logged from inside prepare_raw. ForwardMode is an
# IntEnum, so the f-string renders the value, not the name -- map it back through
# the enum rather than hardcoding numbers, so a renumbering cannot turn this into
# a silent match against the wrong mode.
_SPLIT_LOG_RE = re.compile(r"TboForwardBatchPreparer\.prepare\b.*?forward_mode=(\d+)")

# The phases this feature spans, and why each must appear:
#   TARGET_VERIFY   -- the target's decode-phase forward. Proves TBO is live
#                      end-to-end under speculative decoding (baseline behavior).
#   DRAFT_EXTEND_V2 -- the forward #7892 adds. Proves the draft worker's gate
#                      opened AND a draft-extend batch reached prepare_raw.
# The draft multi-step decode loop must NOT appear: it carries EagleDraftInput,
# not EagleDraftExtendInput, so the gate closes on it. That negative branch is
# not assertable from logs (a draft DECODE and a target DECODE render the same
# integer), so it is pinned by TestDraftExtendChildrenGate on CPU instead; here
# the DECODE count is only reported.
REQUIRED_SPLIT_MODES = (ForwardMode.TARGET_VERIFY, ForwardMode.DRAFT_EXTEND_V2)


class TestMTPDraftExtendTBO(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "4",
                "--enable-dp-attention",
                "--dp-size",
                "4",
                "--enable-two-batch-overlap",
                "--moe-a2a-backend",
                "deepep",
                "--trust-remote-code",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "2",
                "--speculative-eagle-topk",
                "3",
                "--speculative-num-draft-tokens",
                "3",
                "--speculative-draft-model-path",
                DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
                "--chunked-prefill-size",
                "256",
                "--cuda-graph-max-bs-decode",
                "32",
                "--max-running-requests",
                "128",
            ],
            env={
                **os.environ,
                "SGLANG_TBO_DEBUG": "1",
                # The draft-extend cuda graph runner cannot consume tbo_children,
                # so the gate requires the graph to be off. This is the single
                # env var that separates this test from TestMTPWithTBO.
                "SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH": "1",
            },
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None):
            kill_process_tree(cls.process.pid)
        for handle in (getattr(cls, "stdout", None), getattr(cls, "stderr", None)):
            if handle:
                handle.close()

    @staticmethod
    def _read_server_logs() -> str:
        """Both streams: the banner and the TBO debug lines are logger output,
        whose stream depends on the logging config, so never assume one file."""
        chunks = []
        for name in (STDOUT_FILENAME, STDERR_FILENAME):
            if os.path.exists(name):
                with open(name, errors="replace") as f:
                    chunks.append(f.read())
        return "\n".join(chunks)

    @classmethod
    def _split_mode_census(cls) -> Counter:
        """How many batches of each forward mode actually got TBO-split."""
        census = Counter()
        for raw in _SPLIT_LOG_RE.findall(cls._read_server_logs()):
            try:
                census[ForwardMode(int(raw))] += 1
            except ValueError:
                census[f"unknown({raw})"] += 1
        return census

    def _drive_decode_traffic(self, num_requests: int = 32) -> None:
        """Generate decode rounds with a batch big enough to actually split.

        The split index is `bs // 2` *per DP rank*, so with dp_size=4 a handful
        of requests can leave a rank with bs=1 and an empty child A. 32 in
        flight keeps every rank above that.
        """

        def _one(i: int):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": f"Question {i}: describe the water cycle step by step.",
                    "sampling_params": {"max_new_tokens": 64, "temperature": 0},
                },
                timeout=180,
            )
            self.assertEqual(response.status_code, 200)

        with ThreadPoolExecutor(max_workers=num_requests) as pool:
            list(pool.map(_one, range(num_requests)))

    def test_draft_extend_tbo_gate_opened(self):
        """The six-term gate in EagleDraftWorker.__init__ actually opened.

        Needs no traffic -- the banner is emitted at worker init. Kept separate
        from the census so a closed gate is distinguishable from a gate that
        opened but never saw a draft-extend batch.
        """
        self.assertIn(
            GATE_BANNER,
            self._read_server_logs(),
            "draft-extend TBO never engaged: the gate stayed closed at worker "
            "init, so this run is exercising the monolithic path. Check the TBO "
            "flag, SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH, the NextN draft model "
            "class, and that speculative_moe_a2a_backend still inherits the "
            "target's deepep default.",
        )

    def test_tbo_split_forward_mode_census(self):
        """Measure which forward modes really produced two micro-batches.

        This is the coverage proof for the whole file. `prepare_raw` is reached
        only after `tbo_split_seq_index` is non-None *and* (for the draft worker)
        the draft-extend gate opens, so one debug line per mode is direct
        evidence that phase split -- not an inference from accuracy holding up.
        """
        self._drive_decode_traffic()

        census = self._split_mode_census()
        print("\n###TBO split census (forward mode -> number of split batches):")
        for mode, count in sorted(census.items(), key=lambda kv: str(kv[0])):
            name = mode.name if isinstance(mode, ForwardMode) else str(mode)
            print(f"    {name:<16} {count}")
        print(
            f"    (DECODE reported only; the draft decode loop must stay "
            f"monolithic, which TestDraftExtendChildrenGate pins on CPU)\n"
        )

        missing = [m.name for m in REQUIRED_SPLIT_MODES if census[m] == 0]
        self.assertFalse(
            missing,
            f"these forward modes never split: {missing}. Modes seen: "
            f"{ {getattr(m, 'name', m): c for m, c in census.items()} }. "
            f"DRAFT_EXTEND_V2 missing means #7892's path did not run; "
            f"TARGET_VERIFY missing means TBO itself is not engaging and the "
            f"whole comparison against TestMTPWithTBO is void.",
        )

    def test_gsm8k_accuracy_and_accept_length(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(metrics)

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(
            f"###test_gsm8k (deepseek-v3 mtp + dp + tbo + draft-extend tbo):\n"
            f"accuracy={metrics['score']:.3f}\n"
            f"{avg_spec_accept_length=:.3f}\n"
        )

        # Same thresholds as TestMTPWithTBO: splitting draft-extend must be
        # output-equivalent, so neither number is allowed to move.
        self.assertGreater(metrics["score"], 0.60)
        self.assertGreater(avg_spec_accept_length, 2.1)


if __name__ == "__main__":
    unittest.main()
