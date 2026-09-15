"""Qwen3.8-Flash-Next (Qwen4-Exp) collocated prefill CP E2E on 4x B200.

Collocated CP is the only prefill-CP layout Qwen4-Exp supports: the CP group
*is* the TP group, the residual stream / QSA attention / QSA indexer are
CP-sharded, and the MoE and linear (GDN) attention keep their TP partition.

The flag that turns it on is derived, not requested -- ``resolve_collocated_cp``
returns early when ``attn_cp_size <= 1`` -- so a missing or wrong
``--attn-cp-size`` would boot plain TP4 and still pass an accuracy gate. Hence
``test_collocated_cp_is_derived`` pins the resolved flag, and the accuracy gate
then covers the numerics of the path it proves is live.

Launching with the plain recipe also covers ``should_run_flashinfer_autotune``
turning the tuner off for this topology: with it on, the tuner's decode-shaped
dummy forward poisons the CUDA context and the scheduler dies during init.

Registry: extra-b-test-4-gpu-b200 (label-gated extra CI, 4x B200)
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=750, stage="extra-b", runner_config="4-gpu-b200")

MODEL = "RadixArk/Qwen3.8-Flash-Next-NVFP4"

SERVER_LAUNCH_TIMEOUT = 3600
CP_SIZE = 4
# Measured 0.9750 on 4x B200 with this recipe -- the same score the plain-TP4
# sibling (test/registered/e2e/models/test_qwen4_exp_models.py) gates at 0.94,
# so CP's reordered reductions cost nothing here. Kept at the sibling's 0.94
# so both Qwen4-Exp gates move together.
GSM8K_SCORE_THRESHOLD = 0.94

CP_ARGS = [
    # The CP group is the TP group; attn_cp_size != tp_size is rejected.
    "--tp-size",
    str(CP_SIZE),
    "--attn-cp-size",
    str(CP_SIZE),
    "--enable-prefill-cp",
    "--cp-strategy",
    "zigzag",
    # The QSA CP path attends over the whole gathered sequence and asserts a
    # zero prefix, so both sources of a partial prefill must be off.
    "--chunked-prefill-size",
    "-1",
    "--disable-radix-cache",
    # Attention TP collapses to width 1 under collocated CP, so every rank
    # holds the full attention weights and writes the full sequence's KV.
    # Leave more headroom than the plain-TP recipe.
    "--mem-fraction-static",
    "0.8",
    "--linear-attn-prefill-backend",
    "flashinfer",
    "--linear-attn-decode-backend",
    "flashinfer",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--reasoning-parser",
    "qwen3-thinking",
]


class TestQwen4ExpCollocatedCP(GSM8KMixin, CustomTestCase):
    """Collocated prefill CP4 == TP4, NVFP4 weights."""

    gsm8k_backend = "sgl_eval"
    gsm8k_thinking = True
    gsm8k_num_examples = 200
    gsm8k_num_threads = 32
    gsm8k_max_tokens = 16384
    gsm8k_score_threshold = GSM8K_SCORE_THRESHOLD

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=CP_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_collocated_cp_is_derived(self):
        """Guard the silent fallback: /server_info reports the resolved args."""
        info = requests.get(self.base_url + "/server_info", timeout=120).json()
        self.assertTrue(
            info["enable_collocated_cp"],
            "collocated CP was not derived; the server is running plain TP "
            f"(attn_cp_size={info.get('attn_cp_size')}, "
            f"enable_prefill_cp={info.get('enable_prefill_cp')})",
        )
        self.assertEqual(info["attn_cp_size"], CP_SIZE)
        self.assertEqual(info["tp_size"], CP_SIZE)

    def test_short_prompt_falls_back(self):
        """A prompt below ``cp_size * 2`` tokens takes the non-CP branch.

        ``ZigzagCPStrategy.can_apply`` needs every request to be at least
        ``2 * cp_size`` tokens long; shorter extends run the plain TP path,
        where the GDN output has to be all-reduced over the TP group because
        its heads are folded over the whole group. Cheap probe that the
        fallback still serves.
        """
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": "1+1=",
                "sampling_params": {"temperature": 0.0, "max_new_tokens": 8},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["text"])


if __name__ == "__main__":
    unittest.main()
