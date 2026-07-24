"""MI35x Kimi-K2.6 single-node DECODE-metadata probe (diagnostic, DO NOT MERGE).

Single-node counterpart to the disagg `1p1d-metadump` recipe, for the non-MTP
GSM8K drop investigation (disagg ~0.88 vs single-node 0.944). It launches the
SAME server config as test/registered/amd/accuracy/mi35x/test_kimi_k26_eval_mi35x.py
(TP8, aiter prefill / triton decode) but EAGER (--disable-cuda-graph) and with
the decode-metadata probe enabled, then runs a few GSM8K 8-shot questions at
parallel=1 (bs=1) so the `[DDM]` dump has the same shape as the disagg run.

Purpose: the disagg probe already showed the transferred prefix is present
(zero_rows=0), correctly indexed (idx==rtt), and seq_len is monotonic -- so the
only remaining variable is the transferred KV *values*. The 8-shot GSM8K prefix
is identical here and in the disagg gate, so the per-token latent norm sequence
must match if the values are preserved. Compare this run's step=1 `KV L0 norm`
head against the disagg run's:

    KV L0 norm head = [19.617, 22.029, 17.145, 23.779, 17.698, 19.234]   (kv_idx = first 6 prefix tokens)

  * norms MATCH  -> transferred KV values are faithful; the bug is in the decode
    compute / missing prefill-established state, not the KV read.
  * norms DIFFER -> the MORI transfer alters the latent values (layout / stride /
    dtype / partial-head), i.e. a transfer-fidelity bug.

Run (single MI35x node, inside the ROCm container):
    python3 test/manual/test_kimi_k26_decodemeta_mi35x.py
then: grep -A6 "\[DDM" on the server stdout.

Manual-only diagnostic (lives under test/manual/, not registered for CI).
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

KIMI_K26_MODEL_PATH = "moonshotai/Kimi-K2.6"
SERVER_LAUNCH_TIMEOUT = 5400
TP_SIZE = 8


class TestKimiK26DecodeMetaMI35x(CustomTestCase):
    """Single-node decode-metadata probe run (diagnostic)."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_kimi_k26_decode_metadata_probe(self):
        other_args = [
            "--tp",
            str(TP_SIZE),
            "--decode-attention-backend",
            "triton",
            "--prefill-attention-backend",
            "aiter",
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--watchdog-timeout",
            "1200",
            # Eager decode so forward_decode is entered in Python (the probe
            # never fires under HIP graph replay).
            "--disable-cuda-graph",
        ]
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_ROCM_FUSED_DECODE_MLA"] = "0"
        # Decode-metadata probe (see debug_utils/disagg_decode_meta_probe.py).
        env["SGLANG_DEBUG_DISAGG_DECODE_META"] = "1"
        env["SGLANG_DEBUG_DISAGG_DECODE_META_MINLEN"] = "128"
        env["SGLANG_DEBUG_DISAGG_DECODE_META_STEPS"] = "16"
        env["SGLANG_DEBUG_DISAGG_DECODE_META_KVNORM"] = "1"

        process = popen_launch_server(
            KIMI_K26_MODEL_PATH,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

        try:
            requests.get(self.base_url + "/flush_cache")

            # parallel=1 -> bs=1 decode, so the [DDM] dump matches the disagg
            # run's shape. A handful of questions is enough; only the first
            # request's first STEPS decode steps are dumped. Accuracy is not the
            # point here, so no threshold assertion.
            args = SimpleNamespace(
                num_shots=8,
                data_path=None,
                num_questions=8,
                parallel=1,
                max_new_tokens=32,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            print(
                f"[decodemeta] single-node accuracy over "
                f"{args.num_questions} q = {metrics['accuracy']:.3f} "
                f"(diagnostic; grep '[DDM' in server log for the metadata dump)"
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
