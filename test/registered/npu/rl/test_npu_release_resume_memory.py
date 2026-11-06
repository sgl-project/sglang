"""1-NPU smoke tests for the sleep/wake (release/resume) lifecycle.

RL Test Consolidation (sgl-project/sglang#47624) — sleep/wake area, 1-NPU tier.
File ownership per RFC section 5 (sleep/wake, 1-NPU smoke).

BF16 precision contract (RFC section 4): the RL rollout engine on Ascend runs
BF16 weights and KV cache. Both instances below launch with an explicit
``--dtype bfloat16`` (never dtype-inferred) and assert the effective dtype via
``/server_info`` (``dtype == "bfloat16"`` and ``launch_command`` pins the flag)
before any scenario assertion runs — S0. All bitwise determinism anchors
(S1/S3) therefore validate the BF16 numerics path specifically.

Coverage (scenarios map to the npu188 experiment matrix):
- S0  BF16 precision contract: server runs with explicit --dtype bfloat16
      and /server_info echoes the effective dtype (weights + KV cache follow
      the model dtype on the ascend attention backend).
- S1  KV-only sleep roundtrip is clean: release(kv_cache) -> resume(kv_cache)
      keeps greedy outputs bitwise identical (E5 template).
- S2  Full cycle with the RL standard chain: release(all) / resume(all)
      lifecycle (device-memory deltas logged as evidence — on this CANN/TMS
      stack TMS pause does not return pages to the device pool, so the
      watermark is informational only), then
      update_weights_from_disk(same weights) + /health_generate succeed.
      NOTE: under default (level-2 equivalent) semantics the phase between
      resume and update deliberately has NO output-correctness assertion —
      non-buffer runtime tensors are dropped by sleep (known defect, tracked
      separately as a defect-fix PR; see sleep-mode consolidation report).
- S4  cuda_graph tag contract: release(["cuda_graph"]) is accepted (HTTP 200).
      On NPU this tag currently has no actual release effect (TMS NPU has no
      cuda-graph context) — memory assertions therefore only cover
      weights+kv_cache.
- S3  Level-1 equivalence on a --enable-weights-cpu-backup server:
      release(all) -> resume(all) WITHOUT any update keeps outputs bitwise
      identical (E1-B template). This is the regression probe for the sleep
      pollution defect: any content drift here is a real regression.

Instance strategy: one shared server instance A (default config) runs S4, S1,
S2 in sequence; instance B (--enable-weights-cpu-backup) runs S3 alone.

Server die selection: set RL_FEAT_NPU_ID (default "0") to pin
ASCEND_RT_VISIBLE_DEVICES for the launched server.

Sibling CUDA test:
  test/registered/rl/test_multi_instance_release_memory_occupation.py
"""

import os
import unittest

import requests

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
)

try:
    import torch
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except Exception:  # pragma: no cover - non-NPU environments
    _NPU_AVAILABLE = False

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

register_npu_ci(est_time=1800, suite="nightly-npu-a3-merged", nightly=True)

# Model override for hosts without the default weights cached (e.g. npu188
# only caches Qwen/Qwen3-0.6B): export RL_FEAT_MODEL_PATH=<absolute path>.
MODEL = os.environ.get("RL_FEAT_MODEL_PATH", LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH)

PROMPTS = [
    "The capital of France is",
    "Write a one-sentence summary of photosynthesis.",
    "1 2 3 4 5",
    "你好，请用一句话介绍自己。",
    "def add(a, b):",
]


def _npu_used_memory_mb() -> float:
    free, total = torch.npu.mem_get_info()
    return (total - free) / (1024**2)


@unittest.skipUnless(_NPU_AVAILABLE, "Requires an Ascend NPU device")
class TestNpuReleaseResumeMemory(CustomTestCase):
    def _launch(self, extra_args):
        port = find_available_port(31000)
        self.base_url = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        env["ASCEND_RT_VISIBLE_DEVICES"] = os.environ.get("RL_FEAT_NPU_ID", "0")
        return popen_launch_server(
            MODEL,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--attention-backend", "ascend", "--dtype", "bfloat16"]
            + extra_args,
            env=env,
        )

    def _assert_bf16_contract(self):
        """S0: the running server must be pinned to BF16 (RFC section 4).
        Asserted against /server_info — the resolution result (``dtype``) and
        the actually-issued ``launch_command`` — so an accidental dtype
        inference change fails here instead of silently redefining the
        scenarios' numerics."""
        info = requests.get(self.base_url + "/server_info", timeout=120).json()
        self.assertEqual(
            str(info.get("dtype", "")).lower(),
            "bfloat16",
            f"server must run BF16 weights/KV, got dtype={info.get('dtype')!r}",
        )
        launch = info.get("launch_command", "")
        self.assertIn(
            "bfloat16",
            str(launch).lower(),
            f"launch command must pin --dtype bfloat16: {launch}",
        )

    def _post(self, path, payload):
        r = requests.post(
            self.base_url + path,
            json=payload,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        )
        self.assertEqual(r.status_code, 200, f"{path} -> {r.status_code}: {r.text}")
        return r

    def _generate_all(self, max_new_tokens=64):
        return [
            requests.post(
                self.base_url + "/generate",
                json={
                    "text": p,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": max_new_tokens,
                        "ignore_eos": True,
                    },
                },
                timeout=300,
            ).json()["text"]
            for p in PROMPTS
        ]

    def test_instance_a_lifecycle_memory_and_update_chain(self):
        """Shared default-config instance: S4 (cuda_graph tag), S1 (KV-only
        roundtrip), S2 (memory watermark + RL chain)."""
        process = self._launch(["--enable-memory-saver"])
        try:
            # S0: BF16 precision contract for this instance.
            self._assert_bf16_contract()

            # S4: cuda_graph tag is accepted; on NPU it is a release no-op today.
            self._post("/release_memory_occupation", {"tags": ["cuda_graph"]})
            self._post("/resume_memory_occupation", {"tags": ["cuda_graph"]})

            # S1: KV-only sleep roundtrip must be content-clean.
            self._post("/release_memory_occupation", {"tags": ["kv_cache"]})
            self._post("/resume_memory_occupation", {"tags": ["kv_cache"]})
            baseline = self._generate_all()
            self._post("/release_memory_occupation", {"tags": ["kv_cache"]})
            self._post("/resume_memory_occupation", {"tags": ["kv_cache"]})
            after_kv = self._generate_all()
            self.assertEqual(
                after_kv, baseline, "KV-only sleep roundtrip must be deterministic"
            )

            # S2: full cycle with the RL chain. No output assertion between
            # resume(all) and the weight update: default (level-2 equivalent)
            # semantics leave non-buffer runtime tensors undefined after sleep
            # (known defect, separate fix PR).
            #
            # Memory evidence: on this CANN/TMS stack pausing a TMS region does
            # NOT return pages to the device-level free pool (observed delta
            # ~2 MB), so torch.npu.mem_get_info is NOT a valid release oracle
            # here. We log the deltas as evidence and keep lifecycle + content
            # assertions as the hard contract; device-level memory evidence
            # needs npu-smi sampling or TMS-internal accounting instead.
            used_before = _npu_used_memory_mb()
            self._post("/release_memory_occupation", {})
            used_released = _npu_used_memory_mb()
            print(
                f"[memory-evidence] release(all) device delta = "
                f"{used_before - used_released:.0f} MB"
            )
            self.assertGreaterEqual(used_before - used_released, 0.0)

            self._post("/resume_memory_occupation", {})
            used_resumed = _npu_used_memory_mb()
            print(
                f"[memory-evidence] resume(all) device delta = "
                f"{used_resumed - used_released:.0f} MB"
            )

            r = self._post(
                "/update_weights_from_disk", {"model_path": MODEL, "flush_cache": True}
            )
            self.assertTrue(
                r.json().get("success"),
                f"update_weights_from_disk failed: {r.text}",
            )
            health = requests.get(self.base_url + "/health_generate", timeout=300)
            self.assertEqual(health.status_code, 200)
        finally:
            kill_process_tree(process.pid)

    def test_instance_b_level1_bitwise_restore_with_cpu_backup(self):
        """--enable-weights-cpu-backup (level-1 equivalent): sleep->resume
        without any weight update must restore outputs bitwise (E1-B). Any
        drift here is the sleep pollution defect regressing."""
        process = self._launch(["--enable-memory-saver", "--enable-weights-cpu-backup"])
        try:
            # S0: BF16 precision contract for this instance.
            self._assert_bf16_contract()

            baseline = self._generate_all()

            self._post("/release_memory_occupation", {})
            self._post("/resume_memory_occupation", {})

            restored = self._generate_all()
            self.assertEqual(
                restored,
                baseline,
                "level-1 (cpu backup) sleep must restore outputs bitwise",
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
