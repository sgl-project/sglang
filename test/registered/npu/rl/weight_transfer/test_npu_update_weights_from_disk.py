"""Phase 3 NPU tests: refit from a checkpoint on disk.

``/update_weights_from_disk`` is the path a trainer uses when it already wrote
a checkpoint. The refit is checked against a server that loaded the *same*
checkpoint at startup, and ``keep_pause`` is asserted to hold the engine paused
until ``/continue_generation`` so the caller can finish its own work first.
"""

import os
import shutil
import tempfile
import threading
import unittest

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_rl_utils import (
    assert_logprob_signature_equal,
    continue_generation,
    generate,
    logprob_signature,
    npu_rl_env,
    npu_server,
    output_stays_frozen,
    pause_generation,
    post,
    start_stream,
    update_weights_from_disk,
    wait_for_first_token,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=900, suite="extra-b-test-1-npu-a3")
register_npu_ci(est_time=900, suite="nightly-test-1-npu-a3", nightly=True)

MODEL = QWEN3_0_6B_WEIGHTS_PATH
HAVE_MODEL = os.path.isfile(os.path.join(MODEL, "config.json"))
PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
)
# One layer-0 projection, scaled: large enough that the greedy answer must
# change, small enough that the checkpoint stays a valid model.
PERTURBED_TENSOR = "model.layers.0.self_attn.o_proj.weight"
PERTURBED_SCALE = 1.5
PERTURBED_BIAS = 0.1
BASE_ARGS = (
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
    "--disable-overlap-schedule",
    "--disable-radix-cache",
    "--context-length",
    "4096",
    "--max-running-requests",
    "4",
    # Two servers share one card: the reference loads the perturbed checkpoint
    # at startup, the lane holds the original one and refits into it.
    "--mem-fraction-static",
    "0.35",
)
WEIGHT_SUFFIXES = (".safetensors", ".bin")


def _is_weight_file(name: str) -> bool:
    return name.endswith(WEIGHT_SUFFIXES)


def write_refit_checkpoints(source: str, perturbed: str, malformed: str) -> None:
    """Write a valid refit target and a partial checkpoint with a bad shape."""
    from safetensors.torch import load_file, save_file

    os.makedirs(perturbed)
    os.makedirs(malformed)
    shards = sorted(
        name
        for name in os.listdir(source)
        if _is_weight_file(name) and "index" not in name
    )
    if not shards:
        raise FileNotFoundError(f"no weight file under {source}")

    state = {}
    for shard in shards:
        state.update(load_file(os.path.join(source, shard)))
    if PERTURBED_TENSOR not in state:
        raise AssertionError(f"{PERTURBED_TENSOR} is not in {source}")

    weight = state[PERTURBED_TENSOR]
    # The loader sorts safetensors keys, so o_proj is copied before the norm
    # fails. The partial write would undo the perturbation if left in place.
    save_file(
        {
            PERTURBED_TENSOR: weight,
            "model.norm.weight": state["model.norm.weight"][:2].clone(),
        },
        os.path.join(malformed, "model.safetensors"),
        metadata={"format": "pt"},
    )
    state[PERTURBED_TENSOR] = (
        (weight.float() * PERTURBED_SCALE + PERTURBED_BIAS)
        .to(weight.dtype)
        .contiguous()
    )

    for name in sorted(os.listdir(source)):
        path = os.path.join(source, name)
        if not os.path.isfile(path) or _is_weight_file(name):
            continue
        if name.endswith(".index.json"):
            continue
        shutil.copyfile(path, os.path.join(perturbed, name))
    save_file(
        state,
        os.path.join(perturbed, "model.safetensors"),
        metadata={"format": "pt"},
    )


@unittest.skipUnless(HAVE_MODEL, f"model not found: {MODEL}")
class TestNpuUpdateWeightsFromDisk(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._perturbed_dir = tempfile.TemporaryDirectory(
            prefix="npu-disk-refit-target-"
        )
        cls.perturbed = os.path.join(cls._perturbed_dir.name, "perturbed")
        cls.malformed = os.path.join(cls._perturbed_dir.name, "malformed")
        write_refit_checkpoints(MODEL, cls.perturbed, cls.malformed)
        env = npu_rl_env()
        with npu_server(cls.perturbed, extra_args=BASE_ARGS, env=env) as base_url:
            cls.perturbed_reference = logprob_signature(base_url, PROMPTS)
        cls._lane = npu_server(MODEL, extra_args=BASE_ARGS, env=env)
        cls.base_url = cls._lane.__enter__()
        cls.baseline = logprob_signature(cls.base_url, PROMPTS)

    @classmethod
    def tearDownClass(cls):
        cls._lane.__exit__(None, None, None)
        cls._perturbed_dir.cleanup()

    def tearDown(self):
        try:
            continue_generation(self.base_url)
        except Exception:
            pass

    def _signature(self):
        return logprob_signature(self.base_url, PROMPTS)

    def test_01_disk_refit_matches_startup_load(self):
        self.assertNotEqual(
            [item[0] for item in self.baseline],
            [item[0] for item in self.perturbed_reference],
            "the perturbed checkpoint produced the original output -- the oracle "
            "would not detect a missed refit",
        )

        # A refit that cannot be loaded must report failure and keep the model
        # it already has, not leave a half-written one behind.
        failed = post(
            self.base_url,
            "/update_weights_from_disk",
            {"model_path": os.path.join(self.perturbed, "missing")},
            expect=None,
        )
        self.assertEqual(failed.status_code, 400, "a missing path was accepted")
        self.assertFalse(failed.json()["success"])
        assert_logprob_signature_equal(
            self._signature(), self.baseline, "a failed refit changed the output"
        )

        update_weights_from_disk(self.base_url, MODEL, load_format="auto")
        self.assertEqual(
            [item[0] for item in self._signature()],
            [item[0] for item in self.baseline],
            "reloading the same checkpoint changed the output",
        )

        update_weights_from_disk(self.base_url, self.perturbed, load_format="auto")
        assert_logprob_signature_equal(
            self._signature(),
            self.perturbed_reference,
            "refit from disk did not reproduce a startup load of the same checkpoint",
        )

        failed = post(
            self.base_url,
            "/update_weights_from_disk",
            {"model_path": self.malformed, "load_format": "auto"},
            expect=None,
        )
        self.assertEqual(failed.status_code, 400, failed.text)
        result = failed.json()
        self.assertFalse(result["success"])
        self.assertIn("Attempted to load weight", result["message"])
        self.assertIn("Rolling back to original weights", result["message"])
        assert_logprob_signature_equal(
            self._signature(),
            self.perturbed_reference,
            "rollback did not restore the last successful refit",
        )

        # Refitting back has to restore the original model, not a blend.
        update_weights_from_disk(self.base_url, MODEL, load_format="auto")
        assert_logprob_signature_equal(
            self._signature(), self.baseline, "refitting back did not restore the model"
        )

    def test_02_keep_pause_holds_the_engine_until_continue(self):
        # The paused request keeps its KV, so this refit must not flush it.
        update_weights_from_disk(
            self.base_url, MODEL, load_format="auto", flush_cache=False
        )

        request = start_stream(self.base_url, PROMPTS[0], max_new_tokens=256)
        wait_for_first_token(request, label="keep_pause")
        pause_generation(self.base_url, mode="in_place")

        result = update_weights_from_disk(
            self.base_url,
            MODEL,
            load_format="auto",
            keep_pause=True,
            flush_cache=False,
        )
        self.assertTrue(result.get("success"), "keep_pause refit reported failure")

        # A concurrent short request must stay queued while the engine is held
        # paused; the streaming request must not grow either.
        outcomes = {}

        def _run():
            try:
                outcomes["response"] = generate(
                    self.base_url, PROMPTS[1], max_new_tokens=8
                )
            except Exception as exc:  # pragma: no cover - surfaced in the assert
                outcomes["error"] = exc

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        self.assertTrue(
            output_stays_frozen(lambda: len(request.events), window=3.0),
            "the engine emitted tokens while paused with keep_pause",
        )
        self.assertTrue(
            thread.is_alive() and "response" not in outcomes,
            "a new request completed while the engine was held paused",
        )

        continue_generation(self.base_url)
        thread.join(timeout=180)
        self.assertFalse(thread.is_alive(), "request did not finish after continue")
        self.assertNotIn("error", outcomes, f"request failed: {outcomes.get('error')}")

        request.thread.join(timeout=180)
        self.assertFalse(request.thread.is_alive(), "stream did not end after continue")
        self.assertFalse(request.errors, f"stream failed: {request.errors}")
        finish = request.events[-1].get("meta_info", {}).get("finish_reason", {})
        self.assertNotEqual(
            finish.get("type"), "abort", "keep_pause aborted the running request"
        )
        self.assertGreater(
            len(request.events),
            1,
            "the paused request produced no tokens after continue",
        )


if __name__ == "__main__":
    unittest.main()
