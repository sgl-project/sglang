import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import requests
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from sglang.srt.state_capturer.routed_experts import (
    extract_routed_experts_from_meta_info,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=420, stage="extra-a", runner_config="2-gpu-large")

MODEL = DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST
_QWEN3_30B_A3B_NUM_LAYERS = 48
_QWEN3_30B_A3B_TOPK = 8
PROMPT = (
    "Write a short, careful explanation of why the sum of the first n odd numbers "
    "equals n squared, and then check the claim for n = 1, 2, 3 and 4."
)
TOOL_REPLY = "Tool result: the check passed for every n you listed. Continue."
# Turn 1 is retracted once this many of its tokens have streamed out.
RETRACT_AFTER_TOKENS = 128
# Below this share of differing rows, an unforced recompute under the perturbed
# routers cannot tell forcing from a no-op.
MIN_CONTROL_FLIP_FRAC = 0.01


def _perturbed_routers(out_dir: Path, *, seed: int, scale: float = 0.2) -> Path:
    """A checkpoint holding only the MoE routers, each scaled by 1 + scale * N(0, 1);
    /update_weights_from_disk loads the tensors it finds and keeps the rest."""
    model_dir = Path(snapshot_download(MODEL, allow_patterns=["*.safetensors"]))
    generator = torch.Generator().manual_seed(seed)
    routers = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                if name.endswith(".mlp.gate.weight"):
                    weight = f.get_tensor(name)
                    noise = torch.randn(weight.shape, generator=generator)
                    routers[name] = (weight.float() * (1 + scale * noise)).to(
                        weight.dtype
                    )
    assert len(routers) == _QWEN3_30B_A3B_NUM_LAYERS, len(routers)
    out_dir.mkdir(parents=True)
    save_file(routers, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    return out_dir


def _differing_row_frac(a: np.ndarray, b: np.ndarray) -> float:
    return float((np.sort(a, axis=-1) != np.sort(b, axis=-1)).any(axis=-1).mean())


class TestRoutedExpertsForcingRetract(CustomTestCase):
    """Positions a session already holds keep their routed experts when the engine
    recomputes them after a retract and after router weight updates.

    Rows are compared with the rows the engine held, never with a fresh prefill:
    SGLang prefill is not run-to-run exact. Rows are compared as sets, since a
    forced position whose experts the gate picks again keeps the gate's order.
    The triton attention backend with page size 1 runs on Hopper and Blackwell
    alike, and lets a radix hit cover every held token.
    """

    @classmethod
    def setUpClass(cls):
        workdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(workdir.cleanup)
        cls.perturbed = [
            _perturbed_routers(Path(workdir.name) / f"routers-{seed}", seed=seed)
            for seed in (1, 2)
        ]
        process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                2,
                "--enable-return-routed-experts",
                "--enable-routed-experts-forcing",
                "--moe-runner-backend",
                "triton",
                "--attention-backend",
                "triton",
                "--page-size",
                1,
            ],
        )
        cls.addClassCleanup(terminate_and_kill_process_tree, process)

    def test_forcing_across_retract_and_weight_updates(self):
        key = f"session-{time.time_ns()}"
        prompt = self._tokenize(PROMPT)

        # Turn 1 is retracted mid-decode; its re-prefill runs under new routers.
        turn1, retracted_at = self._generate_retracted(prompt, 2048, key=key)
        self.assertIsNotNone(
            retracted_at,
            f"turn 1 finished before {RETRACT_AFTER_TOKENS} tokens streamed out",
        )
        stats = turn1["meta_info"]["routed_experts_forcing"]
        self.assertGreaterEqual(
            stats["retract_snapshots"],
            1,
            f"the retract after {retracted_at} tokens held no rows: {stats}",
        )
        self.assertGreater(stats["forced_tokens"], 0, stats)
        self.assertEqual(stats["unforced_tokens"], 0, stats)
        tokens = prompt + turn1["output_ids"]
        held = len(tokens) - 1
        held_routes = self._routes(turn1)
        self.assertEqual(held_routes.shape[0], held)

        # Turn 2 re-prefills every held position under newer routers (the update
        # flushes the radix cache, as an RL weight sync does).
        self._post("/update_weights_from_disk", {"model_path": str(self.perturbed[1])})
        turn2_ids = tokens + self._tokenize(TOOL_REPLY, add_special_tokens=False)
        turn2 = self._generate(turn2_ids, 8, key=key, force_len=held)
        self.assertEqual(turn2["meta_info"]["cached_tokens"], 0)
        self.assertEqual(
            turn2["meta_info"]["routed_experts_forcing"],
            {
                "held_tokens": held,
                "forced_tokens": held,
                "unforced_tokens": 0,
                "retract_snapshots": 0,
            },
        )

        # Turn 2 returns only rows past the held ones. A probe of the same session
        # that holds nothing (force length 0) hits the radix KV turn 2 computed,
        # so it reads turn 2's forced rows back without recomputing them.
        probe = self._generate(turn2_ids, 1, key=key)
        self.assertGreaterEqual(probe["meta_info"]["cached_tokens"], held)
        forced_routes = self._routes(probe)[:held]
        np.testing.assert_array_equal(
            np.sort(forced_routes, axis=-1), np.sort(held_routes, axis=-1)
        )
        control = self._routes(self._generate(turn2_ids, 1))[:held]
        flips = _differing_row_frac(control, held_routes)
        self.assertGreater(flips, MIN_CONTROL_FLIP_FRAC, "perturbation too small")

    def test_missing_routes_refused(self):
        info = requests.get(f"{DEFAULT_URL_FOR_TEST}/server_info", timeout=60).json()
        self.assertEqual(info["routed_experts_forcing"]["version"], 1)

        message = "routed_experts_missing: held=0 requested=5"
        meta = self._generate(self._tokenize(PROMPT), 8, key="never-held", force_len=5)
        meta = meta["meta_info"]
        self.assertEqual(meta["finish_reason"]["type"], "abort")
        self.assertEqual(meta["finish_reason"]["message"], message)
        self.assertEqual(
            meta["routed_experts_forcing"]["refused"], "routed_experts_missing"
        )
        self.assertEqual(meta["completion_tokens"], 0)

        # The same refusal through the chat API.
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": PROMPT}],
                "max_tokens": 8,
                "return_routed_experts": True,
                "routed_experts_start_len": 5,
                "routed_experts_session_key": "never-held-chat",
                "return_meta_info": True,
            },
            timeout=600,
        )
        self.assertEqual(response.status_code, 200, response.text)
        choice = response.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "abort")
        self.assertEqual(choice["meta_info"]["finish_reason"]["message"], message)
        self.assertEqual(
            choice["meta_info"]["routed_experts_forcing"]["refused"],
            "routed_experts_missing",
        )

    def _generate_retracted(self, input_ids, max_new_tokens, *, key):
        """Stream a turn, retract it once RETRACT_AFTER_TOKENS tokens are out while
        the routers change, and return its final output and where it was retracted
        (None if it finished first)."""
        payload = {**self._payload(input_ids, max_new_tokens, key=key), "stream": True}
        retracted_at = None
        with requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate", json=payload, stream=True, timeout=600
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                out = json.loads(line[len(b"data: ") :])
                meta = out["meta_info"]
                if (
                    retracted_at is None
                    and meta["finish_reason"] is None
                    and meta["completion_tokens"] >= RETRACT_AFTER_TOKENS
                ):
                    retracted_at = meta["completion_tokens"]
                    self._post("/pause_generation", {"mode": "retract"})
                    self._post(
                        "/update_weights_from_disk",
                        {"model_path": str(self.perturbed[0])},
                    )
                    self._post("/continue_generation", {})
        return out, retracted_at

    def _tokenize(self, text: str, *, add_special_tokens: bool = True) -> list:
        return self._post(
            "/tokenize", {"prompt": text, "add_special_tokens": add_special_tokens}
        )["tokens"]

    def _payload(self, input_ids, max_new_tokens, *, key=None, force_len=0):
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_routed_experts": True,
            "routed_experts_start_len": force_len,
        }
        if key is not None:
            payload["routed_experts_session_key"] = key
        return payload

    def _generate(self, input_ids, max_new_tokens, *, key=None, force_len=0):
        return self._post(
            "/generate",
            self._payload(input_ids, max_new_tokens, key=key, force_len=force_len),
        )

    def _post(self, path: str, body: dict) -> dict:
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}{path}", json=body, timeout=600
        )
        response.raise_for_status()
        return response.json()

    def _routes(self, output: dict) -> np.ndarray:
        return extract_routed_experts_from_meta_info(output).reshape(
            -1, _QWEN3_30B_A3B_NUM_LAYERS, _QWEN3_30B_A3B_TOPK
        )


if __name__ == "__main__":
    unittest.main()
