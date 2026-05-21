"""CI coverage for the public Nemotron-Labs-Diffusion 3B TinyStories DLLM.

The public 8B target checkpoint is available as
``nvidia/Nemotron-Labs-Diffusion-8B`` and uses the same
``NemotronLabsDiffusionModel`` architecture. This file keeps default CI on a
smaller public 3B checkpoint trained on TinyStories. The two scheduling
algorithms (FastDiffuser iterative denoising and LinearSpec speculative
decoding) get one server-launch fixture each.

The model is a story-completion model, not a benchmark model, so we do not
assert downstream-task accuracy. The asserts here are intentionally loose:
they catch boot/serving regressions (graph capture failures, empty
generations, blocking errors) without coupling to hardware-dependent
throughput numbers.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="stage-b-test-1-gpu-small")

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

_MODEL = "MMaghoumi/Nemotron-Labs-Diffusion-TinyStories-3b"
_CONFIG_DIR = "test/registered/dllm/configs"
# 3B BF16 weights (~6 GB) plus KV cache fit on a 5090; keep the context cap
# conservative so this also runs on smaller-VRAM CI hosts.
_COMMON_SERVER_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "1",
    "--mem-fraction-static",
    "0.85",
    "--max-running-requests",
    "4",
    "--attention-backend",
    "flashinfer",
    "--cuda-graph-bs",
    "1",
    "2",
    "3",
    "4",
    "--context-length",
    "4096",
]


# A TinyStories-trained checkpoint is expected to continue this prompt with a
# short children's story about a gingerbread man. We use the continuation as a
# coherence check: with greedy decoding the on-topic word should appear early
# in the generated text.
_STORY_PROMPT = "Once upon a time, there was a little gingerbread man who jumped out of the oven and"
_STORY_KEYWORDS = ("gingerbread", "ran", "fox")


def _check_generates_text(base_url, model, max_new_tokens=128):
    """Send a gingerbread-man story prompt and verify the model produces a
    coherent continuation.

    Asserts:
      - HTTP request round-trips successfully.
      - At least one token is generated.
      - The continuation contains at least one expected on-topic keyword,
        which catches degenerate outputs (empty / all-whitespace / mask
        repetitions) that a bare ``completion_tokens > 0`` check would miss.
    """
    response = requests.post(
        base_url + "/v1/completions",
        json={
            "model": model,
            "prompt": _STORY_PROMPT,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        },
        timeout=120,
    )
    if response.status_code != 200:
        raise AssertionError(response.text)
    body = response.json()
    completion_tokens = body.get("usage", {}).get("completion_tokens", 0)
    text = body["choices"][0]["text"]
    return completion_tokens, text


class TestNemotronLabsDiffusion3BFastDiffuser(CustomTestCase):
    """FastDiffuser (iterative denoising) on the 3B public DLLM."""

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_COMMON_SERVER_ARGS
            + [
                "--dllm-algorithm",
                "FastDiffuser",
                "--dllm-algorithm-config",
                f"{_CONFIG_DIR}/nemotron_labs_diffusion_3b_fastdiffuser.yaml",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_generates_story_continuation(self):
        completion_tokens, text = _check_generates_text(self.base_url, self.model)
        self.assertGreater(completion_tokens, 0, "model produced zero tokens")
        self.assertIsInstance(text, str)
        lowered = text.lower()
        self.assertTrue(
            any(kw in lowered for kw in _STORY_KEYWORDS),
            f"output missing all on-topic keywords {_STORY_KEYWORDS!r}: {text!r}",
        )

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=512)
        acc_length, speed = send_one_prompt(args)
        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (nemotron-labs-diffusion-3b-tinystories, fastdiffuser) tp=1\n"
                f"{speed=:.2f} token/s\n"
            )
            # Floor only — exact throughput is hardware-dependent. The intent
            # is to catch graph-capture regressions that collapse to single-
            # token-per-second eager mode, not to police kernel-level
            # performance on local developer machines.
            self.assertGreater(speed, 20)


class TestNemotronLabsDiffusion3BLinearSpec(CustomTestCase):
    """LinearSpec (1 draft + 1 verify per block) on the 3B public DLLM."""

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_COMMON_SERVER_ARGS
            + [
                "--dllm-algorithm",
                "LinearSpec",
                "--dllm-algorithm-config",
                f"{_CONFIG_DIR}/nemotron_labs_diffusion_3b_linearspec.yaml",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_generates_story_continuation(self):
        completion_tokens, text = _check_generates_text(self.base_url, self.model)
        self.assertGreater(completion_tokens, 0, "model produced zero tokens")
        self.assertIsInstance(text, str)
        lowered = text.lower()
        self.assertTrue(
            any(kw in lowered for kw in _STORY_KEYWORDS),
            f"output missing all on-topic keywords {_STORY_KEYWORDS!r}: {text!r}",
        )

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=512)
        acc_length, speed = send_one_prompt(args)
        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (nemotron-labs-diffusion-3b-tinystories, linearspec) tp=1\n"
                f"{speed=:.2f} token/s\n"
            )
            # LinearSpec should clear FastDiffuser by a comfortable margin on
            # the same hardware; keep the floor low to stay portable across
            # runners.
            self.assertGreater(speed, 40)


if __name__ == "__main__":
    unittest.main()
