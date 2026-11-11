"""Paged experts match the unpaged model, with decode captured (prompts one at a time) and host-planned (one batch)."""

import io
import json
import os
import re
import statistics
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoConfig

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
    DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=490, stage="extra-a", runner_config="1-gpu-large")

PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
    "clips in May. How many clips did Natalia sell altogether in April and May?",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
    "The ocean's waves crash against the shore in an endless rhythm,",
]
# Decode graphs only at bs 1, so a batch of prompts decodes without one; prefill graphs only at
# the sizes the prompts need.
SERVER_ARGS = [
    "--cuda-graph-backend-decode",
    "full",
    "--cuda-graph-backend-prefill",
    "breakable",
    "--cuda-graph-max-bs-decode",
    "1",
    "--cuda-graph-bs-prefill",
    "64",
    "128",
    "--skip-server-warmup",
    # Small enough that the unpaged and paged servers share one GPU.
    "--max-total-tokens",
    "4096",
    "--mem-fraction-static",
    "0.6",
    "--disable-shared-experts-fusion",
    "--disable-radix-cache",
    "--enable-deterministic-inference",
    "--attention-backend",
    "triton",
]
PAGED_ARGS = ["--enable-paged-experts", "--paged-experts-num-resident", "8"]
_HOST, _PORT = DEFAULT_URL_FOR_TEST.rsplit(":", 1)
PAGED_URL = f"{_HOST}:{find_available_port(int(_PORT) + 100)}"


class _PagedMatchesUnpaged:
    model: str
    #: False for MoE kernels that are not batch-invariant: compare the median logprob difference
    #: per prompt, decode ones up to the first greedy token the servers differ on.
    exact = True
    median_abs_tolerance = 0.0
    #: Extra arguments for both servers.
    server_args = ()

    @classmethod
    def setUpClass(cls):
        # Both servers boot and score at the same time.
        with ThreadPoolExecutor(2) as pool:
            reference = pool.submit(_reference, cls.model, cls.server_args)
            paged = pool.submit(
                _serve_and_score, cls.model, PAGED_URL, [*cls.server_args, *PAGED_ARGS]
            )
            cls.reference, cls.paged = reference.result(), paged.result()

    def test_captured_decode_matches(self):
        self._assert_match(self.reference["single"], self.paged["single"])

    def test_host_planned_decode_matches(self):
        self._assert_match(self.reference["batch"], self.paged["batch"])

    def _assert_match(self, reference, paged):
        decode_compared = decode_total = 0
        for prompt, ref, got in zip(PROMPTS, reference, paged):
            with self.subTest(prompt=prompt[:30]):
                if self.exact:
                    self.assertEqual(got, ref)
                    continue
                diffs = [abs(a - b) for a, b in zip(got[0], ref[0]) if a is not None]
                for (ref_tok, ref_lp), (tok, lp) in zip(ref[1], got[1]):
                    if tok != ref_tok:
                        break
                    diffs.append(abs(lp - ref_lp))
                    decode_compared += 1
                decode_total += len(ref[1])
                self.assertLessEqual(
                    statistics.median(diffs), self.median_abs_tolerance
                )
        if not self.exact:
            self.assertGreaterEqual(decode_compared, decode_total // 2)


class TestPagedExpertsBf16(_PagedMatchesUnpaged, CustomTestCase):
    model = DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE


class TestPagedExpertsFp8Block(_PagedMatchesUnpaged, CustomTestCase):
    """Block-FP8 experts (128x128 scales) on the triton runner."""

    model = DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE


class TestPagedExpertsGptq(_PagedMatchesUnpaged, CustomTestCase):
    """GPTQ int4 experts on the Marlin runner, whose reduction depends on the batch shape: the
    unpaged server alone differs by up to 4 nats at a near-tied position between one prompt and
    a batch, with medians below 0.05."""

    model = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"
    exact = False
    median_abs_tolerance = 0.1


class TestPagedExpertsAuto(_PagedMatchesUnpaged, CustomTestCase):
    """``--paged-experts-num-resident auto`` with a static budget that holds about half the
    experts: the startup sizing, the slot tables it allocates and the decode capture cap it sets,
    through captured and host-planned decode. The paged server boots after the unpaged one has
    stopped, so the memory it measures is its own."""

    model = DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE
    resident_share = 0.5  # of the experts, for the budget

    @classmethod
    def setUpClass(cls):
        cls.reference = _reference(cls.model)
        fraction = _mem_fraction_for_experts(cls.model, cls.resident_share)
        args = [
            "--mem-fraction-static",
            f"{fraction:.3f}",
            "--enable-paged-experts",
            "--paged-experts-num-resident",
            "auto",
        ]
        log = io.StringIO()
        cls.paged = _serve_and_score(cls.model, PAGED_URL, args, log=log)
        cls.log = log.getvalue()

    def test_auto_keeps_part_of_the_experts_resident(self):
        """Without this, a budget that holds every expert would pass the comparisons with no
        paging at all."""
        found = re.search(r"Paged experts: K=(\d+)(?:-\d+)? of (\d+) experts", self.log)
        self.assertIsNotNone(found, "auto logged no K")
        k, num_experts = map(int, found.groups())
        top_k = AutoConfig.from_pretrained(self.model).num_experts_per_tok
        self.assertLess(top_k, k)
        self.assertLess(k, num_experts)


class TestPagedExpertsDisk(_PagedMatchesUnpaged, CustomTestCase):
    """``--paged-experts-enable-disk``: a first start writes every MoE layer's experts to the
    expert cache (in an empty ``SGLANG_CACHE_DIR``), a second start reads them after checking
    a few groups; both serve what the unpaged model does."""

    model = DEFAULT_SMALL_MOE_MODEL_NAME_FOR_TEST_BASE

    @classmethod
    def setUpClass(cls):
        cls.reference = _reference(cls.model)
        with tempfile.TemporaryDirectory() as folder:
            cls.starts = _serve_twice_with_cache(cls.model, folder, PAGED_ARGS)
        cls.paged = cls.starts[1]["scores"]

    def test_the_first_start_writes_every_layer_and_the_second_reads_them(self):
        layers = AutoConfig.from_pretrained(self.model).num_hidden_layers
        written, read = (_cache_outcomes(start["log"]) for start in self.starts)
        self.assertEqual(written, {"read": 0, "rebuilt": 0, "written": layers})
        self.assertEqual(read, {"read": layers, "rebuilt": 0, "written": 0})

    def test_the_writing_start_matches_too(self):
        self.assertEqual(self.starts[0]["scores"], self.reference)


def _serve_twice_with_cache(model, folder, paged_args, server_args=()):
    """Two starts with the expert cache under ``folder``: their scores and logs."""
    starts = []
    for _ in range(2):
        log = io.StringIO()
        scores = _serve_and_score(
            model,
            PAGED_URL,
            [*server_args, *paged_args, "--paged-experts-enable-disk"],
            log=log,
            env=_cache_env(folder),
        )
        starts.append({"scores": scores, "log": log.getvalue()})
    return starts


def _cache_env(folder):
    """``SGLANG_CACHE_DIR`` at ``folder``, for the expert cache; the compile caches that
    default to it stay where they are, so the servers do not compile again."""
    from sglang.srt.environ import envs, third_party_cache_defaults

    env = {k: os.environ.get(k, v) for k, v in third_party_cache_defaults().items()}
    env["SGLANG_DG_CACHE_DIR"] = envs.SGLANG_DG_CACHE_DIR.get()
    env["SGLANG_CUTE_AOT_CACHE_DIR"] = envs.SGLANG_CUTE_AOT_CACHE_DIR.get()
    env["SGLANG_CACHE_DIR"] = folder
    return env


def _cache_outcomes(log):
    """What the expert cache's log line after loading counts."""
    found = re.search(r"Paged experts: expert cache \S+: (.*)", log)
    if found is None:
        return None
    counts = {"read": 0, "rebuilt": 0, "written": 0}
    for verb, key in (("read", "read"), ("rebuilt", "rebuilt"), ("wrote", "written")):
        n = re.search(rf"\b{verb} (\d+) layers?\b", found.group(1))
        if n is not None:
            counts[key] = int(n.group(1))
    return counts


_references = {}


def _reference(model, server_args=()):
    """The unpaged server's scores, once per model and arguments: the classes of one model
    share them."""
    key = (model, tuple(server_args))
    if key not in _references:
        _references[key] = _serve_and_score(model, DEFAULT_URL_FOR_TEST, [*server_args])
    return _references[key]


def _mem_fraction_for_experts(model, share):
    """A --mem-fraction-static whose budget holds the model's non-expert weights, 2 GiB for the
    KV cache and buffers, and ``share`` of its experts (bf16 Qwen MoE)."""
    config = AutoConfig.from_pretrained(model)
    experts = (
        config.num_hidden_layers
        * config.num_experts
        * 3
        * config.hidden_size
        * config.moe_intermediate_size
        * 2
    )
    with open(hf_hub_download(model, "model.safetensors.index.json")) as f:
        weights = json.load(f)["metadata"]["total_size"]
    free, _ = torch.cuda.mem_get_info()
    return (weights - experts + share * experts + (2 << 30)) / free


def _serve_and_score(model, base_url, other_args, log=None, env=None):
    """With ``log``, the server's output is copied into it; ``env`` adds to the server's
    environment."""
    args = SERVER_ARGS
    if "--mem-fraction-static" in other_args:
        i = args.index("--mem-fraction-static")
        args = args[:i] + args[i + 2 :]
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=args + other_args,
        # Compile DeepGEMM kernels lazily for the few shapes the prompts use.
        env={"SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0", **(env or {})},
        return_stdout_stderr=(log, log) if log is not None else None,
    )
    try:
        return {
            "single": [_generate(base_url, [p])[0] for p in PROMPTS],
            "batch": _generate(base_url, PROMPTS),
        }
    finally:
        terminate_and_kill_process_tree(process)


def _generate(base_url, prompts):
    """Prompt logprobs and greedy decode (token, logprob) pairs for ``prompts``, sent as one
    batched request."""
    metas = requests.post(
        base_url + "/generate",
        json={
            "text": prompts,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 16,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
        },
    ).json()
    return [
        (
            [lp for lp, _, _ in m["meta_info"]["input_token_logprobs"]],
            [(tok, lp) for lp, tok, _ in m["meta_info"]["output_token_logprobs"]],
        )
        for m in metas
    ]


if __name__ == "__main__":
    unittest.main(verbosity=3)
