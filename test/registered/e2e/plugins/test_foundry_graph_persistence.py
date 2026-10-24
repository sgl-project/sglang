"""End-to-end test of the external Foundry plugin (CUDA graph save/restore).

Per class, three engines run in sequence on one GPU with the same CUDA graph
sets: native capture, Foundry SAVE (captures and writes the archive) and
Foundry LOAD (rebuilds the graphs from the archive instead of capturing). LOAD
must produce the same greedy tokens as native, decode at the same speed,
restore its graphs quickly, and actually run the plugin.

Qwen3.5-2B (hybrid gated DeltaNet) runs decode graphs only: its full-backend
prefill capture fails in SGLang (the GDN out-of-graph metadata is decode-only).
Qwen3-1.7B covers full-backend prefill plus decode graphs.

Needs the `foundry` package in the serving environment (it registers the
`sglang.srt.plugins` entry point); skipped otherwise.

Run:  python3 test/registered/e2e/plugins/test_foundry_graph_persistence.py
"""

import glob
import importlib.util
import os
import re
import shutil
import tempfile
import unittest

import requests

from sglang.benchmark.serving import run_benchmark
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    get_benchmark_args,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# 1-gpu-large-foundry installs Foundry (scripts/ci/cuda/ci_install_foundry.sh); skips elsewhere.
register_cuda_ci(est_time=420, stage="nightly", runner_config="1-gpu-large-foundry")

_CONFIG_ENV = "FOUNDRY_GRAPH_EXTENSION_CONFIG"

# Upper bound on LOAD's graph restore (decode + prefill lines summed). Measured on
# H200 with foundry 056b154: 0.032 s (prefill 4 graphs 0.012 s, decode 8 graphs 0.020 s);
# older foundry counted the whole prefill loop in this line (max 1.18 s over 5 runs).
_RESTORE_BOUND_S = 2.0
# How much slower LOAD's median TPOT may be than native's (one-sided). Measured on a
# quiet H200: LOAD within 2% of native whenever both benches were clean (3 of 4 runs).
# In the 4th the native engine's prefills stalled (TTFT 151 vs 85 ms, ITL unchanged)
# and native was the slow side (LOAD -18.7%); a slow native says nothing about Foundry.
_TPOT_REL_TOL = 0.15

_PROMPTS = [
    "The capital of France is",
    "1, 2, 3, 4, 5,",
    "def fibonacci(n):",
    "Explain why the sky is blue in one sentence.",
    "Translate to German: The weather is nice today.",
    "The three primary colors are",
    "Write a haiku about the ocean.",
    "In 1969, the first person to walk on the moon was",
]
_SAMPLING = {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True}

_ACTIVE = re.compile(r"\[Foundry\] sglang plugin active: pid=(\d+)")
_DECODE_LOADED = re.compile(r"\[Foundry\] Loaded (\d+) SGLang graphs in ([0-9.]+)s")
_PREFILL_LOADED = re.compile(
    r"\[Foundry\] Loaded (\d+) SGLang prefill graphs in ([0-9.]+)s"
)
_PREFILL_PASSES = re.compile(
    r'^sglang:cuda_graph_passes_total\{[^}]*mode="prefill_cuda_graph"[^}]*\}'
    r"\s+([0-9.eE+-]+)$",
    re.MULTILINE,
)


def _write_toml(path: str, mode: str, workspace: str) -> str:
    # The region is virtual address space for everything allocated after setup,
    # the KV pool included; 256GB covers an 80-192 GB GPU at this memory fraction.
    with open(path, "w") as f:
        f.write(
            f'mode = "{mode}"\nbase_addr = 0x600000000000\nregion_size = "256GB"\n'
            f'workspace_root = "{workspace}"\nscratch_space_size = "1024MB"\n'
        )
    return path


def _prefill_graph_passes(base_url: str) -> float:
    metrics = requests.get(base_url + "/metrics", timeout=30).text
    return sum(map(float, _PREFILL_PASSES.findall(metrics)), 0.0)


def _generate(base_url: str, text):
    response = requests.post(
        base_url + "/generate",
        json={"text": text, "sampling_params": _SAMPLING},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


@unittest.skipIf(importlib.util.find_spec("foundry") is None, "foundry not installed")
class TestFoundryGraphPersistence(CustomTestCase):
    model = "Qwen/Qwen3.5-2B"
    decode_bs = [1, 2, 3, 4, 5, 6, 7, 8]
    # Empty = prefill graphs disabled; the prefill assertions then expect none.
    prefill_bs = []
    # None = SGLang's default (fa3 on Hopper).
    attention_backend = None
    process = None
    tmp = None
    saved_config_env = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tmp = tempfile.mkdtemp(prefix="foundry_e2e_")
        cls.workspace = os.path.join(cls.tmp, "archive")
        # popen_launch_server merges os.environ, so a caller's export would turn
        # the native engine into a Foundry one.
        cls.saved_config_env = os.environ.pop(_CONFIG_ENV, None)
        if cls.prefill_bs:
            prefill_args = [
                "full",
                "--cuda-graph-bs-prefill",
                *map(str, cls.prefill_bs),
            ]
        else:
            prefill_args = ["disabled"]
        cls.server_args = [
            "--cuda-graph-backend-prefill",
            *prefill_args,
            "--cuda-graph-bs-decode",
            *map(str, cls.decode_bs),
            "--disable-cuda-graph-padding",
            "--mem-fraction-static",
            "0.6",
            "--max-running-requests",
            str(max(cls.decode_bs)),
            "--random-seed",
            "0",
            "--enable-metrics",
        ]
        if cls.attention_backend is not None:
            cls.server_args += ["--attention-backend", cls.attention_backend]
        # Native first: it also fills the JIT caches, which SAVE must not compile
        # inside its capture window.
        cls.runs = {"native": cls._run_engine("native", env=None, bench=True)}
        save_toml = _write_toml(
            os.path.join(cls.tmp, "save.toml"), "save", cls.workspace
        )
        cls.runs["save"] = cls._run_engine(
            "save", env={_CONFIG_ENV: save_toml}, bench=False
        )
        load_toml = _write_toml(
            os.path.join(cls.tmp, "load.toml"), "load", cls.workspace
        )
        cls.runs["load"] = cls._run_engine(
            "load", env={_CONFIG_ENV: load_toml}, bench=True
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
            cls.process = None
        if cls.saved_config_env is not None:
            os.environ[_CONFIG_ENV] = cls.saved_config_env
        if cls.tmp is not None:
            shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def _run_engine(cls, name, env, bench):
        out_path = os.path.join(cls.tmp, f"{name}.out")
        err_path = os.path.join(cls.tmp, f"{name}.err")
        result = {}
        with open(out_path, "w") as out, open(err_path, "w") as err:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=cls.server_args,
                env=env,
                return_stdout_stderr=(out, err),
            )
            try:
                passes_before = _prefill_graph_passes(cls.base_url)
                # One request at a time: every engine sees the same batch composition.
                result["sequential"] = []
                for prompt in _PROMPTS:
                    # A radix hit would shrink the prefill below the smallest bucket.
                    requests.post(
                        cls.base_url + "/flush_cache", timeout=30
                    ).raise_for_status()
                    result["sequential"].append(
                        _generate(cls.base_url, prompt)["output_ids"]
                    )
                result["prefill_passes"] = (
                    _prefill_graph_passes(cls.base_url) - passes_before
                )
                result["batched"] = [
                    r["output_ids"] for r in _generate(cls.base_url, _PROMPTS)
                ]
                if bench:
                    result.update(cls._bench_tpot(name))
            finally:
                terminate_and_kill_process_tree(cls.process)
                cls.process = None
        with open(err_path) as f_err, open(out_path) as f_out:
            result["log"] = f_err.read() + f_out.read()
        return result

    @classmethod
    def _bench_tpot(cls, name):
        # One throwaway bench with the same parameters before the timed one, on
        # every benched engine, so first-request costs stay out of the timed run.
        for run in ("warmup", "timed"):
            args = get_benchmark_args(
                base_url=cls.base_url,
                dataset_name="random-ids",
                tokenizer=cls.model,
                num_prompts=32,
                random_input_len=256,
                random_output_len=128,
                max_concurrency=8,
                seed=0,
            )
            args.output_file = os.path.join(cls.tmp, f"bench_{name}_{run}.jsonl")
            metrics = run_benchmark(args)
        return {
            "median_tpot_ms": metrics["median_tpot_ms"],
            "median_itl_ms": metrics["median_itl_ms"],
        }

    def test_save_writes_archive(self):
        rank_dir = os.path.join(self.workspace, "rank_0")
        decode = glob.glob(os.path.join(rank_dir, "graph_*_FULL_t*.json"))
        prefill = glob.glob(os.path.join(rank_dir, "graph_*_PREFILL_t*.json"))
        self.assertEqual(len(decode), len(self.decode_bs), sorted(decode))
        if self.prefill_bs:
            self.assertGreaterEqual(len(prefill), len(self.prefill_bs), sorted(prefill))
        else:
            self.assertEqual(prefill, [])

    def test_greedy_output_matches_native(self):
        for engine in ("save", "load"):
            for kind in ("sequential", "batched"):
                with self.subTest(engine=engine, kind=kind):
                    self.assertEqual(self.runs[engine][kind], self.runs["native"][kind])

    def test_load_restores_and_replays_every_graph(self):
        log = self.runs["load"]["log"]
        decode = [int(n) for n, _ in _DECODE_LOADED.findall(log)]
        prefill = [int(n) for n, _ in _PREFILL_LOADED.findall(log)]
        archived_prefill = len(
            glob.glob(os.path.join(self.workspace, "rank_0", "graph_*_PREFILL_t*.json"))
        )
        self.assertEqual(decode, [len(self.decode_bs)])
        if not self.prefill_bs:
            self.assertEqual(prefill, [])
            return
        self.assertEqual(prefill, [archived_prefill])
        # A restored prefill graph that the runner never selects would pass the
        # count checks and silently run eager prefills.
        self.assertGreaterEqual(self.runs["native"]["prefill_passes"], len(_PROMPTS))
        self.assertGreaterEqual(self.runs["load"]["prefill_passes"], len(_PROMPTS))

    def test_load_restore_time(self):
        log = self.runs["load"]["log"]
        times = [float(t) for _, t in _DECODE_LOADED.findall(log)]
        times += [float(t) for _, t in _PREFILL_LOADED.findall(log)]
        self.assertTrue(times, "no [Foundry] Loaded line in the LOAD log")
        self.assertLess(sum(times), _RESTORE_BOUND_S, times)

    def test_plugin_active_without_hook_errors(self):
        self.assertIsNone(
            _ACTIVE.search(self.runs["native"]["log"]),
            "plugin active without the env var",
        )
        for engine in ("save", "load"):
            log = self.runs[engine]["log"]
            with self.subTest(engine=engine):
                # Launcher and scheduler each load the plugin.
                self.assertGreaterEqual(len(set(_ACTIVE.findall(log))), 2)
                # An allocation outside the region is only logged by the hook.
                self.assertNotIn("[HOOK] ERROR", log)

    def test_load_tpot_matches_native(self):
        native = self.runs["native"]["median_tpot_ms"]
        load = self.runs["load"]["median_tpot_ms"]
        native_itl = self.runs["native"]["median_itl_ms"]
        load_itl = self.runs["load"]["median_itl_ms"]
        # Median ITL is only logged, not asserted yet: it stayed within 1.5% of
        # native in all 4 quiet-host runs (including the stalled-native one), too
        # few samples to pick a bound.
        print(
            f"median TPOT native {native:.3f} ms, LOAD {load:.3f} ms; "
            f"median ITL native {native_itl:.3f} ms, LOAD {load_itl:.3f} ms"
        )
        # One-sided: fail only when LOAD is slower than native.
        self.assertLessEqual(
            (load - native) / native,
            _TPOT_REL_TOL,
            f"median TPOT native {native:.3f} ms, LOAD {load:.3f} ms",
        )


class TestFoundryGraphPersistencePrefill(TestFoundryGraphPersistence):
    model = "Qwen/Qwen3-1.7B"
    # Tokens per bucket. A prefill replays only when bucket padding is at most 2x,
    # so the smallest bucket must cover the 5-12 token prompts.
    prefill_bs = [8, 16, 32, 64]
    # Full-backend prefill needs capture-stable EXTEND metadata, which flashinfer provides.
    attention_backend = "flashinfer"


if __name__ == "__main__":
    unittest.main(verbosity=3)
