"""End-to-end test of --cuda-graph-persistence: native, SAVE and LOAD on one GPU;
LOAD restores every archived graph and matches native's greedy tokens."""

import atexit
import glob
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from importlib import metadata
from typing import Optional

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.perf_bench_kit import check_perf, reported
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=560, stage="base-b", runner_config="1-gpu-large")

# The env of Foundry's recipes (recipe/sglang/set_env.sh),
# set for native, SAVE and LOAD alike so the three start from the same pins.
_ENGINE_ENV = {
    # Foundry pins it on SAVE / LOAD anyway.
    "TORCHDYNAMO_DISABLE": "1",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
    # The FlashInfer decode wrapper pins 8 MiB per captured shape;
    # registering is ~1.5x faster than one cudaHostAlloc each.
    "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF",
        "pinned_use_cuda_host_register:True,pinned_num_register_threads:8",
    ),
}
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

_ACTIVE = re.compile(r"\[Foundry\] cuda graph persistence active: pid=(\d+) mode=(\w+)")
# "Loaded N SGLang graphs" (target decode) or "Loaded N SGLang <kind> graphs",
# <kind> being target_verify, draft_decode or draft_extend.
_DECODE_LOADED = re.compile(
    r"\[Foundry\] Loaded (\d+) SGLang (?!prefill )(?:(\w+) )?graphs in ([0-9.]+)s"
)
_PREFILL_LOADED = re.compile(
    r"\[Foundry\] Loaded (\d+) SGLang prefill graphs in ([0-9.]+)s"
)
# LOAD's capture loop per graph set: wall, shapes, restore and SGLang's per-shape prep.
_LOAD_LOOP = re.compile(
    r"\[Foundry\] SGLang (\w+) capture loop on LOAD: ([0-9.]+)s \((\d+) shapes, "
    r"restore ([0-9.]+)s, per-shape prep ([0-9.]+)s\)"
)
# The decode-type batch sizes SGLang captured, as rank 0 prints them;
# the EP and attention-TP layouts clamp or align the requested list.
_CAPTURED_BS = re.compile(
    r"Capture target (?:decode|verify) CUDA graph end\. bs=\[([0-9, ]*)\]"
)
# Rank 0 prints the first only when a layout has such state (EP dispatchers, TP gatherers).
_PREBUILT_STATE = re.compile(
    r"Prebuilt capture state: (\d+) MoE dispatchers, (\d+) logits gatherers\."
)
_CAPTURE_BEGIN = re.compile(r"Capture [\w ]*CUDA graph begin\.")
# Multi-GPU ranks share one log; each line's prefix names its process.
_LOG_PID = re.compile(r" PID\d+\]")
# SGLang's own startup timers (the launcher prints them once the schedulers are up).
_STARTUP_TIMINGS = re.compile(r"Engine startup timings \(s\): (.*)")
_STARTUP_FIELDS = re.compile(r"scheduler_e2e=([0-9.]+), cuda_graph=\{([^}]*)\}")
# Hybrid GDN models compile their prefill kernels in process on every start, LOAD included.
_GDN_PREWARM = re.compile(
    r"Prewarmed FlashInfer GDN prefill with \d+ runs in ([0-9.]+) s\."
)
_PREFILL_PASSES = re.compile(
    r'^sglang:cuda_graph_passes_total\{[^}]*mode="prefill_cuda_graph"[^}]*\}'
    r"\s+([0-9.eE+-]+)$",
    re.MULTILINE,
)


# Keep in sync with the foundry extra in python/pyproject.toml.
_FOUNDRY_REQUIREMENT = "foundry-core>=0.1.0rc5,<0.2"
# CI installs foundry-core in setUpClass (_foundry_env), after this guard is evaluated.
_HAS_FOUNDRY = is_in_ci() or importlib.util.find_spec("foundry") is not None
_foundry_pythonpath: Optional[str] = None


def _foundry_env() -> dict:
    # Server env importing Foundry; in CI a scratch --target dir on PYTHONPATH,
    # since CI runners share site-packages between jobs.
    global _foundry_pythonpath
    if not is_in_ci():
        return {}
    if _foundry_pythonpath is None:
        target = tempfile.mkdtemp(prefix="foundry_core_")
        atexit.register(shutil.rmtree, target, ignore_errors=True)
        install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-deps", "--target"]
            + [target, _FOUNDRY_REQUIREMENT]
        )
        if install.returncode != 0:
            _skip_in_job_log(f"pip could not install {_FOUNDRY_REQUIREMENT}")
        _skip_unless_built_for_this_torch(target)
        _foundry_pythonpath = target
    return {
        "PYTHONPATH": os.pathsep.join(
            filter(None, [_foundry_pythonpath, os.environ.get("PYTHONPATH")])
        )
    }


def _skip_unless_built_for_this_torch(target: str) -> None:
    # A foundry-core wheel requires torch==<its build torch>; --no-deps skips that check.
    import torch
    from packaging.requirements import Requirement

    (dist,) = metadata.distributions(name="foundry-core", path=[target])
    runner = torch.__version__.split("+")[0]
    for req in map(Requirement, dist.requires or []):
        if req.name == "torch" and not req.specifier.contains(runner):
            _skip_in_job_log(
                f"foundry-core {dist.version} requires {req}, this runner has "
                f"torch {runner}"
            )


def _skip_in_job_log(reason: str) -> None:
    # Visible in the job log, not only in the skip summary.
    print(f"[Foundry e2e] SKIPPED: {reason}", file=sys.stderr, flush=True)
    raise unittest.SkipTest(reason)


def _write_toml(path: str, workspace: str) -> str:
    # The region is VA space for every allocation after setup, the KV pool included;
    # 256GB covers an 80-192 GB GPU at this memory fraction; the flag gives the mode.
    with open(path, "w") as f:
        f.write(
            'base_addr = 0x600000000000\nregion_size = "256GB"\n'
            f'workspace_root = "{workspace}"\nscratch_space_size = "1024MB"\n'
        )
    return path


def _line_prefix(log: str, match: re.Match) -> str:
    return log[log.rfind("\n", 0, match.start()) + 1 : match.start()]


def _capture_begin_of(log: str, prebuilt: re.Match) -> Optional[re.Match]:
    # The first capture-begin line from the process that printed `prebuilt`;
    # another rank's line can land first in the shared log.
    pid = _LOG_PID.search(_line_prefix(log, prebuilt))
    for capture in _CAPTURE_BEGIN.finditer(log):
        if pid is None or pid.group(0) in _line_prefix(log, capture):
            return capture
    return None


def _startup_seconds(log: str):
    # (scheduler init, time in the runners' capture loops over every graph set),
    # from the engine's startup-timings line; None without one.
    match = _STARTUP_FIELDS.search(log)
    if match is None:
        return None
    graph_s = sum(float(item.split("=")[1]) for item in match.group(2).split(", "))
    return float(match.group(1)), graph_s


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


@unittest.skipUnless(
    _HAS_FOUNDRY, "Foundry is not installed (pip install 'sglang[foundry]')"
)
class TestFoundryGraphPersistence(CustomTestCase):
    # Hybrid gated DeltaNet; SGLang's full-backend prefill capture fails for it,
    # the GDN out-of-graph metadata being decode-only.
    model = "Qwen/Qwen3.5-2B"
    # Every decode batch size up to 128; EP / attention-TP layouts clamp or align it.
    decode_bs = list(range(1, 129))
    # Empty = prefill graphs disabled; the prefill assertions then expect none.
    prefill_bs = []
    # None = SGLang's default (fa3 on Hopper).
    attention_backend = None
    # Decode-type graph kinds SAVE archives and LOAD restores: [None] = target decode;
    # speculative decoding replaces it by target_verify, draft_decode, draft_extend.
    decode_kinds = [None]
    # Appended to the server args of every engine (subclasses: speculative decoding).
    extra_server_args = []
    # Environment of every engine; the nightly comparison runs without the pins.
    engine_env = _ENGINE_ENV
    # Added to engine_env (subclasses: the EP rows' NVSHMEM / single-node pins).
    extra_env = {}
    # Batched outputs are compared with native unless a subclass opts out;
    # DP attention composes batches per rank, which differs run to run natively.
    compare_batched = True
    # LOAD's capture loops against the same run's native ones;
    # restore is a few percent, SGLang's per-shape prep is the rest.
    load_capture_ratio = 0.25
    # Skip, not fail, a layout whose native engine cannot start on this machine,
    # such as DeepEP without a usable NVSHMEM transport.
    native_failure_skips = False
    process = None
    tmp = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Every engine, native included, so the three see the same environment.
        cls.foundry_env = _foundry_env()
        cls.tmp = tempfile.mkdtemp(prefix="foundry_e2e_")
        cls.workspace = os.path.join(cls.tmp, "archive")
        cls.toml = _write_toml(
            path=os.path.join(cls.tmp, "foundry.toml"), workspace=cls.workspace
        )
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
            # The batched request reaches the scheduler as one batch,
            # so every engine prefills the same batch composition.
            "--enable-tokenizer-batch-encode",
        ]
        if cls.attention_backend is not None:
            cls.server_args += ["--attention-backend", cls.attention_backend]
        cls.server_args += cls.extra_server_args
        # Native first: it also fills the JIT caches,
        # which SAVE must not compile inside its capture window.
        try:
            cls.runs = {"native": cls._run_engine(name="native", mode=None)}
        except Exception as e:
            if not cls.native_failure_skips:
                raise
            # setUpClass raised: unittest skips tearDownClass, so clean up here.
            tail = ""
            err_path = os.path.join(cls.tmp, "native.err")
            if os.path.exists(err_path):
                with open(err_path) as f:
                    tail = "".join(f.readlines()[-8:])
            cls.tearDownClass()
            raise unittest.SkipTest(
                f"{cls.__name__}: the native engine of this layout does not start "
                f"here ({e}); log tail:\n{tail}"
            ) from e
        cls.runs["save"] = cls._run_engine(name="save", mode="save")
        cls.runs["load"] = cls._run_engine(name="load", mode="load")

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
            cls.process = None
        if cls.tmp is not None:
            shutil.rmtree(cls.tmp, ignore_errors=True)

    @classmethod
    def _run_engine(cls, name, mode):
        out_path = os.path.join(cls.tmp, f"{name}.out")
        err_path = os.path.join(cls.tmp, f"{name}.err")
        result = {}
        with open(out_path, "w") as out, open(err_path, "w") as err:
            t_launch = time.perf_counter()
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=cls.server_args + cls._persistence_args(mode),
                env={**cls.engine_env, **cls.extra_env, **cls.foundry_env},
                return_stdout_stderr=(out, err),
            )
            result["ready_s"] = time.perf_counter() - t_launch
            try:
                passes_before = _prefill_graph_passes(cls.base_url)
                # One request at a time, then all prompts as one batch (batch encode).
                result["sequential"] = []
                for prompt in _PROMPTS:
                    # A radix hit would shrink the prefill below the smallest bucket.
                    requests.post(
                        cls.base_url + "/flush_cache",
                        # Wait for the scheduler to drain the previous request;
                        # without it /flush_cache answers 400 while busy.
                        params={"timeout": 30},
                        timeout=60,
                    ).raise_for_status()
                    result["sequential"].append(
                        _generate(base_url=cls.base_url, text=prompt)["output_ids"]
                    )
                result["prefill_passes"] = (
                    _prefill_graph_passes(cls.base_url) - passes_before
                )
                result["batched"] = [
                    r["output_ids"]
                    for r in _generate(base_url=cls.base_url, text=_PROMPTS)
                ]
            finally:
                terminate_and_kill_process_tree(cls.process)
                cls.process = None
        with open(err_path) as f_err, open(out_path) as f_out:
            result["log"] = f_err.read() + f_out.read()
        # Startup numbers for the log, not asserted;
        # compared across engines and against the nightly run without the env pins.
        loops = "; ".join(
            f"{kind}: {int(n)} shapes, restore {restore}s, prep {prep}s"
            for kind, _, n, restore, prep in _LOAD_LOOP.findall(result["log"])
        )
        # The launch wait polls /health every 10 s; the engine's own timers are exact.
        timings = _STARTUP_TIMINGS.search(result["log"])
        startup = (
            timings.group(1)
            if timings
            else f"launch -> ready {result['ready_s']:.0f}s (10 s poll)"
        )
        print(
            f"[Foundry e2e] {cls.__name__} {name}: {startup}"
            + (f"; LOAD capture loop {loops}" if loops else "")
        )
        return result

    @classmethod
    def _persistence_args(cls, mode):
        if mode is None:
            return []
        return [
            "--cuda-graph-persistence",
            mode,
            "--cuda-graph-persistence-config",
            cls.toml,
        ]

    def _captured_decode_bs(self):
        match = _CAPTURED_BS.search(self.runs["native"]["log"])
        if match is None:
            return list(self.decode_bs)
        return [int(b) for b in match.group(1).split(",") if b.strip()]

    def _archived_decode_graphs(self, kind):
        pattern = (
            "graph_*_FULL_t*.json" if kind is None else f"graph_*_DECODE-{kind}_t*.json"
        )
        return glob.glob(os.path.join(self.workspace, "rank_0", pattern))

    def test_restore_replays_every_archived_graph(self):
        """Catches too few graphs archived or restored, an unused prefill graph,
        Foundry in the wrong engine or mode, or a hook error; outputs miss these."""
        rank_dir = os.path.join(self.workspace, "rank_0")
        archived_prefill = glob.glob(os.path.join(rank_dir, "graph_*_PREFILL_t*.json"))
        # SGLang clamps or aligns the requested decode batch sizes per layout,
        # so the archive must hold what native captured, one graph per kind and size.
        captured_bs = self._captured_decode_bs()
        for kind in self.decode_kinds:
            with self.subTest(archived_kind=kind):
                decode = self._archived_decode_graphs(kind)
                self.assertEqual(
                    len(decode), len(captured_bs), (captured_bs, sorted(decode))
                )
        if self.prefill_bs:
            self.assertGreaterEqual(
                len(archived_prefill), len(self.prefill_bs), sorted(archived_prefill)
            )
        else:
            self.assertEqual(archived_prefill, [])

        log = self.runs["load"]["log"]
        decode = {kind or None: int(n) for n, kind, _ in _DECODE_LOADED.findall(log)}
        # One Loaded line per rank: the set of counts.
        prefill = sorted({int(n) for n, _ in _PREFILL_LOADED.findall(log)})
        # One Loaded line per decode-type kind, each with every archived graph.
        self.assertEqual(decode, {kind: len(captured_bs) for kind in self.decode_kinds})
        if self.prefill_bs:
            self.assertEqual(prefill, [len(archived_prefill)])
            # Count checks pass even if the runner never picks a restored prefill graph;
            # the prefills would then silently run eagerly.
            self.assertGreaterEqual(
                self.runs["native"]["prefill_passes"], len(_PROMPTS)
            )
            self.assertGreaterEqual(self.runs["load"]["prefill_passes"], len(_PROMPTS))
        else:
            self.assertEqual(prefill, [])

        times = [float(t) for _, _, t in _DECODE_LOADED.findall(log)]
        times += [float(t) for _, t in _PREFILL_LOADED.findall(log)]
        self.assertTrue(times, "no [Foundry] Loaded line in the LOAD log")
        print(f"[Foundry] LOAD graph restore total: {sum(times):.3f}s {times}")

        self.assertIsNone(
            _ACTIVE.search(self.runs["native"]["log"]),
            "Foundry active without --cuda-graph-persistence",
        )
        for engine in ("save", "load"):
            engine_log = self.runs[engine]["log"]
            with self.subTest(active_engine=engine):
                active = _ACTIVE.findall(engine_log)
                # Launcher (resolution) and scheduler each activate it, in the flag's mode.
                self.assertGreaterEqual(len({pid for pid, _ in active}), 2)
                self.assertEqual({mode for _, mode in active}, {engine})
                # An allocation outside the region is only logged by the hook.
                self.assertNotIn("[HOOK] ERROR", engine_log)

    def test_capture_state_is_built_before_capture(self):
        """Catches the lazy comm state built after the first capture began,
        or built on one of SAVE / LOAD only; one layout's outputs can miss both."""
        # Without persistence the first forward builds it, so native prints no line.
        self.assertEqual(_PREBUILT_STATE.findall(self.runs["native"]["log"]), [])
        # What is prebuilt follows the layout and the machine (multicast), not a constant.
        save = _PREBUILT_STATE.findall(self.runs["save"]["log"])
        self.assertEqual(_PREBUILT_STATE.findall(self.runs["load"]["log"]), save)
        for engine in ("save", "load"):
            log = self.runs[engine]["log"]
            with self.subTest(engine=engine):
                prebuilt = _PREBUILT_STATE.search(log)
                if prebuilt is not None:
                    capture = _capture_begin_of(log, prebuilt)
                    self.assertIsNotNone(capture, "no capture-begin line in the log")
                    self.assertLess(prebuilt.start(), capture.start())

    def test_load_startup_stays_bounded(self):
        """Catches LOAD paying capture work again (a warmup forward per shape,
        a compile inside the capture loop), which the output and count checks miss."""
        native, load = (
            _startup_seconds(self.runs[engine]["log"]) for engine in ("native", "load")
        )
        self.assertIsNotNone(native, "no startup-timings line in the native log")
        self.assertIsNotNone(load, "no startup-timings line in the LOAD log")
        # Not persistence's time; drop once these kernels are cached on disk.
        native_init_s, load_init_s = (
            times[0] - sum(map(float, _GDN_PREWARM.findall(self.runs[engine]["log"])))
            for engine, times in (("native", native), ("load", load))
        )
        check_perf(
            self,
            reported("load_cuda_graph", load[1], unit="s"),
            reported("load_scheduler_e2e", load_init_s, unit="s"),
            reported("native_cuda_graph", native[1], unit="s"),
            reported("native_scheduler_e2e", native_init_s, unit="s"),
            suffix=f"_{type(self).__name__}",
        )
        # The same run's native engine is the reference, whatever the machine:
        # it warms every shape up twice and captures it once.
        self.assertLess(load[1], self.load_capture_ratio * native[1])
        self.assertLess(load_init_s, native_init_s)

    def test_greedy_output_matches_native(self):
        for engine in ("save", "load"):
            kinds = (
                ("sequential", "batched") if self.compare_batched else ("sequential",)
            )
            for kind in kinds:
                with self.subTest(engine=engine, kind=kind):
                    self.assertEqual(self.runs[engine][kind], self.runs["native"][kind])


class TestFoundryGraphPersistenceSpecPrefill(TestFoundryGraphPersistence):
    """EAGLE3 with full prefill graphs: the prefill set plus three decode-type kinds;
    dummy weights avoid a 31 GB download, so draft acceptance is not exercised."""

    # Ungated target and draft (the default EAGLE3 pair's Llama target is gated).
    model = "Qwen/Qwen3-30B-A3B-FP8"
    attention_backend = "flashinfer"
    # Tokens per bucket. A prefill replays only when bucket padding is at most 2x,
    # so the smallest bucket must cover the 5-12 token prompts.
    prefill_bs = [8, 16, 32, 64]
    decode_kinds = ["target_verify", "draft_decode", "draft_extend"]
    extra_server_args = [
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        "Tengyunw/qwen3_30b_moe_eagle3",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--load-format",
        "dummy",
    ]
    # Prefill's per-shape prep runs on LOAD too and is most of its capture loop.
    load_capture_ratio = 0.5


if __name__ == "__main__":
    unittest.main(verbosity=3)
