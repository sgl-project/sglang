"""End-to-end test of --cuda-graph-persistence (Foundry CUDA graph save/restore).

Three engines run in sequence on one GPU with the same CUDA graph
sets: native capture, Foundry SAVE (captures and writes the archive) and
Foundry LOAD (rebuilds the graphs from the archive instead of capturing). LOAD
must produce the same greedy tokens as native, restore every archived graph
(the summed restore time is printed, not bounded), and actually run Foundry.
Two cases: test_restore_replays_every_archived_graph and
test_greedy_output_matches_native.

Two configurations: Qwen3.5-2B (hybrid gated DeltaNet) with decode graphs
only, since its full-backend prefill capture fails in SGLang (the GDN
out-of-graph metadata is decode-only); and Qwen3-30B-A3B-FP8 with EAGLE3
speculative decoding plus full-backend prefill graphs, which exercises the
prefill set and the three decode-type graph kinds. Every engine runs with the
environment pins of Foundry's recipes (_ENGINE_ENV); each engine prints its
launch-to-ready time and LOAD its per-set capture-loop numbers.
test_cuda_graph_persistence_nightly.py runs the second configuration without
the pins, as the comparison.

Needs the optional dependency sglang[foundry] (the CUDA CI installs it).

Run:  python3 test/registered/e2e/cuda_graph/persistence/test_cuda_graph_persistence.py
"""

import glob
import os
import re
import shutil
import tempfile
import time
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=560, stage="base-b", runner_config="1-gpu-large")

# The environment Foundry's recipes (recipe/sglang/set_env.sh) give every engine, set
# here for native, SAVE and LOAD alike so the three start from the same pins:
# dynamo off (Foundry pins it on SAVE / LOAD anyway), no DeepGEMM precompile sweep,
# and host-registered pinned buffers (the FlashInfer decode wrapper pins 8 MiB per
# captured shape; registering is ~1.5x faster than one cudaHostAlloc each).
_ENGINE_ENV = {
    "TORCHDYNAMO_DISABLE": "1",
    "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "0",
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
# "Loaded N SGLang graphs" (target decode) or "Loaded N SGLang <kind> graphs" (the
# speculative-decoding kinds target_verify / draft_decode / draft_extend).
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
_PREFILL_PASSES = re.compile(
    r'^sglang:cuda_graph_passes_total\{[^}]*mode="prefill_cuda_graph"[^}]*\}'
    r"\s+([0-9.eE+-]+)$",
    re.MULTILINE,
)


def _write_toml(path: str, workspace: str) -> str:
    # The region is virtual address space for everything allocated after setup,
    # the KV pool included; 256GB covers an 80-192 GB GPU at this memory fraction.
    # No mode key: --cuda-graph-persistence gives it.
    with open(path, "w") as f:
        f.write(
            'base_addr = 0x600000000000\nregion_size = "256GB"\n'
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


class TestFoundryGraphPersistence(CustomTestCase):
    model = "Qwen/Qwen3.5-2B"
    decode_bs = [1, 2, 3, 4, 5, 6, 7, 8]
    # Empty = prefill graphs disabled; the prefill assertions then expect none.
    prefill_bs = []
    # None = SGLang's default (fa3 on Hopper).
    attention_backend = None
    # Decode-type graph kinds SAVE archives and LOAD restores: [None] = the target
    # decode graphs (archive files graph_*_FULL_t*); speculative decoding replaces
    # them by target_verify / draft_decode / draft_extend (graph_*_DECODE-<kind>_t*).
    decode_kinds = [None]
    # Appended to the server args of every engine (subclasses: speculative decoding).
    extra_server_args = []
    # Environment of every engine; the nightly comparison runs without the pins.
    engine_env = _ENGINE_ENV
    process = None
    tmp = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tmp = tempfile.mkdtemp(prefix="foundry_e2e_")
        cls.workspace = os.path.join(cls.tmp, "archive")
        cls.toml = _write_toml(os.path.join(cls.tmp, "foundry.toml"), cls.workspace)
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
            # The batched request reaches the scheduler as one batch, so every
            # engine prefills the same batch composition.
            "--enable-tokenizer-batch-encode",
        ]
        if cls.attention_backend is not None:
            cls.server_args += ["--attention-backend", cls.attention_backend]
        cls.server_args += cls.extra_server_args
        # Native first: it also fills the JIT caches, which SAVE must not compile
        # inside its capture window.
        cls.runs = {"native": cls._run_engine("native", mode=None)}
        cls.runs["save"] = cls._run_engine("save", mode="save")
        cls.runs["load"] = cls._run_engine("load", mode="load")

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
                env=cls.engine_env,
                return_stdout_stderr=(out, err),
            )
            result["ready_s"] = time.perf_counter() - t_launch
            try:
                passes_before = _prefill_graph_passes(cls.base_url)
                # One request at a time, then all prompts as one batch (batch encode).
                result["sequential"] = []
                for prompt in _PROMPTS:
                    # A radix hit would shrink the prefill below the smallest bucket.
                    # timeout: wait for the scheduler to drain the previous
                    # request; without it /flush_cache answers 400 while busy.
                    requests.post(
                        cls.base_url + "/flush_cache",
                        params={"timeout": 30},
                        timeout=60,
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
            finally:
                terminate_and_kill_process_tree(cls.process)
                cls.process = None
        with open(err_path) as f_err, open(out_path) as f_out:
            result["log"] = f_err.read() + f_out.read()
        # Startup numbers for the log: compared across engines (and against the
        # nightly run without the environment pins), not asserted.
        loops = "; ".join(
            f"{kind}: {int(n)} shapes, restore {restore}s, prep {prep}s"
            for kind, _, n, restore, prep in _LOAD_LOOP.findall(result["log"])
        )
        print(
            f"[Foundry e2e] {cls.__name__} {name}: launch -> ready {result['ready_s']:.1f}s"
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

    def _archived_decode_graphs(self, kind):
        pattern = (
            "graph_*_FULL_t*.json" if kind is None else f"graph_*_DECODE-{kind}_t*.json"
        )
        return glob.glob(os.path.join(self.workspace, "rank_0", pattern))

    def test_restore_replays_every_archived_graph(self):
        """Without it a SAVE that archives too few graphs, a LOAD that restores
        fewer than archived or never selects a restored prefill graph, Foundry
        active in the native run or in another mode than the flag's, or a hook
        allocation error all pass the output comparison."""
        rank_dir = os.path.join(self.workspace, "rank_0")
        archived_prefill = glob.glob(os.path.join(rank_dir, "graph_*_PREFILL_t*.json"))
        for kind in self.decode_kinds:
            with self.subTest(archived_kind=kind):
                decode = self._archived_decode_graphs(kind)
                self.assertEqual(len(decode), len(self.decode_bs), sorted(decode))
        if self.prefill_bs:
            self.assertGreaterEqual(
                len(archived_prefill), len(self.prefill_bs), sorted(archived_prefill)
            )
        else:
            self.assertEqual(archived_prefill, [])

        log = self.runs["load"]["log"]
        decode = {kind or None: int(n) for n, kind, _ in _DECODE_LOADED.findall(log)}
        prefill = [int(n) for n, _ in _PREFILL_LOADED.findall(log)]
        # One Loaded line per decode-type kind, each with every archived graph.
        self.assertEqual(
            decode, {kind: len(self.decode_bs) for kind in self.decode_kinds}
        )
        if self.prefill_bs:
            self.assertEqual(prefill, [len(archived_prefill)])
            # A restored prefill graph that the runner never selects would pass the
            # count checks and silently run eager prefills.
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

    def test_greedy_output_matches_native(self):
        for engine in ("save", "load"):
            for kind in ("sequential", "batched"):
                with self.subTest(engine=engine, kind=kind):
                    self.assertEqual(self.runs[engine][kind], self.runs["native"][kind])


class TestFoundryGraphPersistenceSpecPrefill(TestFoundryGraphPersistence):
    """EAGLE3 speculative decoding plus full-backend prefill graphs: the archive
    holds the prefill graphs and the three decode-type kinds the EAGLE worker
    captures (target_verify, draft_decode, draft_extend) in place of the target
    decode graphs, and LOAD restores each set. A small chain (3 steps, top-k 1, 4
    draft tokens) keeps the run short; greedy outputs stay deterministic. Dummy
    weights (--load-format dummy, target and draft) keep CI from downloading
    31 GB: the graphs, their restore and the target's outputs are exercised,
    draft acceptance is not (a random draft accepts no token)."""

    # Ungated target and draft (the default EAGLE3 pair's Llama target is gated).
    model = "Qwen/Qwen3-30B-A3B-FP8"
    attention_backend = "flashinfer"
    decode_bs = [1, 2, 4, 8]
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


if __name__ == "__main__":
    unittest.main(verbosity=3)
