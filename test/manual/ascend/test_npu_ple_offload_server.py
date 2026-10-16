"""Functional PLE parameter coverage on NPU; run this file with python3.

Each successful server is a real TP8 Qwen3.8-Flash-Next load. A test-only import
observer reports the actual table placement at first execution, without changing
weights, IDs, sampling or model calculations. No shared test utility is edited.
The child cache and all table directories are private to each test.

Covers explicit enable/disable, NPU automatic default, default/explicit pinned,
file directory creation/default/restart reuse, argument rejection, filesystem
errors, required UVA and ordinary graphs, sequential decode and
concurrent requests.
This is not a task-accuracy or performance benchmark. A restart reuses files but
the production loader still rewrites their weights; it is not a no-load test.
"""

import argparse
import contextlib
import io
import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import requests
import sympy
import torch

from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
    pad_vocab_size,
)
from sglang.srt.models.qwen4_exp_ple_table import ple_table_file_name
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers import get_config
from sglang.test.ascend.test_ascend_utils import MODEL_WEIGHTS_DIR
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

MODEL = os.path.join(MODEL_WEIGHTS_DIR, "Qwen3.8-Flash-Next")
TP_SIZE = "8"
EXTRA_ARGS = ["--mamba-ssm-dtype", "bfloat16"]
SERVER_LAUNCH_TIMEOUT = max(1800, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)

TABLE_FILE = re.compile(
    r"^ple_table_(?P<dims>[0-9x]+)_(?P<dtype>[a-z0-9_]+)_(?P<nbytes>\d+)B"
    r"_rows(?P<start>\d+)-(?P<end>\d+)\.bin$"
)

# Generated in a private PYTHONPATH directory, so spawn workers inherit the
# observer. Importing the real module still runs its original loader. Only the
# first compute_ngram_ids call reads and reports metadata; its return is intact.
RUNTIME_OBSERVER = r"""
import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys

TARGET = "sglang.srt.models.qwen4_exp"

def mapping_for(pointer):
    with open("/proc/self/maps") as maps:
        for line in maps:
            parts = line.split(None, 5)
            start, end = (int(value, 16) for value in parts[0].split("-"))
            if start <= pointer < end:
                return parts[5].strip() if len(parts) == 6 else None
    return None

class ObserverLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        return self.original.create_module(spec)

    def exec_module(self, module):
        self.original.exec_module(module)
        original = module.Qwen4ExpNGramEmbedding.compute_ngram_ids

        @functools.wraps(original)
        def observed(instance, *args, **kwargs):
            if not getattr(instance, "_ple_test_reported", False):
                embedding = instance.ngram_embedding
                weight = embedding.weight
                cpu = weight.device.type == "cpu"
                data = {
                    "pid": os.getpid(),
                    "tp_rank": module.get_parallel().tp_rank,
                    "layer": instance.ple_layer_index,
                    "embedding_class": type(embedding).__name__,
                    "device": weight.device.type,
                    "pinned": bool(weight.is_pinned()) if cpu else False,
                    "mapping": mapping_for(weight.data_ptr()) if cpu else None,
                    "dtype": str(weight.dtype).removeprefix("torch."),
                    "shape": list(weight.shape),
                    "start": embedding.shard_indices.org_vocab_start_index,
                    "end": embedding.shard_indices.org_vocab_end_index,
                    "npu_uva_enabled": bool(
                        getattr(embedding, "npu_uva_enabled", False)
                    ),
                }
                os.write(1, ("PLE_TEST_TABLE " + json.dumps(data) + "\n").encode())
                instance._ple_test_reported = True
            return original(instance, *args, **kwargs)

        module.Qwen4ExpNGramEmbedding.compute_ngram_ids = observed

    def __getattr__(self, name):
        return getattr(self.original, name)

class ObserverFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == TARGET:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec is not None and spec.loader is not None:
                spec.loader = ObserverLoader(spec.loader)
                return spec
        return None

sys.meta_path.insert(0, ObserverFinder())
"""


def ensure_pinned_memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    if soft == resource.RLIM_INFINITY:
        return
    try:
        resource.setrlimit(
            resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Cannot raise pinned PLE memlock (soft={soft}, hard={hard} bytes); "
            "the host/container must permit raising the locked-memory limit."
        ) from exc


def expected_table_files(dtype: torch.dtype, tp_size: int) -> list[str]:
    text_config = get_config(MODEL, trust_remote_code=True).text_config
    ngram_heads = (text_config.ngram_size - 1) * text_config.heads_per_ngram
    head_dim = text_config.ple_embed_dim // ngram_heads
    divisible_by = text_config.make_ngram_vocab_size_divisible_by
    prime = text_config.ngram_vocab_size_base - 1
    names = []
    for layer_id in sorted(set(text_config.ple_layer_ids)):
        vocab = 0
        for _ in range(ngram_heads):
            prime = int(sympy.nextprime(prime))
            vocab += prime
        vocab = -(-vocab // divisible_by) * divisible_by
        if not 1 <= layer_id <= text_config.num_hidden_layers:
            continue
        padded = pad_vocab_size(vocab)
        for rank in range(tp_size):
            shard = VocabParallelEmbedding._get_indices(
                padded, padded, vocab, vocab, rank, tp_size
            )
            names.append(
                ple_table_file_name(
                    (padded // tp_size, head_dim),
                    dtype,
                    tag=f"rows{shard.org_vocab_start_index}-{shard.org_vocab_end_index}",
                )
            )
    return names


class TestNpuPleOffloadCli(CustomTestCase):
    def parse(self, flags):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        return parser.parse_args(["--model-path", MODEL, *flags])

    def test_cli_defaults_and_explicit_switches(self):
        args = self.parse([])
        self.assertIsNone(args.ple_offload_embedding)
        self.assertEqual(args.ple_offload_backend, "pinned")
        self.assertIsNone(args.ple_offload_dir)
        self.assertIs(
            self.parse(["--ple-offload-embedding"]).ple_offload_embedding, True
        )
        self.assertIs(
            self.parse(["--no-ple-offload-embedding"]).ple_offload_embedding, False
        )

    def test_invalid_backend_rejected_by_cli(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(
            SystemExit
        ) as result:
            self.parse(["--ple-offload-backend", "invalid-backend"])
        self.assertEqual(result.exception.code, 2)
        self.assertIn("invalid choice", stderr.getvalue())
        self.assertIn("--ple-offload-backend", stderr.getvalue())


class TestNpuPleOffloadServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.isdir(MODEL):
            raise FileNotFoundError(
                f"Model weights not found in shared directory: {MODEL}"
            )
        cls.base_url = os.environ.get("SGLANG_TEST_PLE_URL", DEFAULT_URL_FOR_TEST)

    def setUp(self):
        self.process = None
        self.owns_process_group = False
        self.work_dir = tempfile.mkdtemp(prefix="ple_server_test_")
        self.addCleanup(shutil.rmtree, self.work_dir, ignore_errors=True)
        self.table_dir = os.path.join(self.work_dir, "tables")
        os.mkdir(self.table_dir)
        self.cache_dir = os.path.join(self.work_dir, "cache")
        self.observer_dir = os.path.join(self.work_dir, "observer")
        os.mkdir(self.observer_dir)
        Path(self.observer_dir, "sitecustomize.py").write_text(
            RUNTIME_OBSERVER, encoding="utf-8"
        )
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def tearDown(self):
        self.stop_server()

    def stop_server(self):
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                kill_process_tree(process.pid)
            if self.owns_process_group:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait(timeout=30)
        finally:
            self.process = None
            self.owns_process_group = False

    def common_args(self):
        return [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--tp-size",
            TP_SIZE,
            "--mem-fraction-static",
            "0.85",
            # These functional requests run singly or with four-way concurrency.
            # Keep decode graphs enabled without capturing 52 A5 default sizes.
            "--cuda-graph-max-bs-decode",
            "4",
            *EXTRA_ARGS,
        ]

    @contextlib.contextmanager
    def isolated_cache(self):
        # Internally isolate the default-directory test; the caller exports nothing.
        with patch.dict(os.environ, {"SGLANG_CACHE_DIR": self.cache_dir}):
            os.environ.pop("SGLANG_QWEN4_PLE_FILE_DIR", None)
            yield

    def launch(self, backend=None, *, enabled=True, table_dir=None):
        if enabled is True and backend in (None, "pinned"):
            ensure_pinned_memory_limit()
        args = self.common_args()
        if enabled is not None:
            args += [
                "--ple-offload-embedding" if enabled else "--no-ple-offload-embedding"
            ]
        if backend is not None:
            args += ["--ple-offload-backend", backend]
        if table_dir is not None:
            args += ["--ple-offload-dir", table_dir]
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        env = {
            "PYTHONPATH": os.pathsep.join(
                part
                for part in (self.observer_dir, os.environ.get("PYTHONPATH", ""))
                if part
            )
        }
        with self.isolated_cache():
            self.process = popen_launch_server(
                MODEL,
                self.base_url,
                timeout=SERVER_LAUNCH_TIMEOUT,
                other_args=args,
                env=env,
                return_stdout_stderr=(self.stdout, self.stderr),
            )

    def log(self):
        return self.stdout.getvalue() + "\n" + self.stderr.getvalue()

    def server_info(self):
        response = requests.get(f"{self.base_url}/server_info", timeout=30)
        self.assertEqual(response.status_code, 200)
        return response.json()

    def generate(self, prompt, *, force_decode=True):
        params = {"temperature": 0.0, "max_new_tokens": 16}
        if force_decode:
            params["ignore_eos"] = True
        response = requests.post(
            f"{self.base_url}/generate",
            json={"text": prompt, "sampling_params": params},
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        output = response.json()
        self.assertTrue(output["output_ids"])
        if force_decode:
            self.assertEqual(output["meta_info"]["completion_tokens"], 16)
            self.assertEqual(len(output["output_ids"]), 16)
        else:
            self.assertGreaterEqual(output["meta_info"]["completion_tokens"], 1)
        return output

    def check_runtime_table(self, enabled, backend, directory=None):
        records = []
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            records = [
                json.loads(match[1])
                for match in re.finditer(r"PLE_TEST_TABLE (\{[^\n]*\})", self.log())
            ]
            if records:
                dtype = getattr(torch, records[0]["dtype"])
                expected_count = len(expected_table_files(dtype, int(TP_SIZE)))
                if len(records) == expected_count:
                    break
            time.sleep(0.1)
        self.assertTrue(records, "No real-worker PLE table observation was recorded")
        self.assertEqual(len(records), expected_count)
        self.assertEqual(
            len({(r["layer"], r["start"], r["end"]) for r in records}), expected_count
        )
        for record in records:
            if enabled:
                self.assertEqual(
                    record["embedding_class"], "Qwen4ExpPinnedHostEmbedding"
                )
                self.assertEqual(record["device"], "cpu")
                # A registered file mapping can also report is_pinned=True.
                # Validate file storage through its real /proc/self/maps path.
                if backend == "pinned":
                    self.assertTrue(record["pinned"])
                self.assertTrue(record["npu_uva_enabled"])
                self.assertEqual(record["dtype"], "bfloat16")
                if backend == "file":
                    self.assertIsNotNone(record["mapping"])
                    self.assertEqual(
                        os.path.dirname(os.path.realpath(record["mapping"])),
                        os.path.realpath(directory),
                    )
                    self.assertRegex(os.path.basename(record["mapping"]), TABLE_FILE)
            else:
                self.assertEqual(record["embedding_class"], "VocabParallelEmbedding")
                self.assertEqual(record["device"], "npu")
                self.assertFalse(record["npu_uva_enabled"])
        self.assertEqual(
            {int(record["tp_rank"]) for record in records}, set(range(int(TP_SIZE)))
        )
        captures = re.findall(
            r"TP(\d+)\].*Capture target decode NPU graph end\.", self.log()
        )
        self.assertEqual(set(map(int, captures)), set(range(int(TP_SIZE))))
        routes = re.findall(
            r"TP(\d+)\].*NPU PLE lookup route: uva \(backend=(pinned|file)\)",
            self.log(),
        )
        expected_routes = Counter(
            (int(record["tp_rank"]), backend) for record in records if enabled
        )
        self.assertEqual(
            Counter((int(rank), actual_backend) for rank, actual_backend in routes),
            expected_routes,
        )

    def check_serving(self, enabled=True, backend="pinned", directory=None):
        info = self.server_info()
        self.assertIs(info["ple_offload_embedding"], enabled)
        self.assertEqual(info["ple_offload_backend"], backend)
        self.generate("The capital of France is")
        self.generate("Count from one to twenty: " * 64, force_decode=False)
        self.check_runtime_table(enabled, backend, directory)

    def check_table_layout(self, directory):
        files = sorted(os.listdir(directory))
        self.assertTrue(files)
        dtypes = set()
        for name in files:
            match = TABLE_FILE.fullmatch(name)
            self.assertIsNotNone(match, f"Unexpected table file: {name}")
            dtypes.add(match["dtype"])
            stat = os.stat(os.path.join(directory, name))
            self.assertEqual(stat.st_size, int(match["nbytes"]))
            self.assertGreater(stat.st_blocks * 512, stat.st_size // 2)
        self.assertEqual(len(dtypes), 1)
        expected = expected_table_files(getattr(torch, dtypes.pop()), int(TP_SIZE))
        self.assertEqual(len(expected), len(set(expected)))
        self.assertEqual(files, sorted(expected))

    def reject_startup(self, ple_args, expected_message):
        _, host, port = self.base_url.split(":")
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            MODEL,
            *self.common_args(),
            "--device",
            "npu",
            "--host",
            host[2:],
            "--port",
            port,
            *ple_args,
        ]
        with self.isolated_cache():
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        self.owns_process_group = True
        try:
            output, _ = self.process.communicate(timeout=180)
            print(output, flush=True)
            self.assertNotEqual(self.process.returncode, 0)
            self.assertIn(expected_message, output)
        finally:
            self.stop_server()

    def test_default_ple_switch_is_disabled_on_npu(self):
        self.launch(enabled=None)
        self.check_serving(enabled=False)

    def test_explicit_disabled_ple_stays_on_npu(self):
        self.launch(enabled=False)
        self.check_serving(enabled=False)

    def test_default_backend_is_pinned(self):
        self.launch()
        self.check_serving()

    def test_pinned_backend_serves(self):
        # dir must not turn pinned storage into file storage.
        self.launch("pinned", table_dir=self.table_dir)
        self.check_serving()
        self.assertEqual(os.listdir(self.table_dir), [])
        prompts = [
            "Explain photosynthesis.",
            "计算 17 加 28。",
            "List three colors.",
            "What is an atom?",
        ]
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            outputs = list(executor.map(self.generate, prompts))
        self.assertEqual(len(outputs), len(prompts))

    def test_file_backend_serves_and_reuses_directory(self):
        self.launch("file", table_dir=self.table_dir)
        self.check_serving(backend="file", directory=self.table_dir)
        self.assertEqual(self.server_info()["ple_offload_dir"], self.table_dir)
        self.check_table_layout(self.table_dir)
        files = {
            name: (
                os.stat(os.path.join(self.table_dir, name)).st_ino,
                os.stat(os.path.join(self.table_dir, name)).st_size,
            )
            for name in os.listdir(self.table_dir)
        }
        self.stop_server()
        self.launch("file", table_dir=self.table_dir)
        self.check_serving(backend="file", directory=self.table_dir)
        self.check_table_layout(self.table_dir)
        self.assertEqual(
            files,
            {
                name: (
                    os.stat(os.path.join(self.table_dir, name)).st_ino,
                    os.stat(os.path.join(self.table_dir, name)).st_size,
                )
                for name in os.listdir(self.table_dir)
            },
        )

    def test_file_backend_table_layout_and_directory_creation(self):
        directory = os.path.join(self.work_dir, "not_created", "nested_tables")
        self.assertFalse(os.path.exists(directory))
        self.launch("file", table_dir=directory)
        self.check_serving(backend="file", directory=directory)
        self.assertEqual(self.server_info()["ple_offload_dir"], directory)
        self.assertTrue(os.path.isdir(directory))
        self.check_table_layout(directory)

    def test_file_backend_default_directory(self):
        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", MODEL.rstrip("/")).strip("_")
        directory = os.path.join(self.cache_dir, "ple", safe_model or "model")
        self.assertFalse(os.path.exists(directory))
        self.launch("file")
        self.check_serving(backend="file", directory=directory)
        self.assertIsNone(self.server_info()["ple_offload_dir"])
        self.check_table_layout(directory)

    def test_incompatible_offload_arguments_rejected(self):
        cases = [
            (
                ["--no-ple-offload-embedding", "--ple-offload-backend", "file"],
                "file requires --ple-offload-embedding",
            ),
            (
                ["--ple-offload-embedding", "--cpu-offload-gb", "1"],
                "--ple-offload-embedding cannot be combined",
            ),
            (
                ["--ple-offload-embedding", "--offload-group-size", "1"],
                "--ple-offload-embedding cannot be combined",
            ),
        ]
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                self.reject_startup(arguments, message)

    def test_file_directory_occupied_by_regular_file_rejected(self):
        path = os.path.join(self.work_dir, "occupied")
        Path(path).write_text("keep this file", encoding="utf-8")
        self.reject_startup(
            [
                "--ple-offload-embedding",
                "--ple-offload-backend",
                "file",
                "--ple-offload-dir",
                path,
            ],
            "FileExistsError",
        )
        self.assertEqual(Path(path).read_text(encoding="utf-8"), "keep this file")

    def test_unwritable_file_directory_rejected(self):
        # chmod alone is ineffective for root; procfs cannot create table dirs.
        path = os.path.join("/proc", os.path.basename(self.work_dir))
        self.reject_startup(
            [
                "--ple-offload-embedding",
                "--ple-offload-backend",
                "file",
                "--ple-offload-dir",
                path,
            ],
            "FileNotFoundError",
        )
        self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
