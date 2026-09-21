"""Validate --radix-cache-backend with Llama-3.2-1B-Instruct on one NPU.

Run from the repository root in an installed SGLang Ascend environment:
    ASCEND_RT_VISIBLE_DEVICES=0 \
    SGLANG_TEST_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
    SGLANG_TEST_LOG_DIR=/tmp/npu-radix-backend \
    python test/registered/npu/basic_function/parameter/test_npu_radix_cache_backend.py -v

Uses the Ascend model-path convention and the cold/warm cache assertions in
test/registered/radix_cache/test_radix_cache_hit.py. The dedicated registry
tests in test/registered/unit/mem_cache/test_registry.py cover invalid names
and duplicate registration; this test exercises actual NPU serving.
"""

import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import requests
from transformers import AutoConfig, AutoTokenizer

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.srt.utils.network import get_open_port
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=240, suite="full-1-npu-a3", nightly=True)

BACKEND_NAME = "ascend_test_radix"
PLUGIN_NAME = "ascend_radix_regression"


@unittest.skipUnless(is_npu(), "Requires Ascend NPU")
class TestNpuRadixCacheBackend(CustomTestCase):
    """[Test Category] Parameter; [Test Target] --radix-cache-backend."""

    PAGE_SIZE = 128
    PREFIX_LEN = 512
    SUFFIX_LEN = 256
    OUTPUT_LEN = 8

    def setUp(self):
        super().setUp()
        self.model = os.environ.get(
            "SGLANG_TEST_MODEL_PATH", LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        )
        self.assertEqual(
            AutoConfig.from_pretrained(self.model).model_type,
            "llama",
            "This fixture requires a Llama full-attention model.",
        )
        temporary = tempfile.TemporaryDirectory(prefix="npu-radix-backend-")
        self.addCleanup(temporary.cleanup)
        self.work_dir = Path(temporary.name)
        self.log_dir = Path(os.environ.get("SGLANG_TEST_LOG_DIR", temporary.name))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)

        # importlib.metadata discovers this fixture in every spawned worker.
        # Both launches load the plugin; only the second selects its factory.
        dist = self.work_dir / "ascend_radix_regression-0.0.0.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text(
            "Metadata-Version: 2.1\nName: ascend-radix-regression\nVersion: 0.0.0\n",
            encoding="utf-8",
        )
        (dist / "entry_points.txt").write_text(
            "[sglang.srt.plugins]\n"
            f"{PLUGIN_NAME} = sglang.test.ascend.radix_cache_backend:register\n",
            encoding="utf-8",
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model)
        prefix = tokenizer.encode(
            "This document describes a library, its books, and its readers. " * 100,
            add_special_tokens=False,
        )[: self.PREFIX_LEN]
        suffix_a = tokenizer.encode(
            "Alpha readers enjoy science and discuss discoveries. " * 100,
            add_special_tokens=False,
        )[: self.SUFFIX_LEN]
        suffix_b = tokenizer.encode(
            "Beta readers enjoy history and discuss ancient cities. " * 100,
            add_special_tokens=False,
        )[: self.SUFFIX_LEN]
        self.assertEqual(len(prefix), self.PREFIX_LEN)
        self.assertEqual(len(suffix_a), self.SUFFIX_LEN)
        self.assertEqual(len(suffix_b), self.SUFFIX_LEN)
        self.assertNotEqual(suffix_a[0], suffix_b[0])
        self.prompt_a = prefix + suffix_a
        self.prompt_b = prefix + suffix_b

    def _request(self, method, path, **kwargs):
        response = self.session.request(
            method, self.base_url + path, timeout=180, **kwargs
        )
        self.assertEqual(response.status_code, 200, response.text)
        return response

    def _save(self, name, value):
        (self.log_dir / f"{name}.json").write_text(
            json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @contextmanager
    def _server(self, backend):
        label = backend or "default"
        self.base_url = f"http://127.0.0.1:{get_open_port()}"
        # CustomTestCase may retry the test method without rerunning setUp.
        events_dir = Path(tempfile.mkdtemp(prefix=f"{label}-", dir=self.work_dir))
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(self.work_dir), os.environ.get("PYTHONPATH")))
            ),
            "SGLANG_PLUGINS": PLUGIN_NAME,
            "SGLANG_TEST_RADIX_EVENT_DIR": str(events_dir),
            "SGLANG_EXPERIMENTAL_CPP_RADIX_TREE": "0",
            "SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND": "python",
            "TOKENIZERS_PARALLELISM": "false",
        }
        args = [
            "--device",
            "npu",
            "--attention-backend",
            "ascend",
            "--dtype",
            "bfloat16",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.3",
            "--context-length",
            "2048",
            "--max-total-tokens",
            "4096",
            "--max-running-requests",
            "1",
            "--page-size",
            str(self.PAGE_SIZE),
            "--chunked-prefill-size",
            "256",
            "--random-seed",
            "0",
            "--disable-cuda-graph",
            "--enable-cache-report",
        ]
        if backend:
            args += ["--radix-cache-backend", backend]
        log_path = self.log_dir / f"{label}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = None
            try:
                process = popen_launch_server(
                    self.model,
                    self.base_url,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=args,
                    env=env,
                    device="npu",
                    return_stdout_stderr=(log, log),
                )
                info = self._request("GET", "/server_info").json()
                self._save(f"{label}_server_info", info)
                self.assertEqual(info["device"], "npu")
                self.assertEqual(info["radix_cache_backend"], backend)
                self.assertFalse(info["disable_radix_cache"])
                yield
            finally:
                if process is not None:
                    kill_process_tree(process.pid)
                # Preserve operation evidence even on a failed assertion.
                events = [
                    json.loads(line)
                    for path in sorted(events_dir.glob("*.jsonl"))
                    for line in path.read_text(encoding="utf-8").splitlines()
                ]
                self._save(f"{label}_events", events)
                print(f"Server log: {log_path}", flush=True)

        log_text = log_path.read_text(encoding="utf-8")
        registrations = [e for e in events if e["event"] == "register"]
        self.assertTrue(registrations, "The plugin must register in server processes")
        self.assertTrue(all(e["backend"] == BACKEND_NAME for e in registrations))
        if backend:
            self.assertIn(f"source=registered('{BACKEND_NAME}')", log_text)
            self.assertIn("impl=ObservedNpuRadixCache", log_text)
            factories = [e for e in events if e["event"] == "factory"]
            self.assertEqual(len(factories), 1)
            self.assertIn(factories[0]["pid"], {e["pid"] for e in registrations})
            self.assertEqual(factories[0]["tp_rank"], 0)
            self.assertTrue(factories[0]["device"].startswith("npu"))
            hits = [e for e in events if e["event"] == "hit"]
            self.assertTrue(hits, "The selected backend must match real NPU KV indices")
            self.assertTrue(all(e["device"].startswith("npu") for e in hits))
            self.assertTrue(
                any(e["event"] == "insert" and e["tokens"] > 0 for e in events)
            )
            self.assertGreaterEqual(sum(e["event"] == "reset" for e in events), 3)
        else:
            self.assertIn("source=default impl=UnifiedRadixCache", log_text)
            self.assertEqual(
                events,
                registrations,
                "Loading a plugin must not select its cache factory",
            )

    def _generate(self, input_ids):
        result = self._request(
            "POST",
            "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": self.OUTPUT_LEN,
                    "ignore_eos": True,
                },
            },
        ).json()
        self.assertEqual(result["meta_info"]["prompt_tokens"], len(input_ids))
        self.assertEqual(result["meta_info"]["completion_tokens"], self.OUTPUT_LEN)
        self.assertEqual(len(result["output_ids"]), self.OUTPUT_LEN)
        return result

    def _flush(self):
        self._request("POST", "/flush_cache", params={"timeout": 30})

    def _assert_same_output(self, first, second):
        self.assertEqual(first["output_ids"], second["output_ids"])
        self.assertEqual(first["text"], second["text"])

    def _exercise_cache(self, label):
        self._flush()
        cold = self._generate(self.prompt_a)
        warm = self._generate(self.prompt_a)
        branch = self._generate(self.prompt_b)
        self._flush()
        branch_cold = self._generate(self.prompt_b)
        self._flush()
        after_flush = self._generate(self.prompt_a)
        results = dict(
            cold=cold,
            warm=warm,
            branch=branch,
            branch_cold=branch_cold,
            after_flush=after_flush,
        )
        self._save(f"{label}_responses", results)

        for name in ("cold", "branch_cold", "after_flush"):
            self.assertEqual(results[name]["meta_info"]["cached_tokens"], 0, name)
        # At least one input token is recomputed; account for page rounding.
        expected_warm = (len(self.prompt_a) - 1) // self.PAGE_SIZE * self.PAGE_SIZE
        self.assertGreaterEqual(warm["meta_info"]["cached_tokens"], expected_warm)
        self.assertLess(warm["meta_info"]["cached_tokens"], len(self.prompt_a))
        self.assertEqual(branch["meta_info"]["cached_tokens"], self.PREFIX_LEN)
        self._assert_same_output(cold, warm)
        self._assert_same_output(cold, after_flush)
        self._assert_same_output(branch, branch_cold)
        return results

    def test_registered_backend_matches_default(self):
        with self._server(None):
            baseline = self._exercise_cache("default")
        with self._server(BACKEND_NAME):
            custom = self._exercise_cache(BACKEND_NAME)
        for name in baseline:
            with self.subTest(request=name):
                self._assert_same_output(baseline[name], custom[name])
                self.assertEqual(
                    baseline[name]["meta_info"]["cached_tokens"],
                    custom[name]["meta_info"]["cached_tokens"],
                )


if __name__ == "__main__":
    unittest.main()
