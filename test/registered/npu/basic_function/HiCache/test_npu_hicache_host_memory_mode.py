"""Ascend regression for --hicache-host-memory-mode using Llama or Qwen3.

Migrates the backup/flush/prefetch scenario from the GPU file-backend test
and adds explicit cache vs buffer_only lifetime and cache-source assertions.
Uses the model-path convention from sglang.test.ascend.test_ascend_utils.

Run on one idle NPU from the repository root:
    ASCEND_RT_VISIBLE_DEVICES=0 \
    SGLANG_TEST_MODEL_PATH=/path/to/Llama-3.2-1B-Instruct \
    python test/registered/npu/basic_function/HiCache/test_npu_hicache_host_memory_mode.py -v

Set SGLANG_TEST_LOG_DIR to retain server logs, responses, and metrics.
SGLANG_TEST_MODEL_PATH may also select Qwen3-0.6B. Hybrid Qwen3.5/3.6
models contain a MAMBA component and cannot run the buffer_only roundtrip;
the component rejection is covered separately without allocating a KV pool.
"""

import json
import os
import tempfile
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import requests
import torch
from prometheus_client.parser import text_string_to_metric_families
from transformers import AutoConfig, AutoTokenizer

from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_npu, kill_process_tree
from sglang.srt.utils.network import get_open_port
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase, popen_launch_server

register_npu_ci(est_time=300, suite="nightly-1-npu-a3", nightly=True)


@unittest.skipUnless(is_npu(), "Requires Ascend NPU")
class TestNpuHiCacheHostMemoryMode(CustomTestCase):
    PAGE_SIZE = 128

    def setUp(self):
        self.model = os.environ.get(
            "SGLANG_TEST_MODEL_PATH", LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        )
        self.assertIn(
            AutoConfig.from_pretrained(self.model).model_type,
            ("llama", "qwen3"),
            "The two-mode roundtrip requires a full-attention model. "
            "Use Llama-3.2-1B-Instruct or Qwen3-0.6B; Qwen3.5/3.6 hybrid "
            "models are not supported by buffer_only.",
        )
        log_dir = os.environ.get("SGLANG_TEST_LOG_DIR")
        if log_dir is None:
            directory = tempfile.TemporaryDirectory(prefix="npu-hicache-mode-")
            self.addCleanup(directory.cleanup)
            log_dir = directory.name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)

    def test_argument_defaults_and_rejections(self):
        for mode, expected_ratio in (("cache", 2.0), ("buffer_only", 1.2)):
            with self.subTest(mode=mode):
                args = ServerArgs(
                    model_path=self.model,
                    device="npu",
                    enable_hierarchical_cache=True,
                    hicache_host_memory_mode=mode,
                    hicache_storage_backend="file",
                )
                args.resolve_once()
                self.assertEqual(
                    resolution_result(args, "hicache_ratio"), expected_ratio
                )
                self.assertEqual(
                    resolution_result(args, "hicache_io_backend"), "kernel_ascend"
                )
                self.assertEqual(
                    resolution_result(args, "hicache_mem_layout"), "page_first_direct"
                )
        cases = [
            ({"hicache_storage_backend": None}, "requires a storage backend"),
            ({"hicache_write_policy": "write_back"}, "does not support"),
            ({"disaggregation_mode": "decode"}, "not supported on decode"),
            (
                {"hicache_host_memory_mode": "invalid"},
                "must be 'cache' or 'buffer_only'",
            ),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                kwargs = dict(
                    model_path=self.model,
                    device="npu",
                    enable_hierarchical_cache=True,
                    hicache_host_memory_mode="buffer_only",
                    hicache_storage_backend="file",
                )
                kwargs.update(overrides)
                with self.assertRaisesRegex(ValueError, message):
                    ServerArgs(**kwargs).resolve_once()

    def test_mamba_component_rejected_before_host_pool_allocation(self):
        from sglang.srt.mem_cache.unified_cache.components import ComponentType
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        # Exercise the production guard, not a reimplementation of it. Hybrid
        # Qwen3.5/3.6 trees include these components. The rejection precedes
        # pool construction, so no model weights or NPU allocations are needed.
        cache = SimpleNamespace(
            tree_components=(ComponentType.FULL, ComponentType.MAMBA)
        )
        with (
            patch(
                "sglang.srt.mem_cache.unified_radix_cache.get_memory",
                return_value=SimpleNamespace(hicache_host_memory_mode="buffer_only"),
            ),
            self.assertRaisesRegex(
                ValueError, r"buffer_only supports only FULL/SWA.*MAMBA"
            ),
        ):
            UnifiedRadixCache.init_hicache(cache, None, None)

    @contextmanager
    def _server(self, mode, storage_dir=None):
        self.base_url = f"http://127.0.0.1:{get_open_port()}"
        label = mode or "baseline"
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
            "--random-seed",
            "0",
            "--decode-log-interval",
            "1",
            "--disable-cuda-graph",
            "--enable-cache-report",
            "--enable-metrics",
        ]
        env = {"TOKENIZERS_PARALLELISM": "false"}
        if mode:
            args += [
                "--enable-hierarchical-cache",
                "--hicache-host-memory-mode",
                mode,
                "--hicache-ratio",
                "1.2",
                "--hicache-write-policy",
                "write_through",
                "--hicache-storage-backend",
                "file",
                "--hicache-storage-prefetch-policy",
                "wait_complete",
            ]
            env["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = str(storage_dir)
        log_path = self.log_dir / f"{label}.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = None
            try:
                process = popen_launch_server(
                    self.model,
                    self.base_url,
                    timeout=600,
                    other_args=args,
                    env=env,
                    return_stdout_stderr=(log, log),
                )
                info = self._request("get", "/server_info").json()
                self._save(f"{label}_server_info", info)
                self.assertEqual(info["device"], "npu")
                self.assertEqual(info["enable_hierarchical_cache"], mode is not None)
                if mode:
                    self.assertEqual(info["hicache_host_memory_mode"], mode)
                    self.assertEqual(info["hicache_io_backend"], "kernel_ascend")
                    self.assertEqual(info["hicache_mem_layout"], "page_first_direct")
                    self.assertEqual(info["hicache_storage_backend"], "file")
                self._flush()
                yield
            except Exception:
                log.flush()
                print(log_path.read_text(errors="replace")[-16000:])
                raise
            finally:
                if process is not None:
                    kill_process_tree(process.pid)
                    process.wait(timeout=30)

    def _request(self, method, path, **kwargs):
        response = self.session.request(
            method, f"{self.base_url}{path}", timeout=120, **kwargs
        )
        response.raise_for_status()
        return response

    def _save(self, name, value):
        (self.log_dir / f"{name}.json").write_text(
            json.dumps(value, indent=2), encoding="utf-8"
        )

    def _flush(self):
        self._request("post", "/flush_cache", params={"timeout": 30})

    def _generate(self, name, input_ids):
        result = self._request(
            "post",
            "/generate",
            json={
                "input_ids": input_ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                "return_logprob": True,
            },
        ).json()
        self._save(name, result)
        self.assertIn("paris", result["text"].lower())
        self.assertGreater(result["meta_info"]["completion_tokens"], 0)
        print(
            f"{name}: {result['text']!r}, {result['meta_info'].get('cached_tokens_details')}"
        )
        return result

    def _assert_same_generation(self, expected, actual):
        self.assertEqual(actual["text"], expected["text"])
        reference = expected["meta_info"]["output_token_logprobs"]
        observed = actual["meta_info"]["output_token_logprobs"]
        self.assertEqual([x[1] for x in observed], [x[1] for x in reference])
        # BF16 prefill with and without a cached prefix can change rounding.
        torch.testing.assert_close(
            torch.tensor([x[0] for x in observed]),
            torch.tensor([x[0] for x in reference]),
            atol=0.02,
            rtol=0,
        )

    def _wait_host_state(self, mode, name, min_backed_up):
        deadline = time.monotonic() + 75
        metrics = {}
        while time.monotonic() < deadline:
            # Host-pool gauges refresh on forward passes, not on /metrics reads.
            # A sub-page probe forces a fresh sample without creating another
            # cacheable page. This avoids accepting a stale zero from startup.
            probe = self._request(
                "post",
                "/generate",
                json={
                    "text": "Hello",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 2,
                        "ignore_eos": True,
                    },
                },
            ).json()
            self.assertLess(
                probe["meta_info"]["prompt_tokens"]
                + probe["meta_info"]["completion_tokens"],
                self.PAGE_SIZE,
            )
            self._save(f"{name}_metric_probe", probe)
            raw = self._request("get", "/metrics").text
            metrics = {}
            for family in text_string_to_metric_families(raw):
                for sample in family.samples:
                    metrics[sample.name] = metrics.get(sample.name, 0) + sample.value
            host_used = metrics.get("sglang:hicache_host_used_tokens")
            total = metrics.get("sglang:hicache_host_total_tokens", 0)
            backed_up = metrics.get("sglang:backuped_tokens_total", 0)
            # Do not treat a missing metric, or an uninitialized zero-capacity
            # gauge, as proof that buffer_only has released its host staging.
            ready = total > 0 and backed_up >= min_backed_up
            if mode == "buffer_only":
                ready = ready and host_used == 0
            else:
                ready = ready and host_used is not None and host_used >= min_backed_up
            (self.log_dir / f"{name}.prom").write_text(raw, encoding="utf-8")
            if ready:
                print(
                    f"{name}: host_used={host_used}, total={total}, backed_up={backed_up}"
                )
                return
            time.sleep(0.5)
        self.fail(f"Host pool did not reach {mode} state: {metrics}")

    def test_cache_and_buffer_only_storage_roundtrip(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": "Answer the user's question briefly and accurately. "
                    * 72,
                },
                {
                    "role": "user",
                    "content": "What is the capital of France? Reply with only the city name.",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        reusable = (len(input_ids) - 1) // self.PAGE_SIZE * self.PAGE_SIZE
        self.assertGreaterEqual(reusable, 512)
        with self._server(None):
            baseline = self._generate("baseline_cold", input_ids)
            self.assertEqual(baseline["meta_info"]["cached_tokens"], 0)
        for mode in ("cache", "buffer_only"):
            with (
                self.subTest(mode=mode),
                tempfile.TemporaryDirectory(
                    prefix=f"npu-hicache-{mode}-"
                ) as storage_dir,
            ):
                with self._server(mode, storage_dir):
                    cold = self._generate(f"{mode}_cold", input_ids)
                    self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
                    self._assert_same_generation(baseline, cold)
                    self._wait_host_state(mode, f"{mode}_after_backup", reusable)
                    files = [
                        {"name": p.name, "bytes": p.stat().st_size}
                        for p in sorted(Path(storage_dir).glob("*.bin"))
                    ]
                    self._save(f"{mode}_storage_files", files)
                    self.assertGreaterEqual(len(files), reusable // self.PAGE_SIZE)
                    self.assertTrue(all(f["bytes"] > 0 for f in files))
                    warm = self._generate(f"{mode}_warm", input_ids)
                    self._assert_same_generation(baseline, warm)
                    self.assertGreaterEqual(
                        warm["meta_info"]["cached_tokens_details"]["device"], reusable
                    )
                    # Clear both L1 and L2; only this test's L3 directory survives.
                    # Repeat to exercise reuse/release after an earlier load-back.
                    for cycle in range(2):
                        self._flush()
                        loaded = self._generate(f"{mode}_storage_{cycle}", input_ids)
                        details = loaded["meta_info"]["cached_tokens_details"]
                        self.assertEqual(details["device"], 0)
                        self.assertEqual(details["host"], 0)
                        self.assertGreaterEqual(details["storage"], reusable)
                        self.assertEqual(details["storage_backend"], "HiCacheFile")
                        self._assert_same_generation(baseline, loaded)
                        self._wait_host_state(
                            mode, f"{mode}_after_load_{cycle}", reusable
                        )


if __name__ == "__main__":
    unittest.main()
