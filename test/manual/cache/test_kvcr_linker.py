"""Manual GPU acceptance for direct linkers; not registered in CI.

See python/sglang/srt/mem_cache/storage/kvcr/README.md for prerequisites and
acceptance criteria; run `python test/manual/cache/test_kvcr_linker.py --help`
for CLI options.
"""

import argparse
import importlib.metadata
import json
import random
import subprocess
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path

import requests
import torch

from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.kits.unified_radix_cache_kit import UnifiedRadixTreeTestMixin
from sglang.test.kl_multiturn_utils import (
    _extract_output_logprobs,
    _flush_cache,
    _replay_and_compare_kl,
    get_input_ids,
)
from sglang.test.mooncake_utils import MooncakeTestServices
from sglang.test.test_utils import (
    CustomTestCase,
    find_available_port,
    popen_launch_server,
    terminate_and_kill_process_tree,
    unified_radix_tree_server_env,
)

MODELS = {
    "qwen3": {
        "model": "Qwen/Qwen3-32B-FP8",
        "tp": 1,
        "page_size": 64,
        # Pending FP8 calibration; inherited from the BF16 Qwen3-32B cache test.
        "kl_threshold": 0.0025,
        "max_total_tokens": 8192,
        "args": [
            "--attention-backend",
            "flashinfer",
            "--kv-cache-dtype",
            "bfloat16",
            "--mem-fraction-static",
            "0.8",
        ],
    },
    "glm52": {
        "model": "zai-org/GLM-5.2-FP8",
        "tp": 8,
        "page_size": 64,
        "kl_threshold": 0.03,
        "max_total_tokens": 12000,
        "args": [
            "--mem-fraction-static",
            "0.8",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ],
    },
    "dsv4": {
        "model": "sgl-project/DeepSeek-V4-Flash-FP8",
        "tp": 4,
        "page_size": 256,
        "kl_threshold": 0.01,
        "max_total_tokens": 8192,
        "args": [
            "--attention-backend",
            "compressed",
            "--chunked-prefill-size",
            "8192",
            "--mem-fraction-static",
            "0.92",
            "--disable-shared-experts-fusion",
            "--swa-full-tokens-ratio",
            "0.25",
        ],
    },
}


def peer_hint(tokens, page_size):
    hashes = get_storage_hash_str(tokens, page_size=page_size)
    return {
        "protocol_version": "0.1",
        "message_id": uuid.uuid4().hex,
        "actions": [
            {
                "action_id": "load",
                "action_type": "kv.fetch",
                "action_version": "1.0",
                "payload": {
                    "source_control_endpoint": "tcp://127.0.0.1:19500",
                    "block_hashes": [int(key[:16], 16) for key in hashes],
                },
            }
        ],
    }


def save_runtime_versions(output_dir, backend):
    versions = {"backend": backend}
    repos = {"sglang": Path(__file__).resolve().parents[3]}
    if backend == "kvcr":
        import kvcr

        source = Path(kvcr.__file__).resolve()
        try:
            version = importlib.metadata.version("kvcr")
        except importlib.metadata.PackageNotFoundError:
            version = "uninstalled source checkout"
        versions.update(kvcr_version=version, kvcr_source=str(source))
        for parent in source.parents:
            if (parent / ".git").exists() and (parent / "src" / "kvcr").exists():
                repos["kvcr"] = parent
                break
    for name, repo in repos.items():
        for field, args in (
            ("commit", ["rev-parse", "HEAD"]),
            ("changes", ["status", "--short"]),
        ):
            result = subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                versions[f"{name}_{field}"] = result.stdout.strip()
    (output_dir / "runtime_versions.json").write_text(
        json.dumps(versions, indent=2) + "\n"
    )


class TestKVCRLinker(UnifiedRadixTreeTestMixin, CustomTestCase):
    # Run only the cache correctness methods from this mixin.
    test_gsm8k = None
    sampling_temperature = 0
    max_new_tokens = 64
    prefix_len = 2048
    decode_hit_request_batch_size = 3
    decode_hit_inter_batch_delay_s = 0.5
    options = None

    @classmethod
    def setUpClass(cls):
        if cls.options is None:
            raise RuntimeError("Run this manual test as a script with --model.")
        cls.processes, cls.log_files = [], []
        cls.mooncake = None
        cls.config = MODELS[cls.options.model]
        needed = cls.config["tp"] * (2 if cls.options.case != "local" else 1)
        if torch.cuda.device_count() < needed:
            raise RuntimeError(f"This acceptance run requires {needed} visible GPUs.")
        cls.model = cls.options.model_path or cls.config["model"]
        cls.page_size = cls.config["page_size"]
        cls.kl_threshold = cls.config["kl_threshold"]
        cls.output_dir = Path(cls.options.output_dir).resolve()
        cls.output_dir.mkdir(parents=True, exist_ok=True)
        save_runtime_versions(cls.output_dir, cls.options.backend)
        print(f"Acceptance artifacts: {cls.output_dir}")
        if cls.options.backend == "mooncake":
            cls.mooncake = MooncakeTestServices()
            cls.mooncake.start()
        cls.base_url = cls.launch(0)
        cls.input_ids = get_input_ids(cls.model, num_samples=18)

    @classmethod
    def launch(cls, replica):
        base_url = f"http://127.0.0.1:{find_available_port(30000 + 1000 * replica)}"
        log = (cls.output_dir / f"worker-{replica}.log").open("w")
        cls.log_files.append(log)
        env = (
            {"SGLANG_DSV4_FP4_EXPERTS": "0"}
            if cls.options.model == "dsv4"
            else unified_radix_tree_server_env("rust")
        )
        extra_config = {
            "dram_size_gb": cls.options.dram_size_gb,
            "guard_index": replica * cls.config["tp"],
            "control_host": "127.0.0.1",
            "advertise_host": "127.0.0.1",
            "control_port": 19500 + 100 * replica,
            "nixl_port": 20500 + 100 * replica,
        }
        if cls.mooncake is not None:
            env.update(cls.mooncake.server_env())
            extra_config = {"enable_group_semantics": True}
        process = popen_launch_server(
            cls.model,
            base_url,
            timeout=3600,
            device="cuda",
            env=env,
            return_stdout_stderr=(log, log),
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                str(cls.config["tp"]),
                "--base-gpu-id",
                str(replica * cls.config["tp"]),
                "--page-size",
                str(cls.page_size),
                "--max-total-tokens",
                str(cls.config["max_total_tokens"]),
                "--max-running-requests",
                "1",
                "--enable-cache-report",
                "--log-level",
                "debug",
                "--enable-unified-cache-external-linker",
                "--unified-cache-external-linker-backend",
                cls.options.backend,
                "--hicache-storage-backend-extra-config",
                json.dumps(extra_config),
                *cls.config["args"],
            ],
        )
        cls.processes.append(process)
        return base_url

    @classmethod
    def tearDownClass(cls):
        with ExitStack() as cleanup:
            if getattr(cls, "mooncake", None) is not None:
                cleanup.callback(cls.mooncake.stop)
            for log in getattr(cls, "log_files", []):
                cleanup.callback(log.close)
            for process in getattr(cls, "processes", []):
                cleanup.callback(terminate_and_kill_process_tree, process)

    @contextmanager
    def profile(self, base_url, label):
        if self.options.profile:
            response = requests.post(
                base_url + "/start_profile",
                json={
                    "output_dir": str(self.output_dir / label),
                    "activities": ["CPU", "GPU"],
                    "with_stack": False,
                    "record_shapes": False,
                },
                timeout=60,
            )
            response.raise_for_status()
        try:
            yield
        finally:
            if self.options.profile:
                requests.post(
                    base_url + "/stop_profile", timeout=120
                ).raise_for_status()

    def prefill_cache_assert(self, result, prefix_len, label):
        meta = result["meta_info"]
        self.assertGreaterEqual(
            meta["cached_tokens"], prefix_len - self.page_size, label
        )
        self.host_tokens += int(
            (meta.get("cached_tokens_details") or {}).get("host", 0)
        )

    def decode_cache_assert(self, result, history_len, output_len, label):
        self.prefill_cache_assert(result, history_len + output_len, label)

    def run_local_case(self, case):
        self.host_tokens = 0
        with self.profile(self.base_url, case.__name__):
            case()
        self.assertGreater(
            self.host_tokens, 0, f"No {self.options.backend} load-back occurred"
        )
        print(
            f"{case.__name__}: restored {self.host_tokens} tokens through "
            f"{self.options.backend}"
        )

    def test_multiturn_logprobs_match(self):
        self.run_local_case(super().test_multiturn_logprobs_match)

    def test_multiturn_prefill_cache_hit_branching(self):
        self.run_local_case(super().test_multiturn_prefill_cache_hit_branching)

    def test_multiturn_decode_cache_hit_branching(self):
        self.run_local_case(super().test_multiturn_decode_cache_hit_branching)

    def generate(
        self, base_url, tokens, *, hint=None, max_new_tokens=64, logprob_start_len=-1
    ):
        payload = {
            "input_ids": tokens,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": logprob_start_len,
            "return_text_in_logprobs": False,
        }
        if hint is not None:
            payload["kv_hints"] = hint
        response = requests.post(base_url + "/generate", json=payload, timeout=300)
        response.raise_for_status()
        return response.json()

    def pressure(self, base_url, seed):
        # Exceed L1 capacity without resetting deposited DRAM.
        rng = random.Random(seed)
        for _ in range(self.config["max_total_tokens"] // self.prefix_len + 2):
            self.generate(
                base_url,
                [rng.randint(1, 30000) for _ in range(self.prefix_len)],
                max_new_tokens=1,
            )

    def test_peer_idle_and_concurrent_pressure(self):
        _flush_cache(self.base_url)
        prompts = []
        for ids in self.input_ids:
            if len(ids) >= self.prefix_len and all(
                ids[: self.page_size] != prompt[: self.page_size] for prompt in prompts
            ):
                prompts.append(ids[: self.prefix_len])
            if len(prompts) == 2:
                break
        self.assertEqual(len(prompts), 2, "Need two page-aligned LongBench prefixes")
        baselines = [
            self.generate(self.base_url, prompt, logprob_start_len=0)
            for prompt in prompts
        ]
        self.assertTrue(
            all(item["meta_info"]["cached_tokens"] == 0 for item in baselines)
        )
        self.pressure(self.base_url, seed=90210)
        destination = self.launch(1)
        time.sleep(2)
        rng = random.Random(12345)
        fresh_tokens = [rng.randint(1, 30000) for _ in range(self.prefix_len * 2)]
        # From here until both loads finish, no request wakes the source scheduler.
        with self.profile(destination, "peer_idle_and_concurrent_pressure"):
            first = self.generate(
                destination, prompts[0], hint=peer_hint(prompts[0], self.page_size)
            )
            with ThreadPoolExecutor(max_workers=2) as executor:
                fresh = executor.submit(
                    self.generate,
                    destination,
                    fresh_tokens,
                )
                remote = executor.submit(
                    self.generate,
                    destination,
                    prompts[1],
                    hint=peer_hint(prompts[1], self.page_size),
                )
                fresh_result = fresh.result(timeout=300)
                second = remote.result(timeout=300)
        results = [first, second]
        for index, (result, baseline) in enumerate(zip(results, baselines)):
            meta = result["meta_info"]
            host = int((meta.get("cached_tokens_details") or {}).get("host", 0))
            self.assertGreater(
                host, 0, f"peer request {index} silently recomputed its prefix"
            )
            self.assertGreaterEqual(
                meta["cached_tokens"], self.prefix_len - self.page_size
            )
            self.assertEqual(
                result["output_ids"], baseline["output_ids"], f"peer request {index}"
            )
        self.assertEqual(fresh_result["meta_info"]["cached_tokens"], 0)
        self.pressure(destination, seed=54321)
        reloaded = self.generate(destination, fresh_tokens)
        host = int(
            (reloaded["meta_info"].get("cached_tokens_details") or {}).get("host", 0)
        )
        self.assertGreater(host, 0, "The concurrent offload was not reloadable")
        self.assertGreaterEqual(
            reloaded["meta_info"]["cached_tokens"], len(fresh_tokens) - self.page_size
        )
        self.assertEqual(reloaded["output_ids"], fresh_result["output_ids"])
        prompts.append(fresh_tokens)
        results.append(reloaded)
        _replay_and_compare_kl(
            destination,
            self.model,
            self.kl_threshold,
            [prompt + result["output_ids"] for prompt, result in zip(prompts, results)],
            [_extract_output_logprobs(result) for result in results],
            label="kvcr_peer",
            sampling_temperature=0,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--model-path", help="Local checkpoint override")
    parser.add_argument("--backend", choices=("kvcr", "mooncake"), default="kvcr")
    parser.add_argument("--case", choices=("local", "peer", "all"), default="local")
    parser.add_argument("--dram-size-gb", type=float, default=8)
    parser.add_argument("--output-dir", default=f"/tmp/sglang-kvcr-{int(time.time())}")
    parser.add_argument("--profile", action="store_true")
    options = parser.parse_args()
    if options.backend == "mooncake" and options.case != "local":
        parser.error("Mooncake supports --case local here; peer hints require KVCR.")
    TestKVCRLinker.options = options
    names = []
    if options.case != "peer":
        names.extend(
            [
                "test_multiturn_logprobs_match",
                "test_multiturn_prefill_cache_hit_branching",
                "test_multiturn_decode_cache_hit_branching",
            ]
        )
    if options.case != "local":
        names.append("test_peer_idle_and_concurrent_pressure")
    suite = unittest.TestSuite(TestKVCRLinker(name) for name in names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
