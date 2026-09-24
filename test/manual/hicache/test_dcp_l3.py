"""H200 release gate: fresh engines, fixed topology, file-backed MLA DCP.

Run with four free GPUs visible and the model cached:
  python -m pytest test/manual/hicache/test_dcp_l3.py -q -s
Set DCP_L3_OUTPUT_DIR to retain server logs and the JSON result table.
"""

import json
import os
import re
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import requests
from transformers import AutoConfig, AutoTokenizer

from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.test_utils import popen_launch_server, terminate_and_kill_process_tree

MODEL = "deepseek-ai/DeepSeek-V2-Lite-Chat"


class TestDcpL3(unittest.TestCase):
    def test_fresh_engines(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(os.environ.get("DCP_L3_OUTPUT_DIR", temp))
            output.mkdir(parents=True, exist_ok=True)
            tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
            config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
            text = (
                "The library keeps a record of each book. A reader can borrow a book and return it later. "
                * 150
            )
            tokens = tokenizer.encode(text)[:1025]
            self.assertEqual(len(tokens), 1025)
            payload_bytes = (
                64
                * config.num_hidden_layers
                * (config.kv_lora_rank + config.qk_rope_head_dim)
                * 2
            )
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
            self.assertGreaterEqual(len(visible), 4)
            reports = []

            @contextmanager
            def server(name, tp, dcp, storage, devices=None, runtime_attach=False):
                log = output / f"{name}.log"
                port = 31002 if name.endswith("parallel-b") else 31000
                url = f"http://127.0.0.1:{port}"
                env = dict(
                    os.environ,
                    CUDA_VISIBLE_DEVICES=",".join(devices or visible[:tp]),
                    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=str(storage),
                )
                args = [
                    "--trust-remote-code",
                    "--tp-size",
                    str(tp),
                    "--dcp-size",
                    str(dcp),
                    "--attention-backend",
                    "flashinfer",
                    "--dcp-comm-backend",
                    "ag_rs",
                    "--dtype",
                    "bfloat16",
                    "--kv-cache-dtype",
                    "auto",
                    "--page-size",
                    "64",
                    "--context-length",
                    "8192",
                    "--chunked-prefill-size",
                    "2048",
                    "--max-running-requests",
                    "4",
                    "--mem-fraction-static",
                    "0.4",
                    "--cuda-graph-backend-decode",
                    "disabled",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    "--enable-hierarchical-cache",
                    "--hicache-size",
                    "4",
                    "--hicache-write-policy",
                    "write_through",
                    "--hicache-mem-layout",
                    "page_first",
                    "--hicache-io-backend",
                    "kernel",
                    "--hicache-storage-backend",
                    "file",
                    "--hicache-storage-prefetch-policy",
                    "wait_complete",
                    "--hicache-storage-backend-extra-config",
                    '{"prefetch_threshold": 1}',
                    "--enable-cache-report",
                    "--log-level",
                    "debug",
                ]
                if runtime_attach:
                    backend_index = args.index("--hicache-storage-backend")
                    del args[backend_index : backend_index + 2]
                    args.extend(["--admin-api-key", "dcp-l3-test-admin"])
                with log.open("w") as handle:
                    process = popen_launch_server(
                        MODEL,
                        url,
                        timeout=300,
                        other_args=args,
                        env=env,
                        return_stdout_stderr=(handle, handle),
                    )
                    try:
                        if runtime_attach:
                            # HTTP readiness can precede the warmup request's
                            # cache transfers draining. Attach requires idle.
                            deadline = time.monotonic() + 30
                            while True:
                                response = requests.put(
                                    url + "/hicache/storage-backend",
                                    headers={
                                        "Authorization": "Bearer dcp-l3-test-admin"
                                    },
                                    json={
                                        "hicache_storage_backend": "file",
                                        "hicache_storage_backend_extra_config_json": '{"prefetch_threshold": 1}',
                                        "hicache_storage_prefetch_policy": "wait_complete",
                                        "hicache_write_policy": "write_through",
                                    },
                                    timeout=30,
                                )
                                if response.ok:
                                    break
                                if (
                                    "scheduler is not idle" not in response.text
                                    or time.monotonic() >= deadline
                                ):
                                    self.fail(
                                        f"Runtime attach failed: {response.status_code} {response.text}"
                                    )
                                time.sleep(0.1)
                        yield url, log
                    finally:
                        terminate_and_kill_process_tree(process)

            def generate(url):
                response = requests.post(
                    url + "/generate",
                    json={
                        "input_ids": tokens,
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 8,
                            "ignore_eos": True,
                        },
                    },
                    timeout=120,
                )
                response.raise_for_status()
                return response.json()

            def storage_tokens(result):
                return result["meta_info"]["cached_tokens_details"]["storage"]

            def wait_files(storage, hashes, dcp):
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    files = list(storage.glob("*.bin"))
                    if all(len(list(storage.glob(h + "*.bin"))) == dcp for h in hashes):
                        self.assertEqual(len(files), len(hashes) * dcp)
                        self.assertEqual(
                            {p.stat().st_size for p in files}, {payload_bytes}
                        )
                        return files
                    time.sleep(0.1)
                self.fail(
                    "Timed out waiting for complete files from every shard writer"
                )

            def read_counts(log, tp, expected):
                matches = re.findall(
                    r"DCP L3 prefetch: tp_rank=(\d+) tokens=(\d+)", log.read_text()
                )
                counts = {int(rank): int(count) for rank, count in matches}
                self.assertEqual(counts, {rank: expected for rank in range(tp)})
                return [counts[rank] for rank in range(tp)]

            topologies = os.environ.get("DCP_L3_TOPOLOGIES", "2:2,4:2,4:4")
            for topology in topologies.split(","):
                tp, dcp = map(int, topology.split(":"))
                name = f"tp{tp}-dcp{dcp}"
                storage = Path(temp) / name
                storage.mkdir()
                page = 64 * dcp
                hashes = get_storage_hash_str(tokens[:1024], None, page_size=page)
                with server(name + "-writer", tp, dcp, storage) as (url, _):
                    cold = generate(url)
                    self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
                    files = wait_files(storage, hashes, dcp)
                with server(
                    name + "-reader",
                    tp,
                    dcp,
                    storage,
                    runtime_attach=(tp, dcp) == (4, 4),
                ) as (url, log):
                    warm = generate(url)
                    self.assertEqual(warm["output_ids"], cold["output_ids"])
                    self.assertEqual(storage_tokens(warm), 1024)
                    counts = read_counts(log, tp, 1024)
                report = dict(
                    tp=tp,
                    dcp=dcp,
                    pages=len(hashes),
                    objects=len(files),
                    bytes_per_object=payload_bytes,
                    restored_tokens=counts,
                    cold_output_ids=cold["output_ids"],
                    restored_output_ids=warm["output_ids"],
                )
                if (tp, dcp) == (4, 2):
                    missing = next(
                        p
                        for p in storage.glob(hashes[1] + "*.bin")
                        if "_dcp1_" in p.name
                    )
                    missing.unlink()
                    with server(name + "-missing", tp, dcp, storage) as (url, log):
                        partial = generate(url)
                        self.assertEqual(partial["output_ids"], cold["output_ids"])
                        self.assertEqual(storage_tokens(partial), page)
                        report["missing_shard_restored_tokens"] = read_counts(
                            log, tp, page
                        )
                reports.append(report)
                (output / "results.json").write_text(json.dumps(reports, indent=2))
                print("DCP_L3_RESULT=" + json.dumps(report), flush=True)

            if os.environ.get("DCP_L3_SKIP_CONCURRENCY") != "1":
                storage = Path(temp) / "concurrent"
                storage.mkdir()
                hashes = get_storage_hash_str(tokens[:1024], None, page_size=128)
                with server("parallel-a", 2, 2, storage, visible[:2]) as (url_a, _):
                    with server("parallel-b", 2, 2, storage, visible[2:4]) as (
                        url_b,
                        _,
                    ):
                        with ThreadPoolExecutor(2) as executor:
                            a, b = list(executor.map(generate, (url_a, url_b)))
                        self.assertEqual(a["output_ids"], b["output_ids"])
                        files = wait_files(storage, hashes, 2)
                with server("parallel-reader", 2, 2, storage) as (url, log):
                    result = generate(url)
                    self.assertEqual(result["output_ids"], a["output_ids"])
                    self.assertEqual(storage_tokens(result), 1024)
                    counts = read_counts(log, 2, 1024)
                reports.append(
                    dict(
                        concurrent_engines=2,
                        tp=2,
                        dcp=2,
                        objects=len(files),
                        restored_tokens=counts,
                        output_ids=result["output_ids"],
                    )
                )
                (output / "results.json").write_text(json.dumps(reports, indent=2))
                print("DCP_L3_CONCURRENCY=" + json.dumps(reports[-1]), flush=True)


if __name__ == "__main__":
    unittest.main()
