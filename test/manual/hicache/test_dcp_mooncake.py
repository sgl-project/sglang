"""Real Mooncake TCP tests for MLA DCP (four free GPUs, cached DeepSeek V2 Lite).

Run with CUDA_VISIBLE_DEVICES set to four free GPUs:
  python -m pytest test/manual/hicache/test_dcp_mooncake.py -q -s
Requires mooncake_master on PATH and mooncake-transfer-engine installed.
DCP_MOONCAKE_OUTPUT_DIR retains logs and results; DCP_MOONCAKE_PORT defaults
to 32000. DCP_MOONCAKE_LAYOUT selects page_first, layer_first or
page_first_direct. DCP_MOONCAKE_SIZE selects 2 or 4 for the fresh-engine test.
"""

import json
import os
import re
import socket
import subprocess
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


class TestDcpMooncake(unittest.TestCase):
    def setUp(self):
        from mooncake.store import MooncakeDistributedStore

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(os.environ.get("DCP_MOONCAKE_OUTPUT_DIR", temp.name))
        self.output.mkdir(parents=True, exist_ok=True)
        self.devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
        self.assertGreaterEqual(len(self.devices), 4)
        self.port = int(os.environ.get("DCP_MOONCAKE_PORT", "32000"))
        self.layout = os.environ.get("DCP_MOONCAKE_LAYOUT", "page_first")
        self.tag = self.id().split(".")[-1]
        self.extra = dict(
            master_server_address=f"127.0.0.1:{self.port + 10}",
            local_hostname="127.0.0.1",
            metadata_server=f"http://127.0.0.1:{self.port + 11}/metadata",
            protocol="tcp",
            device_name="",
            # Keep all objects in the test's storage client across engine exits.
            global_segment_size=0,
            check_server=False,
            prefetch_threshold=1,
            extra_backend_tag=self.tag,
            enable_group_semantics=False,
        )
        log = (self.output / f"{self.tag}-master.log").open("w")
        self.addCleanup(log.close)
        master = subprocess.Popen(
            [
                "mooncake_master",
                f"--rpc_port={self.port + 10}",
                "--enable_http_metadata_server=true",
                f"--http_metadata_server_port={self.port + 11}",
                f"--metrics_port={self.port + 12}",
            ],
            stdout=log,
            stderr=log,
        )
        self.addCleanup(terminate_and_kill_process_tree, master)
        for port in (self.port + 10, self.port + 11):
            deadline = time.monotonic() + 30
            while True:
                self.assertIsNone(master.poll(), "Mooncake master exited")
                with socket.socket() as sock:
                    ready = sock.connect_ex(("127.0.0.1", port)) == 0
                if ready:
                    break
                self.assertLess(time.monotonic(), deadline, "Master startup timed out")
                time.sleep(0.1)
        self.store = MooncakeDistributedStore()
        self.addCleanup(self.store.close)
        self.assertEqual(
            self.store.setup(
                "127.0.0.1",
                self.extra["metadata_server"],
                1024**3,
                16 * 1024**2,
                "tcp",
                "",
                self.extra["master_server_address"],
            ),
            0,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        self.prompts = [
            tokenizer.encode(text * 150)[:1025]
            for text in (
                "The library keeps a record of each book. A reader can borrow a book and return it later. ",
                "Mountains and rivers shape the valley. Each spring the snow melts and the river rises. ",
            )
        ]
        self.assertEqual([len(p) for p in self.prompts], [1025, 1025])
        config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
        self.page_bytes = (
            64
            * config.num_hidden_layers
            * (config.kv_lora_rank + config.qk_rope_head_dim)
            * 2
        )

    @contextmanager
    def server(self, name, tp, dcp, devices, runtime_attach=False, port_offset=0):
        url = f"http://127.0.0.1:{self.port + port_offset}"
        path = self.output / f"{self.tag}-{name}.log"
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
            self.layout,
            "--hicache-io-backend",
            "direct" if self.layout == "page_first_direct" else "kernel",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--enable-cache-report",
            "--log-level",
            "debug",
        ]
        if runtime_attach:
            args.extend(["--admin-api-key", "dcp-mooncake-test"])
        else:
            args.extend(
                [
                    "--hicache-storage-backend",
                    "mooncake",
                    "--hicache-storage-backend-extra-config",
                    json.dumps(self.extra),
                ]
            )
        with path.open("w") as log:
            process = popen_launch_server(
                MODEL,
                url,
                timeout=300,
                other_args=args,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(devices)),
                return_stdout_stderr=(log, log),
            )
            try:
                info = requests.get(
                    url + "/server_info",
                    headers={"Authorization": "Bearer dcp-mooncake-test"},
                    timeout=30,
                )
                info.raise_for_status()
                self.assertEqual(info.json()["hicache_mem_layout"], self.layout)
                if runtime_attach:
                    deadline = time.monotonic() + 30
                    while True:
                        response = requests.put(
                            url + "/hicache/storage-backend",
                            headers={"Authorization": "Bearer dcp-mooncake-test"},
                            json={
                                "hicache_storage_backend": "mooncake",
                                "hicache_storage_backend_extra_config_json": json.dumps(
                                    self.extra
                                ),
                                "hicache_storage_prefetch_policy": "wait_complete",
                                "hicache_write_policy": "write_through",
                            },
                            timeout=30,
                        )
                        if response.ok:
                            break
                        self.assertIn("scheduler is not idle", response.text)
                        self.assertLess(time.monotonic(), deadline, response.text)
                        time.sleep(0.1)
                yield url, path
            finally:
                terminate_and_kill_process_tree(process)

    def generate(self, url, tokens):
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

    def keys(self, tokens, tp, dcp):
        prefix = f"{self.tag}_deepseek-ai-DeepSeek-V2-Lite-Chat_tp{tp}_dcp{dcp}_page{64 * dcp}_pp1_cp1"
        return [
            f"{prefix}_{page_hash}_dcp{shard}_cp0_k"
            for page_hash in get_storage_hash_str(
                tokens[:1024], None, page_size=64 * dcp
            )
            for shard in range(dcp)
        ]

    def wait_pages(self, keys):
        deadline = time.monotonic() + 60
        while self.store.batch_is_exist(keys) != [1] * len(keys):
            self.assertLess(time.monotonic(), deadline, "Shard writes timed out")
            time.sleep(0.1)
        self.assertEqual({self.store.get_size(key) for key in keys}, {self.page_bytes})

    def check_restore(self, result, cold, log, tp, expected=1024):
        self.assertEqual(result["output_ids"], cold["output_ids"])
        self.assertEqual(
            result["meta_info"]["cached_tokens_details"]["storage"], expected
        )
        counts = {
            int(rank): int(tokens)
            for rank, tokens in re.findall(
                r"DCP L3 prefetch: tp_rank=(\d+) tokens=(\d+)", log.read_text()
            )
        }
        self.assertEqual(counts, {rank: expected for rank in range(tp)})
        return dict(
            storage_tokens=expected, rank_tokens=counts, output_ids=result["output_ids"]
        )

    def finish(self, report, keys):
        self.wait_pages(keys)
        # Removing only this test's namespace also counts every stored object,
        # including any accidental replicas under extra rank keys.
        count = self.store.remove_by_regex(f"^{self.tag}_.*", force=True)
        self.assertEqual(count, len(keys))
        report.update(objects=count, page_bytes=self.page_bytes, layout=self.layout)
        (self.output / f"{self.tag}.json").write_text(json.dumps(report, indent=2))
        print("DCP_MOONCAKE=" + json.dumps(report), flush=True)

    def test_two_engines_read_each_other(self):
        tokens_a, tokens_b = self.prompts
        keys = self.keys(tokens_a, 2, 2) + self.keys(tokens_b, 2, 2)
        self.assertEqual(len(set(keys)), 32)
        with self.server("a", 2, 2, self.devices[:2]) as (url_a, log_a):
            with self.server(
                "b", 2, 2, self.devices[2:4], runtime_attach=True, port_offset=2
            ) as (url_b, log_b):
                with ThreadPoolExecutor(2) as executor:
                    a = executor.submit(self.generate, url_a, tokens_a)
                    b = executor.submit(self.generate, url_b, tokens_b)
                    cold_a, cold_b = a.result(), b.result()
                self.assertEqual(cold_a["meta_info"]["cached_tokens"], 0)
                self.assertEqual(cold_b["meta_info"]["cached_tokens"], 0)
                self.wait_pages(keys)
                with ThreadPoolExecutor(2) as executor:
                    a = executor.submit(self.generate, url_a, tokens_b)
                    b = executor.submit(self.generate, url_b, tokens_a)
                    read_a, read_b = a.result(), b.result()
                report = dict(
                    tp=2,
                    dcp=2,
                    a_reads_b=self.check_restore(read_a, cold_b, log_a, 2),
                    b_reads_a=self.check_restore(read_b, cold_a, log_b, 2),
                )
        self.finish(report, keys)

    def test_fresh_engine_and_missing_shard(self):
        tp, dcp = 4, int(os.environ.get("DCP_MOONCAKE_SIZE", "2"))
        tokens = self.prompts[0]
        keys = self.keys(tokens, tp, dcp)
        with self.server("writer", tp, dcp, self.devices[:tp]) as (url, _):
            cold = self.generate(url, tokens)
            self.assertEqual(cold["meta_info"]["cached_tokens"], 0)
            self.wait_pages(keys)
        with self.server("reader", tp, dcp, self.devices[:tp]) as (url, log):
            result = self.generate(url, tokens)
            full = self.check_restore(result, cold, log, tp)
        # Remove shard 1 of logical page 2, leaving later pages present.
        missing = keys[2 * dcp + 1]
        self.assertEqual(
            self.store.remove_by_regex(f"^{re.escape(missing)}$", force=True), 1
        )
        self.assertEqual(self.store.is_exist(missing), 0)
        with self.server("partial", tp, dcp, self.devices[:tp]) as (url, log):
            result = self.generate(url, tokens)
            partial = self.check_restore(result, cold, log, tp, expected=2 * 64 * dcp)
            self.wait_pages(keys)
        self.finish(dict(tp=tp, dcp=dcp, fresh=full, missing_shard=partial), keys)


if __name__ == "__main__":
    unittest.main()
