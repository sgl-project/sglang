"""File HiCache round trip through two independent server processes."""

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import requests
from huggingface_hub import snapshot_download

from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
    terminate_and_kill_process_tree,
)


class FileHiCacheRestoreMixin:
    model = "deepseek-ai/DeepSeek-V2-Lite-Chat"
    revision = "85864749cd611b4353ce1decdb286193298f64c7"
    tp = 2
    dcp = 2
    hybrid = False
    other_args = ["--attention-backend", "flashinfer", "--dcp-comm-backend", "ag_rs"]

    @contextmanager
    def server(self, directory, model_path):
        process = None
        try:
            process = popen_launch_server(
                model_path,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                env=dict(
                    os.environ,
                    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=str(directory),
                    SGLANG_ENABLE_RANK_CONSENSUS_CHECKER="1",
                ),
                other_args=[
                    "--trust-remote-code",
                    "--tp-size",
                    str(self.tp),
                    "--dcp-size",
                    str(self.dcp),
                    "--page-size",
                    "64",
                    "--dtype",
                    "bfloat16",
                    "--random-seed",
                    "0",
                    "--context-length",
                    "2048",
                    "--chunked-prefill-size",
                    "1024",
                    "--max-running-requests",
                    "4",
                    "--max-total-tokens",
                    "4096",
                    "--mem-fraction-static",
                    "0.65",
                    "--cuda-graph-max-bs-decode",
                    "4",
                    "--cuda-graph-backend-prefill",
                    "disabled",
                    "--enable-hierarchical-cache",
                    "--hicache-size",
                    "4",
                    "--hicache-mem-layout",
                    "page_first",
                    "--hicache-io-backend",
                    "kernel",
                    "--hicache-write-policy",
                    "write_through",
                    "--hicache-storage-backend",
                    "file",
                    "--hicache-storage-prefetch-policy",
                    "wait_complete",
                    "--hicache-storage-backend-extra-config",
                    json.dumps({"prefetch_threshold": 1}),
                    "--enable-cache-report",
                    *self.other_args,
                ],
            )
            yield
        finally:
            if process is not None:
                terminate_and_kill_process_tree(process)

    def generate(self, tokens):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/generate",
            json={
                "input_ids": tokens,
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        self.assertNotEqual(result["meta_info"]["finish_reason"]["type"], "abort")
        return result

    def test_fresh_process_restores_rank_shards(self):
        model_path = snapshot_download(
            self.model,
            revision=self.revision,
            allow_patterns=["*.json", "*.py", "*.safetensors", "*.model"],
        )
        tokenizer = get_tokenizer(model_path, trust_remote_code=True)
        task = tokenizer.encode(
            "\nWhat is 17 plus 25? Explain the calculation briefly."
        )
        notes = tokenizer.encode(
            "Reference notes: the library lends books to readers. "
        )
        tokens = (notes * (513 // len(notes) + 1))[: 513 - len(task)] + task
        hashes = get_storage_hash_str(tokens[:512], None, page_size=64 * self.dcp)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.server(directory, model_path):
                self.generate(tokens)
                reference = self.generate(tokens)
                deadline = time.monotonic() + 60
                while True:
                    files = [p for p in directory.glob("*.bin") if p.stat().st_size]
                    kv_ready = all(
                        sum(
                            p.name.startswith(h) and "mamba" not in p.name
                            for p in files
                        )
                        == self.dcp
                        for h in hashes
                    )
                    state_ready = not self.hybrid or all(
                        any(f"mamba_tp{rank}_" in p.name for p in files)
                        for rank in range(self.tp)
                    )
                    if kv_ready and state_ready:
                        break
                    self.assertLess(time.monotonic(), deadline, "Incomplete L3 backup")
                    time.sleep(0.1)
            # A new process rules out accidental L1/L2 reuse.
            with self.server(directory, model_path):
                restored = self.generate(tokens)
            details = restored["meta_info"]["cached_tokens_details"]
            self.assertEqual(details["storage"], 512)
            self.assertEqual(details["device"], 0)
            self.assertEqual(details["host"], 0)
            self.assertEqual(restored["output_ids"], reference["output_ids"])
