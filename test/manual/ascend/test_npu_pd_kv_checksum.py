"""PD KV checksum (--disaggregation-enable-kv-checksum) end to end on NPU.

Prefill computes an Adler-32 over the request's KV pages and writes it to the
metadata buffer; decode recomputes it and compares. Decode skips the comparison
when the checksum it received is 0, so requests returning 200 would also pass if
prefill never wrote one. test_corrupted_checksum_is_rejected rules that out:
prefill adds 1 to one request's checksum and decode must reject that request.
The kernel's arithmetic is tested in sgl-kernel-npu/tests/manual/test_kv_checksum.py.
The SGLang page/state integration is covered by test_npu_kv_checksum.py.
"""

import json
import os
import random
import shlex
import shutil
import tempfile
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional

import requests

from sglang.bench_serving import get_tokenizer
from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import popen_with_error_check

CORRUPT_RID_PREFIX = "kv-checksum-corrupt-"
CHUNKED_PREFILL_SIZE = 8192

# Installed as sitecustomize.py at the front of the prefill server's PYTHONPATH.
# Schedulers are spawned, so a patch made in this process would not reach them;
# every spawned interpreter runs sitecustomize and inherits PYTHONPATH.
CORRUPT_CHECKSUM_HOOK = """
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

_TARGET = "sglang.srt.disaggregation.utils"


def _patch(module):
    original = module.MetadataBuffers.set_kv_checksum

    def set_kv_checksum(self, req, value):
        # Nonzero, so decode compares it instead of skipping it as unwritten.
        if str(req.rid).startswith(_PREFIX):
            value += 1
        original(self, req, value)

    module.MetadataBuffers.set_kv_checksum = set_kv_checksum


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _TARGET:
            return None
        sys.meta_path.remove(self)
        spec = importlib.util.find_spec(name)
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            _patch(module)

        spec.loader.exec_module = exec_and_patch
        return spec


sys.meta_path.insert(0, _PatchOnImport())

# Run the sitecustomize this file shadows, if the interpreter has one.
_here = os.path.dirname(os.path.abspath(__file__))
_rest = [p for p in sys.path if os.path.abspath(p or os.curdir) != _here]
_shadowed = importlib.machinery.PathFinder.find_spec("sitecustomize", _rest)
if _shadowed is not None:
    _shadowed.loader.exec_module(importlib.util.module_from_spec(_shadowed))
"""


class _NpuPdKvChecksumBase(PDDisaggregationServerBase):
    """Testcase: KV checksum rejects a corrupted checksum and passes intact KV.

    [Test Category] Functional
    [Test Target] --disaggregation-enable-kv-checksum on NPU
    """

    prefill_tp_size = 2
    decode_tp_size = 2
    # Prefill takes the first two devices, decode the next two.
    decode_base_gpu_id = int(
        os.environ.get("SGLANG_TEST_DECODE_BASE_GPU_ID", str(prefill_tp_size))
    )
    # In CI a mismatch raises in the decode scheduler and takes the server down;
    # outside CI it aborts only that request with HTTP 500.
    extra_decode_env = {"SGLANG_IS_IN_CI": "false"}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = QWEN3_32B_WEIGHTS_PATH
        cls.tokenizer = get_tokenizer(cls.model)

        cls.hook_dir = tempfile.mkdtemp(prefix="kv_checksum_hook_")
        with open(os.path.join(cls.hook_dir, "sitecustomize.py"), "w") as f:
            f.write(f"_PREFIX = {CORRUPT_RID_PREFIX!r}\n{CORRUPT_CHECKSUM_HOOK}")
        python_path = [cls.hook_dir, os.environ.get("PYTHONPATH")]
        cls.extra_prefill_env = {
            "PYTHONPATH": os.pathsep.join(p for p in python_path if p)
        }

        # The base class picks mooncake plus RDMA devices from the CI
        # environment; NPU transfers go over the ascend backend instead.
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        cls.rdma_devices = []

        common_args = [
            "--attention-backend",
            "ascend",
            "--mem-fraction-static",
            "0.9",
            "--disable-cuda-graph",
            "--chunked-prefill-size",
            str(CHUNKED_PREFILL_SIZE),
            "--disaggregation-enable-kv-checksum",
        ]
        cls.extra_prefill_args = list(common_args)
        cls.extra_decode_args = list(common_args)

        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            shutil.rmtree(cls.hook_dir, ignore_errors=True)

    @classmethod
    def rdma_devices_for(cls, gpu_indices) -> list:
        return []

    @classmethod
    def launch_lb(cls):
        # The bootstrap-port form that test_npu_pd_disaggregation.py uses, rather
        # than the base class's --mini-lb.
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--prefill",
            cls.prefill_url,
            cls.bootstrap_port,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
            # The router retries a 500 by default, which would replay the
            # corrupted request and could hide a sporadic mismatch.
            "--disable-retries",
        ]
        print("Starting load balancer:", shlex.join(lb_command))
        cls.process_lb = popen_with_error_check(lb_command)
        cls.wait_server_ready(cls.lb_url + "/health", process=cls.process_lb)

        # Router health can pass before its PD workers finish registering.
        # Probe the actual route before testing checksum mismatch rejection.
        deadline = time.monotonic() + 180
        last_error = "No readiness response"
        while time.monotonic() < deadline:
            for name, process in (
                ("prefill", cls.process_prefill),
                ("decode", cls.process_decode),
                ("router", cls.process_lb),
            ):
                if process.poll() is not None:
                    raise RuntimeError(f"PD {name} exited before route readiness")
            try:
                response = requests.post(
                    cls.lb_url + "/generate",
                    json={
                        "text": "PD checksum route readiness",
                        "rid": f"kv-checksum-ready-{uuid.uuid4().hex}",
                        "sampling_params": {
                            "temperature": 0.0,
                            "max_new_tokens": 1,
                        },
                    },
                    timeout=min(60, max(1, deadline - time.monotonic())),
                )
            except requests.RequestException as exc:
                last_error = str(exc)
            else:
                if response.status_code == 200:
                    print("PD checksum route is ready", flush=True)
                    return
                if response.status_code != 503:
                    raise RuntimeError(
                        "PD readiness probe failed: "
                        f"HTTP {response.status_code}: {response.text[:500]}"
                    )
                last_error = f"HTTP 503: {response.text[:500]}"
            time.sleep(1)
        raise TimeoutError(f"PD route was not ready within 180s: {last_error}")

    def gen_prompt(self, token_num: int) -> str:
        all_available_tokens = list(self.tokenizer.get_vocab().values())
        selected_tokens = random.choices(all_available_tokens, k=token_num)
        return self.tokenizer.decode(selected_tokens)

    def post_generate(
        self, prompt: str, rid: Optional[str] = None
    ) -> requests.Response:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 32,
                "ignore_eos": True,
            },
        }
        if rid is not None:
            payload["rid"] = rid
        return requests.post(f"{self.lb_url}/generate", json=payload, timeout=120)

    def send_request(self, prompt: str) -> Dict:
        response = self.post_generate(prompt)
        self.assertEqual(
            response.status_code,
            200,
            f"Request failed: {response.status_code} - {response.text}",
        )
        return response.json()


class TestNpuPdKvChecksum(_NpuPdKvChecksumBase):
    def test_corrupted_checksum_is_rejected(self):
        response = self.post_generate(
            self.gen_prompt(300), rid=f"{CORRUPT_RID_PREFIX}{uuid.uuid4().hex}"
        )
        self.assertEqual(response.status_code, 500, response.text)
        self.assertIn("KV checksum mismatch", response.text)
        # Only the corrupted request is dropped; decode keeps serving.
        self.assertTrue(self.send_request(self.gen_prompt(300))["text"])

    def test_requests_across_page_counts(self):
        # The longest prompt spans more than 64 pages, so each checksum program
        # walks several pages, and prefill sends its KV in more than one chunk.
        for token_num in (1, 17, 300, 800, 2000, 10000):
            with self.subTest(token_num=token_num):
                response = self.send_request(self.gen_prompt(token_num))
                self.assertTrue(response["text"])
                if token_num > CHUNKED_PREFILL_SIZE:
                    self.assertGreater(
                        response["meta_info"]["prompt_tokens"], CHUNKED_PREFILL_SIZE
                    )

    def test_repeated_prompt_reuses_prefix(self):
        # A second pass over the same prompt takes the cached-prefix path, where
        # prefill checksums a different page set than the first request did.
        prompt = self.gen_prompt(800)
        self.send_request(prompt)
        response = self.send_request(prompt)
        self.assertTrue(response["text"])
        self.assertGreater(response["meta_info"]["cached_tokens"], 0)


class TestNpuPdKvChecksumMatrix(_NpuPdKvChecksumBase):
    PAGE_SIZE = 128
    matrix_records = []
    _record_lock = threading.Lock()
    COLD_LENGTHS = (
        1, 2, 17, 127, 128, 129, 255, 256, 257, 511, 512, 513,
        1023, 1024, 1025, 4095, 4096, 4097, 8191, 8192, 8193, 10000,
        16383, 16384, 16385, 24577, 32768,
    )

    @classmethod
    def launch_all(cls):
        for args in (cls.extra_prefill_args, cls.extra_decode_args):
            args.extend(["--page-size", str(cls.PAGE_SIZE)])
            if "--disable-overlap-schedule" in args:
                raise RuntimeError("This regression suite requires overlap enabled")
        super().launch_all()

    def token_ids(self, length, seed):
        # Direct IDs prevent decode/re-tokenize from changing boundary lengths.
        pool = self.tokenizer.encode(
            "The quick brown fox checks the shared key value cache. "
            "Numbers 0123456789 and different words exercise token pages. "
            "\u8fd9\u662f\u7528\u4e8e\u9a8c\u8bc1\u7f13\u5b58\u4f20\u8f93\u6821\u9a8c\u7684\u6d4b\u8bd5\u6587\u672c\u3002",
            add_special_tokens=False,
        )
        special = set(self.tokenizer.all_special_ids)
        pool = [token for token in pool if token not in special]
        self.assertTrue(pool)
        return random.Random(seed).choices(pool, k=length)

    def flush(self, *, prefill=True, decode=True):
        urls = ([self.prefill_url] if prefill else []) + ([self.decode_url] if decode else [])
        for url in urls:
            deadline = time.monotonic() + 30
            while True:
                response = requests.get(f"{url}/flush_cache", timeout=10)
                if response.status_code == 200:
                    break
                self.assertEqual(response.status_code, 400, response.text)
                if time.monotonic() >= deadline:
                    self.fail(f"Cache could not be flushed at {url}: {response.text}")
                time.sleep(0.25)

    def request_ids(self, ids, label, *, output_tokens=4, corrupt=False):
        rid = f"matrix-{label}-{uuid.uuid4().hex}"
        if corrupt:
            rid = CORRUPT_RID_PREFIX + rid
        start = time.monotonic()
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "input_ids": ids,
                "rid": rid,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": output_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=300,
        )
        record = {
            "label": label, "rid": rid, "input_tokens": len(ids),
            "requested_output_tokens": output_tokens,
            "expected_corruption": corrupt, "http_status": response.status_code,
            "elapsed_s": round(time.monotonic() - start, 4),
        }
        data = None
        if response.status_code == 200:
            data = response.json()
            meta = data["meta_info"]
            record.update(
                prompt_tokens=meta["prompt_tokens"],
                completion_tokens=meta["completion_tokens"],
                cached_tokens=meta.get("cached_tokens", 0),
            )
        else:
            record["error"] = response.text[:1000]
        with self._record_lock:
            self.matrix_records.append(record)
            print("MATRIX_ITEM " + json.dumps(record, sort_keys=True), flush=True)
        if corrupt:
            self.assertEqual(response.status_code, 500, response.text)
            self.assertIn("KV checksum mismatch", response.text)
            return None
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(data["meta_info"]["prompt_tokens"], len(ids), record)
        self.assertEqual(data["meta_info"]["completion_tokens"], output_tokens, record)
        return data

    def test_matrix_cold_lengths(self):
        for length in self.COLD_LENGTHS:
            with self.subTest(input_tokens=length):
                self.flush()
                data = self.request_ids(self.token_ids(length, length), f"cold-{length}")
                self.assertEqual(data["meta_info"].get("cached_tokens", 0), 0)

    def test_matrix_chunk_repetition_and_page_reuse(self):
        # Alternate long/short inputs so released KV pages are reused.
        for repeat in range(3):
            self.flush()
            for length in (8193, 129, 16385, 257):
                with self.subTest(repeat=repeat, input_tokens=length):
                    self.request_ids(
                        self.token_ids(length, 100000 + repeat * 100 + length),
                        f"reuse-{repeat}-{length}",
                    )

    def test_matrix_cached_prefix_boundaries(self):
        for prefix_len, suffix_len in (
            (128, 1), (129, 129), (256, 257),
            (257, 8193), (8192, 1), (8193, 129),
        ):
            with self.subTest(prefix_tokens=prefix_len, suffix_tokens=suffix_len):
                self.flush()
                prefix = self.token_ids(prefix_len, 200000 + prefix_len)
                self.request_ids(
                    prefix + self.token_ids(37, 300000 + prefix_len),
                    f"prefix-fill-{prefix_len}",
                )
                # Retain prefill's prefix, force decode to receive it.
                self.flush(prefill=False)
                data = self.request_ids(
                    prefix + self.token_ids(suffix_len, 400000 + prefix_len),
                    f"prefix-probe-{prefix_len}-{suffix_len}",
                )
                expected_hit = prefix_len // self.PAGE_SIZE * self.PAGE_SIZE
                self.assertGreaterEqual(data["meta_info"]["cached_tokens"], expected_hit)

    def concurrent_group(self, requests_to_send, min_cached_tokens=0):
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(self.request_ids, ids, label): label
                for ids, label in requests_to_send
            }
            for future in as_completed(futures):
                with self.subTest(label=futures[future]):
                    data = future.result()
                    if min_cached_tokens:
                        self.assertGreaterEqual(
                            data["meta_info"]["cached_tokens"], min_cached_tokens
                        )

    def test_matrix_concurrent_mixed_lengths(self):
        for repeat, lengths in enumerate(
            ((129, 8193, 16385, 32768), (128, 8192, 8193, 16384))
        ):
            self.flush()
            self.concurrent_group([
                (self.token_ids(length, 500000 + repeat * 100 + i), f"mixed-{repeat}-{length}")
                for i, length in enumerate(lengths)
            ])

    def test_matrix_concurrent_shared_prefix(self):
        self.flush()
        prefix = self.token_ids(1024, 600000)
        self.request_ids(prefix + self.token_ids(37, 600001), "shared-prefix-fill")
        for repeat in range(3):
            # Flush only between fully completed concurrent batches.
            self.flush(prefill=False)
            self.concurrent_group([
                (
                    prefix + self.token_ids(length, 610000 + repeat * 100 + i),
                    f"shared-{repeat}-{length}",
                )
                for i, length in enumerate((1, 129, 8193, 16385))
            ], min_cached_tokens=1024)

    def test_matrix_output_lengths(self):
        for output_tokens in (1, 32, 128):
            with self.subTest(output_tokens=output_tokens):
                self.flush()
                self.request_ids(
                    self.token_ids(1025, 700000 + output_tokens),
                    f"output-{output_tokens}", output_tokens=output_tokens,
                )

    def test_matrix_corruption_and_recovery(self):
        for length in (129, 8193, 16385):
            with self.subTest(corrupted_input_tokens=length):
                self.flush()
                self.request_ids(
                    self.token_ids(length, 800000 + length),
                    f"corrupt-{length}", corrupt=True,
                )
                self.request_ids(
                    self.token_ids(129, 900000 + length), f"recovery-{length}"
                )


if __name__ == "__main__":
    unittest.main()
