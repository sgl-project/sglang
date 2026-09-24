"""PD KV checksum (--disaggregation-enable-kv-checksum) end to end on NPU.

Prefill computes an Adler-32 over the request's KV pages and writes it to the
metadata buffer; decode recomputes it and compares. Decode skips the comparison
when the checksum it received is 0, so requests returning 200 would also pass if
prefill never wrote one. test_corrupted_checksum_is_rejected rules that out:
prefill adds 1 to one request's checksum and decode must reject that request.
The kernel's own arithmetic is covered by test/manual/ascend/test_npu_kv_checksum.py.
"""

import os
import random
import shlex
import shutil
import tempfile
import unittest
import uuid
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


class TestNpuPdKvChecksum(PDDisaggregationServerBase):
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


if __name__ == "__main__":
    unittest.main()
