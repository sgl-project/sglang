import os
import shutil
import tempfile
import time
import unittest
import uuid
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace
from unittest.mock import patch

import requests
import torch
from prometheus_client.parser import text_string_to_metric_families

from sglang.kernels.ops.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.arg_groups.model_override_base import resolving_view
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.arg_groups.serving_hook import handle_other_validations
from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.disaggregation.prefill import (
    SchedulerDisaggregationPrefillMixin,
    should_force_retry,
)
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST, CustomTestCase

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")


FORCE_RETRY_PROB = 0.1


def rid_that_forces_retry(prefix: str) -> str:
    """Return a rid that the test retry sampler will select."""
    for _ in range(1000):
        rid = f"{prefix}{uuid.uuid4().hex}"
        req = SimpleNamespace(
            rid=rid,
            is_retracted=False,
            prefill_attempt_count=0,
        )
        if should_force_retry(req):
            return rid
    raise RuntimeError("Failed to sample an optimistic prefill retry rid")


class OptimisticPrefillRetryCounterMixin:
    def _get_counter_total(self, family_name: str) -> float:
        """Sum of a prefill-side Prometheus counter across its label sets."""
        response = requests.get(f"{self.prefill_url}/metrics")
        response.raise_for_status()
        total = 0.0
        for family in text_string_to_metric_families(response.text):
            if family.name != family_name:
                continue
            for sample in family.samples:
                if sample.name == f"{family_name}_total":
                    total += sample.value
        return total

    def _get_retry_counter(self) -> float:
        return self._get_counter_total("sglang:num_prefill_retries")

    def assert_retry_counter_increases(self, fn):
        before_retries = self._get_retry_counter()
        result = fn()
        after_retries = self._get_retry_counter()
        self.assertGreater(after_retries, before_retries)
        return result


class TestOptimisticPrefill(
    OptimisticPrefillRetryCounterMixin, PDDisaggregationServerBase
):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._force_retry_prob_was_set = (
            envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.is_set()
        )
        cls._force_retry_prob_value = (
            envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
        )
        envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.set(FORCE_RETRY_PROB)
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = [
            "--optimistic-prefill-attempts",
            "3",
            "--chunked-prefill-size",
            "128",
            "--enable-hierarchical-cache",
            "--hicache-write-policy",
            "write_through",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
        ]
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            if getattr(cls, "_force_retry_prob_was_set", False):
                envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.set(
                    cls._force_retry_prob_value
                )
            else:
                envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.clear()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = self.assert_retry_counter_increases(lambda: run_eval(args))
        print(f"Evaluation metrics: {metrics}")
        self.assertGreater(metrics["score"], 0.62)
        time.sleep(1)  # trigger memory check

    def test_logprob(self):
        request_id = rid_that_forces_retry("logprob-retry-")
        prompt = f"{request_id}: " + "The capital of France is Paris. " * 900
        j = self.assert_retry_counter_increases(
            lambda: requests.post(
                self.lb_url + "/generate",
                json={
                    "rid": request_id,
                    "text": prompt,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 8},
                    "return_logprob": True,
                    "return_input_logprob": True,
                    "logprob_start_len": 0,
                },
            ).json()
        )
        completion_tokens = j["meta_info"]["completion_tokens"]
        input_logprobs = j["meta_info"]["input_token_logprobs"]
        output_logprobs = j["meta_info"]["output_token_logprobs"]

        self.assertGreater(j["meta_info"]["prompt_tokens"], 512)
        assert len(output_logprobs) == completion_tokens
        # Input logprobs must be complete: retried or pending chunks must not
        # drop their logprobs.
        self.assertGreaterEqual(
            len(input_logprobs), j["meta_info"]["prompt_tokens"] - 1
        )


class TestOptimisticPrefillFailure(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # enable optimistic prefill retry sampling and disagg failure prob
        cls._force_retry_ctx = (
            envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.override(
                FORCE_RETRY_PROB
            )
        )
        cls._force_retry_ctx.__enter__()
        cls._disagg_failure_ctx = envs.SGLANG_TEST_DISAGG_FAILURE_PROB.override(
            FORCE_RETRY_PROB
        )
        cls._disagg_failure_ctx.__enter__()

        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = [
            "--optimistic-prefill-attempts",
            "3",
            "--chunked-prefill-size",
            "128",
            "--enable-hierarchical-cache",
            "--hicache-write-policy",
            "write_through",
            "--enable-metrics",
            "--enable-request-time-stats-logging",
            "--load-format",
            "dummy",
        ]
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            if getattr(cls, "_force_retry_ctx", None):
                cls._force_retry_ctx.__exit__(None, None, None)
            if getattr(cls, "_disagg_failure_ctx", None):
                cls._disagg_failure_ctx.__exit__(None, None, None)

    def test_survive_requests(self):
        # send many small requests to ensure the engine survives injected failures
        n = 100
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = []
            for i in range(n):
                rid = f"survive-{i}-{uuid.uuid4().hex}"
                futures.append(
                    executor.submit(
                        requests.post,
                        self.lb_url + "/generate",
                        json={
                            "rid": rid,
                            "text": "Hello world",
                            "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                        },
                        timeout=30,
                    )
                )
            for future in as_completed(futures):
                try:
                    _ = future.result()
                except Exception:
                    pass
        time.sleep(1)  # trigger memory check


class TestOptimisticPrefillL3BufferWriteThrough(
    OptimisticPrefillRetryCounterMixin, PDDisaggregationServerBase
):
    """Optimistic prefill with buffer-only L3 (write-through, file backend).

    Small prefill and decode pools keep yielded prefixes evictable while their
    retries wait for decode, so retries that recover the prefix from L3 run
    under real load; the gsm8k score is the correctness check."""

    @classmethod
    def setUpClass(cls):
        cls.hicache_dir = tempfile.mkdtemp(prefix="sglang-hicache-")
        os.environ["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = cls.hicache_dir
        super().setUpClass()
        cls._force_retry_prob_was_set = (
            envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.is_set()
        )
        cls._force_retry_prob_value = (
            envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.get()
        )
        envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.set(FORCE_RETRY_PROB)
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = [
            "--optimistic-prefill-attempts",
            "2",
            "--chunked-prefill-size",
            "128",
            "--max-total-tokens",
            "16384",
            "--enable-metrics",
            "--enable-hierarchical-cache",
            "--hicache-size",
            "4",
            "--hicache-host-memory-mode",
            "buffer_only",
            "--hicache-write-policy",
            "write_through",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
        ]
        # A small decode pool gates bootstrap, so a yielded request waits long
        # enough for the prefill pool above to evict its cached prefix.
        cls.extra_decode_args = ["--max-total-tokens", "16384"]
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        try:
            super().tearDownClass()
        finally:
            if getattr(cls, "_force_retry_prob_was_set", False):
                envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.set(
                    cls._force_retry_prob_value
                )
            else:
                envs.SGLANG_TEST_FORCE_OPTIMISTIC_PREFILL_RETRY_PROB.clear()
            os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR", None)
            shutil.rmtree(cls.hicache_dir, ignore_errors=True)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=f"http://{self.base_host}:{self.lb_port}",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = self.assert_retry_counter_increases(lambda: run_eval(args))
        print(f"Evaluation metrics: {metrics}")
        self.assertGreater(metrics["score"], 0.62)
        # Write-through published prefixes to L3; report what retries fetched back.
        self.assertGreater(self._get_counter_total("sglang:backuped_tokens"), 0)
        print(
            "L3 prefetch hit tokens: "
            f"{self._get_counter_total('sglang:storage_prefetch_hit_tokens')}"
        )
        time.sleep(1)  # trigger memory check


class TestOptimisticPrefillMambaAdmission(CustomTestCase):
    """Mamba radix-cache models must keep optimistic prefill enabled.

    Regression guard: the validation hook used to zero
    ``optimistic_prefill_attempts`` whenever resolution marked the config as
    using the mamba radix cache, silently disabling optimistic prefill for
    hybrid-mamba models in disaggregated prefill. The PP and HiCache
    write-policy restrictions are independent of that gate and must remain.
    """

    BASE = dict(disaggregation_mode="prefill", optimistic_prefill_attempts=3)

    def _make_args(self, **fields) -> ServerArgs:
        server_args = ServerArgs(model_path="dummy")
        for name, value in fields.items():
            setattr(server_args, name, value)
        return server_args

    def _resolved_attempts(self, server_args) -> int:
        handle_other_validations(server_args)
        return resolving_view(server_args).optimistic_prefill_attempts

    def test_mamba_radix_cache_keeps_optimistic_prefill(self):
        server_args = self._make_args(**self.BASE)
        # Simulate resolution having identified a mamba radix-cache model.
        declare_resolution(server_args, "test", uses_mamba_radix_cache=True)
        self.assertEqual(self._resolved_attempts(server_args), 3)

    def test_pp_restriction_retained(self):
        server_args = self._make_args(pp_size=2, **self.BASE)
        self.assertEqual(self._resolved_attempts(server_args), 0)

    def test_l2_write_through_allowed(self):
        server_args = self._make_args(
            enable_hierarchical_cache=True,
            hicache_write_policy="write_through",
            **self.BASE,
        )
        self.assertEqual(self._resolved_attempts(server_args), 3)


class TestOptimisticPrefillMambaRetryRelease(CustomTestCase):
    """An optimistic-prefill retry leaves the donated prefix and exactly one
    Mamba checkpoint in the tree -- not zero, and not a second pinned clone.
    """

    SIZE = 128
    MAMBA_SIZE = 8
    TRACK_SEQLEN = 4
    PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]

    def _setup_mamba_tree(self):
        server_args = ServerArgs(model_path="dummy", page_size=1)
        # The mamba component reads mamba_cache_chunk_size, whose property
        # otherwise loads the HF config for the dummy model.
        server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
        set_global_server_args_for_scheduler(server_args)
        num_layers = 48
        global_interval = 4
        full_attention_layer_ids = [
            i for i in range(global_interval - 1, num_layers, global_interval)
        ]
        mamba_layers = [
            i for i in range(num_layers) if i not in full_attention_layer_ids
        ]
        device = get_device()
        with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
            shape = Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=4096,
                n_groups=16,
                num_heads=32,
                head_dim=128,
                state_size=128,
                conv_kernel=4,
            )
            cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)
        req_to_token_pool = HybridReqToTokenPool(
            size=4,
            mamba_size=self.MAMBA_SIZE,
            mamba_spec_state_size=4,
            max_context_len=64,
            device=device,
            enable_memory_saver=False,
            cache_params=cache_params,
            mamba_layer_ids=mamba_layers,
            enable_mamba_extra_buffer=True,
            speculative_num_draft_tokens=3,
        )
        pool = HybridLinearKVPool(
            size=self.SIZE,
            dtype=torch.bfloat16,
            page_size=1,
            head_num=2,
            head_dim=256,
            full_attention_layer_ids=full_attention_layer_ids,
            device=device,
            enable_memory_saver=False,
            mamba_pool=req_to_token_pool.mamba_pool,
        )
        allocator = TokenToKVPoolAllocator(
            size=self.SIZE,
            dtype=torch.bfloat16,
            device=device,
            kvcache=pool,
            need_sort=False,
        )
        tree = UnifiedRadixCache(
            params=CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                enable_mamba_extra_buffer=True,
                tree_components=(ComponentType.FULL, ComponentType.MAMBA),
            )
        )
        return tree, allocator, req_to_token_pool

    def test_retry_release_retains_donated_checkpoint(self):
        tree, allocator, req_to_token_pool = self._setup_mamba_tree()

        # A request that finished optimistic prefill of PROMPT and tracked one
        # Mamba checkpoint at TRACK_SEQLEN.
        req = Req(
            rid="optimistic-mamba-retry",
            origin_input_text="",
            origin_input_ids=array("q", self.PROMPT),
            sampling_params=SamplingParams(max_new_tokens=1),
        )
        req_to_token_pool.alloc([req])
        kv_indices = allocator.alloc(len(self.PROMPT))
        req_to_token_pool.write(
            (req.kv.req_pool_idx, slice(0, len(self.PROMPT))), kv_indices
        )
        req.full_untruncated_fill_ids = array("q", self.PROMPT)
        req.set_extend_range(0, len(self.PROMPT))
        req.kv.kv_committed_len = len(self.PROMPT)
        req.kv.kv_allocated_len = len(self.PROMPT)
        req.kv.mamba_last_track_seqlen = self.TRACK_SEQLEN
        req.last_node = tree.root_node_handle()

        scheduler = SimpleNamespace(
            tree_cache=tree,
            waiting_queue=[],
            disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[]),
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            processed_tokens_counter=0,
            _release_aborted_request=lambda req: None,
            clear_pending_chunk_send=lambda req: None,
        )
        with patch(
            "sglang.srt.disaggregation.prefill.get_disagg",
            return_value=SimpleNamespace(optimistic_prefill_attempts=3),
        ):
            SchedulerDisaggregationPrefillMixin.optimistic_release_and_requeue(
                scheduler, req
            )

        self.assertEqual(tree.total_size(), (self.TRACK_SEQLEN, 1))
        match = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", self.PROMPT)))
        )
        self.assertIsNotNone(
            tree.tree_core.get_component_device_value(
                match.last_device_node, ComponentType.MAMBA
            )
        )

        # Only the uncached tail and the request-owned Mamba buffers are
        # freed; the tree keeps the donated checkpoint slot.
        self.assertEqual(allocator.available_size(), self.SIZE - self.TRACK_SEQLEN)
        self.assertEqual(
            req_to_token_pool.mamba_allocator.available_size(), self.MAMBA_SIZE - 1
        )


if __name__ == "__main__":
    unittest.main()
