"""External connector loading and scheduler lifecycle tests without a GPU."""

import importlib
import json
import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_policy import AddReqResult
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.output_streamer import (
    SchedulerOutputStreamer,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    CacheRequestHandle,
    CacheRequestOutcome,
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.kv_transfer_config import KVTransferConfig
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.registry import TreeCacheBuildContext, create_tree_cache
from sglang.srt.runtime_context import get_context, get_server_args
from sglang.srt.server_args import ServerArgs, prepare_server_args

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

CONFIG = {
    "kv_connector": "RecordingConnector",
    "kv_connector_module_path": "external_test_provider",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"host": "tcp://localhost", "port": 5555},
}

# Written outside the source tree, so loading cannot rely on registration side
# effects or an in-tree backend. Only scheduler/control-plane behavior is faked;
# these tests do not claim to exercise a device transfer.
PROVIDER = textwrap.dedent(
    """
    from sglang.srt.mem_cache.base_kv_connector import BaseKVConnector

    class RecordingConnector(BaseKVConnector):
        @classmethod
        def validate_config(cls, context, config):
            if context.is_hybrid_swa:
                raise ValueError("SWA is not supported by this test provider")

        def __init__(self, context, config):
            self.context = context
            self.config = config
            self.pending = set()
            self.ready = set()
            self.calls = []

        def prefetch_request(self, req):
            self.calls.append(("lookup", req.cache_request_handle))
            self.pending.add(req.cache_request_handle)

        def check_prefetch_progress(self, handle):
            return handle in self.ready

        def check_hicache_events(self):
            self.calls.append(("poll",))
            self.ready.update(self.pending)
            self.pending.clear()

        def ready_to_load_host_cache(self):
            self.calls.append(("before_forward",))
            return -1

        def has_pending_cache_operations(self):
            return bool(self.pending)

        def release_host_resources(self):
            self.calls.append(("close",))
            self.pending.clear()
            self.ready.clear()

        def release_aborted_request(self, handle):
            self.pending.discard(handle)
            self.ready.discard(handle)

        def reset(self):
            self.calls.append(("reset",))
            self.ready.clear()

        def match_prefix(self, params):
            raise NotImplementedError("This fixture has no KV tensors")

        def cache_finished_req(self, req, is_insert=True, *, owned_kv_len, **kwargs):
            self.pending.add(req.cache_request_handle)

        def cache_unfinished_req(self, req, **kwargs):
            pass

        def evict(self, params):
            raise NotImplementedError

        def inc_lock_ref(self, node):
            raise NotImplementedError

        def dec_lock_ref(self, node, params=None):
            raise NotImplementedError
    """
)


class TestKVTransferConfig(CustomTestCase):
    def test_defaults_and_private_extra_config_copy(self):
        raw = {key: value for key, value in CONFIG.items() if key != "kv_role"}
        config = KVTransferConfig.from_dict(raw)
        self.assertEqual(config.kv_role, "kv_both")
        config.kv_connector_extra_config["port"] = 9999
        self.assertEqual(raw["kv_connector_extra_config"]["port"], 5555)

    def test_invalid_config(self):
        cases = [
            [],
            {},
            {**CONFIG, "kv_connector": ""},
            {**CONFIG, "kv_connector": "a.b"},
            {**CONFIG, "kv_connector_module_path": "/tmp/provider.py"},
            {**CONFIG, "kv_role": "reader"},
            {**CONFIG, "kv_connector_extra_config": []},
            {**CONFIG, "typo": 1},
        ]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                KVTransferConfig.from_dict(value)

    def test_cli_python_and_pickle_without_importing_provider(self):
        with patch("importlib.import_module", wraps=importlib.import_module) as load:
            cli = prepare_server_args(
                ["--model-path", "dummy", "--kv-transfer-config", json.dumps(CONFIG)]
            )
            api = ServerArgs(model_path="dummy", kv_transfer_config=CONFIG)
            cli.resolve_once()
            api.resolve_once()
        self.assertNotIn(
            CONFIG["kv_connector_module_path"],
            [call.args[0] for call in load.call_args_list],
        )
        for args in (cli, api, pickle.loads(pickle.dumps(api))):
            self.assertEqual(args.kv_transfer_config, CONFIG)

    def test_rejects_conflicting_cache_selection_before_model_loading(self):
        for name, value in (
            ("radix_cache_backend", "custom"),
            ("enable_lmcache", True),
            ("enable_flexkv", True),
            ("enable_hierarchical_cache", True),
            ("hicache_storage_backend", "file"),
            ("enable_unified_cache_external_linker", True),
            ("disable_radix_cache", True),
            ("enable_hisparse", True),
            ("disaggregation_mode", "prefill"),
            ("enable_streaming_session", True),
        ):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "kv-transfer-config"),
            ):
                ServerArgs(
                    model_path="dummy", kv_transfer_config=CONFIG, **{name: value}
                ).resolve_once()


class TestExternalKVConnector(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        Path(self.tmp.name, "external_test_provider.py").write_text(PROVIDER)
        sys.path.insert(0, self.tmp.name)
        self.addCleanup(sys.path.remove, self.tmp.name)
        self.addCleanup(sys.modules.pop, "external_test_provider", None)
        importlib.invalidate_caches()
        override = get_context().override_server_args(kv_transfer_config=CONFIG)
        override.install()
        self.addCleanup(override.restore)

    def context(self, **overrides):
        values = dict(
            server_args=get_server_args(),
            params=MagicMock(),
            is_hybrid_swa=False,
            is_hybrid_ssm=False,
            enable_hierarchical_cache=False,
            disable_radix_cache=False,
            effective_chunked_prefill_size=512,
            tp_worker=MagicMock(),
            model_config=MagicMock(),
            tp_size=2,
            tp_rank=1,
            tp_group=MagicMock(),
        )
        return TreeCacheBuildContext(**(values | overrides))

    def scheduler(self):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.tree_cache = create_tree_cache(self.context())
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = False
        scheduler.enable_hicache_storage = False
        scheduler.running_batch = MagicMock()
        scheduler.running_batch.is_empty.return_value = True
        scheduler.chunked_req = None
        scheduler.dllm_manager = MagicMock()
        scheduler.dllm_manager.any_staging_reqs.return_value = False
        scheduler.last_batch = None
        scheduler.enable_overlap = False
        scheduler.waiting_queue = []
        scheduler._pp_microbatches_drained = lambda: True
        scheduler.grammar_manager = SimpleNamespace(grammar_queue=[])
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.enable_hisparse = False
        scheduler._engine_paused = False
        return scheduler

    def test_import_context_and_extra_config(self):
        context = self.context()
        cache = create_tree_cache(context)
        self.assertIs(cache.context, context)
        self.assertEqual(cache.config.kv_role, "kv_both")
        self.assertEqual(
            cache.config.kv_connector_extra_config, CONFIG["kv_connector_extra_config"]
        )

    def test_provider_validates_before_constructor(self):
        with self.assertRaisesRegex(ValueError, "SWA is not supported"):
            create_tree_cache(self.context(is_hybrid_swa=True))

    def test_bad_module_and_bad_class_fail_without_default_fallback(self):
        for changes, error in (
            ({"kv_connector_module_path": "missing_sglang_test_provider"}, ImportError),
            ({"kv_connector": "MissingClass"}, TypeError),
            ({"kv_connector": "BaseKVConnector"}, TypeError),
        ):
            with self.subTest(changes=changes):
                override = get_context().override_server_args(
                    kv_transfer_config=CONFIG | changes
                )
                with override, self.assertRaises(error):
                    create_tree_cache(self.context())

    def test_import_in_fresh_python_worker(self):
        # No provider registration is copied from the parent process.
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {self.tmp.name!r})
            from sglang.test.test_utils import maybe_stub_sgl_kernel
            maybe_stub_sgl_kernel()
            from sglang.srt.runtime_context import get_context
            from sglang.srt.mem_cache.registry import TreeCacheBuildContext, create_tree_cache
            with get_context().override_server_args(kv_transfer_config={CONFIG!r}) as args:
                context = TreeCacheBuildContext(
                    server_args=args, params=None, is_hybrid_swa=False,
                    is_hybrid_ssm=False, enable_hierarchical_cache=False,
                    disable_radix_cache=False, effective_chunked_prefill_size=None,
                    tp_worker=None, model_config=None, tp_size=1, tp_rank=0, tp_group=None,
                )
                cache = create_tree_cache(context)
                assert cache.config.kv_connector_extra_config['port'] == 5555
            """)
        subprocess.run([sys.executable, "-c", script], check=True, timeout=60)

    def prefill_scheduler(self):
        scheduler = self.scheduler()
        scheduler.grammar_manager = MagicMock()
        scheduler.grammar_manager.has_waiting_grammars.return_value = False
        scheduler.running_batch.batch_is_full = False
        scheduler.running_batch.reqs = []
        scheduler.enable_priority_preemption = False
        scheduler.enable_priority_scheduling = False
        scheduler.is_hybrid_swa = False
        scheduler.min_free_slots_delayer = None
        scheduler.get_num_allocatable_reqs = MagicMock(return_value=8)
        scheduler.policy = MagicMock()
        scheduler.processed_tokens_counter = 0
        scheduler.chunked_prefill_size = 512
        scheduler.tp_worker = MagicMock()
        scheduler.page_size = 1
        scheduler.token_to_kv_pool_allocator = MagicMock()
        scheduler.new_token_ratio_tracker = SimpleNamespace(current=1.0)
        scheduler.max_prefill_tokens = 512
        scheduler.is_mixed_chunk = False
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.max_prefill_bs = 8
        scheduler.max_running_requests = 8
        scheduler.dllm_config = None
        scheduler.enable_lora = False
        scheduler.req_to_token_pool = SimpleNamespace(mamba_allocator=None)
        scheduler.truncation_align_size = None
        scheduler.model_config = MagicMock()
        scheduler.spec_algorithm = MagicMock()
        req = MagicMock()
        req.cache_request_handle = CacheRequestHandle("req", 1)
        req.beam_group = None
        scheduler.waiting_queue = [req]
        return scheduler, req

    def test_admission_waits_for_lookup_and_orders_before_forward(self):
        scheduler, req = self.prefill_scheduler()
        adder = MagicMock()
        adder.can_run_list = []
        adder.preempt_list = []
        adder.new_chunked_req = None

        def admit(*args, **kwargs):
            adder.can_run_list.append(req)
            return AddReqResult.CONTINUE

        adder.add_one_req.side_effect = admit
        scheduler._prefetch_kvcache(req)
        with patch("sglang.srt.managers.scheduler.PrefillAdder", return_value=adder):
            batch, _ = scheduler._get_new_batch_prefill_raw(
                None, scheduler.running_batch
            )
            self.assertIsNone(batch)
            adder.add_one_req.assert_not_called()

            scheduler._process_hicache_events()
            batch = MagicMock()
            # Stop after testing the ordering boundary, before model execution.
            batch.prepare_for_extend.side_effect = RuntimeError("forward boundary")
            with patch(
                "sglang.srt.managers.scheduler.ScheduleBatch.init_new",
                return_value=batch,
            ):
                with self.assertRaisesRegex(RuntimeError, "forward boundary"):
                    scheduler._get_new_batch_prefill_raw(None, scheduler.running_batch)
            self.assertEqual(batch.hicache_consumer_index, -1)
            self.assertEqual(scheduler.tree_cache.calls[-1], ("before_forward",))

    def test_pending_transfer_does_not_mark_empty_running_batch_full(self):
        scheduler, req = self.prefill_scheduler()
        scheduler.tree_cache.ready.add(req.cache_request_handle)
        adder = MagicMock()
        adder.can_run_list = []
        adder.add_one_req.return_value = AddReqResult.NO_TOKEN
        req.kv.holds_mamba = False
        with patch("sglang.srt.managers.scheduler.PrefillAdder", return_value=adder):
            batch, _ = scheduler._get_new_batch_prefill_raw(
                None, scheduler.running_batch
            )
        self.assertIsNone(batch)
        self.assertFalse(scheduler.running_batch.batch_is_full)

    def test_lookup_polls_with_no_hicache_and_abort_releases_attempt(self):
        scheduler = self.scheduler()
        handle = CacheRequestHandle("req", 1)
        req = SimpleNamespace(
            cache_request_handle=handle, init_next_round_input=MagicMock()
        )
        scheduler._prefetch_kvcache(req)
        req.init_next_round_input.assert_called_once_with(
            scheduler.tree_cache, cow_mamba=False
        )
        self.assertFalse(scheduler.tree_cache.check_prefetch_progress(handle))
        scheduler._process_hicache_events()
        self.assertTrue(scheduler.tree_cache.check_prefetch_progress(handle))
        scheduler.tree_cache.finish(handle, CacheRequestOutcome.ABORT)
        self.assertFalse(scheduler.tree_cache.check_prefetch_progress(handle))
        self.assertFalse(scheduler.tree_cache.has_pending_cache_operations())

    def test_pending_store_blocks_flush_until_idle_poll_completes(self):
        scheduler = self.scheduler()
        req = SimpleNamespace(cache_request_handle=CacheRequestHandle("req", 1))
        scheduler.tree_cache.cache_finished_req(req, owned_kv_len=16)
        scheduler.tree_cache.finish(
            req.cache_request_handle, CacheRequestOutcome.SUCCESS
        )
        self.assertFalse(scheduler.is_fully_idle())
        self.assertTrue(scheduler.is_fully_idle(for_health_check=True))
        self.assertFalse(scheduler.flush_cache())
        self.assertNotIn(("reset",), scheduler.tree_cache.calls)
        scheduler._process_hicache_events()
        self.assertTrue(scheduler.is_fully_idle())

    def test_storage_metadata_and_optional_clear(self):
        scheduler = self.scheduler()
        streamer = SchedulerOutputStreamer.__new__(SchedulerOutputStreamer)
        self.assertEqual(streamer._get_storage_backend_type(), "RecordingConnector")
        self.assertFalse(scheduler.clear_hicache_storage_wrapped(None).success)

    def test_noop_example_reuses_unified_cache_on_cpu(self):
        example_dir = (
            Path(__file__).resolve().parents[4] / "examples/runtime/kv_connector"
        )
        sys.path.insert(0, str(example_dir))
        self.addCleanup(sys.path.remove, str(example_dir))
        self.addCleanup(sys.modules.pop, "noop_connector", None)
        config = {
            "kv_connector": "NoOpKVConnector",
            "kv_connector_module_path": "noop_connector",
        }
        with get_context().override_server_args(
            kv_transfer_config=config, device="cpu"
        ):
            allocator = TokenToKVPoolAllocator(32, torch.float32, "cpu", None, False)
            params = CacheInitParams(
                disable=False,
                req_to_token_pool=ReqToTokenPool(2, 16, "cpu", False),
                token_to_kv_pool_allocator=allocator,
                page_size=1,
            )
            cache = create_tree_cache(self.context(params=params))
            indices = allocator.alloc(4)
            key = RadixKey([1, 2, 3, 4], extra_key="tenant-a")
            cache.insert(InsertParams(key=key, value=indices))
            match = cache.match_prefix(MatchPrefixParams(key=key))
            self.assertEqual(match.device_indices.tolist(), indices.tolist())
            other_key = RadixKey([1, 2, 3, 4], extra_key="tenant-b")
            self.assertEqual(
                len(
                    cache.match_prefix(MatchPrefixParams(key=other_key)).device_indices
                ),
                0,
            )
            cache.evict(EvictParams(num_tokens=4))
            self.assertEqual(allocator.available_size(), 32)
            cache.release_host_resources()
            cache.release_host_resources()


if __name__ == "__main__":
    unittest.main()
