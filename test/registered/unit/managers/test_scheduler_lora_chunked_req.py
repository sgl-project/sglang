import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import MatchResult
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class MockLoRAManager:
    def __init__(self, max_loras_per_batch: int):
        self.memory_pool = MagicMock()
        self.max_loras_per_batch = max_loras_per_batch
        self.num_pinned_loras = 0
        self.loras = None
        self.lora_modules = None
        self.lora_refs = {}
        self.embed_tokens_module = None
        self.lm_head_module = None

    def validate_lora_batch(self, lora_ids):
        return LoRAManager.validate_lora_batch(self, lora_ids)

    def fetch_new_loras(self, new_loras):
        return LoRAManager.fetch_new_loras(self, new_loras)


class MockRunningBatch:
    def __init__(self):
        self.reqs = []
        self.batch_is_full = False


class MockTreeCache:
    def supports_mamba(self):
        return False

    def supports_swa(self):
        return False

    def evictable_size(self):
        return 0

    def inc_lock_ref(self, _node):
        return None

    def dec_lock_ref(self, _node, params=None):
        return None

    def match_prefix(self, params):
        return MatchResult(
            device_indices=torch.empty((0,), dtype=torch.int64),
            last_device_node=None,
            last_host_node=None,
            host_hit_length=0,
            mamba_branching_seqlen=None,
            cache_protected_len=0,
        )


class TestSchedulerLoRAChunkedPrefill(unittest.TestCase):
    def setUp(self):
        server_args = ServerArgs(model_path="dummy")
        set_global_server_args_for_scheduler(server_args)
        self.server_args = server_args

    def make_req(self, rid: str, token_ids: list[int], lora_id: str) -> Req:
        return Req(
            rid=rid,
            origin_input_text=rid,
            origin_input_ids=token_ids,
            sampling_params=SamplingParams(max_new_tokens=1),
            lora_id=lora_id,
        )

    def make_scheduler(self, lora_manager: MockLoRAManager) -> Scheduler:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.server_args = self.server_args
        scheduler.enable_lora = True
        scheduler.enable_lora_overlap_loading = False
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_priority_preemption = False
        scheduler.enable_priority_scheduling = False
        scheduler.enable_hicache_storage = False
        scheduler.enable_dynamic_chunking = False
        scheduler.enable_overlap = False
        scheduler.chunked_prefill_size = 2
        scheduler.page_size = 1
        scheduler.new_token_ratio = 1.0
        scheduler.max_prefill_tokens = 64
        scheduler.max_prefill_bs = 0
        scheduler.max_running_requests = None
        scheduler.is_mixed_chunk = False
        scheduler.priority_scheduling_preemption_threshold = 0
        scheduler.truncation_align_size = None
        scheduler.disaggregation_mode = None
        scheduler.dllm_config = None
        scheduler.spec_algorithm = None
        scheduler.model_config = None
        scheduler.policy = SimpleNamespace(
            calc_priority=lambda waiting_queue, running_batch: None
        )
        scheduler.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        scheduler.tree_cache = MockTreeCache()
        scheduler.req_to_token_pool = SimpleNamespace(device="cpu")
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            available_size=lambda: 16,
        )
        scheduler.running_batch = MockRunningBatch()
        scheduler.get_num_allocatable_reqs = lambda running_bs: 16
        scheduler._get_num_pending_tokens = lambda chunk_deduct=0: 0
        scheduler.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(lora_manager=lora_manager)
        )
        return scheduler

    def _run_get_new_batch_prefill(self, scheduler):
        with (
            patch.object(ScheduleBatch, "prepare_for_extend", autospec=True),
            patch(
                "sglang.srt.managers.schedule_policy.is_nsa_prefill_cp_in_seq_split",
                return_value=False,
            ),
            patch(
                "sglang.srt.managers.schedule_policy.is_prefill_context_parallel_enabled",
                return_value=False,
            ),
        ):
            return scheduler._get_new_batch_prefill_raw(
                prefill_delayer_single_pass=None
            )

    def test_chunked_request_lora_admission_rejected(self):
        lora_manager = MockLoRAManager(max_loras_per_batch=1)

        scheduler = self.make_scheduler(lora_manager)
        scheduler.chunked_req = self.make_req(
            rid="chunked", token_ids=[1, 2, 3, 4], lora_id="lora-chunked"
        )
        scheduler.chunked_req.prefix_indices = torch.tensor(
            [0, 1, 2], dtype=torch.int64
        )
        scheduler.chunked_req.cache_protected_len = 3
        scheduler.waiting_queue = [
            self.make_req(rid="waiting", token_ids=[5], lora_id="lora-waiting")
        ]

        # With only one slot, lora-waiting must be rejected because lora-chunked should occupy it
        batch = self._run_get_new_batch_prefill(scheduler)

        self.assertIsNotNone(batch)
        self.assertEqual([req.lora_id for req in batch.reqs], ["lora-chunked"])

        # Ensure that the batch was able to be processed
        lora_manager.fetch_new_loras({req.lora_id for req in batch.reqs})
        lora_manager.memory_pool.prepare_lora_batch.assert_called_once()

    def test_chunked_request_lora_admission(self):
        lora_manager = MockLoRAManager(max_loras_per_batch=2)

        scheduler = self.make_scheduler(lora_manager)
        scheduler.chunked_req = self.make_req(
            rid="chunked", token_ids=[1, 2, 3, 4], lora_id="lora-chunked"
        )
        scheduler.chunked_req.prefix_indices = torch.tensor(
            [0, 1, 2], dtype=torch.int64
        )
        scheduler.chunked_req.cache_protected_len = 3
        scheduler.waiting_queue = [
            self.make_req(rid="waiting", token_ids=[5], lora_id="lora-waiting")
        ]

        # With two slots, lora-chunked and lora-waiting should be accepted
        batch = self._run_get_new_batch_prefill(scheduler)

        self.assertIsNotNone(batch)
        self.assertEqual(
            [req.lora_id for req in batch.reqs], ["lora-chunked", "lora-waiting"]
        )

        # Ensure that the batch was able to be processed
        lora_manager.fetch_new_loras({req.lora_id for req in batch.reqs})
        lora_manager.memory_pool.prepare_lora_batch.assert_called_once()


if __name__ == "__main__":
    unittest.main()
