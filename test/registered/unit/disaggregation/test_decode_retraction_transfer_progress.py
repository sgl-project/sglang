"""PD decode: requests whose KV has arrived keep moving while a retracted request waits.

Drives the real decode-side queues (admission, transfer polling, prebuilt batches,
retraction and resume) on CPU; only the KV transport and the model forward are stubbed.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeReqToTokenPool,
    DecodeRequest,
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.disaggregation.utils import (
    MetadataBuffers,
    ReqToMetadataIdxAllocator,
    TransferBackend,
)
from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.scheduler_components.new_token_ratio_tracker import (
    NewTokenRatioTracker,
)
from sglang.srt.managers.scheduler_components.request_list import (
    IndexedReqList,
    KeyedReqList,
    ListKind,
    move,
)
from sglang.srt.managers.scheduler_components.request_retirer import RequestRetirer
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.static_kv_storage import StaticPoolShape, kv_storage_of
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_disagg
from sglang.srt.sampling.sampler_params import SamplerParamsPool
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import idle_penalty_status_pool
from sglang.test.test_utils import CustomTestCase, enter_override

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

SMALL_POOL = 16384
ROWS = 16
EOS = 0
TOKEN = 7


class _Receiver:
    """Stands in for the KV transport: the prefill's write lands at step `done_at`,
    together with the room id and first token it writes into the metadata slot."""

    abort_notified = True
    kv_mgr = SimpleNamespace(enable_deferred_decode_kv_release=False)

    def __init__(self, node, room, done_at):
        self.node = node
        self.room = room
        self.done_at = done_at
        self.decode_req = None
        self.prefill_info = None
        self.landed = False

    def poll(self):
        if self.node.step_count < self.done_at:
            return KVPoll.Transferring
        if not self.landed:
            idx = self.decode_req.metadata_buffer_index
            self.node.metadata_buffers.bootstrap_room[idx, 0] = self.room
            self.node.metadata_buffers.output_ids[idx, 0] = TOKEN
            self.landed = True
        return KVPoll.Success

    def ensure_abort_notified(self, force_arm=False):
        pass

    def failure_exception(self):
        raise RuntimeError("stub transfer failed")

    def abort(self):
        pass

    def clear(self):
        pass


class _DecodeNode:
    """A single-rank PD decode scheduler: real queues, pools, batches and retraction."""

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.step_count = 0
        self.retractions = []
        self.next_room = 1000
        self.req_to_token_pool = DecodeReqToTokenPool(
            size=ROWS,
            max_context_len=pool_size,
            device="cpu",
            enable_memory_saver=False,
            pre_alloc_size=ROWS,
        )
        kv_pool = MHATokenToKVPool(
            size=pool_size,
            page_size=1,
            dtype=torch.float16,
            head_num=1,
            head_dim=4,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        self.allocator = TokenToKVPoolAllocator(
            pool_size,
            dtype=torch.float16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        kv_storage_of(self.req_to_token_pool, self.allocator).bind_window_retention(
            None, {}
        )
        storage = kv_storage_of(
            self.req_to_token_pool,
            self.allocator,
            shape=StaticPoolShape(
                has_window=False, has_state=False, token_capacity=pool_size
            ),
        )
        self.tree_cache = UnifiedRadixCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.allocator,
                page_size=1,
                tree_slots=tuple(storage.tree_slots()),
            )
        )
        self.metadata_buffers = MetadataBuffers(
            size=4 * ROWS,
            hidden_size=2,
            hidden_states_dtype=torch.float32,
            max_sampling_mask_tokens=0,
        )
        self.scheduler = self._build_scheduler()
        meta_alloc = ReqToMetadataIdxAllocator(4 * ROWS)
        self.scheduler.disagg_decode_transfer_queue = self._build_transfer_queue(
            meta_alloc
        )
        self.scheduler.disagg_decode_prealloc_queue = self._build_prealloc_queue(
            meta_alloc
        )

    def _build_scheduler(self):
        s = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.DECODE,
            enable_overlap=False,
            enable_hisparse=False,
            enable_lora=False,
            enable_decode_hicache=False,
            enable_priority_scheduling=False,
            enable_skip_finishing_decode=False,
            scheduler_stage_metrics=None,
            uocr_runtime=None,
            decode_offload_manager=None,
            hisparse_coordinator=None,
            kv_checksum_computer=None,
            forward_ct=0,
            last_batch=None,
            tree_cache=self.tree_cache,
            req_to_token_pool=self.req_to_token_pool,
            penalty_status_pool=idle_penalty_status_pool(),
            sampler_params_pool=SamplerParamsPool(
                num_rows=self.req_to_token_pool.req_to_token.shape[0], device="cpu"
            ),
            token_to_kv_pool_allocator=self.allocator,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            max_running_requests=ROWS,
            model_config=SimpleNamespace(
                context_len=self.pool_size,
                vocab_size=16,
                token_id_vocab_size=16,
                is_encoder_decoder=False,
                is_multimodal=False,
                num_key_value_heads=1,
                hidden_size=2,
                hf_eos_token_id={EOS},
                think_end_ids=None,
                self_opened_think_start_ids=None,
            ),
            metrics_reporter=SimpleNamespace(
                num_retracted_reqs=0, enable_metrics=False
            ),
            grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
            new_token_ratio_tracker=NewTokenRatioTracker.from_config(),
            invariant_checker=MagicMock(),
            waiting_queue=IndexedReqList(ListKind.WAITING),
            running_reqs=KeyedReqList(ListKind.DECODING),
            output_streamer=SimpleNamespace(stream_output=lambda *a, **k: None),
            running_batch=ScheduleBatch(reqs=[], device="cpu"),
        )
        s.request_retirer = RequestRetirer(
            tree_cache=self.tree_cache,
            beam_coordinator=SimpleNamespace(retire_group=lambda req: None),
            hisparse_coordinator=None,
            uocr_runtime=None,
            penalty_status_pool=s.penalty_status_pool,
            send_abort_output=lambda abort_req, req: None,
            stream_output=lambda *a, **k: None,
        )
        s._set_or_validate_priority = lambda req: True
        s._reqs_finishing_inflight = lambda batch: Scheduler._reqs_finishing_inflight(
            s, batch
        )
        s._add_request_to_queue = lambda req, is_retracted=False: (
            Scheduler._add_request_to_queue(s, req, is_retracted=is_retracted)
        )
        s._preallocate_new_requests = lambda: (
            SchedulerDisaggregationDecodeMixin._preallocate_new_requests(s)
        )
        processor = SimpleNamespace(
            model_config=s.model_config, logprob_result_processor=None
        )
        processor._maybe_update_reasoning_tokens = lambda req, tok: (
            SchedulerBatchResultProcessor._maybe_update_reasoning_tokens(
                processor, req, tok
            )
        )
        processor.commit_handoff_token = lambda req, tok, **kw: (
            SchedulerBatchResultProcessor.commit_handoff_token(
                processor, req, tok, **kw
            )
        )
        s.batch_result_processor = processor
        return s

    def _build_transfer_queue(self, meta_alloc):
        tq = DecodeTransferQueue.__new__(DecodeTransferQueue)
        tq.queue = IndexedReqList(
            ListKind.PD_RECEIVING, abort=tq._abort_receiving, req_of=lambda x: x.req
        )
        tq.gloo_group = None
        tq.req_to_metadata_buffer_idx_allocator = meta_alloc
        tq.tp_rank = 0
        tq.metadata_buffers = self.metadata_buffers
        tq.scheduler = self.scheduler
        tq.enable_host_receive = False
        tq.tree_cache = self.tree_cache
        tq.spec_algorithm = SpeculativeAlgorithm.NONE
        tq._skipped_checksum_signatures = set()
        tq.enable_staging = False
        tq.staging_handler = None
        tq.enable_deferred_kv_release = True
        tq.deferred_kv_release_timeout = 30
        tq._deferred_releases = IndexedReqList(
            ListKind.PD_RELEASING, req_of=lambda x: x[0].req
        )
        return tq

    def _build_prealloc_queue(self, meta_alloc):
        pq = DecodePreallocQueue.__new__(DecodePreallocQueue)
        pq.req_to_token_pool = self.req_to_token_pool
        pq.token_to_kv_pool_allocator = self.allocator
        pq.token_to_kv_pool = self.allocator.get_kvcache()
        pq.draft_token_to_kv_pool = None
        pq.draft_kv_ring = None
        pq.is_mla_backend = False
        pq.metadata_buffers = self.metadata_buffers
        pq.req_to_metadata_buffer_idx_allocator = meta_alloc
        pq.scheduler = self.scheduler
        pq.transfer_queue = self.scheduler.disagg_decode_transfer_queue
        pq.tree_cache = self.tree_cache
        pq.host_pool = None
        pq.host_reserved_tokens = 0
        pq.host_window_entries = []
        pq.gloo_group = None
        pq._num_published_destinations = 0
        pq._defer_publish = False
        pq._pending_publish = []
        pq._publish_stream = None
        pq.tp_rank = 0
        pq.gpu_id = 0
        pq.bootstrap_port = 8998
        pq.max_total_num_tokens = self.pool_size
        pq.pp_rank = 0
        pq.pp_size = 1
        pq.num_reserved_decode_tokens = get_disagg().num_reserved_decode_tokens
        pq.transfer_backend = TransferBackend.MOONCAKE
        pq.queue = IndexedReqList(
            ListKind.PD_PREALLOC, abort=pq._abort_prealloc, req_of=lambda x: x.req
        )
        pq.retracted_queue = IndexedReqList(
            ListKind.PD_RETRACTED, abort=pq._abort_retracted
        )
        pq._prefill_dp_rank_queries = {}
        pq.held_rebootstrap_reqs = IndexedReqList(
            ListKind.PD_HELD, abort=pq._abort_held
        )
        pq.enable_staging = False
        pq.kv_manager = SimpleNamespace(
            kv_args=SimpleNamespace(state_types=[], state_sub_pools=[])
        )
        pq.kv_checksum_signature = 0
        pq._host_staged_admission_stats = None
        # The bootstrap handshake and the send of the destination pages to prefill.
        pq._resolve_pending_reqs = lambda: None
        pq._update_handshake_waiters = lambda *a, **k: None

        def publish_destination(*, decode_req, **_):
            decode_req.metadata_buffer_index = meta_alloc.alloc()

        pq._publish_destination = publish_destination
        return pq

    def arrive(self, rid, prompt_len, max_new_tokens, done_at):
        sampling_params = SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0)
        sampling_params.normalize(None)
        self.next_room += 1
        req = Req(
            rid,
            "",
            array("q", [TOKEN] * prompt_len),
            sampling_params,
            eos_token_ids={EOS},
            bootstrap_host="127.0.0.1",
            bootstrap_port=8998,
            bootstrap_room=self.next_room,
            disagg_mode=DisaggregationMode.DECODE,
            vocab_size=16,
        )
        receiver = _Receiver(self, self.next_room, done_at)
        decode_req = DecodeRequest(req=req, kv_receiver=receiver)
        receiver.decode_req = decode_req
        decode_req.waiting_for_input = True
        move([decode_req], self.scheduler.disagg_decode_prealloc_queue.queue)
        return req

    def _admit_prebuilt(self):
        s = self.scheduler
        new_batch = SchedulerDisaggregationDecodeMixin._get_new_prebuilt_batch(
            s, s.running_batch
        )
        if new_batch is None:
            return
        new_batch.process_prebuilt(SimpleNamespace(stash=lambda *a, **k: None))
        for req in new_batch.reqs:
            req.update_finish_state()
            if req.finished():
                s.request_retirer.release(req, checkpoint=True)
        new_batch.filter_batch()
        if new_batch.is_empty():
            return
        if s.running_batch.is_empty():
            s.running_batch = new_batch
        else:
            s.running_batch.merge_batch(new_batch)

    def step(self):
        """One normal decode loop iteration; the forward emits one token per request."""
        s = self.scheduler
        SchedulerDisaggregationDecodeMixin.process_decode_queue(s)
        self._admit_prebuilt()
        batch = s.running_batch
        if not batch.is_empty():
            batch = Scheduler.update_running_batch(s, batch)
            if batch is not None:
                s.running_batch = batch
        retracted = self.retracted_rids()
        if retracted and (not self.retractions or self.retractions[-1][1] != retracted):
            self.retractions.append((self.step_count, retracted, self.transfer_rids()))
        if batch is not None and not batch.is_empty():
            for req in batch.reqs:
                req.output_ids.append(TOKEN)
                req.update_finish_state()
                if req.finished():
                    s.request_retirer.release(req, checkpoint=True)
            batch.filter_batch()
            s.last_batch = batch
        else:
            s.last_batch = None
        self.step_count += 1

    def transfer_rids(self):
        queue = self.scheduler.disagg_decode_transfer_queue.queue
        return [decode_req.req.rid for decode_req in queue.items()]

    def retracted_rids(self):
        queue = self.scheduler.disagg_decode_prealloc_queue.retracted_queue
        return [req.rid for req in queue]


class TestDecodeRetractionTransferProgress(CustomTestCase):
    def setUp(self):
        # The transfer queue sizes its poll collectives from the gloo group.
        world_size = patch("torch.distributed.get_world_size", return_value=1)
        world_size.start()
        self.addCleanup(world_size.stop)
        enter_override(
            self,
            get_context().override_server_args(
                disaggregation_mode="decode", disable_radix_cache=True, page_size=1
            ),
        )
        get_context().override(
            "test", disaggregation_decode_retraction_backup="cpu_tensor"
        )

    def _run(self, *, pool_size, transfer_done_at, steps):
        node = _DecodeNode(pool_size)
        reqs = [
            node.arrive("short", 100, 1200, done_at=0),
            node.arrive("long", 8000, 1500, done_at=0),
        ]
        node.step()
        reqs += [
            node.arrive("x1", 3000, 1200, done_at=transfer_done_at),
            node.arrive("x2", 3200, 1200, done_at=transfer_done_at),
        ]
        retracted_at_arrival = None
        for _ in range(steps):
            node.step()
            if node.step_count == transfer_done_at + 1:
                retracted_at_arrival = node.retracted_rids()
            if all(req.finished() for req in reqs):
                break
        return node, reqs, retracted_at_arrival

    def _assert_all_finished(self, node, reqs):
        for req in reqs:
            self.assertIsInstance(req.finished_reason, FINISH_LENGTH, req.rid)
        self.assertEqual(node.allocator.available_size(), node.pool_size)

    def test_arrived_transfers_run_while_a_retracted_request_waits(self):
        """KV that arrives while a request sits in the retracted queue must still be
        admitted, or the retracted request and the transfers wait on each other."""
        node, reqs, retracted_at_arrival = self._run(
            pool_size=SMALL_POOL, transfer_done_at=1100, steps=4000
        )
        # The retraction happened while both transfers were still in flight.
        self.assertEqual(node.retractions[0][1:], (["long"], ["x1", "x2"]))
        self.assertEqual(retracted_at_arrival, ["long"])
        self.assertEqual(node.transfer_rids(), [])
        self.assertEqual(node.retracted_rids(), [])
        self._assert_all_finished(node, reqs)

    def test_retraction_without_pending_transfers_resumes(self):
        """Control: the same small pool, the transfers land before the pool fills."""
        node, reqs, _ = self._run(pool_size=SMALL_POOL, transfer_done_at=2, steps=4000)
        self.assertTrue(node.retractions)
        for _, _, transfers in node.retractions:
            self.assertEqual(transfers, [])
        self._assert_all_finished(node, reqs)

    def test_large_pool_never_retracts(self):
        """Control: a pool large enough for every request."""
        node, reqs, _ = self._run(
            pool_size=4 * SMALL_POOL, transfer_done_at=1100, steps=4000
        )
        self.assertEqual(node.retractions, [])
        self._assert_all_finished(node, reqs)


if __name__ == "__main__":
    unittest.main()
