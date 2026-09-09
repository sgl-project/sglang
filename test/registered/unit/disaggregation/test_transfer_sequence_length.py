"""CPU-only coverage of PD length validation without importing GPU backends."""

from __future__ import annotations

import ast
import enum
import hashlib
import logging
import multiprocessing
import random
import sys
import tempfile
import time
import traceback
import unittest
from collections import deque
from contextlib import nullcontext
from datetime import timedelta
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"


def load_definitions(path, names, namespace, class_name=None):
    tree = ast.parse(path.read_text())
    body = tree.body
    if class_name is not None:
        body = next(n for n in body if getattr(n, "name", None) == class_name).body
    selected = [n for n in body if getattr(n, "name", None) in names]
    assert {n.name for n in selected} == set(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias("annotations")], level=0
            )
        ]
        + selected,
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


# Load the CI marker without importing the public API and its GPU dependencies.
_ci = {}
load_definitions(
    ROOT / "python/sglang/test/ci/ci_register.py", ["register_cpu_ci"], _ci
)
register_cpu_ci = _ci["register_cpu_ci"]
register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def run_hicache_rank(rank, init_method, results, step_barrier):
    case = TestTransferSequenceLength()
    phase = "initialization"
    calls = []
    try:
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=5),
        )
        case.setUp()
        case.all_reduce.stop()
        all_reduce = dist.all_reduce

        def tracked_reduce(tensor, op, group):
            calls.append(tensor.tolist())
            return all_reduce(tensor, op=op, group=group)

        restore = case.ns["HiCacheRestoreResult"]
        scenarios = [
            ("pending", restore.READY, restore.PENDING, "success", 9, False, False),
            ("failed", restore.READY, restore.FAILED, "success", 9, False, True),
            (
                "failed_pending",
                restore.FAILED,
                restore.PENDING,
                "success",
                9,
                False,
                True,
            ),
            (
                "failed_rdma",
                restore.FAILED,
                restore.READY,
                "transferring",
                9,
                False,
                True,
            ),
            ("failed_scatter", restore.FAILED, restore.READY, "success", 9, True, True),
            (
                "network_failed_pending",
                restore.READY,
                restore.PENDING,
                "failed",
                9,
                False,
                True,
            ),
            (
                "length_pending",
                restore.READY,
                restore.PENDING,
                "success",
                1,
                False,
                True,
            ),
        ]
        observations = []
        with patch.object(dist, "all_reduce", side_effect=tracked_reduce):
            for staging in (False, True):
                for (
                    name,
                    left,
                    right,
                    network,
                    length,
                    scatter_pending,
                    fails,
                ) in scenarios:
                    if scatter_pending and not staging:
                        continue
                    case.ns["release_kv_cache"].reset_mock()
                    queue, dr, receiver = case.make_queue(
                        case.make_req(), length if rank == 0 else 9, staging
                    )
                    queue.gloo_group = dist.group.WORLD
                    queue.tp_rank = rank
                    queue.metadata_buffers.get_buf = Mock(
                        wraps=queue.metadata_buffers.get_buf
                    )
                    queue.scheduler.enable_decode_hicache = True
                    dr.hicache_restore_status = (left, right)[rank]
                    dr.prefix_match = SimpleNamespace(needs_local_restore=True)
                    dr.hicache_restored_node = object()
                    dr.hicache_load_consumer_index = 0
                    queue.tree_cache = SimpleNamespace(
                        is_load_back_event_done=Mock(return_value=False),
                        cache_controller=SimpleNamespace(
                            layer_done_counter=SimpleNamespace(
                                producer_index=0, num_counters=2
                            )
                        ),
                    )
                    network_rank = 1 if network == "transferring" else 0
                    receiver.poll.return_value = (
                        {
                            "success": case.poll.Success,
                            "transferring": case.poll.Transferring,
                            "failed": case.poll.Failed,
                        }[network]
                        if rank == network_rank
                        else case.poll.Success
                    )
                    queue.staging_handler.is_done.return_value = not (
                        scatter_pending and rank == 1
                    )
                    waits = (
                        restore.PENDING in (left, right)
                        or network == "transferring"
                        or scatter_pending
                    )
                    for tick in range(2 if waits else 1):
                        phase = (name, staging, tick)
                        calls.clear()
                        if tick:
                            queue.tree_cache.is_load_back_event_done.return_value = True
                            if network != "failed":
                                receiver.poll.return_value = case.poll.Success
                            queue.staging_handler.is_done.return_value = True
                        transferred = queue.pop_transferred()
                        waiting = waits and tick == 0
                        expected_outcome = (
                            "waiting" if waiting else "failed" if fails else "committed"
                        )
                        outcome = (
                            "waiting"
                            if queue.queue
                            else "failed"
                            if dr.req.finished_reason is not None
                            else "committed"
                        )
                        state = (name, staging, tick, outcome, len(calls))
                        # Keep the next tick from matching an omitted collective.
                        step_barrier.wait(timeout=10)
                        case.assertEqual(outcome, expected_outcome)
                        if waiting:
                            restore_gate = network == "failed" or (
                                staging and network == "success" and not scatter_pending
                            )
                            expected_shapes = [1, 2] if restore_gate else [1]
                        else:
                            expected_shapes = (
                                [1, 2, 1]
                                if not fails or name == "length_pending"
                                else [1, 2]
                            )
                        case.assertEqual([len(call) for call in calls], expected_shapes)
                        if waiting:
                            case.assertEqual(transferred, [])
                            case.assertEqual(dr.req.output_ids, [])
                            case.assertIsNone(dr.req.finished_reason)
                            receiver.clear.assert_not_called()
                            case.ns["release_kv_cache"].assert_not_called()
                            queue._commit_hicache_local_restore_to_req.assert_not_called()
                            queue.metadata_buffers.get_buf.assert_not_called()
                            case.assertEqual(
                                queue.req_to_metadata_buffer_idx_allocator.available_size(),
                                0,
                            )
                        else:
                            receiver.clear.assert_called_once()
                            case.assertIsNone(dr.kv_receiver)
                            case.assertEqual(
                                queue.req_to_metadata_buffer_idx_allocator.available_size(),
                                1,
                            )
                            if fails:
                                case.assertEqual(transferred, [])
                                case.assertEqual(dr.req.output_ids, [])
                                case.assertEqual(
                                    dr.req.finished_reason.status_code,
                                    HTTPStatus.INTERNAL_SERVER_ERROR,
                                )
                                case.ns["release_kv_cache"].assert_called_once_with(
                                    dr.req, queue.tree_cache, is_insert=False
                                )
                                queue._commit_hicache_local_restore_to_req.assert_not_called()
                            else:
                                case.assertEqual(transferred, [dr.req])
                                case.assertEqual(dr.req.output_ids, [17])
                                case.ns["release_kv_cache"].assert_not_called()
                        observations.append(state)
        results.put((rank, observations, None))
    except Exception:
        results.put(
            (rank, None, f"{phase=}, collectives={calls}\n{traceback.format_exc()}")
        )
    finally:
        case.doCleanups()
        if dist.is_initialized():
            dist.destroy_process_group()


class TestTransferSequenceLength(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(disaggregation_transfer_backend="mooncake")
        self.ns = dict(
            torch=torch,
            enum=enum,
            dist=dist,
            random=random,
            hashlib=hashlib,
            deque=deque,
            nullcontext=nullcontext,
            HTTPStatus=HTTPStatus,
            logger=logging.getLogger(__name__),
            is_npu=lambda: False,
            _is_npu=False,
            get_disagg=lambda: self.config,
            FAKE_BOOTSTRAP_HOST="2.2.2.2",
            envs=SimpleNamespace(
                SGLANG_MOONCAKE_CUSTOM_MEM_POOL=Mock(get=lambda: None),
                SGLANG_TEST_DISAGG_FAILURE_PROB=Mock(get=lambda: 0),
            ),
            release_kv_cache=Mock(),
            Enum=enum.Enum,
        )
        load_definitions(
            SRT / "disaggregation/base/conn.py", ["KVPoll", "StateType"], self.ns
        )
        load_definitions(
            SRT / "disaggregation/decode_hicache_mixin.py",
            ["HiCacheRestoreResult", "HiCacheRestoreGatedKVReceiver"],
            self.ns,
        )
        load_definitions(
            SRT / "managers/schedule_batch.py",
            ["BaseFinishReason", "FINISH_ABORT"],
            self.ns,
        )
        schedule_batch = patch.dict(
            sys.modules,
            {
                "sglang.srt.managers.schedule_batch": SimpleNamespace(
                    FINISH_ABORT=self.ns["FINISH_ABORT"]
                )
            },
        )
        schedule_batch.start()
        self.addCleanup(schedule_batch.stop)
        load_definitions(
            SRT / "disaggregation/utils.py",
            [
                "MetadataBuffers",
                "ReqToMetadataIdxAllocator",
                "_is_fake_transfer",
                "_poll_with_failure_injection",
                "_apply_metadata_gate",
                "_all_reduce_polls",
                "poll_and_all_reduce",
                "poll_and_all_reduce_with_staging",
                "prepare_abort",
            ],
            self.ns,
        )
        load_definitions(
            SRT / "disaggregation/decode.py",
            ["_generate_fake_prefill_handoff_output_id"],
            self.ns,
        )
        load_definitions(
            SRT / "disaggregation/prefill.py",
            ["send_kv_chunk"],
            self.ns,
            class_name="SchedulerDisaggregationPrefillMixin",
        )
        load_definitions(SRT / "mem_cache/common.py", ["kv_to_page_indices"], self.ns)
        load_definitions(
            SRT / "disaggregation/decode.py",
            ["_pre_alloc_fill_len"],
            self.ns,
            class_name="DecodePreallocQueue",
        )
        methods = {}
        load_definitions(
            SRT / "disaggregation/decode.py",
            [
                "_commit_transfer_to_req",
                "_poll_with_metadata_gate",
                "_poll_with_staging",
                "pop_transferred",
            ],
            self.ns,
            class_name="DecodeTransferQueue",
        )
        for name in (
            "_commit_transfer_to_req",
            "_poll_with_metadata_gate",
            "_poll_with_staging",
            "pop_transferred",
        ):
            methods[name] = self.ns[name]
        load_definitions(
            SRT / "disaggregation/decode_hicache_mixin.py",
            ["_process_hicache_local_restores"],
            self.ns,
            class_name="DecodeHiCacheTransferMixin",
        )
        methods["_process_hicache_local_restores"] = self.ns[
            "_process_hicache_local_restores"
        ]
        self.queue_type = type("DecodeTransferQueue", (), methods)
        self.poll = self.ns["KVPoll"]
        self.all_reduce = patch.object(
            dist, "all_reduce", side_effect=lambda *a, **kw: None
        )
        self.reduce = self.all_reduce.start()
        self.addCleanup(self.all_reduce.stop)

    def make_buffers(self):
        return self.ns["MetadataBuffers"](
            size=1,
            hidden_size=2,
            hidden_states_dtype=torch.float32,
            max_sampling_mask_tokens=0,
        )

    def make_req(self, prompt_len=9, prefix_len=0, outputs=(), rebootstrap=False):
        req = SimpleNamespace(
            rid="length-test",
            origin_input_ids=list(range(prompt_len)),
            output_ids=list(outputs),
            bootstrap_host="127.0.0.1",
            bootstrap_room=42,
            vocab_size=100,
            metadata_buffer_index=0,
            cached_tokens=prefix_len,
            cached_tokens_device=prefix_len,
            cached_tokens_host=0,
            cached_tokens_storage=0,
            multimodal_inputs=None,
            return_logprob=False,
            return_sampling_mask=False,
            hidden_states_tensor=None,
            pd_rebootstrap_in_progress=rebootstrap,
            pd_rebootstrap_forced_output_id=99 if rebootstrap else None,
            time_stats=Mock(),
            finished_reason=None,
        )
        fill_len = self.ns["_pre_alloc_fill_len"](req)
        req.kv = SimpleNamespace(kv_committed_len=fill_len)
        req.extend_range = SimpleNamespace(start=prefix_len, end=fill_len)
        return req

    def make_queue(self, req, transferred_len, staging=False):
        buffers = self.make_buffers()
        buffers.output_ids[0, 0] = 17
        buffers.bootstrap_room[0, 0] = req.bootstrap_room
        buffers.cached_tokens[0, 0] = req.cached_tokens
        buffers.cached_tokens[0, 7] = transferred_len
        receiver = Mock(require_staging=staging, abort_notified=False)
        receiver.poll.return_value = self.poll.Success
        dr = SimpleNamespace(
            req=req,
            kv_receiver=receiver,
            metadata_buffer_index=0,
            is_rebootstrap=req.pd_rebootstrap_in_progress,
            hicache_restore_status=self.ns["HiCacheRestoreResult"].READY,
        )
        queue = self.queue_type()
        queue.queue = [dr]
        queue.gloo_group = object()
        queue.tp_rank = 0
        queue.metadata_buffers = buffers
        queue.enable_staging = staging
        queue.staging_handler = Mock()
        queue.staging_handler.is_done.return_value = True
        queue.staging_handler.is_failed.return_value = False
        queue.staging_handler.is_staging_room.return_value = staging
        queue.enable_deferred_kv_release = False
        queue.tree_cache = object()
        queue.req_to_metadata_buffer_idx_allocator = self.ns[
            "ReqToMetadataIdxAllocator"
        ](1)
        self.assertEqual(queue.req_to_metadata_buffer_idx_allocator.alloc(), 0)
        queue.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        queue.scheduler = SimpleNamespace(
            enable_decode_hicache=False,
            enable_hisparse=False,
            metrics_reporter=SimpleNamespace(enable_metrics=False),
            batch_result_processor=Mock(),
            output_streamer=Mock(),
        )
        queue._commit_hicache_local_restore_to_req = Mock()
        queue._clean_hicache_prefetch_resources = Mock()
        return queue, dr, receiver

    def assert_rejected(self, queue, dr, receiver, outputs=()):
        self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(dr.req.output_ids, list(outputs))
        self.assertIsInstance(dr.req.finished_reason, self.ns["FINISH_ABORT"])
        self.assertEqual(
            dr.req.finished_reason.status_code, HTTPStatus.INTERNAL_SERVER_ERROR
        )
        queue._commit_hicache_local_restore_to_req.assert_not_called()
        queue.scheduler.batch_result_processor._maybe_update_reasoning_tokens.assert_not_called()
        queue.scheduler.output_streamer.stream_output.assert_called_once_with(
            [dr.req], False
        )
        self.ns["release_kv_cache"].assert_called_once_with(
            dr.req, queue.tree_cache, is_insert=False
        )
        receiver.clear.assert_called_once()
        self.assertIsNone(dr.kv_receiver)
        self.assertEqual(queue.queue, [])
        self.assertEqual(queue.metadata_buffers.bootstrap_room[0, 0].item(), 0)
        self.assertEqual(queue.req_to_metadata_buffer_idx_allocator.available_size(), 1)
        self.assertEqual(queue.pop_transferred(), [])
        receiver.clear.assert_called_once()

    def test_metadata_publishes_logical_coverage_and_overwrites_reused_slot(self):
        buffers = self.make_buffers()
        ptrs, sizes, item_sizes = buffers.get_buf_infos()
        self.assertEqual(buffers.cached_tokens.shape, (1, 16))
        self.assertEqual(buffers.cached_tokens.dtype, torch.int32)
        self.assertEqual(ptrs[1], buffers.cached_tokens.data_ptr())
        self.assertEqual(item_sizes[1], 64)
        req = self.make_req(prompt_len=9, prefix_len=4, outputs=[17])
        for end, expected in ((9, 9), (12, 9), (5, 5)):
            with self.subTest(end=end):
                req.extend_range.end = end
                buffers.set_buf(req)
                self.assertEqual(buffers.get_buf(0)[1][7].item(), expected)
                self.assertEqual(
                    buffers.cached_tokens[0, :7].tolist(), [4, 4, 0, 0, 0, 0, 0]
                )
                self.assertEqual(buffers.get_buf_infos(), (ptrs, sizes, item_sizes))

    def test_correct_length_includes_cached_prefix_and_excludes_handoff(self):
        req = self.make_req(prefix_len=4)
        queue, dr, receiver = self.make_queue(req, 9)
        self.assertEqual(queue.pop_transferred(), [req])
        self.assertEqual(req.output_ids, [17])
        self.assertIsNone(req.finished_reason)
        self.ns["release_kv_cache"].assert_not_called()
        receiver.clear.assert_called_once()

    def test_only_final_chunk_publishes_total_coverage_before_send(self):
        buffers = self.make_buffers()
        req = self.make_req(prefix_len=4, outputs=[17])
        req.kv.req_pool_idx = 0
        req.start_send_idx = 4
        req.disagg_decode_prefix_len = 4
        req.disagg_kv_sender = Mock()
        req.disagg_kv_sender.should_send_kv_chunk.return_value = True
        prefill = SimpleNamespace(
            enable_staging=False,
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=4,
                translate_kv_indices_for_transfer=lambda indices: indices,
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.arange(12).reshape(1, 12)
            ),
            disagg_metadata_buffers=buffers,
            disagg_prefill_bootstrap_queue=SimpleNamespace(
                kv_manager=SimpleNamespace(kv_args=SimpleNamespace(state_types=[]))
            ),
            disagg_prefill_pending_chunk_rids=set(),
        )
        sends = []
        req.disagg_kv_sender.send.side_effect = lambda *a, **kw: sends.append(
            (buffers.cached_tokens[0, 7].item(), kw["num_kv_tokens"])
        )
        self.ns["send_kv_chunk"](prefill, req, last_chunk=False)
        self.ns["send_kv_chunk"](prefill, req, last_chunk=True)
        self.assertEqual(sends, [(0, 4), (9, 1)])
        self.assertEqual(req.start_send_idx, 9)

    def test_stale_length_is_not_read_before_metadata_arrives(self):
        queue, dr, receiver = self.make_queue(self.make_req(), 1)
        queue.metadata_buffers.bootstrap_room[0, 0] = 0
        self.assertEqual(queue.pop_transferred(), [])
        self.assertIsNone(dr.req.finished_reason)
        receiver.clear.assert_not_called()
        self.ns["release_kv_cache"].assert_not_called()
        self.assertEqual(queue.req_to_metadata_buffer_idx_allocator.available_size(), 0)

    def test_rejects_short_long_delta_and_page_rounded_lengths(self):
        for length in (1, 5, 10, 16, -1):
            with self.subTest(length=length):
                self.ns["release_kv_cache"].reset_mock()
                self.assert_rejected(
                    *self.make_queue(self.make_req(prefix_len=4), length)
                )

    def test_rebootstrap_validates_preallocated_prompt_plus_existing_outputs(self):
        req = self.make_req(outputs=[31, 32], rebootstrap=True)
        queue, dr, receiver = self.make_queue(req, 11)
        self.assertEqual(queue.pop_transferred(), [req])
        self.assertEqual(req.output_ids, [31, 32, 99])
        self.assertIsNone(req.finished_reason)
        req = self.make_req(outputs=[31, 32], rebootstrap=True)
        self.assert_rejected(*self.make_queue(req, 9), outputs=[31, 32])
        self.assertEqual(req.pd_rebootstrap_forced_output_id, 99)

    def test_zero_legacy_length_fails_open_after_slot_reuse(self):
        req = self.make_req()
        queue, dr, receiver = self.make_queue(req, 9)
        self.assertEqual(queue.pop_transferred(), [req])
        # Older prefill sends the entire 16-int row, with spare slots zeroed.
        legacy = self.make_buffers()
        legacy.output_ids[0, 0] = 18
        legacy.bootstrap_room[0, 0] = 43
        for dst, src in (
            (queue.metadata_buffers.cached_tokens, legacy.cached_tokens),
            (queue.metadata_buffers.output_ids, legacy.output_ids),
            (queue.metadata_buffers.bootstrap_room, legacy.bootstrap_room),
        ):
            dst.copy_(src)
        self.assertEqual(queue.req_to_metadata_buffer_idx_allocator.alloc(), 0)
        req = self.make_req(prompt_len=20)
        req.bootstrap_room = 43
        dr.req = req
        dr.kv_receiver = receiver
        queue.queue = [dr]
        self.assertEqual(queue.pop_transferred(), [req])
        self.assertEqual(req.output_ids, [18])
        self.assertEqual(queue.metadata_buffers.cached_tokens[0, 7].item(), 0)

    def test_fake_host_and_fake_backend_bypass_length_validation(self):
        for host in ("2.2.2.2", None):
            with self.subTest(host=host):
                self.config.disaggregation_transfer_backend = "fake"
                req = self.make_req()
                req.bootstrap_host = host
                queue, dr, receiver = self.make_queue(req, 1)
                queue.metadata_buffers.bootstrap_room[0, 0] = 0
                self.assertEqual(queue.pop_transferred(), [req])
                self.assertIsNone(req.finished_reason)
                self.assertEqual(len(req.output_ids), 1)

    def test_length_validation_waits_for_all_ranks_to_finish_transfer(self):
        queue, dr, receiver = self.make_queue(self.make_req(), 1)

        def peer_still_transferring(tensor, op, group):
            self.assertEqual(op, dist.ReduceOp.MIN)
            tensor.fill_(self.poll.Transferring)

        self.reduce.side_effect = peer_still_transferring
        self.assertEqual(queue.pop_transferred(), [])
        self.assertIsNone(dr.req.finished_reason)
        receiver.clear.assert_not_called()
        self.ns["release_kv_cache"].assert_not_called()
        self.assertEqual(len(queue.queue), 1)

    def test_each_rank_rejects_if_any_rank_has_wrong_length(self):
        for staging in (False, True):
            for local_length in (1, 9):
                with self.subTest(staging=staging, local_length=local_length):
                    self.ns["release_kv_cache"].reset_mock()
                    queue, dr, receiver = self.make_queue(
                        self.make_req(), local_length, staging
                    )
                    calls = []

                    def consensus(tensor, op, group):
                        self.assertEqual(op, dist.ReduceOp.MIN)
                        self.assertIs(group, queue.gloo_group)
                        calls.append(tensor.tolist())
                        if len(calls) == 2:
                            self.assertEqual(dr.req.output_ids, [])
                            self.assertEqual(
                                tensor.tolist(),
                                [
                                    self.poll.Success
                                    if local_length == 9
                                    else self.poll.Failed
                                ],
                            )
                            tensor.fill_(self.poll.Failed)

                    self.reduce.side_effect = consensus
                    self.assert_rejected(queue, dr, receiver)
                    self.assertEqual(len(calls), 2)

    @unittest.skipUnless(dist.is_gloo_available(), "requires CPU Gloo")
    def test_two_rank_hicache_collective_order(self):
        ctx = multiprocessing.get_context("spawn")
        results = ctx.Queue()
        step_barrier = ctx.Barrier(2)
        with tempfile.TemporaryDirectory() as directory:
            init_method = f"file://{directory}/gloo"
            processes = [
                ctx.Process(
                    target=run_hicache_rank,
                    args=(rank, init_method, results, step_barrier),
                )
                for rank in range(2)
            ]
            try:
                for process in processes:
                    process.start()
                deadline = time.monotonic() + 45
                reports = [
                    results.get(timeout=max(0.1, deadline - time.monotonic()))
                    for _ in processes
                ]
                for process in processes:
                    process.join(timeout=5)
                    self.assertEqual(process.exitcode, 0)
                self.assertEqual({rank for rank, _, _ in reports}, {0, 1})
                for _, _, error in reports:
                    self.assertIsNone(error, error)
                self.assertEqual(reports[0][1], reports[1][1])
                self.assertEqual(len(reports[0][1]), 24)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.kill()
                    process.join(timeout=5)
                results.close()
                results.join_thread()


if __name__ == "__main__":
    unittest.main()
