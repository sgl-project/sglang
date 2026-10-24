"""CPU contracts for portable request reconstruction, not a GPU restore test.

The store, reconstruction, batch mutations and input counter methods are real.
Only device allocation/readback, scheduler collectives and sampling hooks are
mocked. Run examples/runtime/token_kv_restore.py separately on real engines.
"""

import copy
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers import token_kv_snapshot as snapshot
from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def cpu_batch(requests, rows):
    # Native filter/merge run on CPU tensors. No physical KV pool is fabricated.
    batch = object.__new__(ScheduleBatch)
    batch.reqs = list(requests)
    batch.beam_tail = None
    batch.req_pool_indices = torch.tensor(rows, dtype=torch.int64)
    batch.req_pool_indices_cpu = batch.req_pool_indices.clone()
    batch.seq_lens = torch.full((len(rows),), 3, dtype=torch.int64)
    batch.seq_lens_cpu = batch.seq_lens.clone()
    batch.orig_seq_lens = batch.seq_lens.to(torch.int32)
    batch.seq_lens_sum = int(batch.seq_lens.sum())
    batch.input_ids = batch.out_cache_loc = None
    batch.model_config = NS(is_encoder_decoder=False)
    batch.sampling_info = Mock()
    batch.return_logprob = batch.has_grammar = False
    batch.return_hidden_states_mode = CaptureHiddenMode.NULL
    batch.return_hidden_states = batch.is_prefill_only = False
    batch.multimodal_inputs = [None] * len(rows)
    batch.spec_info = None
    return batch


class TokenKVSnapshotContracts(CustomTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = snapshot.TokenKVSnapshotStore(self.temp.name)
        self.identity = dict(
            weight_version="v0",
            tp_size=2,
            kv_dtype="bfloat16",
            layout="full_attention_and_gdn",
        )
        params = SamplingParams(max_new_tokens=8, is_normalized=True)
        self.metadata = dict(
            request_schema_version=2,
            rid="original-source-rid",
            origin_input_ids=[11, 22],
            output_ids=[33, 44],
            computed_tokens=3,
            origin_input_ids_sha256=snapshot._json_digest([11, 22]),
            pending_input_token=44,
            remaining_generation_tokens=6,
            resolved_sampling_params=snapshot._resolved_sampling_params(params),
            source_binding=dict(
                source_engine_instance_id="source-instance", generation=7
            ),
            request_flags=dict(return_logprob=True),
            output_logprobs=[-0.125, -0.25],
            output_logprob_token_ids=[33, 44],
            output_weight_versions=[dict(version="v0", start=0, end=2)],
            temp_scaled_logprobs=False,
            top_p_normalized_logprobs=False,
            decode_batch_idx=9,
            reasoning_state=dict(
                is_reasoning_over=True, reasoning_tokens=0, think_end_match_len=0
            ),
            source_stream_cursors={},
        )
        self.state = dict(
            attention_kv=[torch.arange(4, dtype=torch.float32)],
            mamba_conv=[torch.zeros(2)],
            mamba_temporal=[torch.ones(2)],
        )
        raw = io.BytesIO()
        torch.save(self.state, raw)
        for rank in range(2):
            self.store.write_rank(
                "portable",
                tp_rank=rank,
                tp_size=2,
                identity=self.identity,
                request_metadata=self.metadata,
                tensor_metadata=snapshot._state_receipt(self.state),
                payload=raw.getvalue(),
            )
        self.store.commit("portable", tp_size=2)
        self.descriptor = dict(
            storage_dir=self.temp.name,
            snapshot_id="portable",
            identity=copy.deepcopy(self.identity),
            expected_weight_version="v0",
            source_binding=copy.deepcopy(self.metadata["source_binding"]),
            manifest_ref=self.store.receipt("portable")["manifest_ref"],
        )

    def tearDown(self):
        self.temp.cleanup()

    def request_input(self, rid="new-target-rid"):
        return NS(
            rid=rid,
            token_kv_restore=copy.deepcopy(self.descriptor),
            stream=True,
            text="must be replaced",
            input_ids=None,
            input_embeds=None,
            contains_mm_input=lambda: False,
            session_params=None,
            session_id=None,
            lora_path=None,
            lora_id=None,
            custom_logit_processor=None,
            positional_embed_overrides=None,
            return_hidden_states=False,
            return_routed_experts=False,
            return_indexer_topk=False,
            return_sampling_mask=False,
            sampling_logprobs_mode=None,
        )

    def manager(self, rank=0, running=None):
        manager = object.__new__(snapshot.TokenKVSnapshotManager)
        manager.scheduler = NS(ps=NS(tp_rank=rank, tp_size=2), running_batch=running)
        parallel = patch.object(
            snapshot,
            "get_parallel",
            create=True,
            return_value=NS(tp_rank=rank, tp_size=2),
        )
        parallel.start()
        self.addCleanup(parallel.stop)
        manager.engine_instance_id = "fresh-target-instance"
        manager.resume_metrics, manager.portable_restore_by_rid = {}, {}
        manager._check_runtime = Mock(return_value=[])
        manager._drain_overlap_results = Mock()
        manager._identity = Mock(return_value=copy.deepcopy(self.identity))

        # Actual TP agreement is tested by the real store. No distributed
        # executor is started in this method-level fixture.
        def reject(error):
            if error:
                raise ValueError(error)

        manager._rank_errors = reject

        def allocate(req, computed, payload, expected):
            self.assertEqual(computed, 3)
            self.assertEqual(snapshot._state_receipt(payload), expected)
            req.req_pool_idx = 10 + rank
            req.kv.req_pool_idx = req.req_pool_idx
            return [req.req_pool_idx], expected

        manager._allocate_and_load = Mock(side_effect=allocate)
        manager._prepare_portable_decode_batch = Mock(
            side_effect=lambda req, rows, *_: cpu_batch([req], rows)
        )
        manager._slots = Mock(return_value=dict(request_row_generation=1))
        return manager

    def native_request(self, rid="new-target-rid"):
        return NS(
            rid=rid,
            origin_input_ids=[11, 22],
            sampling_params=snapshot.sampling_params_from_snapshot(
                self.metadata["resolved_sampling_params"]
            ),
            logprob=NS(),
            req_pool_idx=None,
            mamba_pool_idx=None,
            kv=ReqKvInfo(),
        )

    def test_public_sampling_default_projection(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["request_flags"].update(
            sampling_logprobs_mode="selected",
            logprob_start_len=2,
            top_logprobs_num=0,
            token_ids_logprob=None,
            require_reasoning=False,
        )
        raw = io.BytesIO()
        torch.save(self.state, raw)
        for rank in range(2):
            self.store.write_rank(
                "selected-default",
                tp_rank=rank,
                tp_size=2,
                identity=self.identity,
                request_metadata=metadata,
                tensor_metadata=snapshot._state_receipt(self.state),
                payload=raw.getvalue(),
            )
        self.store.commit("selected-default", tp_size=2)
        descriptor = copy.deepcopy(self.descriptor)
        descriptor.update(
            snapshot_id="selected-default",
            manifest_ref=self.store.receipt("selected-default")["manifest_ref"],
        )
        before = {
            p.name: p.read_bytes()
            for p in (Path(self.temp.name) / "selected-default").iterdir()
            if p.is_file()
        }
        obj = self.request_input()
        obj.token_kv_restore = descriptor
        with patch.dict("os.environ", {"SGLANG_ENABLE_TOKEN_KV_SNAPSHOT": "1"}):
            snapshot.prepare_portable_generate_input(obj)
        self.assertIsNone(obj.sampling_logprobs_mode)
        self.assertIs(obj.return_sampling_mask, False)
        self.assertEqual(obj.input_ids, metadata["origin_input_ids"])
        for key, value in metadata["request_flags"].items():
            if key != "sampling_logprobs_mode":
                self.assertEqual(getattr(obj, key), value)
        self.assertEqual(
            before,
            {
                p.name: p.read_bytes()
                for p in (Path(self.temp.name) / "selected-default").iterdir()
                if p.is_file()
            },
        )

    def test_public_explicit_sampling_mode_remains_rejected(self):
        for mode in ("selected", "all"):
            obj = self.request_input()
            obj.sampling_logprobs_mode = mode
            with (
                self.subTest(mode=mode),
                patch.dict("os.environ", {"SGLANG_ENABLE_TOKEN_KV_SNAPSHOT": "1"}),
                self.assertRaisesRegex(ValueError, "explicit sampling_logprobs_mode"),
            ):
                snapshot.prepare_portable_generate_input(obj)

    def test_new_rid_reconstruction_preserves_both_tp_histories(self):
        with patch.dict("os.environ", {"SGLANG_ENABLE_TOKEN_KV_SNAPSHOT": "1"}):
            obj = self.request_input()
            snapshot.prepare_portable_generate_input(obj)
            self.assertEqual(obj.input_ids, [11, 22])
            self.assertEqual(
                obj.sampling_params, self.metadata["resolved_sampling_params"]
            )
            for invalid_rid in ("original-source-rid", ["target-a", "target-b"]):
                with (
                    self.subTest(rid=invalid_rid),
                    self.assertRaisesRegex(ValueError, "new single target request ID"),
                ):
                    snapshot.prepare_portable_generate_input(
                        self.request_input(invalid_rid)
                    )
        for rank in range(2):
            with self.subTest(tp_rank=rank):
                manager, req = self.manager(rank), self.native_request()
                receipt = manager.import_for_generation(req, self.descriptor)
                self.assertEqual(list(req.output_ids), [33, 44])
                self.assertEqual(req.logprob.output_token_logprobs_val, [-0.125, -0.25])
                self.assertEqual(req.logprob.output_token_logprobs_idx, [33, 44])
                self.assertEqual(
                    req._token_kv_output_version_spans,
                    [dict(version="v0", start=0, end=2)],
                )
                self.assertEqual(
                    (receipt["source_rid"], receipt["target_rid"]),
                    ("original-source-rid", "new-target-rid"),
                )
                self.assertEqual(receipt["remaining_generation_tokens"], 6)
                self.assertEqual(req.sampling_params.max_new_tokens, 8)
                self.assertEqual(
                    manager.resume_metrics[req.rid]["actual_input_counter_status"],
                    "ARMED_NOT_YET_EXECUTED",
                )
                self.assertEqual(
                    manager.resume_metrics[req.rid]["actual_completed_forward_iters"],
                    [],
                )

    def test_version_layout_and_history_rejected_before_device_allocation(self):
        for change in ({"weight_version": "v1"}, {"layout": "different_layout"}):
            with self.subTest(change=change):
                manager = self.manager()
                descriptor = copy.deepcopy(self.descriptor)
                descriptor["identity"].update(change)
                with self.assertRaises(ValueError):
                    manager.import_for_generation(self.native_request(), descriptor)
                manager._allocate_and_load.assert_not_called()
        manager, req = self.manager(), self.native_request()
        req.origin_input_ids = [11, 99]
        with self.assertRaisesRegex(ValueError, "input differs"):
            manager.import_for_generation(req, self.descriptor)
        manager._allocate_and_load.assert_not_called()

    def test_missing_tp_payload_rejected_before_allocation(self):
        from pathlib import Path

        (Path(self.temp.name) / "portable/rank-1.bin").unlink()
        manager = self.manager(rank=1)
        with self.assertRaises(ValueError):
            manager.import_for_generation(self.native_request(), self.descriptor)
        manager._allocate_and_load.assert_not_called()

    def test_naturally_empty_target_drops_stale_native_rows(self):
        old = cpu_batch([self.native_request("finished")], [7])
        old.filter_batch(keep_indices=[])
        self.assertTrue(old.is_empty())
        self.assertEqual(old.seq_lens.numel(), 1)  # Native filter leaves stale rows.
        imported = cpu_batch([self.native_request()], [10])
        manager = self.manager(running=old)
        merged = manager._merge_portable_decode_batch(imported)
        self.assertIs(merged, imported)
        self.assertEqual(len(merged.reqs), merged.seq_lens.numel())
        self.assertEqual(merged.req_pool_indices.tolist(), [10])

    def test_nonempty_target_preserves_existing_member_and_rows(self):
        old_req = self.native_request("unrelated-active-rid")
        old_req.output_ids = [71, 72]
        old_req.logprob.output_token_logprobs_val = [-0.3, -0.4]
        old = cpu_batch([old_req], [7])
        imported_req = self.native_request()
        manager = self.manager(running=old)
        merged = manager._merge_portable_decode_batch(cpu_batch([imported_req], [10]))
        self.assertIs(merged, old)
        self.assertEqual(
            [req.rid for req in merged.reqs], ["unrelated-active-rid", "new-target-rid"]
        )
        self.assertEqual(merged.req_pool_indices.tolist(), [7, 10])
        self.assertEqual(len(merged.reqs), merged.seq_lens.numel())
        self.assertEqual(old_req.output_ids, [71, 72])
        self.assertEqual(old_req.logprob.output_token_logprobs_val, [-0.3, -0.4])

    def test_counter_requires_processed_result_not_initial_zero_or_capture(self):
        manager, req = self.manager(), self.native_request()
        manager.import_for_generation(req, self.descriptor)
        metrics = manager.resume_metrics[req.rid]
        batch = NS(
            reqs=[req],
            seq_lens_cpu=torch.tensor([4]),
            forward_mode=ForwardMode.DECODE,
            forward_iter=17,
        )
        with patch.object(
            snapshot, "get_serving", return_value=NS(weight_version="v0")
        ):
            intervals = manager.capture_native_input_intervals(batch)
            self.assertEqual(intervals[0][3:6], (3, 4, 3))
            batch.seq_lens_cpu[0] = 99  # Overlap may mutate the carrier later.
            self.assertEqual(intervals[0][3:6], (3, 4, 3))
            self.assertEqual(
                metrics["actual_input_counter_status"], "ARMED_NOT_YET_EXECUTED"
            )
            manager.observe_native_input_intervals(intervals)
            manager.observe_native_input_intervals(intervals)
        self.assertEqual(
            metrics["actual_input_counter_status"], "OBSERVED_COMPLETED_NATIVE_FORWARD"
        )
        self.assertEqual(metrics["actual_recomputed_tokens"], 0)
        self.assertEqual(metrics["actual_processed_input_tokens"], 1)
        self.assertEqual(metrics["actual_completed_forward_iters"], [17])
        with patch.object(
            snapshot, "get_serving", return_value=NS(weight_version="v1")
        ):
            manager.observe_native_input_intervals(intervals)
        self.assertEqual(metrics["actual_input_counter_status"], "UNAVAILABLE")
        self.assertIsNone(metrics["actual_recomputed_tokens"])
        # A real overlap with committed positions must be counted; zero cannot
        # be a constant stub hidden behind a completed status.
        other, req = self.manager(), self.native_request("overlap-history-rid")
        other.import_for_generation(req, self.descriptor)
        req.prefix_len, req.extend_end = 1, 5  # Current-main native range.
        req.extend_range = NS(start=1, end=5)  # Pinned older source adapter.
        batch = NS(reqs=[req], forward_mode=ForwardMode.EXTEND, forward_iter=18)
        with patch.object(
            snapshot, "get_serving", return_value=NS(weight_version="v0")
        ):
            intervals = other.capture_native_input_intervals(batch)
            other.observe_native_input_intervals(intervals)
        self.assertEqual(other.resume_metrics[req.rid]["actual_recomputed_tokens"], 2)
        self.assertEqual(
            other.resume_metrics[req.rid]["actual_processed_input_tokens"], 4
        )

    def test_allocation_keeps_original_native_resource_record_and_mamba_rows(self):
        manager, req = self.manager(), self.native_request()
        original_kv = req.kv
        mamba_row = torch.tensor([9])
        track_rows = torch.tensor([17, 18])

        def allocate_row(requests):
            self.assertIs(requests[0].kv, original_kv)
            req.kv.req_pool_idx = 2
            req.kv.mamba_pool_idx = mamba_row
            req.kv.mamba_ping_pong_track_buffer = track_rows
            return [2]

        pool = NS(
            alloc=Mock(side_effect=allocate_row),
            req_to_token=torch.zeros((4, 8), dtype=torch.int32),
        )
        allocator = NS(page_size=1, alloc=Mock(return_value=torch.tensor([4, 5, 6])))
        root, lock = object(), object()
        cache = NS(
            disable=False,
            root_node_handle=Mock(return_value=root),
            lock=Mock(return_value=lock),
        )
        manager.scheduler.req_to_token_pool = pool
        manager.scheduler.token_to_kv_pool_allocator = allocator
        manager.scheduler.tree_cache = cache
        req.extra_key = None
        req.refresh_fill_ids = Mock()
        manager._load = Mock()
        manager._copy = Mock(return_value=self.state)
        expected = snapshot._state_receipt(self.state)
        rows, observed = snapshot.TokenKVSnapshotManager._allocate_and_load(
            manager, req, 3, self.state, expected
        )
        self.assertEqual(rows, [2])
        self.assertEqual(observed, expected)
        self.assertIs(req.kv, original_kv)
        self.assertIs(req.kv.mamba_pool_idx, mamba_row)
        self.assertIs(req.kv.mamba_ping_pong_track_buffer, track_rows)
        self.assertEqual(pool.req_to_token[2, :3].tolist(), [4, 5, 6])
        self.assertEqual((req.kv.kv_committed_len, req.kv.kv_allocated_len), (3, 3))
        self.assertEqual((req.prefix_len, req.extend_end), (3, 3))
        req.refresh_fill_ids.assert_called_once_with()
        self.assertIs(req.lock, lock)
        cache.lock.assert_called_once_with(root)
        for kv in (
            ReqKvInfo(req_pool_idx=1),
            ReqKvInfo(mamba_pool_idx=torch.tensor([1])),
        ):
            with self.subTest(holds_kv=kv.holds_kv, holds_mamba=kv.holds_mamba):
                req.kv = kv
                pool.alloc.reset_mock()
                allocator.alloc.reset_mock()
                with self.assertRaisesRegex(
                    ValueError, "unowned request resource record"
                ):
                    snapshot.TokenKVSnapshotManager._allocate_and_load(
                        manager, req, 3, self.state, expected
                    )
                pool.alloc.assert_not_called()
                allocator.alloc.assert_not_called()

    def test_release_rejects_changed_allocator_generation_before_any_free(self):
        manager, req = self.manager(), self.native_request()
        manager._request = Mock(return_value=(req, cpu_batch([req], [2]), 3, 0))
        manager._request_metadata = Mock(return_value=copy.deepcopy(self.metadata))
        manager._slots = Mock(return_value=dict(request_row_generation=8))
        manager._read = Mock()
        record = dict(
            released=False,
            metadata=copy.deepcopy(self.metadata),
            slots=dict(request_row_generation=7),
        )
        request = NS(
            source_binding=self.metadata["source_binding"], release_full_gpu_pool=False
        )
        with patch.object(snapshot, "release_kv_cache", create=True) as release:
            with self.assertRaisesRegex(ValueError, "generation changed"):
                manager._release(request, self.identity, self.store, record)
            release.assert_not_called()
        manager._read.assert_not_called()
        self.assertFalse(record["released"])

    def test_waiting_cancel_uses_native_request_cleanup_not_rid_prefix(self):
        # Single-rank control-flow contract; distributed agreement is not mocked
        # into a claim that a TP collective ran.
        manager, req = self.manager(), self.native_request("exact-waiting-rid")
        identity = {**self.identity, "tp_size": 1}
        manager._identity.return_value = identity
        req.output_ids, req.prefix_len = [], 0
        req.finished = lambda: False
        binding = dict(
            request_id=req.rid,
            source_engine_instance_id=manager.engine_instance_id,
            source_execution_generation=7,
            trajectory_id="trajectory",
            source_owner_id="owner",
            source_lease_id="lease",
            policy_identity_sha256="policy-digest",
            origin_input_ids_sha256=snapshot._json_digest(req.origin_input_ids),
        )
        original = {
            key: value
            for key, value in binding.items()
            if key != "source_engine_instance_id"
        }
        original.update(schema_version=1, policy_version="v0")
        req.sampling_params.custom_params = {
            "trajectory_source_request_v1": json.dumps(original)
        }
        scheduler = manager.scheduler
        scheduler._engine_paused = True
        scheduler.running_batch = cpu_batch([], [])
        scheduler.waiting_queue, scheduler.chunked_req = [req], None
        scheduler._release_aborted_request = Mock()
        scheduler.ipc_channels = NS(send_to_tokenizer=NS(send_output=Mock()))
        manager._drain_overlap_results.return_value = 0
        request = NS(
            snapshot_id="pending",
            storage_dir=self.temp.name,
            target_rid=req.rid,
            rid=None,
            source_binding=binding,
            identity=identity,
            expected_weight_version="v0",
        )
        with patch.object(
            snapshot, "get_parallel", return_value=NS(tp_rank=0, tp_size=1)
        ):
            result = manager._preempt(request, [])
        scheduler._release_aborted_request.assert_called_once_with(req)
        self.assertEqual(scheduler.waiting_queue, [])
        self.assertEqual(result["source_disposition"], "PENDING_CANCELLED")
        self.assertTrue(result["pending_removed"])
        self.assertFalse(result["kv_allocated"])


class PublicRestoreEvidenceContracts(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        path = (
            Path(__file__).resolve().parents[4] / "examples/runtime/token_kv_restore.py"
        )
        spec = importlib.util.spec_from_file_location("public_token_kv_restore", path)
        cls.public = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.public)

    def test_complete_unique_tp_receipts_required(self):
        complete = dict(ranks=[dict(tp_rank=i, status="PASS") for i in range(2)])
        self.assertEqual(set(self.public.rank_rows(complete, 2)), {0, 1})
        for rows in (
            [complete["ranks"][0]],
            [complete["ranks"][0], complete["ranks"][0]],
            [complete["ranks"][0], dict(tp_rank=2, status="PASS")],
            [complete["ranks"][0], dict(tp_rank=1, status="FAILED")],
        ):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.public.rank_rows(dict(ranks=rows), 2)

    def test_native_token_logprob_and_version_evidence_must_be_complete(self):
        row = dict(
            meta_info=dict(
                output_token_logprobs=[[-0.125, 33], [-0.25, 44]],
                weight_version="v0",
                weight_versions=[dict(version="v0", start=0, end=2)],
            )
        )
        output = self.public.token_output(row)
        self.assertEqual(output["token_ids"], [33, 44])
        self.assertEqual(output["logprobs"], [-0.125, -0.25])
        for changes in (
            dict(weight_version="v1"),
            dict(weight_versions=[dict(version="v0", start=0, end=1)]),
            dict(output_token_logprobs=[[float("nan"), 33], [-0.25, 44]]),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.public.token_output(
                    dict(meta_info={**row["meta_info"], **changes})
                )


if __name__ == "__main__":
    unittest.main()


class CapacityLayoutContracts(CustomTestCase):
    def setUp(self):
        self.serving = NS(weight_version="v0")
        self.parallel = NS(tp_size=2, tp_rank=0)
        for name, value in (
            ("get_serving", self.serving),
            ("get_parallel", self.parallel),
        ):
            patched = patch.object(snapshot, name, return_value=value)
            patched.start()
            self.addCleanup(patched.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.model_dir = Path(self.tmp.name)
        self.config = {
            "model_type": "hybrid",
            "text_config": {
                "model_type": "qwen3_mamba",
                "num_hidden_layers": 2,
                "layer_types": ["mamba", "full_attention"],
            },
        }
        self.config_file = self.model_dir / "config.json"
        self.config_file.write_text(json.dumps(self.config))

    def tearDown(self):
        self.tmp.cleanup()

    def _hf_config(self):
        return NS(to_dict=lambda: copy.deepcopy(self.config), _commit_hash="rev0")

    def _tensor(self, capacity, dtype=torch.float32):
        return torch.zeros((2, capacity, 3, 4), dtype=dtype)

    def _pool(self, capacity):
        return NS(
            mamba_cache=NS(
                conv=[self._tensor(capacity), self._tensor(capacity)],
                temporal=self._tensor(capacity),
            ),
            mamba_layer_ids=[0],
        )

    def _allocator(self, page_size=16):
        return NS(
            page_size=page_size,
            _kvcache=NS(full_attention_layer_id_mapping={1: 1}),
            alloc=Mock(),
        )

    def _scheduler(self, capacity=461):
        req_pool = NS(
            alloc=Mock(),
            free=Mock(),
            free_mamba_cache=Mock(),
            req_to_token=torch.zeros((8, 128), dtype=torch.int32),
            mamba_pool=self._pool(capacity),
            req_to_token_pool=None,
        )
        return NS(
            model_config=NS(hf_config=self._hf_config()),
            server_args=NS(model_path=self.model_dir),
            token_to_kv_pool_allocator=self._allocator(),
            req_to_token_pool=req_pool,
            tree_cache=NS(disable=True),
        )

    def _manager_for(self, capacity=461):
        manager = object.__new__(snapshot.TokenKVSnapshotManager)
        manager.scheduler = self._scheduler(capacity)
        return manager

    def _identity_with_patches(self, manager):
        return manager._identity()

    def test_capacity_461_466_identity_and_layout(self):
        left, right = self._manager_for(461), self._manager_for(466)
        left_id = self._identity_with_patches(left)
        right_id = self._identity_with_patches(right)
        self.assertEqual(left_id, right_id)
        left._validate_identity(NS(expected_weight_version="v0", identity=right_id))
        right._validate_identity(NS(expected_weight_version="v0", identity=left_id))
        self.assertEqual(right._cache_layout()["conv"][0]["shape"][1], 466)
        full = left._cache_layout()
        self.assertEqual(full["conv"][0]["shape"][1], 461)
        self.assertEqual(full["temporal"]["shape"][1], 461)
        self.assertEqual(left_id["cache_layout"]["conv"][0]["shape"][1], 1)
        self.assertEqual(left_id["cache_layout"]["temporal"]["shape"][1], 1)
        self.assertTrue(
            torch.equal(
                left.scheduler.req_to_token_pool.mamba_pool.mamba_cache.conv[0],
                self._tensor(461),
            )
        )

    def test_layout_mutations_rejected(self):
        identity = self._manager_for()._identity()
        for name in ("conv_dtype", "temporal_dtype", "conv_dim", "temporal_dim"):
            manager = self._manager_for(466)
            cache = manager.scheduler.req_to_token_pool.mamba_pool.mamba_cache
            if name == "conv_dtype":
                cache.conv[0] = cache.conv[0].to(torch.float64)
            elif name == "temporal_dtype":
                cache.temporal = cache.temporal.to(torch.bfloat16)
            elif name == "conv_dim":
                cache.conv[0] = torch.zeros((2, 466, 5, 4))
            else:
                cache.temporal = torch.zeros((2, 466, 3, 5))
            with self.subTest(mutation=name), self.assertRaises(ValueError):
                manager._validate_identity(
                    NS(expected_weight_version="v0", identity=identity)
                )

    def test_identity_mismatches_rejected(self):
        manager = self._manager_for()
        identity = manager._identity()
        for key, value in (
            ("tp_size", 1),
            ("weight_version", "v1"),
            ("model_config_sha256", "0" * 64),
            ("engine_source_sha256", "0" * 64),
            ("cache_layout_sha256", "0" * 64),
        ):
            bad = copy.deepcopy(identity)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                manager._validate_identity(
                    NS(expected_weight_version="v0", identity=bad)
                )
        for name in ("page_size", "attention_layers", "mamba_layers", "model_config"):
            manager = self._manager_for()
            scheduler = manager.scheduler
            if name == "page_size":
                scheduler.token_to_kv_pool_allocator.page_size = 32
            elif name == "attention_layers":
                scheduler.token_to_kv_pool_allocator._kvcache.full_attention_layer_id_mapping = {
                    2: 2
                }
            elif name == "mamba_layers":
                scheduler.req_to_token_pool.mamba_pool.mamba_layer_ids = [9]
            else:
                scheduler.model_config.hf_config.to_dict = lambda: {
                    "model_type": "other"
                }
            with self.subTest(actual_mismatch=name), self.assertRaises(ValueError):
                manager._validate_identity(
                    NS(expected_weight_version="v0", identity=identity)
                )
        self.serving.weight_version = "v1"
        with self.assertRaisesRegex(ValueError, "weight version"):
            self._manager_for()._validate_identity(
                NS(expected_weight_version="v0", identity=identity)
            )

    def test_native_layout_validation_rejects_malformed_metadata(self):
        manager = self._manager_for()
        pool = manager.scheduler.req_to_token_pool.mamba_pool
        malformed = [torch.zeros((3,)), torch.zeros((2, 0, 3, 4))]
        for tensor in malformed:
            with self.subTest(tensor=tensor.shape):
                pool.mamba_cache.conv[0] = tensor
                with self.assertRaises(ValueError):
                    manager._cache_layout()
        pool.mamba_cache.temporal = torch.zeros((3,))
        with self.assertRaises(ValueError):
            manager._cache_layout()

    def test_allocate_and_load_failure_guards(self):
        manager = self._manager_for()
        scheduler = manager.scheduler
        pool, allocator = (
            scheduler.req_to_token_pool,
            scheduler.token_to_kv_pool_allocator,
        )
        req = NS(
            kv=NS(
                holds_kv=False,
                holds_mamba=False,
                req_pool_idx=0,
                kv_committed_len=0,
                kv_allocated_len=0,
                cache_protected_len=0,
                cache_inserted_len=0,
                mamba_needs_clear=True,
            ),
            prefix_len=0,
            extra_key="x",
            refresh_fill_ids=Mock(),
            extend_end=0,
        )
        pool.alloc.return_value = None
        with patch.object(type(manager), "_load", create=True) as load:
            with self.assertRaises(ValueError):
                manager._allocate_and_load(req, 4, None, None)
            load.assert_not_called()
        self.assertEqual(allocator.alloc.call_count, 0)

        pool.alloc.return_value = [1]
        allocator.alloc.return_value = None
        pool.free_mamba_cache.reset_mock()
        pool.free.reset_mock()
        with patch.object(type(manager), "_load", create=True) as load:
            with self.assertRaises(ValueError):
                manager._allocate_and_load(req, 4, None, None)
            load.assert_not_called()
        pool.free_mamba_cache.assert_called_once_with(req)
        pool.free.assert_called_once_with(req)

        pool.alloc.reset_mock()
        allocator.alloc.reset_mock()
        req.kv.holds_mamba = True
        with patch.object(type(manager), "_load", create=True) as load:
            with self.assertRaises(ValueError):
                manager._allocate_and_load(req, 4, None, None)
            load.assert_not_called()
            pool.alloc.assert_not_called()
            allocator.alloc.assert_not_called()
