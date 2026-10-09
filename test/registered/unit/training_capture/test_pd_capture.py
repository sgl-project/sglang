"""P/D handoff preserves raw teacher ownership and fences partial samples."""

import tempfile
import threading
import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import msgspec
import numpy as np
import torch
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.training_capture.catalog import HTTPCaptureCatalog
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.pd_capture import (
    DecodeCaptureCoordinator,
    DecodeCaptureMixin,
    PrefillCaptureCoordinator,
)
from sglang.srt.training_capture.pd_protocol import (
    MAX_HANDOFF_BYTES,
    CaptureTransferContext,
    PrefillTeacherHandoff,
    decode_handoff,
    encode_handoff,
)
from sglang.srt.training_capture.protocol import ContractError, validate_tensors
from sglang.srt.training_capture.snapshot import SnapshotMetadata, assemble_snapshot
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_utils import (
    BufferStore,
    CaptureTestRequest,
    FakeReplicateConfig,
    Registrar,
    make_snapshot,
    read_snapshot,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPDCapture(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.catalog = TestCaptureCatalog()
        manifest, _ = make_snapshot()
        self.store = MooncakeSnapshotStore(BufferStore(), FakeReplicateConfig())
        config = CaptureConfig(
            dataset_id=manifest.dataset_id,
            model_id="test",
            producer_revision="test",
            selected_layer_ids=manifest.kv.selected_layer_ids,
            catalog_endpoint=self.catalog.endpoint,
            journal_directory=self.directory.name + "/journal",
            store=StoreSetup(
                local_hostname="localhost", master_server_addr="localhost:1"
            ),
            max_sample_tokens=8,
            max_inflight_samples=1,
            max_host_bytes=2 << 20,
            sample_ratio=1.0,
        )
        self.buffers = {
            f"target_{c}.{g.layer_id}": torch.randn(16, 2, 4).bfloat16()
            for g in manifest.kv.layers
            for c in ("k", "v")
        }
        self.decode = DecodeCaptureCoordinator(
            config=config,
            teacher=manifest.teacher,
            kv=manifest.kv,
            exporter=SelectedLayerKVExporter(manifest.kv, self.buffers),
            req_to_token=SimpleNamespace(req_to_token=torch.tensor([[7, 3, 6, 1]])),
            store=self.store,
            catalog=HTTPCaptureCatalog(self.catalog.endpoint),
            pin_memory=False,
            capture_mode="pd_autoregressive",
        )
        self.prefill = PrefillCaptureCoordinator(
            config=config, teacher=manifest.teacher, kv=manifest.kv
        )

    def tearDown(self):
        self.prefill.close()
        self.decode.close()
        self.catalog.close()
        self.directory.cleanup()

    def wait_until(self, predicate):
        deadline = time.monotonic() + 5
        while not predicate():
            self.assertLess(time.monotonic(), deadline, self.decode.stats())
            time.sleep(0.01)

    def begin(self, max_new_tokens=1):
        self.wait_until(lambda: len(self.decode.available) == 1)
        req = CaptureTestRequest("decode", max_new_tokens)
        req.bootstrap_room, req.req_pool_idx = 19, 0
        payload = self.decode.begin_pd_transfer(req)
        self.assertIsNotNone(payload)
        return req, payload

    def prefill_request(self, payload, **overrides):
        req = CaptureTestRequest("prefill", 1)
        req.bootstrap_room = 19
        req.bootstrap_host = "127.0.0.1"
        req.disagg_kv_sender = SimpleNamespace(
            get_training_capture_context=lambda: payload
        )
        for name, value in overrides.items():
            setattr(req, name, value)
        self.prefill.before_forward([req])
        return req

    def handoff(self, payload):
        req = self.prefill_request(payload)
        self.assertIsNotNone(req.training_capture_pd)
        logits = torch.randn(1, 256, generator=torch.Generator().manual_seed(917))
        raw = logits.clone()
        batch_logits = torch.cat([logits + 10, logits])
        # An unselected batch row changes the packed position offset.
        other = CaptureTestRequest("unselected")
        self.prefill.after_forward(
            SimpleNamespace(reqs=[other, req], seq_lens_cpu=[1, 2]),
            SimpleNamespace(
                extend_seq_lens_cpu=[1, 2], positions=torch.tensor([0, 0, 1])
            ),
            SimpleNamespace(next_token_logits=batch_logits),
        )
        batch_logits.fill_(-1000)
        req.output_ids = [10]
        return self.prefill.finish_handoff(req), raw

    def test_single_token_response_publishes_owned_prompt_and_raw_teacher(self):
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        req, context = self.begin()
        payload, raw = self.handoff(context)
        expected = {name: buf[[7, 3]].clone() for name, buf in self.buffers.items()}
        req.output_ids = [10]
        self.decode.accept_pd_handoff(req, payload)
        for buffer in self.buffers.values():
            buffer.zero_()
        req.finished_len, req.finished_reason = 1, FINISH_LENGTH(1)
        self.decode.on_release(req)
        manifest, tensors = read_snapshot(
            self.store, self.catalog.wait_publications(1)[0]
        )
        self.assertEqual(manifest.provenance.capture_mode, "pd_autoregressive")
        self.assertEqual(tensors["token_ids"].tolist(), [3, 4, 10])
        self.assertEqual(tensors["loss_mask"].tolist(), [0, 0, 1])
        self.assertEqual(tensors["kv_valid"].tolist(), [1, 1, 0])
        values, ids = raw.topk(128)
        torch.testing.assert_close(
            tensors["teacher_topk_logits"], values, rtol=0, atol=0
        )
        torch.testing.assert_close(tensors["teacher_topk_ids"], ids.int())
        torch.testing.assert_close(tensors["teacher_logsumexp"], raw.logsumexp(-1))
        for name, value in expected.items():
            torch.testing.assert_close(tensors[name], value, rtol=0, atol=0)

    def test_prefill_pause_drains_but_abort_epoch_survives_resume(self):
        _, payload = self.begin()
        old = self.prefill_request(payload)
        state = old.training_capture_pd
        self.prefill.control("pause")
        self.assertTrue(self.prefill._live_state(state))
        excluded = self.prefill_request(payload)
        self.assertIsNone(excluded.training_capture_pd)
        self.prefill.control("abort")
        self.prefill.control("resume")
        self.assertFalse(self.prefill._live_state(state))
        self.prefill.before_forward([excluded])
        self.assertIsNone(excluded.training_capture_pd)
        with patch("sglang.srt.training_capture.pd_capture.capture_teacher") as teacher:
            batch = SimpleNamespace(reqs=[old], seq_lens_cpu=[2])
            self.prefill.after_forward(
                batch,
                SimpleNamespace(extend_seq_lens_cpu=[2]),
                SimpleNamespace(next_token_logits=torch.zeros(1, 256)),
            )
            teacher.assert_not_called()
            self.assertIsNone(self.prefill.pack_pp_handoffs(batch, [10]))
        self.assertIsNone(self.prefill.finish_handoff(old))
        fresh = self.prefill_request(payload)
        self.assertTrue(self.prefill._live_state(fresh.training_capture_pd))

    def test_decode_abort_rejects_late_prefill_handoff(self):
        req, payload = self.begin()
        record = req.training_capture_context
        handoff, _ = self.handoff(payload)
        self.decode.control("abort")
        self.decode.control("resume")
        req.output_ids = [10]
        self.decode.accept_pd_handoff(req, handoff)
        self.assertIsNone(req.training_capture_context)
        self.wait_until(lambda: record.state == "done")
        self.assertFalse(self.catalog.publications)
        self.assertEqual(
            self.catalog.captures[record.lease.capture_id]["state"], "FAILED"
        )

    def test_prefill_weight_update_fences_each_teacher_stage_after_resume(self):
        for stage in ("admitted", "copied", "encoded"):
            with self.subTest(stage=stage):
                self.prefill = PrefillCaptureCoordinator(
                    config=self.prefill.config,
                    teacher=self.prefill.teacher,
                    kv=self.prefill.kv,
                )
                decode, payload = self.begin()
                record = decode.training_capture_context
                req = self.prefill_request(payload)
                req.output_ids = [10]
                batch = SimpleNamespace(reqs=[req], seq_lens_cpu=[2])
                if stage != "admitted":
                    self.prefill.after_forward(
                        batch,
                        SimpleNamespace(
                            extend_seq_lens_cpu=[2], positions=torch.tensor([0, 1])
                        ),
                        SimpleNamespace(
                            next_token_logits=torch.arange(256).float()[None]
                        ),
                    )
                    self.assertIsNotNone(req.training_capture_pd.teacher)
                if stage == "encoded":
                    self.assertIsNotNone(self.prefill.pack_pp_handoffs(batch, [10]))
                    self.assertIsNotNone(req.training_capture_pd.handoff)
                self.prefill.disable("target_weights_update")
                self.prefill.control("resume")
                self.assertEqual(self.prefill.disabled_reason, "target_weights_update")
                self.assertIsNone(self.prefill.pack_pp_handoffs(batch, [10]))
                self.assertIsNone(self.prefill.finish_handoff(req))
                fresh = self.prefill_request(payload)
                self.assertIsNone(fresh.training_capture_pd)
                decode.output_ids = [10]
                self.decode.accept_pd_handoff(decode, None)
                self.wait_until(lambda record=record: record.state == "done")
                self.assertEqual(
                    self.catalog.captures[record.lease.capture_id]["reason"],
                    "pd_handoff_failed",
                )
                self.assertFalse(self.catalog.publications)

    def test_decode_weight_update_rejects_late_teacher_and_new_admission(self):
        req, payload = self.begin()
        record = req.training_capture_context
        handoff, _ = self.handoff(payload)
        self.decode.disable("target_weights_update")
        self.decode.control("resume")
        req.output_ids = [10]
        self.decode.accept_pd_handoff(req, handoff)
        self.assertIsNone(req.training_capture_context)
        self.wait_until(lambda: record.state == "done")
        self.assertEqual(
            self.catalog.captures[record.lease.capture_id]["reason"],
            "target_weights_update",
        )
        self.assertFalse(self.catalog.publications)
        fresh = CaptureTestRequest("after-weight-update", 1)
        self.assertIsNone(self.decode.begin_pd_transfer(fresh))
        self.assertIsNone(fresh.training_capture_context)
        self.assertEqual(self.decode.counters["admitted"], 1)

    def test_prefill_abort_fences_materialized_and_encoded_teacher_after_resume(self):
        decode, payload = self.begin()
        record = decode.training_capture_context
        for encoded in (False, True):
            with self.subTest(encoded=encoded):
                req = self.prefill_request(payload)
                req.output_ids = [10]
                batch = SimpleNamespace(reqs=[req], seq_lens_cpu=[2])
                self.prefill.after_forward(
                    batch,
                    SimpleNamespace(extend_seq_lens_cpu=[2], positions=torch.arange(2)),
                    SimpleNamespace(next_token_logits=torch.arange(256).float()[None]),
                )
                state = req.training_capture_pd
                self.assertIsNotNone(state.teacher)
                if encoded:
                    wire = self.prefill.pack_pp_handoffs(batch, [10])
                    self.assertIsNotNone(wire)
                    self.assertIsNotNone(state.handoff)
                    self.assertIsNone(state.teacher)
                self.prefill.control("abort")
                self.prefill.control("resume")
                self.assertIsNone(self.prefill.pack_pp_handoffs(batch, [10]))
                if encoded:
                    self.prefill.accept_pp_handoffs(batch, wire)
                self.assertIsNone(self.prefill.finish_handoff(req))
                self.assertIsNone(req.training_capture_pd)
                self.assertEqual(req.output_ids, [10])
        self.assertEqual(self.prefill.counters["pd_teacher_copied"], 2)
        self.assertEqual(self.prefill.counters["pd_handoff_ready"], 0)
        self.assertEqual(self.prefill.counters["pd_pp_handoff_received"], 0)
        decode.output_ids = [10]
        self.decode.accept_pd_handoff(decode, None)
        self.wait_until(lambda: record.state == "done")
        self.assertFalse(self.catalog.publications)
        self.assertEqual(
            self.catalog.captures[record.lease.capture_id]["state"], "FAILED"
        )

    def test_pd_verify_trims_accepted_tail_after_length_or_eos(self):
        from sglang.srt.managers.schedule_batch import (
            FINISH_LENGTH,
            FINISH_MATCHED_TOKEN,
        )

        for output_length, reason in (
            (3, FINISH_LENGTH(3)),
            (2, FINISH_MATCHED_TOKEN(20)),
        ):
            with self.subTest(output_length=output_length):
                self.decode.capture_mode = "pd_speculative_accepted_target_path"
                req, wire = self.begin(max_new_tokens=3)
                payload, prefill_raw = self.handoff(wire)
                req.output_ids, req.eos_token_ids = [10], {20}
                self.decode.accept_pd_handoff(req, payload)
                logits = torch.randn(
                    4, 256, generator=torch.Generator().manual_seed(818)
                )
                raw = logits.clone()
                ticket = self.decode.after_verify_forward(
                    SimpleNamespace(reqs=[req], seq_lens_cpu=[2]),
                    SimpleNamespace(
                        input_ids=torch.tensor([10, 20, 30, 40]),
                        positions=torch.arange(2, 6),
                        out_cache_loc=torch.tensor([6, 1, 9, 5]),
                    ),
                    SimpleNamespace(next_token_logits=logits),
                    width=4,
                    can_run_cuda_graph=False,
                )
                logits.zero_()
                self.decode.after_verify_accept(
                    ticket,
                    commit_lens=torch.tensor([3]),
                    out_tokens=torch.tensor([[20, 30, 55, 0]]),
                )
                req.output_ids = [10, 20, 30, 55]
                req.finished_len, req.finished_reason = output_length, reason
                previous = len(self.catalog.publications)
                self.decode.on_release(req)
                manifest, tensors = read_snapshot(
                    self.store, self.catalog.wait_publications(previous + 1)[-1]
                )
                self.assertEqual(
                    manifest.provenance.capture_mode, self.decode.capture_mode
                )
                self.assertEqual(
                    tensors["token_ids"].tolist(),
                    [3, 4, 10, 20, 30][: 2 + output_length],
                )
                self.assertEqual(
                    tensors["kv_valid"].tolist(), [1] * (2 + output_length)
                )
                expected = torch.cat(
                    (
                        prefill_raw.topk(128).values,
                        raw[: output_length - 1].topk(128).values,
                    )
                )
                torch.testing.assert_close(
                    tensors["teacher_topk_logits"], expected, rtol=0, atol=0
                )
                for name, buffer in self.buffers.items():
                    torch.testing.assert_close(
                        tensors[name],
                        buffer[[7, 3, 6, 1, 9][: 2 + output_length]],
                        rtol=0,
                        atol=0,
                    )

    def test_missing_corrupt_or_replayed_teacher_fails_only_capture(self):
        for failure in (
            "missing",
            "oversized",
            "fence",
            "token",
            "duplicate_ids",
            "nan",
        ):
            with self.subTest(failure=failure):
                req, context = self.begin()
                record = req.training_capture_context
                payload, _ = self.handoff(context)
                handoff = decode_handoff(payload, PrefillTeacherHandoff)
                if failure == "missing":
                    payload = None
                elif failure == "oversized":
                    payload = b"x" * (MAX_HANDOFF_BYTES + 1)
                else:
                    fields = {
                        "fence": {
                            "context": msgspec.structs.replace(
                                handoff.context,
                                fencing_token=handoff.context.fencing_token + 1,
                            )
                        },
                        "token": {"output_token_id": 11},
                        "duplicate_ids": {"topk_ids": [0] * 128},
                        "nan": {"logsumexp": float("nan")},
                    }[failure]
                    payload = encode_handoff(msgspec.structs.replace(handoff, **fields))
                req.output_ids = [10]
                self.decode.accept_pd_handoff(req, payload)
                self.wait_until(lambda record=record: record.state == "done")
                self.assertIsNone(req.training_capture_context)
                self.assertIsNone(req.training_capture_pd)
                self.assertFalse(req.finished())
                self.assertEqual(req.output_ids, [10])
                self.assertFalse(self.catalog.publications)

    def test_prefill_rejects_prompt_identity_sampling_and_room_mismatches(self):
        _, payload = self.begin()
        context = decode_handoff(payload, CaptureTransferContext)
        for fields in (
            {"prompt_length": 3},
            {"prompt_sha256": "0" * 64},
            {"contract_sha256": "0" * 64},
            {"sampling_sha256": "0" * 64},
            {"bootstrap_room": 20},
        ):
            with self.subTest(fields=fields):
                req = self.prefill_request(
                    encode_handoff(msgspec.structs.replace(context, **fields))
                )
                self.assertIsNone(req.training_capture_pd)
                self.assertIsNone(self.prefill.finish_handoff(req))
        with self.assertRaises(ContractError):
            decode_handoff(b"bad", CaptureTransferContext)

    def test_warmup_fake_sender_does_not_require_a_capture_interface(self):
        from sglang.srt.disaggregation.utils import FAKE_BOOTSTRAP_HOST

        req = CaptureTestRequest("warmup")
        req.disagg_kv_sender = object()
        req.bootstrap_host = FAKE_BOOTSTRAP_HOST
        self.prefill.before_forward([req])
        self.assertIsNone(req.training_capture_pd)
        self.assertIsNone(self.prefill.finish_handoff(req))
        self.assertNotIn("pd_context_rejected", self.prefill.counters)

    def test_prefill_unsupported_inputs_cannot_reuse_an_ordinary_decode_lease(self):
        _, payload = self.begin()
        for field in (
            "input_embeds",
            "multimodal_inputs",
            "positional_embed_overrides",
            "session",
            "lora_id",
            "custom_logit_processor",
        ):
            with self.subTest(field=field):
                req = self.prefill_request(payload, **{field: object()})
                self.assertIsNone(req.training_capture_pd)
                self.assertIsNone(self.prefill.finish_handoff(req))

    def test_handoff_inbox_ignores_late_rooms_and_poisoned_duplicates(self):
        from sglang.srt.disaggregation.mooncake.conn import (
            MooncakeKVManager,
            MooncakeKVReceiver,
        )

        mgr = object.__new__(MooncakeKVManager)
        mgr.training_capture_lock = threading.Lock()
        mgr.training_capture_handoffs = {19: None}
        mgr.request_status = {19: 1}
        mgr.required_prefill_response_num_table = {19: 1}
        mgr.prefill_response_tracker = {19: set()}
        receiver = object.__new__(MooncakeKVReceiver)
        receiver.kv_mgr, receiver.bootstrap_room = mgr, 19
        for payload in (b"first", b"first", b"conflicting", b"first"):
            mgr._receive_training_capture_handoff(
                [mgr.TRAINING_CAPTURE_HEADER, b"19", payload]
            )
        self.assertEqual(receiver.take_training_capture_handoff(), b"")
        receiver.clear()
        mgr._receive_training_capture_handoff(
            [mgr.TRAINING_CAPTURE_HEADER, b"19", b"late"]
        )
        mgr._receive_training_capture_handoff(
            [mgr.TRAINING_CAPTURE_HEADER, b"999", b"orphan"]
        )
        self.assertFalse(mgr.training_capture_handoffs)
        self.assertFalse(mgr.request_status)

    def test_capture_metadata_extends_legacy_wire_without_changing_kv_indices(self):
        from sglang.srt.disaggregation.mooncake.conn import (
            MooncakeKVReceiver,
            TransferInfo,
        )

        receiver = object.__new__(MooncakeKVReceiver)
        receiver.bootstrap_room, receiver.session_id = 19, "session"
        receiver.required_dst_info_num = 1
        receiver.bootstrap_infos = [{"is_dummy": False}]
        receiver.kv_mgr = SimpleNamespace(
            local_ip="127.0.0.1",
            rank_port=1234,
            enable_staging=False,
            training_capture_lock=threading.Lock(),
            training_capture_handoffs={},
        )
        socket = Mock()
        receiver._connect_to_bootstrap_server = Mock(
            return_value=(socket, threading.Lock())
        )
        indices = np.array([7, 3, 1], dtype=np.int32)
        for context in (None, b"bounded-context"):
            with self.subTest(capture=context is not None):
                receiver.send_metadata(
                    indices, 2, decode_prefix_len=4, training_capture_context=context
                )
                message = socket.send_multipart.call_args.args[0]
                self.assertEqual(len(message), 10 if context is None else 11)
                info = TransferInfo.from_zmq(message)
                self.assertEqual(info.training_capture_context, context)
                self.assertEqual(info.decode_prefix_len, 4)
                np.testing.assert_array_equal(info.dst_kv_indices, indices)

    def test_pd_import_partitions_prompt_heads_and_writes_aux_only_once(self):
        req, wire = self.begin()
        payload, raw = self.handoff(wire)
        transfer = decode_handoff(wire, CaptureTransferContext)
        # Two physical heads replicated over four ranks leaves an inactive rank;
        # aux ownership is deliberately on a rank without canonical KV heads.
        layout = plan_capture_layout(
            self.decode.kv, tp_size=4, pp_layer_ranges=[(0, 4)], aux_tp_rank=1
        )
        parts, objects = [], {}
        with ExitStack() as stack:
            for partition in layout.partitions:
                with self.subTest(owner=partition.owner_id):
                    pool, slot = None, None
                    if partition.active:
                        pool = HostBufferPool(
                            kv=self.decode.kv,
                            max_tokens=8,
                            slots=1,
                            max_bytes=2 << 20,
                            registrar=Registrar(),
                            pin_memory=False,
                            partition=partition,
                        )
                        stack.callback(pool.close)
                        slot = pool.acquire()
                        stack.callback(pool.release, slot, transfer_complete=True)
                    context = RequestCaptureContext(
                        slot=slot,
                        prompt_ids=(3, 4),
                        max_tokens=8,
                        vocab_size=256,
                        partition=partition,
                    )
                    buffers = {
                        f"target_{c}.{heads.layer_id}": self.buffers[
                            f"target_{c}.{heads.layer_id}"
                        ][:, heads.start : heads.end]
                        for heads in partition.heads
                        for c in ("k", "v")
                    }
                    exporter = (
                        SelectedLayerKVExporter(
                            self.decode.kv, buffers, partition=partition
                        )
                        if buffers
                        else None
                    )
                    record = SimpleNamespace(context=context, invalid_reason=None)
                    local_req = SimpleNamespace(
                        req_pool_idx=0,
                        output_ids=[10],
                        training_capture_context=record,
                        training_capture_pd=transfer,
                    )
                    coordinator = SimpleNamespace(
                        teacher=self.decode.teacher,
                        req_to_token=self.decode.req_to_token,
                        exporter=exporter,
                        _count=Mock(),
                        _fail_request=Mock(
                            side_effect=AssertionError("PD import failed")
                        ),
                    )
                    DecodeCaptureMixin.accept_pd_handoff(
                        coordinator, local_req, payload
                    )
                    self.assertEqual(context.token_ids, [3, 4, 10])
                    self.assertEqual(context.kv_end, 2)
                    self.assertEqual(context.teacher_rows, int(partition.include_aux))
                    self.assertIsNone(local_req.training_capture_pd)
                    context.seal("length")
                    if not partition.active:
                        continue
                    metadata = SnapshotMetadata(
                        dataset_id=transfer.dataset_id,
                        sample_id=transfer.sample_id,
                        generation_id=transfer.generation_id,
                        teacher=self.decode.teacher,
                        sequence=context.sequence,
                        kv=self.decode.kv,
                        provenance=req.training_capture_context.provenance,
                        topology=layout.topology,
                    )
                    part, tensors = context.prepare_partition(
                        **{
                            name: getattr(metadata, name)
                            for name in SnapshotMetadata.__struct_fields__
                            if name != "sequence"
                        }
                    )
                    parts.append(part)
                    objects.update(tensors)
            manifest = assemble_snapshot(metadata, parts, layout=layout)
            validate_tensors(manifest, objects)
            aux = {
                obj.name: objects[obj.key]
                for obj in manifest.objects
                if obj.kind == "aux"
            }
            self.assertEqual(aux["token_ids"].tolist(), [3, 4, 10])
            torch.testing.assert_close(aux["teacher_topk_logits"], raw.topk(128).values)
            for obj in manifest.objects:
                if obj.kind == "kv":
                    start, end = obj.head_range
                    expected = self.buffers[obj.name][[7, 3], start:end]
                    torch.testing.assert_close(
                        objects[obj.key], expected, rtol=0, atol=0
                    )

    def test_recomputed_prefill_replaces_cached_pp_teacher(self):
        _, wire = self.begin()
        req = self.prefill_request(wire)
        batch = SimpleNamespace(reqs=[req], seq_lens_cpu=[2])
        forward = SimpleNamespace(extend_seq_lens_cpu=[2], positions=torch.arange(2))
        logits = torch.arange(256).float().reshape(1, -1)
        payloads = []
        for raw in (logits, -logits):
            self.prefill.after_forward(
                batch, forward, SimpleNamespace(next_token_logits=raw)
            )
            payload = self.prefill.pack_pp_handoffs(batch, torch.tensor([10]))[0]
            rows = decode_handoff(payload, PrefillTeacherHandoff).teacher_rows(256)
            torch.testing.assert_close(rows.logits, raw.topk(128).values)
            torch.testing.assert_close(rows.token_ids, raw.topk(128).indices.int())
            payloads.append(payload)
        self.assertNotEqual(*payloads)
        req.output_ids = [10]
        self.assertEqual(self.prefill.finish_handoff(req), payloads[-1])

    def test_pp_output_circulates_owned_teacher_before_final_kv_send(self):
        from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
        from sglang.srt.model_executor.forward_batch_info import PPProxyTensors

        _, wire = self.begin()
        source, receiver = self.prefill_request(wire), self.prefill_request(wire)
        other = CaptureTestRequest("unselected")
        batch = SimpleNamespace(
            reqs=[other, source],
            seq_lens_cpu=[1, 2],
            return_logprob=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        received_batch = SimpleNamespace(
            reqs=[other, receiver],
            seq_lens_cpu=[1, 2],
            return_logprob=False,
            req_pool_indices=torch.tensor([0, 1]),
            input_ids=torch.tensor([3, 4]),
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        forward = SimpleNamespace(
            extend_seq_lens_cpu=[1, 2], positions=torch.tensor([0, 0, 1])
        )
        self.prefill.after_forward(received_batch, forward, None)
        self.assertFalse(receiver.training_capture_pd.failed)
        logits = torch.randn(2, 256, generator=torch.Generator().manual_seed(901))
        raw = logits[1].clone()
        self.prefill.after_forward(
            batch, forward, SimpleNamespace(next_token_logits=logits)
        )
        scheduler = SimpleNamespace(
            tp_worker=SimpleNamespace(training_capture=self.prefill),
            future_map=SimpleNamespace(stash=Mock()),
        )
        with patch(
            "sglang.srt.managers.scheduler_pp_mixin.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode="prefill"),
        ):
            packet = SchedulerPPMixin._pp_prepare_tensor_dict(
                scheduler, SimpleNamespace(next_token_ids=torch.tensor([5, 10])), batch
            )
            payloads = packet["training_capture_pd_handoffs"]
            self.assertIsNone(payloads[0])
            self.assertLessEqual(len(payloads[1]), MAX_HANDOFF_BYTES)
            logits.fill_(-1000)
            result = SchedulerPPMixin._pp_prep_batch_result(
                scheduler,
                received_batch,
                SimpleNamespace(can_run_cuda_graph=False),
                PPProxyTensors(packet),
            )
        self.assertEqual(result.next_token_ids.tolist(), [5, 10])
        self.assertIsNone(source.training_capture_pd.teacher)
        self.assertIsNone(receiver.training_capture_pd.teacher)
        source.output_ids = receiver.output_ids = [10]
        self.assertEqual(self.prefill.finish_handoff(source), payloads[1])
        self.assertEqual(self.prefill.finish_handoff(receiver), payloads[1])
        row = decode_handoff(payloads[1], PrefillTeacherHandoff).teacher_rows(256)
        torch.testing.assert_close(row.logits[0], raw.topk(128).values, rtol=0, atol=0)

    def test_pp_missing_misaligned_or_foreign_teacher_cannot_finish_handoff(self):
        _, wire = self.begin()
        payload, _ = self.handoff(wire)
        handoff = decode_handoff(payload, PrefillTeacherHandoff)
        stale = encode_handoff(
            msgspec.structs.replace(
                handoff,
                context=msgspec.structs.replace(
                    handoff.context, fencing_token=handoff.context.fencing_token + 1
                ),
            )
        )
        for packet in (None, (), (None,), (payload, payload), (stale,), (b"bad",)):
            with self.subTest(packet=packet is None or len(packet)):
                req = self.prefill_request(wire)
                self.prefill.accept_pp_handoffs(
                    SimpleNamespace(reqs=[req], seq_lens_cpu=[2]), packet
                )
                self.assertTrue(req.training_capture_pd.failed)
                req.output_ids = [10]
                self.assertIsNone(self.prefill.finish_handoff(req))
                self.assertFalse(req.finished())
        req = self.prefill_request(wire)
        batch = SimpleNamespace(reqs=[req], seq_lens_cpu=[1])
        self.prefill.accept_pp_handoffs(batch, None)
        self.assertFalse(req.training_capture_pd.failed)
        batch.seq_lens_cpu = [2]
        self.prefill.accept_pp_handoffs(batch, (payload,))
        req.output_ids = [11]
        self.assertIsNone(self.prefill.finish_handoff(req))


if __name__ == "__main__":
    unittest.main()
