"""Tests for KV checksum integration in PD disaggregation."""

import random
import unittest
import zlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.checksum import (
    KvChecksumComputer,
    is_health_check_req,
)
from sglang.srt.disaggregation.decode import (
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import MetadataBuffers, kv_layout_signature
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _ref_strided_adler32(tensors, indices, strides) -> int:
    parts = []
    for tensor, idx, stride in zip(tensors, indices, strides):
        raw = tensor.cpu().contiguous().flatten().view(torch.uint8)
        for i in idx.cpu().tolist():
            parts.append(raw[i * stride : (i + 1) * stride].numpy().tobytes())
    return zlib.adler32(b"".join(parts))


def _make_buf(size=4, *, kv_checksum_enabled=True, output_dsa_topk_indices_dim=0):
    return MetadataBuffers(
        size=size,
        hidden_size=16,
        hidden_states_dtype=torch.float32,
        max_sampling_mask_tokens=128,
        output_dsa_topk_indices_dim=output_dsa_topk_indices_dim,
        kv_checksum_enabled=kv_checksum_enabled,
    )


class TestMetadataBuffers(unittest.TestCase):
    def test_extends_aux_with_kv_checksum(self):
        for sampling_mask in (False, True):
            for seed_dim in (0, 32):
                with (
                    self.subTest(sampling_mask=sampling_mask, seed_dim=seed_dim),
                    envs.SGLANG_ENABLE_DISAGG_SAMPLING_MASK.override(sampling_mask),
                ):
                    buf = _make_buf(size=8, output_dsa_topk_indices_dim=seed_dim)
                    disabled = _make_buf(
                        size=8,
                        kv_checksum_enabled=False,
                        output_dsa_topk_indices_dim=seed_dim,
                    )
                    ptrs, data_lens, item_lens = buf.get_buf_infos()
                    base_ptrs, base_data_lens, base_item_lens = disabled.get_buf_infos()
                    self.assertEqual(len(ptrs), len(base_ptrs) + 1)
                    self.assertEqual(data_lens[:-1], base_data_lens)
                    self.assertEqual(item_lens[:-1], base_item_lens)
                    self.assertEqual(ptrs[-1], buf.kv_checksum.data_ptr())
                    self.assertEqual(data_lens[-1], buf.kv_checksum.nbytes)
                    self.assertEqual(item_lens[-1], buf.kv_checksum[0].nbytes)
                    self.assertEqual(item_lens[-1], item_lens[-2])
                    self.assertEqual(len(buf.get_buf(2)), len(disabled.get_buf(2)))

    def test_set_get_kv_checksum_roundtrip(self):
        buf = _make_buf()
        buf.set_kv_checksum(SimpleNamespace(metadata_buffer_index=1), 0xDEADBEEF, 0xAB)
        self.assertEqual(buf.get_kv_checksum(1), (0xDEADBEEF, 0xAB))
        # An untouched row reads as "no digest".
        self.assertEqual(buf.get_kv_checksum(0), (0, 0))

    def test_clear_slot_resets_what_the_next_owner_reads(self):
        buf = _make_buf()
        buf.set_kv_checksum(SimpleNamespace(metadata_buffer_index=1), 0xDEADBEEF, 0xAB)
        buf.bootstrap_room[1, 0] = 42
        buf.clear_slot(1)
        self.assertEqual(buf.get_kv_checksum(1), (0, 0))
        self.assertEqual(int(buf.bootstrap_room[1, 0]), 0)
        _make_buf(kv_checksum_enabled=False).clear_slot(1)


class TestKvLayoutSignature(unittest.TestCase):
    def test_tracks_layout_and_is_never_zero(self):
        def make(**kw):
            base = dict(
                page_size=1,
                kv_cache_dtype_str="bf16",
                kv_layout=None,
                swa_kv_cache_layout=None,
                state_types=[StateType.SWA],
                kv_item_lens=[64, 64],
                state_item_lens=[[32]],
            )
            base.update(kw)
            return kv_layout_signature(SimpleNamespace(**base))

        same = make()
        self.assertNotEqual(same, 0)
        self.assertEqual(same, make())
        # A different TP width shows up as different per-page item lengths.
        self.assertNotEqual(same, make(kv_item_lens=[32, 32]))
        # A layer-sharded or PP prefill owns fewer buffers.
        self.assertNotEqual(same, make(kv_item_lens=[64]))
        self.assertNotEqual(same, make(page_size=16))
        self.assertNotEqual(same, make(state_item_lens=[[16]]))
        # Same byte sizes, other bytes: another dtype of the same width, or
        # another page layout of the same pool.
        self.assertNotEqual(same, make(kv_cache_dtype_str="fp16"))
        self.assertNotEqual(same, make(kv_layout="bucketed"))
        self.assertNotEqual(same, make(state_types=[StateType.DSA]))


class TestKvChecksumComputerConfig(unittest.TestCase):
    def test_flattens_nested_state_descriptor_components(self):
        computer = KvChecksumComputer(
            torch.device("cpu"),
            kv_data_ptrs=[11, 22],
            kv_item_lens=[33, 44],
            state_data_ptrs=[[55, 66], [77]],
            state_item_lens=[[88, 99], [111]],
        )

        self.assertEqual(computer._state_data_ptrs, [55, 66, 77])
        self.assertEqual(computer._state_item_lens, [88, 99, 111])


class TestKvChecksumHealthCheck(unittest.TestCase):
    def test_detects_health_check_request(self):
        self.assertTrue(is_health_check_req(SimpleNamespace(rid="HEALTH_CHECK_1")))
        self.assertFalse(is_health_check_req(SimpleNamespace(rid="user_req")))
        self.assertFalse(is_health_check_req(SimpleNamespace(rid=None)))


def _make_kv(num_layers, num_pages, page_elems, dtype=torch.float16):
    return [
        torch.randn(num_pages, page_elems, dtype=dtype, device="cuda:0")
        for _ in range(2 * num_layers)
    ]


def _make_computer(kv, item_len, state=None, state_item_lens=None):
    return KvChecksumComputer(
        torch.device("cuda:0"),
        kv_data_ptrs=[t.data_ptr() for t in kv],
        kv_item_lens=[item_len] * len(kv),
        state_data_ptrs=[t.data_ptr() for t in (state or [])],
        state_item_lens=state_item_lens or [],
    )


class TestKvChecksumComputer(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda:0")

    def test_kv_only_matches_reference(self):
        kv = _make_kv(num_layers=4, num_pages=32, page_elems=128, dtype=torch.bfloat16)
        idx = torch.tensor([3, 7, 8, 15, 31], dtype=torch.int64, device=self.device)
        item_len = 128 * 2
        value = _make_computer(kv, item_len).compute(idx)
        expected = _ref_strided_adler32(kv, [idx] * len(kv), [item_len] * len(kv))
        self.assertEqual(value, expected)

    def test_kv_corruption_detected(self):
        kv = _make_kv(num_layers=2, num_pages=16, page_elems=64)
        idx = torch.tensor([5], dtype=torch.int64, device=self.device)
        computer = _make_computer(kv, 64 * 2)
        v1 = computer.compute(idx)
        kv[0][5, 0] += 1
        self.assertNotEqual(v1, computer.compute(idx))

    def test_state_only_matches_reference(self):
        state = _make_kv(num_layers=1, num_pages=4, page_elems=16)
        idx = torch.tensor([1, 3], dtype=torch.int64, device=self.device)
        value = _make_computer([], 0, state, [32, 32]).compute(idx, idx)
        self.assertEqual(value, _ref_strided_adler32(state, [idx, idx], [32, 32]))

    def test_kv_plus_state_matches_reference(self):
        kv = _make_kv(num_layers=2, num_pages=8, page_elems=32)
        state = [
            torch.randn(4, 16, dtype=torch.float16, device=self.device)
            for _ in range(2)
        ]
        kv_idx = torch.tensor([0, 1, 2], dtype=torch.int64, device=self.device)
        state_idx = torch.tensor([1], dtype=torch.int64, device=self.device)
        kv_len, state_lens = 32 * 2, [16 * 2, 16 * 2]
        computer = _make_computer(kv, kv_len, state, state_lens)
        value = computer.compute(kv_idx, state_idx)
        expected = _ref_strided_adler32(
            kv + state,
            [kv_idx] * len(kv) + [state_idx] * len(state),
            [kv_len] * len(kv) + state_lens,
        )
        self.assertEqual(value, expected)
        state[0][1, 0] += 1
        self.assertNotEqual(value, computer.compute(kv_idx, state_idx))

    def test_nested_state_components_match_reference(self):
        kv = _make_kv(num_layers=1, num_pages=8, page_elems=32)
        state_components = [
            [torch.randn(4, 16, dtype=torch.float16, device=self.device)],
            [
                torch.randn(4, 8, dtype=torch.float16, device=self.device),
                torch.randn(4, 12, dtype=torch.float16, device=self.device),
            ],
        ]
        kv_idx = torch.tensor([0, 3, 7], dtype=torch.int64, device=self.device)
        state_idx = torch.tensor([1, 2], dtype=torch.int64, device=self.device)
        state_tensors = [tensor for comp in state_components for tensor in comp]
        kv_len = 32 * 2
        state_lens = [[16 * 2], [8 * 2, 12 * 2]]
        computer = KvChecksumComputer(
            self.device,
            kv_data_ptrs=[t.data_ptr() for t in kv],
            kv_item_lens=[kv_len] * len(kv),
            state_data_ptrs=[
                [tensor.data_ptr() for tensor in comp] for comp in state_components
            ],
            state_item_lens=state_lens,
        )

        value = computer.compute(kv_idx, state_idx)
        expected = _ref_strided_adler32(
            kv + state_tensors,
            [kv_idx] * len(kv) + [state_idx] * len(state_tensors),
            [kv_len] * len(kv) + [item for comp in state_lens for item in comp],
        )
        self.assertEqual(value, expected)


class _FakeScheduler(SchedulerDisaggregationDecodeMixin):
    def __init__(self, computer, req_to_token):
        self.kv_checksum_computer = computer
        self.waiting_queue = []
        self.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=1, get_kvcache=lambda: SimpleNamespace()
        )
        self.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        self.tree_cache = None
        self.output_streamer = SimpleNamespace(stream_output=self.stream_output)
        self.metrics_reporter = SimpleNamespace(enable_metrics=True)
        self.metrics_collector = Mock()
        self.streamed_aborts = []
        self.ngram_embedding_manager = Mock()
        self.chunked_req = None
        self.future_map = None
        self.attn_tp_cpu_group = None
        self.disagg_decode_prealloc_queue = SimpleNamespace(
            kv_manager=SimpleNamespace(
                should_verify_kv_checksum=lambda bootstrap_addr: True
            ),
            kv_checksum_mismatch_injector=random.Random(0),
        )

    def stream_output(self, reqs, return_logprob):
        self.streamed_aborts.extend(reqs)


class _FakePrefillScheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self, computer=None):
        self.kv_checksum_computer = computer or object()
        self.disagg_metadata_buffers = SimpleNamespace(set_kv_checksum=Mock())
        self.disagg_prefill_bootstrap_queue = SimpleNamespace(
            kv_checksum_signature=0x1234
        )


def _make_req(
    expected_chksum,
    num_input_tokens,
    rid="r0",
    bootstrap_host="127.0.0.1",
    start_idx=0,
):
    return SimpleNamespace(
        rid=rid,
        bootstrap_host=bootstrap_host,
        bootstrap_port=8998,
        bootstrap_room=12345,
        # start_idx: the decode-cached prefix, as each role records it.
        kv=SimpleNamespace(req_pool_idx=0, cache_protected_len=start_idx),
        origin_input_ids=list(range(num_input_tokens)),
        fill_ids=list(range(num_input_tokens)),
        extend_end=num_input_tokens,
        expected_kv_checksum=expected_chksum,
        disagg_decode_prefix_len=start_idx,
        return_logprob=False,
    )


class TestPrefillChecksum(unittest.TestCase):
    def _send_last_chunk(self, sched, req):
        page_indices = Mock(return_value=None)
        with (
            patch.object(
                SchedulerDisaggregationPrefillMixin,
                "_send_kv_chunk",
                lambda *args, **kwargs: None,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.page_indices_for_request",
                page_indices,
            ),
            patch(
                "sglang.srt.disaggregation.prefill.state_indices_for_request",
                lambda *args, **kwargs: None,
            ),
        ):
            sched.send_kv_chunk(req, last_chunk=True)
        return page_indices

    def test_health_check_clears_metadata_checksum(self):
        sched = _FakePrefillScheduler()
        req = _make_req(0xDEADBEEF, 1, rid="HEALTH_CHECK_1")
        self._send_last_chunk(sched, req)
        sched.disagg_metadata_buffers.set_kv_checksum.assert_called_once_with(req, 0, 0)

    def test_last_chunk_ships_digest_with_layout_signature(self):
        sched = _FakePrefillScheduler(SimpleNamespace(compute=lambda *args: 0x55))
        req = _make_req(0, 7)
        self._send_last_chunk(sched, req)
        sched.disagg_metadata_buffers.set_kv_checksum.assert_called_once_with(
            req, 0x55, 0x1234
        )

    def test_digest_starts_where_the_decode_cached_prefix_ends(self):
        """A decode-side radix hit makes prefill send only the suffix; the
        digest must cover that suffix, not the whole prompt."""
        sched = _FakePrefillScheduler(SimpleNamespace(compute=lambda *args: 0x55))
        req = _make_req(0, 7, start_idx=4)
        page_indices = self._send_last_chunk(sched, req)
        page_indices.assert_called_once_with(sched, req, 7, start_idx=4)


class TestKvChecksumComparable(unittest.TestCase):
    def _queue(self, signature=0x77):
        return SimpleNamespace(
            scheduler=SimpleNamespace(
                disagg_decode_prealloc_queue=SimpleNamespace(
                    kv_checksum_signature=signature
                )
            ),
            _skipped_checksum_signatures=set(),
        )

    def test_same_layout_compares(self):
        self.assertTrue(
            DecodeTransferQueue._kv_checksum_comparable(self._queue(), 0x77)
        )

    def test_no_digest_skips_quietly(self):
        queue = self._queue()
        self.assertFalse(DecodeTransferQueue._kv_checksum_comparable(queue, 0))
        self.assertEqual(queue._skipped_checksum_signatures, set())

    def test_other_layout_skips_and_logs_once(self):
        queue = self._queue()
        with self.assertLogs(
            "sglang.srt.disaggregation.decode", level="WARNING"
        ) as logs:
            self.assertFalse(DecodeTransferQueue._kv_checksum_comparable(queue, 0x99))
        self.assertEqual(len(logs.output), 1)
        with patch("sglang.srt.disaggregation.decode.logger.warning") as warning:
            self.assertFalse(DecodeTransferQueue._kv_checksum_comparable(queue, 0x99))
        warning.assert_not_called()


class TestGetNewPrebuiltBatchChecksum(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        self.device = torch.device("cuda:0")
        self.num_pages = 8
        self.kv = _make_kv(num_layers=2, num_pages=self.num_pages, page_elems=32)
        self.item_len = 32 * 2
        self.req_to_token = torch.arange(
            self.num_pages, dtype=torch.int64, device=self.device
        ).view(1, self.num_pages)
        self.true_chksum = _ref_strided_adler32(
            self.kv,
            [torch.arange(self.num_pages, dtype=torch.int64, device=self.device)]
            * len(self.kv),
            [self.item_len] * len(self.kv),
        )

    def _make_sched(self, computer=None):
        if computer is _SENTINEL:
            computer = _make_computer(self.kv, self.item_len)
        return _FakeScheduler(computer, self.req_to_token)

    def _run_once(self, sched, batch_ret=None, reduce=None):
        batch = Mock(reqs=sched.waiting_queue) if batch_ret is None else batch_ret
        batch.is_empty.side_effect = lambda: not batch.reqs

        def filter_batch(*, keep_indices):
            batch.reqs = [batch.reqs[i] for i in keep_indices]

        batch.filter_batch.side_effect = filter_batch
        sched.waiting_queue = []
        running_batch = SimpleNamespace()
        with (
            patch.object(
                SchedulerDisaggregationDecodeMixin,
                "_get_new_prebuilt_batch",
                lambda s, rb: batch,
            ),
            # Single rank unless a test stands in for the peers.
            patch(
                "sglang.srt.disaggregation.decode.all_reduce_flags",
                reduce or (lambda flags, group, op: flags),
            ),
        ):
            return sched.get_new_prebuilt_batch(running_batch)

    def test_match_keeps_req(self):
        sched = self._make_sched(_SENTINEL)
        sched.waiting_queue = [_make_req(self.true_chksum, self.num_pages)]
        batch = self._run_once(sched)
        self.assertEqual(len(batch.reqs), 1)
        self.assertEqual(sched.streamed_aborts, [])

    def test_mismatch_aborts(self):
        sched = self._make_sched(_SENTINEL)
        req = _make_req(0xDEADBEEF, self.num_pages)
        valid = _make_req(self.true_chksum, self.num_pages, rid="valid")
        sched.waiting_queue = [req, valid]
        with (
            envs.SGLANG_IS_IN_CI.override(False),
            patch("sglang.srt.disaggregation.decode.prepare_abort") as mock_abort,
            patch("sglang.srt.disaggregation.decode.release_kv_cache") as mock_release,
        ):
            batch = self._run_once(sched)
            self.assertIsNone(self._run_once(sched))
        self.assertEqual(batch.reqs, [valid])
        batch.process_prebuilt.assert_called_once_with(sched.future_map)
        self.assertEqual(sched.waiting_queue, [])
        self.assertEqual(sched.streamed_aborts, [req])
        mock_abort.assert_called_once()
        mock_release.assert_called_once()
        sched.metrics_collector.increment_transfer_failed_reqs.assert_called_once_with()

    def test_mismatch_raises_in_ci(self):
        sched = self._make_sched(_SENTINEL)
        sched.waiting_queue = [_make_req(0xDEADBEEF, self.num_pages)]
        with (
            envs.SGLANG_IS_IN_CI.override(True),
            patch("sglang.srt.disaggregation.decode.prepare_abort") as mock_abort,
            self.assertRaisesRegex(RuntimeError, "KV checksum mismatch"),
        ):
            self._run_once(sched)
        mock_abort.assert_not_called()
        sched.metrics_collector.increment_transfer_failed_reqs.assert_not_called()

    def test_health_check_skips_checksum(self):
        sched = self._make_sched(_SENTINEL)
        req = _make_req(0xDEADBEEF, self.num_pages, rid="HEALTH_CHECK_1")
        sched.waiting_queue = [req]
        with (
            patch("sglang.srt.disaggregation.decode.prepare_abort") as mock_abort,
            patch("sglang.srt.disaggregation.decode.release_kv_cache") as mock_release,
        ):
            batch = self._run_once(sched)
        self.assertEqual(batch.reqs, [req])
        self.assertEqual(sched.streamed_aborts, [])
        mock_abort.assert_not_called()
        mock_release.assert_not_called()

    def test_injected_mismatch_aborts_without_raising_in_ci(self):
        """The injector must not trip the in-CI raise it exists to exercise."""
        sched = self._make_sched(_SENTINEL)
        req = _make_req(self.true_chksum, self.num_pages)
        sched.waiting_queue = [req]
        with (
            envs.SGLANG_IS_IN_CI.override(True),
            envs.SGLANG_TEST_DISAGG_KV_CHECKSUM_MISMATCH_PROB.override(1.0),
            patch("sglang.srt.disaggregation.decode.prepare_abort") as mock_abort,
            patch("sglang.srt.disaggregation.decode.release_kv_cache"),
        ):
            self.assertIsNone(self._run_once(sched))
        self.assertEqual(sched.streamed_aborts, [req])
        mock_abort.assert_called_once()

    def test_mismatch_decision_is_all_reduced(self):
        """Every rank drops the same requests, even if only a peer saw the fault."""
        sched = self._make_sched(_SENTINEL)
        clean = _make_req(self.true_chksum, self.num_pages, rid="clean")
        sched.waiting_queue = [clean]
        with (
            envs.SGLANG_IS_IN_CI.override(False),
            patch("sglang.srt.disaggregation.decode.prepare_abort"),
            patch("sglang.srt.disaggregation.decode.release_kv_cache"),
        ):
            batch = self._run_once(
                sched, reduce=lambda flags, group, op: [1] * len(flags)
            )
        self.assertIsNone(batch)
        self.assertEqual(sched.streamed_aborts, [clean])

    def test_verified_once(self):
        """A request is digested when it first enters a batch, not again after
        a retraction, by which time decode has written into its last page."""
        sched = self._make_sched(_SENTINEL)
        req = _make_req(self.true_chksum, self.num_pages)
        sched.waiting_queue = [req]
        batch = self._run_once(sched)
        self.assertEqual(req.expected_kv_checksum, 0)
        self.kv[0][0, 0] += 1
        sched.waiting_queue = batch.reqs
        self.assertEqual(self._run_once(sched).reqs, [req])
        self.assertEqual(sched.streamed_aborts, [])

    def test_suffix_digest_skips_decode_cached_prefix(self):
        """After a decode-side radix hit the digest covers only the pages
        prefill sent; the prefix decode cached itself is not corruption."""
        start = self.num_pages // 2
        suffix = torch.arange(
            start, self.num_pages, dtype=torch.int64, device=self.device
        )
        expected = _ref_strided_adler32(
            self.kv, [suffix] * len(self.kv), [self.item_len] * len(self.kv)
        )
        sched = self._make_sched(_SENTINEL)
        req = _make_req(expected, self.num_pages, start_idx=start)
        sched.waiting_queue = [req]
        self.kv[0][0, 0] += 1
        self.assertEqual(self._run_once(sched).reqs, [req])
        self.assertEqual(sched.streamed_aborts, [])

    def test_disabled_delegates_to_batch_builder(self):
        sched = self._make_sched(computer=None)
        sched.waiting_queue = [_make_req(0xABCD, 4)]
        sentinel = Mock(reqs=sched.waiting_queue)
        self.assertIs(self._run_once(sched, batch_ret=sentinel), sentinel)

    def test_zero_expected_skips_checksum(self):
        sched = self._make_sched(_SENTINEL)
        sched.waiting_queue = [_make_req(0, self.num_pages)]
        self.assertEqual(len(self._run_once(sched).reqs), 1)

    def test_backend_can_skip_incomparable_checksum(self):
        """A request whose sender the backend declares incomparable is admitted
        without a digest, decided per bootstrap address, and skipping it must not
        disturb the index-based filtering of a mismatching sibling."""
        sched = self._make_sched(_SENTINEL)
        sched.disagg_decode_prealloc_queue.kv_manager.should_verify_kv_checksum = (
            lambda bootstrap_addr: not bootstrap_addr.startswith("10.0.0.1:")
        )
        aborted = _make_req(0xDEADBEEF, self.num_pages, rid="aborted")
        skipped = _make_req(
            0xDEADBEEF, self.num_pages, rid="skipped", bootstrap_host="10.0.0.1"
        )
        sched.waiting_queue = [aborted, skipped]
        with (
            envs.SGLANG_IS_IN_CI.override(False),
            patch("sglang.srt.disaggregation.decode.prepare_abort") as mock_abort,
            patch("sglang.srt.disaggregation.decode.release_kv_cache"),
        ):
            batch = self._run_once(sched)
        self.assertEqual(batch.reqs, [skipped])
        self.assertEqual(sched.streamed_aborts, [aborted])
        mock_abort.assert_called_once()


_SENTINEL = object()


if __name__ == "__main__":
    unittest.main()
