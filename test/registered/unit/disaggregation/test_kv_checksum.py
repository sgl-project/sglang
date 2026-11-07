"""Tests for KV checksum integration in PD disaggregation."""

import random
import unittest
import zlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.checksum import (
    KvChecksumComputer,
    is_health_check_req,
    page_indices_for_request,
    state_indices_for_request,
)
from sglang.srt.disaggregation.decode import (
    DecodeTransferQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import MetadataBuffers, kv_layout_signature
from sglang.srt.environ import envs
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.scheduler_components.request_list import (
    KeyedReqList,
    ListKind,
)
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpaceKind
from sglang.srt.mem_cache.kv.router import PlainPoolRouter
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import reset_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scheduler_test_utils import FakeReq, release_only_retirer
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


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
        # A prefill keeping a longer window tail than its decode, a pairing
        # the PD bootstrap accepts, digests more pages of that window.
        tails = make(state_window_tails=[128])
        self.assertEqual(tails, make(state_window_tails=[128]))
        self.assertNotEqual(tails, make(state_window_tails=[192]))


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


def _plain_kv_pool():
    """A one-sub-pool KV pool: no window sub-pool, no wrapped view."""
    pool = SimpleNamespace(start_layer=0, layer_num=1)
    pool.layer_router = PlainPoolRouter(pool, IdSpaceKind.FULL)
    return pool


class _FakeScheduler(SchedulerDisaggregationDecodeMixin):
    def __init__(self, computer, req_to_token):
        self.kv_checksum_computer = computer
        self.enable_overlap = False
        self.waiting_queue = []
        self.running_reqs = KeyedReqList(ListKind.DECODING)
        # A static pool: no unified storage, and ids the transfer uses as they are.
        self.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=1,
            get_kvcache=_plain_kv_pool,
            storage=None,
            translate_kv_indices_for_transfer=lambda kv_indices: kv_indices,
        )
        self.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
        self.tree_cache = None
        self.request_retirer = Mock()
        self.enable_hisparse = False
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
    return FakeReq(
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
        ):
            batch = self._run_once(sched)
            self.assertIsNone(self._run_once(sched))
        self.assertEqual(batch.reqs, [valid])
        batch.process_prebuilt.assert_called_once_with(sched.future_map)
        self.assertEqual(sched.waiting_queue, [])
        self.assertEqual(sched.streamed_aborts, [req])
        mock_abort.assert_called_once()
        sched.request_retirer.release.assert_called_once_with(req, checkpoint=False)
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
        with patch("sglang.srt.disaggregation.decode.prepare_abort") as mock_abort:
            batch = self._run_once(sched)
        self.assertEqual(batch.reqs, [req])
        self.assertEqual(sched.streamed_aborts, [])
        mock_abort.assert_not_called()
        sched.request_retirer.release.assert_not_called()

    def test_injected_mismatch_aborts_without_raising_in_ci(self):
        """The injector must not trip the in-CI raise it exists to exercise."""
        sched = self._make_sched(_SENTINEL)
        req = _make_req(self.true_chksum, self.num_pages)
        sched.waiting_queue = [req]
        with (
            envs.SGLANG_IS_IN_CI.override(True),
            envs.SGLANG_TEST_DISAGG_KV_CHECKSUM_MISMATCH_PROB.override(1.0),
            patch("sglang.srt.disaggregation.decode.prepare_abort") as mock_abort,
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
        ):
            batch = self._run_once(sched)
        self.assertEqual(batch.reqs, [skipped])
        self.assertEqual(sched.streamed_aborts, [aborted])
        mock_abort.assert_called_once()


class TestKvChecksumMismatchHiSparse(unittest.TestCase):
    def test_mismatch_releases_hisparse_before_kv(self):
        # Transfer already admitted the request into HiSparse; the abort must
        # free its host pages before the KV itself.
        sched = _FakeScheduler(None, None)
        calls = []
        hisparse_coordinator = Mock()
        hisparse_coordinator.release_request.side_effect = lambda req: calls.append(
            ("hisparse", req)
        )
        sched.request_retirer = release_only_retirer(Mock(), hisparse_coordinator)
        req = _make_req(0xDEADBEEF, 4)
        req.kv = SimpleNamespace(
            holds_kv=True, holds_mamba=False, retraction_backup=None
        )
        # What the patched `prepare_abort` would set.
        req.finished_reason = FINISH_ABORT("KV checksum mismatch")
        req.cache_request_handle = None
        req.session = None
        req.get_mm_token_counts = Mock()
        req.release_mm_inputs_on_abort = Mock()
        with (
            envs.SGLANG_IS_IN_CI.override(False),
            patch("sglang.srt.disaggregation.decode.prepare_abort"),
            patch(
                "sglang.srt.managers.scheduler_components.request_retirer.release_kv_cache",
                side_effect=lambda req, *a, **kw: calls.append(("kv", req)),
            ),
        ):
            sched._handle_kv_checksum_mismatch(req, "KV checksum mismatch")
        self.assertEqual(calls, [("hisparse", req), ("kv", req)])


def _row_of_pages(pages, page_size):
    """A request-table row holding the given pages' tokens in order."""
    return [page * page_size + offset for page in pages for offset in range(page_size)]


def _owner_from_top(page_size, num_pages=100):
    """Token ids as an owner band growing down from the top hands them out."""
    return lambda ids: (num_pages - ids // page_size) * page_size + ids % page_size


def _window_router():
    """A pool with a window sub-pool beside its full one."""
    return SimpleNamespace(
        wrapped_view=lambda: None,
        sub_pool=lambda kind: (
            SimpleNamespace(name="swa") if kind == IdSpaceKind.SLIDING_WINDOW else None
        ),
    )


def _wrapping_router():
    """A pool wrapping one token KV view beside its recurrent state."""
    return SimpleNamespace(wrapped_view=lambda: object(), sub_pool=lambda kind: None)


def _index_sched(
    *,
    page_size,
    row,
    owner=lambda ids: ids,
    window=lambda ids: ids,
    slots=lambda ids: ids,
    mamba_slot=0,
    window_size=8,
    router=None,
    entries=(StateType.SWA,),
):
    """Scheduler fields the index selection reads: each id space's
    virtual-to-physical translation as a function, and a storage whose entries
    are no sub-pools (a static pool's) registering ``entries`` in order."""
    storage = SimpleNamespace(
        entry_sub_pools=lambda: (),
        retained_window_lengths=lambda: {"swa": window_size},
        wire_entries=lambda: tuple(SimpleNamespace(state_type=t) for t in entries),
    )
    allocator = SimpleNamespace(
        page_size=page_size,
        storage=storage,
        translate_kv_indices_for_transfer=owner,
        state_ids={"mamba": SimpleNamespace(ids=slots)}.__getitem__,
        id_spaces=lambda: {"swa": SimpleNamespace(ids=window)},
        get_kvcache=lambda: SimpleNamespace(layer_router=router or _window_router()),
    )
    table = SimpleNamespace(
        req_to_token=torch.tensor([row], dtype=torch.int64),
        req_index_to_mamba_index_mapping=torch.tensor([[mamba_slot]]),
        state_sub_pool="mamba",
    )
    return SimpleNamespace(
        token_to_kv_pool_allocator=allocator, req_to_token_pool=table
    )


_ROW0 = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=0))


class TestChecksumIndexSelection(CustomTestCase):
    """The units each role digests are the ones the transfer addresses: a
    unified pool registers physical ids, the request table holds virtual ones."""

    page_size = 4

    def _sched(self, **kw):
        row = _row_of_pages([2, 5, 9], self.page_size)
        return _index_sched(page_size=self.page_size, row=row, **kw)

    def _units(self, sched, start_idx=0):
        rows = state_indices_for_request(sched, _ROW0, 12, start_idx=start_idx)
        if isinstance(rows, torch.Tensor):
            return rows.tolist()
        return [entry.tolist() for entry in rows]

    def test_owner_pages_are_translated(self):
        sched = self._sched(owner=_owner_from_top(self.page_size))
        pages = page_indices_for_request(sched, _ROW0, 12)
        self.assertEqual(pages.tolist(), [98, 95, 91])

    def test_static_pool_selects_the_table_pages(self):
        self.assertEqual(
            page_indices_for_request(self._sched(), _ROW0, 12).tolist(), [2, 5, 9]
        )

    def test_suffix_after_cached_prefix_selects_its_translated_pages(self):
        # One cached page, then a suffix ending inside its last page.
        sched = self._sched(owner=_owner_from_top(self.page_size))
        pages = page_indices_for_request(sched, _ROW0, 10, start_idx=4)
        self.assertEqual(pages.tolist(), [95, 91])

    def test_mamba_slot_is_translated(self):
        sched = self._sched(
            slots=lambda ids: 3 * ids + 1,
            mamba_slot=6,
            router=_wrapping_router(),
            entries=(StateType.MAMBA,),
        )
        self.assertEqual(self._units(sched), [19])

    def test_a_window_beside_state_reads_each_entry_at_its_own_units(self):
        # The window [6, 12) widens to its first page: table pages 5 and 9.
        window, slot = [12, 16], [19]
        for entries, expected in (
            ((StateType.SWA, StateType.MAMBA), [window, slot]),
            ((StateType.MAMBA, StateType.SWA), [slot, window]),
            ((StateType.BLOCK_SCALE_SWA, StateType.MAMBA), [window, slot]),
        ):
            with self.subTest(entries=entries):
                sched = self._sched(
                    window=lambda ids: ids + 7 * self.page_size,
                    slots=lambda ids: 3 * ids + 1,
                    mamba_slot=6,
                    window_size=6,
                    entries=entries,
                )
                self.assertEqual(self._units(sched), expected)

    def test_entries_without_their_own_rule_keep_the_window_pages(self):
        sched = self._sched(
            window=lambda ids: ids + 7 * self.page_size,
            slots=lambda ids: 3 * ids + 1,
            mamba_slot=6,
            window_size=6,
            entries=(StateType.SWA, StateType.BLOCK_SCALE, StateType.MAMBA),
        )
        rows = state_indices_for_request(sched, _ROW0, 12)
        self.assertEqual([entry.tolist() for entry in rows], [[12, 16], [12, 16], [19]])
        self.assertIs(rows[0], rows[1])

    def test_a_window_pool_without_state_shares_one_vector(self):
        sched = self._sched(
            window=lambda ids: ids + 7 * self.page_size,
            window_size=6,
            entries=(StateType.SWA, StateType.BLOCK_SCALE_SWA),
        )
        self.assertEqual(self._units(sched), [12, 16])

    def test_window_starts_after_cached_prefix(self):
        sched = self._sched(window_size=6)
        self.assertEqual(self._units(sched, start_idx=8), [9])


class TestStaticWindowAndMambaChecksumUnits(CustomTestCase):
    """BUG REGRESSION: on real static pools, a window pool beside a Mamba state
    pool digests the Mamba entry at the request's slot, not at window pages."""

    def setUp(self):
        setup_allocator_context()

    def tearDown(self):
        reset_context()

    def test_each_registered_entry_is_digested_at_its_own_units(self):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=256,
            n_groups=1,
            num_heads=2,
            head_dim=16,
            state_size=16,
            conv_kernel=4,
        )
        table = HybridReqToTokenPool(
            size=4,
            mamba_size=16,
            mamba_spec_state_size=4,
            max_context_len=64,
            device="cpu",
            enable_memory_saver=False,
            cache_params=Mamba2CacheParams(shape=shape, layers=[2]),
            mamba_layer_ids=[2],
            enable_mamba_extra_buffer=False,
            speculative_num_draft_tokens=3,
        )
        pool = SWAKVPool(
            size=64,
            size_swa=64,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            swa_attention_layer_ids=[1],
            full_attention_layer_ids=[0],
            device="cpu",
        )
        allocator = SWATokenToKVPoolAllocator(
            size=64,
            size_swa=64,
            page_size=1,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=pool,
            need_sort=False,
        )
        allocator.state = table.state
        storage = kv_storage_of(table, allocator)
        storage.bind_window_retention(2, None)
        tokens = allocator.alloc(4)
        table.req_to_token[0, :4] = tokens
        slot = allocator.state_slots.alloc(1)
        table.req_index_to_mamba_index_mapping[0] = slot
        sched = SimpleNamespace(
            req_to_token_pool=table, token_to_kv_pool_allocator=allocator
        )
        self.assertEqual(
            [entry.state_type for entry in storage.wire_entries()],
            [StateType.SWA, StateType.MAMBA],
        )

        rows = state_indices_for_request(sched, _ROW0, 4)

        self.assertIsInstance(rows, list, "one window vector serves both entries")
        window_space = allocator.id_spaces()[
            pool.layer_router.sub_pool(IdSpaceKind.SLIDING_WINDOW).name
        ]
        self.assertEqual(
            [entry.tolist() for entry in rows],
            [window_space.ids(tokens[-2:]).tolist(), slot.reshape(-1).tolist()],
        )
        self.assertNotEqual(rows[0].tolist(), rows[1].tolist())


class _UnitDigest:
    """Digests the values stored at the selected physical units, as the real
    computer digests the registered bytes there."""

    def __init__(self, units, entries):
        self.units = units
        self.entries = entries

    def _value(self, part, unit):
        return self.units[part].get(unit, ("unrelated", self.units["side"], unit))

    def compute(self, kv_pages, state_indices):
        if not isinstance(state_indices, list):
            state_indices = [state_indices] * len(self.entries)
        values = [self._value("kv", unit) for unit in kv_pages.tolist()]
        for entry, indices in zip(self.entries, state_indices, strict=True):
            values += [self._value(entry, unit) for unit in indices.tolist()]
        return zlib.adler32(repr(values).encode())


def _side(
    side, entries, router, *, owner_base, window_base, slot_base, table_base, mamba_slot
):
    """One role's tables and stored units for a 4-token request whose last two
    tokens are its window. Each role gives the request different virtual ids
    and maps them to different physical units; the transfer copied the values."""
    row = [table_base + token for token in range(4)]
    sched = _index_sched(
        page_size=1,
        row=row,
        owner=lambda ids: owner_base - ids,
        window=lambda ids: ids + window_base,
        slots=lambda ids: ids + slot_base,
        mamba_slot=mamba_slot,
        window_size=2,
        router=router,
        entries=entries,
    )
    units = {
        "side": side,
        "kv": {
            owner_base - virtual: ("kv", token) for token, virtual in enumerate(row)
        },
        StateType.SWA: {
            virtual + window_base: ("swa", token)
            for token, virtual in enumerate(row)
            if token >= 2
        },
        StateType.MAMBA: {mamba_slot + slot_base: ("mamba", "final state")},
    }
    return sched, units


class TestChecksumAgreesAcrossRoles(CustomTestCase):
    """Prefill's send and decode's verify digest the same request data when
    the two roles hold it at different virtual and physical ids."""

    LAYOUTS = {
        "window beside state": ((StateType.SWA, StateType.MAMBA), _window_router),
        "token KV beside state": ((StateType.MAMBA,), _wrapping_router),
    }

    def _prefill_digest(self, layout, start_idx):
        entries, router = self.LAYOUTS[layout]
        sched, units = _side(
            "prefill",
            entries,
            router(),
            owner_base=50,
            window_base=20,
            slot_base=100,
            table_base=0,
            mamba_slot=3,
        )
        prefill = _FakePrefillScheduler(_UnitDigest(units, entries))
        prefill.token_to_kv_pool_allocator = sched.token_to_kv_pool_allocator
        prefill.req_to_token_pool = sched.req_to_token_pool
        req = _make_req(0, 4, start_idx=start_idx)
        with patch.object(
            SchedulerDisaggregationPrefillMixin,
            "_send_kv_chunk",
            lambda *args, **kwargs: None,
        ):
            prefill.send_kv_chunk(req, last_chunk=True)
        (_, digest, _), _ = prefill.disagg_metadata_buffers.set_kv_checksum.call_args
        return digest

    def _decode_verdict(self, layout, expected, start_idx, corrupt=None):
        entries, router = self.LAYOUTS[layout]
        sched, units = _side(
            "decode",
            entries,
            router(),
            owner_base=90,
            window_base=40,
            slot_base=200,
            table_base=10,
            mamba_slot=5,
        )
        if corrupt == "kv":
            # The last token's KV, at decode's physical unit for virtual id 13.
            units["kv"][90 - 13] = ("kv", "flipped")
        elif corrupt == "state":
            units[StateType.MAMBA][5 + 200] = ("mamba", "flipped")
        decode = _FakeScheduler(_UnitDigest(units, entries), None)
        decode.token_to_kv_pool_allocator = sched.token_to_kv_pool_allocator
        decode.req_to_token_pool = sched.req_to_token_pool
        req = _make_req(expected, 4, start_idx=start_idx)
        batch = SimpleNamespace(reqs=[req], filter_batch=Mock())
        with (
            envs.SGLANG_IS_IN_CI.override(False),
            patch(
                "sglang.srt.disaggregation.decode.all_reduce_flags",
                lambda flags, group, op: flags,
            ),
            patch("sglang.srt.disaggregation.decode.prepare_abort"),
        ):
            decode._verify_kv_checksums(batch)
        return decode.streamed_aborts

    def test_intact_transfer_verifies(self):
        for layout in self.LAYOUTS:
            for start_idx in (0, 2):
                with self.subTest(layout=layout, start_idx=start_idx):
                    digest = self._prefill_digest(layout, start_idx)
                    verdict = self._decode_verdict(layout, digest, start_idx)
                    self.assertEqual(verdict, [])

    def test_changed_unit_is_dropped(self):
        for layout in self.LAYOUTS:
            for corrupt in ("kv", "state"):
                with self.subTest(layout=layout, corrupt=corrupt):
                    digest = self._prefill_digest(layout, 0)
                    verdict = self._decode_verdict(layout, digest, 0, corrupt=corrupt)
                    self.assertEqual(len(verdict), 1)


_SENTINEL = object()


if __name__ == "__main__":
    unittest.main()
