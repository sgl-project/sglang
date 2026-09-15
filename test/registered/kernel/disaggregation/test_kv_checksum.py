"""Tests for KV checksum integration in PD disaggregation."""

import unittest
import zlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.checksum import (
    KvChecksumComputer,
    is_health_check_req,
)
from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import MetadataBuffers
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
        buf.set_kv_checksum(SimpleNamespace(metadata_buffer_index=1), 0xDEADBEEF)
        self.assertEqual(buf.get_kv_checksum(1), 0xDEADBEEF)
        self.assertEqual(buf.get_kv_checksum(0), 0)


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

    def stream_output(self, reqs, return_logprob):
        self.streamed_aborts.extend(reqs)


class _FakePrefillScheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self):
        self.kv_checksum_computer = object()
        self.disagg_metadata_buffers = SimpleNamespace(set_kv_checksum=Mock())


def _make_req(expected_chksum, num_input_tokens, rid="r0"):
    return SimpleNamespace(
        rid=rid,
        bootstrap_room=12345,
        kv=SimpleNamespace(req_pool_idx=0),
        origin_input_ids=list(range(num_input_tokens)),
        fill_ids=list(range(num_input_tokens)),
        expected_kv_checksum=expected_chksum,
        return_logprob=False,
    )


class TestPrefillHealthCheckChecksum(unittest.TestCase):
    def test_health_check_clears_metadata_checksum(self):
        sched = _FakePrefillScheduler()
        req = _make_req(0xDEADBEEF, 1, rid="HEALTH_CHECK_1")
        with patch.object(
            SchedulerDisaggregationPrefillMixin,
            "_send_kv_chunk",
            lambda *args, **kwargs: None,
        ):
            sched.send_kv_chunk(req, last_chunk=True)
        sched.disagg_metadata_buffers.set_kv_checksum.assert_called_once_with(req, 0)


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

    def _run_once(self, sched, batch_ret=None):
        running_batch = SimpleNamespace()
        with patch.object(
            SchedulerDisaggregationDecodeMixin,
            "_get_new_prebuilt_batch",
            lambda s, rb: batch_ret,
        ):
            return sched.get_new_prebuilt_batch(running_batch)

    def test_match_keeps_req(self):
        sched = self._make_sched(_SENTINEL)
        sched.waiting_queue = [_make_req(self.true_chksum, self.num_pages)]
        self._run_once(sched)
        self.assertEqual(len(sched.waiting_queue), 1)
        self.assertEqual(sched.streamed_aborts, [])

    def test_mismatch_aborts(self):
        sched = self._make_sched(_SENTINEL)
        req = _make_req(0xDEADBEEF, self.num_pages)
        sched.waiting_queue = [req]
        with (
            envs.SGLANG_IS_IN_CI.override(False),
            patch("sglang.srt.disaggregation.decode.prepare_abort") as mock_abort,
            patch("sglang.srt.disaggregation.decode.release_kv_cache") as mock_release,
        ):
            self._run_once(sched)
            self._run_once(sched)
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
            self._run_once(sched)
        self.assertEqual(sched.waiting_queue, [req])
        self.assertEqual(sched.streamed_aborts, [])
        mock_abort.assert_not_called()
        mock_release.assert_not_called()

    def test_retract_re_verifies(self):
        sched = self._make_sched(_SENTINEL)
        req = _make_req(self.true_chksum, self.num_pages)
        sched.waiting_queue = [req]
        self._run_once(sched)
        self._run_once(sched)
        self.assertEqual(sched.waiting_queue, [req])
        self.assertEqual(sched.streamed_aborts, [])

    def test_disabled_delegates_to_batch_builder(self):
        sched = self._make_sched(computer=None)
        sched.waiting_queue = [_make_req(0xABCD, 4)]
        sentinel = object()
        self.assertIs(self._run_once(sched, batch_ret=sentinel), sentinel)

    def test_zero_expected_skips_checksum(self):
        sched = self._make_sched(_SENTINEL)
        sched.waiting_queue = [_make_req(0, self.num_pages)]
        self._run_once(sched)
        self.assertEqual(len(sched.waiting_queue), 1)


_SENTINEL = object()


if __name__ == "__main__":
    unittest.main()
