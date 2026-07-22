import unittest
from unittest.mock import Mock

import torch
from torch import nn

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferIndicesUpdaterDecode,
    FlashInferIndicesUpdaterPrefill,
    FlashInferWrapperMetadata,
    _resolve_wrapper_metadata,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_attention(num_heads: int, layer_id: int, window: int) -> RadixAttention:
    return RadixAttention(
        num_heads=num_heads,
        head_dim=64,
        scaling=64**-0.5,
        num_kv_heads=8,
        layer_id=layer_id,
        sliding_window_size=window,
    )


def _get_swa_wrapper_idx(layer: RadixAttention) -> int:
    return 1 if layer.sliding_window_size == -1 else 0


class TestFlashInferWrapperGeometry(CustomTestCase):
    def test_resolves_tp_local_heads_per_wrapper(self):
        model = nn.ModuleList(
            [
                _make_attention(72, 0, 511),
                _make_attention(72, 1, 511),
                _make_attention(48, 2, -1),
            ]
        )

        metadata = _resolve_wrapper_metadata(
            model,
            num_wrappers=2,
            get_wrapper_idx=_get_swa_wrapper_idx,
            default_metadata=[
                FlashInferWrapperMetadata(48, 511),
                FlashInferWrapperMetadata(48, -1),
            ],
        )

        self.assertEqual(
            metadata,
            [
                FlashInferWrapperMetadata(72, 511),
                FlashInferWrapperMetadata(48, -1),
            ],
        )

    def test_falls_back_for_missing_layers(self):
        metadata = _resolve_wrapper_metadata(
            None,
            num_wrappers=2,
            get_wrapper_idx=_get_swa_wrapper_idx,
            default_metadata=[
                FlashInferWrapperMetadata(48, 511),
                FlashInferWrapperMetadata(48, -1),
            ],
        )

        self.assertEqual(
            metadata,
            [
                FlashInferWrapperMetadata(48, 511),
                FlashInferWrapperMetadata(48, -1),
            ],
        )

    def test_rejects_mixed_heads_within_one_wrapper(self):
        model = nn.ModuleList(
            [
                _make_attention(64, 0, 511),
                _make_attention(72, 1, 511),
            ]
        )

        with self.assertRaisesRegex(
            ValueError, r"wrapper 0 has \[\(64, 511\), \(72, 511\)\]"
        ):
            _resolve_wrapper_metadata(
                model,
                num_wrappers=2,
                get_wrapper_idx=_get_swa_wrapper_idx,
                default_metadata=[
                    FlashInferWrapperMetadata(48, 511),
                    FlashInferWrapperMetadata(48, -1),
                ],
            )

    def test_rejects_mixed_windows_within_one_wrapper(self):
        model = nn.ModuleList(
            [
                _make_attention(72, 0, 255),
                _make_attention(72, 1, 511),
            ]
        )

        with self.assertRaisesRegex(ValueError, "sliding-window geometry"):
            _resolve_wrapper_metadata(
                model,
                num_wrappers=2,
                get_wrapper_idx=_get_swa_wrapper_idx,
                default_metadata=[
                    FlashInferWrapperMetadata(48, 511),
                    FlashInferWrapperMetadata(48, -1),
                ],
            )

    def test_decode_selects_wrapper_heads(self):
        updater = FlashInferIndicesUpdaterDecode.__new__(FlashInferIndicesUpdaterDecode)
        updater.sliding_window_size = 511
        updater.num_qo_heads_per_wrapper = [72, 48]
        updater.window_left_per_wrapper = [511, -1]
        updater.kv_indptr = [torch.zeros(2, dtype=torch.int32) for _ in range(2)]
        updater._swa_kv_pool = None
        updater.call_begin_forward = Mock()

        updater.update_sliding_window(
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([1024], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([1024], dtype=torch.int32),
            seq_lens_sum=1024,
            decode_wrappers=[object(), object()],
            encoder_lens=None,
            spec_info=None,
        )

        self.assertEqual(
            [
                call.kwargs["num_qo_heads"]
                for call in updater.call_begin_forward.call_args_list
            ],
            [72, 48],
        )

    def test_prefill_selects_wrapper_heads_and_plan_windows(self):
        updater = FlashInferIndicesUpdaterPrefill.__new__(
            FlashInferIndicesUpdaterPrefill
        )
        updater.sliding_window_size = 511
        updater.num_qo_heads_per_wrapper = [72, 48]
        updater.window_left_per_wrapper = [511, -1]
        updater.prefill_wrappers_ragged = ["swa-ragged", "full-ragged"]
        updater.kv_indptr = [torch.zeros(2, dtype=torch.int32) for _ in range(2)]
        updater.qo_indptr = [torch.zeros(2, dtype=torch.int32) for _ in range(2)]
        updater._swa_kv_pool = None
        updater.call_begin_forward = Mock()

        updater.update_sliding_window(
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([1024], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([1024], dtype=torch.int32),
            seq_lens_sum=1024,
            prefix_lens=torch.tensor([0], dtype=torch.int32),
            prefill_wrappers=["swa-paged", "full-paged"],
            use_ragged=True,
            encoder_lens=None,
            spec_info=None,
        )

        self.assertEqual(
            [
                (
                    call.args[0],
                    call.kwargs["num_qo_heads"],
                    call.kwargs["window_left"],
                    call.kwargs["ragged_window_left"],
                )
                for call in updater.call_begin_forward.call_args_list
            ],
            [
                ("swa-ragged", 72, -1, 511),
                ("full-ragged", 48, -1, -1),
            ],
        )

        updater.call_begin_forward.reset_mock()
        updater.update_sliding_window(
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([1024], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([1024], dtype=torch.int32),
            seq_lens_sum=1024,
            prefix_lens=torch.tensor([1000], dtype=torch.int32),
            prefill_wrappers=["swa-verify", "full-verify"],
            use_ragged=False,
            encoder_lens=None,
            spec_info=object(),
        )

        self.assertEqual(
            [
                (
                    call.kwargs["num_qo_heads"],
                    call.kwargs["window_left"],
                    call.kwargs["ragged_window_left"],
                )
                for call in updater.call_begin_forward.call_args_list
            ],
            [(72, -1, -1), (48, -1, -1)],
        )

        updater.call_begin_forward.reset_mock()
        updater.update_sliding_window(
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([1024], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([1024], dtype=torch.int32),
            seq_lens_sum=1024,
            prefix_lens=torch.tensor([1000], dtype=torch.int32),
            prefill_wrappers=["swa-paged", "full-paged"],
            use_ragged=False,
            encoder_lens=None,
            spec_info=None,
        )

        self.assertEqual(
            [
                (
                    call.kwargs["num_qo_heads"],
                    call.kwargs["window_left"],
                    call.kwargs["ragged_window_left"],
                )
                for call in updater.call_begin_forward.call_args_list
            ],
            [(72, 511, -1), (48, -1, -1)],
        )

        updater.call_begin_forward.reset_mock()
        multi_item_params = Mock()
        multi_item_params.is_enabled.return_value = True
        updater.update_sliding_window(
            req_pool_indices=torch.tensor([0], dtype=torch.int32),
            seq_lens=torch.tensor([1024], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([1024], dtype=torch.int32),
            seq_lens_sum=1024,
            prefix_lens=torch.tensor([1000], dtype=torch.int32),
            prefill_wrappers=["swa-mis", "full-mis"],
            use_ragged=False,
            encoder_lens=None,
            spec_info=None,
            multi_item_params=multi_item_params,
        )

        self.assertEqual(
            [
                (
                    call.kwargs["num_qo_heads"],
                    call.kwargs["window_left"],
                    call.kwargs["ragged_window_left"],
                )
                for call in updater.call_begin_forward.call_args_list
            ],
            [(72, -1, -1), (48, -1, -1)],
        )


if __name__ == "__main__":
    unittest.main()
