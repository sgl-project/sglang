"""Correctness coverage for Qwen4 packed-varlen PLE short convolution."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.mamba import qwen4_short_conv
from sglang.kernels.ops.mamba.qwen4_short_conv import (
    can_fuse_qwen4_varlen_conv,
    fused_qwen4_varlen_conv,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen4_exp
from sglang.srt.models.qwen4_exp import _use_qwen4_varlen_prefill
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")


def can_fuse_qwen4_direct_decode_conv(*args, **kwargs):
    assert hasattr(qwen4_short_conv, "can_fuse_qwen4_direct_decode_conv")
    return qwen4_short_conv.can_fuse_qwen4_direct_decode_conv(*args, **kwargs)


def fused_qwen4_direct_decode_conv(*args, **kwargs):
    assert hasattr(qwen4_short_conv, "fused_qwen4_direct_decode_conv")
    return qwen4_short_conv.fused_qwen4_direct_decode_conv(*args, **kwargs)


def use_qwen4_direct_decode(*args, **kwargs):
    assert hasattr(qwen4_exp, "_use_qwen4_direct_decode")
    return qwen4_exp._use_qwen4_direct_decode(*args, **kwargs)


def _metadata(lengths, state_indices):
    lengths_tensor = torch.tensor(lengths, device="cuda", dtype=torch.long)
    query_start_loc = torch.cat([lengths_tensor.new_zeros(1), lengths_tensor.cumsum(0)])
    req_indices = torch.repeat_interleave(
        torch.arange(len(lengths), device="cuda"), lengths_tensor
    )
    token_offsets = (
        torch.arange(sum(lengths), device="cuda") - query_start_loc[req_indices]
    )
    return (
        query_start_loc,
        req_indices,
        token_offsets,
        torch.tensor(state_indices, device="cuda", dtype=torch.long),
    )


def _inputs(
    tokens,
    channels,
    slots,
    kernel_size,
    dilation,
    seed,
    *,
    dtype=torch.bfloat16,
    state_dtype=torch.bfloat16,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state_len = (kernel_size - 1) * dilation
    x = torch.randn(
        (tokens, channels),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    weight = torch.randn(
        (channels, 1, kernel_size),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    state = torch.randn(
        (slots, channels, state_len),
        device="cuda",
        dtype=state_dtype,
        generator=generator,
    )
    return x, weight, state


def _reference(x, weight, state, query_start_loc, state_indices, dilation):
    output = torch.empty_like(x)
    next_state = state.clone()
    state_len = state.shape[2]
    for req, slot in enumerate(state_indices.tolist()):
        start = int(query_start_loc[req])
        end = int(query_start_loc[req + 1])
        if start == end:
            continue
        conv_input = torch.cat(
            [state[slot].to(x.dtype), x[start:end].T], dim=1
        ).unsqueeze(0)
        conv = F.conv1d(
            conv_input.float(),
            weight.float(),
            dilation=dilation,
            groups=x.shape[1],
        ).to(x.dtype)
        output[start:end] = conv.squeeze(0).T
        if slot != 0:
            next_state[slot] = conv_input[0, :, end - start : end - start + state_len]
    return output, next_state


def _fused_output_reference(
    x, residual, weight, state, query_start_loc, state_indices, dilation
):
    conv, next_state = _reference(
        x, weight, state, query_start_loc, state_indices, dilation
    )
    return residual + F.silu(conv), next_state


def _decode_reference(
    x, residual, weight, state, state_indices, dilation, track_indices=None
):
    selected = state.index_select(0, state_indices).to(dtype=x.dtype)
    conv_input = torch.cat([selected, x.unsqueeze(-1)], dim=-1)
    conv = F.conv1d(
        conv_input,
        weight.to(dtype=x.dtype),
        dilation=dilation,
        groups=x.shape[1],
    ).squeeze(-1)
    output = residual + F.silu(conv)
    next_state = conv_input[:, :, 1:]
    expected_state = state.clone()
    real = state_indices.ne(0)
    expected_state[state_indices[real]] = next_state[real].to(dtype=state.dtype)
    if track_indices is not None:
        tracked = track_indices.ne(0)
        expected_state[track_indices[tracked]] = next_state[tracked].to(
            dtype=state.dtype
        )
    return output, expected_state


@requires_cuda
class TestQwen4ShortConv(CustomTestCase):
    def test_direct_decode_matches_native_for_batch_dtype_state_and_reuse(self):
        for batch_size in (1, 8, 32):
            for dtype in (torch.bfloat16, torch.float16):
                for state_dtype in (dtype, torch.float32):
                    x, weight, initial_state = _inputs(
                        batch_size,
                        257,
                        batch_size + 3,
                        kernel_size=3,
                        dilation=4,
                        seed=1000 + batch_size,
                        state_dtype=state_dtype,
                    )
                    x = x.to(dtype=dtype)
                    weight = weight.to(dtype=dtype)
                    residual = torch.randn_like(x)
                    indices = torch.arange(
                        2, batch_size + 2, device="cuda", dtype=torch.long
                    )
                    actual_state = initial_state.clone()
                    actual_state[indices] = 0
                    # Iteration one is a fresh slot; iteration two reuses the
                    # state advanced by the direct kernel.
                    for _ in range(2):
                        expected, expected_state = _decode_reference(
                            x,
                            residual,
                            weight,
                            actual_state,
                            indices,
                            dilation=4,
                        )
                        actual = fused_qwen4_direct_decode_conv(
                            x,
                            residual,
                            weight,
                            actual_state,
                            indices,
                            dilation=4,
                        )
                        if dtype == torch.bfloat16:
                            self.assertTrue(torch.equal(actual, expected))
                        else:
                            torch.testing.assert_close(
                                actual, expected, rtol=5e-4, atol=5e-4
                            )
                        self.assertTrue(torch.equal(actual_state, expected_state))

    def test_direct_decode_supports_strided_state_track_and_padding_slot(self):
        batch_size = 8
        x, weight, backing = _inputs(
            batch_size,
            129,
            24,
            kernel_size=3,
            dilation=4,
            seed=2001,
            state_dtype=torch.float32,
        )
        strided_backing = torch.empty((24, 129, 16), device="cuda", dtype=torch.float32)
        state = strided_backing[:, :, ::2]
        state.copy_(backing)
        self.assertFalse(state.is_contiguous())
        residual = torch.randn_like(x)
        indices = torch.tensor(
            [0, 2, 4, 6, 8, 10, 12, 14], device="cuda", dtype=torch.long
        )
        track_indices = torch.tensor(
            [0, 3, 5, 7, 9, 11, 13, 15], device="cuda", dtype=torch.long
        )
        slot_zero = state[0].clone()
        expected, expected_state = _decode_reference(
            x,
            residual,
            weight,
            state,
            indices,
            dilation=4,
            track_indices=track_indices,
        )

        self.assertTrue(
            can_fuse_qwen4_direct_decode_conv(
                x, residual, weight, state, indices, 4, track_indices
            )
        )
        actual = fused_qwen4_direct_decode_conv(
            x,
            residual,
            weight,
            state,
            indices,
            dilation=4,
            track_indices=track_indices,
        )

        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        self.assertTrue(torch.equal(state, expected_state))
        self.assertTrue(torch.equal(state[0], slot_zero))

    def test_direct_decode_cuda_graph_replay_uses_live_inputs_and_state(self):
        batch_size = 8
        x, weight, state = _inputs(
            batch_size, 129, 12, kernel_size=3, dilation=4, seed=3001
        )
        residual = torch.randn_like(x)
        indices = torch.arange(2, batch_size + 2, device="cuda", dtype=torch.long)
        static_state = state.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fused_qwen4_direct_decode_conv(
                x, residual, weight, static_state, indices, dilation=4
            )

        for scale in (0.5, -1.25):
            x.copy_(torch.randn_like(x) * scale)
            residual.copy_(torch.randn_like(residual) * scale)
            before = static_state.clone()
            expected, expected_state = _decode_reference(
                x, residual, weight, before, indices, dilation=4
            )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.02)
            self.assertTrue(torch.equal(static_state, expected_state))

    def test_direct_decode_guard_keeps_unsupported_fallbacks(self):
        x, weight, state = _inputs(1, 65, 4, kernel_size=3, dilation=4, seed=4001)
        residual = torch.randn_like(x)
        indices = torch.tensor([2], device="cuda", dtype=torch.long)
        args = (x, residual, weight, state, indices, 4, None)
        self.assertTrue(can_fuse_qwen4_direct_decode_conv(*args))
        self.assertFalse(
            can_fuse_qwen4_direct_decode_conv(
                x.float(), residual.float(), weight, state, indices, 4, None
            )
        )
        self.assertFalse(
            can_fuse_qwen4_direct_decode_conv(
                x, residual[:, ::2], weight, state, indices, 4, None
            )
        )

    def test_direct_decode_dispatch_is_narrow_and_fusion_gated(self):
        x, weight, state = _inputs(8, 65, 12, kernel_size=3, dilation=4, seed=5001)
        residual = torch.randn_like(x)
        indices = torch.arange(2, 10, device="cuda", dtype=torch.long)
        args = (x, residual, weight, state, indices, 4, None)
        self.assertTrue(use_qwen4_direct_decode(ForwardMode.DECODE, *args))
        self.assertFalse(use_qwen4_direct_decode(ForwardMode.EXTEND, *args))
        with patch.object(
            qwen4_exp.envs.SGLANG_ENABLE_QWEN4_PLE_FUSION,
            "get",
            return_value=False,
        ):
            self.assertFalse(use_qwen4_direct_decode(ForwardMode.DECODE, *args))

    def test_packed_varlen_matches_padded_reference(self):
        lengths = [1, 2, 8, 9, 10, 33, 0]
        state_indices = [7, 2, 9, 4, 11, 6, 0]
        x, weight, initial_state = _inputs(
            sum(lengths), 257, 12, kernel_size=3, dilation=4, seed=1
        )
        query_start_loc, req_indices, token_offsets, state_indices_tensor = _metadata(
            lengths, state_indices
        )
        expected, expected_state = _reference(
            x,
            weight,
            initial_state,
            query_start_loc,
            state_indices_tensor,
            dilation=4,
        )
        residual = torch.randn_like(x)
        expected = residual + F.silu(expected)
        actual_state = initial_state.clone()
        actual = fused_qwen4_varlen_conv(
            x,
            weight,
            actual_state,
            state_indices_tensor,
            query_start_loc,
            req_indices,
            token_offsets,
            dilation=4,
            residual=residual,
        )

        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0.02, atol=0.05
        )
        self.assertTrue(torch.equal(actual_state, expected_state))

    def test_varlen_output_fusion_covers_mixed_equal_fp16_and_fp32_state(self):
        cases = (
            ("mixed", [1, 5, 0, 9], [2, 4, 0, 6]),
            ("equal", [4, 4, 4], [2, 4, 6]),
        )
        for name, lengths, state_indices in cases:
            for dtype in (torch.bfloat16, torch.float16):
                for state_dtype in (dtype, torch.float32):
                    with self.subTest(case=name, dtype=dtype, state_dtype=state_dtype):
                        x, weight, initial_state = _inputs(
                            sum(lengths),
                            129,
                            8,
                            kernel_size=3,
                            dilation=4,
                            seed=6000 + sum(lengths),
                            dtype=dtype,
                            state_dtype=state_dtype,
                        )
                        residual = torch.randn_like(x)
                        meta = _metadata(lengths, state_indices)
                        expected, expected_state = _fused_output_reference(
                            x,
                            residual,
                            weight,
                            initial_state,
                            meta[0],
                            meta[3],
                            dilation=4,
                        )
                        actual_state = initial_state.clone()
                        self.assertTrue(
                            can_fuse_qwen4_varlen_conv(
                                x,
                                weight,
                                actual_state,
                                meta[3],
                                *meta[:3],
                                4,
                                residual,
                            )
                        )
                        actual = fused_qwen4_varlen_conv(
                            x,
                            weight,
                            actual_state,
                            meta[3],
                            *meta[:3],
                            dilation=4,
                            residual=residual,
                        )

                        if dtype == torch.bfloat16:
                            self.assertTrue(torch.equal(actual, expected))
                        else:
                            torch.testing.assert_close(
                                actual, expected, rtol=5e-4, atol=5e-4
                            )
                        self.assertTrue(torch.equal(actual_state, expected_state))

    def test_varlen_output_fusion_preserves_low_precision_rounding_points(self):
        for dtype in (torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                conv_value, residual_value = {
                    torch.bfloat16: (-0.1630859375, -0.0250244140625),
                    torch.float16: (0.11907958984375, 0.11846923828125),
                }[dtype]
                x = torch.tensor(
                    [[conv_value]],
                    device="cuda",
                    dtype=dtype,
                )
                # Only the newest tap contributes, so conv1d materializes the
                # chosen low-precision value exactly before SiLU.
                weight = torch.tensor([[[0.0, 0.0, 1.0]]], device="cuda", dtype=dtype)
                state = torch.zeros((3, 1, 8), device="cuda", dtype=torch.float32)
                residual = torch.tensor(
                    [[residual_value]],
                    device="cuda",
                    dtype=dtype,
                )
                meta = _metadata([1], [2])
                conv, _ = _reference(x, weight, state, meta[0], meta[3], dilation=4)
                expected = residual + F.silu(conv)
                unmaterialized = (residual.float() + F.silu(conv.float())).to(
                    dtype=dtype
                )
                self.assertFalse(torch.equal(expected, unmaterialized))

                actual = fused_qwen4_varlen_conv(
                    x,
                    weight,
                    state,
                    meta[3],
                    *meta[:3],
                    dilation=4,
                    residual=residual,
                )
                self.assertTrue(torch.equal(actual, expected))

    def test_varlen_output_fusion_keeps_raw_output_compatibility_api(self):
        x, weight, initial_state = _inputs(
            5, 65, 5, kernel_size=3, dilation=4, seed=7001
        )
        meta = _metadata([2, 3], [2, 4])
        expected, _ = _reference(x, weight, initial_state, meta[0], meta[3], dilation=4)
        actual = fused_qwen4_varlen_conv(
            x, weight, initial_state.clone(), meta[3], *meta[:3], dilation=4
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_writeback_updates_main_and_track_boundaries(self):
        lengths = [12]
        x, weight, initial_state = _inputs(
            12, 129, 10, kernel_size=3, dilation=4, seed=2
        )
        query_start_loc, req_indices, token_offsets, state_indices = _metadata(
            lengths, [3]
        )
        track_indices = torch.tensor([8], device="cuda", dtype=torch.long)
        track_offsets = torch.tensor([5], device="cuda", dtype=torch.long)
        actual_state = initial_state.clone()
        residual = torch.randn_like(x)
        fused_qwen4_varlen_conv(
            x,
            weight,
            actual_state,
            state_indices,
            query_start_loc,
            req_indices,
            token_offsets,
            dilation=4,
            residual=residual,
            track_indices=track_indices,
            track_offsets=track_offsets,
        )

        full_input = torch.cat([initial_state[3], x.T], dim=1)
        torch.testing.assert_close(actual_state[3], full_input[:, 12:20])
        torch.testing.assert_close(actual_state[8], full_input[:, 5:13])

    def test_chunked_writeback_matches_one_shot(self):
        x, weight, initial_state = _inputs(
            12, 129, 8, kernel_size=3, dilation=4, seed=3
        )
        one_state = initial_state.clone()
        one_meta = _metadata([12], [5])
        residual = torch.randn_like(x)
        one_output = fused_qwen4_varlen_conv(
            x,
            weight,
            one_state,
            one_meta[3],
            *one_meta[:3],
            dilation=4,
            residual=residual,
        )

        chunk_state = initial_state.clone()
        outputs = []
        start = 0
        for chunk in (x[:5], x[5:]):
            meta = _metadata([chunk.shape[0]], [5])
            outputs.append(
                fused_qwen4_varlen_conv(
                    chunk,
                    weight,
                    chunk_state,
                    meta[3],
                    *meta[:3],
                    dilation=4,
                    residual=residual[start : start + chunk.shape[0]],
                )
            )
            start += chunk.shape[0]

        torch.testing.assert_close(
            torch.cat(outputs).float(), one_output.float(), rtol=0.02, atol=0.05
        )
        self.assertTrue(torch.equal(chunk_state, one_state))

    def test_writeback_skips_empty_rows_and_dummy_slot_zero(self):
        x, weight, initial_state = _inputs(2, 65, 8, kernel_size=3, dilation=4, seed=4)
        meta = _metadata([2, 0, 0], [0, 4, 5])
        track_indices = torch.tensor([0, 6, 7], device="cuda", dtype=torch.long)
        track_offsets = torch.tensor([1, 0, 0], device="cuda", dtype=torch.long)
        actual_state = initial_state.clone()
        fused_qwen4_varlen_conv(
            x,
            weight,
            actual_state,
            meta[3],
            *meta[:3],
            dilation=4,
            track_indices=track_indices,
            track_offsets=track_offsets,
        )
        self.assertTrue(torch.equal(actual_state, initial_state))

    def test_fp32_state_matches_bf16_materialization_of_padded_path(self):
        lengths = [1, 5]
        x, weight, initial_state = _inputs(
            sum(lengths),
            65,
            6,
            kernel_size=3,
            dilation=4,
            seed=6,
            state_dtype=torch.float32,
        )
        # Make loss at the old path's FP32 -> BF16 boundary deterministic.
        initial_state.add_(0.00390625)
        meta = _metadata(lengths, [2, 5])
        expected, expected_state = _reference(
            x, weight, initial_state, meta[0], meta[3], dilation=4
        )
        residual = torch.randn_like(x)
        expected = residual + F.silu(expected)
        actual_state = initial_state.clone()
        actual = fused_qwen4_varlen_conv(
            x,
            weight,
            actual_state,
            meta[3],
            *meta[:3],
            dilation=4,
            residual=residual,
        )

        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0.02, atol=0.05
        )
        self.assertTrue(torch.equal(actual_state, expected_state))

    def test_metadata_requires_int64_and_same_cuda_device(self):
        x, weight, state = _inputs(3, 65, 5, kernel_size=3, dilation=4, seed=7)
        metadata = _metadata([1, 2], [2, 4])
        args = [x, weight, state, metadata[3], *metadata[:3], 4]

        for index in range(3, 7):
            invalid = list(args)
            invalid[index] = invalid[index].to(dtype=torch.int32)
            self.assertFalse(can_fuse_qwen4_varlen_conv(*invalid))

        invalid = list(args)
        invalid[1] = invalid[1].cpu()
        self.assertFalse(can_fuse_qwen4_varlen_conv(*invalid))
        if torch.cuda.device_count() > 1:
            invalid = list(args)
            invalid[1] = invalid[1].to("cuda:1")
            self.assertFalse(can_fuse_qwen4_varlen_conv(*invalid))

    def test_track_metadata_requires_int64_and_same_device(self):
        x, weight, state = _inputs(3, 65, 8, kernel_size=3, dilation=4, seed=8)
        metadata = _metadata([3], [2])
        track_indices = torch.tensor([6], device="cuda", dtype=torch.long)
        track_offsets = torch.tensor([2], device="cuda", dtype=torch.long)

        for invalid_indices, invalid_offsets in (
            (track_indices.int(), track_offsets),
            (track_indices, track_offsets.int()),
            (track_indices.cpu(), track_offsets),
        ):
            with self.assertRaisesRegex(ValueError, "track metadata"):
                fused_qwen4_varlen_conv(
                    x,
                    weight,
                    state.clone(),
                    metadata[3],
                    *metadata[:3],
                    dilation=4,
                    track_indices=invalid_indices,
                    track_offsets=invalid_offsets,
                )
        if torch.cuda.device_count() > 1:
            with self.assertRaisesRegex(ValueError, "track metadata"):
                fused_qwen4_varlen_conv(
                    x,
                    weight,
                    state.clone(),
                    metadata[3],
                    *metadata[:3],
                    dilation=4,
                    track_indices=track_indices.to("cuda:1"),
                    track_offsets=track_offsets,
                )

    def test_dispatch_is_limited_to_ordinary_bf16_prefill(self):
        x, weight, state = _inputs(3, 65, 5, kernel_size=3, dilation=4, seed=5)
        metadata = _metadata([1, 2], [2, 4])
        kernel_args = (x, weight, state, metadata[3], *metadata[:3], 4)
        dispatch_args = (
            x,
            weight,
            state,
            metadata[3],
            metadata[1],
            metadata[2],
            4,
        )
        self.assertTrue(can_fuse_qwen4_varlen_conv(*kernel_args))
        for mode in (
            ForwardMode.EXTEND,
            ForwardMode.MIXED,
            ForwardMode.SPLIT_PREFILL,
        ):
            self.assertTrue(
                _use_qwen4_varlen_prefill(mode, metadata[0], *dispatch_args)
            )
        self.assertFalse(
            _use_qwen4_varlen_prefill(
                ForwardMode.TARGET_VERIFY, metadata[0], *dispatch_args
            )
        )
        self.assertFalse(
            can_fuse_qwen4_varlen_conv(
                x.float(), weight, state, metadata[3], *metadata[:3], 4
            )
        )
        with patch.object(
            qwen4_exp.envs.SGLANG_ENABLE_QWEN4_PLE_FUSION,
            "get",
            return_value=False,
        ):
            self.assertFalse(
                _use_qwen4_varlen_prefill(
                    ForwardMode.EXTEND, metadata[0], *dispatch_args
                )
            )
        with patch.object(qwen4_exp, "is_cuda", return_value=False, create=True):
            self.assertFalse(
                _use_qwen4_varlen_prefill(
                    ForwardMode.EXTEND, metadata[0], *dispatch_args
                )
            )


if __name__ == "__main__":
    unittest.main()
