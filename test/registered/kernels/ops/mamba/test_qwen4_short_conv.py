"""Correctness coverage for Qwen4 PLE short convolution."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    fused_mamba_state_scatter_with_mask,
)
from sglang.kernels.ops.mamba.qwen4_short_conv import (
    can_fuse_qwen4_varlen_conv,
    fused_qwen4_varlen_conv,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models import qwen4_exp
from sglang.srt.models.qwen4_exp import _use_qwen4_varlen_prefill
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")

requires_cuda = unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")


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
    state_dtype=torch.bfloat16,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    state_len = (kernel_size - 1) * dilation
    x = torch.randn(
        (tokens, channels),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    weight = torch.randn(
        (channels, 1, kernel_size),
        device="cuda",
        dtype=torch.bfloat16,
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


@requires_cuda
class TestQwen4ShortConv(CustomTestCase):
    def test_fused_verify_commits_expected_windows(self):
        width, channels, history = 3, 8, 2
        pool_size = 6
        x = torch.arange(
            1, 3 * width * channels + 1, device="cuda", dtype=torch.bfloat16
        ).reshape(3 * width, channels)
        state_slots = torch.tensor([3, 5, 0], device="cuda")
        # Row 0 is reserved; the maximum request row must remain live.
        request_rows = torch.tensor([4, pool_size, 0], device="cuda")
        accepted = torch.tensor([1, 2, -1], device="cuda")
        lengths = torch.tensor([width, width, 0], device="cuda")
        cache_locs = torch.arange(1, x.shape[0] + 1, device=x.device)
        cache_locs[-width:] = 0
        mode = ForwardMode.TARGET_VERIFY
        forward_batch = SimpleNamespace(
            _original_forward_mode=None,
            forward_mode=mode,
            req_pool_indices=request_rows,
            tbo_parent_token_range=None,
            spec_algorithm=None,
            spec_info=SimpleNamespace(topk=1, draft_token_num=width),
            global_num_token_non_padded_cpu=None,
            extend_seq_lens_cpu=None,
            out_cache_loc=cache_locs,
        )
        layer = qwen4_exp.Qwen4ExpPLELayer.__new__(qwen4_exp.Qwen4ExpPLELayer)
        torch.nn.Module.__init__(layer)
        layer.layer_id = 0
        layer.hidden_size = channels
        layer.hc_count = 1
        layer._prefetch_state = None
        layer.ple_embedding = Mock(return_value=2 * x, enable_ple_fusion=True)
        layer.key_proj = layer.value_proj = lambda value: (value, None)
        layer.norm_key = layer.norm_query = layer.norm_conv = torch.nn.Identity()
        layer.short_conv_dilation = 1
        layer.conv1d = SimpleNamespace(weight=x.new_ones((channels, 1, history + 1)))
        layer._short_conv = Mock(side_effect=AssertionError("expected fused verify"))

        state = x.new_zeros((pool_size + 1, channels, history))
        windows = x.new_zeros((pool_size + 1, width, channels, history))
        view = SimpleNamespace(
            layer_cache=lambda _: state,
            layer_intermediate_cache=lambda _: windows,
        )
        router = SimpleNamespace(
            read=lambda *_: SimpleNamespace(view=view, view_layer_id=0)
        )
        expected = state.clone()
        expected[3] = x[:2].T
        expected[5] = x[width + 1 : 2 * width].T

        for pp_spec in (False, True):
            for graph_padding in (False, True):
                with (
                    self.subTest(pp_spec=pp_spec, graph_padding=graph_padding),
                    qwen4_exp.envs.SGLANG_ENABLE_PP_SPEC.override(pp_spec),
                    patch.object(qwen4_exp.layernorm_sp, "runs_sp", return_value=False),
                    patch.object(
                        qwen4_exp,
                        "get_token_to_kv_pool",
                        return_value=SimpleNamespace(layer_router=router),
                    ),
                    patch.object(
                        qwen4_exp,
                        "get_req_to_token_pool",
                        return_value=SimpleNamespace(size=pool_size),
                    ),
                    patch.object(
                        qwen4_exp,
                        "get_attn_backend",
                        return_value=SimpleNamespace(
                            forward_state_slots=lambda *_: SimpleNamespace(
                                working=state_slots
                            )
                        ),
                    ),
                ):
                    forward_batch.extend_seq_lens = None if graph_padding else lengths
                    batch = qwen4_exp._prepare_ple_batch(
                        x.new_zeros(x.shape[0], dtype=torch.long),
                        forward_batch,
                        ngram_size=None,
                        ngram_eos_token_id=None,
                    )
                    windows.zero_()
                    layer(torch.zeros_like(x), forward_batch, batch)
                    committed = state.clone().unsqueeze(0)
                    fused_mamba_state_scatter_with_mask(
                        committed,
                        windows.unsqueeze(0),
                        state_slots,
                        accepted,
                        request_rows if pp_spec else None,
                    )
                    torch.testing.assert_close(committed[0], expected, atol=0, rtol=0)

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
        )

        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=0.02, atol=0.05
        )
        self.assertTrue(torch.equal(actual_state, expected_state))

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
        fused_qwen4_varlen_conv(
            x,
            weight,
            actual_state,
            state_indices,
            query_start_loc,
            req_indices,
            token_offsets,
            dilation=4,
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
        one_output = fused_qwen4_varlen_conv(
            x, weight, one_state, one_meta[3], *one_meta[:3], dilation=4
        )

        chunk_state = initial_state.clone()
        outputs = []
        for chunk in (x[:5], x[5:]):
            meta = _metadata([chunk.shape[0]], [5])
            outputs.append(
                fused_qwen4_varlen_conv(
                    chunk, weight, chunk_state, meta[3], *meta[:3], dilation=4
                )
            )

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
        actual_state = initial_state.clone()
        actual = fused_qwen4_varlen_conv(
            x, weight, actual_state, meta[3], *meta[:3], dilation=4
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
