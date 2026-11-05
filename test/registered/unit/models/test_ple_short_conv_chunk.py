"""Exact halo, packed output and PLE state commits for opt-in prefill chunking."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.ple_short_conv import chunked_packed_short_conv
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen4_exp as model
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _bits(tensor):
    return tensor.contiguous().view(torch.int16)


class TestPackedShortConv(unittest.TestCase):
    @torch.inference_mode()
    def test_bf16_output_bits_ragged_lengths_kernel_and_dilation(self):
        g = torch.Generator().manual_seed(91)
        for kernel, dilation, lengths, channels in (
            (1, 1, [0, 1, 31], 4),
            (2, 3, [2, 0, 67, 9], 8),
            (4, 1, [129, 1, 33], 16),
            (4, 3, [257, 0, 1, 9, 17], 32),
            (4, 4, [1, 2, 129], 8),
        ):
            halo = (kernel - 1) * dilation
            width = max(lengths)
            inputs = torch.randn(
                len(lengths), channels, width + halo, generator=g
            ).bfloat16()
            weights = torch.randn(channels, 1, kernel, generator=g).bfloat16()
            reference = F.conv1d(inputs, weights, groups=channels, dilation=dilation)
            reference = torch.cat(
                [reference[i, :, :n].T for i, n in enumerate(lengths)]
            )
            for chunk_size in (1, 2, 7, 16, 64, width, width + 1):
                with self.subTest(kernel=kernel, dilation=dilation, chunk=chunk_size):
                    got = chunked_packed_short_conv(
                        inputs,
                        weights,
                        lengths,
                        dilation=dilation,
                        chunk_size=chunk_size,
                    )
                    self.assertTrue(torch.equal(_bits(got), _bits(reference)))

    @torch.inference_mode()
    def test_halo_impulses_signed_zero_and_real_channel_width(self):
        channels, kernel, dilation = 10240, 4, 3
        halo, width, chunk_size = 9, 33, 7
        inputs = torch.zeros(1, channels, width + halo, dtype=torch.bfloat16)
        inputs[:, :, 0] = -0.0
        inputs[:, :, halo - 1] = 4
        for boundary in range(chunk_size, width, chunk_size):
            inputs[:, :, boundary + halo - 1] = -3
            inputs[:, :, boundary + halo] = 2
        weights = torch.tensor([0.25, -0.5, 1, 0.75], dtype=torch.bfloat16)
        weights = weights.expand(channels, 1, kernel).contiguous()
        reference = F.conv1d(inputs, weights, groups=channels, dilation=dilation)[0].T
        got = chunked_packed_short_conv(
            inputs, weights, [width], dilation=dilation, chunk_size=chunk_size
        )
        self.assertTrue(torch.equal(_bits(got), _bits(reference)))

    @torch.inference_mode()
    def test_reset_halo_negative_control_is_detected(self):
        inputs = torch.arange(1, 27, dtype=torch.bfloat16).reshape(1, 1, 26)
        weights = torch.ones(1, 1, 4, dtype=torch.bfloat16)
        reference = F.conv1d(inputs, weights, dilation=3)[0].T
        got = chunked_packed_short_conv(inputs, weights, [17], dilation=3, chunk_size=5)
        wrong = torch.cat(
            [
                F.conv1d(
                    F.pad(inputs[:, :, 9 + i : 9 + min(i + 5, 17)], (9, 0)),
                    weights,
                    dilation=3,
                )[0].T
                for i in range(0, 17, 5)
            ]
        )
        self.assertTrue(torch.equal(_bits(got), _bits(reference)))
        self.assertFalse(torch.equal(_bits(wrong), _bits(reference)))

    @torch.inference_mode()
    def test_input_stride_and_no_input_or_weight_writes(self):
        inputs = torch.randn(2, 8, 88).bfloat16()[:, :, ::2]
        weights = torch.randn(8, 1, 4).bfloat16()
        before, weight_before = inputs.clone(), weights.clone()
        lengths = [35, 7]
        ref = F.conv1d(inputs, weights, groups=8, dilation=3)
        ref = torch.cat([ref[i, :, :n].T for i, n in enumerate(lengths)])
        got = chunked_packed_short_conv(
            inputs, weights, lengths, dilation=3, chunk_size=6
        )
        self.assertTrue(torch.equal(_bits(got), _bits(ref)))
        self.assertTrue(torch.equal(_bits(inputs), _bits(before)))
        self.assertTrue(torch.equal(_bits(weights), _bits(weight_before)))

    @torch.inference_mode()
    def test_bounded_conv_shapes_and_packed_allocation(self):
        inputs = torch.randn(3, 16, 138).bfloat16()
        weights = torch.randn(16, 1, 4).bfloat16()
        lengths, chunk_size, halo = [129, 5, 0], 16, 9
        native = F.conv1d
        with patch("sglang.srt.layers.ple_short_conv.F.conv1d", wraps=native) as spy:
            got = chunked_packed_short_conv(
                inputs, weights, lengths, dilation=3, chunk_size=chunk_size
            )
        self.assertEqual(got.shape, (sum(lengths), 16))
        self.assertEqual(
            spy.call_count, sum(math.ceil(n / chunk_size) for n in lengths)
        )
        for call in spy.call_args_list:
            slab = call.args[0]
            self.assertEqual(slab.shape[0], 1)
            self.assertLessEqual(slab.shape[2], chunk_size + halo)
            self.assertTrue(slab.is_contiguous())

    def test_bad_chunk_size_rejected_without_native_call(self):
        with patch("sglang.srt.layers.ple_short_conv.F.conv1d") as spy:
            with self.assertRaises(ValueError):
                chunked_packed_short_conv(
                    torch.empty(1, 8, 12),
                    torch.empty(8, 1, 4),
                    [3],
                    dilation=3,
                    chunk_size=0,
                )
        spy.assert_not_called()

    def test_selected_native_error_propagates_without_retry(self):
        error = RuntimeError("selected conv failure")
        with patch(
            "sglang.srt.layers.ple_short_conv.F.conv1d", side_effect=error
        ) as spy:
            with self.assertRaisesRegex(RuntimeError, "selected conv failure"):
                chunked_packed_short_conv(
                    torch.empty(1, 8, 26),
                    torch.empty(8, 1, 4),
                    [17],
                    dilation=3,
                    chunk_size=4,
                )
        self.assertEqual(spy.call_count, 1)


class TestPleShortConvIntegration(unittest.TestCase):
    device = "cpu"

    def setUp(self):
        self.g = torch.Generator(device=self.device).manual_seed(112)

    def _run(
        self,
        chunk_size,
        *,
        lengths=(73, 0, 5, 16),
        host_lengths="valid",
        mode=ForwardMode.EXTEND,
        capture=False,
        breakable=False,
        track=True,
        state_dtype=torch.bfloat16,
        input_dtype=torch.bfloat16,
    ):
        self.g.manual_seed(112)
        channels, kernel, dilation = 16, 4, 3
        halo, width = (kernel - 1) * dilation, max(lengths)
        row_count = len(lengths)
        state_indices = torch.arange(1, row_count + 1, device=self.device)
        state = torch.randn(
            2 * row_count + 2, channels, halo, generator=self.g, device=self.device
        ).to(state_dtype)
        x = torch.randn(
            sum(lengths), channels, generator=self.g, device=self.device
        ).to(input_dtype)
        weight = torch.randn(
            channels, 1, kernel, generator=self.g, device=self.device
        ).bfloat16()
        req = torch.repeat_interleave(
            torch.arange(row_count, device=self.device),
            torch.tensor(lengths, device=self.device),
        )
        offsets = torch.cat([torch.arange(n, device=self.device) for n in lengths])
        lengths_tensor = torch.tensor(lengths, device=self.device)
        batch = SimpleNamespace(
            use_decode_fast_path=False,
            lengths=lengths_tensor,
            state_indices=state_indices,
            row_width=width,
            mode=mode,
            req_indices=req,
            token_offsets=offsets,
            valid_tokens=torch.ones(sum(lengths), dtype=torch.bool, device=self.device),
        )
        view = SimpleNamespace(
            layer_cache=lambda _: state,
            layer_intermediate_cache=lambda _: None,
        )
        pool = SimpleNamespace(
            layer_router=SimpleNamespace(
                read=lambda *_: SimpleNamespace(view=view, view_layer_id=0)
            )
        )
        forward_batch = SimpleNamespace(
            extend_seq_lens_cpu=(
                list(lengths) if host_lengths == "valid" else host_lengths
            ),
            mamba_track_indices=torch.arange(
                row_count + 1, 2 * row_count + 1, device=self.device
            ),
            mamba_track_mask=torch.tensor(
                [True, False, True, True][:row_count], device=self.device
            ),
            mamba_track_aligned_lens=lambda: torch.tensor(
                [max(0, n - 3) for n in lengths], device=self.device
            ),
        )
        if not track:
            forward_batch.mamba_track_indices = None
            forward_batch.mamba_track_mask = None
        layer = SimpleNamespace(
            layer_id=0,
            short_conv_state_len=halo,
            short_conv_dilation=dilation,
            conv_channels=channels,
            conv1d=SimpleNamespace(weight=weight),
        )
        native_chunked = model.chunked_packed_short_conv
        with (
            envs.SGLANG_PLE_PREFILL_CONV_CHUNK_SIZE.override(chunk_size),
            patch.object(model, "get_token_to_kv_pool", return_value=pool),
            patch.object(model, "get_is_capture_mode", return_value=capture),
            patch.object(model, "is_in_breakable_cuda_graph", return_value=breakable),
            patch.object(
                model, "chunked_packed_short_conv", wraps=native_chunked
            ) as spy,
            torch.inference_mode(),
        ):
            output = model.Qwen4ExpPLELayer._short_conv(layer, x, forward_batch, batch)
        return output, state, spy.call_count

    def test_actual_method_conv_silu_state_and_track_bits(self):
        for track in (False, True):
            for state_dtype in (torch.bfloat16, torch.float32):
                baseline, state_ref, _ = self._run(
                    0, track=track, state_dtype=state_dtype
                )
                for size in (1, 4, 16, 64):
                    with self.subTest(track=track, state_dtype=state_dtype, chunk=size):
                        got, state, calls = self._run(
                            size, track=track, state_dtype=state_dtype
                        )
                        self.assertEqual(calls, 1)
                        self.assertTrue(torch.equal(_bits(got), _bits(baseline)))
                        self.assertTrue(torch.equal(_bits(state), _bits(state_ref)))

    def test_default_off_and_nonselected_phases_keep_stock(self):
        self.assertEqual(envs.SGLANG_PLE_PREFILL_CONV_CHUNK_SIZE.default, 0)
        for kwargs in (
            {"host_lengths": None},
            {"host_lengths": [73, 0, 5]},
            {"host_lengths": [73, 0, 5, 15]},
            {"host_lengths": [73, 0, -1, 22]},
            {"mode": ForwardMode.TARGET_VERIFY},
            {"capture": True, "breakable": False},
        ):
            with self.subTest(kwargs=kwargs):
                want, state_ref, _ = self._run(0, **kwargs)
                got, state, calls = self._run(16, **kwargs)
                self.assertEqual(calls, 0)
                self.assertTrue(torch.equal(_bits(got), _bits(want)))
                self.assertTrue(torch.equal(_bits(state), _bits(state_ref)))

    def test_breakable_eager_body_uses_live_lengths(self):
        baseline, state_ref, _ = self._run(0, capture=True, breakable=True)
        got, state, calls = self._run(16, capture=True, breakable=True)
        self.assertEqual(calls, 1)
        self.assertTrue(torch.equal(_bits(got), _bits(baseline)))
        self.assertTrue(torch.equal(_bits(state), _bits(state_ref)))

    def test_other_input_dtypes_keep_stock(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                got, state, calls = self._run(16, input_dtype=dtype)
                ref, state_ref, _ = self._run(0, input_dtype=dtype)
                self.assertEqual(calls, 0)
                self.assertTrue(torch.equal(_bits(got), _bits(ref)))
                self.assertTrue(torch.equal(_bits(state), _bits(state_ref)))

    def test_empty_input_returns_without_conv_or_state_commit(self):
        got, state, calls = self._run(16, lengths=(0, 0, 0, 0))
        ref, state_ref, _ = self._run(0, lengths=(0, 0, 0, 0))
        self.assertEqual(got.shape[0], 0)
        self.assertEqual(calls, 0)
        self.assertTrue(torch.equal(_bits(got), _bits(ref)))
        self.assertTrue(torch.equal(_bits(state), _bits(state_ref)))

    def test_short_sequence_keeps_stock(self):
        got, state, calls = self._run(128)
        ref, state_ref, _ = self._run(0)
        self.assertEqual(calls, 0)
        self.assertTrue(torch.equal(_bits(got), _bits(ref)))
        self.assertTrue(torch.equal(_bits(state), _bits(state_ref)))


if __name__ == "__main__":
    unittest.main()
