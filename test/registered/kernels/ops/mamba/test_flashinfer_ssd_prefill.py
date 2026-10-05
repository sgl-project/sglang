"""Checkpoint-preserving FlashInfer SSD prefill correctness on Blackwell."""

import itertools
import json
import unittest

import torch

from sglang.kernels.ops.mamba.flashinfer_ssd import (
    flashinfer_ssd_prefill,
    prepare_ssd_prefill_metadata,
)
from sglang.kernels.ops.mamba.triton_ops.ssd_combined import mamba_chunk_scan_combined
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def check_error(name, actual, expected):
    actual, expected = actual.float(), expected.float()
    rms = expected.square().mean().sqrt().clamp_min(1e-6)
    error = actual - expected
    relative_rms = (error.square().mean().sqrt() / rms).item()
    scaled_max = (error.abs() / (expected.abs() + rms)).max().item()
    print(
        json.dumps(dict(metric=name, relative_rms=relative_rms, scaled_max=scaled_max)),
        flush=True,
    )
    assert torch.isfinite(actual).all(), name
    # BF16 operands, FP32 accumulation, FP16 state: quantify both global and
    # pointwise drift rather than hiding cancellation behind relative error.
    assert relative_rms <= 1 / 128, (name, relative_rms)
    assert scaled_max <= 4 / 128, (name, scaled_max)


class SSDCase:
    def __init__(self, lengths, checkpoints, heads=8, groups=2, warm=True):
        self.lengths, self.checkpoints = lengths, checkpoints
        self.total, self.heads, self.groups = sum(lengths), heads, groups
        device = torch.device("cuda")
        torch.manual_seed(187)
        # Projection slices deliberately have realistic noncompact token strides.
        projected = (
            torch.randn(
                1,
                self.total,
                heads * 64 + 2 * groups * 128,
                device=device,
                dtype=torch.bfloat16,
            )
            * 0.2
        )
        x, B, C = projected.split([heads * 64, groups * 128, groups * 128], dim=-1)
        self.x = x.view(1, self.total, heads, 64)
        self.B = B.view(1, self.total, groups, 128)
        self.C = C.view(1, self.total, groups, 128)
        self.dt = torch.randn(1, self.total, heads, device=device, dtype=torch.bfloat16)
        self.A = -(torch.rand(heads, device=device) + 0.2)
        self.D = torch.randn(heads, device=device, dtype=torch.bfloat16)
        self.bias = torch.full((heads,), -3.0, device=device)
        self.initial = (
            torch.randn(
                len(lengths), heads, 64, 128, device=device, dtype=torch.float16
            )
            * 0.1
        )
        self.initial[::2] = 0
        if not warm:
            self.initial.zero_()
        self.slots = [2 * row + 1 for row in range(len(lengths))]
        self.meta = prepare_ssd_prefill_metadata(
            lengths, device, checkpoints, self.slots
        )
        self.ref_meta = prepare_ssd_prefill_metadata(lengths, device)
        self.cu = torch.tensor(
            [0] + list(itertools.accumulate(lengths)), dtype=torch.int32, device=device
        )
        self.slot_tensor = torch.tensor(self.slots, dtype=torch.long, device=device)
        self.pool = torch.full(
            (2 * len(lengths) + 1, heads, 64, 128),
            13,
            device=device,
            dtype=torch.float16,
        )
        self.out = torch.empty_like(self.x)
        self.ref_out = torch.empty_like(self.x)

    def run(self):
        return flashinfer_ssd_prefill(
            self.x,
            self.dt,
            self.A,
            self.B,
            self.C,
            D=self.D,
            dt_bias=self.bias,
            initial_states=self.initial,
            metadata=self.meta,
            out=self.out,
            checkpoint_states=self.pool,
        )

    def reference(self, checkpoints=True):
        active = [row for row, end in enumerate(self.checkpoints) if end > 0]
        starts = [0] + list(itertools.accumulate(self.lengths))
        kwargs = {}
        if checkpoints:
            kwargs.update(
                track_seq_idx=torch.tensor(active, device="cuda", dtype=torch.int32),
                track_end_locs=torch.tensor(
                    [starts[row] + self.checkpoints[row] for row in active],
                    device="cuda",
                    dtype=torch.int32,
                ),
            )
        # Drop the dummy padding sequence's logical chunk. Triton natively
        # supports a partial last physical chunk, unlike FlashInfer.
        nlogical = sum(
            (starts[row + 1] - 1) // 128 - starts[row] // 128 + 1
            for row in range(len(self.lengths))
        )
        result = mamba_chunk_scan_combined(
            self.x,
            self.dt,
            self.A,
            self.B,
            self.C,
            128,
            D=self.D,
            dt_bias=self.bias,
            dt_softplus=True,
            initial_states=self.initial,
            seq_idx=self.ref_meta.seq_idx[:, : self.total],
            chunk_indices=self.ref_meta.chunk_indices[:nlogical],
            chunk_offsets=self.ref_meta.chunk_offsets[:nlogical],
            cu_seqlens=self.cu,
            return_varlen_states=True,
            return_intermediate_states=True,
            return_final_states=False,
            return_track_states=checkpoints,
            out=self.ref_out,
            state_dtype=torch.float16,
            **kwargs,
        )
        return result

    def validate(self):
        final = self.run().clone()
        _, ref_final, ref_track = self.reference()
        check_error("output", self.out, self.ref_out)
        check_error("final_state", final, ref_final)
        active = [row for row, end in enumerate(self.checkpoints) if end > 0]
        if active:
            check_error(
                "checkpoint", self.pool[[self.slots[row] for row in active]], ref_track
            )
        for row, end in enumerate(self.checkpoints):
            if end == 0:
                torch.testing.assert_close(
                    self.pool[self.slots[row]], self.initial[row], rtol=0, atol=0
                )
        written = [
            self.slots[row] for row, end in enumerate(self.checkpoints) if end >= 0
        ]
        untouched = [slot for slot in range(len(self.pool)) if slot not in written]
        torch.testing.assert_close(
            self.pool[untouched],
            torch.full_like(self.pool[untouched], 13),
            rtol=0,
            atol=0,
        )


class TestFlashInferSSDPrefill(CustomTestCase):
    @torch.inference_mode()
    def test_scalar_fp32_reference(self):
        case = SSDCase([33, 95], [17, 64])
        final = case.run().clone()
        cursor = 0
        for row, length in enumerate(case.lengths):
            state = case.initial[row].float()
            outputs = []
            checkpoint = None
            for position in range(length):
                token = cursor + position
                dt = torch.nn.functional.softplus(case.dt[0, token].float() + case.bias)
                B = (
                    case.B[0, token]
                    .float()
                    .repeat_interleave(case.heads // case.groups, dim=0)
                )
                C = (
                    case.C[0, token]
                    .float()
                    .repeat_interleave(case.heads // case.groups, dim=0)
                )
                x = case.x[0, token].float()
                state = (
                    state * torch.exp(dt * case.A)[:, None, None]
                    + x[:, :, None] * (dt[:, None] * B)[:, None, :]
                )
                outputs.append(
                    (state * C[:, None, :]).sum(dim=-1) + case.D.float()[:, None] * x
                )
                if position + 1 == case.checkpoints[row]:
                    checkpoint = state.clone()
            check_error(
                "scalar_output",
                case.out[0, cursor : cursor + length],
                torch.stack(outputs),
            )
            check_error("scalar_final", final[row], state)
            check_error("scalar_checkpoint", case.pool[case.slots[row]], checkpoint)
            cursor += length

    @torch.inference_mode()
    def test_packed_checkpoints(self):
        for lengths, ends in [
            ([128], [128]),
            ([17], [17]),
            ([129, 255, 17], [128, 128, 17]),
            ([5, 7, 116, 1], [-1, 7, 64, 1]),
            ([19, 109], [-1, 0]),
            ([4096, 512], [2048, 512]),
        ]:
            for warm in (False, True):
                with self.subTest(lengths=lengths, checkpoints=ends, warm=warm):
                    case = SSDCase(lengths, ends, warm=warm)
                    # Runtime receives translated physical slots on GPU. This
                    # covers zero-endpoint restore as well as native capture.
                    case.meta = prepare_ssd_prefill_metadata(
                        lengths, torch.device("cuda"), ends, case.slot_tensor
                    )
                    case.validate()

    @torch.inference_mode()
    def test_model_shape(self):
        SSDCase([1025, 767, 256], [1024, 640, 256], heads=128, groups=8).validate()

    @torch.inference_mode()
    def test_short_strong_decay(self):
        """exp(-sum(A*dt)) can overflow although exp(A*dt) is bounded.

        H128/G8 selects FlashInfer's prefix-factorized short-input route with
        the default lower clamp. The adapter must use a stable scan instead;
        the long model-shape test never exercises this route.
        """
        case = SSDCase([13], [13], heads=128, groups=8, warm=False)
        case.A.fill_(-100)
        case.dt.zero_()
        case.bias.zero_()
        case.validate()

    @torch.inference_mode()
    def test_transformed_dt_precision(self):
        """BF16 transformed dt exceeded the state-error bound at seed 187.

        Compare to an independent token-by-token FP32 recurrence, not another
        chunked implementation with potentially matching rounding errors.
        """
        batch, length, heads, groups = 16, 256, 128, 8
        case = SSDCase([length] * batch, [length] * batch, heads, groups)
        actual = case.run().clone()
        state = case.initial.float()
        x = case.x.reshape(batch, length, heads, 64)
        dt = case.dt.reshape(batch, length, heads)
        B = case.B.reshape(batch, length, groups, 128)
        for token in range(length):
            delta = torch.nn.functional.softplus(dt[:, token].float() + case.bias)
            projection = B[:, token].float().repeat_interleave(heads // groups, dim=1)
            state = (
                state * torch.exp(delta * case.A)[:, :, None, None]
                + x[:, token].float()[:, :, :, None]
                * (delta[:, :, None] * projection)[:, :, None, :]
            )
        check_error("transformed_dt_precision", actual, state)
        self.assertTrue(torch.equal(actual, case.pool[case.slot_tensor]))

    @torch.inference_mode()
    def test_repeated_chunks(self):
        case = SSDCase([129, 255], [128, 128])
        reference_initial = case.initial.clone()
        for step in range(8):
            final = case.run().clone()
            case.initial = reference_initial
            _, ref_final, _ = case.reference()
            reference_initial = ref_final.clone()
            check_error(f"chunk_{step}_output", case.out, case.ref_out)
            check_error(f"chunk_{step}_state", final, ref_final)
            case.initial = final


if __name__ == "__main__":
    unittest.main()
