import os
import time
import unittest

import torch

try:
    from sglang.srt.layers.attention.mamba_state_scatter_triton import (
        fused_mamba_state_scatter,
    )

    _FUSED_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    fused_mamba_state_scatter = None
    _FUSED_IMPORT_ERROR = e


def _dtype_from_str(name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(
            f"Unsupported dtype string {name!r}. Supported: {sorted(mapping.keys())}"
        )
    return mapping[name]


def _ref_scatter(dst, src, dst_indices, src_indices, step_indices):
    """Reference implementation using PyTorch advanced indexing."""
    # dst: [L, C, E]
    # src: [L, S, D, E]
    dst[:, dst_indices] = src[:, src_indices, step_indices].to(dst.dtype, copy=False)


def _old_update_like(
    ssm_states,
    intermediate_ssm,
    conv_states,
    intermediate_conv,
    *,
    state_indices_tensor,
    accepted_steps,
    mamba_track_indices=None,
    mamba_steps_to_track=None,
):
    """Matches the pre-optimization logic in update_mamba_state_after_mtp_verify."""
    request_number = accepted_steps.shape[0]
    intermediate_state_indices = torch.arange(
        request_number, dtype=torch.int32, device=accepted_steps.device
    )

    valid_mask = accepted_steps >= 0
    dst_state_indices = state_indices_tensor[valid_mask].to(torch.int64)
    src_state_indices = intermediate_state_indices[valid_mask].to(torch.int64)
    last_steps = accepted_steps[valid_mask].to(torch.int64)

    if dst_state_indices.numel() == 0:
        return

    _ref_scatter(ssm_states, intermediate_ssm, dst_state_indices, src_state_indices, last_steps)
    _ref_scatter(conv_states, intermediate_conv, dst_state_indices, src_state_indices, last_steps)

    if mamba_track_indices is not None:
        assert mamba_steps_to_track is not None
        track_mask = mamba_steps_to_track >= 0
        if not track_mask.any():
            return
        dst_track_indices = mamba_track_indices[track_mask].to(torch.int64)
        src_track_indices = intermediate_state_indices[track_mask].to(torch.int64)
        track_steps = mamba_steps_to_track[track_mask].to(torch.int64)

        _ref_scatter(
            ssm_states,
            intermediate_ssm,
            dst_track_indices,
            src_track_indices,
            track_steps,
        )
        _ref_scatter(
            conv_states,
            intermediate_conv,
            dst_track_indices,
            src_track_indices,
            track_steps,
        )


def _new_update_like(
    ssm_states,
    intermediate_ssm,
    conv_states,
    intermediate_conv,
    *,
    state_indices_tensor,
    accepted_steps,
    mamba_track_indices=None,
    mamba_steps_to_track=None,
):
    """Matches the current (optimized) logic in update_mamba_state_after_mtp_verify."""
    valid_indices = (accepted_steps >= 0).nonzero(as_tuple=True)[0]
    if valid_indices.numel() == 0:
        return

    dst_state_indices = state_indices_tensor.index_select(0, valid_indices).to(torch.int64)
    src_state_indices = valid_indices.to(torch.int64)
    last_steps = accepted_steps.index_select(0, valid_indices).to(torch.int64)

    fused_mamba_state_scatter(
        ssm_states,
        intermediate_ssm,
        dst_state_indices,
        src_state_indices,
        last_steps,
    )
    fused_mamba_state_scatter(
        conv_states,
        intermediate_conv,
        dst_state_indices,
        src_state_indices,
        last_steps,
    )

    if mamba_track_indices is not None:
        assert mamba_steps_to_track is not None
        track_valid_indices = (mamba_steps_to_track >= 0).nonzero(as_tuple=True)[0]
        if track_valid_indices.numel() == 0:
            return

        dst_track_indices = mamba_track_indices.index_select(0, track_valid_indices).to(
            torch.int64
        )
        src_track_indices = track_valid_indices.to(torch.int64)
        track_steps = mamba_steps_to_track.index_select(0, track_valid_indices).to(torch.int64)

        fused_mamba_state_scatter(
            ssm_states,
            intermediate_ssm,
            dst_track_indices,
            src_track_indices,
            track_steps,
        )
        fused_mamba_state_scatter(
            conv_states,
            intermediate_conv,
            dst_track_indices,
            src_track_indices,
            track_steps,
        )


def _time_cuda_ms(fn, iters=50, warmup=10):
    """Measure average CUDA time (ms) using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


class TestMambaStateScatterCorrectness(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_fused_matches_reference(self):
        if fused_mamba_state_scatter is None:
            self.skipTest(f"fused_mamba_state_scatter import failed: {_FUSED_IMPORT_ERROR}")

        torch.manual_seed(0)
        device = torch.device("cuda")

        # Keep sizes moderate so this test is quick.
        L = 8
        B = 32
        C = 49
        D = 5
        ssm_elems = 1024
        conv_elems = 512

        ssm_states0 = torch.empty((L, C, ssm_elems), device=device, dtype=torch.bfloat16)
        conv_states0 = torch.empty((L, C, conv_elems), device=device, dtype=torch.bfloat16)
        intermediate_ssm = torch.randn((L, B, D, ssm_elems), device=device, dtype=torch.bfloat16)
        intermediate_conv = torch.randn((L, B, D, conv_elems), device=device, dtype=torch.bfloat16)

        # unique cache lines (no duplicates) to avoid nondeterministic write order
        state_indices_tensor = torch.randperm(C, device=device, dtype=torch.int64)[:B].to(
            torch.int32
        )

        accepted_steps = torch.randint(0, D, (B,), device=device, dtype=torch.int64)
        # set ~10% invalid
        invalid = torch.rand((B,), device=device) < 0.1
        accepted_steps[invalid] = -1

        # Optional track update
        mamba_track_indices = torch.randperm(C, device=device, dtype=torch.int64)[:B]
        mamba_steps_to_track = torch.randint(0, D, (B,), device=device, dtype=torch.int64)
        track_invalid = torch.rand((B,), device=device) < 0.7
        mamba_steps_to_track[track_invalid] = -1

        ssm_ref = ssm_states0.clone()
        conv_ref = conv_states0.clone()
        ssm_new = ssm_states0.clone()
        conv_new = conv_states0.clone()

        _old_update_like(
            ssm_ref,
            intermediate_ssm,
            conv_ref,
            intermediate_conv,
            state_indices_tensor=state_indices_tensor,
            accepted_steps=accepted_steps,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
        )
        _new_update_like(
            ssm_new,
            intermediate_ssm,
            conv_new,
            intermediate_conv,
            state_indices_tensor=state_indices_tensor,
            accepted_steps=accepted_steps,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
        )

        torch.testing.assert_close(ssm_new, ssm_ref)
        torch.testing.assert_close(conv_new, conv_ref)


class TestMambaStateScatterPerf(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_perf_report_old_vs_new(self):
        """Optional microbenchmark to confirm/attribute negative optimization.

        Enable with: SGLANG_RUN_MAMBA_SCATTER_PERF_TEST=1
        """
        if os.environ.get("SGLANG_RUN_MAMBA_SCATTER_PERF_TEST", "0") != "1":
            self.skipTest("Set SGLANG_RUN_MAMBA_SCATTER_PERF_TEST=1 to run perf test.")
        if fused_mamba_state_scatter is None:
            self.skipTest(f"fused_mamba_state_scatter import failed: {_FUSED_IMPORT_ERROR}")

        torch.manual_seed(0)
        device = torch.device("cuda")

        # Parameterize sizes via env vars so we can match a real model more closely.
        L = int(os.environ.get("SGLANG_MAMBA_SCATTER_LAYERS", "32"))
        B = int(os.environ.get("SGLANG_MAMBA_SCATTER_BATCH", "48"))
        C = int(os.environ.get("SGLANG_MAMBA_SCATTER_CACHE", "49"))
        D = int(os.environ.get("SGLANG_MAMBA_SCATTER_DRAFT_TOKENS", "5"))
        ssm_elems = int(os.environ.get("SGLANG_MAMBA_SCATTER_SSM_ELEMS", "4096"))
        conv_elems = int(os.environ.get("SGLANG_MAMBA_SCATTER_CONV_ELEMS", "512"))
        invalid_ratio = float(os.environ.get("SGLANG_MAMBA_SCATTER_INVALID_RATIO", "0.0"))
        track_ratio = float(os.environ.get("SGLANG_MAMBA_SCATTER_TRACK_RATIO", "0.0"))
        ssm_dtype = _dtype_from_str(os.environ.get("SGLANG_MAMBA_SCATTER_SSM_DTYPE", "bfloat16"))
        conv_dtype = _dtype_from_str(
            os.environ.get("SGLANG_MAMBA_SCATTER_CONV_DTYPE", "bfloat16")
        )

        # Use zeros for dst so each iteration overwrites the same memory.
        ssm_states = torch.zeros((L, C, ssm_elems), device=device, dtype=ssm_dtype)
        conv_states = torch.zeros((L, C, conv_elems), device=device, dtype=conv_dtype)
        intermediate_ssm = torch.randn((L, B, D, ssm_elems), device=device, dtype=ssm_dtype)
        intermediate_conv = torch.randn(
            (L, B, D, conv_elems), device=device, dtype=conv_dtype
        )

        state_indices_tensor = torch.randperm(C, device=device, dtype=torch.int64)[:B].to(
            torch.int32
        )
        accepted_steps = torch.randint(0, D, (B,), device=device, dtype=torch.int64)
        if invalid_ratio > 0:
            invalid = torch.rand((B,), device=device) < invalid_ratio
            accepted_steps[invalid] = -1

        mamba_track_indices = None
        mamba_steps_to_track = None
        if track_ratio > 0:
            mamba_track_indices = torch.randperm(C, device=device, dtype=torch.int64)[:B]
            mamba_steps_to_track = torch.randint(0, D, (B,), device=device, dtype=torch.int64)
            track_invalid = torch.rand((B,), device=device) >= track_ratio
            mamba_steps_to_track[track_invalid] = -1

        def old_fn():
            _old_update_like(
                ssm_states,
                intermediate_ssm,
                conv_states,
                intermediate_conv,
                state_indices_tensor=state_indices_tensor,
                accepted_steps=accepted_steps,
                mamba_track_indices=mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
            )

        def new_fn():
            _new_update_like(
                ssm_states,
                intermediate_ssm,
                conv_states,
                intermediate_conv,
                state_indices_tensor=state_indices_tensor,
                accepted_steps=accepted_steps,
                mamba_track_indices=mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
            )

        # Warm up JIT compilation for triton kernels (and caches for torch indexing)
        old_fn()
        new_fn()
        torch.cuda.synchronize()

        old_ms = _time_cuda_ms(old_fn)
        new_ms = _time_cuda_ms(new_fn)

        # Best-effort breakdown for the new path.
        # (1) index prep
        def new_prep_only():
            valid_indices = (accepted_steps >= 0).nonzero(as_tuple=True)[0]
            if valid_indices.numel() == 0:
                return
            _ = state_indices_tensor.index_select(0, valid_indices).to(torch.int64)
            _ = valid_indices.to(torch.int64)
            _ = accepted_steps.index_select(0, valid_indices).to(torch.int64)

        # (2) kernel only (use precomputed indices)
        valid_indices = (accepted_steps >= 0).nonzero(as_tuple=True)[0]
        if valid_indices.numel() > 0:
            dst_state_indices = state_indices_tensor.index_select(0, valid_indices).to(torch.int64)
            src_state_indices = valid_indices.to(torch.int64)
            last_steps = accepted_steps.index_select(0, valid_indices).to(torch.int64)

            def new_kernel_only():
                fused_mamba_state_scatter(
                    ssm_states,
                    intermediate_ssm,
                    dst_state_indices,
                    src_state_indices,
                    last_steps,
                )
                fused_mamba_state_scatter(
                    conv_states,
                    intermediate_conv,
                    dst_state_indices,
                    src_state_indices,
                    last_steps,
                )

            new_prep_ms = _time_cuda_ms(new_prep_only)
            new_kernel_ms = _time_cuda_ms(new_kernel_only)
        else:
            new_prep_ms = float("nan")
            new_kernel_ms = float("nan")

        num_valid = int((accepted_steps >= 0).sum().item())
        ratio = new_ms / old_ms if old_ms > 0 else float("inf")

        # Print a concise report to help root-cause "negative optimization".
        print(
            "\n[MambaStateScatterPerf]\n"
            f"  shapes: L={L} B={B} C={C} D={D} ssm_elems={ssm_elems} conv_elems={conv_elems}\n"
            f"  dtypes: ssm={ssm_dtype} conv={conv_dtype}\n"
            f"  valid: {num_valid}/{B}  invalid_ratio={invalid_ratio}  track_ratio={track_ratio}\n"
            f"  old_total_ms: {old_ms:.4f}\n"
            f"  new_total_ms: {new_ms:.4f}  (ratio={ratio:.3f}x)\n"
            f"  new_prep_ms:  {new_prep_ms:.4f}\n"
            f"  new_kernel_ms:{new_kernel_ms:.4f}\n"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

