from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=7, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=7, suite="stage-b-test-1-gpu-small-amd-mi35x")

import os
import unittest

import torch

try:
    from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
        fused_linear_compact_state_replay_with_optional_track,
        fused_mamba_state_scatter_with_mask,
    )

    _FUSED_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    fused_linear_compact_state_replay_with_optional_track = None
    fused_mamba_state_scatter_with_mask = None
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


def _ref_update_like(
    ssm_states,
    intermediate_ssm,
    conv_states,
    intermediate_conv,
    *,
    state_indices_tensor,
    step_indices_raw,
    mamba_track_indices=None,
    mamba_steps_to_track=None,
):
    """Reference implementation using PyTorch advanced indexing for correctness verification."""
    total_requests = step_indices_raw.shape[0]
    intermediate_state_indices = torch.arange(
        total_requests, dtype=torch.int32, device=step_indices_raw.device
    )

    valid_mask = step_indices_raw >= 0
    dst_state_indices = state_indices_tensor[valid_mask].to(torch.int64)
    src_state_indices = intermediate_state_indices[valid_mask].to(torch.int64)
    last_steps = step_indices_raw[valid_mask].to(torch.int64)

    # Only scatter if there are valid indices (but don't early return -
    # mamba_track_indices processing is independent)
    if dst_state_indices.numel() > 0:
        _ref_scatter(
            ssm_states,
            intermediate_ssm,
            dst_state_indices,
            src_state_indices,
            last_steps,
        )
        _ref_scatter(
            conv_states,
            intermediate_conv,
            dst_state_indices,
            src_state_indices,
            last_steps,
        )

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


def _fused_update_like(
    ssm_states,
    intermediate_ssm,
    conv_states,
    intermediate_conv,
    *,
    state_indices_tensor,
    step_indices_raw,
    mamba_track_indices=None,
    mamba_steps_to_track=None,
):
    """Matches the fully fused logic that avoids index_select and nonzero calls."""
    # Use fully fused kernel that handles masking internally
    fused_mamba_state_scatter_with_mask(
        ssm_states,
        intermediate_ssm,
        state_indices_tensor,
        step_indices_raw,
    )
    fused_mamba_state_scatter_with_mask(
        conv_states,
        intermediate_conv,
        state_indices_tensor,
        step_indices_raw,
    )

    if mamba_track_indices is not None:
        assert mamba_steps_to_track is not None
        fused_mamba_state_scatter_with_mask(
            ssm_states,
            intermediate_ssm,
            mamba_track_indices,
            mamba_steps_to_track,
        )
        fused_mamba_state_scatter_with_mask(
            conv_states,
            intermediate_conv,
            mamba_track_indices,
            mamba_steps_to_track,
        )


def _ref_linear_compact_replay(
    dst,
    k_norm,
    delta_v,
    decay,
    *,
    base_indices_raw,
    accepted_dst_indices_raw,
    accepted_step_indices_raw,
    track_dst_indices_raw=None,
    track_step_indices_raw=None,
):
    total_requests = accepted_step_indices_raw.shape[0]
    src_req_size = k_norm.shape[1]
    src_step_size = k_norm.shape[2]
    dst_req_size = dst.shape[1]

    for req in range(total_requests):
        accepted_step = int(accepted_step_indices_raw[req].item())
        accepted_dst_idx = int(accepted_dst_indices_raw[req].item())
        if not (
            0 <= accepted_step < src_step_size
            and 0 <= accepted_dst_idx < dst_req_size
            and req < src_req_size
        ):
            continue

        base_idx = int(base_indices_raw[req].item())
        if not (0 <= base_idx < dst_req_size):
            continue

        h = dst[:, base_idx].float().clone()

        def replay_range(start: int, stop: int):
            nonlocal h
            for t in range(start, stop):
                k_t = k_norm[:, req, t].float()
                v_t = delta_v[:, req, t].float()
                decay_t = decay[:, req, t].float()
                h = h * decay_t[:, :, None, :] + v_t[:, :, :, None] * k_t[:, :, None, :]

        track_valid = False
        track_step = -1
        track_dst_idx = -1
        if track_dst_indices_raw is not None:
            assert track_step_indices_raw is not None
            track_step = int(track_step_indices_raw[req].item())
            track_dst_idx = int(track_dst_indices_raw[req].item())
            track_valid = (
                0 <= track_step <= accepted_step
                and track_step < src_step_size
                and 0 <= track_dst_idx < dst_req_size
            )

        if track_valid:
            replay_range(0, track_step + 1)
            dst[:, track_dst_idx] = h.to(dst.dtype)
            replay_range(track_step + 1, accepted_step + 1)
        else:
            replay_range(0, accepted_step + 1)

        dst[:, accepted_dst_idx] = h.to(dst.dtype)


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
        """Test that fused_mamba_state_scatter_with_mask matches the reference."""
        if fused_mamba_state_scatter_with_mask is None:
            self.skipTest(
                f"fused_mamba_state_scatter_with_mask import failed: {_FUSED_IMPORT_ERROR}"
            )

        torch.manual_seed(42)
        device = torch.device("cuda")

        # Keep sizes moderate so this test is quick.
        L = 8
        B = 32
        C = 49
        D = 5
        ssm_elems = 1024
        conv_elems = 512

        ssm_states0 = torch.randn(
            (L, C, ssm_elems), device=device, dtype=torch.bfloat16
        )
        conv_states0 = torch.randn(
            (L, C, conv_elems), device=device, dtype=torch.bfloat16
        )
        intermediate_ssm = torch.randn(
            (L, B, D, ssm_elems), device=device, dtype=torch.bfloat16
        )
        intermediate_conv = torch.randn(
            (L, B, D, conv_elems), device=device, dtype=torch.bfloat16
        )

        # unique cache lines (no duplicates) to avoid nondeterministic write order
        state_indices_tensor = torch.randperm(C, device=device, dtype=torch.int64)[
            :B
        ].to(torch.int32)

        step_indices_raw = torch.randint(0, D, (B,), device=device, dtype=torch.int64)
        # set ~10% invalid
        invalid = torch.rand((B,), device=device) < 0.1
        step_indices_raw[invalid] = -1

        # Optional track update
        mamba_track_indices = torch.randperm(C, device=device, dtype=torch.int64)[:B]
        mamba_steps_to_track = torch.randint(
            0, D, (B,), device=device, dtype=torch.int64
        )
        track_invalid = torch.rand((B,), device=device) < 0.7
        mamba_steps_to_track[track_invalid] = -1

        ssm_ref = ssm_states0.clone()
        conv_ref = conv_states0.clone()
        ssm_fused = ssm_states0.clone()
        conv_fused = conv_states0.clone()

        _ref_update_like(
            ssm_ref,
            intermediate_ssm,
            conv_ref,
            intermediate_conv,
            state_indices_tensor=state_indices_tensor,
            step_indices_raw=step_indices_raw,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
        )
        _fused_update_like(
            ssm_fused,
            intermediate_ssm,
            conv_fused,
            intermediate_conv,
            state_indices_tensor=state_indices_tensor,
            step_indices_raw=step_indices_raw,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
        )

        torch.testing.assert_close(ssm_fused, ssm_ref)
        torch.testing.assert_close(conv_fused, conv_ref)


class TestLinearCompactStateReplayCorrectness(CustomTestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is None,
        "CUDA without HIP is required for this test.",
    )
    def test_replay_without_track_matches_reference(self):
        if fused_linear_compact_state_replay_with_optional_track is None:
            self.skipTest(
                "fused_linear_compact_state_replay_with_optional_track import failed: "
                f"{_FUSED_IMPORT_ERROR}"
            )

        torch.manual_seed(1)
        device = torch.device("cuda")
        L, C, B, D, H, V, K = 2, 9, 4, 5, 2, 5, 3

        dst0 = torch.randn((L, C, H, V, K), device=device, dtype=torch.float32)
        k_norm = torch.randn((L, B, D, H, K), device=device, dtype=torch.float32)
        delta_v = torch.randn((L, B, D, H, V), device=device, dtype=torch.float32)
        decay = torch.rand((L, B, D, H, K), device=device, dtype=torch.float32) * 0.2
        decay += 0.8

        base_indices = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device)
        dst_indices = torch.tensor([5, 6, 7, 8], dtype=torch.int64, device=device)
        accepted_steps = torch.tensor([0, 3, -1, 2], dtype=torch.int64, device=device)

        ref = dst0.clone()
        actual = dst0.clone()
        _ref_linear_compact_replay(
            ref,
            k_norm,
            delta_v,
            decay,
            base_indices_raw=base_indices,
            accepted_dst_indices_raw=dst_indices,
            accepted_step_indices_raw=accepted_steps,
        )
        fused_linear_compact_state_replay_with_optional_track(
            actual,
            k_norm,
            delta_v,
            decay,
            base_indices,
            dst_indices,
            accepted_steps,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, ref, atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is None,
        "CUDA without HIP is required for this test.",
    )
    def test_replay_with_track_matches_reference(self):
        if fused_linear_compact_state_replay_with_optional_track is None:
            self.skipTest(
                "fused_linear_compact_state_replay_with_optional_track import failed: "
                f"{_FUSED_IMPORT_ERROR}"
            )

        torch.manual_seed(2)
        device = torch.device("cuda")
        L, C, B, D, H, V, K = 2, 12, 3, 5, 2, 5, 3

        dst0 = torch.randn((L, C, H, V, K), device=device, dtype=torch.float32)
        k_norm = torch.randn((L, B, D, H, K), device=device, dtype=torch.float32)
        delta_v = torch.randn((L, B, D, H, V), device=device, dtype=torch.float32)
        decay = torch.rand((L, B, D, H, K), device=device, dtype=torch.float32) * 0.2
        decay += 0.8

        # With track slots, replay first stores the state at track_steps for
        # prefix-cache tracking, then continues to accepted_steps for the final
        # committed request state. A negative track step means no track write.
        base_indices = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
        accepted_dst_indices = torch.tensor([4, 5, 6], dtype=torch.int32, device=device)
        accepted_steps = torch.tensor([3, 4, 2], dtype=torch.int64, device=device)
        track_dst_indices = torch.tensor([7, 8, 9], dtype=torch.int64, device=device)
        track_steps = torch.tensor([1, -1, 2], dtype=torch.int64, device=device)

        ref = dst0.clone()
        actual = dst0.clone()
        _ref_linear_compact_replay(
            ref,
            k_norm,
            delta_v,
            decay,
            base_indices_raw=base_indices,
            accepted_dst_indices_raw=accepted_dst_indices,
            accepted_step_indices_raw=accepted_steps,
            track_dst_indices_raw=track_dst_indices,
            track_step_indices_raw=track_steps,
        )
        fused_linear_compact_state_replay_with_optional_track(
            actual,
            k_norm,
            delta_v,
            decay,
            base_indices,
            accepted_dst_indices,
            accepted_steps,
            track_dst_indices,
            track_steps,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(actual, ref, atol=1e-5, rtol=1e-5)


class TestMambaStateScatterPerf(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_perf_report_old_vs_fused(self):
        """Optional microbenchmark comparing baseline vs fused kernel.

        Enable with: SGLANG_RUN_MAMBA_SCATTER_PERF_TEST=1
        """
        if os.environ.get("SGLANG_RUN_MAMBA_SCATTER_PERF_TEST", "0") != "1":
            self.skipTest("Set SGLANG_RUN_MAMBA_SCATTER_PERF_TEST=1 to run perf test.")
        if fused_mamba_state_scatter_with_mask is None:
            self.skipTest(
                f"fused_mamba_state_scatter_with_mask import failed: {_FUSED_IMPORT_ERROR}"
            )

        torch.manual_seed(0)
        device = torch.device("cuda")

        # Parameterize sizes via env vars so we can match a real model more closely.
        L = int(os.environ.get("SGLANG_MAMBA_SCATTER_LAYERS", "32"))
        B = int(os.environ.get("SGLANG_MAMBA_SCATTER_BATCH", "48"))
        C = int(os.environ.get("SGLANG_MAMBA_SCATTER_CACHE", "49"))
        D = int(os.environ.get("SGLANG_MAMBA_SCATTER_DRAFT_TOKENS", "5"))
        ssm_elems = int(os.environ.get("SGLANG_MAMBA_SCATTER_SSM_ELEMS", "4096"))
        conv_elems = int(os.environ.get("SGLANG_MAMBA_SCATTER_CONV_ELEMS", "512"))
        invalid_ratio = float(
            os.environ.get("SGLANG_MAMBA_SCATTER_INVALID_RATIO", "0.0")
        )
        track_ratio = float(os.environ.get("SGLANG_MAMBA_SCATTER_TRACK_RATIO", "0.0"))
        ssm_dtype = _dtype_from_str(
            os.environ.get("SGLANG_MAMBA_SCATTER_SSM_DTYPE", "bfloat16")
        )
        conv_dtype = _dtype_from_str(
            os.environ.get("SGLANG_MAMBA_SCATTER_CONV_DTYPE", "bfloat16")
        )

        # Use zeros for dst so each iteration overwrites the same memory.
        ssm_states = torch.zeros((L, C, ssm_elems), device=device, dtype=ssm_dtype)
        conv_states = torch.zeros((L, C, conv_elems), device=device, dtype=conv_dtype)
        intermediate_ssm = torch.randn(
            (L, B, D, ssm_elems), device=device, dtype=ssm_dtype
        )
        intermediate_conv = torch.randn(
            (L, B, D, conv_elems), device=device, dtype=conv_dtype
        )

        state_indices_tensor = torch.randperm(C, device=device, dtype=torch.int64)[
            :B
        ].to(torch.int32)
        step_indices_raw = torch.randint(0, D, (B,), device=device, dtype=torch.int64)
        if invalid_ratio > 0:
            invalid = torch.rand((B,), device=device) < invalid_ratio
            step_indices_raw[invalid] = -1

        mamba_track_indices = None
        mamba_steps_to_track = None
        if track_ratio > 0:
            mamba_track_indices = torch.randperm(C, device=device, dtype=torch.int64)[
                :B
            ]
            mamba_steps_to_track = torch.randint(
                0, D, (B,), device=device, dtype=torch.int64
            )
            track_invalid = torch.rand((B,), device=device) >= track_ratio
            mamba_steps_to_track[track_invalid] = -1

        def ref_fn():
            _ref_update_like(
                ssm_states,
                intermediate_ssm,
                conv_states,
                intermediate_conv,
                state_indices_tensor=state_indices_tensor,
                step_indices_raw=step_indices_raw,
                mamba_track_indices=mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
            )

        def fused_fn():
            _fused_update_like(
                ssm_states,
                intermediate_ssm,
                conv_states,
                intermediate_conv,
                state_indices_tensor=state_indices_tensor,
                step_indices_raw=step_indices_raw,
                mamba_track_indices=mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
            )

        # Warm up JIT compilation for triton kernels (and caches for torch indexing)
        ref_fn()
        fused_fn()
        torch.cuda.synchronize()

        ref_ms = _time_cuda_ms(ref_fn)
        fused_ms = _time_cuda_ms(fused_fn)

        num_valid = int((step_indices_raw >= 0).sum().item())
        ratio = fused_ms / ref_ms if ref_ms > 0 else float("inf")
        speedup = ref_ms / fused_ms if fused_ms > 0 else float("inf")

        # Print a concise report
        print(
            "\n[MambaStateScatterPerf]\n"
            f"  shapes: L={L} B={B} C={C} D={D} ssm_elems={ssm_elems} conv_elems={conv_elems}\n"
            f"  dtypes: ssm={ssm_dtype} conv={conv_dtype}\n"
            f"  valid: {num_valid}/{B}  invalid_ratio={invalid_ratio}  track_ratio={track_ratio}\n"
            f"  ref_total_ms (baseline):  {ref_ms:.4f}\n"
            f"  fused_total_ms:           {fused_ms:.4f}  (ratio={ratio:.3f}x, speedup={speedup:.2f}x)\n"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
