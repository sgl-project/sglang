from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)

register_cuda_ci(est_time=7, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=7, suite="stage-b-test-1-gpu-small-amd-mi35x")
# The dst layout-contract tests run on CPU (no kernel launch).
register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

import torch

try:
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        _require_entry_contiguous_dst,
        fused_conv_window_scatter_with_mask,
        fused_mamba_state_scatter_with_mask,
    )

    _FUSED_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    _require_entry_contiguous_dst = None
    fused_conv_window_scatter_with_mask = None
    fused_mamba_state_scatter_with_mask = None
    _FUSED_IMPORT_ERROR = e

from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mamba_views,
    mamba_entry_bytes,
)


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


def _make_envelope_views(device="cpu"):
    """Envelope-strided conv/temporal views, exactly as UnifiedMambaPool /
    the page-major MambaPool serve them ((num_layers, max_slots, *inner) with
    slot stride = the multi-layer entry envelope). Mirrors
    test_flashkda_strided_state_access.py's setup."""
    layers, slots = 2, 16
    temporal_shape = (2, 4, 4)  # (H, V, K)
    conv_shapes = ((8, 3),)  # (dim, K-1) as fused_conv_window_scatter expects
    conv_dtype = torch.bfloat16
    temporal_dtype = torch.float32
    entry = mamba_entry_bytes(
        layer_num=layers,
        conv_state_shapes=conv_shapes,
        conv_dtype=conv_dtype,
        temporal_state_shape=temporal_shape,
        temporal_dtype=temporal_dtype,
    )
    raw = torch.zeros(slots * entry, dtype=torch.uint8, device=device)
    conv_views, temporal = build_page_major_mamba_views(
        raw,
        layer_num=layers,
        conv_state_shapes=conv_shapes,
        conv_dtype=conv_dtype,
        temporal_state_shape=temporal_shape,
        temporal_dtype=temporal_dtype,
        max_slots=slots,
    )
    return conv_views, temporal


class TestScatterDstLayoutContract(unittest.TestCase):
    """The scatter wrappers' dst contract (CPU, no kernel launch).

    Derived property: the Triton kernels index dst through its REAL
    ``stride(0)``/``stride(1)`` plus a FLAT in-entry element offset, so the
    layout contract is "arbitrary layer/slot strides, contiguous trailing
    entry dims" — NOT ``dst.is_contiguous()``. The blanket contiguity assert
    the wrappers used to carry rejected the unified pool's envelope-strided
    views (DSPARK verify commit under --enable-unified-memory); the relaxed
    check must keep accepting them while still rejecting a dst whose entry
    dims the kernels would mis-address."""

    def setUp(self):
        if _require_entry_contiguous_dst is None:
            self.skipTest(f"import failed: {_FUSED_IMPORT_ERROR}")

    def test_envelope_strided_views_accepted(self):
        conv_views, temporal = _make_envelope_views()
        # Precondition: the views really are envelope-strided (else the
        # property below is vacuous).
        self.assertFalse(temporal.is_contiguous())
        self.assertFalse(conv_views[0].is_contiguous())
        # dst = temporal (5-D) for the dense scatter, conv (4-D) for the
        # conv-window scatter; entry dims start at 2 for both.
        _require_entry_contiguous_dst(temporal, 2, "test")
        _require_entry_contiguous_dst(conv_views[0], 2, "test")

    def test_entry_noncontiguous_dst_rejected(self):
        # A dst whose ENTRY dims are strided (inner transpose) would be
        # mis-addressed by the flat in-entry offset; the check must not have
        # degraded to always-pass.
        dst = torch.zeros(2, 4, 8, 3).transpose(-1, -2)  # entry dims strided
        with self.assertRaises(ValueError):
            _require_entry_contiguous_dst(dst, 2, "test")


class TestMambaStateScatterEnvelopeDst(unittest.TestCase):
    """End-to-end: both scatter wrappers accept the unified pool's
    envelope-strided dst views and address slots through the real strides
    (bug regression: the wrappers used to raise 'dst tensor must be
    contiguous' on these views)."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_fused_scatter_envelope_strided_dst(self):
        if fused_mamba_state_scatter_with_mask is None:
            self.skipTest(f"import failed: {_FUSED_IMPORT_ERROR}")

        torch.manual_seed(7)
        device = torch.device("cuda")
        conv_views, temporal = _make_envelope_views(device=device)
        layers, slots = temporal.shape[0], temporal.shape[1]
        temporal_shape = tuple(temporal.shape[2:])  # (H, V, K)
        dim, km1 = conv_views[0].shape[2], conv_views[0].shape[3]

        B, D = 5, 3
        temporal[:] = torch.randn_like(temporal)
        conv_views[0][:] = torch.randn_like(conv_views[0])
        temporal_before = temporal.clone()
        conv_before = conv_views[0].clone()

        # Dense SSM scatter: contiguous per-step src (the intermediate cache).
        src_ssm = torch.randn(
            (layers, B, D) + temporal_shape, device=device, dtype=temporal.dtype
        )
        # Conv-window scatter: overlapping as_strided src over a shared
        # [dim, D+K-2] buffer per (layer, slot) — window t = shared[:, t:t+K-1].
        shared = torch.randn(
            (layers, B, dim, D + km1 - 1), device=device, dtype=conv_views[0].dtype
        )
        src_conv = shared.as_strided(
            (layers, B, D, dim, km1),
            (
                shared.stride(0),
                shared.stride(1),
                1,  # step: window slides by one position
                shared.stride(2),
                1,  # within-window
            ),
        )

        dst_indices = torch.randperm(slots, device=device, dtype=torch.int64)[:B].to(
            torch.int32
        )
        step_indices = torch.randint(0, D, (B,), device=device, dtype=torch.int64)
        step_indices[0] = -1  # one rejected row must be skipped

        fused_mamba_state_scatter_with_mask(
            temporal, src_ssm, dst_indices, step_indices
        )
        fused_conv_window_scatter_with_mask(
            conv_views[0], src_conv, dst_indices, step_indices
        )

        # Reference via advanced indexing (layout-agnostic).
        valid = step_indices >= 0
        d = dst_indices[valid].long()
        s = torch.arange(B, device=device)[valid]
        t = step_indices[valid]
        expect_temporal = temporal_before.clone()
        expect_temporal[:, d] = src_ssm[:, s, t]
        expect_conv = conv_before.clone()
        expect_conv[:, d] = src_conv[:, s, t]

        torch.testing.assert_close(temporal, expect_temporal)
        torch.testing.assert_close(conv_views[0], expect_conv)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
