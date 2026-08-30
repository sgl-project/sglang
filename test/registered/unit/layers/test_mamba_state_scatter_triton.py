import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.layout.page_major import (
    build_page_major_mamba_views,
    mamba_entry_bytes,
)
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")
# The dst layout-contract tests run on CPU (no kernel launch).
register_cpu_ci(est_time=5, suite="base-a-test-cpu")

try:
    import sglang.kernels.ops.mamba.mamba_state_scatter_triton as mamba_state_scatter
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        _require_entry_contiguous_dst,
        copy_mamba_state_extend_rows,
        copy_mamba_state_rows,
        fused_conv_window_scatter_with_mask,
        fused_mamba_state_scatter_with_mask,
    )

    _FUSED_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    _require_entry_contiguous_dst = None
    mamba_state_scatter = None
    copy_mamba_state_extend_rows = None
    copy_mamba_state_rows = None
    fused_conv_window_scatter_with_mask = None
    fused_mamba_state_scatter_with_mask = None
    _FUSED_IMPORT_ERROR = e

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


def _ref_copy_rows(src, dst, src_indices, dst_indices):
    dst[dst_indices] = src[src_indices].to(dst.dtype, copy=False)


class TestCopyMambaStateRows(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_copy_api_is_available(self):
        self.assertIsNotNone(
            copy_mamba_state_rows,
            msg=f"copy_mamba_state_rows import failed: {_FUSED_IMPORT_ERROR}",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_h_to_state_bf16_fp32_and_dtype_conversion(self):
        if copy_mamba_state_rows is None:
            self.skipTest(f"copy_mamba_state_rows import failed: {_FUSED_IMPORT_ERROR}")

        device = torch.device("cuda")
        torch.manual_seed(123)
        src_indices = torch.tensor([1, 5, 7, 9], device=device, dtype=torch.int32)
        dst_indices = torch.tensor([0, 3, 6, 8], device=device, dtype=torch.int32)

        for src_dtype, dst_dtype in [
            (torch.bfloat16, torch.bfloat16),
            (torch.float32, torch.float32),
            (torch.float32, torch.bfloat16),
            (torch.bfloat16, torch.float32),
        ]:
            src = torch.randn((16, 128), device=device, dtype=src_dtype)
            dst = torch.randn((16, 128), device=device, dtype=dst_dtype)
            ref = dst.clone()

            _ref_copy_rows(src, ref, src_indices, dst_indices)
            copy_mamba_state_rows(src, dst, src_indices, dst_indices)
            torch.testing.assert_close(dst, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_nontrivial_row_stride_view_and_empty_indices(self):
        if copy_mamba_state_rows is None:
            self.skipTest(f"copy_mamba_state_rows import failed: {_FUSED_IMPORT_ERROR}")

        device = torch.device("cuda")
        torch.manual_seed(0)

        src_base = torch.randn((48, 300), device=device, dtype=torch.float32)
        dst_base = torch.randn((64, 320), device=device, dtype=torch.bfloat16)

        src = src_base.as_strided(
            (20, 128), (src_base.stride(0) * 2, 1), storage_offset=17
        )
        dst = dst_base.as_strided(
            (24, 128), (dst_base.stride(0) * 2, 1), storage_offset=11
        )
        ref = dst.clone()

        src_indices = torch.tensor([0, 4, 7, 13, 19], device=device, dtype=torch.int32)
        dst_indices = torch.tensor([2, 5, 9, 15, 23], device=device, dtype=torch.int32)
        _ref_copy_rows(src, ref, src_indices, dst_indices)
        copy_mamba_state_rows(src, dst, src_indices, dst_indices)
        torch.testing.assert_close(dst, ref)

        empty = torch.empty((0,), device=device, dtype=torch.int32)
        dst_before = dst.clone()
        copy_mamba_state_rows(src, dst, empty, empty)
        torch.testing.assert_close(dst, dst_before)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_state_to_state_disjoint_copy(self):
        if copy_mamba_state_rows is None:
            self.skipTest(f"copy_mamba_state_rows import failed: {_FUSED_IMPORT_ERROR}")

        device = torch.device("cuda")
        torch.manual_seed(7)
        state = torch.randn((12, 96), device=device, dtype=torch.bfloat16)
        ref = state.clone()

        src_indices = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
        dst_indices = torch.tensor([4, 5, 6, 7], device=device, dtype=torch.int32)

        _ref_copy_rows(ref, ref, src_indices, dst_indices)
        copy_mamba_state_rows(state, state, src_indices, dst_indices)
        torch.testing.assert_close(state, ref)


class TestCopyMambaStateExtendRows(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_non_hip_platform_uses_advanced_index_fallback(self):
        if copy_mamba_state_extend_rows is None or mamba_state_scatter is None:
            self.skipTest(
                f"copy_mamba_state_extend_rows import failed: {_FUSED_IMPORT_ERROR}"
            )

        device = torch.device("cuda")
        h = torch.randn((8, 2, 128, 128), device=device, dtype=torch.float32)
        ssm_states = torch.randn((16, 2, 128, 128), device=device, dtype=torch.bfloat16)
        h_src = torch.tensor([0, 2], device=device, dtype=torch.int32)
        h_dst = torch.tensor([8, 10], device=device, dtype=torch.int32)
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        ref = ssm_states.clone()
        _ref_copy_rows(h, ref, h_src, h_dst)

        with (
            mock.patch.object(mamba_state_scatter, "_is_hip", False),
            mock.patch.object(
                mamba_state_scatter,
                "copy_mamba_state_rows",
                side_effect=AssertionError("direct helper should not run"),
            ),
        ):
            copy_mamba_state_extend_rows(
                h,
                ssm_states,
                h_src,
                h_dst,
                empty,
                empty,
                h_indices_trusted=True,
                final_indices_trusted=True,
                final_state_disjoint=True,
            )
        torch.testing.assert_close(ssm_states, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_unvalidated_dtype_uses_advanced_index_fallback(self):
        if copy_mamba_state_extend_rows is None or mamba_state_scatter is None:
            self.skipTest(
                f"copy_mamba_state_extend_rows import failed: {_FUSED_IMPORT_ERROR}"
            )

        device = torch.device("cuda")
        h = torch.randn((8, 2, 128, 128), device=device, dtype=torch.float16)
        ssm_states = torch.randn((16, 2, 128, 128), device=device, dtype=torch.float16)
        h_src = torch.tensor([0, 2], device=device, dtype=torch.int32)
        h_dst = torch.tensor([8, 10], device=device, dtype=torch.int32)
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        ref = ssm_states.clone()
        _ref_copy_rows(h, ref, h_src, h_dst)

        with mock.patch.object(
            mamba_state_scatter,
            "copy_mamba_state_rows",
            side_effect=AssertionError("direct helper should not run"),
        ):
            copy_mamba_state_extend_rows(
                h,
                ssm_states,
                h_src,
                h_dst,
                empty,
                empty,
                h_indices_trusted=True,
                final_indices_trusted=True,
                final_state_disjoint=True,
            )
        torch.testing.assert_close(ssm_states, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_h_to_state_direct_canonical_shape(self):
        if copy_mamba_state_extend_rows is None:
            self.skipTest(
                f"copy_mamba_state_extend_rows import failed: {_FUSED_IMPORT_ERROR}"
            )

        device = torch.device("cuda")
        torch.manual_seed(13)
        hv, v_dim, k_dim = 4, 128, 128
        h = torch.randn((1, 32, hv, v_dim, k_dim), device=device, dtype=torch.float32)
        ssm_states = torch.randn(
            (64, hv, v_dim, k_dim), device=device, dtype=torch.bfloat16
        )
        h2 = h.squeeze(0)

        h_src = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
        h_dst = torch.tensor([2, 9, 15, 21], device=device, dtype=torch.int32)
        final_src = torch.empty((0,), device=device, dtype=torch.int32)
        final_dst = torch.empty((0,), device=device, dtype=torch.int32)
        ref = ssm_states.clone()

        _ref_copy_rows(h2, ref, h_src, h_dst)
        copy_mamba_state_extend_rows(
            h2,
            ssm_states,
            h_src,
            h_dst,
            final_src,
            final_dst,
            h_indices_trusted=True,
            final_indices_trusted=True,
            final_state_disjoint=True,
        )
        torch.testing.assert_close(ssm_states, ref)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is not None,
        "ROCm GPU is required for direct large-shape coverage.",
    )
    def test_h_to_state_direct_qwen_shapes(self):
        if copy_mamba_state_extend_rows is None or mamba_state_scatter is None:
            self.skipTest(
                f"copy_mamba_state_extend_rows import failed: {_FUSED_IMPORT_ERROR}"
            )

        device = torch.device("cuda")
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        torch.manual_seed(91)

        for hv in (8, 16, 24, 48):
            for tracked_rows in (1, 64, 256):
                with self.subTest(hv=hv, tracked_rows=tracked_rows):
                    h = torch.empty(
                        (tracked_rows, hv, 128, 128),
                        device=device,
                        dtype=torch.bfloat16,
                    ).uniform_(-1, 1)
                    ssm_states = torch.zeros_like(h)
                    indices = torch.arange(
                        tracked_rows, device=device, dtype=torch.int32
                    )

                    with mock.patch.object(
                        mamba_state_scatter,
                        "_copy_rows_advanced_index",
                        side_effect=AssertionError(
                            "large trusted HIP shapes must use direct copy"
                        ),
                    ):
                        copy_mamba_state_extend_rows(
                            h,
                            ssm_states,
                            indices,
                            indices,
                            empty,
                            empty,
                            h_indices_trusted=True,
                            final_indices_trusted=True,
                            final_state_disjoint=True,
                        )

                    self.assertTrue(torch.equal(ssm_states, h))
                    del h, ssm_states, indices

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is None,
        "A CUDA GPU is required for the real-platform fallback test.",
    )
    def test_real_cuda_platform_uses_advanced_index_fallback(self):
        if copy_mamba_state_extend_rows is None or mamba_state_scatter is None:
            self.skipTest(
                f"copy_mamba_state_extend_rows import failed: {_FUSED_IMPORT_ERROR}"
            )

        device = torch.device("cuda")
        h = torch.randn((4, 8, 128, 128), device=device, dtype=torch.bfloat16)
        ssm_states = torch.zeros_like(h)
        indices = torch.arange(4, device=device, dtype=torch.int32)
        empty = torch.empty((0,), device=device, dtype=torch.int32)

        with mock.patch.object(
            mamba_state_scatter,
            "copy_mamba_state_rows",
            side_effect=AssertionError("CUDA must not launch the HIP direct helper"),
        ):
            copy_mamba_state_extend_rows(
                h,
                ssm_states,
                indices,
                indices,
                empty,
                empty,
                h_indices_trusted=True,
                final_indices_trusted=True,
                final_state_disjoint=True,
            )

        self.assertTrue(torch.equal(ssm_states, h))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_state_to_state_disjoint_uses_direct_path(self):
        if copy_mamba_state_extend_rows is None or mamba_state_scatter is None:
            self.skipTest(
                f"copy_mamba_state_extend_rows import failed: {_FUSED_IMPORT_ERROR}"
            )

        device = torch.device("cuda")
        torch.manual_seed(31)
        hv, v_dim, k_dim = 2, 128, 128
        ssm_states = torch.randn(
            (32, hv, v_dim, k_dim), device=device, dtype=torch.bfloat16
        )
        h = torch.randn((16, hv, v_dim, k_dim), device=device, dtype=torch.float32)
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        final_src = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
        final_dst = torch.tensor([8, 9, 10, 11], device=device, dtype=torch.int32)
        ref = ssm_states.clone()
        _ref_copy_rows(ref, ref, final_src, final_dst)

        with mock.patch.object(
            mamba_state_scatter,
            "_copy_rows_advanced_index",
            side_effect=AssertionError("advanced-index fallback should not run"),
        ):
            copy_mamba_state_extend_rows(
                h,
                ssm_states,
                empty,
                empty,
                final_src,
                final_dst,
                h_indices_trusted=True,
                final_indices_trusted=True,
                final_state_disjoint=True,
            )
        torch.testing.assert_close(ssm_states, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_state_to_state_overlap_uses_fallback(self):
        if copy_mamba_state_extend_rows is None:
            self.skipTest(
                f"copy_mamba_state_extend_rows import failed: {_FUSED_IMPORT_ERROR}"
            )

        device = torch.device("cuda")
        torch.manual_seed(52)
        hv, v_dim, k_dim = 2, 128, 128
        ssm_states = torch.randn(
            (24, hv, v_dim, k_dim), device=device, dtype=torch.bfloat16
        )
        h = torch.randn((12, hv, v_dim, k_dim), device=device, dtype=torch.float32)
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        final_src = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
        final_dst = torch.tensor([1, 2, 3, 4], device=device, dtype=torch.int32)
        ref = ssm_states.clone()
        _ref_copy_rows(ref, ref, final_src, final_dst)

        copy_mamba_state_extend_rows(
            h,
            ssm_states,
            empty,
            empty,
            final_src,
            final_dst,
            h_indices_trusted=True,
            final_indices_trusted=True,
            final_state_disjoint=False,
        )
        torch.testing.assert_close(ssm_states, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_page_like_row_stride(self):
        if copy_mamba_state_extend_rows is None:
            self.skipTest(
                f"copy_mamba_state_extend_rows import failed: {_FUSED_IMPORT_ERROR}"
            )

        device = torch.device("cuda")
        torch.manual_seed(77)
        hv, v_dim, k_dim = 2, 128, 128
        row_elems = hv * v_dim * k_dim
        row_stride = row_elems + 37
        storage = torch.randn((64 * row_stride,), device=device, dtype=torch.bfloat16)
        ssm_states = storage.as_strided(
            (64, hv, v_dim, k_dim),
            (row_stride, v_dim * k_dim, k_dim, 1),
        )
        h = torch.randn((32, hv, v_dim, k_dim), device=device, dtype=torch.float32)
        h_src = torch.tensor([0, 3, 6, 9], device=device, dtype=torch.int32)
        h_dst = torch.tensor([7, 11, 14, 20], device=device, dtype=torch.int32)
        final_src = torch.tensor([2, 4, 5], device=device, dtype=torch.int32)
        final_dst = torch.tensor([22, 23, 24], device=device, dtype=torch.int32)
        ref = ssm_states.clone()

        _ref_copy_rows(h, ref, h_src, h_dst)
        _ref_copy_rows(ref, ref, final_src, final_dst)
        with mock.patch.object(
            mamba_state_scatter,
            "_copy_rows_advanced_index",
            side_effect=AssertionError("fallback should not run"),
        ):
            copy_mamba_state_extend_rows(
                h,
                ssm_states,
                h_src,
                h_dst,
                final_src,
                final_dst,
                h_indices_trusted=True,
                final_indices_trusted=True,
                final_state_disjoint=True,
            )
        torch.testing.assert_close(ssm_states, ref)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for this test.")
    def test_untrusted_indices_route_to_pytorch_fallback(self):
        if copy_mamba_state_extend_rows is None:
            self.skipTest(
                f"copy_mamba_state_extend_rows import failed: {_FUSED_IMPORT_ERROR}"
            )

        device = torch.device("cuda")
        hv, v_dim, k_dim = 2, 128, 128
        h = torch.randn((8, hv, v_dim, k_dim), device=device, dtype=torch.float32)
        ssm_states = torch.randn(
            (8, hv, v_dim, k_dim), device=device, dtype=torch.bfloat16
        )
        ref = ssm_states.clone()
        h_src = torch.tensor([0, -1], device=device, dtype=torch.int32)
        h_dst = torch.tensor([1, 6], device=device, dtype=torch.int32)
        empty = torch.empty((0,), device=device, dtype=torch.int32)
        _ref_copy_rows(h, ref, h_src, h_dst)

        with mock.patch.object(
            mamba_state_scatter,
            "copy_mamba_state_rows",
            side_effect=AssertionError("direct helper should not run"),
        ):
            copy_mamba_state_extend_rows(
                h,
                ssm_states,
                h_src,
                h_dst,
                empty,
                empty,
                h_indices_trusted=False,
                final_indices_trusted=False,
                final_state_disjoint=False,
            )
        torch.testing.assert_close(ssm_states, ref)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
