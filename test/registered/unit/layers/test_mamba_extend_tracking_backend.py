import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=8, suite="stage-b-test-1-gpu-small-amd-mi35x")

try:
    import sglang.srt.layers.attention.hybrid_linear_attn_backend as hybrid_backend
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        MambaAttnBackendBase,
    )
    from sglang.srt.layers.attention.mamba.mamba2_metadata import (
        ForwardMetadata,
        Mamba2Metadata,
    )

    _BACKEND_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover
    MambaAttnBackendBase = None
    hybrid_backend = None
    ForwardMetadata = None
    Mamba2Metadata = None
    _BACKEND_IMPORT_ERROR = e


class _FakeReqToTokenPool:
    def __init__(self, *, device: torch.device, num_layers: int, pool_rows: int):
        self._device = device
        self._pool_rows = pool_rows
        self.mamba_pool = SimpleNamespace(
            mamba_cache=SimpleNamespace(
                temporal=torch.empty(
                    (num_layers, pool_rows, 2, 128, 128), device=device
                )
            )
        )

    def get_mamba_indices(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        return req_pool_indices.to(device=self._device, dtype=torch.int64)

    def translate_mamba_indices(self, mamba_indices: torch.Tensor) -> torch.Tensor:
        return mamba_indices


def _build_backend(
    device: torch.device, pool_rows: int, *, num_layers: int = 4
) -> MambaAttnBackendBase:
    backend = MambaAttnBackendBase.__new__(MambaAttnBackendBase)
    backend.device = device
    backend.topk = 0
    backend.enable_unified_memory = False
    backend.req_to_token_pool = _FakeReqToTokenPool(
        device=device, num_layers=num_layers, pool_rows=pool_rows
    )
    backend.conv_states_shape = (num_layers, pool_rows, 32)
    return backend


class TestMambaExtendTrackingBackend(unittest.TestCase):
    def setUp(self):
        if MambaAttnBackendBase is None:
            self.skipTest(f"backend import failed: {_BACKEND_IMPORT_ERROR}")

    def test_init_track_ssm_indices_mixed_alignment_and_trust(self):
        device = torch.device("cpu")
        backend = _build_backend(device, pool_rows=64, num_layers=4)
        chunk = 16

        mamba_cache_indices = torch.tensor([12, 20], device=device, dtype=torch.int64)
        forward_batch = SimpleNamespace(
            mamba_track_mask=torch.tensor([True, True], device=device),
            extend_seq_lens=torch.tensor(
                [chunk + 1, chunk], device=device, dtype=torch.int64
            ),
            mamba_track_indices=torch.tensor(
                [33, 40], device=device, dtype=torch.int64
            ),
            mamba_track_seqlens=torch.tensor(
                [chunk + 1, chunk], device=device, dtype=torch.int64
            ),
            extend_prefix_lens=torch.tensor([0, 0], device=device, dtype=torch.int64),
        )

        with mock.patch.object(
            hybrid_backend,
            "get_server_args",
            return_value=SimpleNamespace(mamba_cache_chunk_size=chunk),
        ):
            (
                h_src,
                h_dst,
                final_src,
                final_dst,
                h_trusted,
                final_trusted,
                final_disjoint,
            ) = backend._init_track_ssm_indices(mamba_cache_indices, forward_batch)

        self.assertEqual(h_src.numel(), 1)
        self.assertEqual(final_src.numel(), 1)
        self.assertTrue(h_trusted)
        self.assertTrue(final_trusted)
        self.assertTrue(final_disjoint)
        self.assertTrue(torch.equal(h_dst.cpu(), torch.tensor([33], dtype=torch.int64)))
        self.assertTrue(
            torch.equal(final_dst.cpu(), torch.tensor([40], dtype=torch.int64))
        )

    def test_init_track_ssm_indices_invalid_routes_untrusted(self):
        device = torch.device("cpu")
        backend = _build_backend(device, pool_rows=32, num_layers=4)
        chunk = 16

        mamba_cache_indices = torch.tensor([12, 14], device=device, dtype=torch.int64)
        forward_batch = SimpleNamespace(
            mamba_track_mask=torch.tensor([True, True], device=device),
            extend_seq_lens=torch.tensor(
                [chunk + 1, chunk], device=device, dtype=torch.int64
            ),
            mamba_track_indices=torch.tensor(
                [28, 100], device=device, dtype=torch.int64
            ),
            mamba_track_seqlens=torch.tensor(
                [chunk + 1, chunk], device=device, dtype=torch.int64
            ),
            extend_prefix_lens=torch.tensor([0, 0], device=device, dtype=torch.int64),
        )

        with mock.patch.object(
            hybrid_backend,
            "get_server_args",
            return_value=SimpleNamespace(mamba_cache_chunk_size=chunk),
        ):
            (
                _h_src,
                _h_dst,
                _final_src,
                _final_dst,
                h_trusted,
                final_trusted,
                _final_disjoint,
            ) = backend._init_track_ssm_indices(mamba_cache_indices, forward_batch)

        self.assertTrue(h_trusted)
        self.assertFalse(final_trusted)

    def test_track_mamba_state_extend_mixed_aligned_unaligned(self):
        device = torch.device("cpu")
        backend = _build_backend(device, pool_rows=64, num_layers=4)

        h = torch.randn((1, 6, 2, 128, 128), device=device, dtype=torch.float32)
        ssm_states = torch.randn((64, 2, 128, 128), device=device, dtype=torch.bfloat16)
        ref = ssm_states.clone()

        h_src = torch.tensor([1, 4], device=device, dtype=torch.int32)
        h_dst = torch.tensor([31, 42], device=device, dtype=torch.int32)
        final_src = torch.tensor([33, 36], device=device, dtype=torch.int32)
        final_dst = torch.tensor([50, 57], device=device, dtype=torch.int32)
        ref[h_dst] = h.squeeze(0)[h_src].to(ref.dtype, copy=False)
        ref[final_dst] = ref[final_src]

        metadata = ForwardMetadata(
            query_start_loc=torch.zeros((1,), device=device, dtype=torch.int32),
            mamba_cache_indices=torch.zeros((1,), device=device, dtype=torch.int32),
            has_mamba_track_mask=True,
            track_ssm_h_src=h_src,
            track_ssm_h_dst=h_dst,
            track_ssm_final_src=final_src,
            track_ssm_final_dst=final_dst,
            track_ssm_h_trusted=True,
            track_ssm_final_trusted=True,
            track_ssm_final_disjoint=True,
        )
        forward_batch = SimpleNamespace()

        def _ref_extend_copy(
            h_local,
            ssm_local,
            h_src_local,
            h_dst_local,
            final_src_local,
            final_dst_local,
            *,
            h_indices_trusted,
            final_indices_trusted,
            final_state_disjoint,
        ):
            self.assertTrue(h_indices_trusted)
            self.assertTrue(final_indices_trusted)
            self.assertTrue(final_state_disjoint)
            ssm_local[h_dst_local] = h_local[h_src_local].to(
                ssm_local.dtype, copy=False
            )
            ssm_local[final_dst_local] = ssm_local[final_src_local]

        with mock.patch.object(
            hybrid_backend, "copy_mamba_state_extend_rows", side_effect=_ref_extend_copy
        ) as patched:
            backend._track_mamba_state_extend(forward_batch, h, ssm_states, metadata)
            self.assertEqual(patched.call_count, 1)
        self.assertTrue(torch.equal(ssm_states, ref))

    def test_track_mamba_state_extend_propagates_overlap_fallback(self):
        device = torch.device("cpu")
        backend = _build_backend(device, pool_rows=32, num_layers=4)

        h = None
        ssm_states = torch.randn((16, 2, 128, 128), device=device, dtype=torch.bfloat16)
        ref = ssm_states.clone()

        final_src = torch.tensor([12, 13, 14], device=device, dtype=torch.int32)
        final_dst = torch.tensor([13, 14, 15], device=device, dtype=torch.int32)
        ref[final_dst] = ref[final_src]

        metadata = ForwardMetadata(
            query_start_loc=torch.zeros((1,), device=device, dtype=torch.int32),
            mamba_cache_indices=torch.zeros((1,), device=device, dtype=torch.int32),
            has_mamba_track_mask=True,
            track_ssm_h_src=torch.empty((0,), device=device, dtype=torch.int32),
            track_ssm_h_dst=torch.empty((0,), device=device, dtype=torch.int32),
            track_ssm_final_src=final_src,
            track_ssm_final_dst=final_dst,
            track_ssm_h_trusted=True,
            track_ssm_final_trusted=True,
            track_ssm_final_disjoint=False,
        )
        forward_batch = SimpleNamespace()

        def _ref_extend_copy(
            h_local,
            ssm_local,
            h_src_local,
            h_dst_local,
            final_src_local,
            final_dst_local,
            *,
            h_indices_trusted,
            final_indices_trusted,
            final_state_disjoint,
        ):
            self.assertIsNone(h_local)
            self.assertTrue(h_indices_trusted)
            self.assertTrue(final_indices_trusted)
            self.assertFalse(final_state_disjoint)
            ssm_local[final_dst_local] = ssm_local[final_src_local]

        with mock.patch.object(
            hybrid_backend, "copy_mamba_state_extend_rows", side_effect=_ref_extend_copy
        ) as patched:
            backend._track_mamba_state_extend(forward_batch, h, ssm_states, metadata)
            self.assertEqual(patched.call_count, 1)
        self.assertTrue(torch.equal(ssm_states, ref))

    def test_extend_metadata_chain_propagates_flags(self):
        device = torch.device("cpu")
        backend = _build_backend(device, pool_rows=96, num_layers=6)
        chunk = 16

        class _FakeMode:
            def is_decode_or_idle(self):
                return False

            def is_extend(self, include_draft_extend_v2=True):
                return True

            def is_draft_extend_v2(self):
                return False

            def is_target_verify(self):
                return False

        forward_batch = SimpleNamespace(
            batch_size=2,
            req_pool_indices=torch.tensor([12, 20], device=device, dtype=torch.int64),
            mamba_track_indices=torch.tensor(
                [41, 63], device=device, dtype=torch.int64
            ),
            _original_batch_size=None,
            forward_mode=_FakeMode(),
            extend_start_loc=torch.tensor(
                [0, chunk + 1], device=device, dtype=torch.int32
            ),
            extend_seq_lens=torch.tensor(
                [chunk + 1, chunk], device=device, dtype=torch.int64
            ),
            extend_seq_lens_cpu=[chunk + 1, chunk],
            mamba_track_mask=torch.tensor([True, True], device=device),
            mamba_track_seqlens=torch.tensor(
                [chunk + 1, chunk], device=device, dtype=torch.int64
            ),
            extend_prefix_lens=torch.tensor([0, 0], device=device, dtype=torch.int64),
            extend_num_tokens=2 * chunk + 1,
            seq_lens=torch.tensor([chunk + 1, chunk], device=device, dtype=torch.int32),
            spec_info=None,
        )
        h = torch.randn((1, 6, 2, 128, 128), device=device, dtype=torch.float32)
        ssm_states = torch.randn((96, 2, 128, 128), device=device, dtype=torch.bfloat16)
        ref = ssm_states.clone()

        with mock.patch.object(
            hybrid_backend,
            "get_server_args",
            return_value=SimpleNamespace(mamba_cache_chunk_size=chunk),
        ):
            forward_metadata = backend._forward_metadata(forward_batch)
        mamba2_meta = Mamba2Metadata.prepare_mixed(
            forward_metadata, chunk_size=chunk, forward_batch=forward_batch
        )

        self.assertTrue(mamba2_meta.track_ssm_h_trusted)
        self.assertTrue(mamba2_meta.track_ssm_final_trusted)
        self.assertTrue(mamba2_meta.track_ssm_final_disjoint)
        self.assertGreater(int(mamba2_meta.track_ssm_h_dst.min().item()), 6)
        self.assertGreater(int(mamba2_meta.track_ssm_final_dst.min().item()), 6)

        ref[mamba2_meta.track_ssm_h_dst] = h.squeeze(0)[mamba2_meta.track_ssm_h_src].to(
            ref.dtype, copy=False
        )
        ref[mamba2_meta.track_ssm_final_dst] = ref[mamba2_meta.track_ssm_final_src]

        def _ref_extend_copy(
            h_local,
            ssm_local,
            h_src_local,
            h_dst_local,
            final_src_local,
            final_dst_local,
            *,
            h_indices_trusted,
            final_indices_trusted,
            final_state_disjoint,
        ):
            self.assertTrue(h_indices_trusted)
            self.assertTrue(final_indices_trusted)
            self.assertTrue(final_state_disjoint)
            ssm_local[h_dst_local] = h_local[h_src_local].to(
                ssm_local.dtype, copy=False
            )
            ssm_local[final_dst_local] = ssm_local[final_src_local]

        with mock.patch.object(
            hybrid_backend, "copy_mamba_state_extend_rows", side_effect=_ref_extend_copy
        ) as patched:
            backend._track_mamba_state_extend(forward_batch, h, ssm_states, mamba2_meta)
            self.assertEqual(patched.call_count, 1)
        self.assertTrue(torch.equal(ssm_states, ref))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
