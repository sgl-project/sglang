"""Row-alignment guard for the mamba track-destination plumbing.

The decode track-save copies conv/ssm state from ``mamba_cache_indices[i]`` to
``mamba_track_indices[i]`` for every row the track mask selects, so it is only
correct while all three per-row tensors agree on what row ``i`` means. Under
cuda graph ``_replay_metadata`` used to hand out the whole ``(max_bs,)`` static
track-dest buffer instead of a batch-length view; combined with the tail-relative
``[-num_decodes:]`` slice at the call site, every tracked row then checkpointed
into whatever slot sat in the untouched buffer tail (zeros -> virtual slot 0),
silently clobbering another request's mamba state.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase

MAX_BS = 16
DRAFT_TOKEN_NUM = 2


class _FakeReqToTokenPool:
    """Identity v2p translate; mamba slot id == req pool index."""

    def __init__(self):
        self.mamba_pool = None

    def get_mamba_indices(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        return req_pool_indices.to(torch.int32).clone()

    def translate_mamba_indices(self, mamba_indices: torch.Tensor) -> torch.Tensor:
        return mamba_indices


def _make_backend() -> MambaAttnBackendBase:
    """A MambaAttnBackendBase carrying only the state _replay_metadata reads,
    laid out exactly as init_cuda_graph_state(max_bs) would."""
    backend = object.__new__(MambaAttnBackendBase)
    backend.pad_slot_id = -1
    backend.device = torch.device("cpu")
    backend.topk = 1
    backend.req_to_token_pool = _FakeReqToTokenPool()
    backend.replayssm_write_pos_list = None
    backend.replayssm_force_flush_list = None
    backend.mamba_track_indices_buf = torch.zeros((MAX_BS,), dtype=torch.int64)
    backend.state_indices_list = [
        torch.full((i + 1,), -1, dtype=torch.int32) for i in range(MAX_BS)
    ]
    backend.query_start_loc_list = [
        torch.zeros((i + 2,), dtype=torch.int32) for i in range(MAX_BS)
    ]
    backend.cached_cuda_graph_decode_query_start_loc = torch.arange(
        0, MAX_BS + 1, dtype=torch.int32
    )
    backend.cached_cuda_graph_verify_query_start_loc = torch.arange(
        0, MAX_BS * DRAFT_TOKEN_NUM + 1, step=DRAFT_TOKEN_NUM, dtype=torch.int32
    )
    return backend


class TestMambaTrackDestAlignment(CustomTestCase):
    def _replay(self, *, bs: int, num_padding: int, forward_mode: ForwardMode):
        """Replay-prep at batch size ``bs``, mimicking the decode-graph caller:
        the track-dest source is the full ``(max_bs,)`` registry buffer, whose
        ``[raw_bs:]`` tail is never populated for this batch."""
        backend = _make_backend()
        registry_track_indices = torch.zeros((MAX_BS,), dtype=torch.int64)
        raw_bs = bs - num_padding
        registry_track_indices[:raw_bs] = torch.arange(
            100, 100 + raw_bs, dtype=torch.int64
        )
        metadata = backend._replay_metadata(
            bs,
            torch.arange(bs, dtype=torch.int64),
            forward_mode,
            SimpleNamespace(draft_token_num=DRAFT_TOKEN_NUM),
            None,
            num_padding=num_padding,
            mamba_track_indices=registry_track_indices,
        )
        return metadata, registry_track_indices

    def test_track_dest_is_batch_length(self):
        """The ForwardMetadata contract says length == batch; the cuda-graph path
        must honour it so per-row slicing stays aligned with the row tensors it
        is zipped against."""
        for forward_mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            for bs in (1, 3, 9, MAX_BS):
                with self.subTest(forward_mode=forward_mode, bs=bs):
                    metadata, _ = self._replay(
                        bs=bs, num_padding=0, forward_mode=forward_mode
                    )
                    self.assertEqual(metadata.mamba_track_indices.shape[0], bs)
                    self.assertEqual(metadata.mamba_cache_indices.shape[0], bs)

    def test_track_dest_rows_match_scheduler_rows(self):
        """Row i must carry the destination the scheduler picked for row i, not
        an entry read off the untouched buffer tail."""
        bs = 9
        metadata, registry_track_indices = self._replay(
            bs=bs, num_padding=0, forward_mode=ForwardMode.TARGET_VERIFY
        )
        torch.testing.assert_close(
            metadata.mamba_track_indices, registry_track_indices[:bs]
        )
        # Pre-fix this held the untouched tail, i.e. all-zero destinations.
        self.assertFalse(bool((metadata.mamba_track_indices == 0).all()))

    def test_padded_batch_keeps_real_rows_at_the_head(self):
        """Padding sits at the tail, so real rows must still start at index 0 in
        both the track dests and the cache indices they are zipped against."""
        bs, num_padding = 10, 1
        metadata, registry_track_indices = self._replay(
            bs=bs, num_padding=num_padding, forward_mode=ForwardMode.TARGET_VERIFY
        )
        raw_bs = bs - num_padding
        torch.testing.assert_close(
            metadata.mamba_track_indices[:raw_bs], registry_track_indices[:raw_bs]
        )
        # Padded rows are poisoned in the cache indices, so the track-save skips
        # them regardless of what their destinations say.
        self.assertTrue(bool((metadata.mamba_cache_indices[raw_bs:] == -1).all()))


if __name__ == "__main__":
    unittest.main()
