"""CPU coverage for eager hybrid-linear metadata after MLP-sync padding."""

from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    MambaAttnBackendBase,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _ReqToTokenPool:
    def get_mamba_indices(self, req_pool_indices):
        return req_pool_indices.clone()

    def translate_mamba_indices(self, mamba_indices):
        return mamba_indices


def _backend():
    backend = object.__new__(MambaAttnBackendBase)
    backend.device = torch.device("cpu")
    backend.topk = 1
    backend.req_to_token_pool = _ReqToTokenPool()
    return backend


def _target_verify_batch(
    *,
    draft_token_num: int,
    physical_tokens: int,
    original_batch_size: int = 1,
    ragged_query_start_loc=None,
):
    ragged_layout = (
        None
        if ragged_query_start_loc is None
        else SimpleNamespace(qo_indptr_device=ragged_query_start_loc)
    )
    return SimpleNamespace(
        batch_size=physical_tokens // draft_token_num,
        _original_batch_size=original_batch_size,
        req_pool_indices=torch.tensor(
            list(range(17, 17 + original_batch_size))
            + [0] * (physical_tokens // draft_token_num - original_batch_size)
        ),
        mamba_track_indices=None,
        mamba_track_mask=None,
        extend_start_loc=None,
        extend_seq_lens=None,
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=SimpleNamespace(
            draft_token_num=draft_token_num,
            ragged_verify_layout=ragged_layout,
        ),
        input_ids=torch.arange(physical_tokens),
        num_token_non_padded_cpu=(
            original_batch_size * draft_token_num
            if ragged_query_start_loc is None
            else int(ragged_query_start_loc[-1])
        ),
    )


class TestHybridLinearAttentionMetadata(CustomTestCase):
    def test_empty_dp_rank_skips_child_metadata_planning(self):
        class _MetadataBackend:
            def __init__(self):
                self.forward_metadata = object()
                self.calls = 0

            def init_forward_metadata(self, _forward_batch):
                self.calls += 1

        full_attn = _MetadataBackend()
        linear_attn = _MetadataBackend()
        backend = object.__new__(HybridLinearAttnBackend)
        backend.attn_backend_list = [full_attn, linear_attn]

        backend.init_forward_metadata(
            SimpleNamespace(batch_size=0, forward_mode=ForwardMode.TARGET_VERIFY)
        )

        self.assertEqual(full_attn.calls, 0)
        self.assertEqual(linear_attn.calls, 0)
        self.assertIsNone(full_attn.forward_metadata)
        self.assertIsNone(linear_attn.forward_metadata)

    def test_empty_dp_target_verify_skips_linear_attention_forward(self):
        backend = object.__new__(HybridLinearAttnBackend)
        backend._is_full_attn = lambda *_args, **_kwargs: False
        mixed_qkv = torch.empty((0, 16))
        layer = SimpleNamespace(num_v_heads=2, head_v_dim=4)
        forward_batch = SimpleNamespace(
            batch_size=0, forward_mode=ForwardMode.TARGET_VERIFY
        )

        output = backend.forward(
            layer=layer,
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
        )

        self.assertEqual(output.shape, (0, 2, 4))

    def test_eager_target_verify_collapses_mlp_sync_padding_rows(self):
        # Adaptive MTP N=4 is physically padded to attn-TP=8. The recurrent
        # backend keeps two request rows for collective shape compatibility,
        # but the fabricated row must describe an empty query interval.
        forward_batch = _target_verify_batch(draft_token_num=4, physical_tokens=8)

        metadata = _backend()._forward_metadata(forward_batch)

        self.assertEqual(metadata.query_start_loc.tolist(), [0, 4, 4])
        self.assertEqual(metadata.mamba_cache_indices.tolist(), [17, -1])
        self.assertEqual(
            int(metadata.query_start_loc[-1]),
            forward_batch.num_token_non_padded_cpu,
        )
        self.assertEqual(
            metadata.query_start_loc.shape[0] - 1,
            metadata.mamba_cache_indices.shape[0],
        )

    def test_eager_target_verify_preserves_unpadded_request_domain(self):
        forward_batch = _target_verify_batch(draft_token_num=6, physical_tokens=6)

        metadata = _backend()._forward_metadata(forward_batch)

        self.assertEqual(metadata.query_start_loc.tolist(), [0, 6])
        self.assertEqual(metadata.mamba_cache_indices.tolist(), [17])

    def test_eager_target_verify_preserves_real_requests_before_padding(self):
        forward_batch = _target_verify_batch(
            draft_token_num=4,
            physical_tokens=16,
            original_batch_size=2,
        )

        metadata = _backend()._forward_metadata(forward_batch)

        self.assertEqual(metadata.query_start_loc.tolist(), [0, 4, 8, 8, 8])
        self.assertEqual(metadata.mamba_cache_indices.tolist(), [17, 18, -1, -1])

    def test_eager_ragged_verify_pads_only_the_request_domain(self):
        forward_batch = _target_verify_batch(
            draft_token_num=4,
            physical_tokens=8,
            ragged_query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        )
        metadata = _backend()._forward_metadata(forward_batch)

        self.assertEqual(metadata.query_start_loc.tolist(), [0, 3, 3])
        self.assertEqual(metadata.mamba_cache_indices.tolist(), [17, -1])


if __name__ == "__main__":
    import unittest

    unittest.main()
