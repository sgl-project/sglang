"""DSv4 decode length buckets: CPU selection and persistent metadata isolation."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.deepseek_v4_backend import (
    DeepseekV4AttnBackend,
    DSV4RawDecodeMetadata,
    _GraphBucket,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.seq_len_buckets import (
    normalize_seq_len_buckets,
    select_seq_len_bucket,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDecodeSeqLenBuckets(unittest.TestCase):
    def test_normalization_and_fallback(self):
        self.assertEqual(
            normalize_seq_len_buckets([8192, 4096, 4096], 32772),
            [4096, 8192, 32772],
        )
        for invalid in ([], [0], [-1], [32773]):
            with self.assertRaises(ValueError):
                normalize_seq_len_buckets(invalid, 32772)

    def test_boundaries_growth_and_shrinking(self):
        buckets = [4096, 8192, 32772]
        for actual, expected in (
            (0, 4096),
            (4095, 4096),
            (4096, 4096),
            (4097, 8192),
            (8192, 8192),
            (8193, 32772),
            (32772, 32772),
            (32773, None),
            (4096, 4096),
        ):
            self.assertEqual(select_seq_len_bucket(buckets, actual), expected)

    def test_mixed_batch_selection_and_missing_mirror(self):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.capture_seq_lens = [4096, 8192, 32772]
        for lengths, expected in (
            (torch.tensor([1, 4097, 2048]), 8192),
            (torch.tensor([4096, 1]), 4096),
            (torch.empty(0, dtype=torch.int64), 4096),
            (None, None),
        ):
            self.assertEqual(
                runner._select_graph_seq_len(SimpleNamespace(seq_lens_cpu=lengths)),
                expected,
            )

    def test_shape_keys_do_not_alias(self):
        keys = {ShapeKey(2), ShapeKey(2, seq_len=4096), ShapeKey(2, seq_len=8192)}
        self.assertEqual(len(keys), 3)

    def test_raw_metadata_bound_is_static(self):
        def raw(bound, length):
            return DSV4RawDecodeMetadata(
                torch.tensor([0]), torch.tensor([length]), torch.tensor([1]), bound
            )

        a, b = raw(4096, 1024), raw(8192, 5000)
        with self.assertRaises(AssertionError):
            a.copy_(b)
        a.copy_(raw(4096, 2048))
        self.assertEqual(a.seq_lens.item(), 2048)

    def test_metadata_keys_isolate_length_buckets(self):
        backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
        backend.cuda_graph_metadata_of_bucket_and_bs = {_GraphBucket.DECODE_OR_IDLE: {}}
        first, second, update = Mock(), Mock(), Mock()
        for key, metadata in (
            ((2, 4096), first),
            ((2, 8192), second),
            ((2, 4096), update),
        ):
            backend.replay_cuda_graph_metadata_from(
                key, metadata, _GraphBucket.DECODE_OR_IDLE
            )
        first.copy_.assert_called_once_with(update)
        second.copy_.assert_not_called()
        self.assertIs(backend.forward_metadata, first)

    def test_in_graph_raw_upgrade_preserves_bound(self):
        backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
        backend.MAX_SEQ_LEN_FOR_CAPTURE = 32772
        backend.req_to_token = None
        backend.token_to_kv_pool = None
        backend.make_core_attn_metadata = Mock(return_value=None)
        backend.init_forward_metadata_indexer = Mock(return_value=None)
        raw = DSV4RawDecodeMetadata(
            torch.tensor([0]), torch.tensor([1000]), torch.tensor([1]), 4096
        )
        with patch(
            "sglang.srt.layers.attention.deepseek_v4_backend.create_paged_compressor_data",
            return_value=None,
        ):
            backend.make_forward_metadata_from_raw_decode(raw)
        self.assertEqual(
            backend.make_core_attn_metadata.call_args.kwargs["max_seq_len"], 4096
        )

    def test_replay_metadata_growth_and_revisit_with_padding(self):
        backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
        backend.gvr_state = None
        backend.needs_cpu_seq_lens = True
        backend.is_dspark_draft = False
        backend.MAX_SEQ_LEN_FOR_CAPTURE = 32772
        backend.online_c128_mtp = Mock()
        backend.cuda_graph_metadata_of_bucket_and_bs = {_GraphBucket.DECODE_OR_IDLE: {}}

        def make_metadata(max_seq_len, req_pool_indices, seq_lens, out_cache_loc):
            return DSV4RawDecodeMetadata(
                req_pool_indices.clone(),
                seq_lens.clone(),
                out_cache_loc.clone(),
                max_seq_len,
            )

        backend.init_forward_metadata_decode = Mock(side_effect=make_metadata)
        previous = {}
        for bound, length in ((4096, 4096), (8192, 4097), (4096, 1024)):
            fb = SimpleNamespace(
                batch_size=2,
                forward_mode=ForwardMode.DECODE,
                req_pool_indices=torch.tensor([3, 0]),
                seq_lens=torch.tensor([length, 1]),
                seq_lens_cpu=torch.tensor([length, 1]),
                seq_lens_sum=length + 1,
                out_cache_loc=torch.tensor([17]),
                max_seq_len_override=bound,
            )
            backend.init_forward_metadata_out_graph(fb)
            metadata = backend.forward_metadata
            if bound in previous:
                self.assertIs(metadata, previous[bound])
            previous[bound] = metadata
            self.assertEqual(metadata.max_seq_len, bound)
            self.assertEqual(metadata.seq_lens.tolist(), [length, 1])
            self.assertEqual(metadata.out_cache_loc.tolist(), [17, 0])
        self.assertEqual(len(previous), 2)

    def test_decode_metadata_bound_with_both_prep_modes(self):
        backend = DeepseekV4AttnBackend.__new__(DeepseekV4AttnBackend)
        backend.cuda_graph_seq_lens_enabled = True
        backend.req_to_token = None
        backend.token_to_kv_pool = None
        backend.make_core_attn_metadata = Mock(return_value=None)
        backend.init_forward_metadata_indexer = Mock(return_value=None)
        for prep_in_graph in (False, True):
            with (
                patch.object(
                    envs.SGLANG_PREP_IN_CUDA_GRAPH, "get", return_value=prep_in_graph
                ),
                patch(
                    "sglang.srt.layers.attention.deepseek_v4_backend.create_paged_compressor_data",
                    return_value=None,
                ),
            ):
                metadata = backend.init_forward_metadata_decode(
                    4096, torch.tensor([0]), torch.tensor([1024]), torch.tensor([1])
                )
            if prep_in_graph:
                self.assertEqual(metadata.max_seq_len, 4096)
            else:
                self.assertEqual(
                    backend.make_core_attn_metadata.call_args.kwargs["max_seq_len"],
                    4096,
                )


if __name__ == "__main__":
    unittest.main()
