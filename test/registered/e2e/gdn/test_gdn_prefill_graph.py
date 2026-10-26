"""Captured GDN extend preserves eager outputs and state across replay shapes."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    GDNKernelDispatcher,
)
from sglang.srt.layers.attention.linear.utils import LinearAttnKernelBackend
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")

QUERY_HEADS, VALUE_HEADS, HEAD_DIM = 2, 4, 128
DIM = (2 * QUERY_HEADS + VALUE_HEADS) * HEAD_DIM
BUCKET, MAX_SEQUENCES, SLOTS = 160, 4, 12


def make_backend(conv_states: torch.Tensor, ssm_states: torch.Tensor) -> GDNAttnBackend:
    backend = object.__new__(GDNAttnBackend)
    state = SimpleNamespace(conv=[conv_states], temporal=ssm_states)
    backend.req_to_token_pool = SimpleNamespace(
        size=MAX_SEQUENCES,
        mamba_pool=SimpleNamespace(mamba_cache=state),
        mamba2_layer_cache=lambda layer_id: state,
        get_mamba_indices=lambda request_indices: (request_indices + 1).int(),
        translate_mamba_indices=lambda indices: indices,
    )
    backend.device = conv_states.device
    backend.mis_metadata = None
    backend.enable_mis = False
    backend._mamba_chunk_size = 64
    backend.conv_states_shape = conv_states.shape
    backend.prefill_graph_max_seqs = MAX_SEQUENCES
    backend.kernel_dispatcher = GDNKernelDispatcher(
        LinearAttnKernelBackend.TRITON, LinearAttnKernelBackend.TRITON
    )
    return backend


def make_batch(lengths: list[int], prefixes: list[int], track: bool) -> ForwardBatch:
    count, total = len(lengths), sum(lengths)
    sequence_lengths = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    prefix_lengths = torch.tensor(prefixes, device="cuda", dtype=torch.int32)
    mask = [track and length >= 64 for length in lengths]
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=count,
        input_ids=torch.zeros(total, device="cuda", dtype=torch.int64),
        positions=torch.arange(total, device="cuda"),
        req_pool_indices=torch.arange(count, device="cuda"),
        seq_lens=prefix_lengths + sequence_lengths,
        seq_lens_sum=total + sum(prefixes),
        out_cache_loc=torch.zeros(total, device="cuda", dtype=torch.int64),
        extend_num_tokens=total,
        extend_start_loc=torch.cat(
            [sequence_lengths.new_zeros(1), sequence_lengths.cumsum(0)[:-1]]
        ).int(),
        extend_seq_lens=sequence_lengths,
        extend_prefix_lens=prefix_lengths,
        extend_seq_lens_cpu=lengths,
        extend_prefix_lens_cpu=prefixes,
        mamba_track_mask=torch.tensor(mask, device="cuda") if track else None,
        mamba_track_indices=(
            torch.arange(6, 6 + count, device="cuda", dtype=torch.int32)
            if track
            else None
        ),
        mamba_track_seqlens=prefix_lengths + sequence_lengths if track else None,
        mamba_prefill_track_mask_cpu=mask if track else None,
        mamba_track_seqlens_cpu=(
            [prefix + length for prefix, length in zip(prefixes, lengths)]
            if track
            else None
        ),
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGDNPrefillGraph(CustomTestCase):
    @torch.inference_mode()
    def test_replay_preserves_outputs_and_owned_states(self):
        """One graph must handle changed lengths, cached state, and padding safely."""
        torch.manual_seed(7)
        layer = RadixLinearAttention(
            layer_id=0,
            num_q_heads=QUERY_HEADS,
            num_k_heads=QUERY_HEADS,
            num_v_heads=VALUE_HEADS,
            head_q_dim=HEAD_DIM,
            head_k_dim=HEAD_DIM,
            head_v_dim=HEAD_DIM,
            conv_weights=torch.randn(DIM, 4, device="cuda", dtype=torch.bfloat16) * 0.1,
            bias=None,
            activation="silu",
            A_log=torch.linspace(0.1, 1, VALUE_HEADS, device="cuda").log(),
            dt_bias=torch.zeros(VALUE_HEADS, device="cuda", dtype=torch.bfloat16),
        )
        cases = [
            (False, 24),
            (False, BUCKET),
            (True, BUCKET),
        ]
        for track, bucket in cases:
            with (
                self.subTest(tracking=track, bucket=bucket),
                get_context().override_server_args(
                    disable_radix_cache=not track,
                    mamba_radix_cache_strategy="extra_buffer",
                    _mamba_cache_chunk_size=64,
                ),
            ):
                conv_states = torch.randn(SLOTS, DIM, 3, device="cuda").bfloat16() * 0.1
                ssm_states = (
                    torch.randn(SLOTS, VALUE_HEADS, HEAD_DIM, HEAD_DIM, device="cuda")
                    * 0.01
                )
                backend = make_backend(conv_states, ssm_states)
                mixed = torch.zeros(bucket, DIM, device="cuda", dtype=torch.bfloat16)
                a = torch.zeros(bucket, VALUE_HEADS, device="cuda", dtype=mixed.dtype)
                b = torch.zeros_like(a)
                batch = make_batch([bucket], [0], track)
                metadata = backend.init_prefill_graph_metadata(batch)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    backend.forward_extend(layer, batch, mixed, a, b)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = backend.forward_extend(layer, batch, mixed, a, b)
                layouts = (
                    [([1, 7], [0, 64])]
                    if bucket < 64
                    else [([1, 23, 65], [0, 128, 64]), ([64, 65], [64, 0])]
                )
                for lengths, prefixes in layouts:
                    batch = make_batch(lengths, prefixes, track)
                    total = sum(lengths)
                    for row, prefix in enumerate(prefixes, start=1):
                        if prefix == 0:
                            conv_states[row].zero_()
                            ssm_states[row].zero_()
                    mixed.normal_()
                    a.normal_()
                    b.normal_()
                    reference_conv, reference_ssm = (
                        conv_states.clone(),
                        ssm_states.clone(),
                    )
                    reference = make_backend(reference_conv, reference_ssm)
                    reference_batch = make_batch(lengths, prefixes, track)
                    reference.init_forward_metadata(reference_batch)
                    expected = reference.forward_extend(
                        layer, reference_batch, mixed[:total], a[:total], b[:total]
                    )
                    backend.refresh_prefill_graph_metadata(metadata, batch)
                    graph.replay()
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(output[:, :total], expected))
                    self.assertTrue(bool((output[:, total:] == 0).all()))
                    self.assertTrue(torch.equal(conv_states, reference_conv))
                    self.assertTrue(torch.equal(ssm_states, reference_ssm))
                backend.init_forward_metadata(batch)
                actual = backend.forward_extend(
                    layer, batch, mixed[:total], a[:total], b[:total]
                )
                expected = reference.forward_extend(
                    layer, reference_batch, mixed[:total], a[:total], b[:total]
                )
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(ssm_states, reference_ssm))


if __name__ == "__main__":
    unittest.main()
