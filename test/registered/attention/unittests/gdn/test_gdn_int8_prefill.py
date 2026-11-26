"""INT8 GDN prefill matches FP32 state math without staging a dense state."""

import unittest

import torch

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule_fwd
from sglang.kernels.ops.attention.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from sglang.kernels.ops.attention.fla.chunk_fwd import (
    chunk_gated_delta_rule_fwd_intra,
)
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None, "requires NVIDIA CUDA"
)
class TestGDNInt8Prefill(unittest.TestCase):
    def test_cached_suffix_cold_row_and_tracking(self):
        # The first sequence starts at a cached 256-token checkpoint and extends
        # by 344 tokens. The second starts cold from a dirty, reused cache slot.
        device = "cuda"
        generator = torch.Generator(device=device).manual_seed(2026)
        slots, heads, value_heads, dim = 8, 2, 6, 128

        def index(values):
            return torch.tensor(values, dtype=torch.int32, device=device)

        def randn(*shape, dtype=torch.bfloat16):
            return torch.randn(shape, dtype=dtype, device=device, generator=generator)

        storage = torch.randint(
            -127,
            128,
            (slots, 2, value_heads, dim, dim),
            dtype=torch.int8,
            device=device,
            generator=generator,
        )
        scale_storage = (
            randn(slots, 2, value_heads, dim, dtype=torch.float32).abs() * 0.01
        )
        # A per-layer view of a slot-major pool keeps the slot stride for both tensors.
        h0, scales = storage[:, 0], scale_storage[:, 0]
        state_before, scale_before = storage.clone(), scale_storage.clone()
        source_slots, track_slots = index([3, 1]), index([6, 5])
        starts, lengths = index([2, 1]), index([0, 344, 474])
        hit_mask = torch.tensor([True, False], device=device)
        q, k = (randn(1, 474, heads, dim) for _ in range(2))
        v = randn(1, 474, value_heads, dim)
        g = -randn(1, 474, value_heads, dtype=torch.float32).abs() * 0.1
        beta = randn(1, 474, value_heads).sigmoid()

        reference = h0[source_slots].float() * scales[source_slots].unsqueeze(-1)
        reference[1].zero_()
        _, expected, _, _, expected_chunks, _ = chunk_gated_delta_rule_fwd(
            q=l2norm_fwd(q),
            k=l2norm_fwd(k),
            v=v,
            g=g,
            beta=beta,
            scale=dim**-0.5,
            initial_state=reference,
            initial_state_indices=index([0, 1]),
            cu_seqlens=lengths,
        )

        # Use a separate FP32 snapshot buffer for the independent tracking
        # reference. The returned h tiles are in the activation dtype.
        track_reference = torch.empty_like(reference)
        norm_k = l2norm_fwd(k)
        g_cumsum = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=lengths)
        w, u, _ = chunk_gated_delta_rule_fwd_intra(
            k=norm_k, v=v, g=g_cumsum, beta=beta, cu_seqlens=lengths
        )
        source = h0[source_slots].float() * scales[source_slots].unsqueeze(-1)
        source[1].zero_()
        chunk_gated_delta_rule_fwd_h(
            k=norm_k,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=source,
            initial_state_indices=index([0, 1]),
            cu_seqlens=lengths,
            inplace_update=False,
            track_state=track_reference,
            track_chunk_idx=starts,
        )

        out, _, chunks = TritonGDNKernel().extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=h0,
            cache_indices=source_slots,
            query_start_loc=lengths,
            initial_state_scale=scales,
            has_initial_state=hit_mask,
            track_chunk_idx=starts,
            track_state_indices=track_slots,
        )
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(chunks, expected_chunks, rtol=0, atol=0)
        for rows, state in ((source_slots, reference), (track_slots, track_reference)):
            scale = (state.abs().amax(dim=-1) / 127).clamp_min(1e-12)
            quantized = (
                (state / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
            )
            torch.testing.assert_close(scales[rows], scale, rtol=2e-6, atol=1e-9)
            torch.testing.assert_close(h0[rows], quantized, rtol=0, atol=1)

        untouched = index([0, 2, 4, 7])
        torch.testing.assert_close(
            storage[untouched], state_before[untouched], rtol=0, atol=0
        )
        torch.testing.assert_close(
            scale_storage[untouched], scale_before[untouched], rtol=0, atol=0
        )
        torch.testing.assert_close(storage[:, 1], state_before[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(
            scale_storage[:, 1], scale_before[:, 1], rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
