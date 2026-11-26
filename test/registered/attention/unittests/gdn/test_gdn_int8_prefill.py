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

register_cuda_ci(est_time=35, stage="base-b", runner_config="1-gpu-large")


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

    def make_inputs(self, tokens):
        generator = torch.Generator(device="cuda").manual_seed(53226)

        def randn(*shape, dtype=torch.bfloat16):
            return torch.randn(shape, device="cuda", dtype=dtype, generator=generator)

        heads, value_heads, dim = 2, 6, 128
        return (
            randn(1, tokens, heads, dim) * 0.2,
            randn(1, tokens, heads, dim) * 0.2,
            randn(1, tokens, value_heads, dim) * 0.2,
            -randn(1, tokens, value_heads, dtype=torch.float32).abs() * 0.1,
            randn(1, tokens, value_heads).sigmoid(),
        )

    @staticmethod
    def index(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    @staticmethod
    def pack(state):
        row_scale = (state.abs().amax(dim=-1) / 127).clamp_min(1e-12)
        data = (state / row_scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)
        return data, row_scale

    def test_rejects_checkpoint_axes_mismatched_to_activations(self):
        kernel = TritonGDNKernel()
        inputs = self.make_inputs(64)
        for shape in ((7, 128, 128), (6, 129, 128), (6, 128, 129)):
            with self.subTest(shape=shape):
                state = torch.full((3, *shape), 37, dtype=torch.int8, device="cuda")
                scales = torch.full((3, *shape[:-1]), 0.005, device="cuda")
                before = state.clone(), scales.clone()
                with self.assertRaisesRegex(ValueError, "to match prefill activations"):
                    kernel.extend(
                        *inputs,
                        ssm_states=state,
                        cache_indices=self.index([1]),
                        query_start_loc=self.index([0, 64]),
                        initial_state_scale=scales,
                    )
                torch.testing.assert_close(state, before[0], rtol=0, atol=0)
                torch.testing.assert_close(scales, before[1], rtol=0, atol=0)

    def test_two_extends_requantize_at_each_boundary(self):
        kernel = TritonGDNKernel()
        active, tracked = self.index([1]), self.index([3])
        state = torch.full((4, 6, 128, 128), 27, dtype=torch.int8, device="cuda")
        scales = torch.full((4, 6, 128), 0.005, device="cuda")
        inputs = self.make_inputs(344)
        expected_state, expected_scale = state[active].clone(), scales[active].clone()
        for start, end in ((0, 128), (128, 344)):
            with self.subTest(length=end - start):
                q, k, v, g, beta = (x[:, start:end].contiguous() for x in inputs)
                cu = self.index([0, end - start])
                initial = expected_state.float() * expected_scale.unsqueeze(-1)
                # A snapshot at this extend's first chunk is the checkpoint
                # restored from the previous extend's quantized writeback.
                expected_track = self.pack(initial)
                reference = initial.clone()
                _, expected_out, _, _, _, _ = chunk_gated_delta_rule_fwd(
                    q=l2norm_fwd(q),
                    k=l2norm_fwd(k),
                    v=v,
                    g=g,
                    beta=beta,
                    scale=128**-0.5,
                    initial_state=reference,
                    initial_state_indices=self.index([0]),
                    cu_seqlens=cu,
                )
                expected_state, expected_scale = self.pack(reference)
                out, _, _ = kernel.extend(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ssm_states=state,
                    cache_indices=active,
                    query_start_loc=cu,
                    initial_state_scale=scales,
                    track_chunk_idx=self.index([0]),
                    track_state_indices=tracked,
                )
                torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
                torch.testing.assert_close(
                    state[active], expected_state, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    scales[active], expected_scale, rtol=2e-6, atol=1e-9
                )
                torch.testing.assert_close(
                    state[tracked], expected_track[0], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    scales[tracked], expected_track[1], rtol=2e-6, atol=1e-9
                )

    def test_cuda_graph_dynamic_slots_match_eager(self):
        from sglang.kernels.ops.attention.fla.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )

        kernel = TritonGDNKernel()
        inputs = self.make_inputs(256)
        cu = self.index([0, 128, 256])
        tables = dict(
            chunk_indices=prepare_chunk_indices(cu, 64),
            chunk_offsets=prepare_chunk_offsets(cu, 64),
        )
        initial = torch.full((8, 2, 6, 128, 128), 23, dtype=torch.int8, device="cuda")
        initial_scale = torch.full((8, 2, 6, 128), 0.005, device="cuda")
        state, scales = initial.clone(), initial_scale.clone()
        slots, destinations, chunks = (
            self.index([1, 2]),
            self.index([5, 6]),
            self.index([0, 1]),
        )
        hit = torch.tensor([True, False], device="cuda")

        def run(source, row_scales, update):
            return kernel.extend(
                *inputs,
                ssm_states=source[:, 0],
                cache_indices=slots,
                query_start_loc=cu,
                initial_state_scale=row_scales[:, 0],
                has_initial_state=hit,
                track_chunk_idx=chunks,
                track_state_indices=destinations,
                inplace_update=update,
                **tables,
            )

        cases = (
            ([1, 2], [5, 6], [True, False], [2], float("nan")),
            ([3, 4], [-1, -1], [False, True], [3], float("inf")),
            ([1, -1], [6, 7], [True, False], [0], float("nan")),
        )
        for update in (True, False):
            run(state, scales, update)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual, _, actual_chunks = run(state, scales, update)

            for active, tracked, has_hit, dirty, value in cases:
                with self.subTest(update=update, active=active, tracked=tracked):
                    slots.copy_(self.index(active))
                    destinations.copy_(self.index(tracked))
                    hit.copy_(torch.tensor(has_hit, device="cuda"))
                    state.copy_(initial)
                    scales.copy_(initial_scale)
                    scales[dirty, 0] = value

                    # The clean eager control exercises the same data and dynamic
                    # metadata. A cold row must ignore its prior state and scale.
                    expected_state, expected_scales = (
                        initial.clone(),
                        initial_scale.clone(),
                    )
                    expected, _, expected_chunks = run(
                        expected_state, expected_scales, update
                    )
                    graph.replay()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        actual_chunks, expected_chunks, rtol=0, atol=0
                    )
                    self.assertTrue(bool(torch.isfinite(actual).all()))

                    touched = {
                        target
                        for slot, target in zip(active, tracked)
                        if slot >= 0 and target >= 0
                    }
                    if update:
                        touched.update(slot for slot in active if slot >= 0)
                    untouched = self.index([i for i in range(8) if i not in touched])
                    if touched:
                        selected = self.index(sorted(touched))
                        torch.testing.assert_close(
                            state[selected], expected_state[selected], rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            scales[selected], expected_scales[selected], rtol=0, atol=0
                        )
                    torch.testing.assert_close(
                        state[untouched], initial[untouched], rtol=0, atol=0
                    )
                    unchanged_scales = initial_scale.clone()
                    unchanged_scales[dirty, 0] = value
                    torch.testing.assert_close(
                        scales[untouched],
                        unchanged_scales[untouched],
                        rtol=0,
                        atol=0,
                        equal_nan=True,
                    )
                    torch.testing.assert_close(
                        state[:, 1], initial[:, 1], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        scales[:, 1], initial_scale[:, 1], rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
