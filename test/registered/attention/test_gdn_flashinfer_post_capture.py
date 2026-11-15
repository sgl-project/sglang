import unittest

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
    FlashInferGDNKernel,
)
from sglang.srt.mem_cache.kv.layout.mha import MHASubPoolSpec
from sglang.srt.mem_cache.kv.layout.state import MambaSubPoolSpec
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import arena_of

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "FlashInfer gather/scatter decode runs on SM90",
)
class TestFlashInferPostCaptureState(CustomTestCase):
    def test_padded_graph_reads_only_finalized_backing(self):
        """Padded SM90 replay must not index the unbacked tail of captured state views."""
        state_spec = MambaSubPoolSpec(
            name="mamba",
            layer_num=1,
            grow_direction="up",
            conv_state_shapes=((1, 16),),
            conv_dtype=torch.float32,
            temporal_state_shape=(1, 128, 128),
            temporal_dtype=torch.float32,
        )
        full_spec = MHASubPoolSpec(
            name="full",
            layer_num=1,
            head_num=1,
            head_dim=128,
            store_dtype=torch.bfloat16,
            grow_direction="down",
        )
        arena = arena_of(
            [state_spec, full_spec],
            total_bytes=64 << 20,
            device="cuda",
            post_capture_active=True,
        )
        states = state_spec.views(arena)[1][0]
        identity = (states.data_ptr(), tuple(states.shape), tuple(states.stride()))
        self.assertNotEqual(states.stride(0), states[0].numel())
        kernel = FlashInferGDNKernel()
        self.assertFalse(kernel.use_state_pool)
        torch.manual_seed(42)
        q = torch.randn((1, 3, 1, 128), dtype=torch.bfloat16, device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        a = torch.zeros((1, 3, 1), dtype=torch.bfloat16, device="cuda")
        b = torch.zeros_like(a)
        A_log = torch.zeros(1, dtype=torch.float32, device="cuda")
        dt_bias = torch.zeros(1, dtype=torch.bfloat16, device="cuda")
        indices = torch.zeros(3, dtype=torch.int32, device="cuda")
        starts = torch.arange(4, dtype=torch.int32, device="cuda")

        def forward():
            return kernel.decode(
                q,
                k,
                v,
                a,
                b,
                A_log=A_log,
                dt_bias=dt_bias,
                ssm_states=states,
                cache_indices=indices,
                query_start_loc=starts,
            )

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = forward()
        arena.finalize_backing(4 << 20)
        self.assertEqual(
            identity, (states.data_ptr(), tuple(states.shape), tuple(states.stride()))
        )
        tail_offset = (
            states.data_ptr()
            - arena.raw.data_ptr()
            + (states.shape[0] - 1) * states.stride(0) * states.element_size()
        )
        self.assertGreaterEqual(tail_offset, arena.backed_bytes)
        states[1:3].copy_(torch.randn_like(states[1:3]) * 0.01)
        reference = states[:3].clone()
        for slot in (1, 2, 1):
            indices.copy_(
                torch.tensor([slot, -1, -1], dtype=torch.int32, device="cuda")
            )
            expected = kernel.decode(
                q[:, :1],
                k[:, :1],
                v[:, :1],
                a[:, :1],
                b[:, :1],
                A_log=A_log,
                dt_bias=dt_bias,
                ssm_states=reference,
                cache_indices=indices[:1],
                query_start_loc=starts[:2],
            )
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output[:, :1], expected, rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(
                states[1:3], reference[1:3], rtol=1e-4, atol=1e-4
            )
            self.assertEqual(torch.count_nonzero(output[:, 1:]).item(), 0)
            self.assertEqual(torch.count_nonzero(states[0]).item(), 0)
        before_idle = states[:3].clone()
        indices.fill_(-1)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(states[:3], before_idle)
        self.assertEqual(torch.count_nonzero(output).item(), 0)


if __name__ == "__main__":
    unittest.main()
