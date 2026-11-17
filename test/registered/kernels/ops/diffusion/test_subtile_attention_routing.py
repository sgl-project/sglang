import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.sdpa import (
    DynamicCudnnSDPAImpl,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    enable_breakable_cuda_graph,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@unittest.skipUnless(
    torch.cuda.is_available() and current_platform.is_blackwell(), "requires Blackwell"
)
class TestSubtileAttentionRouting(unittest.TestCase):
    def test_capture_and_eager_replay_use_fresh_inputs(self):
        for q_len, kv_len in ((126, 1536), (1536, 126)):
            with self.subTest(q_len=q_len, kv_len=kv_len):
                self._check_capture_and_eager_replay(q_len, kv_len)

    def _check_capture_and_eager_replay(self, q_len, kv_len):
        torch.manual_seed(0)
        impl = DynamicCudnnSDPAImpl(
            num_heads=32, head_size=64, causal=False, softmax_scale=64**-0.5
        )
        q = torch.randn(1, q_len, 32, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, kv_len, 32, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        with (
            torch.inference_mode(),
            set_forward_context(
                current_timestep=0, attn_metadata=None, quality="lossless"
            ),
        ):
            reference = impl.forward(q, k, v, None)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), enable_breakable_cuda_graph():
                for _ in range(2):
                    impl.forward(q, k, v, None)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with enable_breakable_cuda_graph(), torch.cuda.graph(graph):
                captured = impl.forward(q, k, v, None)
            graph.replay()
            torch.testing.assert_close(captured, reference, rtol=1e-2, atol=1e-2)
            with (
                set_forward_context(0, None, quality="exact"),
                patch.object(impl.fa_impl, "forward", wraps=impl.fa_impl.forward) as fa,
            ):
                exact = impl.forward(q, k, v, None)
                fa.assert_called_once()
            torch.testing.assert_close(captured, exact, rtol=0, atol=0)
            q.copy_(torch.randn_like(q))
            k.copy_(torch.randn_like(k))
            v.copy_(torch.randn_like(v))
            with patch.object(
                impl.cudnn_impl, "forward", wraps=impl.cudnn_impl.forward
            ) as cudnn:
                reference = impl.forward(q, k, v, None)
                cudnn.assert_called_once()
            graph.replay()
            torch.testing.assert_close(captured, reference, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
