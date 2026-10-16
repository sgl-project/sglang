"""DFlash prefix commits write only each request's accepted MLA rows."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
from sglang.srt.runtime_context import reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.parallel_groups import publish

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/HIP")
class TestMLAPrefixValidPool(unittest.TestCase):
    def test_graph_replay_obeys_updated_commit_lengths(self):
        """Commit lengths are read on device, so a captured commit stays correct."""
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        pool = MLATokenToKVPool(
            size=31,
            page_size=1,
            dtype=torch.bfloat16,
            kv_lora_rank=128,
            qk_rope_head_dim=64,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
        )
        layer = SimpleNamespace(layer_id=0)
        loc = torch.tensor([[5, 6, 7], [9, 10, 11]], device="cuda")
        lengths = torch.tensor([2, 1], dtype=torch.int32, device="cuda")
        latent = torch.randn(6, 1, 128, dtype=torch.bfloat16, device="cuda")
        rope = torch.randn(6, 1, 64, dtype=torch.bfloat16, device="cuda")
        # Compile the MLA scatter before recording the graph.
        pool.set_mla_kv_buffer_prefix_valid(layer, loc, lengths, latent, rope)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            pool.set_mla_kv_buffer_prefix_valid(layer, loc, lengths, latent, rope)

        for commit, rows, slots in (
            ([2, 1], [0, 1, 3], [5, 6, 9]),
            ([0, 3], [3, 4, 5], [9, 10, 11]),
        ):
            pool.kv_buffer[0].zero_()
            lengths.copy_(torch.tensor(commit, dtype=torch.int32, device="cuda"))
            graph.replay()
            expected = torch.zeros_like(pool.kv_buffer[0])
            expected[slots] = torch.cat((latent, rope), dim=-1)[rows]
            self.assertTrue(torch.equal(pool.kv_buffer[0], expected), commit)


if __name__ == "__main__":
    unittest.main()
