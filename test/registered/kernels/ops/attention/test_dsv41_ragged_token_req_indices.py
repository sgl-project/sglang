"""Request mapping for packed DeepSeek-V4.1 target verification."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsv4.dsv41_sparse import token_req_indices
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestV41RaggedTokenRequestMapping(CustomTestCase):
    def make_batch(self, lengths, slots, num_tokens, device="cpu"):
        return SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            req_pool_indices=torch.tensor(slots, dtype=torch.int32, device=device),
            spec_info=SimpleNamespace(
                draft_token_num=6,
                ragged_verify_layout=SimpleNamespace(
                    graph_num_tokens=num_tokens,
                    qo_indptr_device=torch.tensor(
                        [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                        dtype=torch.int32,
                        device=device,
                    ),
                ),
            ),
        )

    def reference(self, lengths, slots, num_tokens, device="cpu"):
        mapped = [slot for length, slot in zip(lengths, slots) for _ in range(length)]
        return torch.tensor(
            mapped + [0] * (num_tokens - len(mapped)), dtype=torch.int64, device=device
        )

    def test_capture_tier_and_padding_tail(self):
        for lengths, slots, num_tokens in (
            ([6] * 4 + [5] * 12, list(range(1, 17)), 84),
            ([6] * 13 + [0] * 3, list(range(16, 0, -1)), 84),
            ([0, 1, 0, 6, 2, 0], [9, 4, 3, 7, 1, 8], 18),
            ([0, 0, 0], [9, 4, 3], 6),
        ):
            batch = self.make_batch(lengths, slots, num_tokens)
            actual = token_req_indices(batch, num_tokens=num_tokens)
            expected = self.reference(lengths, slots, num_tokens)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dense_verify_still_repeats_the_full_window(self):
        batch = self.make_batch([6, 6], [9, 4], 12)
        batch.spec_info.ragged_verify_layout = None
        torch.testing.assert_close(
            token_req_indices(batch, num_tokens=12),
            torch.tensor([9] * 6 + [4] * 6),
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_graph_replay_updates_mapping_and_padding(self):
        slots = list(range(16, 0, -1))
        batch = self.make_batch([6] * 4 + [5] * 12, slots, 84, "cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                token_req_indices(batch, num_tokens=84)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = token_req_indices(batch, num_tokens=84)
        lengths = [6] * 13 + [0] * 3
        batch.spec_info.ragged_verify_layout.qo_indptr_device.copy_(
            self.make_batch(
                lengths, slots, 84, "cuda"
            ).spec_info.ragged_verify_layout.qo_indptr_device
        )
        graph.replay()
        expected = self.reference(lengths, slots, 84, "cuda")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
