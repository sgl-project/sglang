import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestRaggedMropePositions(unittest.TestCase):
    def test_bucket_padding_uses_the_real_ragged_layout(self):
        device = torch.device("cuda")
        forward_batch = ForwardBatch.__new__(ForwardBatch)
        forward_batch.seq_lens = torch.tensor(
            [100, 200, 300], dtype=torch.int32, device=device
        )
        layout = RaggedVerifyLayout.from_verify_lens(
            verify_lens_cpu=[1, 3, 2],
            device=device,
            grid=[8],
        )
        self.assertEqual(layout.bs, 3)
        self.assertEqual(layout.graph_num_tokens, 8)
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(
                positions=torch.tensor(
                    [100, 200, 201, 202, 300, 301, 0, 1],
                    dtype=torch.int64,
                    device=device,
                ),
                ragged_verify_layout=layout,
            ),
            multimodal_inputs=[
                SimpleNamespace(mrope_position_delta=torch.tensor([[10]])),
                None,
                SimpleNamespace(mrope_position_delta=torch.tensor([[-2]])),
            ],
        )

        forward_batch.compute_spec_mrope_positions(
            SimpleNamespace(device=device), batch
        )

        expected = torch.tensor(
            [110, 200, 201, 202, 298, 299, 0, 1],
            dtype=torch.int64,
            device=device,
        )
        torch.testing.assert_close(
            forward_batch.mrope_positions, expected.unsqueeze(0).repeat(3, 1)
        )

    def test_dense_verify_layout_is_unchanged(self):
        device = torch.device("cuda")
        forward_batch = ForwardBatch.__new__(ForwardBatch)
        forward_batch.seq_lens = torch.tensor(
            [100, 200], dtype=torch.int32, device=device
        )
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(
                positions=torch.tensor(
                    [100, 101, 200, 201], dtype=torch.int64, device=device
                ),
                ragged_verify_layout=None,
            ),
            multimodal_inputs=[
                SimpleNamespace(mrope_position_delta=torch.tensor([[10]])),
                SimpleNamespace(mrope_position_delta=torch.tensor([[-2]])),
            ],
        )

        forward_batch.compute_spec_mrope_positions(
            SimpleNamespace(device=device), batch
        )

        expected = torch.tensor([110, 111, 198, 199], dtype=torch.int64, device=device)
        torch.testing.assert_close(
            forward_batch.mrope_positions, expected.unsqueeze(0).repeat(3, 1)
        )


if __name__ == "__main__":
    unittest.main()
