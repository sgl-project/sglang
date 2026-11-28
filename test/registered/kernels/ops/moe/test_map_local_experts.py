import unittest

import torch

from sglang.kernels.ops.moe.map_local_experts import map_topk_ids_to_local_experts
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_DEVICE = "cuda"


@unittest.skipUnless(
    torch.cuda.is_available(), "map_topk_ids_to_local_experts needs CUDA"
)
class TestMapLocalExperts(CustomTestCase):
    def test_matches_eager_and_oob_handling(self):
        torch.manual_seed(17)
        for dtype in (torch.int32, torch.int64):
            for num_experts, topk in ((72, 8), (128, 8), (288, 8)):
                mapping = torch.full(
                    (num_experts,), -1, dtype=torch.int32, device=_DEVICE
                )
                half = num_experts // 2
                mapping[half:] = torch.arange(
                    num_experts - half, dtype=torch.int32, device=_DEVICE
                )
                for m_rows in (0, 1, 16, 64):
                    with self.subTest(dtype=dtype, num_experts=num_experts, m_rows=m_rows):
                        raw = torch.randint(
                            -1, num_experts, (m_rows, topk), dtype=dtype, device=_DEVICE
                        )
                        if m_rows > 0:
                            raw[0, :5] = torch.tensor(
                                [-1, 0, half - 1, half, num_experts - 1],
                                dtype=dtype,
                                device=_DEVICE,
                            )
                        expected = torch.where(
                            raw < 0,
                            torch.tensor(-1, dtype=dtype, device=_DEVICE),
                            mapping[raw.clamp(0, num_experts - 1)].to(dtype),
                        )
                        out = map_topk_ids_to_local_experts(
                            topk_ids=raw, local_expert_mapping=mapping
                        )
                        self.assertEqual(out.dtype, dtype)
                        self.assertTrue(torch.equal(out, expected))

                        if m_rows > 0:
                            oob = raw.clone()
                            oob[0, 0] = -7
                            oob[0, 1] = num_experts
                            oob[0, 2] = num_experts + 5
                            out_oob = map_topk_ids_to_local_experts(
                                topk_ids=oob, local_expert_mapping=mapping
                            )
                            self.assertEqual(int(out_oob[0, 0].item()), -1)
                            self.assertEqual(int(out_oob[0, 1].item()), -1)
                            self.assertEqual(int(out_oob[0, 2].item()), -1)

    def test_cuda_graph_replay(self):
        num_experts, m_rows, topk = 288, 16, 8
        mapping = torch.full((num_experts,), -1, dtype=torch.int32, device=_DEVICE)
        mapping[72:144] = torch.arange(72, dtype=torch.int32, device=_DEVICE)
        topk_ids = torch.randint(
            -1, num_experts, (m_rows, topk), dtype=torch.int32, device=_DEVICE
        )
        out_buf = torch.empty_like(topk_ids)

        def step():
            out_buf.copy_(
                map_topk_ids_to_local_experts(
                    topk_ids=topk_ids, local_expert_mapping=mapping
                )
            )

        stream = torch.cuda.Stream(device=_DEVICE)
        stream.wait_stream(torch.cuda.current_stream(_DEVICE))
        with torch.cuda.stream(stream):
            for _ in range(3):
                step()
        torch.cuda.current_stream(_DEVICE).wait_stream(stream)
        torch.cuda.synchronize(_DEVICE)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            step()
        torch.cuda.synchronize(_DEVICE)

        for fill_vals in ((-1, 0, 72, 143, 144, 287), (-1, 71, 72, 100, -1, 200)):
            topk_ids[0, : len(fill_vals)] = torch.tensor(
                fill_vals, dtype=torch.int32, device=_DEVICE
            )
            graph.replay()
            torch.cuda.synchronize(_DEVICE)
            expected = torch.where(
                topk_ids < 0,
                torch.tensor(-1, dtype=torch.int32, device=_DEVICE),
                mapping[topk_ids.clamp(0, num_experts - 1)],
            )
            self.assertTrue(torch.equal(out_buf, expected))


if __name__ == "__main__":
    unittest.main()
