# Alternating graph shapes must update the shared auxiliary output correctly.
import unittest

import torch

from sglang.srt.model_executor.shared_aux_hidden import SharedAuxHiddenBuffers
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestSharedAuxGraph(CustomTestCase):
    def test_replay_alternating_shapes(self):
        pool = SharedAuxHiddenBuffers()
        outputs, graphs = {}, {}
        source = torch.arange(64 * 12, device="cuda", dtype=torch.float32).reshape(
            64, 12
        )
        torch.cuda.synchronize()
        for rows in (64, 8, 32):
            out = pool.get(
                0, rows=rows, max_rows=64, width=12, dtype=source.dtype, device="cuda"
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out.copy_(source[:rows])
                out.add_(rows)
            outputs[rows], graphs[rows] = out, graph
        for rows in (8, 64, 32, 8):
            graphs[rows].replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(outputs[rows], source[:rows] + rows)


if __name__ == "__main__":
    unittest.main()
