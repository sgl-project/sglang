"""CUDA graphs are captured at SGLANG_PLB_STREAM_PRIORITY: every kernel node of a
graph recorded on the shared capture stream and the alt stream carries it."""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.runner.early_verify_metadata import (
    EarlyVerifyMetadata,
)
from sglang.srt.model_executor.runner.metadata_glue_graph import MetadataGlueGraph
from sglang.srt.model_executor.runner_utils.pool import (
    get_or_create_global_graph_capture_stream,
)
from sglang.srt.runtime_context import get_stream, reset_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

PRIORITY = -1


def _kernel_node_priorities(graph: torch.cuda.CUDAGraph) -> list[int]:
    from cuda.bindings import driver as cu

    raw = cu.CUgraph(int(graph.raw_cuda_graph()))
    count = int(cu.cuGraphGetNodes(raw, 0)[-1])
    nodes = list(cu.cuGraphGetNodes(raw, count)[1])[:count]
    kernel = cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL
    attr = cu.CUkernelNodeAttrID.CU_LAUNCH_ATTRIBUTE_PRIORITY
    return [
        int(cu.cuGraphKernelNodeGetAttribute(node, attr)[1].priority)
        for node in nodes
        if cu.cuGraphNodeGetType(node)[1] == kernel
    ]


def _body(x: torch.Tensor, alt: torch.cuda.Stream) -> torch.Tensor:
    """Three kernels on the current stream, two on ``alt`` forked from it."""
    y = x * 2.0
    alt.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(alt):
        z = (y + 1.0).sin()
    torch.cuda.current_stream().wait_stream(alt)
    return y.cos() + z


class _GlueBackend:
    attn_backend_list = None
    requires_eager_metadata = False
    forward_metadata = None

    def __init__(self):
        self.seq_lens = torch.zeros(4, device="cuda")

    def init_forward_metadata_out_graph(self, fb_view):
        self.seq_lens.add_(1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestGraphStreamPriority(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        self.enterContext(envs.SGLANG_PLB_STREAM_PRIORITY.override(PRIORITY))

    def test_graph_kernels_on_the_capture_and_alt_streams_carry_it(self):
        capture = get_or_create_global_graph_capture_stream()
        alt = get_stream("alt")
        x = torch.randn(1 << 16, device="cuda")
        with torch.cuda.stream(capture):
            _body(x, alt)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=capture):
            out = _body(x, alt)
        self.assertEqual(_kernel_node_priorities(graph), [PRIORITY] * 5)

        graph.instantiate()
        x.copy_(torch.randn_like(x))
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, _body(x, torch.cuda.Stream())))

    def test_verify_metadata_streams_carry_it(self):
        glue = MetadataGlueGraph("cuda", strict=True)
        backend = _GlueBackend()
        for _ in range(MetadataGlueGraph.NUM_WARMUP + 1):
            glue.run(backend, fb_view=None, key="k")
        torch.cuda.synchronize()
        self.assertTrue(glue.is_captured("k"))
        self.assertEqual(glue._capture_stream.priority, PRIORITY)

        early = EarlyVerifyMetadata(device=torch.device("cuda"), log_every=1)
        self.assertEqual(early._stream.priority, PRIORITY)


if __name__ == "__main__":
    unittest.main()
