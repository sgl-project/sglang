"""FlashInfer CUDA-graph metadata: a batch size captured twice must reuse its wrappers.

Replay plans only the wrappers stored for the batch size. If a second capture of the same batch
size (another attention graph variant, or the LoRA / no-LoRA pair) created new wrappers, the first
graph would replay against wrappers that are never re-planned.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention import flashinfer_backend as fib
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _backend():
    backend = object.__new__(fib.FlashInferAttnBackend)
    backend.decode_cuda_graph_metadata = {}
    backend.prefill_cuda_graph_metadata = {}
    backend._create_decode_wrappers = lambda bs, num_tokens: [object()]
    backend._create_prefill_wrappers = lambda bs, use_custom_mask=False: [object()]
    return backend


class TestFlashInferGraphVariantWrappers(CustomTestCase):
    def test_decode_capture_twice_shares_wrappers(self):
        backend = _backend()
        backend._prepare_cuda_graph_metadata(4, 4, ForwardMode.DECODE, None)
        first = backend.forward_metadata.decode_wrappers
        backend._prepare_cuda_graph_metadata(4, 4, ForwardMode.DECODE, None)
        self.assertIs(backend.forward_metadata.decode_wrappers, first)
        self.assertIs(backend.decode_cuda_graph_metadata[4], first)
        backend._prepare_cuda_graph_metadata(8, 8, ForwardMode.DECODE, None)
        self.assertIsNot(backend.decode_cuda_graph_metadata[8], first)

    def test_verify_capture_twice_shares_wrappers(self):
        backend = _backend()
        spec = SimpleNamespace(custom_mask=None)
        backend._prepare_cuda_graph_metadata(4, 32, ForwardMode.TARGET_VERIFY, spec)
        first = backend.forward_metadata.prefill_wrappers
        backend._prepare_cuda_graph_metadata(4, 32, ForwardMode.TARGET_VERIFY, spec)
        self.assertIs(backend.forward_metadata.prefill_wrappers, first)
        self.assertIs(backend.prefill_cuda_graph_metadata[4], first)


if __name__ == "__main__":
    unittest.main()
