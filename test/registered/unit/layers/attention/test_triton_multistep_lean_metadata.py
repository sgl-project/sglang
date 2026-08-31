import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestTritonMultiStepLeanMetadata(unittest.TestCase):
    def test_capture_metadata_keeps_lean_workspaces(self):
        backend = object.__new__(TritonAttnBackend)
        backend.cuda_graph_attn_logits = object()
        backend.cuda_graph_attn_lse = object()
        backend.cuda_graph_num_kv_splits = object()
        backend.window_kv_indptr = object()
        backend.cuda_graph_swa_attn_logits = object()
        backend.cuda_graph_lean_Mp = object()
        backend.cuda_graph_lean_Lp = object()
        backend.cuda_graph_lean_Op = object()
        backend.cuda_graph_lean_locks = object()

        spec_info = SimpleNamespace(kv_indptr=object(), kv_indices=object())
        forward_batch = SimpleNamespace(
            batch_size=1,
            req_pool_indices=object(),
            seq_lens=object(),
            forward_mode=ForwardMode.DECODE,
            spec_info=spec_info,
            encoder_lens=None,
        )

        backend.init_forward_metadata_out_graph(forward_batch, in_capture=True)

        self.assertIs(backend.forward_metadata.lean_Mp, backend.cuda_graph_lean_Mp)
        self.assertIs(backend.forward_metadata.lean_Lp, backend.cuda_graph_lean_Lp)
        self.assertIs(backend.forward_metadata.lean_Op, backend.cuda_graph_lean_Op)
        self.assertIs(
            backend.forward_metadata.lean_locks, backend.cuda_graph_lean_locks
        )


if __name__ == "__main__":
    unittest.main()
