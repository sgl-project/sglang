"""Structural contract for the trtllm_mha backend's class hierarchy (#28808).

``TRTLLMHAAttnBackend`` used to inherit from ``FlashInferAttnBackend`` and
``TRTLLMHAAttnMultiStepDraftBackend`` from ``FlashInferMultiStepDraftBackend``.
Every attention-specific behavior is TRTLLM's own (metadata builders, page
tables, forward paths), so the inherited FlashInfer wrappers / updaters /
workspace plumbing was allocated and then never used. These tests pin the
post-refactor contract so the coupling cannot silently return, and keep the
behavioral flags that external code reads off these backends.
"""

import inspect
import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    FlashInferMultiStepDraftBackend,
)
from sglang.srt.layers.attention.trtllm_mha_backend import (
    TRTLLMHAAttnBackend,
    TRTLLMHAAttnMultiStepDraftBackend,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-large")

# Runtime surface the backend must own itself (it previously inherited
# FlashInfer implementations that only worked by accident of sharing the
# parent's state).
OWNED_BY_TRM_MHA = (
    "shared_read_ends",
    "init_forward_metadata",
    "init_forward_metadata_out_graph",
    "init_cuda_graph_state",
    "get_cuda_graph_seq_len_fill_value",
    "forward_decode",
    "forward_extend",
    # The one helper that used to come from FlashInferAttnBackend.
    "_kv_write_scales",
)


class TestTRTLLMHAAttnBackendInheritance(unittest.TestCase):
    def test_backend_inherits_base_not_flashinfer(self):
        self.assertTrue(issubclass(TRTLLMHAAttnBackend, AttentionBackend))
        self.assertFalse(issubclass(TRTLLMHAAttnBackend, FlashInferAttnBackend))

    def test_backend_owns_its_runtime_surface(self):
        for name in OWNED_BY_TRM_MHA:
            with self.subTest(method=name):
                self.assertIn(name, TRTLLMHAAttnBackend.__dict__)

    def test_constructor_signature(self):
        params = inspect.signature(TRTLLMHAAttnBackend.__init__).parameters
        self.assertEqual(
            list(params),
            ["self", "model_runner", "skip_prefill", "speculative_step_id"],
        )

    def test_behavioral_flags_preserved(self):
        # External readers (flashinfer_autotune, TBO / hybrid / minimax
        # wrappers via getattr, overlap scheduler) depend on these values;
        # they must not flip when the base class changes.
        self.assertIs(TRTLLMHAAttnBackend.needs_cpu_seq_lens, False)
        self.assertIs(TRTLLMHAAttnBackend.supports_ragged_verify_graph, True)
        self.assertIs(TRTLLMHAAttnBackend.extend_dummy_seqs_capped_by_req_pool, True)

    def test_kv_write_scales_dispatches_on_global_scale(self):
        def make_backend(needs_global_scale):
            backend = TRTLLMHAAttnBackend.__new__(TRTLLMHAAttnBackend)
            backend.kv_cache_quant_method = SimpleNamespace(
                needs_global_scale=lambda: needs_global_scale
            )
            return backend

        layer = SimpleNamespace(k_scale=0.5, v_scale=0.25)
        self.assertEqual(
            make_backend(False)._kv_write_scales(layer), (layer.k_scale, layer.v_scale)
        )
        self.assertEqual(make_backend(True)._kv_write_scales(layer), (None, None))


class TestTRTLLMHAAttnMultiStepDraftBackendInheritance(unittest.TestCase):
    def test_draft_backend_is_standalone(self):
        self.assertFalse(
            issubclass(
                TRTLLMHAAttnMultiStepDraftBackend, FlashInferMultiStepDraftBackend
            )
        )

    def test_constructor_signature(self):
        params = inspect.signature(
            TRTLLMHAAttnMultiStepDraftBackend.__init__
        ).parameters
        self.assertEqual(
            list(params),
            ["self", "model_runner", "topk", "speculative_num_steps"],
        )

    def test_behavioral_flag_preserved(self):
        self.assertIs(TRTLLMHAAttnMultiStepDraftBackend.needs_cpu_seq_lens, False)


if __name__ == "__main__":
    unittest.main()
