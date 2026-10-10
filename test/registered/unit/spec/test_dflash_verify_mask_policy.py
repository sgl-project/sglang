"""Tests for DFLASH target-verify custom-mask handling.

- resolve_dflash_verify_mask_policy resolves wrapper backends to the one that
  runs TARGET_VERIFY.
- FlashInfer rejects a mask-less verify plan on a cuda-graph wrapper captured
  with a mask buffer.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import resolve_dflash_verify_mask_policy
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

try:
    from sglang.srt.layers.attention.flashinfer_backend import (
        FlashInferIndicesUpdaterPrefill,
    )

    _HAS_FLASHINFER_BACKEND = True
except ImportError:
    _HAS_FLASHINFER_BACKEND = False

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _leaf(name: str):
    """Stand-in leaf backend; the policy keys leaves by class name only."""
    cls = type(name, (), {})
    backend = cls()
    backend.needs_cpu_seq_lens = True
    backend.token_to_kv_pool = None
    backend.req_to_token_pool = None
    backend.kv_index_translator = None
    return backend


def _split(prefill, decode, mode: str) -> HybridAttnBackend:
    model_runner = SimpleNamespace(
        kv_cache_dtype=torch.float8_e4m3fn,
        token_to_kv_pool=None,
        req_to_token_pool=None,
        kv_index_translator=None,
        model_config=SimpleNamespace(context_len=4096),
    )
    spec = SimpleNamespace(speculative_attention_mode=mode)
    with patch(
        "sglang.srt.layers.attention.hybrid_attn_backend.get_spec", return_value=spec
    ):
        return HybridAttnBackend(
            model_runner=model_runner, prefill_backend=prefill, decode_backend=decode
        )


def _linear_hybrid(full):
    """Stand-in for HybridLinearAttnBackend (unwrapped via full_attn_backend)."""
    return SimpleNamespace(
        full_attn_backend=full,
        token_to_kv_pool=None,
        req_to_token_pool=None,
        kv_index_translator=None,
    )


def _tbo(primary) -> TboAttnBackend:
    return TboAttnBackend(primary=primary, children=[primary, primary])


class TestDFlashVerifyMaskPolicy(CustomTestCase):
    def setUp(self):
        self.flashinfer = _leaf("FlashInferAttnBackend")
        self.trtllm_mha = _leaf("TRTLLMHAAttnBackend")
        self.triton = _leaf("TritonAttnBackend")
        self.unlisted = _leaf("UnlistedAttnBackend")

    def assertPolicy(self, backend, name: str, build_custom_mask: bool):
        self.assertEqual(
            resolve_dflash_verify_mask_policy(backend), (name, build_custom_mask)
        )

    def test_single_backend(self):
        self.assertPolicy(self.flashinfer, "FlashInferAttnBackend", False)
        self.assertPolicy(
            _linear_hybrid(self.flashinfer), "FlashInferAttnBackend", False
        )

    def test_split_prefill_mode_judges_prefill_child(self):
        # The regression: FlashInfer serves verify under prefill mode.
        backend = _linear_hybrid(_split(self.flashinfer, self.trtllm_mha, "prefill"))
        self.assertPolicy(backend, "FlashInferAttnBackend", False)

        backend = _linear_hybrid(_split(self.triton, self.trtllm_mha, "prefill"))
        self.assertPolicy(backend, "TritonAttnBackend", False)

    def test_split_decode_mode_judges_decode_child(self):
        backend = _linear_hybrid(_split(self.flashinfer, self.trtllm_mha, "decode"))
        self.assertPolicy(backend, "TRTLLMHAAttnBackend", False)

    def test_split_ignores_child_that_does_not_serve_verify(self):
        # A "both children listed" rule would attach a mask here.
        backend = _split(self.flashinfer, self.unlisted, "prefill")
        self.assertPolicy(backend, "FlashInferAttnBackend", False)
        # An "either child listed" rule would skip the mask here.
        backend = _split(self.unlisted, self.trtllm_mha, "prefill")
        self.assertPolicy(backend, "UnlistedAttnBackend", True)

    def test_two_batch_overlap_wrapper(self):
        self.assertPolicy(_tbo(self.flashinfer), "FlashInferAttnBackend", False)
        backend = _tbo(
            _linear_hybrid(_split(self.flashinfer, self.trtllm_mha, "prefill"))
        )
        self.assertPolicy(backend, "FlashInferAttnBackend", False)


@unittest.skipUnless(_HAS_FLASHINFER_BACKEND, "requires flashinfer backend import")
class TestFlashInferVerifyMaskGuard(CustomTestCase):
    BS = 2
    DRAFT_TOKENS = 4

    def _updater(self):
        updater = FlashInferIndicesUpdaterPrefill.__new__(
            FlashInferIndicesUpdaterPrefill
        )
        updater.attn_backend = SimpleNamespace(
            kv_index_translator=SimpleNamespace(reads_are_translated=False)
        )
        updater.req_to_token = torch.zeros((self.BS, 64), dtype=torch.int32)
        updater.kv_last_page_len = torch.ones(self.BS, dtype=torch.int32)
        updater.num_qo_heads = updater.num_kv_heads = 1
        updater.head_dim = 64
        updater.q_data_type = updater.data_type = torch.float16
        return updater

    def _spec_info(self, custom_mask):
        qo_indptr = torch.arange(
            0, (self.BS + 1) * self.DRAFT_TOKENS, self.DRAFT_TOKENS, dtype=torch.int32
        )
        kv_indptr = torch.tensor([0, 10, 20], dtype=torch.int32)
        kv_indices = torch.arange(20, dtype=torch.int32)
        # spec= keeps the updater's isinstance(spec_info, SpecInput) check happy
        # without running the real (GPU) kv-index kernel.
        spec_info = MagicMock(spec=DFlashVerifyInput)
        spec_info.spec_input_type = SpecInputType.DFLASH_VERIFY
        spec_info.num_tokens_per_req = self.DRAFT_TOKENS
        spec_info.generate_attn_arg_prefill.return_value = (
            kv_indices,
            kv_indptr,
            qo_indptr,
            custom_mask,
        )
        return spec_info

    def _wrapper(self, *, cuda_graph: bool, has_mask_buf: bool):
        wrapper = MagicMock()
        wrapper.is_cuda_graph_enabled = cuda_graph
        wrapper._custom_mask_buf = (
            torch.zeros(16, dtype=torch.uint8) if has_mask_buf else None
        )
        return wrapper

    def _plan(self, wrapper, custom_mask):
        seq_lens = torch.tensor([6, 6], dtype=torch.int32)
        self._updater().call_begin_forward(
            wrapper_ragged=None,
            wrapper_paged=wrapper,
            req_pool_indices=torch.arange(self.BS, dtype=torch.int32),
            paged_kernel_lens=seq_lens,
            paged_kernel_lens_sum=int(seq_lens.sum()),
            seq_lens=seq_lens,
            prefix_lens=None,
            kv_start_idx=None,
            kv_indptr=torch.zeros(self.BS + 1, dtype=torch.int32),
            qo_indptr=torch.zeros(self.BS + 1, dtype=torch.int32),
            use_ragged=False,
            spec_info=self._spec_info(custom_mask),
        )

    def test_rejects_maskless_plan_on_masked_graph_wrapper(self):
        wrapper = self._wrapper(cuda_graph=True, has_mask_buf=True)
        with self.assertRaisesRegex(RuntimeError, "capture and replay must agree"):
            self._plan(wrapper, custom_mask=None)
        wrapper.begin_forward.assert_not_called()

    def test_allows_consistent_plans(self):
        cases = {
            "graph wrapper captured without a mask": (True, False, None),
            "graph wrapper replayed with a real mask": (
                True,
                True,
                torch.ones(8, dtype=torch.bool),
            ),
            "eager wrapper (plan resets the mask buffer)": (False, True, None),
        }
        for label, (cuda_graph, has_mask_buf, mask) in cases.items():
            with self.subTest(label):
                wrapper = self._wrapper(
                    cuda_graph=cuda_graph, has_mask_buf=has_mask_buf
                )
                self._plan(wrapper, custom_mask=mask)
                wrapper.begin_forward.assert_called_once()


if __name__ == "__main__":
    unittest.main()
