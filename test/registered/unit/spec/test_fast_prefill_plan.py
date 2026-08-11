"""Equivalence tests for the sync-free `fast_prefill_plan`.

`fast_prefill_plan` replaces FlashInfer's `BatchPrefillWithPagedKVCacheWrapper.plan`
in the EAGLE draft-extend CUDA graph: upstream plan() does blocking `.to("cpu")`
copies to build host scheduling metadata, while fast_prefill_plan takes that
metadata as host-known args and reaches `_cached_module.plan` with no readback.

Correctness is proven end-to-end: the same draft-extend attention, planned two
independent ways (upstream plan() vs fast_prefill_plan), must yield the SAME
`run()` output on identical q/kv. A mutation check reverses the kv_indices handed
to fast_prefill_plan and asserts the output DIVERGES, so we know the output is
sensitive to the metadata under test and the equivalence is not vacuous.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    WrapperDispatch,
    _can_use_dflash_fast_prefill_plan,
    _is_supported_dflash_fast_plan_topology,
    fast_prefill_plan,
)
from sglang.srt.speculative.spec_info import SpecInputType
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

try:
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    _HAS_FLASHINFER = True
except ImportError:
    _HAS_FLASHINFER = False

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

# Draft-extend layout: constant qo (num_tokens_per_req per req), page_size 1.
# Non-uniform seq_lens make cumsum non-trivial, so a wrong per-row kv split is
# caught instead of hidden by equal lengths.
NUM_TOKENS_PER_REQ = 8
SEQ_LENS = [37, 12, 89]
NUM_QO_HEADS = 8
NUM_KV_HEADS = 8
HEAD_DIM = 128
DTYPE = torch.float16


class TestDFlashFastPrefillPlanTopology(unittest.TestCase):
    def test_supported_wrapper_layouts(self):
        cases = (
            ("draft", None, True),
            ("draft", WrapperDispatch.SLIDING_WINDOW, True),
            ("draft", WrapperDispatch.CROSS_ATTENTION, False),
            ("target_verify", None, True),
            ("target_verify", WrapperDispatch.SLIDING_WINDOW, False),
            ("target_verify", WrapperDispatch.CROSS_ATTENTION, False),
            ("unknown", None, False),
        )
        for plan_kind, dispatch_reason, expected in cases:
            with self.subTest(
                plan_kind=plan_kind,
                dispatch_reason=dispatch_reason,
            ):
                self.assertEqual(
                    _is_supported_dflash_fast_plan_topology(plan_kind, dispatch_reason),
                    expected,
                )

    def test_compact_draft_cache_uses_original_plan(self):
        for plan_kind, dispatch_reason in (
            ("draft", None),
            ("draft", WrapperDispatch.SLIDING_WINDOW),
            ("target_verify", None),
        ):
            with self.subTest(
                plan_kind=plan_kind,
                dispatch_reason=dispatch_reason,
            ):
                self.assertFalse(
                    _is_supported_dflash_fast_plan_topology(
                        plan_kind,
                        dispatch_reason,
                        use_compact_draft_cache=True,
                    )
                )

    def test_legacy_fallback_uses_original_plan(self):
        final_host_bound = SimpleNamespace(
            spec_input_type=SpecInputType.DFLASH_VERIFY,
            host_seq_lens_include_verify_block=True,
        )
        legacy_fallback = SimpleNamespace(
            spec_input_type=SpecInputType.DFLASH_VERIFY,
            host_seq_lens_include_verify_block=False,
        )

        self.assertTrue(_can_use_dflash_fast_prefill_plan(final_host_bound, "draft"))
        self.assertFalse(_can_use_dflash_fast_prefill_plan(legacy_fallback, "draft"))


@unittest.skipUnless(_HAS_FLASHINFER, "requires flashinfer")
class TestFastPrefillPlan(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = "cuda"
        bs = len(SEQ_LENS)
        self.bs = bs
        seq_lens = torch.tensor(SEQ_LENS, dtype=torch.int32, device=self.device)

        # Device inputs in the exact layout draft-extend feeds plan().
        self.qo_indptr = torch.arange(
            0,
            (bs + 1) * NUM_TOKENS_PER_REQ,
            step=NUM_TOKENS_PER_REQ,
            dtype=torch.int32,
            device=self.device,
        )
        self.kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=self.device)
        self.kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
        self.total_kv = int(self.kv_indptr[-1].item())
        self.total_q = int(self.qo_indptr[-1].item())
        self.kv_indices = torch.arange(
            self.total_kv, dtype=torch.int32, device=self.device
        )
        self.last_page_len = torch.ones(bs, dtype=torch.int32, device=self.device)

        # Host metadata the fast path is handed (page_size==1 -> token-level).
        seq_lens_cpu = seq_lens.cpu()
        self.qo_indptr_host = self.qo_indptr.cpu()
        self.kv_indptr_host = self.kv_indptr.cpu()
        self.kv_lens_host = seq_lens_cpu
        self.max_q_len = NUM_TOKENS_PER_REQ
        self.max_kv_len = int(seq_lens_cpu.max())

        self.workspace = torch.empty(
            384 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )

        # Shared random q/kv so both code paths attend over identical data.
        self.q = torch.randn(
            self.total_q, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device=self.device
        )
        # page_size == 1 -> [num_pages, 1, num_kv_heads, head_dim] (NHD).
        self.k_cache = torch.randn(
            self.total_kv, 1, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=self.device
        )
        self.v_cache = torch.randn(
            self.total_kv, 1, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device=self.device
        )

    def _new_wrapper(self):
        bs = self.bs
        return BatchPrefillWithPagedKVCacheWrapper(
            self.workspace,
            "NHD",
            use_cuda_graph=True,
            backend="fa2",
            qo_indptr_buf=torch.zeros(bs + 1, dtype=torch.int32, device=self.device),
            paged_kv_indptr_buf=torch.zeros(
                bs + 1, dtype=torch.int32, device=self.device
            ),
            paged_kv_indices_buf=torch.zeros(
                self.total_kv, dtype=torch.int32, device=self.device
            ),
            paged_kv_last_page_len_buf=torch.ones(
                bs, dtype=torch.int32, device=self.device
            ),
        )

    def _real_plan(self, w):
        w.plan(
            self.qo_indptr,
            self.kv_indptr,
            self.kv_indices,
            self.last_page_len,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            1,  # page_size
            causal=True,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
        )

    def _forward(self, w):
        return w.run(self.q, (self.k_cache, self.v_cache))

    def _out_upstream(self):
        """Ground truth: a wrapper planned only by upstream plan()."""
        w = self._new_wrapper()
        self._real_plan(w)
        return self._forward(w)

    def _out_fast(
        self,
        *,
        kv_indices=None,
        kv_indptr_host=None,
        kv_lens_host=None,
        max_kv_len=None,
    ):
        """Same attention, planned via the host-known fast path. One real plan()
        first populates `_cached_module` (mirrors capture), then fast_prefill_plan
        re-plans from host metadata."""
        if kv_indices is None:
            kv_indices = self.kv_indices
        if kv_indptr_host is None:
            kv_indptr_host = self.kv_indptr_host
        if kv_lens_host is None:
            kv_lens_host = self.kv_lens_host
        if max_kv_len is None:
            max_kv_len = self.max_kv_len
        w = self._new_wrapper()
        self._real_plan(w)
        fast_prefill_plan(
            w,
            self.qo_indptr,
            self.kv_indptr,
            kv_indices,
            self.last_page_len,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            1,
            causal=True,
            q_data_type=DTYPE,
            kv_data_type=DTYPE,
            qo_indptr_host=self.qo_indptr_host,
            kv_indptr_host=kv_indptr_host,
            kv_lens_host=kv_lens_host,
            max_q_len=self.max_q_len,
            max_kv_len=max_kv_len,
        )
        return self._forward(w)

    def test_fast_plan_matches_upstream(self):
        # Two genuinely independent plan paths over identical q/kv must produce
        # the same attention output.
        out_upstream = self._out_upstream()
        out_fast = self._out_fast()
        torch.testing.assert_close(out_fast, out_upstream, rtol=0, atol=0)

    def test_dflash_full_attention_conservative_bound_matches_upstream(self):
        # DFlash overlap may expose a host planning bound that is larger than
        # the exact device layout. The host values only schedule work; the
        # device indptr/indices remain authoritative for attention addressing.
        kv_lens_host = self.kv_lens_host + NUM_TOKENS_PER_REQ
        kv_indptr_host = torch.zeros(self.bs + 1, dtype=torch.int32, device="cpu")
        kv_indptr_host[1:] = torch.cumsum(kv_lens_host, dim=0)

        out_upstream = self._out_upstream()
        out_fast = self._out_fast(
            kv_indptr_host=kv_indptr_host,
            kv_lens_host=kv_lens_host,
            max_kv_len=int(kv_lens_host.max()),
        )
        torch.testing.assert_close(out_fast, out_upstream, rtol=0, atol=0)

    def test_mutation_changes_output(self):
        """Guards against a vacuous test: the kv_indices fast_prefill_plan installs
        select which physical KV slots the kernel gathers, so reversing them must
        change the attention output. If it did not, the equivalence assertion
        would not be exercising the metadata fast_prefill_plan is responsible for."""
        out_upstream = self._out_upstream()
        reversed_kv = torch.flip(self.kv_indices, dims=[0]).contiguous()
        out_wrong = self._out_fast(kv_indices=reversed_kv)
        self.assertFalse(
            torch.allclose(out_wrong, out_upstream, rtol=1e-3, atol=1e-3),
            "output unchanged under reversed kv_indices; test lacks discriminating power",
        )


if __name__ == "__main__":
    unittest.main()
