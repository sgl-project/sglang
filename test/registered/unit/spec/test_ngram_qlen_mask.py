"""Unit tests for QLEN_ONLY verify-mask dispatch in NGRAMWorker and AttentionBackend."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative import ngram_worker as ngram_worker_mod
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.ngram_worker import NGRAMWorker


def _make_hybrid_attn(prefill_backend, decode_backend, *, spec_attn_is_decode: bool):
    backend = object.__new__(HybridAttnBackend)
    backend.prefill_backend = prefill_backend
    backend.decode_backend = decode_backend
    backend.spec_attn_is_decode = spec_attn_is_decode
    backend.spec_attn_is_prefill = not spec_attn_is_decode
    return backend


def _make_hybrid_linear(full_attn_backend, linear_attn_backend=None):
    backend = object.__new__(HybridLinearAttnBackend)
    backend.full_attn_backend = full_attn_backend
    backend.linear_attn_backend = linear_attn_backend or object.__new__(
        AttentionBackend
    )
    return backend


def _make_tbo(primary):
    backend = object.__new__(TboAttnBackend)
    backend.primary = primary
    backend.children = [primary, primary]
    return backend


def _make_ngram_worker(attn_backend, *, draft_token_num: int = 4, max_bs: int = 8):
    worker = object.__new__(NGRAMWorker)
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(attn_backend=attn_backend, watermark=None)
    )
    worker.model_runner = worker._target_worker.model_runner
    worker.device = torch.device("cpu")
    worker.max_batch_size = max_bs
    worker.draft_token_num = draft_token_num
    worker.max_trie_depth = 8
    worker.speculative_num_draft_tokens = draft_token_num
    worker._prev_decode_rids = set()
    worker.grammar_tree_host = None
    worker.ngram_corpus = NgramCorpus(
        max_trie_depth=8,
        min_bfs_breadth=1,
        max_bfs_breadth=4,
        draft_token_num=draft_token_num,
        capacity=1000,
    )
    worker._init_preallocated_tensors()
    return worker


class TestNgramQlenMask(CustomTestCase):
    def test_attention_backend_capability_and_wrappers(self):
        fa = object.__new__(FlashAttentionBackend)
        fi = object.__new__(FlashInferAttnBackend)
        triton = object.__new__(TritonAttnBackend)

        self.assertTrue(fa.supports_qlen_only_verify_mask)
        self.assertFalse(fi.supports_qlen_only_verify_mask)
        self.assertFalse(triton.supports_qlen_only_verify_mask)

        self.assertTrue(
            _make_hybrid_attn(
                fa, fi, spec_attn_is_decode=False
            ).supports_qlen_only_verify_mask
        )
        self.assertFalse(
            _make_hybrid_attn(
                fa, fi, spec_attn_is_decode=True
            ).supports_qlen_only_verify_mask
        )
        self.assertTrue(
            _make_hybrid_attn(
                fi, fa, spec_attn_is_decode=True
            ).supports_qlen_only_verify_mask
        )

        self.assertTrue(_make_hybrid_linear(fa).supports_qlen_only_verify_mask)
        self.assertFalse(_make_hybrid_linear(fi).supports_qlen_only_verify_mask)
        self.assertTrue(
            _make_hybrid_linear(
                _make_hybrid_attn(fa, fi, spec_attn_is_decode=False)
            ).supports_qlen_only_verify_mask
        )
        self.assertFalse(
            _make_hybrid_linear(
                _make_hybrid_attn(fa, fi, spec_attn_is_decode=True)
            ).supports_qlen_only_verify_mask
        )

        # TBO splits custom_mask using FULL_MASK offsets and must not inherit
        # True from its primary FlashAttentionBackend via __getattr__.
        self.assertFalse(_make_tbo(fa).supports_qlen_only_verify_mask)

    def test_init_attention_backends_resolves_use_full_mask(self):
        fa = object.__new__(FlashAttentionBackend)
        fi = object.__new__(FlashInferAttnBackend)
        cases = [
            (fa, False, False),
            (_make_hybrid_attn(fa, fi, spec_attn_is_decode=False), False, False),
            (_make_hybrid_linear(fa), False, False),
            (fi, False, True),
            (_make_hybrid_attn(fa, fi, spec_attn_is_decode=True), False, True),
            (_make_tbo(fa), False, True),
            (None, False, True),
            (fi, True, False),
        ]
        for backend, is_cpu_flag, expected_use_full_mask in cases:
            with self.subTest(backend=type(backend).__name__, is_cpu_flag=is_cpu_flag):
                worker = _make_ngram_worker(backend)
                with (
                    patch.object(ngram_worker_mod, "_is_cpu", is_cpu_flag),
                    patch.object(
                        ngram_worker_mod,
                        "bind_eagle_runtime_hooks",
                        return_value=ngram_worker_mod.STOCK_SPECULATIVE_RUNTIME_HOOKS,
                    ),
                ):
                    worker.init_attention_backends()
                self.assertEqual(worker.use_full_mask, expected_use_full_mask)

    def test_prepare_for_speculative_decoding_qlen_vs_full_mask_parity(self):
        d = 4
        seq_lens_list = [5, 11]
        corpus_seqs = [
            [10, 11, 12, 13, 14, 15, 16, 17],
            [10, 11, 12, 23, 24, 25],
            [30, 31, 32, 33, 34, 35, 36, 37],
        ]

        def run_prepare(attn_backend, *, pass_seq_lens_cpu: bool):
            worker = _make_ngram_worker(attn_backend, draft_token_num=d)
            worker.ngram_corpus.batch_put(corpus_seqs)
            worker.ngram_corpus.synchronize()
            with (
                patch.object(ngram_worker_mod, "_is_cpu", False),
                patch.object(
                    ngram_worker_mod,
                    "bind_eagle_runtime_hooks",
                    return_value=ngram_worker_mod.STOCK_SPECULATIVE_RUNTIME_HOOKS,
                ),
            ):
                worker.init_attention_backends()

            req0 = Req("r0", "", array("q", [1, 2, 10, 11, 12]), SamplingParams())
            req1 = Req(
                "r1",
                "",
                array("q", [1, 2, 3, 4, 5, 6, 7, 8, 30, 31, 32]),
                SamplingParams(),
            )
            req0.inflight_steps = 1
            req1.inflight_steps = 1

            batch = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                reqs=[req0, req1],
                seq_lens=torch.tensor(seq_lens_list, dtype=torch.int64),
                seq_lens_cpu=(
                    torch.tensor(seq_lens_list, dtype=torch.int64)
                    if pass_seq_lens_cpu
                    else None
                ),
                req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
                req_to_token_pool=SimpleNamespace(
                    req_to_token=torch.zeros((4, 32), dtype=torch.int32)
                ),
                has_grammar=False,
                spec_info=None,
            )
            with (
                patch.object(
                    ngram_worker_mod, "reconstruct_indices_from_tree_mask_func"
                ),
                patch.object(
                    ngram_worker_mod,
                    "assign_extend_cache_locs_func",
                    return_value=torch.zeros(2 * d, dtype=torch.int64),
                ),
                patch.object(ngram_worker_mod, "prepare_mamba_track_for_verify"),
            ):
                worker._prepare_for_speculative_decoding(batch)
            return batch.spec_info

        # QLEN_ONLY path (FlashAttentionBackend) must not touch seq_lens_cpu.
        qlen_info = run_prepare(
            object.__new__(FlashAttentionBackend), pass_seq_lens_cpu=False
        )
        full_info = run_prepare(
            object.__new__(FlashInferAttnBackend), pass_seq_lens_cpu=True
        )

        self.assertTrue(qlen_info.custom_mask_is_qlen_only)
        self.assertFalse(full_info.custom_mask_is_qlen_only)
        self.assertEqual(qlen_info.custom_mask.numel(), 2 * d * d)
        self.assertEqual(
            full_info.custom_mask.numel(),
            sum((sl + d) * d for sl in seq_lens_list),
        )

        # Verify bit-for-bit parity between QLEN_ONLY rows and FULL_MASK trailing suffixes.
        offset = 0
        qlen_3d = qlen_info.custom_mask.view(2, d, d)
        for idx, sl in enumerate(seq_lens_list):
            span = d * (sl + d)
            full_2d = full_info.custom_mask[offset : offset + span].view(d, sl + d)
            self.assertTrue(torch.all(full_2d[:, :sl]))
            self.assertTrue(torch.equal(full_2d[:, sl:], qlen_3d[idx]))
            offset += span

    def test_generate_attn_arg_prefill_rejects_qlen_only(self):
        info = NgramVerifyInput(
            custom_mask=torch.ones(16, dtype=torch.bool),
            draft_token_num=4,
            custom_mask_is_qlen_only=True,
        )
        with self.assertRaises(AssertionError):
            info.generate_attn_arg_prefill(
                req_pool_indices=torch.tensor([0]),
                paged_kernel_lens=torch.tensor([4]),
                paged_kernel_lens_sum=4,
                runner_ids=None,
                plan=None,
            )

    def test_forward_batch_generation_passes_matching_watermark_full_mask(self):
        d = 4
        for backend, expected_watermark_full_mask in (
            (object.__new__(FlashAttentionBackend), False),
            (object.__new__(FlashInferAttnBackend), True),
        ):
            with self.subTest(backend=type(backend).__name__):
                worker = _make_ngram_worker(backend, draft_token_num=d)
                worker.token_to_kv_pool_allocator = None
                worker.prev_token_ids = []
                worker.prev_accept_lens = [0]
                with (
                    patch.object(ngram_worker_mod, "_is_cpu", False),
                    patch.object(
                        ngram_worker_mod,
                        "bind_eagle_runtime_hooks",
                        return_value=ngram_worker_mod.STOCK_SPECULATIVE_RUNTIME_HOOKS,
                    ),
                ):
                    worker.init_attention_backends()

                req = Req("r0", "", array("q", [1, 2, 3, 4]), SamplingParams())
                req.inflight_steps = 1
                verify_input = NgramVerifyInput(
                    draft_token=torch.zeros(d, dtype=torch.int64),
                    custom_mask=torch.ones(d * d, dtype=torch.bool),
                    positions=torch.arange(d, dtype=torch.int64),
                    retrieve_index=torch.arange(d, dtype=torch.int64).view(1, d),
                    retrieve_next_token=torch.full((1, d), -1, dtype=torch.int64),
                    retrieve_next_sibling=torch.full((1, d), -1, dtype=torch.int64),
                    draft_token_num=d,
                    custom_mask_is_qlen_only=not worker.use_full_mask,
                )
                batch = SimpleNamespace(
                    forward_mode=ForwardMode.TARGET_VERIFY,
                    reqs=[req],
                    seq_lens=torch.tensor([4], dtype=torch.int64),
                    has_grammar=False,
                    spec_info=verify_input,
                )

                worker._target_worker.forward_batch_generation = (
                    lambda *args, **kwargs: SimpleNamespace(
                        logits_output=SimpleNamespace(
                            next_token_logits=torch.zeros(d, 8)
                        ),
                        can_run_cuda_graph=False,
                    )
                )

                seen_watermark_full_mask = []

                def fake_prepare(*args, **kwargs):
                    seen_watermark_full_mask.append(kwargs["watermark_full_mask"])
                    return SimpleNamespace(commit=lambda *a, **kw: None)

                fake_accept = SimpleNamespace(
                    tokens=torch.zeros(d, dtype=torch.int64),
                    accept_lens=torch.ones(1, dtype=torch.int32),
                    accept_index=torch.zeros((1, d), dtype=torch.int64),
                )

                with (
                    patch.object(ngram_worker_mod, "record_stream_for_v2_verify"),
                    patch.object(ngram_worker_mod, "set_time_batch"),
                    patch.object(worker, "_prepare_for_speculative_decoding"),
                    patch.object(
                        ngram_worker_mod.TargetVerify,
                        "prepare",
                        side_effect=fake_prepare,
                    ),
                    patch.object(
                        ngram_worker_mod, "accept_draft_tree", return_value=fake_accept
                    ),
                    patch.object(ngram_worker_mod, "commit_mamba_states_after_verify"),
                    patch.object(
                        ngram_worker_mod, "move_accept_tokens_to_target_kvcache"
                    ),
                    patch.object(worker, "_update_ngram_corpus"),
                ):
                    worker.forward_batch_generation(batch)

                self.assertEqual(
                    seen_watermark_full_mask, [expected_watermark_full_mask]
                )


if __name__ == "__main__":
    unittest.main()
