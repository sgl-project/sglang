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


def _new(cls, **attrs):
    obj = object.__new__(cls)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _make_worker(attn_backend, *, draft_token_num: int = 4, is_cpu_flag: bool = False):
    runner = SimpleNamespace(attn_backend=attn_backend, watermark=None)
    worker = _new(
        NGRAMWorker,
        _target_worker=SimpleNamespace(model_runner=runner),
        model_runner=runner,
        device=torch.device("cpu"),
        max_batch_size=8,
        draft_token_num=draft_token_num,
        max_trie_depth=8,
        speculative_num_draft_tokens=draft_token_num,
        _prev_decode_rids=set(),
        grammar_tree_host=None,
        ngram_corpus=NgramCorpus(
            max_trie_depth=8,
            min_bfs_breadth=1,
            max_bfs_breadth=4,
            draft_token_num=draft_token_num,
            capacity=1000,
        ),
    )
    worker._init_preallocated_tensors()
    with (
        patch.object(ngram_worker_mod, "_is_cpu", is_cpu_flag),
        patch.object(
            ngram_worker_mod,
            "bind_eagle_runtime_hooks",
            return_value=ngram_worker_mod.STOCK_SPECULATIVE_RUNTIME_HOOKS,
        ),
    ):
        worker.init_attention_backends()
    return worker


class TestNgramQlenMask(CustomTestCase):
    def test_backend_capability_and_worker_init(self):
        fa = _new(FlashAttentionBackend)
        fi = _new(FlashInferAttnBackend)
        triton = _new(TritonAttnBackend)
        lin = _new(AttentionBackend)
        hyb_prefill_fa = _new(
            HybridAttnBackend,
            prefill_backend=fa,
            decode_backend=fi,
            spec_attn_is_decode=False,
            spec_attn_is_prefill=True,
        )
        hyb_decode_fi = _new(
            HybridAttnBackend,
            prefill_backend=fa,
            decode_backend=fi,
            spec_attn_is_decode=True,
            spec_attn_is_prefill=False,
        )
        hyb_lin_fa = _new(
            HybridLinearAttnBackend, full_attn_backend=fa, linear_attn_backend=lin
        )
        hyb_lin_fi = _new(
            HybridLinearAttnBackend, full_attn_backend=fi, linear_attn_backend=lin
        )
        # TBO splits custom_mask using FULL_MASK offsets and must stay False.
        tbo_fa = _new(TboAttnBackend, primary=fa, children=[fa, fa])

        for backend, expected_qlen in (
            (fa, True),
            (fi, False),
            (triton, False),
            (hyb_prefill_fa, True),
            (hyb_decode_fi, False),
            (hyb_lin_fa, True),
            (hyb_lin_fi, False),
            (tbo_fa, False),
        ):
            self.assertEqual(backend.supports_qlen_only_verify_mask, expected_qlen)
            self.assertEqual(_make_worker(backend).use_full_mask, not expected_qlen)

        self.assertTrue(_make_worker(None).use_full_mask)
        self.assertFalse(_make_worker(fi, is_cpu_flag=True).use_full_mask)

    def test_prepare_qlen_vs_full_mask_parity(self):
        d, seq_lens = 4, [5, 11]
        corpus = [
            [10, 11, 12, 13, 14, 15, 16, 17],
            [10, 11, 12, 23, 24, 25],
            [30, 31, 32, 33, 34, 35, 36, 37],
        ]

        def run(backend, *, pass_seq_lens_cpu: bool):
            worker = _make_worker(backend, draft_token_num=d)
            worker.ngram_corpus.batch_put(corpus)
            worker.ngram_corpus.synchronize()
            reqs = [
                Req("r0", "", array("q", [1, 2, 10, 11, 12]), SamplingParams()),
                Req(
                    "r1",
                    "",
                    array("q", [1, 2, 3, 4, 5, 6, 7, 8, 30, 31, 32]),
                    SamplingParams(),
                ),
            ]
            for r in reqs:
                r.inflight_steps = 1
            batch = SimpleNamespace(
                forward_mode=ForwardMode.DECODE,
                reqs=reqs,
                seq_lens=torch.tensor(seq_lens, dtype=torch.int64),
                seq_lens_cpu=(
                    torch.tensor(seq_lens, dtype=torch.int64)
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

        # QLEN_ONLY path must work with seq_lens_cpu=None.
        qlen_info = run(_new(FlashAttentionBackend), pass_seq_lens_cpu=False)
        full_info = run(_new(FlashInferAttnBackend), pass_seq_lens_cpu=True)

        self.assertTrue(qlen_info.custom_mask_is_qlen_only)
        self.assertFalse(full_info.custom_mask_is_qlen_only)
        self.assertEqual(qlen_info.custom_mask.numel(), 2 * d * d)

        offset = 0
        qlen_3d = qlen_info.custom_mask.view(2, d, d)
        for i, sl in enumerate(seq_lens):
            span = d * (sl + d)
            full_2d = full_info.custom_mask[offset : offset + span].view(d, sl + d)
            self.assertTrue(torch.all(full_2d[:, :sl]))
            self.assertTrue(torch.equal(full_2d[:, sl:], qlen_3d[i]))
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

    def test_watermark_full_mask_matches_mask_layout(self):
        d = 4
        for backend, expected in (
            (_new(FlashAttentionBackend), False),
            (_new(FlashInferAttnBackend), True),
        ):
            worker = _make_worker(backend, draft_token_num=d)
            worker.token_to_kv_pool_allocator = None
            worker.prev_token_ids = []
            worker.prev_accept_lens = [0]
            worker._target_worker.forward_batch_generation = lambda *a, **kw: (
                SimpleNamespace(
                    logits_output=SimpleNamespace(next_token_logits=torch.zeros(d, 8)),
                    can_run_cuda_graph=False,
                )
            )
            req = Req("r0", "", array("q", [1, 2, 3, 4]), SamplingParams())
            req.inflight_steps = 1
            batch = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                reqs=[req],
                seq_lens=torch.tensor([4], dtype=torch.int64),
                has_grammar=False,
                spec_info=NgramVerifyInput(
                    draft_token=torch.zeros(d, dtype=torch.int64),
                    custom_mask=torch.ones(d * d, dtype=torch.bool),
                    positions=torch.arange(d, dtype=torch.int64),
                    retrieve_index=torch.arange(d, dtype=torch.int64).view(1, d),
                    retrieve_next_token=torch.full((1, d), -1, dtype=torch.int64),
                    retrieve_next_sibling=torch.full((1, d), -1, dtype=torch.int64),
                    draft_token_num=d,
                    custom_mask_is_qlen_only=not worker.use_full_mask,
                ),
            )
            seen = []
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
                    side_effect=lambda *a, **kw: (
                        seen.append(kw["watermark_full_mask"]),
                        SimpleNamespace(commit=lambda *x, **y: None),
                    )[1],
                ),
                patch.object(
                    ngram_worker_mod, "accept_draft_tree", return_value=fake_accept
                ),
                patch.object(ngram_worker_mod, "commit_mamba_states_after_verify"),
                patch.object(ngram_worker_mod, "move_accept_tokens_to_target_kvcache"),
                patch.object(worker, "_update_ngram_corpus"),
            ):
                worker.forward_batch_generation(batch)
            self.assertEqual(seen, [expected])


if __name__ == "__main__":
    unittest.main()
