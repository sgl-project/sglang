"""CPU coverage for running the draft extend after a prefill as a KV-only pass
over every token but each request's last, plus a pass over the last tokens."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.speculative.eagle_utils import (
    _MAX_UNSPLIT_PREFILL_DRAFT_EXTEND_TOKENS,
    should_split_prefill_draft_extend,
    split_prefill_draft_extend,
)
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _prefill_batch(prefix_lens, extend_lens, *, mlp_sync=False, multimodal=False):
    num_tokens = sum(extend_lens)
    seq_lens = [p + n for p, n in zip(prefix_lens, extend_lens)]
    # Without attention DP the MLP sync lists hold this rank's counts.
    sync = [num_tokens] if mlp_sync else None
    sync_for_logprob = [len(extend_lens)] if mlp_sync else None
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        input_embeds=None,
        replace_embeds=None,
        tbo_split_seq_index=None,
        reqs=[
            SimpleNamespace(multimodal_inputs=object() if multimodal else None)
            for _ in extend_lens
        ],
        input_ids=torch.arange(100, 100 + num_tokens),
        out_cache_loc=torch.arange(1000, 1000 + num_tokens),
        prefix_lens=list(prefix_lens),
        extend_lens=list(extend_lens),
        extend_num_tokens=num_tokens,
        seq_lens=torch.tensor(seq_lens),
        seq_lens_cpu=torch.tensor(seq_lens),
        seq_lens_sum=sum(seq_lens),
        return_logprob=True,
        extend_input_logprob_token_ids=torch.zeros(num_tokens),
        global_num_tokens=sync,
        global_num_tokens_for_logprob=sync_for_logprob,
        draft_global_num_tokens=sync,
        draft_global_num_tokens_for_logprob=sync_for_logprob,
    )


def _draft_rows(batch, hidden_rows):
    """(draft input, position, KV slot, target hidden row) of every row."""
    positions = [
        p + i
        for p, n in zip(batch.prefix_lens, batch.extend_lens, strict=True)
        for i in range(n)
    ]
    return list(
        zip(
            batch.input_ids.tolist(),
            positions,
            batch.out_cache_loc.tolist(),
            hidden_rows.tolist(),
            strict=True,
        )
    )


def _should_split(batch, *, mm_input_embeds=None, draft_prefill_graph_max_tokens=0):
    return should_split_prefill_draft_extend(
        batch,
        mm_input_embeds=mm_input_embeds,
        draft_prefill_graph_max_tokens=draft_prefill_graph_max_tokens,
    )


# Enough KV-only tokens for the split when no graph is captured, in a request
# longer than DSA's dense-attention length (2048 by default).
_LONG = _MAX_UNSPLIT_PREFILL_DRAFT_EXTEND_TOKENS + 2


class TestSplitPrefillDraftExtend(CustomTestCase):
    def test_split_rows_match_the_single_draft_extend(self):
        """Both passes together must feed every prefill row the input of the
        single draft extend: the next prompt token, or the tail at a request's
        last row, at the same position and KV slot."""
        prefix_lens, extend_lens = [0, 6, 3], [5, 2, 9]
        batch = _prefill_batch(prefix_lens, extend_lens)
        tail_tokens = torch.tensor([7, 8, 9])
        single_ids, start = [], 0
        for tail, n in zip(tail_tokens.tolist(), extend_lens):
            single_ids += batch.input_ids[start + 1 : start + n].tolist() + [tail]
            start += n
        single = _draft_rows(
            SimpleNamespace(
                input_ids=torch.tensor(single_ids),
                out_cache_loc=batch.out_cache_loc,
                prefix_lens=prefix_lens,
                extend_lens=extend_lens,
            ),
            torch.arange(batch.extend_num_tokens),
        )

        kv_batch, last_batch, kv_rows, last_rows = split_prefill_draft_extend(
            batch, tail_tokens=tail_tokens
        )

        split = _draft_rows(kv_batch, kv_rows) + _draft_rows(last_batch, last_rows)
        self.assertEqual(sorted(split, key=lambda row: row[3]), single)
        self.assertEqual(kv_batch.extend_num_tokens, 13)
        self.assertEqual(last_batch.extend_num_tokens, 3)
        # The KV-only batch ends before, and the last-token batch at, the last token.
        self.assertEqual(kv_batch.seq_lens.tolist(), [4, 7, 11])
        self.assertEqual(kv_batch.seq_lens_sum, kv_batch.seq_lens_cpu.sum().item())
        self.assertEqual(last_batch.seq_lens.tolist(), [5, 8, 12])
        for sub_batch in (kv_batch, last_batch):
            self.assertFalse(sub_batch.return_logprob)
            self.assertIsNone(sub_batch.extend_input_logprob_token_ids)
            self.assertIsNone(sub_batch.global_num_tokens)
        self.assertEqual(batch.extend_lens, extend_lens)
        self.assertTrue(batch.return_logprob)

    def test_each_pass_syncs_its_own_token_count(self):
        """With MLP sync, a pass that kept the prefill's token count would be
        padded to every prefill row and compute them all."""
        batch = _prefill_batch([0, 6], [5, 9], mlp_sync=True)
        kv_batch, last_batch, _, _ = split_prefill_draft_extend(
            batch, tail_tokens=torch.tensor([7, 8])
        )
        for sub_batch, num_tokens in ((kv_batch, 12), (last_batch, 2)):
            for counts in (
                sub_batch.global_num_tokens,
                sub_batch.draft_global_num_tokens,
            ):
                self.assertEqual(counts, [num_tokens])
            for counts in (
                sub_batch.global_num_tokens_for_logprob,
                sub_batch.draft_global_num_tokens_for_logprob,
            ):
                self.assertEqual(counts, [2])
        self.assertEqual(batch.global_num_tokens, [14])

    def test_one_token_extend_keeps_the_single_draft_extend(self):
        """The KV-only pass has no rows for a request extending by one token, and
        the KV pool planner rejects an empty extend, so such a batch stays whole."""
        self.assertTrue(_should_split(_prefill_batch([0, 4], [_LONG, 2])))
        self.assertFalse(_should_split(_prefill_batch([0, 4], [_LONG, 1])))

    def test_small_prefill_batch_keeps_the_single_draft_extend(self):
        """Batches with at most the threshold of KV-only tokens stay whole."""
        n = _MAX_UNSPLIT_PREFILL_DRAFT_EXTEND_TOKENS
        # n + 1 tokens leave n for the KV-only pass.
        self.assertFalse(_should_split(_prefill_batch([0], [n + 1])))
        self.assertTrue(_should_split(_prefill_batch([0], [n + 2])))

    def test_draft_prefill_graph_sized_batch_keeps_the_single_draft_extend(self):
        """A captured draft prefill graph replays the full draft layer, so the
        KV-only pass must not take a batch the graph could run."""
        n = 2 * _MAX_UNSPLIT_PREFILL_DRAFT_EXTEND_TOKENS
        self.assertFalse(
            _should_split(
                _prefill_batch([0], [n + 1]), draft_prefill_graph_max_tokens=n
            )
        )
        self.assertTrue(
            _should_split(
                _prefill_batch([0], [n + 2]), draft_prefill_graph_max_tokens=n
            )
        )

    def test_batch_of_dense_attention_requests_keeps_the_single_draft_extend(self):
        """Requests DSA runs as dense attention on fresh KV stay whole however many
        there are: the last-token pass would read their prefixes from the cache."""
        # Enough requests for their KV-only tokens to pass the size bound.
        n = _MAX_UNSPLIT_PREFILL_DRAFT_EXTEND_TOKENS // 1023 + 1
        with envs.SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD.override(1024):
            self.assertFalse(_should_split(_prefill_batch([0] * n, [1024] * n)))
            self.assertTrue(
                _should_split(_prefill_batch([0] * n, [1024] * (n - 1) + [1025]))
            )
            # A prefix counts: the request's attention spans it too.
            self.assertTrue(
                _should_split(_prefill_batch([1] + [0] * (n - 1), [1024] * n))
            )

    def test_multimodal_batch_needs_the_target_embeddings(self):
        """The KV-only pass embeds token ids, so a multimodal batch splits only
        when the target passes its embeddings along."""
        batch = _prefill_batch([0], [_LONG], multimodal=True)
        self.assertFalse(_should_split(batch))
        self.assertTrue(_should_split(batch, mm_input_embeds=torch.zeros(_LONG, 2)))


def _supports_forward_kv_only(*, rope, kv_cache_dtype, prefill, decode):
    attn = SimpleNamespace(use_dsa=True, rotary_emb=object() if rope else None)
    draft = SimpleNamespace(
        model=SimpleNamespace(decoder=SimpleNamespace(self_attn=attn))
    )
    kernel = SimpleNamespace(dsa_prefill_backend=prefill, dsa_decode_backend=decode)
    with mock.patch(
        "sglang.srt.models.deepseek_nextn.get_exec",
        return_value=SimpleNamespace(kernel=kernel),
    ):
        return DeepseekV3ForCausalLMNextN.supports_forward_kv_only(
            draft, kv_cache_dtype=kv_cache_dtype
        )


class TestDraftSupportsForwardKVOnly(CustomTestCase):
    def test_trtllm_fused_rope_fp8_write_keeps_the_single_draft_extend(self):
        """TRT-LLM rotates and quantizes an FP8 draft KV in one kernel, so with RoPE
        the KV-only pass would not write the same bytes and the draft stays whole."""
        fused = dict(rope=True, kv_cache_dtype=torch.float8_e4m3fn)
        self.assertFalse(
            _supports_forward_kv_only(**fused, prefill="trtllm", decode="trtllm")
        )
        self.assertFalse(
            _supports_forward_kv_only(**fused, prefill="flashmla_kv", decode="trtllm")
        )
        self.assertTrue(
            _supports_forward_kv_only(**fused, prefill="tilelang", decode="tilelang")
        )
        for unfused in (
            dict(rope=False, kv_cache_dtype=torch.float8_e4m3fn),
            dict(rope=True, kv_cache_dtype=torch.bfloat16),
        ):
            self.assertTrue(
                _supports_forward_kv_only(**unfused, prefill="trtllm", decode="trtllm")
            )


def _can_split(*, kv_cache_dtype, prefill, dcp=False, return_indexer_topk=False):
    draft = mock.MagicMock(spec=DeepseekV3ForCausalLMNextN)
    draft.supports_forward_kv_only.return_value = True
    worker = SimpleNamespace(
        draft_runner=SimpleNamespace(model=draft, kv_cache_dtype=kv_cache_dtype)
    )
    exec_ = SimpleNamespace(
        kernel=SimpleNamespace(dsa_prefill_backend=prefill),
        features=SimpleNamespace(enable_return_indexer_topk=return_indexer_topk),
    )
    parallel = SimpleNamespace(
        attn_dp_enabled=False, enable_prefill_cp=False, dcp_enabled=dcp
    )
    with mock.patch.multiple(
        "sglang.srt.speculative.eagle_worker_v2",
        _is_cuda=True,
        get_exec=mock.Mock(return_value=exec_),
        get_parallel=mock.Mock(return_value=parallel),
    ):
        can_split = EagleDraftWorker._can_split_prefill_draft_extend(worker)
    draft.supports_forward_kv_only.assert_called_once_with(
        kv_cache_dtype=kv_cache_dtype
    )
    return can_split


class TestCanSplitPrefillDraftExtend(CustomTestCase):
    def test_fp8_sparse_prefill_on_fresh_kv_keeps_the_single_draft_extend(self):
        """With an FP8 cache these DSA prefills attend over a prefix-free batch's
        fresh KV, while the last-token pass reads its prefixes from the cache."""
        fp8, bf16 = torch.float8_e4m3fn, torch.bfloat16
        for prefill in (
            "flashmla_sparse",
            "flashmla_sparse_q8",
            "triton_sparse_mla",
            "flashmla_auto",
        ):
            self.assertFalse(_can_split(kv_cache_dtype=fp8, prefill=prefill))
        for prefill in ("flashmla_kv", "tilelang", "trtllm"):
            self.assertTrue(_can_split(kv_cache_dtype=fp8, prefill=prefill))
        # A BF16 cache reads back the bytes the fresh KV holds.
        self.assertTrue(_can_split(kv_cache_dtype=bf16, prefill="flashmla_sparse_q8"))

    def test_indexer_topk_capture_keeps_the_single_draft_extend(self):
        """The capturer copies top-k rows that the KV-only pass never computes, and
        it is created after the draft worker, so the gate reads the server flag."""
        bf16 = torch.bfloat16
        self.assertTrue(_can_split(kv_cache_dtype=bf16, prefill="tilelang"))
        self.assertFalse(
            _can_split(
                kv_cache_dtype=bf16, prefill="tilelang", return_indexer_topk=True
            )
        )

    def test_decode_context_parallel_keeps_the_single_draft_extend(self):
        self.assertFalse(
            _can_split(kv_cache_dtype=torch.bfloat16, prefill="tilelang", dcp=True)
        )


if __name__ == "__main__":
    unittest.main()
