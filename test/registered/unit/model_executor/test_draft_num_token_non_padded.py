"""EAGLE draft ``ForwardBatch.init_new`` must not report a live token count of 0.

Since #39574 the MoE padded-region mask keys off ``global_num_token_non_padded``.
Draft batches reach ``init_new`` with ``input_ids is None`` (tokens are rebuilt
after); a count of 0 then masks every draft ``topk_ids`` row to -1. Guard: a
spec batch with ``input_ids=None`` and ``spec_info.num_tokens_per_req > 0``
must publish ``bs * num_tokens_per_req``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_BS = 4
_TOKENS_PER_REQ = 1


def _req():
    return SimpleNamespace(lora_id=None, rid="r0", token_type_ids=None)


def _draft_schedule_batch(*, input_ids, spec_info):
    seq_lens = torch.tensor([8, 9, 7, 10], dtype=torch.int64)
    return SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        sampling_info=None,
        seq_lens_cpu=seq_lens,
        seq_lens_sum=int(seq_lens.sum()),
        seq_lens=seq_lens,
        input_ids=input_ids,
        req_pool_indices=torch.arange(_BS, dtype=torch.int64),
        out_cache_loc=torch.zeros(_BS, dtype=torch.int64),
        orig_seq_lens=None,
        out_cache_loc_dsv4=None,
        engram_history=None,
        mamba_track_indices=None,
        mamba_track_mask=None,
        mamba_track_seqlens=None,
        mamba_cow_src_indices=None,
        mamba_cow_dst_indices=None,
        mamba_clear_indices=None,
        encoder_lens=None,
        encoder_out_cache_loc=None,
        input_embeds=None,
        replace_embeds=None,
        replace_positions=None,
        return_logprob=False,
        is_extend_in_batch=False,
        can_run_decode_cuda_graph=False,
        can_run_dp_prefill_cuda_graph=False,
        dp_prefill_cuda_graph_max_prefix_len=None,
        global_forward_mode=None,
        is_prefill_only=False,
        spec_algorithm=SpeculativeAlgorithm.EAGLE,
        tbo_split_seq_index=None,
        top_logprobs_nums=None,
        token_ids_logprobs=None,
        multimodal_inputs=None,
        encoder_cached=None,
        encoder_lens_cpu=None,
        reqs=[_req() for _ in range(_BS)],
        spec_info=spec_info,
        extend_input_logprob_token_ids=None,
        global_num_tokens=None,
        dllm_config=None,
    )


def _draft_model_runner():
    translator = SimpleNamespace(rebind_write_loc=MagicMock())
    return SimpleNamespace(
        device="cpu",
        model_config=SimpleNamespace(
            requires_mm_token_modalities=False,
            model_is_mrope=False,
        ),
        kv_index_translator=translator,
        ngram_embedding_manager=SimpleNamespace(enabled=False),
        lora_manager=None,
        ps=SimpleNamespace(attn_dcp_size=1),
    )


def _init_draft_forward_batch(*, input_ids, spec_info):
    batch = _draft_schedule_batch(input_ids=input_ids, spec_info=spec_info)
    with get_parallel().override(moe_ep_size=4):
        return ForwardBatch.init_new(
            batch,
            _draft_model_runner(),
            capture_hidden_mode=CaptureHiddenMode.LAST,
            return_hidden_states_before_norm=False,
        )


class TestDraftNumTokenNonPadded(CustomTestCase):
    def test_draft_input_ids_none_uses_spec_info_width(self):
        """input_ids is None + spec_info.num_tokens_per_req > 0 must not
        publish a live count of 0 (that masks every draft topk id to -1)."""
        spec_info = SimpleNamespace(
            num_tokens_per_req=_TOKENS_PER_REQ,
            positions=torch.arange(_BS, dtype=torch.int64),
        )
        ret = _init_draft_forward_batch(input_ids=None, spec_info=spec_info)
        want = _BS * _TOKENS_PER_REQ
        self.assertEqual(ret.global_num_token_non_padded_cpu, want)
        self.assertIsNotNone(ret.global_num_token_non_padded)
        self.assertEqual(int(ret.global_num_token_non_padded), want)

    def test_input_ids_present_keeps_tensor_width(self):
        input_ids = torch.arange(6, dtype=torch.int64)
        spec_info = SimpleNamespace(
            num_tokens_per_req=_TOKENS_PER_REQ,
            positions=torch.arange(6, dtype=torch.int64),
        )
        ret = _init_draft_forward_batch(input_ids=input_ids, spec_info=spec_info)
        self.assertEqual(ret.global_num_token_non_padded_cpu, 6)
        self.assertEqual(int(ret.global_num_token_non_padded), 6)


if __name__ == "__main__":
    import unittest

    unittest.main()
