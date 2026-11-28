from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.speculative import dfly_worker_v2
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def test_dfly_accept_uses_grammar_and_verify_adjustments(monkeypatch):
    device = torch.device("cpu")
    draft_input = DFlashDraftInputV2.create_idle_input(device=device)
    draft_input.bonus_tokens = torch.tensor([1])
    batch = SimpleNamespace(
        spec_info=draft_input,
        seq_lens=torch.tensor([11]),
        req_pool_indices=torch.tensor([0]),
        input_ids=None,
        out_cache_loc=None,
        sampling_info=object(),
        has_grammar=True,
    )
    candidates = torch.tensor([[1, 2, 3, 4, 5, 6]])
    positions = torch.arange(11, 17)
    logits = torch.zeros(6, 16)
    patched_logits = torch.ones_like(logits)
    patched_info = object()
    logits_output = SimpleNamespace(next_token_logits=logits, hidden_states=None)
    result = SimpleNamespace(
        logits_output=logits_output,
        can_run_cuda_graph=True,
        routed_experts_output=None,
        indexer_topk_output=None,
    )
    barrier = Mock()
    tree = object()
    seen = []

    def build_mask(tree_arg, *, logits_output, barrier):
        assert tree_arg is tree
        assert logits_output.next_token_logits is logits
        barrier()
        seen.append("grammar")

    def adjust(logits_arg, tokens, *, positions):
        assert seen == ["grammar"]
        assert logits_arg is logits
        torch.testing.assert_close(tokens, candidates)
        torch.testing.assert_close(positions, torch.arange(11, 17).view(1, 6))
        seen.append("adjust")
        return patched_logits, patched_info

    def accept(info, inp, tokens, logits_arg):
        assert seen == ["grammar", "adjust"]
        assert (
            info is patched_info and inp is draft_input and logits_arg is patched_logits
        )
        torch.testing.assert_close(tokens, candidates)
        seen.append("accept")
        return torch.tensor([0]), torch.tensor([9])

    step = SimpleNamespace(build_grammar_mask=build_mask, adjust=adjust, commit=Mock())
    worker = object.__new__(dfly_worker_v2.DFlyWorkerV2)
    worker.device = device
    worker.block_size = candidates.shape[1]
    worker.dfly = SimpleNamespace(candidates_out=candidates.flatten())
    worker._target_worker = SimpleNamespace(
        forward_batch_generation=Mock(return_value=result)
    )
    worker.model_runner = SimpleNamespace(
        model=object(),
        attn_backend=object(),
        sampling_pipeline=SimpleNamespace(plan_verify=Mock(return_value=step)),
    )
    worker._prepare_verify = Mock(return_value=SimpleNamespace(positions=positions))
    worker._commit_state = Mock()
    worker._accept = accept
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(
        dfly_worker_v2.GrammarTree, "from_linear_chain", Mock(return_value=tree)
    )
    monkeypatch.setattr(dfly_worker_v2, "make_draft_input_v2", lambda **_: None)

    decoded = worker._forward_decode(batch, None, barrier)

    assert seen == ["grammar", "adjust", "accept"]
    barrier.assert_called_once()
    assert decoded.accept_lens.tolist() == [1]
    assert decoded.new_seq_lens.tolist() == [12]
    assert decoded.next_token_ids[0].item() == 9
    step.commit.assert_called_once()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
