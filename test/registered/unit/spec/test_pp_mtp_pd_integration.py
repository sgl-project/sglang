from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.arg_groups import validation_hook
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.srt.mem_cache.kv.alloc.iteration_ids import IterationIds
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.pp_spec_relay import PPSpecRelayInput


def config(dp=1):
    return SimpleNamespace(
        disable_overlap_schedule=True,
        speculative_algorithm="EAGLE",
        enable_multi_layer_eagle=False,
        disaggregation_mode="decode",
        speculative_adaptive=False,
        attn_dp_size=dp,
        tp_size=8,
        attn_cp_size=1,
        ep_join_mode=None,
        enable_return_routed_experts=False,
        scheduler_recv_interval=1,
        min_free_slots_delay=None,
    )


def test_pp_eagle_accepts_pd_decode_with_attention_dp(monkeypatch):
    monkeypatch.setattr(validation_hook.envs.SGLANG_ENABLE_PP_SPEC, "get", lambda: True)
    validation_hook.check_pipeline_parallel_compat(config(dp=8))


@pytest.mark.parametrize("dp,cp", [(3, 1), (8, 2)])
def test_pp_eagle_rejects_invalid_attention_lane_layout(monkeypatch, dp, cp):
    monkeypatch.setattr(validation_hook.envs.SGLANG_ENABLE_PP_SPEC, "get", lambda: True)
    cfg = config(dp)
    cfg.attn_cp_size = cp
    with pytest.raises(AssertionError):
        validation_hook.check_pipeline_parallel_compat(cfg)


def test_pd_draft_input_is_normalized_before_batch_merge():
    scheduler = object.__new__(SchedulerPPMixin)
    scheduler._pp_spec_relay = True
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(rid="r1"), SimpleNamespace(rid="r2")],
        spec_info=EagleDraftInput(bonus_tokens=torch.tensor([17, 29])),
    )

    with patch(
        "sglang.srt.managers.scheduler_pp_mixin.get_spec",
        return_value=SimpleNamespace(speculative_num_draft_tokens=6),
    ):
        scheduler._pp_spec_normalize_relay_input(batch)

    assert isinstance(batch.spec_info, PPSpecRelayInput)
    assert batch.spec_info.rids == ["r1", "r2"]
    torch.testing.assert_close(
        batch.spec_info.tokens,
        torch.tensor([[17, 0, 0, 0, 0, 0], [29, 0, 0, 0, 0, 0]]),
    )


def test_eagle_tail_draft_restores_scheduler_batch():
    worker = object.__new__(EAGLEWorkerV2)
    next_verify_input = SimpleNamespace(draft_token=torch.tensor([7, 11]))
    parent_list = torch.tensor([0, 1])
    top_scores_index = torch.tensor([1, 0])
    worker._draft_worker = SimpleNamespace(
        parallel_layout=object(),
        draft=MagicMock(
            return_value=(next_verify_input, parent_list, top_scores_index)
        ),
    )
    original = {
        "spec_info": object(),
        "seq_lens": torch.tensor([8, 13]),
        "forward_mode": ForwardMode.DECODE,
        "input_ids": torch.tensor([3, 5]),
        "seq_lens_cpu": torch.tensor([8, 13]),
        "seq_lens_sum": 21,
    }
    batch = SimpleNamespace(**original)
    output = SimpleNamespace(
        next_draft_input=object(),
        new_seq_lens=torch.tensor([10, 16]),
    )

    prefix = "sglang.srt.speculative.eagle_worker_v2."
    with (
        patch(prefix + "draft_tp_context", return_value=nullcontext()),
        patch(prefix + "speculative_moe_backend_context", return_value=nullcontext()),
        patch(
            prefix + "speculative_moe_a2a_backend_context",
            return_value=nullcontext(),
        ),
        patch(prefix + "spec_stage_span", return_value=nullcontext()),
    ):
        worker.draft_next_pp_round(batch, output)

    for name, value in original.items():
        assert getattr(batch, name) is value
    torch.testing.assert_close(output.next_verify_chain, torch.tensor([7, 11]))
    torch.testing.assert_close(output.next_verify_parent_list, parent_list)
    torch.testing.assert_close(output.next_verify_top_scores_index, top_scores_index)


def test_empty_attention_dp_lane_preserves_empty_write_ids():
    plan = object.__new__(IterationIds)
    plan.req_pool_indices = torch.empty(0, dtype=torch.int64)
    plan.write_virtual = torch.empty(0, dtype=torch.int64)
    assert plan.virtual_write_ids(cols=slice(0, 1)).numel() == 0
