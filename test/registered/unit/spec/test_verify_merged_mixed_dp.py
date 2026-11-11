"""DP scheduler counts for a merged prefill and speculative verify step."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import dp_attn
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("bad_start", [False, True])
def test_scheduler_exchanges_all_mixed_verify_logits(bad_start):
    batch = SimpleNamespace(
        forward_mode=ForwardMode.MIXED,
        batch_size=lambda: 3,
        num_prefill_rows=1,
        extend_lens=[5, 6, 6],
        extend_logprob_start_lens=[4, 0, int(bad_start)],
        extend_num_tokens=17,
        encoder_swa_reset=None,
        return_logprob=False,
        spec_info=None,
        reqs=[],
    )
    tbo = Mock()
    tbo.prepare_all_gather.return_value = (False, ForwardMode.MIXED.value)
    tbo.compute_output.return_value = (None, ForwardMode.MIXED)
    parallel = SimpleNamespace(
        num_dp_ranks=1,
        attn_tp_size=1,
        attn_cp_size=1,
        tp_group=SimpleNamespace(device_group=None, cpu_group=None, device="cpu"),
    )
    with (
        envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.override(False),
        envs.SGLANG_ENABLE_MIXED_PREFILL_SPEC_COORDINATION.override(True),
        patch.object(dp_attn, "get_parallel", return_value=parallel),
        patch.object(dp_attn, "world_dp_gather_enabled", return_value=False),
        patch.object(dp_attn, "check_cuda_graph_backend", return_value=False),
        patch.object(dp_attn, "TboDPAttentionPreparer", return_value=tbo),
        patch.object(dp_attn, "_local_attention_pairs", return_value=0),
        patch.object(dp_attn, "_ENABLE_METRICS_DP_ATTENTION", False),
    ):

        def prepare():
            return dp_attn.prepare_mlp_sync_batch_raw(
                batch,
                model_runner=SimpleNamespace(
                    spec_algorithm=SpeculativeAlgorithm.DSPARK,
                    model_config=None,
                ),
                get_idle_batch=Mock(),
                disable_cuda_graph=False,
                require_mlp_tp_gather=True,
                disable_overlap_schedule=False,
                offload_tags=set(),
            )

        if bad_start:
            # Preserve the invariant: do not fix the original crash by
            # accepting a missing verify-position logit.
            with pytest.raises(AssertionError):
                prepare()
            return
        result = prepare()
    assert result.global_num_tokens == [17]
    assert result.global_num_tokens_for_logprob == [13]
    assert result.dp_spec_prefill_mixed_decode_counts == [2]
    counts, logits, prefills = result.dp_spec_prefill_coordination_metadata
    assert counts == [17] and logits == [13] and prefills.tolist() == [1]


def test_mixed_counts_survive_cross_rank_gather_and_idle_fallback():
    infos = [
        dp_attn.MLPSyncBatchInfo(
            num_dp_ranks=2,
            tp_size=1,
            cp_size=1,
            num_tokens=tokens,
            num_tokens_for_logprob=logits,
            num_mixed_decode_reqs=tails,
            can_run_decode_cuda_graph=False,
            can_run_draft_cuda_graph=True,
            can_run_prefill_cuda_graph=False,
            is_extend_in_batch=bool(tails),
            local_can_run_tbo=False,
            local_forward_mode=mode.value,
        )
        for tokens, logits, tails, mode in (
            (17, 13, 2, ForwardMode.MIXED),
            (0, 0, 0, ForwardMode.IDLE),
        )
    ]
    values = torch.cat([info._get_local_tensor("cpu") for info in infos])
    parallel = SimpleNamespace(tp_group=SimpleNamespace(active_ranks_cpu=torch.ones(2)))

    def gather(output, *_args, **_kwargs):
        output.copy_(values)

    with (
        patch.object(dp_attn, "get_parallel", return_value=parallel),
        patch.object(dp_attn, "exchange_sync_rows", side_effect=gather),
    ):
        for info in infos:
            info.all_gather(device="cpu", group=None)
            assert info.global_num_mixed_decode_reqs == [2, 0]
            assert info.global_num_tokens_for_logprob == [13, 0]
            assert info._get_fallback_tensor("cpu")[13].item() == 0
