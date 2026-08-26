from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.dp_attn import (
    MLPSyncBatchInfo,
    prepare_mlp_sync_batch_raw,
    should_skip_scheduler_all_gather,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def test_dp1_skips_scheduler_all_gather_by_default():
    with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False):
        assert should_skip_scheduler_all_gather(dp_size=1)


def test_multi_dp_preserves_default_and_explicit_override():
    with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False):
        assert not should_skip_scheduler_all_gather(dp_size=2)
    with envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(True):
        assert should_skip_scheduler_all_gather(dp_size=2)


def test_dp1_skip_preserves_local_tbo_metadata():
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        batch_size=lambda: 4,
    )
    model_runner = SimpleNamespace(prefill_cuda_graph_runner=None)
    tp_group = SimpleNamespace(
        device_group=object(),
        device="cpu",
        cpu_group=object(),
    )
    get_idle_batch = Mock(side_effect=AssertionError("DP1 must not emit idle batch"))
    tbo_preparer = Mock()
    tbo_preparer.prepare_all_gather.return_value = (
        True,
        ForwardMode.DECODE.value,
    )
    tbo_preparer.compute_output.return_value = (2, ForwardMode.DECODE)

    with (
        envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.override(False),
        patch(
            "sglang.srt.managers.scheduler_components.dp_attn."
            "TboDPAttentionPreparer",
            return_value=tbo_preparer,
        ),
        patch(
            "sglang.srt.managers.scheduler_components.dp_attn."
            "world_dp_gather_enabled",
            return_value=False,
        ),
        patch(
            "sglang.srt.managers.scheduler_components.dp_attn."
            "check_cuda_graph_backend",
            return_value=False,
        ),
        patch.object(MLPSyncBatchInfo, "all_gather") as all_gather,
    ):
        result = prepare_mlp_sync_batch_raw(
            batch,
            model_runner=model_runner,
            dp_size=1,
            attn_tp_size=4,
            attn_cp_size=1,
            tp_group=tp_group,
            get_idle_batch=get_idle_batch,
            disable_cuda_graph=False,
            require_mlp_tp_gather=False,
            disable_overlap_schedule=True,
            offload_tags=set(),
        )

    all_gather.assert_not_called()
    get_idle_batch.assert_not_called()
    assert result.global_num_tokens == [4]
    assert result.global_num_tokens_for_logprob == [4]
    assert result.tbo_split_seq_index == 2
    assert result.global_forward_mode == ForwardMode.DECODE
    assert result.recv_skipper_forward_mode == ForwardMode.DECODE
    assert result.can_run_decode_cuda_graph
    assert not result.can_run_dp_prefill_cuda_graph
    assert tbo_preparer.compute_output.call_args.args[0].tolist() == [
        [1, ForwardMode.DECODE.value]
    ]
