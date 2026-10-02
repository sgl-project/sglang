"""Actual IDLE inputs and fixed capture modes through the decode runner."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from nccl_ep_test.runner_inputs import SyntheticDecodeRunner, input_batch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.token_dispatcher.nccl_ep_admission import NcclEpGraphDecision
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def context():
    from sglang.srt.runtime_context import get_context

    with get_context().override_server_args(enable_nccl_ep_cuda_graph=True):
        yield


def _runner():
    def forward(batch):
        row = torch.arange(batch.batch_size, device="cuda")
        value = torch.where(row < batch.num_token_non_padded, batch.input_ids * 2, -1)
        return LogitsProcessorOutput(next_token_logits=value[:, None])

    runner = SyntheticDecodeRunner(
        forward, SimpleNamespace(barrier=lambda: None), buckets=(1, 8, 16, 32)
    )
    runner.require_mlp_sync = True
    return runner


def test_true_idle_preserves_zero_valid_tokens_and_empty_result():
    runner = _runner()
    try:
        for size, bucket in ((9, 16), (0, 1), (17, 32), (0, 1), (3, 8)):
            incoming = input_batch(list(range(size)))
            incoming.forward_mode = (
                ForwardMode.IDLE if size == 0 else ForwardMode.DECODE
            )
            incoming.can_run_decode_cuda_graph = True
            assert runner.can_run_graph(incoming)
            output = runner.execute(incoming)
            assert output.next_token_logits.shape == (size, 1)
            assert runner.attn_backend.views[-1].batch_size == bucket
            assert runner.buffers.num_token_non_padded.item() == size
            torch.testing.assert_close(
                output.next_token_logits[:, 0], incoming.input_ids * 2
            )
    finally:
        runner.backend.cleanup()


def test_hidden_mode_incompatibility_falls_back_without_recapture():
    runner = _runner()
    try:
        high = input_batch([1], hidden_mode=CaptureHiddenMode.FULL)
        high.can_run_decode_cuda_graph = True
        assert not runner.can_run_graph(high)
        with pytest.raises(RuntimeError, match="fixed CUDA graph"):
            runner.execute(high)
        normal = input_batch([1])
        normal.can_run_decode_cuda_graph = True
        assert runner.can_run_graph(normal)
        runner.execute(normal)
        assert runner.capture_generations == 1
    finally:
        runner.backend.cleanup()


@pytest.mark.parametrize(
    "mode,local_eligible,admitted",
    [
        (ForwardMode.IDLE, True, False),
        (ForwardMode.DECODE, False, False),
        (ForwardMode.EXTEND, False, False),
        (ForwardMode.IDLE, True, True),
    ],
)
def test_model_runner_votes_before_any_graph_or_eager_work(
    monkeypatch, mode, local_eligible, admitted
):
    from sglang.srt.model_executor import model_runner as module

    events = []
    decision = NcclEpGraphDecision(admitted)

    def vote(**kwargs):
        events.append("vote")
        assert kwargs["eligible"] == local_eligible
        return decision

    def replay(batch, **kwargs):
        events.append("graph")
        return LogitsProcessorOutput(next_token_logits=None)

    def eager(batch, **kwargs):
        events.append("eager")
        return LogitsProcessorOutput(next_token_logits=None)

    runner = SimpleNamespace(
        attn_backend=None,
        device="cuda",
        server_args=SimpleNamespace(
            enable_nccl_ep_cuda_graph=True, moe_a2a_backend="nccl_ep"
        ),
        decode_cuda_graph_runner=SimpleNamespace(
            can_run_graph=lambda batch: local_eligible,
            required_capture_hidden_mode=lambda batch: 0,
            capture_hidden_mode=0,
            execute=replay,
        ),
        _nccl_ep_graph_admission=SimpleNamespace(decide=vote),
        hisparse_coordinator=None,
        _prepare_eager_forward_batch=lambda batch: events.append("prepare"),
        _maybe_execute_deferred_mamba_cow_and_clear=lambda batch: None,
        prefill_cuda_graph_runner=None,
        eager_runner=SimpleNamespace(execute=eager),
    )
    batch = SimpleNamespace(
        forward_mode=mode,
        global_num_tokens_cpu=None,
        capture_hidden_mode=0,
        token_indices_to_pool=None,
    )
    monkeypatch.setattr(module, "get_global_dwdp_manager", lambda: None)
    result = module.ModelRunner._forward_raw(runner, batch, None)
    assert result.can_run_graph == admitted
    assert events == (["vote", "graph"] if admitted else ["vote", "prepare", "eager"])


@pytest.mark.parametrize("counts", [(0, 8), (8, 0), (0, 0)])
def test_scheduler_creates_idle_only_when_a_peer_has_work(monkeypatch, counts):
    from sglang.srt.managers.scheduler_components import dp_attn

    monkeypatch.setattr(dp_attn, "world_dp_gather_enabled", lambda: False)
    monkeypatch.setattr(dp_attn, "check_cuda_graph_backend", lambda *args: False)
    monkeypatch.setattr(
        dp_attn,
        "TboDPAttentionPreparer",
        lambda: SimpleNamespace(
            prepare_all_gather=lambda batch: (False, ForwardMode.DECODE),
            compute_output=lambda info: (None, ForwardMode.DECODE),
        ),
    )

    def gathered(info, **kwargs):
        info.global_num_tokens = list(counts)
        info.global_num_tokens_for_logprob = list(counts)
        info.can_run_decode_cuda_graph = True
        info.is_extend_in_batch = False
        info.tp0_info_cpu = torch.zeros(2, 7, dtype=torch.int64)
        info.tp0_info_cpu[:, 5] = int(ForwardMode.DECODE)

    monkeypatch.setattr(dp_attn.MLPSyncBatchInfo, "all_gather", gathered)
    monkeypatch.setattr(
        dp_attn,
        "get_parallel",
        lambda: SimpleNamespace(
            dp_size=2,
            attn_tp_size=1,
            attn_cp_size=1,
            tp_group=SimpleNamespace(cpu_group=None),
        ),
    )
    created = []

    def idle():
        created.append(True)
        return SimpleNamespace(forward_mode=ForwardMode.IDLE)

    local = (
        SimpleNamespace(forward_mode=ForwardMode.DECODE, batch_size=lambda: counts[0])
        if counts[0]
        else None
    )
    actual = dp_attn.prepare_mlp_sync_batch_raw(
        local,
        model_runner=SimpleNamespace(spec_algorithm=None, model_config=None),
        get_idle_batch=idle,
        disable_cuda_graph=False,
        require_mlp_tp_gather=False,
        disable_overlap_schedule=False,
        offload_tags={"test"},
    )
    if counts == (0, 0):
        assert actual is None and not created
    else:
        assert actual is not None
        assert actual.global_num_tokens == [counts[0]]
        assert actual.can_run_decode_cuda_graph
        assert bool(created) == (counts[0] == 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
