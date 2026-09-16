"""Graph admission must precede disabling NCCL EP eager overlap."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("backend", ["nccl_ep", "deepep"])
@pytest.mark.parametrize(
    "mode", [ForwardMode.EXTEND, ForwardMode.DECODE, ForwardMode.IDLE]
)
@pytest.mark.parametrize(
    "local_eligible,admitted", [(False, False), (True, False), (True, True)]
)
def test_runner_disables_tbo_only_after_graph_admission(
    monkeypatch, backend, mode, local_eligible, admitted
):
    from sglang.srt.model_executor.model_runner import ModelRunner

    batch = ForwardBatch(
        forward_mode=mode,
        batch_size=2,
        input_ids=torch.zeros(2, dtype=torch.long),
        req_pool_indices=torch.zeros(2, dtype=torch.long),
        seq_lens=torch.ones(2, dtype=torch.int32),
        out_cache_loc=torch.zeros(2, dtype=torch.long),
        seq_lens_sum=2,
        tbo_split_seq_index=1,
        global_forward_mode=mode,
    )
    selected = (
        mode.is_cuda_graph()
        and local_eligible
        and (admitted if backend == "nccl_ep" else True)
    )
    events = []

    def can_run_graph(candidate):
        assert candidate is batch and candidate.can_run_tbo
        events.append("eligibility")
        return local_eligible

    def decide(**kwargs):
        assert batch.can_run_tbo
        events.append("admission")
        return SimpleNamespace(can_run=kwargs["eligible"] and admitted)

    def execute(candidate, **kwargs):
        assert candidate is batch and candidate.can_run_tbo
        events.append("graph")
        return "graph output"

    class PreparedEager(Exception):
        pass

    def prepare(candidate):
        events.append("eager")
        assert candidate.can_run_tbo == (backend != "nccl_ep")
        if backend == "nccl_ep":
            assert candidate.global_forward_mode is None
            assert candidate.tbo_children is None
            TboForwardBatchPreparer.prepare(candidate)
            assert candidate.tbo_children is None
        raise PreparedEager

    split = Mock(side_effect=AssertionError("eager must not prepare TBO children"))
    monkeypatch.setattr(TboForwardBatchPreparer, "prepare_raw", split)
    monkeypatch.setattr(
        "sglang.srt.distributed.parallel_state.get_moe_ep_group", lambda: None
    )
    runner = SimpleNamespace(
        device="cuda",
        attn_backend=None,
        server_args=SimpleNamespace(
            enable_nccl_ep_cuda_graph=backend == "nccl_ep", moe_a2a_backend=backend
        ),
        decode_cuda_graph_runner=SimpleNamespace(
            can_run_graph=can_run_graph,
            execute=execute,
            required_capture_hidden_mode=lambda batch: 0,
            capture_hidden_mode=0,
        ),
        _nccl_ep_graph_admission=SimpleNamespace(decide=decide),
        hisparse_coordinator=None,
        _prepare_eager_forward_batch=prepare,
    )
    if selected:
        result = ModelRunner._forward_raw(runner, batch, None)
        assert result.can_run_graph and result.logits_output == "graph output"
    else:
        with pytest.raises(PreparedEager):
            ModelRunner._forward_raw(runner, batch, None)
    assert events[-1] == ("graph" if selected else "eager")
    assert ("admission" in events) == (backend == "nccl_ep")
    split.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
