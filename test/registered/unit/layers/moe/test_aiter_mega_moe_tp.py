"""AITER MegaMoE-TP: which forwards take the kernel and which keep the stock path.

Runs on CPU. ``maybe_forward`` decides per forward; the router, the kernel runner and the
CUDA graph capture flags are replaced, the decision itself is not. The kernel's numerics
are covered by the GPU test next to this file.
"""

import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import sglang.srt.layers.moe.aiter_mega_moe_tp as mmtp
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.utils.common import print_info_once
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MAX_TOKENS = 16
HIP_MODULE = "sglang.srt.models.deepseek_common.amd.deepseek_v2_hip_moe"


@pytest.mark.parametrize(
    ("num_tokens", "mode", "capturing", "compiled", "takes_kernel"),
    [
        (MAX_TOKENS, ForwardMode.EXTEND, False, False, True),
        (MAX_TOKENS + 1, ForwardMode.EXTEND, False, False, False),
        (4, ForwardMode.DECODE, False, False, False),
        (4, ForwardMode.EXTEND, True, False, False),
        (4, ForwardMode.DECODE, False, True, False),
    ],
    ids=["window-edge", "above-window", "decode", "graph-capture", "torch-compile"],
)
def test_only_an_eager_extend_forward_inside_the_window_takes_the_kernel(
    monkeypatch, caplog, num_tokens, mode, capturing, compiled, takes_kernel
):
    print_info_once.cache_clear()
    runner = SimpleNamespace(max_tokens=MAX_TOKENS, run=Mock(return_value="reduced"))
    topk_output = SimpleNamespace(topk_ids="ids", topk_weights="weights")
    moe = SimpleNamespace(
        layer_id=7,
        topk=Mock(return_value=topk_output),
        _mega_moe_tp=mmtp._Hosted(runner=runner, weights={"w1": "w"}),
    )
    gate = Mock(return_value=("logits", None))
    monkeypatch.setattr(f"{HIP_MODULE}.forward_gate", gate)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(mmtp, "get_is_capture_mode", lambda: capturing)
    monkeypatch.setattr(mmtp, "is_in_tc_piecewise_cuda_graph", lambda: False)
    batch = SimpleNamespace(forward_mode=mode)
    hidden = torch.zeros(num_tokens, 4, dtype=torch.bfloat16)

    def took_kernel(hidden):
        return mmtp.maybe_forward(moe, hidden, batch, "allocator") is not None

    if compiled:
        # A logger call in maybe_forward would break this trace.
        took_kernel = torch.compile(took_kernel, backend="eager", fullgraph=True)
    with caplog.at_level(logging.INFO, logger="sglang.srt.utils.common"):
        assert took_kernel(hidden) is takes_kernel

    # Routing and the kernel run only on the kernel path.
    assert gate.call_count == runner.run.call_count == int(takes_kernel)
    # A stock forward logs its reason once, except under torch.compile.
    logged = [m for m in caplog.messages if "stock MoE path" in m]
    assert len(logged) == (0 if takes_kernel or compiled else 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
