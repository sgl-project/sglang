import torch.nn as nn

from sglang.srt.models.minicpmv import MiniCPMV
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_minicpmv_eval_keeps_wrapper_and_shared_experts_config():
    inner = nn.Linear(1, 1)
    inner.train()

    wrapper = MiniCPMV.__new__(MiniCPMV)
    wrapper.minicpmv = inner
    marker = object()
    wrapper.shared_experts_config = marker

    assert wrapper.eval() is wrapper
    assert wrapper.shared_experts_config is marker
    assert not inner.training

    assert wrapper.train(True) is wrapper
    assert wrapper.shared_experts_config is marker
    assert inner.training
