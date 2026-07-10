import torch

from sglang.srt.speculative.dspark_components.kernels import softmax_temp as mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_flashinfer_dispatch_falls_back_to_torch_when_unavailable(monkeypatch):
    monkeypatch.setattr(mod, "_KERNEL_IMPL", "flashinfer")
    monkeypatch.setattr(mod, "_flashinfer_softmax", None)
    monkeypatch.setattr(mod, "_warned_flashinfer_unavailable", False)

    logits = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]])
    temperatures = torch.ones(1)

    ref = mod.softmax_temp(
        logits=logits, temperatures=temperatures, rows_per_request=2
    )
    got = mod.SoftmaxTemp.execute(
        logits=logits, temperatures=temperatures, rows_per_request=2
    )

    torch.testing.assert_close(got, ref)
