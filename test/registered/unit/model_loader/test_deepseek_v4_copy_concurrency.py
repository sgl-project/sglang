"""Exercise the real V4 weight-loading loop with small CPU parameters."""

import concurrent.futures
import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class LoaderProbe(nn.Module):
    load_weights = DeepseekV4ForCausalLM.load_weights
    remap_weight_name_to_dpsk_hf_format = staticmethod(
        DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format
    )

    def __init__(self, loader):
        super().__init__()
        self.config = SimpleNamespace(n_routed_experts=1, num_hidden_layers=0)
        self.quant_config = None
        self.wo_a_fp8 = True
        self.num_fused_shared_experts = 0
        self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        for i in range(8):
            param = nn.Parameter(torch.zeros(2), requires_grad=False)
            param.weight_loader = loader
            self.register_parameter(f"probe_{i}", param)

    def post_load_weights(self, **kwargs):
        pass

    def _prewarm_mhc_kernels(self):
        pass


def test_copy_worker_limit_bounds_loading_and_preserves_weights(monkeypatch):
    monkeypatch.setenv("SGLANG_DSV4_WEIGHT_LOADER_MAX_WORKERS", "2")
    release = threading.Event()
    started = threading.Event()
    exceeded = threading.Event()
    lock = threading.Lock()
    active = 0

    def loader(param, weight):
        nonlocal active
        with lock:
            active += 1
            if active == 2:
                started.set()
            if active > 2:
                exceeded.set()
        assert release.wait(10), "copy workers did not get released"
        param.data.copy_(weight)
        with lock:
            active -= 1

    model = LoaderProbe(loader)
    weights = [(f"probe_{i}", torch.full((2,), float(i))) for i in range(8)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as driver:
        future = driver.submit(model.load_weights, weights)
        try:
            assert started.wait(10), "loader did not start both copy workers"
            assert not exceeded.wait(0.5), "configured copy concurrency exceeded"
        finally:
            release.set()
        future.result(timeout=10)
    for name, weight in weights:
        torch.testing.assert_close(getattr(model, name), weight)


@pytest.mark.parametrize("workers", ["0", "-1"])
def test_invalid_copy_limit_fails_before_reading_weights(monkeypatch, workers):
    monkeypatch.setenv("SGLANG_DSV4_WEIGHT_LOADER_MAX_WORKERS", workers)

    def weights():
        pytest.fail("invalid worker count consumed checkpoint tensors")
        yield

    with pytest.raises(ValueError, match="max_workers"):
        LoaderProbe(lambda param, weight: param.data.copy_(weight)).load_weights(
            weights()
        )


def test_unset_limit_preserves_loading(monkeypatch):
    monkeypatch.delenv("SGLANG_DSV4_WEIGHT_LOADER_MAX_WORKERS", raising=False)
    model = LoaderProbe(lambda param, weight: param.data.copy_(weight))
    weights = [(f"probe_{i}", torch.full((2,), float(i))) for i in range(8)]
    model.load_weights(weights)
    for name, weight in weights:
        torch.testing.assert_close(getattr(model, name), weight)


def test_limited_copy_workers_propagate_loader_errors(monkeypatch):
    monkeypatch.setenv("SGLANG_DSV4_WEIGHT_LOADER_MAX_WORKERS", "1")

    def fail_copy(param, weight):
        raise RuntimeError("checkpoint copy failed")

    with pytest.raises(RuntimeError, match="checkpoint copy failed"):
        LoaderProbe(fail_copy).load_weights([("probe_0", torch.ones(2))])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
