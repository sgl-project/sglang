"""Exercise the real DeepSeek weight-loading loops with small CPU parameters."""

import argparse
import concurrent.futures
import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    DeepseekV2WeightLoaderMixin,
)
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


class SharedDeepseekLoaderProbe(LoaderProbe, DeepseekV2WeightLoaderMixin):
    load_weights = DeepseekV2WeightLoaderMixin.do_load_weights


@pytest.fixture(
    params=[LoaderProbe, SharedDeepseekLoaderProbe], ids=["v4", "shared-v2-v3-r1"]
)
def loader_probe(request):
    config = SimpleNamespace(weight_loader_copy_num_threads=None)
    request.getfixturevalue("monkeypatch").setattr(
        "sglang.srt.models.deepseek_v4.get_model", lambda: config
    )
    request.getfixturevalue("monkeypatch").setattr(
        "sglang.srt.models.deepseek_common.deepseek_weight_loader.get_model",
        lambda: config,
    )
    return request.param, config


def test_copy_worker_limit_bounds_loading_and_preserves_weights(
    monkeypatch, loader_probe
):
    loader_probe, config = loader_probe
    config.weight_loader_copy_num_threads = 2
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

    model = loader_probe(loader)
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
def test_invalid_copy_limit_fails_before_reading_weights(
    monkeypatch, workers, loader_probe
):
    loader_probe, config = loader_probe
    config.weight_loader_copy_num_threads = int(workers)

    def weights():
        pytest.fail("invalid worker count consumed checkpoint tensors")
        yield

    with pytest.raises(ValueError, match="max_workers"):
        loader_probe(lambda param, weight: param.data.copy_(weight)).load_weights(
            weights()
        )


def test_unset_limit_preserves_loading(monkeypatch, loader_probe):
    loader_probe, config = loader_probe
    model = loader_probe(lambda param, weight: param.data.copy_(weight))
    weights = [(f"probe_{i}", torch.full((2,), float(i))) for i in range(8)]
    model.load_weights(weights)
    for name, weight in weights:
        torch.testing.assert_close(getattr(model, name), weight)


def test_limited_copy_workers_propagate_loader_errors(monkeypatch, loader_probe):
    loader_probe, config = loader_probe
    config.weight_loader_copy_num_threads = 1

    def fail_copy(param, weight):
        raise RuntimeError("checkpoint copy failed")

    with pytest.raises(RuntimeError, match="checkpoint copy failed"):
        loader_probe(fail_copy).load_weights([("probe_0", torch.ones(2))])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


def test_copy_threads_cli_round_trip():
    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        ["--model-path", "dummy", "--weight-loader-copy-num-threads", "2"]
    )
    assert args.weight_loader_copy_num_threads == 2
    assert (
        parser.parse_args(["--model-path", "dummy"]).weight_loader_copy_num_threads
        is None
    )
