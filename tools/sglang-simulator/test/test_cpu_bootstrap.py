"""CPU bootstrap preserves native quantization validation and ordinary imports."""

import importlib.util
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

requires_cpu_runtime = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("sglang") is None,
    reason="requires the SGLang CPU test environment",
)


def run_python(code, *, bootstrap=False, cpu=False):
    env = os.environ.copy()
    for key in ("SGLANG_SIMULATOR_BOOTSTRAP", "SGLANG_USE_CPU_ENGINE"):
        env.pop(key, None)
    if bootstrap:
        env["SGLANG_SIMULATOR_BOOTSTRAP"] = "1"
    if cpu:
        env["SGLANG_USE_CPU_ENGINE"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "bootstrap,cpu", [(False, False), (True, False), (False, True)]
)
@requires_cpu_runtime
def test_bootstrap_requires_both_flags(bootstrap, cpu):
    run_python(
        """
import sys, torch
original = torch.cuda.is_available
hip = torch.version.hip
import usercustomize
usercustomize.apply_cpu_simulation_compat()
assert torch.cuda.is_available is original
assert torch.version.hip == hip
assert 'sglang.srt.layers.quantization' not in sys.modules
""",
        bootstrap=bootstrap,
        cpu=cpu,
    )


@requires_cpu_runtime
def test_cpu_bootstrap_preserves_native_quantization():
    run_python(
        """
import sys
import usercustomize
usercustomize.apply_cpu_simulation_compat()
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.quantization import QUANTIZATION_METHODS
from sglang.srt.layers.quantization.fp8 import Fp8Config
assert QUANTIZATION_METHODS['fp8'] is Fp8Config
assert 'aiter' not in sys.modules

def validate(method, hf=None):
    config = ModelConfig.__new__(ModelConfig)
    config.quantization = method
    config.is_draft_model = False
    config._parse_quant_hf_config = lambda: hf
    config._find_quant_modelslim_config = lambda: None
    config._verify_quantization()
    return config.quantization

for method in (None, 'fp8', 'awq', 'compressed-tensors', 'w8a8_fp8', 'mxfp8'):
    assert validate(method) == method
assert validate('AWQ') == 'awq'
assert validate(None, {'quant_method': 'fp8'}) == 'fp8'
assert validate('w8a8_fp8', {'quant_method': 'compressed-tensors'}) == 'w8a8_fp8'
for method, hf in [('unknown', None), ('awq', {'quant_method': 'fp8'})]:
    try:
        validate(method, hf)
    except ValueError:
        pass
    else:
        raise AssertionError((method, hf))
""",
        bootstrap=True,
        cpu=True,
    )


@pytest.fixture
def model_runner(monkeypatch):
    """Exercise the actual pool hooks with no GPU operations or SDK imports."""
    name = "sglang_simulator.simulation.sglang.model_runner"
    if importlib.util.find_spec("torch") is None:
        # The module only needs nn.Module at definition time. Restore both
        # modules afterwards so later native-runtime tests never inherit a stub.
        torch = ModuleType("torch")
        torch.nn = SimpleNamespace(Module=object)
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.delitem(sys.modules, name, raising=False)
        module = importlib.import_module(name)
        monkeypatch.setitem(sys.modules, name, module)
        return module
    return importlib.import_module(name)


class LinearPool:
    def __init__(self, *, cell_bytes=3, fixed_bytes=11, alignment=None):
        self.cell_bytes = cell_bytes
        self.fixed_bytes = fixed_bytes
        self.alignment = alignment
        self.calls = 0

    def calculate_pool_sizes(self, budget_bytes, page_size):
        self.calls += 1
        if budget_bytes < self.fixed_bytes:
            raise ValueError("cannot hold even one request")
        alignment = self.alignment or page_size
        tokens = (budget_bytes - self.fixed_bytes) // self.cell_bytes
        return SimpleNamespace(max_total_num_tokens=tokens // alignment * alignment)


@pytest.mark.parametrize("prepared", [False, True])
def test_mock_forward_attaches_only_its_own_opted_in_snapshot(
    model_runner, monkeypatch, prepared
):
    from sglang_simulator.simulation.sglang.prepared_batch import (
        SNAPSHOT_ATTRIBUTE,
        predictor_batch_from_result,
    )
    from sglang_simulator.simulation.sglang.scheduler import C_SchedulerHook

    monkeypatch.setattr(
        C_SchedulerHook,
        "INFERENCE_PREDICTOR",
        SimpleNamespace(requires_prepared_batch=prepared),
    )
    logits_module = ModuleType("sglang.srt.layers.logits_processor")
    logits_module.LogitsProcessorOutput = SimpleNamespace
    native_runner = ModuleType("sglang.srt.model_executor.model_runner")
    native_runner.ModelRunnerOutput = SimpleNamespace
    monkeypatch.setitem(sys.modules, logits_module.__name__, logits_module)
    monkeypatch.setitem(sys.modules, native_runner.__name__, native_runner)
    target_type = type("Target", (), {})
    model_runner.C_ModelRunnerHook.hook(target_type)
    target = target_type()
    target.model_config = SimpleNamespace(vocab_size=16)
    target.device = "cpu"
    batch = SimpleNamespace(
        forward_mode=SimpleNamespace(name="MIXED"),
        batch_size=2,
        seq_lens_cpu=[101, 8],
        extend_seq_lens_cpu=[1, 8],
        extend_prefix_lens_cpu=[100, 0],
    )

    def allocate(**kwargs):
        batch.seq_lens_cpu[:] = [2, 2]
        batch.extend_seq_lens_cpu[:] = [2, 2]
        batch.extend_prefix_lens_cpu[:] = [0, 0]
        return SimpleNamespace(shape=kwargs["size"])

    monkeypatch.setattr(model_runner.torch, "empty", allocate, raising=False)
    first = target.forward(forward_batch=batch)
    second = target.forward(batch)
    if prepared:
        assert predictor_batch_from_result(
            first, expected_batch_size=2
        ).request_info() == [
            [1, 100],
            [8, 0],
        ]
        assert predictor_batch_from_result(
            second, expected_batch_size=2
        ).request_info() == [
            [2, 0],
            [2, 0],
        ]
    else:
        assert not hasattr(first.logits_output, SNAPSHOT_ATTRIBUTE)
        assert not hasattr(second.logits_output, SNAPSHOT_ATTRIBUTE)
        legacy = target.forward(SimpleNamespace(batch_size=2))
        assert not hasattr(legacy.logits_output, SNAPSHOT_ATTRIBUTE)


def test_native_pool_budget_inverts_exact_aligned_capacity(model_runner):
    pool = LinearPool()
    budget = model_runner._native_pool_budget(pool, 16, 4, require_exact=True)
    assert budget == 59  # Fixed 11 B + sixteen native rows at 3 B each.
    assert pool.calculate_pool_sizes(budget, 4).max_total_num_tokens == 16
    assert pool.calculate_pool_sizes(budget - 1, 4).max_total_num_tokens == 12


def test_native_pool_budget_rejects_zero_cell_constant_capacity(model_runner):
    class ZeroCellPool:
        calls = 0

        def calculate_pool_sizes(self, budget_bytes, page_size):
            self.calls += 1
            # Native zero-cell layouts can return context_len for every budget.
            return SimpleNamespace(max_total_num_tokens=8)

    pool = ZeroCellPool()
    with pytest.raises(ValueError, match="cannot represent"):
        model_runner._native_pool_budget(pool, 16, 4, require_exact=True)
    assert pool.calls == 64


def test_native_pool_budget_rejects_unreachable_rounded_target(model_runner):
    pool = LinearPool(fixed_bytes=0, alignment=8)
    with pytest.raises(ValueError, match="exact InferCast capacity"):
        model_runner._native_pool_budget(pool, 12, 4, require_exact=True)


@pytest.mark.parametrize("tokens", [True, 0, 1, 6, 8.0])
def test_native_pool_budget_rejects_invalid_exact_capacity(model_runner, tokens):
    pool = LinearPool()
    with pytest.raises(ValueError, match="page-aligned"):
        model_runner._native_pool_budget(pool, tokens, 4, require_exact=True)
    assert pool.calls == 0


def test_native_pool_budget_preserves_legacy_undersized_errors(model_runner):
    class Pool(LinearPool):
        def calculate_pool_sizes(self, budget_bytes, page_size):
            if budget_bytes < self.fixed_bytes:
                raise RuntimeError("undersized probe")
            return super().calculate_pool_sizes(budget_bytes, page_size)

    assert model_runner._native_pool_budget(Pool(), 13, 4) == 59
    with pytest.raises(RuntimeError, match="undersized probe"):
        model_runner._native_pool_budget(Pool(), 16, 4, require_exact=True)


@pytest.mark.parametrize("quantization", [None, "fp8"])
def test_resource_pool_guard_accepts_plain_dense_and_fp8_mla(
    model_runner, quantization
):
    configurator = SimpleNamespace(
        server_args=SimpleNamespace(json_model_override_args="{}"),
        model_config=SimpleNamespace(
            quantization=quantization,
            hf_text_config=SimpleNamespace(
                quantization_config=(
                    {"quant_method": quantization} if quantization else None
                )
            ),
        ),
        sliding_window_size=-1,
    )
    model_runner._validate_resource_pool(configurator)


@pytest.mark.parametrize(
    "override",
    [
        {"enable_lora": True},
        {"speculative_algorithm": "EAGLE"},
        {"enable_hisparse": True},
        {"enable_unified_memory": True},
        {"enable_kv_cache_sharding": True},
        {"json_model_override_args": '{"hidden_size": 256}'},
    ],
)
def test_resource_pool_guard_rejects_unmodeled_runtime_allocations(
    model_runner, override
):
    configurator = SimpleNamespace(
        server_args=SimpleNamespace(**override),
        model_config=SimpleNamespace(
            quantization=None,
            hf_text_config=SimpleNamespace(quantization_config=None),
        ),
    )
    with pytest.raises(ValueError, match="does not model"):
        model_runner._validate_resource_pool(configurator)


@pytest.mark.parametrize("dtype", ["mxfp8", "nvfp4", "fp4_mx_block16"])
def test_resource_pool_guard_rejects_packed_cache_metadata(model_runner, dtype):
    configurator = SimpleNamespace(
        server_args=SimpleNamespace(),
        model_config=SimpleNamespace(
            quantization=None,
            hf_text_config=SimpleNamespace(quantization_config=None),
        ),
        kv_cache_dtype_str=dtype,
    )
    with pytest.raises(ValueError, match="packed KV dtype"):
        model_runner._validate_resource_pool(configurator)


@pytest.fixture
def pool_hook(model_runner, monkeypatch):
    class Target:
        def configure(self, *args, **kwargs):
            pass

        def _init_pools(self, *args, **kwargs):
            pass

    pool = LinearPool()
    native = ModuleType("sglang.srt.model_executor.pool_configurator")
    native.create_memory_pool_configurator = lambda configurator: pool
    monkeypatch.setitem(sys.modules, native.__name__, native)
    model_runner.C_KVCacheConfiguratorHook.hook(Target)
    target = Target()
    target.server_args = SimpleNamespace(max_total_tokens=16, page_size=4)
    target.page_size = 4
    return target


def test_explicit_capacity_bypasses_sdk_and_clears_budget_metrics(
    pool_hook, model_runner, monkeypatch
):
    from sglang_simulator.simulation.manager.config import ConfigManager

    monkeypatch.setattr(
        ConfigManager, "_raw_config", {"memory_budget": {"invalid": True}}
    )
    monkeypatch.setattr(ConfigManager, "_memory_budget_metrics", {"stale": True})
    original_import = __import__

    def no_sdk(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"infercast", "aiconfigurator"}:
            raise AssertionError(
                "explicit token capacity must not import a predictor SDK"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", no_sdk)
    assert pool_hook._profile_available_bytes(0) == 59
    assert ConfigManager.get_memory_budget_metrics() is None


def test_without_budget_preserves_legacy_capacity_estimator(
    pool_hook, model_runner, monkeypatch
):
    from sglang_simulator.simulation.manager.config import ConfigManager
    from sglang_simulator.simulation.types import SchedulerConfig

    pool_hook.server_args.max_total_tokens = None
    pool_hook.model_config = SimpleNamespace()
    pool_hook.mambaish_config = None
    model = SimpleNamespace()
    hardware = SimpleNamespace()
    physical = SchedulerConfig(tp_size=1, ep_size=1)
    monkeypatch.setattr(ConfigManager, "_raw_config", {"scheduler": {"tp_size": 8}})
    monkeypatch.setattr(ConfigManager, "_model_info", model)
    monkeypatch.setattr(
        ConfigManager, "get_accelerator_info", classmethod(lambda cls: hardware)
    )
    monkeypatch.setattr(
        model_runner, "resolve_scheduler_config", lambda **kwargs: physical
    )
    observed = []

    def legacy_estimator(**kwargs):
        observed.append(kwargs)
        return 1234

    monkeypatch.setattr(
        model_runner, "profile_device_available_bytes", legacy_estimator
    )
    assert pool_hook._profile_available_bytes(0) == 1234
    assert observed == [
        {"model": model, "device": hardware, "scheduler_config": physical}
    ]
    assert physical.tp_size == 1


def test_target_config_is_a_copy_and_legacy_setter_still_mutates(monkeypatch):
    from sglang_simulator.simulation.manager.config import ConfigManager
    from sglang_simulator.simulation.types import SchedulerConfig
    from sglang_simulator.spec import DataType

    physical = SchedulerConfig(tp_size=1, ep_size=1, kv_cache_data_type=DataType.BF16)
    monkeypatch.setattr(
        ConfigManager,
        "_raw_config",
        {"scheduler": {"tp_size": 8, "ep_size": 2, "kv_cache_data_type": "FP8"}},
    )
    monkeypatch.setattr(ConfigManager, "_scheduler_config", None)
    target = ConfigManager.get_target_scheduler_config(physical)
    assert target is not physical
    assert (target.tp_size, target.ep_size, target.kv_cache_data_type) == (
        8,
        2,
        DataType.FP8,
    )
    assert (physical.tp_size, physical.ep_size, physical.kv_cache_data_type) == (
        1,
        1,
        DataType.BF16,
    )
    assert ConfigManager.get_scheduler_config() is None
    ConfigManager.set_scheduler_config(physical)
    assert ConfigManager.get_scheduler_config() is physical
    assert (physical.tp_size, physical.ep_size) == (8, 2)


def test_config_cache_reset_clears_memory_budget_metrics(monkeypatch):
    from sglang_simulator.simulation.manager.config import ConfigManager

    for name in (
        "_raw_config",
        "_model_info",
        "_platform_config",
        "_scheduler_config",
        "_memory_budget_metrics",
    ):
        monkeypatch.setattr(ConfigManager, name, {"previous": True})
    ConfigManager.reset_config_cache()
    assert ConfigManager.get_memory_budget_metrics() is None
    assert ConfigManager.get_scheduler_config() is None
    assert ConfigManager.get_model_info() is None
