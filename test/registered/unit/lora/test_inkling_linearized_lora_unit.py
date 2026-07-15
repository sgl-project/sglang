"""Regression tests for Inkling's linearized shared-sink LoRA path.

The production LoRA module has optional GPU/runtime imports that are unavailable
in lightweight unit-test environments. The tests compile selected production
methods directly so every tensor operation remains the real implementation.
"""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

REPO_ROOT = Path(__file__).resolve().parents[4]
LORA_LAYERS_PATH = REPO_ROOT / "python/sglang/srt/lora/layers.py"
LORA_MANAGER_PATH = REPO_ROOT / "python/sglang/srt/lora/lora_manager.py"
INKLING_UTIL_PATH = REPO_ROOT / "python/sglang/srt/models/inkling_common/util.py"
DENSE_MLP_PATH = REPO_ROOT / "python/sglang/srt/models/inkling_common/dense_mlp.py"
INKLING_LAYER_PATH = REPO_ROOT / "python/sglang/srt/models/inkling_common/lora.py"
INKLING_DENSE_PATH = (
    REPO_ROOT / "python/sglang/srt/lora/trtllm_lora_temp/inkling_dense.py"
)


class _Flag:
    def __init__(self, value: bool, *, is_set: bool = False):
        self.value = value
        self.explicitly_set = is_set

    def get(self) -> bool:
        return self.value

    def is_set(self) -> bool:
        return self.explicitly_set


class _RefreshableSharedSink:
    is_shared_fused_moe = True

    def __init__(self, callback):
        self._callback = callback

    def on_lora_slots_updated(self, slot_ids):
        self._callback(slot_ids)


def _load_batch_dense_lora_class(monkeypatch):
    """Import the permanent Inkling LoRA layer with lightweight dependencies."""
    impl = _load_inkling_dense_impl()
    side_streams: dict[torch.cuda.Stream, torch.cuda.Stream] = {}

    def get_lora_side_stream():
        consumer_stream = torch.cuda.current_stream()
        if consumer_stream not in side_streams:
            side_streams[consumer_stream] = torch.cuda.Stream()
        return side_streams[consumer_stream]

    _stub_module(
        monkeypatch,
        "sglang.srt.lora.backend.base_backend",
        BaseLoRABackend=object,
    )
    _stub_module(
        monkeypatch,
        "sglang.srt.models.inkling_common.dense_mlp",
        InklingBatchDenseMLP=_FakeSink,
    )
    temp_package = _stub_module(
        monkeypatch,
        "sglang.srt.lora.trtllm_lora_temp",
        get_lora_side_stream=get_lora_side_stream,
    )
    temp_package.__path__ = [str(INKLING_DENSE_PATH.parent)]
    _stub_module(
        monkeypatch,
        "sglang.srt.lora.trtllm_lora_temp.inkling_dense",
        forward_with_lora=impl.forward_with_lora,
    )
    _stub_module(monkeypatch, "sglang.srt.models.inkling_common")
    module_name = "sglang.srt.models.inkling_common.lora"
    spec = __import__("importlib.util").util.spec_from_file_location(
        module_name, INKLING_LAYER_PATH
    )
    module = __import__("importlib.util").util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(
        sys.modules["sglang.srt.models.inkling_common"], "lora", module, raising=False
    )
    spec.loader.exec_module(module)
    return module.InklingBatchDenseMLPWithLoRA


def _load_bf16_materialization_class():
    return _load_selected_class_methods(
        DENSE_MLP_PATH,
        "InklingBatchDenseMLP",
        {
            "weight_loader_fused",
            "process_weights_after_loading",
            "get_bf16_linearized_weights",
            "_refresh_bf16_linearized",
        },
        {
            "torch": torch,
            "FusedMoELoadingMixin": SimpleNamespace(
                weight_loader_fused=lambda _self, param, loaded, *_: param.data.copy_(
                    loaded
                )
            ),
            "logger": SimpleNamespace(
                info=lambda *args: None, info_once=lambda *args: None
            ),
            "SharedExpertFp4Strategy": SimpleNamespace(FP4=object()),
        },
    )


def _load_inkling_dense_impl():
    function_names = {
        "_apply_per_expert_lora",
        "_shared_sink_routing",
        "apply_multi_lora",
        "forward_with_lora",
    }
    tree = ast.parse(INKLING_DENSE_PATH.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in function_names
    ]
    assert {function.name for function in functions} == function_names
    namespace = {
        "torch": torch,
        "envs": SimpleNamespace(
            SGLANG_OPT_USE_INKLING_MULTI_STREAM_OVERLAP=_Flag(True)
        ),
        "symm_mem_all_reduce": lambda value, _group: value,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
            str(INKLING_DENSE_PATH),
            "exec",
        ),
        namespace,
    )
    return SimpleNamespace(
        **{function_name: namespace[function_name] for function_name in function_names},
    )


def _load_selected_class_methods(path, class_name, method_names, namespace):
    """Compile selected production methods into a dependency-free test class."""
    tree = ast.parse(path.read_text())
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node
        for node in source_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in method_names
    ]
    assert {method.name for method in methods} == set(method_names)
    test_class = ast.ClassDef(
        name=f"_{class_name}MethodsUnderTest",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module_ast = ast.fix_missing_locations(
        ast.Module(body=[test_class], type_ignores=[])
    )
    exec(compile(module_ast, str(path), "exec"), namespace)
    return namespace[test_class.name]


def _load_function(path, function_name, namespace):
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    module_ast = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module_ast, str(path), "exec"), namespace)
    return namespace[function_name]


def _load_manager_methods(method_names, namespace=None):
    return _load_selected_class_methods(
        LORA_MANAGER_PATH, "LoRAManager", method_names, namespace or {}
    )


def _stub_module(monkeypatch, name: str, **attributes):
    """Install a small importable module hierarchy for a production-file load."""
    parts = name.split(".")
    for end in range(1, len(parts)):
        package_name = ".".join(parts[:end])
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, package_name, package)
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    if len(parts) > 1:
        parent = sys.modules[".".join(parts[:-1])]
        monkeypatch.setattr(parent, parts[-1], module, raising=False)
    return module


def _load_inkling_util(monkeypatch, state):
    class _Dummy:
        pass

    for module_name, symbol in (
        ("sglang.srt.layers.moe.fused_moe_triton.layer", "FusedMoE"),
        ("sglang.srt.layers.moe.moe_runner.base", "MoeRunnerConfig"),
        ("sglang.srt.layers.quantization.base_config", "QuantizationConfig"),
        ("sglang.srt.layers.quantization.unquant", "UnquantizedFusedMoEMethod"),
    ):
        _stub_module(monkeypatch, module_name, **{symbol: _Dummy})
    _stub_module(
        monkeypatch,
        "sglang.srt.server_args",
        get_global_server_args=lambda: state.args,
    )
    _stub_module(monkeypatch, "sglang.srt.environ", envs=state.envs)
    _stub_module(
        monkeypatch,
        "sglang.srt.layers.moe",
        get_moe_runner_backend=lambda: None,
    )

    module_name = "_inkling_linearized_util_under_test"
    spec = __import__("importlib.util").util.spec_from_file_location(
        module_name, INKLING_UTIL_PATH
    )
    module = __import__("importlib.util").util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    (
        "enable_lora",
        "interleaved",
        "serves_fp4",
        "expected_fused",
    ),
    [
        (False, True, False, False),
        (False, False, False, True),
        (False, True, True, True),
        (True, True, False, False),
    ],
)
def test_linearized_sink_config_eligibility(
    monkeypatch,
    enable_lora,
    interleaved,
    serves_fp4,
    expected_fused,
):
    state = SimpleNamespace(
        args=SimpleNamespace(enable_lora=enable_lora),
        envs=SimpleNamespace(
            SGLANG_OPT_USE_INKLING_SHARED_FUSED_MOE=_Flag(False),
        ),
    )
    util = _load_inkling_util(monkeypatch, state)

    assert (
        util.use_inkling_shared_fused_moe(
            inference_moe_w13_interleaved=interleaved,
            shared_sink_serves_fp4=serves_fp4,
        )
        is expected_fused
    )


@pytest.mark.parametrize("override", [False, True])
def test_lora_ignores_fused_shared_expert_override(monkeypatch, override):
    state = SimpleNamespace(
        args=SimpleNamespace(enable_lora=True),
        envs=SimpleNamespace(
            SGLANG_OPT_USE_INKLING_SHARED_FUSED_MOE=_Flag(override, is_set=True),
        ),
    )
    util = _load_inkling_util(monkeypatch, state)
    assert not util.use_inkling_shared_fused_moe()


class _FakeSink(nn.Module):
    def __init__(self, *, hidden_size=3, num_experts=2, expert_size=2, seed=7):
        super().__init__()
        self.moe_tp_size = 1
        self.moe_tp_rank = 0
        self.intermediate_size_per_partition = expert_size
        self.n_shared_experts = num_experts
        self.layer_id = 0
        self.inference_moe_w13_interleaved = True
        self._linearized_bf16_enabled = True
        self._fp4_strategy = SimpleNamespace(serves_fp4=False)
        self.tp_group = None
        generator = torch.Generator().manual_seed(seed)
        self._w13_lin = torch.randn(
            num_experts * 2 * expert_size, hidden_size, generator=generator
        )
        self._w2_lin = torch.randn(
            num_experts * expert_size, hidden_size, generator=generator
        )
        self.seen_gammas = []

    def get_bf16_linearized_weights(self):
        return self._w13_lin, self._w2_lin

    def _swiglu(self, gate_up, gammas):
        self.seen_gammas.append(gammas.detach().clone())
        gate = gate_up[..., 0::2]
        up = gate_up[..., 1::2]
        return F.silu(gate) * up * gammas.unsqueeze(-1)

    def _forward_bf16_linearized(
        self, x_td, gammas_ts, linearized_weights, use_reduce_scatter
    ):
        w13_lin, w2_lin = linearized_weights
        t = x_td.shape[0]
        y = torch.mm(x_td, w13_lin.T).view(t, self.n_shared_experts, -1)
        act = self._swiglu(y, gammas_ts)
        return torch.mm(act.reshape(t, -1), w2_lin)

    def forward(self, x, gammas, use_reduce_scatter=False):
        x_td = x.view(-1, x.size(-1)) if x.ndim != 2 else x
        gammas_ts = gammas.view(-1, gammas.size(-1)) if gammas.ndim != 2 else gammas
        out_td = self._forward_bf16_linearized(
            x_td,
            gammas_ts,
            self.get_bf16_linearized_weights(),
            use_reduce_scatter,
        )
        return out_td.view_as(x) if x.ndim == 2 else out_td


def test_manager_promotes_dense_sink_in_place(monkeypatch):
    lora_cls = _load_batch_dense_lora_class(monkeypatch)
    manager_cls = _load_manager_methods(
        {"init_lora_modules"},
        {
            "BaseLayerWithLoRA": nn.Module,
            "Dict": dict,
            "FusedMoE": type("_UnusedFusedMoE", (), {}),
            "List": list,
            "Optional": Optional,
            "ParallelLMHead": type("_UnusedParallelLMHead", (), {}),
            "VocabParallelEmbedding": type("_UnusedEmbedding", (), {}),
            "get_layer_id": lambda _name: 0,
            "torch": torch,
        },
    )
    layer = _FakeSink()
    backend = SimpleNamespace(
        max_loras_per_batch=1,
        name="torch-test",
        is_moe_lora=False,
    )
    module_name = "model.layers.0.mlp.shared_experts"
    manager = manager_cls()
    manager.base_hf_config = SimpleNamespace(num_hidden_layers=1)
    manager.base_model = SimpleNamespace(named_modules=lambda: [(module_name, layer)])
    manager.target_modules = {"gate_up_proj", "down_proj"}
    manager.lora_backend = backend

    manager.init_lora_modules()

    promoted = manager.lora_modules[0][module_name]
    assert promoted is layer
    assert type(layer) is lora_cls
    assert layer.is_shared_fused_moe is True
    assert layer.lora_backend is backend
    assert backend.is_moe_lora is True


def _make_pool(*, slots: int, max_rank: int, active_rank: int, scale: float):
    """Build the real shared-outer memory-pool layouts with zero rank padding."""
    n, f, hidden = 2, 2, 3
    gate_a = torch.zeros(slots, 1, 2 * max_rank, hidden)
    gate_b = torch.zeros(slots, n, 2 * f, max_rank)
    down_a = torch.zeros(slots, n, max_rank, f)
    down_b = torch.zeros(slots, 1, hidden, max_rank)

    base = torch.arange(1, active_rank * hidden + 1, dtype=torch.float32).view(
        active_rank, hidden
    )
    for slot in range(slots):
        slot_scale = scale * (slot + 1)
        gate_a[slot, 0, :active_rank] = base * (0.03 * slot_scale)
        gate_a[slot, 0, max_rank : max_rank + active_rank] = base * (-0.02 * slot_scale)
        gate_b[slot, ..., :active_rank] = 0.05 * slot_scale
        down_a[slot, ..., :active_rank, :] = 0.04 * slot_scale
        down_b[slot, ..., :active_rank] = -0.06 * slot_scale
    return gate_a, gate_b, down_a, down_b


def _install_capture_mode(monkeypatch, capture_state):
    _stub_module(
        monkeypatch,
        "sglang.srt.model_executor.runner_utils.capture_mode",
        get_is_capture_mode=lambda: capture_state.value,
    )


def _make_layer(monkeypatch, *, slots=1, max_rank=2, active_rank=2, scale=1.0):
    capture_state = SimpleNamespace(value=False)
    _install_capture_mode(monkeypatch, capture_state)
    batch_info = SimpleNamespace(
        has_active_lora=False,
        lora_ranks=[active_rank],
        moe_lora_info=SimpleNamespace(
            token_lora_mapping=torch.tensor([-1, -1], dtype=torch.int32)
        ),
    )
    backend = SimpleNamespace(
        name="triton" if slots > 1 else "torch-test",
        batch_info=batch_info,
        max_loras_per_batch=slots,
        is_moe_lora=False,
    )
    layer = _load_batch_dense_lora_class(monkeypatch)()
    layer.initialize_lora(backend)
    adapter_values = _make_pool(
        slots=slots, max_rank=max_rank, active_rank=active_rank, scale=scale
    )
    pool = tuple(torch.zeros_like(tensor) for tensor in adapter_values)
    layer.set_lora_info(*pool)
    _replace_slot(pool, adapter_values)
    layer.on_lora_slots_updated(None)
    return layer, pool, batch_info, capture_state


def _forward(layer, batch_info, *, active: bool, mapping):
    batch_info.has_active_lora = active
    moe_lora_info = getattr(batch_info, "moe_lora_info", None)
    if moe_lora_info is not None:
        mapping_tensor = torch.as_tensor(mapping, dtype=torch.int32).flatten()
        if mapping_tensor.numel() == 1:
            moe_lora_info.token_lora_mapping.fill_(mapping_tensor.item())
        else:
            moe_lora_info.token_lora_mapping.copy_(mapping_tensor)
    x = torch.tensor([[0.5, -1.0, 0.25], [1.25, 0.75, -0.5]])
    gammas = torch.tensor([[0.2, 0.8], [0.65, 0.35]])
    output = layer(x, gammas=gammas)
    return output, gammas


def _replace_slot(pool, replacement):
    with torch.no_grad():
        for target, source in zip(pool, replacement):
            target.copy_(source)


def _clear_slot(pool):
    """Mirror production None loading by zeroing both factors."""
    with torch.no_grad():
        for tensor in pool:
            tensor.zero_()


def test_bf16_materialization_and_w2_reload_refresh_stable_storage():
    layer = _load_bf16_materialization_class()()
    layer._linearized_bf16_enabled = True
    layer._fp4_strategy = object()
    layer._bf16_linearized_ready = False
    layer.n_shared_experts = 2
    layer.w13_weight = nn.Parameter(torch.arange(24.0).view(2, 4, 3))
    layer.w2_weight = nn.Parameter(torch.arange(12.0).view(2, 3, 2))
    layer._w2_lin = torch.empty(4, 3)

    layer.process_weights_after_loading()
    w13, w2 = layer.get_bf16_linearized_weights()
    torch.testing.assert_close(w13, layer.w13_weight.view(8, 3))
    torch.testing.assert_close(
        w2, layer.w2_weight.detach().transpose(1, 2).reshape(4, 3)
    )
    storage = w2.data_ptr()

    replacement = layer.w2_weight.detach().add(100)
    layer.weight_loader_fused(layer.w2_weight, replacement, "w2_weight", "w2")
    assert layer._w2_lin.data_ptr() == storage
    torch.testing.assert_close(layer._w2_lin, replacement.transpose(1, 2).reshape(4, 3))


def test_capture_like_base_adapter_base_replay_and_direct_gammas(monkeypatch):
    layer, pool, batch_info, capture_state = _make_layer(monkeypatch)
    capture_state.value = True

    adapter_pool = tuple(tensor.clone() for tensor in pool)
    _clear_slot(pool)
    layer.on_lora_slots_updated(None)
    assert torch.count_nonzero(layer._w1_delta) == 0
    assert torch.count_nonzero(layer._a_cat) == 0
    base_before, gammas = _forward(layer, batch_info, active=False, mapping=-1)
    _replace_slot(pool, adapter_pool)
    layer.on_lora_slots_updated(None)
    adapter, _ = _forward(layer, batch_info, active=True, mapping=0)
    _clear_slot(pool)
    layer.on_lora_slots_updated(None)
    base_after, _ = _forward(layer, batch_info, active=False, mapping=-1)

    torch.testing.assert_close(base_before, base_after, rtol=0, atol=0)
    assert not torch.allclose(adapter, base_before)
    for seen in layer.seen_gammas:
        torch.testing.assert_close(seen, gammas, rtol=0, atol=0)


def test_adapter_hot_swap_refreshes_in_place_for_graph_replay(monkeypatch):
    layer, pool, batch_info, capture_state = _make_layer(monkeypatch, scale=1.0)
    capture_state.value = True
    adapter_a, _ = _forward(layer, batch_info, active=True, mapping=0)
    pointers_before = (layer._w1_delta.data_ptr(), layer._a_cat.data_ptr())
    contents_before = (layer._w1_delta.clone(), layer._a_cat.clone())

    adapter_b_pool = _make_pool(slots=1, max_rank=2, active_rank=2, scale=2.5)
    _replace_slot(pool, adapter_b_pool)
    layer.on_lora_slots_updated(None)
    adapter_b, _ = _forward(layer, batch_info, active=True, mapping=0)

    assert (layer._w1_delta.data_ptr(), layer._a_cat.data_ptr()) == pointers_before
    assert not torch.equal(layer._w1_delta, contents_before[0])
    assert not torch.equal(layer._a_cat, contents_before[1])
    assert not torch.allclose(adapter_a, adapter_b)


def test_slot_update_hook_only_refreshes_changed_slots(monkeypatch):
    layer, pool, _, _ = _make_layer(monkeypatch, slots=3)
    pointers = (layer._w1_delta.data_ptr(), layer._a_cat.data_ptr())
    running_before = (layer._w1_delta[:2].clone(), layer._a_cat[:2].clone())
    changed_before = (layer._w1_delta[2].clone(), layer._a_cat[2].clone())

    replacement = _make_pool(slots=3, max_rank=2, active_rank=2, scale=3.0)
    with torch.no_grad():
        for target, source in zip(pool, replacement):
            target[:2].add_(10)
            target[2].copy_(source[2])
    layer.on_lora_slots_updated({2})

    assert (layer._w1_delta.data_ptr(), layer._a_cat.data_ptr()) == pointers
    torch.testing.assert_close(layer._w1_delta[:2], running_before[0], rtol=0, atol=0)
    torch.testing.assert_close(layer._a_cat[:2], running_before[1], rtol=0, atol=0)
    assert not torch.equal(layer._w1_delta[2], changed_before[0])
    assert not torch.equal(layer._a_cat[2], changed_before[1])


def test_rank_smaller_than_max_rank_matches_compact_rank(monkeypatch):
    padded, _, padded_info, padded_capture = _make_layer(
        monkeypatch, max_rank=3, active_rank=1, scale=1.3
    )
    padded_capture.value = True
    padded_output, _ = _forward(padded, padded_info, active=True, mapping=0)

    compact, _, compact_info, compact_capture = _make_layer(
        monkeypatch, max_rank=1, active_rank=1, scale=1.3
    )
    compact_capture.value = True
    compact_output, _ = _forward(compact, compact_info, active=True, mapping=0)

    torch.testing.assert_close(padded_output, compact_output, rtol=1e-5, atol=1e-6)


def _selected_slot_reference(layer, mapping):
    x = layer._w13_lin.new_tensor([[0.5, -1.0, 0.25], [1.25, 0.75, -0.5]])
    gammas = layer._w13_lin.new_tensor([[0.2, 0.8], [0.65, 0.35]])
    t = x.shape[0]
    n = layer.n_shared_experts
    y = torch.mm(x, layer._w13_lin.T).view(t, n, -1)
    for token, slot in enumerate(mapping):
        if slot >= 0:
            shrink = torch.mm(
                x[token : token + 1], layer.gate_up_lora_a_weights[slot, 0].T
            )
            y[token : token + 1] += torch.mm(shrink, layer._w1_delta[slot].T).view(
                1, n, -1
            )
    gate = y[..., 0::2]
    up = y[..., 1::2]
    act = F.silu(gate) * up * gammas.unsqueeze(-1)
    out = torch.mm(act.reshape(t, -1), layer._w2_lin)
    for token, slot in enumerate(mapping):
        if slot >= 0:
            shrink = torch.mm(
                act[token : token + 1].reshape(1, -1), layer._a_cat[slot].T
            )
            out[token : token + 1] += torch.mm(
                shrink, layer.down_lora_b_weights[slot, 0].T
            )
    return out


def test_dense_sink_tp_slices_and_flat_factor_normalization(monkeypatch):
    layer, _, _, _ = _make_layer(monkeypatch)
    layer.moe_tp_size = 2
    layer.intermediate_size_per_partition = 2
    n, rank, full_intermediate = layer.n_shared_experts, 2, 4
    down_a = torch.arange(n * rank * full_intermediate).view(n, rank, full_intermediate)
    gate_up_b = torch.arange(n * 2 * full_intermediate * rank).view(
        n, 2 * full_intermediate, rank
    )
    expected_b = torch.stack(
        [torch.cat([weight[2:4], weight[6:8]], dim=0) for weight in gate_up_b]
    )

    for a, b in (
        (down_a, gate_up_b),
        (down_a.transpose(0, 1).reshape(rank, -1), gate_up_b.reshape(-1, rank)),
    ):
        torch.testing.assert_close(
            layer.slice_moe_lora_a_weights(a, 1, "down_proj_moe"), down_a[..., 2:4]
        )
        torch.testing.assert_close(
            layer.slice_moe_lora_b_weights(b, 1, "gate_up_proj_moe"), expected_b
        )

    hidden_size = layer.gate_up_lora_a_weights.shape[-1]
    gate_a = torch.zeros(2 * rank, hidden_size)
    down_b = torch.zeros(hidden_size, rank)
    assert layer.slice_moe_lora_a_weights(gate_a, 1, "gate_up_proj_moe").shape == (
        1,
        2 * rank,
        hidden_size,
    )
    assert layer.slice_moe_lora_b_weights(down_b, 1, "down_proj_moe").shape == (
        1,
        hidden_size,
        rank,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("slots", [1, 2, 4, 5, 8, 16])
def test_multi_slot_cuda_graph_replay(monkeypatch, slots):
    from sglang.srt.lora.backend.triton_backend import TritonLoRABackend
    from sglang.srt.lora.utils import LoRABatchInfo, MoELoRABatchInfo

    capture_state = SimpleNamespace(value=True)
    _install_capture_mode(monkeypatch, capture_state)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    rank = 1
    max_rank = 3
    moe_info = MoELoRABatchInfo(
        seg_indptr=torch.tensor([0, 1, 2], device=device, dtype=torch.int32),
        req_to_lora=torch.tensor([0, slots - 1], device=device, dtype=torch.int32),
        adapter_enabled=torch.ones(slots, device=device, dtype=torch.int32),
        token_lora_mapping=torch.tensor(
            [0, slots - 1], device=device, dtype=torch.int32
        ),
    )
    batch_info = LoRABatchInfo(
        use_cuda_graph=True,
        bs=2,
        num_segments=2,
        seg_indptr=moe_info.seg_indptr,
        weight_indices=moe_info.req_to_lora,
        lora_ranks=torch.full((slots,), rank, device=device, dtype=torch.int32),
        scalings=torch.full((slots,), 9.0, device=device),
        max_len=1,
        seg_lens=torch.ones(2, device=device, dtype=torch.int32),
        permutation=None,
        req_seg_indptr=moe_info.seg_indptr,
        req_weight_indices=moe_info.req_to_lora,
        moe_lora_info=moe_info,
        has_active_lora=True,
    )
    backend = TritonLoRABackend(max_loras_per_batch=slots, device=device)
    backend.batch_info = batch_info

    layer = _load_batch_dense_lora_class(monkeypatch)()
    layer._w13_lin = layer._w13_lin.to(device=device, dtype=dtype)
    layer._w2_lin = layer._w2_lin.to(device=device, dtype=dtype)
    layer.initialize_lora(backend)
    pool = tuple(
        tensor.to(device=device, dtype=dtype)
        for tensor in _make_pool(
            slots=slots, max_rank=max_rank, active_rank=rank, scale=1.0
        )
    )
    layer.set_lora_info(*pool)
    x = layer._w13_lin.new_tensor([[0.5, -1.0, 0.25], [1.25, 0.75, -0.5]])
    gammas = layer._w13_lin.new_tensor([[0.2, 0.8], [0.65, 0.35]])

    for _ in range(3):
        layer(x, gammas=gammas)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = layer(x, gammas=gammas)
    graph.replay()
    torch.testing.assert_close(
        graph_output,
        _selected_slot_reference(layer, [0, slots - 1]),
        rtol=2e-2,
        atol=2e-2,
    )
    if slots == 1:
        return

    batch_info.weight_indices.copy_(
        torch.tensor([slots - 1, 0], device=device, dtype=torch.int32)
    )
    batch_info.lora_ranks[0] = 0
    moe_info.token_lora_mapping.copy_(
        torch.tensor([slots - 1, -1], device=device, dtype=torch.int32)
    )
    graph.replay()
    torch.testing.assert_close(
        graph_output,
        _selected_slot_reference(layer, [slots - 1, -1]),
        rtol=2e-2,
        atol=2e-2,
    )

    batch_info.lora_ranks[0] = rank
    batch_info.weight_indices.copy_(
        torch.tensor([0, 1], device=device, dtype=torch.int32)
    )
    moe_info.token_lora_mapping.copy_(
        torch.tensor([0, 1], device=device, dtype=torch.int32)
    )
    graph.replay()
    torch.testing.assert_close(
        graph_output,
        _selected_slot_reference(layer, [0, 1]),
        rtol=2e-2,
        atol=2e-2,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 is required",
)
def test_split_k_shrink_fp32_feeds_temp_bf16_expand(monkeypatch):
    monkeypatch.setenv("SGLANG_EXPERIMENTAL_LORA_OPTI", "1")
    monkeypatch.setenv("SGLANG_ENABLE_LORA_SHRINK_SPLIT_K", "1")
    monkeypatch.setenv("SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC", "1")
    monkeypatch.setenv("SGLANG_OPT_LORA_CUBLAS", "0")
    monkeypatch.setenv("SGLANG_OPT_LORA_CUBLAS_A", "0")
    monkeypatch.setenv("SGLANG_OPT_LORA_CUBLAS_B", "1")

    from sglang.kernels.ops.gemm.trtllm_lora_temp import sgemm_lora_a as triton_ops
    from sglang.srt.lora.trtllm_lora_temp import attention
    from sglang.srt.lora.utils import LoRABatchInfo

    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokens, input_dim, rank, output_dim = 256, 4096, 64, 128
    batch_info = LoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, tokens], device=device, dtype=torch.int32),
        weight_indices=torch.zeros(1, device=device, dtype=torch.int32),
        lora_ranks=torch.full((1,), rank, device=device, dtype=torch.int32),
        scalings=torch.full((1,), 0.5, device=device),
        max_len=tokens,
        seg_lens=torch.full((1,), tokens, device=device, dtype=torch.int32),
        permutation=None,
    )
    x = torch.ones(tokens, input_dim, device=device, dtype=dtype)
    a = torch.full((1, rank, input_dim), 1 / input_dim, device=device, dtype=dtype)
    b = torch.full((1, output_dim, rank), 1 / rank, device=device, dtype=dtype)
    shrink_dtypes = []
    original_shrink = triton_ops.sgemm_lora_a_fwd

    def record_shrink_dtype(*args, **kwargs):
        output = original_shrink(*args, **kwargs)
        shrink_dtypes.append(output.dtype)
        return output

    def reject_common_expand(**_kwargs):
        pytest.fail("two-stream attention must use the temporary expand kernel")

    class QuantMethod:
        @staticmethod
        def apply(_layer, inputs, bias=None):
            return torch.full(
                (inputs.shape[0], output_dim),
                0.25,
                device=inputs.device,
                dtype=inputs.dtype,
            )

    monkeypatch.setattr(triton_ops, "sgemm_lora_a_fwd", record_shrink_dtype)
    monkeypatch.setattr(attention, "is_two_stream_active", lambda _inputs: True)
    layer = SimpleNamespace(
        set_lora=True,
        base_layer=SimpleNamespace(
            input_is_parallel=True,
            tp_rank=0,
            tp_size=1,
            skip_bias_add=True,
            bias=None,
            reduce_results=False,
            quant_method=QuantMethod(),
        ),
        lora_backend=SimpleNamespace(
            _sgemm_info=lambda: batch_info,
            run_lora_b_sgemm=reject_common_expand,
        ),
        A_buffer=a,
        B_buffer=b,
    )

    output, output_bias = attention.row_parallel_lora_forward(layer, x)
    torch.cuda.synchronize()
    assert shrink_dtypes == [torch.float32]
    assert output.dtype == dtype
    assert output_bias is None
    torch.testing.assert_close(output, torch.full_like(output, 0.75), rtol=0, atol=0)


def test_fused_moe_wrapper_reports_local_expert_dimension(monkeypatch):
    _stub_module(
        monkeypatch,
        "sglang.srt.lora.lora_moe_runners",
        LoRAInfo=SimpleNamespace,
    )
    wrapper_cls = _load_selected_class_methods(
        LORA_LAYERS_PATH,
        "FusedMoEWithLoRA",
        {"_get_lora_info"},
        {},
    )
    moe_lora_info = SimpleNamespace(
        seg_indptr=torch.tensor([0, 2], dtype=torch.int32),
        req_to_lora=torch.tensor([0], dtype=torch.int32),
        adapter_enabled=torch.tensor([1], dtype=torch.int32),
        token_lora_mapping=torch.tensor([0, 0], dtype=torch.int32),
    )
    wrapper = wrapper_cls()
    wrapper._lora_runner_backend = SimpleNamespace(
        is_experimental_sgl_trtllm=lambda: True,
        is_experimental_sgl_marlin=lambda: False,
    )
    wrapper.lora_backend = SimpleNamespace(
        batch_info=SimpleNamespace(
            lora_ranks=torch.tensor([4], dtype=torch.int32),
            moe_lora_info=moe_lora_info,
            has_active_lora=True,
        ),
        moe_cg_buffers={"routing": object()},
    )
    wrapper.base_layer = SimpleNamespace(
        num_experts=128, num_local_experts=32, hidden_size=64
    )
    wrapper.gate_up_lora_a_weights = torch.empty(1, 1, 8, 64)
    wrapper.gate_up_lora_b_weights = torch.empty(1, 32, 16, 4)
    wrapper.down_lora_a_weights = torch.empty(1, 32, 4, 8)
    wrapper.down_lora_b_weights = torch.empty(1, 1, 64, 4)
    wrapper.experts_shared_outer_loras = True
    wrapper.lora_use_virtual_experts = True
    wrapper.tp_size = 4
    wrapper.tp_rank = 3

    info = wrapper._get_lora_info()

    assert info.num_experts == 32
    assert info.num_experts == wrapper.down_lora_a_weights.shape[1]
    assert info.max_lora_rank == 4
    assert info.has_active_lora is True


def test_manager_refresh_follows_slot_copy_and_only_runs_on_changes():
    manager_cls = _load_manager_methods(
        {"fetch_new_loras", "_notify_lora_slots_updated"},
        {"Optional": Optional},
    )
    events = []

    class _Pool:
        def __init__(self):
            self.uid_to_buffer_id = {}

        def prepare_lora_batch(self, *, cur_uids, **kwargs):
            events.append(("pool", set(cur_uids)))
            for uid in cur_uids:
                if uid not in self.uid_to_buffer_id:
                    used = set(self.uid_to_buffer_id.values())
                    slot = next((i for i in range(4) if i not in used), 1)
                    if slot == 1:
                        self.uid_to_buffer_id = {
                            resident: resident_slot
                            for resident, resident_slot in self.uid_to_buffer_id.items()
                            if resident_slot != slot
                        }
                    self.uid_to_buffer_id[uid] = slot

    refreshable = _RefreshableSharedSink(
        lambda slots: events.append(("refresh", set(slots)))
    )
    manager = manager_cls()
    manager.max_loras_per_batch = 4
    manager.memory_pool = _Pool()
    manager.loras = {"adapter-a": object(), "adapter-b": object()}
    manager.lora_modules = [{"sink": refreshable}]
    manager.lora_refs = {}
    manager.embed_tokens_module = None
    manager.lm_head_module = None

    manager.fetch_new_loras({"adapter-a"})
    assert events == [("pool", {"adapter-a"}), ("refresh", {0})]

    events.clear()
    manager.fetch_new_loras({"adapter-a"})
    assert events == [("pool", {"adapter-a"})]

    events.clear()
    manager.fetch_new_loras({"adapter-b"})
    assert events == [("pool", {"adapter-b"}), ("refresh", {1})]

    events.clear()
    manager.fetch_new_loras({None})
    assert events == [("pool", {None}), ("refresh", {2})]

    events.clear()
    manager.loras.update({"adapter-c": object(), "adapter-d": object()})
    manager.fetch_new_loras({"adapter-c", "adapter-d"}, running_loras={"adapter-a"})
    assert events[0] == ("pool", {"adapter-a", "adapter-c", "adapter-d"})
    assert events[1] == ("refresh", {1, 3})


def test_manager_unload_reload_same_uid_refreshes_changed_derived_operands():
    manager_cls = _load_manager_methods(
        {
            "create_lora_update_result",
            "fetch_new_loras",
            "_notify_lora_slots_updated",
            "unload_lora_adapter",
        },
        {
            "Dict": dict,
            "LoRAAdapter": object,
            "LoRARef": object,
            "LoRAUpdateOutput": SimpleNamespace,
            "Optional": Optional,
        },
    )

    class _Pool:
        def __init__(self):
            self.uid_to_buffer_id = {"same-uid": 0}
            self.slot_value = torch.tensor([1.0])

        def remove_lora(self, uid):
            slot = self.uid_to_buffer_id.pop(uid, None)
            if slot is not None:
                self.slot_value.zero_()
            return slot

        def prepare_lora_batch(self, *, cur_uids, lora_adapters, **kwargs):
            for uid in cur_uids:
                if uid not in self.uid_to_buffer_id:
                    self.uid_to_buffer_id[uid] = 0
                    self.slot_value.fill_(lora_adapters[uid].value)

    pool = _Pool()
    derived = torch.tensor([-1.0])
    sink = _RefreshableSharedSink(lambda slots: derived.copy_(pool.slot_value))
    ref = SimpleNamespace(
        lora_id="same-uid", lora_name="same", lora_path="old", pinned=False
    )
    manager = manager_cls()
    manager.max_loras_per_batch = 1
    manager.memory_pool = pool
    manager.configs = {"same-uid": object()}
    manager.loras = {"same-uid": SimpleNamespace(value=1.0)}
    manager.lora_refs = {"same-uid": ref}
    manager.num_pinned_loras = 0
    manager.lora_modules = [{"sink": sink}]
    manager.embed_tokens_module = None
    manager.lm_head_module = None

    result = manager.unload_lora_adapter(ref)
    assert result.success
    torch.testing.assert_close(derived, torch.zeros_like(derived))

    manager.configs["same-uid"] = object()
    manager.loras["same-uid"] = SimpleNamespace(value=9.0)
    manager.lora_refs["same-uid"] = SimpleNamespace(
        lora_id="same-uid", lora_name="same", lora_path="new", pinned=False
    )
    manager.fetch_new_loras({"same-uid"})

    torch.testing.assert_close(pool.slot_value, torch.tensor([9.0]))
    torch.testing.assert_close(derived, torch.tensor([9.0]))


def _make_unconfigured_sink(monkeypatch, *, linearized=True):
    layer = _load_batch_dense_lora_class(monkeypatch)()
    layer._linearized_bf16_enabled = linearized
    return layer


@pytest.mark.parametrize(
    ("max_loras", "backend_name", "linearized", "expected_error"),
    [
        (1, "torch-test", True, None),
        (4, "triton", True, None),
        (5, "triton", True, None),
        (16, "triton", True, None),
        (8, "csgmv", True, "requires the Triton backend"),
        (1, "torch-test", False, "does not use linearized BF16"),
    ],
)
def test_dense_sink_lora_initialization_contract(
    monkeypatch, max_loras, backend_name, linearized, expected_error
):
    layer = _make_unconfigured_sink(monkeypatch, linearized=linearized)
    backend = SimpleNamespace(
        name=backend_name,
        max_loras_per_batch=max_loras,
        is_moe_lora=False,
    )

    if expected_error is None:
        layer.initialize_lora(backend)
        assert layer.lora_backend is backend
        assert layer.is_shared_fused_moe is True
        assert backend.is_moe_lora is True
    else:
        with pytest.raises(ValueError, match=expected_error) as exc_info:
            layer.initialize_lora(backend)
        assert "InklingBatchDenseMLPWithLoRA is ineligible" in str(exc_info.value)


@pytest.mark.parametrize(
    ("case", "expected_error"),
    [
        ("valid", None),
        ("ndim", "four 4D MoE buffers"),
        ("gate_outer", "same expert layout"),
        ("down_outer", "same expert layout"),
        ("per_expert", None),
        ("expert_count", "expert count does not match"),
        ("rank128", None),
        ("rank_mismatch", "rank dimensions do not match"),
    ],
)
def test_dense_sink_requires_canonical_4d_moe_buffers(
    monkeypatch, case, expected_error
):
    layer = _make_unconfigured_sink(monkeypatch)
    layer.initialize_lora(
        SimpleNamespace(name="torch-test", max_loras_per_batch=1, is_moe_lora=False)
    )
    weights = list(_make_pool(slots=1, max_rank=2, active_rank=2, scale=1.0))
    if case == "ndim":
        weights[0] = weights[0][0]
    elif case == "gate_outer":
        weights[0] = weights[0].expand(-1, 2, -1, -1).clone()
    elif case == "down_outer":
        weights[3] = weights[3].expand(-1, 2, -1, -1).clone()
    elif case == "per_expert":
        weights[0] = weights[0].expand(-1, 2, -1, -1).clone()
        weights[3] = weights[3].expand(-1, 2, -1, -1).clone()
    elif case == "expert_count":
        weights[1] = torch.zeros(1, 3, 4, 2)
    elif case == "rank128":
        weights = [
            torch.zeros(1, 1, 256, 3),
            torch.zeros(1, 2, 4, 128),
            torch.zeros(1, 2, 128, 2),
            torch.zeros(1, 1, 3, 128),
        ]
    elif case == "rank_mismatch":
        weights[0] = torch.zeros(1, 1, 3, 3)

    if expected_error is None:
        layer.set_lora_info(*weights)
        if case == "per_expert":
            assert layer.experts_shared_outer_loras is False
            assert layer._w1_delta is None
            assert layer._a_cat is None
        elif case == "rank128":
            assert layer.experts_shared_outer_loras is True
            assert layer._w1_delta.shape == (1, 8, 256)
            assert layer._a_cat.shape == (1, 128, 4)
        else:
            assert layer._w1_delta.shape == (1, 8, 4)
            assert layer._a_cat.shape == (1, 2, 4)
    else:
        with pytest.raises(ValueError, match=expected_error):
            layer.set_lora_info(*weights)


def test_outer_factor_detection_bool_and_mixed_rejected():
    manager_cls = _load_manager_methods(
        {"_detect_shared_outer_loras"},
        {
            "Optional": __import__("typing").Optional,
            "re": __import__("re"),
        },
    )
    routed_shared = "model.layers.0.mlp.experts.gate_up_proj.lora_A.weight"
    routed_expert = "model.layers.0.mlp.experts.0.gate_up_proj.lora_A.weight"

    shared_only = manager_cls()
    shared_only.loras = {
        "shared": SimpleNamespace(
            layers=[SimpleNamespace(weights={routed_shared: torch.empty(1, 8, 4)})]
        ),
    }
    assert shared_only._detect_shared_outer_loras() is True

    per_expert_only = manager_cls()
    per_expert_only.loras = {
        "per-expert": SimpleNamespace(
            # numbered 2D expert weights must be visible as per-expert layout
            layers=[SimpleNamespace(weights={routed_expert: torch.empty(4, 4)})]
        ),
    }
    assert per_expert_only._detect_shared_outer_loras() is False

    mixed = manager_cls()
    mixed.loras = {
        "shared": SimpleNamespace(
            layers=[SimpleNamespace(weights={routed_shared: torch.empty(1, 8, 4)})]
        ),
        "per-expert": SimpleNamespace(
            layers=[SimpleNamespace(weights={routed_expert: torch.empty(4, 4)})]
        ),
    }
    with pytest.raises(RuntimeError, match="Mixed shared-outer LoRA formats"):
        mixed._detect_shared_outer_loras()
