"""CPU tests for architecture-independent whole-forward FX planning."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from packaging.version import Version

from sglang.srt.hardware_backend.mlx.fx_lowering import (
    MlxFxCaptureBackend,
    MlxFxLoweringRegistry,
    UnsupportedMlxFxGraphError,
    build_mlx_fx_plan,
    export_mlx_plan,
    make_mlx_fx_executor,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=1, suite="stage-a-unit-test-mlx")

_HAS_WHOLE_GRAPH_MLX_RUNTIME = torch.backends.mps.is_available() and Version(
    torch.__version__
) >= Version("2.13")


@torch.library.custom_op("sglang_mlx_fx_test::kv_commit", mutates_args={"cache"})
def _kv_commit(cache: torch.Tensor, value: torch.Tensor) -> None:
    cache.copy_(value)


@_kv_commit.register_fake
def _kv_commit_fake(cache: torch.Tensor, value: torch.Tensor) -> None:
    return None


class _TinyDecoder(torch.nn.Module):
    """Qwen/Llama-shaped graph without an architecture-specific contract."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 8)
        self.norm = torch.nn.RMSNorm(8)
        self.qkv = torch.nn.Linear(8, 24, bias=False)
        self.out = torch.nn.Linear(8, 8, bias=False)
        self.gate_up = torch.nn.Linear(8, 16, bias=False)
        self.down = torch.nn.Linear(8, 8, bias=False)

    def forward(self, input_ids: torch.Tensor, cache: torch.Tensor):
        hidden = self.norm(self.embedding(input_ids))
        q, k, v = self.qkv(hidden).chunk(3, dim=-1)
        attention = torch.nn.functional.scaled_dot_product_attention(
            q[None, None], k[None, None], v[None, None]
        )[0, 0]
        hidden = self.out(attention)
        gate, up = self.gate_up(hidden).chunk(2, dim=-1)
        hidden = self.down(torch.nn.functional.silu(gate) * up)
        _kv_commit(cache, hidden)
        return hidden


class _TinyMlp(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.RMSNorm(8)
        self.gate_up = torch.nn.Linear(8, 16, bias=False)
        self.down = torch.nn.Linear(8, 8, bias=False)

    def forward(self, value):
        gate, up = self.gate_up(self.norm(value)).chunk(2, dim=-1)
        return self.down(torch.nn.functional.silu(gate) * up)


def _decoder_registry() -> MlxFxLoweringRegistry:
    registry = MlxFxLoweringRegistry.standard_decoder()
    registry.register_function(
        torch.ops.sglang_mlx_fx_test.kv_commit.default, "kv_commit"
    )
    return registry


def test_dynamo_captures_dense_decoder_and_kv_commit_as_one_supported_graph():
    torch._dynamo.reset()
    model = _TinyDecoder().eval()
    backend = MlxFxCaptureBackend(_decoder_registry())
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    input_ids = torch.tensor([1, 2])
    expected_cache = torch.zeros(2, 8)
    actual_cache = torch.zeros(2, 8)

    expected = model(input_ids, expected_cache)
    actual = compiled(input_ids, actual_cache)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_cache, expected_cache)
    assert len(backend.plans) == 1
    assert backend.plans[0].fully_supported
    lowerings = {node.lowering for node in backend.plans[0].nodes}
    assert {
        "embedding",
        "rms_norm",
        "linear",
        "sdpa",
        "silu",
        "multiply",
        "kv_commit",
    } <= lowerings


def test_executor_factory_receives_the_complete_graph_once():
    torch._dynamo.reset()
    executor_plans = []

    def executor_factory(plan, _example_inputs):
        executor_plans.append(plan)
        return plan.graph_module.forward

    backend = MlxFxCaptureBackend(
        _decoder_registry(), executor_factory=executor_factory
    )
    compiled = torch.compile(_TinyDecoder().eval(), backend=backend, fullgraph=True)
    compiled(torch.tensor([1, 2]), torch.zeros(2, 8))

    assert executor_plans == backend.plans
    assert len(executor_plans) == 1


@pytest.mark.skipif(
    not _HAS_WHOLE_GRAPH_MLX_RUNTIME,
    reason="requires Torch 2.13 and MLX on MPS",
)
def test_generic_mlx_executor_matches_torch_in_one_region():
    torch._dynamo.reset()
    model = _TinyMlp().to(device="mps", dtype=torch.bfloat16).eval()
    backend = MlxFxCaptureBackend(
        executor_factory=make_mlx_fx_executor,
    )
    value = torch.randn(4, 8, device="mps", dtype=torch.bfloat16)
    expected = model(value)
    actual = torch.compile(model, backend=backend, fullgraph=True)(value)

    torch.mps.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.008, rtol=0.03)
    assert len(backend.plans) == 1
    assert backend.plans[0].fully_supported


@pytest.mark.skipif(
    not _HAS_WHOLE_GRAPH_MLX_RUNTIME,
    reason="requires Torch 2.13 and MLX on MPS",
)
def test_generic_mlx_executor_reuses_one_torch_graph_across_token_counts():
    torch._dynamo.reset()
    model = _TinyMlp().to(device="mps", dtype=torch.bfloat16).eval()
    backend = MlxFxCaptureBackend(
        executor_factory=make_mlx_fx_executor,
    )
    first = torch.randn(4, 8, device="mps", dtype=torch.bfloat16)
    torch._dynamo.mark_dynamic(first, 0, min=2, max=16)
    compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)

    for value in (
        first,
        torch.randn(7, 8, device="mps", dtype=torch.bfloat16),
        torch.randn(12, 8, device="mps", dtype=torch.bfloat16),
    ):
        expected = model(value)
        actual = compiled(value)
        torch.mps.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.008, rtol=0.03)

    assert len(backend.plans) == 1


@pytest.mark.skipif(
    not _HAS_WHOLE_GRAPH_MLX_RUNTIME,
    reason="requires Torch 2.13 and MLX on MPS",
)
def test_generic_mlx_executor_observes_weight_mutation_and_replacement():
    torch._dynamo.reset()
    model = _TinyMlp().to(device="mps", dtype=torch.bfloat16).eval()
    backend = MlxFxCaptureBackend(executor_factory=make_mlx_fx_executor)
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    value = torch.randn(4, 8, device="mps", dtype=torch.bfloat16)

    first = compiled(value)
    with torch.no_grad():
        model.gate_up.weight.add_(0.25)
    mutated_expected = model(value)
    mutated_actual = compiled(value)

    replacement = torch.nn.Parameter(
        torch.randn_like(model.gate_up.weight) * 0.05,
        requires_grad=False,
    )
    model.gate_up.weight = replacement
    replaced_expected = model(value)
    replaced_actual = compiled(value)

    torch.mps.synchronize()
    assert not torch.allclose(first.cpu(), mutated_actual.cpu())
    torch.testing.assert_close(
        mutated_actual.cpu(), mutated_expected.cpu(), atol=0.008, rtol=0.03
    )
    torch.testing.assert_close(
        replaced_actual.cpu(), replaced_expected.cpu(), atol=0.008, rtol=0.03
    )


@pytest.mark.skipif(
    not _HAS_WHOLE_GRAPH_MLX_RUNTIME,
    reason="requires Torch 2.13 and MLX on MPS",
)
@pytest.mark.parametrize("architecture", ["qwen3", "llama"])
def test_same_mlx_executor_runs_real_sglang_mlp_architectures(architecture):
    from sglang.srt.models.llama import LlamaMLP
    from sglang.srt.models.qwen3 import Qwen3MLP
    from sglang.srt.runtime_context import _CONTEXT, get_parallel

    old_server_args = _CONTEXT._server_args
    old_config_bags = _CONTEXT._config_bags
    old_parallel_config = get_parallel()._config
    _CONTEXT.set_server_args(
        SimpleNamespace(rl_on_policy_target=None, enable_torch_compile=False)
    )
    deterministic_exec = SimpleNamespace(
        deterministic=SimpleNamespace(rl_on_policy_target=None)
    )

    def get_deterministic_exec():
        return deterministic_exec

    try:
        with (
            get_parallel().override(tp_size=1, tp_rank=0),
            mock.patch(
                "sglang.srt.layers.activation.get_exec",
                new=get_deterministic_exec,
            ),
            mock.patch(
                "sglang.srt.models.qwen2.get_exec",
                new=get_deterministic_exec,
            ),
        ):
            if architecture == "qwen3":
                model = Qwen3MLP(8, 16, "silu")
            else:
                model = LlamaMLP(8, 16, "silu", tp_rank=0, tp_size=1)
            model = model.to(device="mps", dtype=torch.bfloat16).eval()
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.normal_(mean=0.0, std=0.02)
            gate_up_projection = model.gate_up_proj
            down_projection = model.down_proj

            def single_rank_gate_up_projection(input_):
                return (
                    torch.nn.functional.linear(
                        input_, gate_up_projection.weight, gate_up_projection.bias
                    ),
                    None,
                )

            def single_rank_down_projection(
                input_, skip_all_reduce=False, forward_batch=None
            ):
                return (
                    torch.nn.functional.linear(
                        input_, down_projection.weight, down_projection.bias
                    ),
                    None,
                )

            gate_up_projection.forward = single_rank_gate_up_projection
            down_projection.forward = single_rank_down_projection
            model.act_fn.enter_torch_compile(num_tokens=4)

            torch._dynamo.reset()
            backend = MlxFxCaptureBackend(
                executor_factory=make_mlx_fx_executor,
            )
            value = torch.randn(4, 8, device="mps", dtype=torch.bfloat16)
            expected = model(value)
            actual = torch.compile(model, backend=backend, fullgraph=True)(value)
    finally:
        _CONTEXT._server_args = old_server_args
        _CONTEXT._config_bags = old_config_bags
        get_parallel()._config = old_parallel_config

    torch.mps.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0.008, rtol=0.03)
    assert len(backend.plans) == 1
    assert backend.plans[0].fully_supported
    assert {"linear", "silu", "multiply"} <= {
        node.lowering for node in backend.plans[0].nodes
    }


def test_export_functionalizes_kv_mutation_into_explicit_graph_output():
    registry = MlxFxLoweringRegistry.standard_export_decoder()
    registry.register_function(
        torch.ops.sglang_mlx_fx_test.kv_commit.default, "kv_commit"
    )
    model = _TinyDecoder().eval()
    plan = export_mlx_plan(
        model,
        (torch.tensor([1, 2]), torch.zeros(2, 8)),
        registry,
    )

    assert plan.mutated_user_inputs == ("cache",)
    lowerings = {node.lowering for node in plan.graph_plan.nodes}
    assert "functionalized_kv_commit" in lowerings
    # Core ATen decomposition is intentionally not fully registered yet. This
    # test proves mutation/state representation, not MLX operation coverage.
    assert not plan.graph_plan.fully_supported
    output_kinds = {
        str(spec.kind).rsplit(".", 1)[-1]
        for spec in plan.functional_program.graph_signature.output_specs
    }
    assert output_kinds == {"USER_INPUT_MUTATION", "USER_OUTPUT"}


def test_real_qwen3_mlp_exports_without_an_architecture_registry():
    from sglang.srt.models.qwen3 import Qwen3MLP
    from sglang.srt.runtime_context import _CONTEXT, get_parallel

    old_server_args = _CONTEXT._server_args
    old_config_bags = _CONTEXT._config_bags
    old_parallel_config = get_parallel()._config
    _CONTEXT.set_server_args(
        SimpleNamespace(rl_on_policy_target=None, enable_torch_compile=False)
    )
    deterministic_exec = SimpleNamespace(
        deterministic=SimpleNamespace(rl_on_policy_target=None)
    )

    def get_deterministic_exec():
        return deterministic_exec

    try:
        with (
            get_parallel().override(tp_size=1, tp_rank=0),
            mock.patch(
                "sglang.srt.layers.activation.get_exec",
                new=get_deterministic_exec,
            ),
            mock.patch(
                "sglang.srt.models.qwen2.get_exec",
                new=get_deterministic_exec,
            ),
        ):
            model = Qwen3MLP(8, 16, "silu").eval()
            gate_up_projection = model.gate_up_proj
            down_projection = model.down_proj

            # Export TP=1 compute without initializing process groups or selecting
            # hardware-specific GEMM wrappers. Those are separate lowerings.
            def single_rank_gate_up_projection(input_):
                return (
                    torch.nn.functional.linear(
                        input_, gate_up_projection.weight, gate_up_projection.bias
                    ),
                    None,
                )

            def single_rank_down_projection(
                input_, skip_all_reduce=False, forward_batch=None
            ):
                return (
                    torch.nn.functional.linear(
                        input_, down_projection.weight, down_projection.bias
                    ),
                    None,
                )

            gate_up_projection.forward = single_rank_gate_up_projection
            down_projection.forward = single_rank_down_projection
            model.act_fn.enter_torch_compile(num_tokens=2)
            exported = torch.export.export(
                model,
                (torch.randn(2, 8),),
                strict=True,
            )
    finally:
        _CONTEXT._server_args = old_server_args
        _CONTEXT._config_bags = old_config_bags
        get_parallel()._config = old_parallel_config

    plan = build_mlx_fx_plan(
        exported.graph_module,
        MlxFxLoweringRegistry.standard_export_decoder(),
    )

    assert plan.fully_supported
    lowerings = [node.lowering for node in plan.nodes]
    assert lowerings.count("linear") == 2
    assert {"silu", "slice", "multiply"} <= set(lowerings)


def test_unsupported_operation_rejects_the_complete_region():
    class UnsupportedModel(torch.nn.Module):
        def forward(self, value):
            return torch.sin(value)

    graph = torch.fx.symbolic_trace(UnsupportedModel())
    plan = build_mlx_fx_plan(graph, MlxFxLoweringRegistry.standard_decoder())

    assert not plan.fully_supported
    assert len(plan.unsupported) == 1
    try:
        plan.require_fully_supported()
    except UnsupportedMlxFxGraphError as exc:
        assert "sin" in str(exc)
    else:
        raise AssertionError("unsupported FX graph was accepted")


def test_unsupported_compiled_graph_falls_back_as_one_torch_region():
    class UnsupportedModel(torch.nn.Module):
        def forward(self, value):
            return torch.sin(value) + 1

    torch._dynamo.reset()
    backend = MlxFxCaptureBackend(fallback_to_torch=True)
    model = UnsupportedModel()
    value = torch.randn(4)
    actual = torch.compile(model, backend=backend, fullgraph=True)(value)

    torch.testing.assert_close(actual, model(value))
    assert len(backend.plans) == 1
    assert not backend.plans[0].fully_supported


@pytest.mark.skipif(
    not _HAS_WHOLE_GRAPH_MLX_RUNTIME,
    reason="requires Torch 2.13 and MLX on MPS",
)
def test_review_flagged_ops_match_eager_through_the_export_path():
    """mm routing, add/sub alpha, slice bounds, and plain-attribute get_attr.

    Each construct was flagged in review: a bare ``mm`` must compute
    ``a @ b`` rather than ride the transposing linear lowering; ``add`` and
    ``sub`` must honour ``alpha``; negative and sentinel slice bounds must
    match Torch's normalization; and a constant tensor hung directly on the
    module (neither parameter nor registered buffer) must resolve as a
    ``get_attr``.
    """

    class ReviewOps(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shift = torch.full((4, 4), 0.25, device="mps", dtype=torch.bfloat16)

        def forward(self, x, w):
            product = torch.mm(x, w)
            subbed = torch.sub(product, self.shift, alpha=2.0)
            added = torch.add(product, self.shift, alpha=0.5)
            return added, subbed, x[:, :-1], x[:, 1:], x[:, -3:]

    model = ReviewOps().eval()
    x = torch.randn(4, 8, device="mps", dtype=torch.bfloat16)
    w = torch.randn(8, 4, device="mps", dtype=torch.bfloat16)
    expected = model(x, w)

    graph_module = torch.export.export(model, (x, w), strict=False).module()
    # The export guard submodule is dead weight the consumer erases before
    # planning; do the same here.
    for node in tuple(graph_module.graph.nodes):
        if node.op == "call_module" and str(node.target) == "_guards_fn":
            assert not node.users
            graph_module.graph.erase_node(node)
    graph_module.recompile()
    plan = build_mlx_fx_plan(
        graph_module, MlxFxLoweringRegistry.standard_export_decoder()
    )
    plan.require_fully_supported()
    executor = make_mlx_fx_executor(plan, [x, w])
    actual = executor(x, w)

    torch.mps.synchronize()
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        torch.testing.assert_close(
            actual_item.cpu(), expected_item.cpu(), atol=0.008, rtol=0.03
        )
