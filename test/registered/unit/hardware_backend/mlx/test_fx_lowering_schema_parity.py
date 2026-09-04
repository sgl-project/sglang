"""Schema-driven parity tests for every registered ATen-to-MLX lowering.

For each ``@_lowering`` registration, read the live ATen ``_schema`` of its
targets and enumerate argument variants mechanically: the base case, plus one
case per defaulted argument with the default overridden. Each case runs the
real op eagerly on CPU torch (the oracle) and the registered lowering on MLX,
and the two must agree.

The acceptance rule per case is *match or raise*: a lowering may reject a
case with ``UnsupportedMlxFxGraphError`` (serving falls back to eager), but
silent numeric disagreement fails. The base case must match, never raise, so
an op cannot pass by rejecting everything.

Because the schemas are read from the installed torch at collection time, a
version bump that adds or changes an argument on any op we lower shows up
here as new generated cases, without anyone editing this file. A new
``@_lowering`` registration without a recipe below fails the coverage test.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from sglang.srt.hardware_backend.mlx.fx_lowering import (
    _LOWERINGS,
    UnsupportedMlxFxGraphError,
    _lower_mlx_node,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_mlx_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")
register_mlx_ci(est_time=2, suite="stage-a-unit-test-mlx")

_HAS_MLX_RUNTIME = (
    importlib.util.find_spec("mlx") is not None and torch.backends.mps.is_available()
)

# Lowerings with no comparable eager output: assert_metadata returns None,
# getitem's only target carries no ATen schema.
_EXEMPT = {"assert_metadata", "getitem"}


def _t(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, dtype=torch.float32)


def _positive(*shape: int) -> torch.Tensor:
    return torch.rand(*shape, dtype=torch.float32) + 0.5


def _indices(count: int, high: int) -> torch.Tensor:
    return torch.randint(0, high, (count,))


# One recipe per lowering: the base arguments (schema order, tensors plus
# required scalars), optional value pools for defaulted arguments the generic
# type pools cannot guess, arguments excluded from overriding (e.g. ones that
# make eager nondeterministic), and extra hand-picked cases. ``compare`` may
# be "values" (default) or "shape" for ops with unspecified contents.
_RECIPES: dict[str, dict] = {
    "alias": dict(args=lambda: (_t(4, 6),)),
    "add": dict(
        args=lambda: (_t(4, 4), _t(4, 4)),
        pools={"alpha": [2, 0.5]},
        extra=[lambda: ((_t(4, 4), _t(1, 4)), {})],  # broadcasting
    ),
    "subtract": dict(
        args=lambda: (_t(4, 4), _t(4, 4)),
        pools={"alpha": [2, 0.5]},
    ),
    "multiply": dict(
        args=lambda: (_t(4, 4), _t(4, 4)),
        extra=[lambda: ((_t(4, 4), _t(1, 4)), {})],
    ),
    "concatenate": dict(args=lambda: ([_t(3, 4), _t(3, 4)],), pools={"dim": [1]}),
    "stack": dict(args=lambda: ([_t(3, 4), _t(3, 4)],), pools={"dim": [1]}),
    "chunk": dict(args=lambda: (_t(5, 6), 3), pools={"dim": [1]}),
    "split": dict(args=lambda: (_t(6, 4), 2), pools={"dim": [1]}),
    "split_with_sizes": dict(args=lambda: (_t(6, 4), [2, 4])),
    "embedding": dict(args=lambda: (_t(10, 4), _indices(6, 10))),
    "empty_like": dict(args=lambda: (_t(4, 6),), compare="shape"),
    "flatten": dict(
        args=lambda: (_t(2, 3, 4),), pools={"start_dim": [1], "end_dim": [1]}
    ),
    "gelu": dict(args=lambda: (_t(4, 4),), pools={"approximate": ["tanh"]}),
    "relu": dict(args=lambda: (_t(4, 4),)),
    "silu": dict(args=lambda: (_t(4, 4),)),
    "sigmoid": dict(args=lambda: (_t(4, 4),)),
    "tanh": dict(args=lambda: (_t(4, 4),)),
    "rsqrt": dict(args=lambda: (_positive(4, 4),)),
    "index_select": dict(args=lambda: (_t(5, 6), 1, _indices(3, 6))),
    "layer_norm": dict(
        args=lambda: (_t(4, 8), [8]),
        pools={"weight": [_t(8)], "bias": [_t(8)], "eps": [1e-3]},
        rejects=[lambda: ((_t(4, 8), [4, 8]), {})],  # not last-axis only
    ),
    "rms_norm": dict(
        args=lambda: (_t(4, 8), [8], _t(8)),
        pools={"eps": [1e-3]},
    ),
    "linear": dict(args=lambda: (_t(3, 8), _t(5, 8)), pools={"bias": [_t(5)]}),
    "matmul": dict(args=lambda: (_t(3, 4), _t(4, 5))),
    "mean": dict(args=lambda: (_t(3, 4, 5), [1]), exclude={"dtype"}),
    "numpy_transpose": dict(args=lambda: (_t(3, 4),)),
    "power": dict(
        args=lambda: (_positive(4, 4), 2.0),
        extra=[lambda: ((_positive(4, 4), 0.5), {})],
    ),
    "reshape": dict(args=lambda: (_t(4, 6), [6, 4])),
    "sdpa": dict(
        args=lambda: (_t(2, 3, 5, 4), _t(2, 3, 5, 4), _t(2, 3, 5, 4)),
        pools={"is_causal": [True], "scale": [0.5]},
        exclude={"dropout_p", "attn_mask", "enable_gqa"},
    ),
    "slice": dict(args=lambda: (_t(4, 8), 1, 1, -1), pools={"step": [2]}),
    "to_dtype": dict(args=lambda: (_t(4, 4), torch.float16), exclude={"non_blocking"}),
    "unsqueeze": dict(args=lambda: (_t(3, 4), 1)),
}

_TYPE_POOLS = {
    "bool": lambda default: [not default],
    "float": lambda default: [(default or 0.0) + 0.5],
    "int": lambda default: [(default or 0) + 1],
}


def _schema_targets(spec):
    return [t for t in spec.aten if getattr(t, "_schema", None) is not None]


def _override_pool(recipe: dict, argument) -> list:
    pools = recipe.get("pools", {})
    if argument.name in pools:
        return pools[argument.name]
    type_pool = _TYPE_POOLS.get(str(argument.type))
    if type_pool is None:
        return []
    default = argument.default_value if argument.has_default_value() else None
    return type_pool(default)


def _generate_cases():
    """(case_id, lowering_name, target, args_factory, kwargs, expect_raise)."""
    cases = []
    seen_specs = set()
    for name, spec in _LOWERINGS.items():
        if name in _EXEMPT or id(spec) in seen_specs:
            continue
        seen_specs.add(id(spec))
        recipe = _RECIPES.get(name)
        if recipe is None:
            continue  # the coverage test reports it
        for target in _schema_targets(spec):
            op = str(target).replace("aten.", "")
            cases.append((f"{op}-base", name, target, recipe["args"], {}, False))
            for argument in target._schema.arguments:
                if not argument.has_default_value():
                    continue
                if argument.name in recipe.get("exclude", ()):
                    continue
                for value in _override_pool(recipe, argument):
                    label = value if not isinstance(value, torch.Tensor) else "tensor"
                    cases.append(
                        (
                            f"{op}-{argument.name}={label}",
                            name,
                            target,
                            recipe["args"],
                            {argument.name: value},
                            False,
                        )
                    )
            for index, extra in enumerate(recipe.get("extra", ())):
                cases.append(
                    (f"{op}-extra{index}", name, target, *_split(extra), False)
                )
            for index, reject in enumerate(recipe.get("rejects", ())):
                cases.append(
                    (f"{op}-reject{index}", name, target, *_split(reject), True)
                )
    return cases


def _split(case_factory):
    def args_factory():
        return case_factory()[0]

    kwargs = case_factory()[1]
    return args_factory, kwargs


def _to_mlx(value):
    import mlx.core as mx

    if isinstance(value, torch.Tensor):
        return mx.array(value.numpy())
    if isinstance(value, (list, tuple)) and any(
        isinstance(item, torch.Tensor) for item in value
    ):
        return [_to_mlx(item) for item in value]
    return value


def _to_torch_outputs(value):
    import numpy as np

    if isinstance(value, (list, tuple)):
        return [_to_torch_outputs(item) for item in value]
    return torch.from_numpy(np.array(value)).to(torch.float32)


_CASES = _generate_cases()


def test_every_schema_bearing_lowering_has_a_recipe():
    # Aliases ("view" -> the reshape spec) are covered through the spec they
    # share, so coverage is judged per spec object, not per name.
    covered_specs = {
        id(spec) for name, spec in _LOWERINGS.items() if name in set(_RECIPES) | _EXEMPT
    }
    missing = sorted(
        name
        for name, spec in _LOWERINGS.items()
        if id(spec) not in covered_specs and _schema_targets(spec)
    )
    assert not missing, (
        "new lowerings need a parity recipe so their schema variants are "
        f"exercised: {missing}"
    )


@pytest.mark.skipif(not _HAS_MLX_RUNTIME, reason="requires MLX on Apple Silicon")
@pytest.mark.parametrize(
    "name,target,args_factory,kwargs,expect_raise",
    [case[1:] for case in _CASES],
    ids=[case[0] for case in _CASES],
)
def test_lowering_matches_eager_for_schema_variants(
    name, target, args_factory, kwargs, expect_raise
):
    import mlx.core as mx

    torch.manual_seed(0)
    args = args_factory()
    try:
        expected = target(*args, **kwargs)
    except Exception as error:  # eager itself rejects this combination
        pytest.skip(f"eager rejects the case: {error}")

    mlx_args = tuple(_to_mlx(value) for value in args)
    mlx_kwargs = {key: _to_mlx(value) for key, value in kwargs.items()}
    try:
        actual = _lower_mlx_node(name, mlx_args, mlx_kwargs)
    except UnsupportedMlxFxGraphError:
        if expect_raise or kwargs:
            return  # declared rejection, or a variant the lowering declines
        raise AssertionError(
            f"lowering {name!r} rejected its base case; the plain-arguments "
            "form must be supported"
        )
    assert not expect_raise, f"lowering {name!r} accepted a case it must reject"

    mx.eval(actual) if not isinstance(actual, (list, tuple)) else mx.eval(*actual)
    actual = _to_torch_outputs(actual)
    expected_items = (
        list(expected) if isinstance(expected, (list, tuple)) else [expected]
    )
    actual_items = actual if isinstance(actual, list) else [actual]
    assert len(actual_items) == len(expected_items)
    for actual_item, expected_item in zip(actual_items, expected_items):
        if _RECIPES[name].get("compare") == "shape":
            assert tuple(actual_item.shape) == tuple(expected_item.shape)
            continue
        torch.testing.assert_close(
            actual_item,
            expected_item.to(torch.float32),
            atol=5e-3,
            rtol=5e-3,
            msg=lambda default, name=name: f"lowering {name!r} diverged: {default}",
        )
