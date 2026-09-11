"""CPU unit tests for the runner seam (design section 6.8).

``BaseCudaGraphRunner.materialize_shape`` is the one place a runner hands a
prepared shape to its materializer. These tests check the ``ShapePlan`` it
builds field by field with a recording materializer, the
``materializer_or_capture_only`` fallback, the forward-free
``planned_shape_keys`` enumerations against the capture loops, and read the
decode and prefill runner sources (AST, no CUDA) to pin that
``capture_one_shape`` goes through ``materialize_shape``, that
``self.materializer`` is resolved from the constructor's plan right after
``self.backend`` and that ``capture`` drives ``plan`` / ``session``. The
speculative runners are deliberately not checked: they stay capture-only in
v1 (design section 9.3).
"""

import ast
import inspect
import pathlib
import sys
from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.graph_serialization.materializer import (
    CaptureOnlyMaterializer,
    ShapePlan,
)
from sglang.srt.model_executor.runner import (
    decode_cuda_graph_runner,
    prefill_cuda_graph_runner,
)
from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
    BaseCudaGraphRunner,
)
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
    PrefillCudaGraphRunner,
    _chunked_prefix_variant,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Recording:
    """A materializer that only records the plan it was handed."""

    def __init__(self) -> None:
        self.plans: list[ShapePlan] = []

    def materialize(self, plan: ShapePlan) -> None:
        self.plans.append(plan)


class _Runner(BaseCudaGraphRunner):
    """Minimal concrete subclass; built with ``__new__`` so no ModelRunner is
    needed."""

    def can_run_graph(self, forward_batch):
        raise NotImplementedError

    def load_batch(self, forward_batch, **kwargs):
        raise NotImplementedError

    def execute(self, *args, **kwargs):
        raise NotImplementedError

    def capture_prepare(self, size, *args, **kwargs):
        raise NotImplementedError

    def capture(self):
        raise NotImplementedError

    def capture_one_shape(self, size, *args, **kwargs):
        raise NotImplementedError


def _runner() -> tuple[_Runner, _Recording]:
    runner = _Runner.__new__(_Runner)
    runner.materializer = _Recording()
    return runner, runner.materializer


def test_materialize_shape_packs_every_field_and_defaults_event_roles_to_empty():
    runner, recording = _runner()
    shape_key = ShapeKey(size=8, variant_label="lora")
    forward_fn = object()
    capture_inputs = object()
    hook = object()

    result = runner.materialize_shape(
        shape_key,
        forward_fn,
        capture_inputs=capture_inputs,
        post_warmup_hook=hook,
        event_roles=None,
    )

    assert result is None
    assert len(recording.plans) == 1
    plan = recording.plans[0]
    assert isinstance(plan, ShapePlan)
    assert plan.shape_key is shape_key
    assert plan.forward_fn is forward_fn
    assert plan.capture_inputs is capture_inputs
    assert plan.post_warmup_hook is hook
    assert plan.event_roles == {}
    assert isinstance(plan.event_roles, dict)


def test_materialize_shape_keyword_defaults():
    runner, recording = _runner()
    shape_key = ShapeKey(size=1)
    runner.materialize_shape(shape_key, print)
    plan = recording.plans[0]
    assert plan.shape_key is shape_key
    assert plan.forward_fn is print
    assert plan.capture_inputs is None
    assert plan.post_warmup_hook is None
    assert plan.event_roles == {}


def test_materialize_shape_passes_event_roles_through():
    runner, recording = _runner()
    event = SimpleNamespace(cuda_event=123)
    runner.materialize_shape(
        ShapeKey(size=2), print, event_roles={"metadata_prep_done": event}
    )
    assert recording.plans[0].event_roles == {"metadata_prep_done": event}
    assert recording.plans[0].event_roles["metadata_prep_done"] is event


def test_base_runner_declares_materializer_next_to_backend():
    annotations = BaseCudaGraphRunner.__annotations__
    assert "backend" in annotations
    assert "materializer" in annotations
    assert "GraphMaterializer" in str(annotations["materializer"])


def test_materializer_or_capture_only_installs_a_default_once():
    # A runner that never resolved a materializer (the speculative runners
    # reuse DecodeCudaGraphRunner.capture without its __init__) captures.
    runner = _Runner.__new__(_Runner)
    runner.backend = object()
    materializer = runner.materializer_or_capture_only()
    assert type(materializer) is CaptureOnlyMaterializer
    assert materializer.backend is runner.backend
    assert runner.materializer is materializer
    assert runner.materializer_or_capture_only() is materializer


def test_materializer_or_capture_only_returns_the_resolved_one():
    runner, recording = _runner()
    assert runner.materializer_or_capture_only() is recording


# --- planned_shape_keys mirrors the capture loops -----------------------------


def _decode_runner(**attrs) -> DecodeCudaGraphRunner:
    runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
    defaults = dict(
        capture_bs=[1, 2, 4],
        captured_req_width=1,
        ragged_verify_mode=False,
        enable_pdmux=False,
        record_nolora_graph=False,
        dsa_dual_graph=False,
    )
    defaults.update(attrs)
    for name, value in defaults.items():
        setattr(runner, name, value)
    return runner


def test_decode_planned_shape_keys_are_buckets_largest_first():
    assert _decode_runner().planned_shape_keys() == [
        ShapeKey(size=4),
        ShapeKey(size=2),
        ShapeKey(size=1),
    ]


def test_decode_planned_shape_keys_cover_lora_and_dsa_variants_in_loop_order():
    runner = _decode_runner(
        capture_bs=[2], record_nolora_graph=True, dsa_dual_graph=True
    )
    assert runner.planned_shape_keys() == [
        ShapeKey(size=2, variant_label="lora", dsa_variant="dense"),
        ShapeKey(size=2, variant_label="lora", dsa_variant="sparse"),
        ShapeKey(size=2, variant_label="nolora", dsa_variant="dense"),
        ShapeKey(size=2, variant_label="nolora", dsa_variant="sparse"),
    ]


def test_decode_planned_shape_keys_use_token_sizes_in_ragged_verify_mode():
    runner = _decode_runner(
        capture_bs=[1, 2], captured_req_width=4, ragged_verify_mode=True
    )
    assert runner.planned_shape_keys() == [ShapeKey(size=8), ShapeKey(size=4)]


def test_decode_planned_shape_keys_enumerate_pdmux_streams():
    runner = _decode_runner(
        capture_bs=[1], enable_pdmux=True, stream_groups=["s0", "s1"]
    )
    assert runner.planned_shape_keys() == [
        ShapeKey(size=1, stream_idx=0),
        ShapeKey(size=1, stream_idx=1),
    ]


def _prefill_runner(**attrs) -> PrefillCudaGraphRunner:
    runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
    defaults = dict(
        capture_num_tokens=[128, 512],
        _capture_chunked_prefix=False,
        _prefix_capture_variants=(),
    )
    defaults.update(attrs)
    for name, value in defaults.items():
        setattr(runner, name, value)
    return runner


def test_prefill_planned_shape_keys_are_buckets_largest_first():
    assert _prefill_runner().planned_shape_keys() == [
        ShapeKey(size=512),
        ShapeKey(size=128),
    ]


def test_prefill_planned_shape_keys_follow_each_bucket_with_its_prefix_variants():
    runner = _prefill_runner(
        _capture_chunked_prefix=True, _prefix_capture_variants=(1, 2)
    )
    assert runner.planned_shape_keys() == [
        ShapeKey(size=512),
        ShapeKey(size=512, variant_label=_chunked_prefix_variant(1)),
        ShapeKey(size=512, variant_label=_chunked_prefix_variant(2)),
        ShapeKey(size=128),
        ShapeKey(size=128, variant_label=_chunked_prefix_variant(1)),
        ShapeKey(size=128, variant_label=_chunked_prefix_variant(2)),
    ]


@pytest.mark.parametrize("cls", [DecodeCudaGraphRunner, PrefillCudaGraphRunner])
def test_runner_constructors_take_the_plan_as_an_optional_keyword(cls):
    parameter = inspect.signature(cls.__init__).parameters["graph_serialization_plan"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


# --- source-level checks on the two production runners ------------------------


def _class_def(module, name: str) -> ast.ClassDef:
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {module.__file__}")


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{cls.name}.{name} not found")


def _attribute_calls(fn: ast.AST, attr: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]


@pytest.mark.parametrize(
    "module, class_name, backend_resolver",
    [
        (decode_cuda_graph_runner, "DecodeCudaGraphRunner", "resolve_decode_backend"),
        (
            prefill_cuda_graph_runner,
            "PrefillCudaGraphRunner",
            "resolve_prefill_backend",
        ),
    ],
)
def test_capture_one_shape_goes_through_materialize_shape(
    module, class_name, backend_resolver
):
    cls = _class_def(module, class_name)
    fn = _method(cls, "capture_one_shape")

    direct = [
        call
        for call in _attribute_calls(fn, "capture_one")
        if ast.unparse(call.func) == "self.backend.capture_one"
    ]
    assert direct == [], "capture_one_shape must not call backend.capture_one"

    seam = _attribute_calls(fn, "materialize_shape")
    assert len(seam) == 1
    assert ast.unparse(seam[0].func) == "self.materialize_shape"
    # Both runners forward the two backend keywords; the decode runner also
    # names the external event role (design section 9.1).
    keywords = {kw.arg for kw in seam[0].keywords}
    assert {"capture_inputs", "post_warmup_hook"} <= keywords
    if class_name == "DecodeCudaGraphRunner":
        assert "event_roles" in keywords
        assert "metadata_prep_done" in ast.unparse(seam[0])


@pytest.mark.parametrize(
    "module, class_name, backend_resolver",
    [
        (decode_cuda_graph_runner, "DecodeCudaGraphRunner", "resolve_decode_backend"),
        (
            prefill_cuda_graph_runner,
            "PrefillCudaGraphRunner",
            "resolve_prefill_backend",
        ),
    ],
)
def test_materializer_is_resolved_immediately_after_the_backend(
    module, class_name, backend_resolver
):
    init = _method(_class_def(module, class_name), "__init__")
    body = init.body
    backend_index = next(
        i
        for i, stmt in enumerate(body)
        if f"self.backend = {backend_resolver}(" in ast.unparse(stmt)
    )
    # The backend keeps raw graphs exactly when the plan is enabled, and the
    # materializer is resolved from that same plan, never from the config.
    backend_stmt = ast.unparse(body[backend_index])
    assert "keep_graph=keeps_raw_graphs(graph_serialization_plan)" in backend_stmt
    following = ast.unparse(body[backend_index + 1])
    assert following == (
        "self.materializer = resolve_materializer(self, graph_serialization_plan)"
    )


@pytest.mark.parametrize(
    "module, class_name, session_count",
    [
        (decode_cuda_graph_runner, "DecodeCudaGraphRunner", 2),
        (prefill_cuda_graph_runner, "PrefillCudaGraphRunner", 1),
    ],
)
def test_capture_drives_plan_and_session_through_the_materializer(
    module, class_name, session_count
):
    capture = _method(_class_def(module, class_name), "capture")

    direct = [
        call
        for call in _attribute_calls(capture, "capture_session")
        if ast.unparse(call.func) == "self.backend.capture_session"
    ]
    assert direct == [], "capture must not bypass the materializer's session"

    fallback = _attribute_calls(capture, "materializer_or_capture_only")
    assert [ast.unparse(call) for call in fallback] == [
        "self.materializer_or_capture_only()"
    ]
    plans = _attribute_calls(capture, "plan")
    assert [ast.unparse(call) for call in plans] == ["materializer.plan(self)"]
    sessions = _attribute_calls(capture, "session")
    assert [ast.unparse(call) for call in sessions] == (
        ["materializer.session(self.stream)"] * session_count
    )
    # plan() runs before the shape loop, i.e. before the first session.
    assert plans[0].lineno < min(call.lineno for call in sessions)


def test_speculative_runners_are_untouched():
    # Spec runners keep calling backend.capture_one directly (design section
    # 9.3); their sources must not have grown a materialize_shape call.
    spec_dir = (
        pathlib.Path(decode_cuda_graph_runner.__file__).parents[2] / "speculative"
    )
    sources = list(spec_dir.glob("*cuda_graph_runner.py"))
    assert sources, f"no spec runners found under {spec_dir}"
    for path in sources:
        assert "materialize_shape(" not in path.read_text(encoding="utf-8"), path


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
