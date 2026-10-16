"""The speculative draft runners inherit the decode graph runner's capture path
without calling its ``__init__``, so anything that path reads must be a
class-level default on the base. The width ladder shipped without them and
draft-graph capture raised ``AttributeError`` with no widths configured.
"""

import ast
import inspect
import textwrap
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import sglang.srt
from sglang.srt.hardware_backend.npu.graph_runner.multi_layer_eagle_draft_extend_npu_graph_runner import (
    MultiLayerEagleDraftExtendNpuGraphRunner,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.deepseek_v4_backend import DecodeGraphWidths
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner import (
    FrozenKVMTPCudaGraphRunner,
)
from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
    MultiLayerEagleDraftExtendCudaGraphRunner,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Every runner that reaches the inherited capture path with a ``__init__`` that
# never calls ``super().__init__()`` -- directly, or inherited from one that
# does not. TestExposedRunnerRatchet keeps this in step with the source tree.
DRAFT_RUNNERS = (
    EAGLEDraftCudaGraphRunner,
    EAGLEDraftExtendCudaGraphRunner,
    FrozenKVMTPCudaGraphRunner,
    MultiLayerEagleDraftExtendCudaGraphRunner,
    MultiLayerEagleDraftExtendNpuGraphRunner,
)

_SRT_ROOT = Path(next(iter(sglang.srt.__path__)))
# Pre-filter token. Must be broad enough for TRANSITIVE subclasses: a file
# defining `class X(MultiLayerEagleDraftExtendCudaGraphRunner)` never names
# DecodeCudaGraphRunner itself, so filtering on the base class name alone
# silently loses it. "GraphRunner" is the common suffix of every class in the
# family. Under-filtering is caught loudly by the ratchet's shrunk check.
_FILTER_TOKEN = "GraphRunner"

# Set by the base __init__ for the width ladder, read by inherited methods.
WIDTH_ATTRS = ("decode_graph_widths", "_active_decode_graph_width")

RUNNER_MOD = "sglang.srt.model_executor.runner.decode_cuda_graph_runner"


def _self_attrs(source_obj, ctx):
    """Names of ``self.<name>`` loads or stores in an object's source."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(source_obj)))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and isinstance(node.ctx, ctx)
    }


def _self_stores_anywhere(cls):
    """``self.<name>`` stores anywhere in the classes below the base runner.

    Not only ``__init__``: one runner initialises in
    ``init_buffers_and_capture``, which then calls ``capture()``. Not only the
    class itself: the NPU variants define no ``__init__`` and run their
    parent's.
    """
    names = set()
    for klass in cls.__mro__:
        if klass is DecodeCudaGraphRunner:
            break
        names |= _self_attrs(klass, ast.Store)
    return names


def _discover_exposed_runners():
    """Names of ``DecodeCudaGraphRunner`` subclasses that skip ``super().__init__()``.

    Static scan, so it sees runners this module never imports. A class with no
    ``__init__`` of its own inherits its parent's verdict.
    """
    bases, own_init = {}, {}
    for path in _SRT_ROOT.rglob("*.py"):
        try:
            source = path.read_text()
        except UnicodeDecodeError:
            continue
        # Cheap pre-filter: only files naming some *GraphRunner can define a
        # subclass in the family. Keeps whole-tree coverage without parsing
        # ~1500 files that cannot contribute.
        if _FILTER_TOKEN not in source:
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            # Accept both `class X(Base)` and `class X(module.Base)`; a
            # dotted base yields an ast.Attribute whose .attr is the class
            # name. Dropping those would let a subclass declared that way
            # escape the ratchet entirely.
            bases[node.name] = [
                b.id if isinstance(b, ast.Name) else b.attr
                for b in node.bases
                if isinstance(b, (ast.Name, ast.Attribute))
            ]
            own_init[node.name] = next(
                (
                    m
                    for m in node.body
                    if isinstance(m, ast.FunctionDef) and m.name == "__init__"
                ),
                None,
            )

    family = {DecodeCudaGraphRunner.__name__}
    while True:
        grown = {c for c, bs in bases.items() if family.intersection(bs)} - family
        if not grown:
            break
        family |= grown

    def skips_super(name, seen=()):
        init = own_init.get(name)
        if init is None:  # inherits one; follow the first base in the family
            parents = [b for b in bases.get(name, []) if b in family and b not in seen]
            return any(skips_super(p, (*seen, name)) for p in parents)
        return not any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "__init__"
            and isinstance(n.func.value, ast.Call)
            and getattr(n.func.value.func, "id", None) == "super"
            for n in ast.walk(init)
        )

    return {
        name for name in family - {DecodeCudaGraphRunner.__name__} if skips_super(name)
    }


def _draft_runner(cls):
    """Only the state a draft runner's own ``__init__`` would have set."""
    runner = cls.__new__(cls)
    runner.model_runner = SimpleNamespace(
        device="cpu", gpu_id=0, model=object(), tp_group=None
    )
    runner.capture_bs = [1]
    runner.compile_bs = set()
    runner.captured_req_width = 1
    runner.attn_backend = Mock()
    runner.capture_one_shape = Mock()
    return runner


@contextmanager
def _capture_loop_isolated():
    """Stub the device and compile plumbing so the loop body is observable."""

    @contextmanager
    def fake_patch_model(*_args, **_kwargs):
        yield "forward"

    with patch(f"{RUNNER_MOD}.get_available_gpu_memory", return_value=0.0), patch(
        f"{RUNNER_MOD}.get_parallel", return_value=SimpleNamespace(tp_rank=1)
    ), patch(
        f"{RUNNER_MOD}.torch_compile_decoration.patch_model", fake_patch_model
    ), patch(
        f"{RUNNER_MOD}._set_capture_lora_variant"
    ):
        yield


@contextmanager
def _without_class_defaults():
    """Remove the defaults so a test can observe the original defect."""
    saved = {name: DecodeCudaGraphRunner.__dict__[name] for name in WIDTH_ATTRS}
    for name in WIDTH_ATTRS:
        delattr(DecodeCudaGraphRunner, name)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(DecodeCudaGraphRunner, name, value)


class TestDraftRunnersInheritWidthDefaults(CustomTestCase):
    def test_base_class_carries_none_defaults(self):
        # The base class owns them, so a fifth draft runner inherits them free.
        for name in WIDTH_ATTRS:
            with self.subTest(attr=name):
                self.assertIn(name, DecodeCudaGraphRunner.__dict__)
                self.assertIsNone(DecodeCudaGraphRunner.__dict__[name])

    def test_both_attributes_resolve_on_an_instance_built_without_super(self):
        # The capture loop needs decode_graph_widths; execute()'s replay log
        # needs _active_decode_graph_width, and
        # MultiLayerEagleDraftExtendCudaGraphRunner inherits execute rather
        # than overriding it. Both must resolve on a runner that never ran the
        # base __init__.
        for cls in DRAFT_RUNNERS:
            runner = cls.__new__(cls)
            for name in WIDTH_ATTRS:
                with self.subTest(runner=cls.__name__, attr=name):
                    self.assertIsNone(getattr(runner, name))

    def test_draft_runners_do_not_shadow_the_defaults(self):
        # Assigning either name would opt the draft side into the ladder.
        for cls in DRAFT_RUNNERS:
            stores = _self_stores_anywhere(cls)
            for name in WIDTH_ATTRS:
                with self.subTest(runner=cls.__name__, attr=name):
                    self.assertIsNone(inspect.getattr_static(cls, name))
                    self.assertNotIn(name, stores)


class TestExposedRunnerRatchet(CustomTestCase):
    """Keep DRAFT_RUNNERS in step with the tree, in both directions.

    A new runner that skips ``super().__init__()`` is exposed to this defect
    class and the tests below would not cover it; a runner that starts calling
    ``super().__init__()`` should leave the roster.
    """

    def test_roster_matches_the_source_tree(self):
        pinned = {cls.__name__ for cls in DRAFT_RUNNERS}
        discovered = _discover_exposed_runners()

        grown = discovered - pinned
        self.assertFalse(
            grown,
            f"{sorted(grown)} subclass DecodeCudaGraphRunner without calling "
            "super().__init__(), so they inherit the capture path with only "
            "part of its state. Add them to DRAFT_RUNNERS, or call "
            "super().__init__().",
        )
        shrunk = pinned - discovered
        self.assertFalse(
            shrunk,
            f"{sorted(shrunk)} no longer skip super().__init__(); drop them "
            "from DRAFT_RUNNERS to lock in the progress.",
        )


class TestBackendCarriesTheDefault(CustomTestCase):
    """The runner reads `self.attn_backend.decode_graph_widths` directly.

    That read is only safe because every backend inherits the attribute. A
    backend that never assigns it must still answer None rather than raise.
    """

    def test_a_backend_that_never_assigns_it_still_answers_none(self):
        self.assertIn("decode_graph_widths", AttentionBackend.__dict__)
        self.assertIsNone(AttentionBackend.__dict__["decode_graph_widths"])

        class _BareBackend(AttentionBackend):
            def __init__(self):
                pass

        self.assertIsNone(_BareBackend().decode_graph_widths)


class TestDraftRunnerCaptureLoop(CustomTestCase):
    def test_capture_loop_runs_and_never_narrows_the_draft_backend(self):
        for cls in DRAFT_RUNNERS:
            with self.subTest(runner=cls.__name__):
                runner = _draft_runner(cls)
                with _capture_loop_isolated():
                    runner._capture_one_stream()

                runner.capture_one_shape.assert_called_once_with(
                    1, "forward", None, None
                )
                runner.attn_backend.set_decode_graph_width.assert_not_called()
                self.assertIsNone(runner._active_decode_graph_width)

    def test_capture_loop_fails_without_the_defaults(self):
        # Control: if this stops failing, the test above guards nothing.
        runner = _draft_runner(EAGLEDraftCudaGraphRunner)
        with _without_class_defaults(), _capture_loop_isolated():
            with self.assertRaisesRegex(AttributeError, "decode_graph_widths"):
                runner._capture_one_stream()

    def test_draft_graph_keys_carry_no_width_even_with_a_ladder(self):
        # Draft graphs are keyed by batch size alone, so two widths would
        # collide. The base implementation would key "dsv4_seq=8192" here.
        for cls in DRAFT_RUNNERS:
            with self.subTest(runner=cls.__name__):
                runner = _draft_runner(cls)
                runner.decode_graph_widths = DecodeGraphWidths(
                    widths=(8192, 262148), label="dsv4_seq"
                )
                runner._active_decode_graph_width = 8192
                key = runner._make_graph_key(4)
                self.assertEqual(key, ShapeKey(size=4))
                self.assertIsNone(key.variant_label)


class TestCaptureLoopReadsOnlyDraftProvidedState(CustomTestCase):
    """Guard the defect class, not only the two attributes that hit it: a new
    instance attribute set in the base ``__init__`` and read by
    ``_capture_one_stream`` fails here rather than at capture on a server.
    """

    def test_every_read_is_resolvable_on_each_draft_runner(self):
        reads = _self_attrs(DecodeCudaGraphRunner._capture_one_stream, ast.Load)
        self.assertIn("decode_graph_widths", reads)  # the original crash site
        for cls in DRAFT_RUNNERS:
            assigned = _self_stores_anywhere(cls)
            unresolved = {
                name
                for name in reads
                if name not in assigned
                and not callable(inspect.getattr_static(cls, name, None))
                and not isinstance(inspect.getattr_static(cls, name, None), property)
                and name not in DecodeCudaGraphRunner.__dict__
            }
            with self.subTest(runner=cls.__name__):
                self.assertEqual(unresolved, set())


if __name__ == "__main__":
    unittest.main()
