"""CPU unit tests for the backend serialization seam (design section 6.7).

Every in-tree ``BaseCudaGraphBackend`` implementation must be concrete after
the ABC gained ``export_shape`` / ``import_shape``:

* ``TcPiecewiseCudaGraphBackend``, ``NPUCudaGraphBackend`` and
  ``FullXPUGraphBackend`` are out of scope for v1 (design section 9.3): they
  export a ``needs_recapture`` artifact and refuse every import.
* ``FullCudaGraphBackend`` encodes ``raw_cuda_graph()`` through the codec on
  export and installs nothing on a failed import (all-or-nothing). Its
  ``capture_one`` still constructs ``torch.cuda.CUDAGraph()`` with no
  arguments unless ``keep_graph`` was requested, which
  ``resolve_decode_backend`` / ``resolve_prefill_backend`` do exactly when
  the runner's graph-serialization plan is enabled.
* ``BreakableCudaGraphBackend`` encodes one graph per segment and its import
  preamble is all-or-nothing too; the break-site rebuild is a documented stub.

The real capture / codec paths need CUDA, so the backends are built with
``__new__`` and the contexts are ``SimpleNamespace`` fakes.
"""

import contextlib
import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
    NPUCudaGraphBackend,
)
from sglang.srt.hardware_backend.xpu.graph_runner.xpu_full_graph_backend import (
    FullXPUGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.graph_serialization.format import (
    GraphVerdict,
    OutputSchema,
    SerializedGraph,
    ShapeArtifact,
    ShapeKeyRecord,
    shape_artifact_verdict,
)
from sglang.srt.model_executor.graph_serialization.materializer import (
    GraphImportError,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend import utils as backend_utils
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend import (
    TcPiecewiseCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_ALL_BACKENDS = (
    FullCudaGraphBackend,
    BreakableCudaGraphBackend,
    TcPiecewiseCudaGraphBackend,
    NPUCudaGraphBackend,
    FullXPUGraphBackend,
)
_UNSUPPORTED = (
    (TcPiecewiseCudaGraphBackend, "tc_piecewise"),
    (NPUCudaGraphBackend, "npu"),
    (FullXPUGraphBackend, "xpu"),
)


def _serialized(verdict=GraphVerdict.SERIALIZABLE) -> SerializedGraph:
    return SerializedGraph(
        nodes=(), edges=(), param_bytes=b"", slots=(), signature="sig", verdict=verdict
    )


def _artifact(backend: str, *, size: int = 4, n_graphs: int = 1) -> ShapeArtifact:
    return ShapeArtifact(
        shape_key=ShapeKeyRecord(size=size),
        backend=backend,
        graphs=tuple(_serialized() for _ in range(n_graphs)),
        output=OutputSchema(kind="none"),
    )


def _save_ctx(encode=None) -> SimpleNamespace:
    return SimpleNamespace(
        codec=SimpleNamespace(
            encode=encode or mock.Mock(name="encode", return_value=_serialized())
        ),
        registry="REGISTRY",
        resolver="RESOLVER",
        policy="POLICY",
        event_roles={7: "metadata_prep_done"},
    )


def _load_ctx(materialize) -> SimpleNamespace:
    return SimpleNamespace(
        codec=SimpleNamespace(materialize=materialize),
        reloc="RELOC",
        resolver="RESOLVER",
        events="EVENTS",
        model=None,
        device_ctx=0,
        dedup=None,
    )


class _TorchStyleGraph:
    """Stands in for ``torch.cuda.CUDAGraph(keep_graph=True)``."""

    def __init__(self, raw: int) -> None:
        self._raw = raw

    def raw_cuda_graph(self) -> int:
        return self._raw


class TestAbcAndConcreteness(CustomTestCase):
    def test_abc_declares_export_and_import(self):
        self.assertIn("export_shape", BaseCudaGraphBackend.__abstractmethods__)
        self.assertIn("import_shape", BaseCudaGraphBackend.__abstractmethods__)

    def test_every_in_tree_backend_is_concrete(self):
        for cls in _ALL_BACKENDS:
            with self.subTest(backend=cls.__name__):
                self.assertFalse(inspect.isabstract(cls))
                self.assertTrue(issubclass(cls, BaseCudaGraphBackend))


class TestUnsupportedBackends(CustomTestCase):
    def test_export_is_needs_recapture_and_import_refuses(self):
        shape_key = ShapeKey(size=8, stream_idx=1)
        for cls, tag in _UNSUPPORTED:
            with self.subTest(backend=cls.__name__):
                backend = cls.__new__(cls)
                artifact = backend.export_shape(shape_key, object())
                self.assertIsInstance(artifact, ShapeArtifact)
                self.assertEqual(artifact.backend, tag)
                self.assertEqual(
                    artifact.shape_key, ShapeKeyRecord(size=8, stream_idx=1)
                )
                self.assertIs(
                    shape_artifact_verdict(artifact), GraphVerdict.NEEDS_RECAPTURE
                )
                self.assertIn("9.3", artifact.graphs[0].verdict_reason)
                with self.assertRaises(GraphImportError) as caught:
                    backend.import_shape(shape_key, artifact, object())
                self.assertIn("9.3", str(caught.exception))


def _make_runner() -> SimpleNamespace:
    device_module = SimpleNamespace(
        synchronize=mock.Mock(name="synchronize"),
        graph=mock.Mock(
            name="graph", side_effect=lambda **kw: contextlib.nullcontext()
        ),
    )
    return SimpleNamespace(
        device_module=device_module,
        model_runner=SimpleNamespace(tp_group=SimpleNamespace(barrier=mock.Mock())),
        enable_profile_cuda_graph=False,
    )


def _make_full_backend(runner=None) -> FullCudaGraphBackend:
    """Build without ``__init__`` (which touches CUDA), like
    ``test_full_cuda_graph_backend.py`` does."""
    runner = runner or _make_runner()
    backend = FullCudaGraphBackend.__new__(FullCudaGraphBackend)
    backend._graphs = {}
    backend._outputs = {}
    backend._pool = None
    backend._capture_stream = None
    backend._memory_saver_adapter = None
    backend._cuda_graph_runner = runner
    backend._device_module = runner.device_module
    backend._tp_group = runner.model_runner.tp_group
    return backend


class TestFullBackendExport(CustomTestCase):
    def test_unknown_shape_raises_key_error_before_encoding(self):
        backend = _make_full_backend()
        ctx = _save_ctx()
        with self.assertRaises(KeyError):
            backend.export_shape(ShapeKey(size=3), ctx)
        ctx.codec.encode.assert_not_called()

    def test_encodes_raw_graph_then_stops_at_the_output_schema_stub(self):
        backend = _make_full_backend()
        shape_key = ShapeKey(size=4)
        backend._graphs[shape_key] = _TorchStyleGraph(raw=0xBEEF)
        backend._outputs[shape_key] = object()
        ctx = _save_ctx()

        with self.assertRaises(NotImplementedError) as caught:
            backend.export_shape(shape_key, ctx)

        self.assertIn("section 6.7", str(caught.exception))
        ctx.codec.encode.assert_called_once_with(
            0xBEEF,
            registry="REGISTRY",
            resolver="RESOLVER",
            policy="POLICY",
            event_roles={7: "metadata_prep_done"},
        )


class TestFullBackendImport(CustomTestCase):
    def _seeded_backend(self):
        backend = _make_full_backend()
        backend._graphs[ShapeKey(size=1)] = "EXISTING_GRAPH"
        backend._outputs[ShapeKey(size=1)] = "EXISTING_OUT"
        return backend, dict(backend._graphs), dict(backend._outputs)

    def test_codec_failure_leaves_tables_untouched(self):
        backend, graphs_before, outputs_before = self._seeded_backend()
        boom = RuntimeError("kernel unresolved")
        ctx = _load_ctx(mock.Mock(side_effect=boom))

        with self.assertRaises(GraphImportError) as caught:
            backend.import_shape(ShapeKey(size=4), _artifact("full"), ctx)

        self.assertIs(caught.exception.__cause__, boom)
        self.assertEqual(backend._graphs, graphs_before)
        self.assertEqual(backend._outputs, outputs_before)

    def test_foreign_backend_artifact_is_refused_without_materializing(self):
        backend, graphs_before, outputs_before = self._seeded_backend()
        materialize = mock.Mock()
        with self.assertRaises(GraphImportError):
            backend.import_shape(
                ShapeKey(size=4), _artifact("breakable"), _load_ctx(materialize)
            )
        materialize.assert_not_called()
        self.assertEqual(backend._graphs, graphs_before)
        self.assertEqual(backend._outputs, outputs_before)

    def test_materialized_graph_is_not_installed_until_the_output_is_rebuilt(self):
        # The output rebuild is a stub in this draft: it must surface as a
        # chained GraphImportError and, being all-or-nothing, install nothing.
        backend, graphs_before, outputs_before = self._seeded_backend()
        artifact = _artifact("full")
        materialize = mock.Mock(return_value="LOADED")
        ctx = _load_ctx(materialize)

        with self.assertRaises(GraphImportError) as caught:
            backend.import_shape(ShapeKey(size=4), artifact, ctx)

        self.assertIsInstance(caught.exception.__cause__, NotImplementedError)
        self.assertIn("section 6.7", str(caught.exception.__cause__))
        materialize.assert_called_once_with(
            artifact.graphs[0],
            kernels=artifact.kernels,
            reloc="RELOC",
            resolver="RESOLVER",
            events="EVENTS",
            device_ctx=0,
        )
        self.assertEqual(backend._graphs, graphs_before)
        self.assertEqual(backend._outputs, outputs_before)


class TestFullBackendCaptureConstructsGraph(CustomTestCase):
    def test_default_constructs_cuda_graph_with_no_arguments(self):
        backend = _make_full_backend()
        self.assertFalse(backend._keep_graph)
        forward_fn = mock.Mock(return_value=object())
        shape_key = ShapeKey(size=2)

        with mock.patch("torch.cuda.CUDAGraph", return_value="GRAPH") as graph_cls:
            backend.capture_one(shape_key, forward_fn)

        graph_cls.assert_called_once_with()
        self.assertEqual(backend._graphs[shape_key], "GRAPH")
        self.assertEqual(forward_fn.call_count, 3)

    def test_keep_graph_constructs_keep_graph_and_instantiates(self):
        backend = _make_full_backend()
        backend._keep_graph = True
        graph = mock.Mock(name="graph")
        shape_key = ShapeKey(size=2)

        with mock.patch("torch.cuda.CUDAGraph", return_value=graph) as graph_cls:
            backend.capture_one(shape_key, mock.Mock(return_value=object()))

        graph_cls.assert_called_once_with(keep_graph=True)
        graph.instantiate.assert_called_once_with()
        self.assertIs(backend._graphs[shape_key], graph)

    def test_constructor_keyword_defaults_to_off(self):
        parameter = inspect.signature(FullCudaGraphBackend.__init__).parameters[
            "keep_graph"
        ]
        self.assertIs(parameter.default, False)
        self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(FullCudaGraphBackend._keep_graph, False)


def _fake_exec(decode=Backend.FULL, prefill=Backend.FULL) -> SimpleNamespace:
    return SimpleNamespace(
        graph=SimpleNamespace(
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend=decode),
                prefill=SimpleNamespace(backend=prefill),
            ),
            debug_cuda_graph=False,
        ),
        features=SimpleNamespace(enable_memory_saver=False),
    )


class TestBackendResolversKeepGraph(CustomTestCase):
    """The resolvers are the seam that turns ``keep_graph`` on: they forward
    the runner's ``keeps_raw_graphs(plan)`` to the Full backend and nowhere
    else (design section 6.7)."""

    def setUp(self):
        self.runner = SimpleNamespace(model_runner=SimpleNamespace(device="cuda"))

    def test_keep_graph_is_keyword_only_and_off_by_default(self):
        for resolver in (
            backend_utils.resolve_decode_backend,
            backend_utils.resolve_prefill_backend,
        ):
            with self.subTest(resolver=resolver.__name__):
                parameter = inspect.signature(resolver).parameters["keep_graph"]
                self.assertIs(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(parameter.default, False)

    def test_decode_resolver_forwards_keep_graph_to_the_full_backend(self):
        with (
            mock.patch.object(backend_utils, "get_exec", return_value=_fake_exec()),
            mock.patch.object(backend_utils, "FullCudaGraphBackend") as full,
        ):
            backend_utils.resolve_decode_backend(self.runner, keep_graph=True)
            full.assert_called_once_with(
                self.runner, enable_memory_saver=False, keep_graph=True
            )
            full.reset_mock()
            backend_utils.resolve_decode_backend(self.runner)
            full.assert_called_once_with(
                self.runner, enable_memory_saver=False, keep_graph=False
            )

    def test_prefill_resolver_forwards_keep_graph_to_the_full_backend(self):
        with (
            mock.patch.object(backend_utils, "get_exec", return_value=_fake_exec()),
            mock.patch.object(backend_utils, "FullCudaGraphBackend") as full,
        ):
            backend_utils.resolve_prefill_backend(self.runner, keep_graph=True)
            full.assert_called_once_with(
                self.runner, enable_memory_saver=False, keep_graph=True
            )
            full.reset_mock()
            backend_utils.resolve_prefill_backend(self.runner)
            full.assert_called_once_with(
                self.runner, enable_memory_saver=False, keep_graph=False
            )

    def test_breakable_backend_ignores_keep_graph(self):
        with (
            mock.patch.object(
                backend_utils,
                "get_exec",
                return_value=_fake_exec(
                    decode=Backend.BREAKABLE, prefill=Backend.BREAKABLE
                ),
            ),
            mock.patch.object(backend_utils, "BreakableCudaGraphBackend") as bcg,
        ):
            backend_utils.resolve_decode_backend(self.runner, keep_graph=True)
            backend_utils.resolve_prefill_backend(self.runner, keep_graph=True)
        for call in bcg.call_args_list:
            self.assertNotIn("keep_graph", call.kwargs)
        self.assertEqual(bcg.call_count, 2)


def _make_bcg_backend() -> BreakableCudaGraphBackend:
    backend = BreakableCudaGraphBackend.__new__(BreakableCudaGraphBackend)
    backend._graphs = {}
    backend._outputs = {}
    backend._capture_inputs = {}
    return backend


class TestBreakableBackendExport(CustomTestCase):
    def test_unknown_shape_raises_key_error(self):
        with self.assertRaises(KeyError):
            _make_bcg_backend().export_shape(ShapeKey(size=3), _save_ctx())

    def test_encodes_one_graph_per_segment_then_stops_at_break_site_stub(self):
        backend = _make_bcg_backend()
        shape_key = ShapeKey(size=16)
        # One torch-style segment and one dedup-registry wrapper (raw_graph).
        backend._graphs[shape_key] = SimpleNamespace(
            _segments=[_TorchStyleGraph(raw=11), SimpleNamespace(raw_graph=22)],
            _break_fns=[object()],
        )
        backend._outputs[shape_key] = object()
        ctx = _save_ctx()

        with self.assertRaises(NotImplementedError) as caught:
            backend.export_shape(shape_key, ctx)

        self.assertIn("section 6.9", str(caught.exception))
        self.assertEqual(
            [call.args[0] for call in ctx.codec.encode.call_args_list], [11, 22]
        )
        for call in ctx.codec.encode.call_args_list:
            self.assertEqual(
                call.kwargs,
                {
                    "registry": "REGISTRY",
                    "resolver": "RESOLVER",
                    "policy": "POLICY",
                    "event_roles": {7: "metadata_prep_done"},
                },
            )

    def test_segment_without_a_handle_is_a_type_error(self):
        with self.assertRaises(TypeError):
            BreakableCudaGraphBackend._segment_raw_graph(object())


class TestBreakableBackendImport(CustomTestCase):
    def _seeded_backend(self):
        backend = _make_bcg_backend()
        backend._graphs[ShapeKey(size=1)] = "EXISTING_GRAPH"
        backend._outputs[ShapeKey(size=1)] = "EXISTING_OUT"
        return backend, dict(backend._graphs), dict(backend._outputs)

    def test_codec_failure_is_a_chained_import_error(self):
        backend, graphs_before, outputs_before = self._seeded_backend()
        boom = RuntimeError("segment 2 cannot be rebuilt")
        materialize = mock.Mock(side_effect=["SEG0", boom])

        with self.assertRaises(GraphImportError) as caught:
            backend.import_shape(
                ShapeKey(size=4),
                _artifact("breakable", n_graphs=2),
                _load_ctx(materialize),
            )

        self.assertIs(caught.exception.__cause__, boom)
        self.assertEqual(backend._graphs, graphs_before)
        self.assertEqual(backend._outputs, outputs_before)

    def test_foreign_backend_and_empty_artifacts_are_refused(self):
        backend, graphs_before, outputs_before = self._seeded_backend()
        materialize = mock.Mock()
        with self.assertRaises(GraphImportError):
            backend.import_shape(
                ShapeKey(size=4), _artifact("full"), _load_ctx(materialize)
            )
        with self.assertRaises(GraphImportError):
            backend.import_shape(
                ShapeKey(size=4),
                _artifact("breakable", n_graphs=0),
                _load_ctx(materialize),
            )
        materialize.assert_not_called()
        self.assertEqual(backend._graphs, graphs_before)
        self.assertEqual(backend._outputs, outputs_before)

    def test_successful_preamble_ends_in_not_implemented_with_no_partial_state(self):
        backend, graphs_before, outputs_before = self._seeded_backend()
        artifact = _artifact("breakable", n_graphs=3)
        materialize = mock.Mock(side_effect=["SEG0", "SEG1", "SEG2"])

        with self.assertRaises(NotImplementedError) as caught:
            backend.import_shape(ShapeKey(size=4), artifact, _load_ctx(materialize))

        self.assertIn("section 6.9", str(caught.exception))
        self.assertEqual(materialize.call_count, 3)
        self.assertEqual(backend._graphs, graphs_before)
        self.assertEqual(backend._outputs, outputs_before)


if __name__ == "__main__":
    unittest.main()
