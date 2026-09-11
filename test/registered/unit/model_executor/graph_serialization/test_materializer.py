"""CPU unit tests for the per-shape materializer seam (design section 6.8).

Backends, stores and runners are fakes: the logic under test (argument
forwarding, verdicts, fallback, cross-rank agreement, resolution from the
plan the runner was handed) is pure Python.
"""

import contextlib
import sys
from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.graph_serialization import materializer as M
from sglang.srt.model_executor.graph_serialization.fingerprint import (
    GraphArtifactFingerprint,
)
from sglang.srt.model_executor.graph_serialization.format import (
    GraphVerdict,
    OutputSchema,
    RankManifest,
    RunnerBundle,
    SerializedGraph,
    ShapeArtifact,
    ShapeKeyRecord,
    unsupported_shape_artifact,
)
from sglang.srt.model_executor.graph_serialization.materializer import (
    ArtifactGraphMaterializer,
    CaptureOnlyMaterializer,
    GraphImportError,
    GraphLoadContext,
    GraphMaterializer,
    GraphSaveContext,
    ShapePlan,
    agree_verdicts,
    keeps_raw_graphs,
    resolve_materializer,
)
from sglang.srt.model_executor.graph_serialization.plan import (
    CacheMode,
    GraphSerializationPlan,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

RUNNER = "DecodeCudaGraphRunner"


# -- fakes ---------------------------------------------------------------------


class FakeBackend:
    """Records every call; ``import_shape`` optionally raises."""

    def __init__(self, *, import_error=None):
        self.calls = []
        self.sessions = []
        self.import_error = import_error

    @contextlib.contextmanager
    def capture_session(self, stream):
        self.sessions.append(("enter", stream))
        try:
            yield
        finally:
            self.sessions.append(("exit", stream))

    def capture_one(self, *args, **kwargs):
        self.calls.append(("capture_one", args, kwargs))

    def export_shape(self, shape_key, ctx):
        self.calls.append(("export_shape", (shape_key, ctx), {}))
        return _serializable_artifact(shape_key)

    def import_shape(self, shape_key, artifact, ctx):
        self.calls.append(("import_shape", (shape_key, artifact, ctx), {}))
        if self.import_error is not None:
            raise self.import_error

    def names(self):
        return [name for name, _, _ in self.calls]


class FakeStore:
    """Duck-types ``GraphArtifactStore.has_rank`` / ``read_rank``."""

    def __init__(self, bundles=None, *, error=None):
        self.bundles = bundles
        self.error = error
        self.artifact_dir = "<fake>"
        self.reads = 0

    def has_rank(self, rank):
        return self.bundles is not None or self.error is not None

    def read_rank(self, rank):
        self.reads += 1
        if self.error is not None:
            raise self.error
        manifest = RankManifest(
            fingerprint=GraphArtifactFingerprint(), regions=(), placement="relocate"
        )
        return manifest, dict(self.bundles)


def _serializable_artifact(shape_key):
    return ShapeArtifact(
        shape_key=ShapeKeyRecord.from_shape_key(shape_key),
        backend="full",
        graphs=(
            SerializedGraph(
                nodes=(), edges=(), param_bytes=b"", slots=(), signature="sig"
            ),
        ),
        output=OutputSchema(kind="none"),
    )


def _bundle(*artifacts, runner=RUNNER):
    return RunnerBundle(runner=runner, backend="full", shapes=tuple(artifacts))


def _plan(mode, **kw):
    return GraphSerializationPlan(mode=CacheMode(mode), cache_dir="/nonexistent", **kw)


def _save_ctx():
    return GraphSaveContext(
        codec=object(), registry=object(), resolver=object(), policy=object()
    )


def _load_ctx():
    return GraphLoadContext(
        codec=object(), reloc=object(), resolver=object(), events=object(), model=None
    )


def _shape_plan(shape_key, **kw):
    kw.setdefault("forward_fn", lambda: None)
    return ShapePlan(shape_key=shape_key, **kw)


def _planned(materializer, *shape_keys):
    """Agree verdicts over ``shape_keys`` single-process, as a runner's
    ``capture`` does through ``plan(self)`` before its shape loop."""
    materializer.plan(SimpleNamespace(planned_shape_keys=lambda: list(shape_keys)))
    return materializer


SK = ShapeKey(size=8)
SK_LORA = ShapeKey(size=16, variant_label="lora")


# -- records -------------------------------------------------------------------


def test_records_are_frozen_keyword_only():
    plan = _shape_plan(SK)
    assert plan.capture_inputs is None
    assert plan.post_warmup_hook is None
    assert plan.event_roles == {}
    with pytest.raises(AttributeError):
        plan.shape_key = SK_LORA
    with pytest.raises(TypeError):
        ShapePlan(SK, lambda: None)
    # Mutable defaults are per instance.
    assert _shape_plan(SK).event_roles is not _shape_plan(SK).event_roles
    ctx = _load_ctx()
    assert ctx.device_ctx == 0 and ctx.dedup is None
    assert _save_ctx().event_roles == {}


# -- CaptureOnlyMaterializer ---------------------------------------------------


def test_capture_only_forwards_exactly_four_arguments():
    backend = FakeBackend()
    materializer = CaptureOnlyMaterializer(backend)
    forward_fn = lambda: "fwd"  # noqa: E731
    capture_inputs = object()
    hook = lambda: None  # noqa: E731

    materializer.materialize(
        ShapePlan(
            shape_key=SK,
            forward_fn=forward_fn,
            capture_inputs=capture_inputs,
            post_warmup_hook=hook,
            event_roles={"metadata_prep_done": object()},
        )
    )

    assert backend.calls == [
        (
            "capture_one",
            (SK, forward_fn),
            {"capture_inputs": capture_inputs, "post_warmup_hook": hook},
        )
    ]
    assert isinstance(materializer, GraphMaterializer)
    assert materializer.backend is backend
    assert materializer.plan(runner=object()) is None
    assert materializer.finish() is None


def test_default_session_wraps_backend_capture_session():
    backend = FakeBackend()
    materializer = CaptureOnlyMaterializer(backend)
    stream = object()
    with materializer.session(stream):
        assert backend.sessions == [("enter", stream)]
    assert backend.sessions == [("enter", stream), ("exit", stream)]


# -- ArtifactGraphMaterializer: capture path -----------------------------------


def test_capture_path_exports_when_saving():
    backend = FakeBackend()
    store = FakeStore()
    save_ctx = _save_ctx()
    materializer = ArtifactGraphMaterializer(
        backend, plan=_plan("save"), store=store, runner_name=RUNNER, save_ctx=save_ctx
    )
    forward_fn = lambda: None  # noqa: E731
    inputs = object()

    materializer.materialize(
        _shape_plan(SK, forward_fn=forward_fn, capture_inputs=inputs)
    )

    assert backend.names() == ["capture_one", "export_shape"]
    assert backend.calls[0] == (
        "capture_one",
        (SK, forward_fn),
        {"capture_inputs": inputs, "post_warmup_hook": None},
    )
    assert backend.calls[1][1] == (SK, save_ctx)
    label = ShapeKeyRecord.from_shape_key(SK).label()
    assert set(materializer.exports) == {label}
    assert materializer.exports[label].shape_key.size == 8
    # Save mode never reads the store.
    assert store.reads == 0


def test_capture_path_does_not_export_when_not_saving():
    backend = FakeBackend()
    materializer = ArtifactGraphMaterializer(
        backend, plan=_plan("load"), store=FakeStore(), runner_name=RUNNER
    )
    materializer.materialize(_shape_plan(SK))
    assert backend.names() == ["capture_one"]
    assert dict(materializer.exports) == {}
    assert materializer.finish() is None


def test_saving_without_save_context_raises_not_implemented():
    backend = FakeBackend()
    materializer = ArtifactGraphMaterializer(
        backend, plan=_plan("save"), store=FakeStore(), runner_name=RUNNER
    )
    with pytest.raises(NotImplementedError, match="section 6.8"):
        materializer.materialize(_shape_plan(SK))
    # Fails before the (wasted) capture.
    assert backend.calls == []


# -- ArtifactGraphMaterializer: load path --------------------------------------


def test_load_path_imports_matching_serializable_artifact():
    backend = FakeBackend()
    artifact = _serializable_artifact(SK)
    store = FakeStore({RUNNER: _bundle(artifact)})
    load_ctx = _load_ctx()
    materializer = ArtifactGraphMaterializer(
        backend, plan=_plan("load"), store=store, runner_name=RUNNER, load_ctx=load_ctx
    )

    # Nothing is agreed before plan() ran, so nothing loads (decision 4).
    assert materializer.verdict_for(SK) == "capture"
    assert store.reads == 0
    _planned(materializer, SK, SK_LORA)
    assert materializer.verdict_for(SK) == "load"
    assert materializer.verdict_for(SK_LORA) == "capture"
    materializer.materialize(_shape_plan(SK))

    assert backend.calls == [("import_shape", (SK, artifact, load_ctx), {})]
    label = ShapeKeyRecord.from_shape_key(SK).label()
    assert dict(materializer.imported) == {label: artifact}
    # The store is read once (by plan) and cached.
    materializer.materialize(_shape_plan(SK_LORA))
    assert backend.names() == ["import_shape", "capture_one"]
    assert store.reads == 1


def test_no_shape_loads_before_plan_agreed_it():
    backend = FakeBackend()
    store = FakeStore({RUNNER: _bundle(_serializable_artifact(SK))})
    materializer = ArtifactGraphMaterializer(
        backend,
        plan=_plan("load"),
        store=store,
        runner_name=RUNNER,
        load_ctx=_load_ctx(),
    )
    # A materialize() without a preceding plan() captures: the per-rank
    # fail-open decision 4 forbids never happens, and the store is not even read.
    materializer.materialize(_shape_plan(SK))
    assert backend.names() == ["capture_one"]
    assert store.reads == 0
    assert dict(materializer.imported) == {}


def test_needs_recapture_artifact_captures():
    backend = FakeBackend()
    artifact = unsupported_shape_artifact(SK, "tc_piecewise", "compile owns graphs")
    materializer = ArtifactGraphMaterializer(
        backend,
        plan=_plan("load"),
        store=FakeStore({RUNNER: _bundle(artifact)}),
        runner_name=RUNNER,
        load_ctx=_load_ctx(),
    )
    _planned(materializer, SK)
    assert materializer.verdict_for(SK) == "capture"
    materializer.materialize(_shape_plan(SK))
    assert backend.names() == ["capture_one"]


def test_bundle_for_other_runner_captures():
    backend = FakeBackend()
    store = FakeStore({"PrefillCudaGraphRunner": _bundle(_serializable_artifact(SK))})
    materializer = ArtifactGraphMaterializer(
        backend, plan=_plan("load"), store=store, runner_name=RUNNER
    )
    _planned(materializer, SK)
    assert materializer.verdict_for(SK) == "capture"


def test_import_error_falls_back_to_capture_when_not_strict():
    backend = FakeBackend(import_error=GraphImportError("kernel unresolved"))
    artifact = _serializable_artifact(SK)
    materializer = ArtifactGraphMaterializer(
        backend,
        plan=_plan("load", strict=False),
        store=FakeStore({RUNNER: _bundle(artifact)}),
        runner_name=RUNNER,
        load_ctx=_load_ctx(),
    )
    _planned(materializer, SK)
    forward_fn = lambda: None  # noqa: E731
    materializer.materialize(_shape_plan(SK, forward_fn=forward_fn))
    assert backend.names() == ["import_shape", "capture_one"]
    assert backend.calls[1] == (
        "capture_one",
        (SK, forward_fn),
        {"capture_inputs": None, "post_warmup_hook": None},
    )
    assert dict(materializer.imported) == {}
    assert dict(materializer.exports) == {}


def test_import_error_reraises_when_strict():
    backend = FakeBackend(import_error=GraphImportError("kernel unresolved"))
    materializer = ArtifactGraphMaterializer(
        backend,
        plan=_plan("load", strict=True),
        store=FakeStore({RUNNER: _bundle(_serializable_artifact(SK))}),
        runner_name=RUNNER,
        load_ctx=_load_ctx(),
    )
    _planned(materializer, SK)
    with pytest.raises(GraphImportError, match="kernel unresolved"):
        materializer.materialize(_shape_plan(SK))
    assert backend.names() == ["import_shape"]


def test_auto_mode_imports_then_captures_and_exports_the_rest():
    backend = FakeBackend()
    materializer = ArtifactGraphMaterializer(
        backend,
        plan=_plan("auto"),
        store=FakeStore({RUNNER: _bundle(_serializable_artifact(SK))}),
        runner_name=RUNNER,
        save_ctx=_save_ctx(),
        load_ctx=_load_ctx(),
    )
    _planned(materializer, SK, SK_LORA)
    materializer.materialize(_shape_plan(SK))
    materializer.materialize(_shape_plan(SK_LORA))
    assert backend.names() == ["import_shape", "capture_one", "export_shape"]


def test_loading_without_load_context_raises_not_implemented():
    backend = FakeBackend()
    materializer = ArtifactGraphMaterializer(
        backend,
        plan=_plan("load"),
        store=FakeStore({RUNNER: _bundle(_serializable_artifact(SK))}),
        runner_name=RUNNER,
    )
    _planned(materializer, SK)
    with pytest.raises(NotImplementedError, match="GraphLoadContext"):
        materializer.materialize(_shape_plan(SK))
    assert backend.calls == []


def test_unreadable_store_captures_unless_strict():
    lenient = ArtifactGraphMaterializer(
        FakeBackend(),
        plan=_plan("load"),
        store=FakeStore(error=ValueError("format_version 7")),
        runner_name=RUNNER,
    )
    _planned(lenient, SK)
    assert lenient.verdict_for(SK) == "capture"

    strict = ArtifactGraphMaterializer(
        FakeBackend(),
        plan=_plan("load", strict=True),
        store=FakeStore(error=ValueError("format_version 7")),
        runner_name=RUNNER,
    )
    with pytest.raises(ValueError, match="format_version 7"):
        _planned(strict, SK)


# -- finish --------------------------------------------------------------------


def test_finish_raises_not_implemented_after_export():
    materializer = ArtifactGraphMaterializer(
        FakeBackend(),
        plan=_plan("save"),
        store=FakeStore(),
        runner_name=RUNNER,
        save_ctx=_save_ctx(),
    )
    materializer.materialize(_shape_plan(SK))
    with pytest.raises(NotImplementedError) as info:
        materializer.finish()
    assert "section 6.8" in str(info.value)
    assert "section 12" in str(info.value)


def test_finish_raises_not_implemented_after_import():
    materializer = ArtifactGraphMaterializer(
        FakeBackend(),
        plan=_plan("load"),
        store=FakeStore({RUNNER: _bundle(_serializable_artifact(SK))}),
        runner_name=RUNNER,
        load_ctx=_load_ctx(),
    )
    _planned(materializer, SK)
    materializer.materialize(_shape_plan(SK))
    with pytest.raises(NotImplementedError, match="section 6.10"):
        materializer.finish()


# -- agreement -----------------------------------------------------------------


def test_agree_verdicts_happy_path():
    assert agree_verdicts([]) == []
    assert agree_verdicts([["load", "capture", "load"]]) == ["load", "capture", "load"]
    assert agree_verdicts(
        [
            ["load", "load", "capture", "load"],
            ["load", "capture", "capture", "load"],
            ("load", "load", "load", "load"),
        ]
    ) == ["load", "capture", "capture", "load"]


def test_agree_verdicts_rejects_length_mismatch_and_unknown_tokens():
    with pytest.raises(ValueError, match="rank 1 has 1 entries; rank 0 has 2"):
        agree_verdicts([["load", "load"], ["load"]])
    with pytest.raises(ValueError, match="unknown verdict"):
        agree_verdicts([["load", "serializable"]])


def test_plan_computes_and_agrees_verdicts_over_planned_shapes(monkeypatch):
    backend = FakeBackend()
    store = FakeStore({RUNNER: _bundle(_serializable_artifact(SK))})
    runner = SimpleNamespace(planned_shape_keys=lambda: [SK, SK_LORA])

    alone = ArtifactGraphMaterializer(
        backend,
        plan=_plan("load"),
        store=store,
        runner_name=RUNNER,
        load_ctx=_load_ctx(),
    )
    alone.plan(runner)
    assert alone.verdict_for(SK) == "load"
    assert alone.verdict_for(SK_LORA) == "capture"
    # A shape that was not planned is never agreed, so it captures.
    assert alone.verdict_for(ShapeKey(size=32)) == "capture"

    # Another rank cannot load SK: the agreed verdict demotes it here too.
    demoted = ArtifactGraphMaterializer(
        backend,
        plan=_plan("load"),
        store=store,
        runner_name=RUNNER,
        load_ctx=_load_ctx(),
    )
    seen = {}

    def fake_gather(local):
        seen["local"] = list(local)
        return [list(local), ["capture", "capture"]]

    monkeypatch.setattr(demoted, "_gather_verdicts", fake_gather)
    demoted.plan(runner)
    assert seen["local"] == ["load", "capture"]
    assert demoted.verdict_for(SK) == "capture"
    demoted.materialize(_shape_plan(SK))
    assert backend.names() == ["capture_one"]


def test_plan_without_planned_shape_keys_captures_everything(caplog):
    backend = FakeBackend()
    materializer = ArtifactGraphMaterializer(
        backend,
        plan=_plan("load"),
        store=FakeStore({RUNNER: _bundle(_serializable_artifact(SK))}),
        runner_name=RUNNER,
        load_ctx=_load_ctx(),
    )
    with caplog.at_level("WARNING", logger=M.__name__):
        assert materializer.plan(SimpleNamespace()) is None
    # No shape list, no agreement, no load (fail closed) -- and it is said.
    assert materializer.verdict_for(SK) == "capture"
    materializer.materialize(_shape_plan(SK))
    assert backend.names() == ["capture_one"]
    assert any("every shape captures" in rec.getMessage() for rec in caplog.records)


def test_gather_without_cpu_group_is_local_only():
    materializer = ArtifactGraphMaterializer(
        FakeBackend(), plan=_plan("load"), store=FakeStore(), runner_name=RUNNER
    )
    assert materializer._gather_verdicts(["load", "capture"]) == [["load", "capture"]]
    # A CPU group without an initialized torch.distributed is single-rank too.
    grouped = ArtifactGraphMaterializer(
        FakeBackend(),
        plan=_plan("load"),
        store=FakeStore(),
        runner_name=RUNNER,
        cpu_group=object(),
    )
    assert grouped._gather_verdicts(["capture"]) == [["capture"]]


# -- resolve_materializer ------------------------------------------------------


def _leaf_declared(name):
    from sglang.srt.server_args import ServerArgs

    return name in ServerArgs.__dataclass_fields__


def test_resolve_materializer_without_a_plan_is_capture_only():
    # A runner built outside the cuda_graph_setup lifecycle (spec, platform
    # and out-of-tree runners) hands over None and is capture-only.
    backend = FakeBackend()
    materializer = resolve_materializer(SimpleNamespace(backend=backend), None)
    assert type(materializer) is CaptureOnlyMaterializer
    assert materializer.backend is backend


def test_resolve_materializer_disabled_plan_is_capture_only():
    backend = FakeBackend()
    runner = SimpleNamespace(backend=backend)
    assert type(resolve_materializer(runner, GraphSerializationPlan())) is (
        CaptureOnlyMaterializer
    )
    # The component's device / draft-worker gate: mode was "save" in the
    # config, the plan says off, and the plan wins.
    forced_off = _plan("save").disabled("device 'npu' is not cuda")
    assert not forced_off.enabled
    materializer = resolve_materializer(runner, forced_off)
    assert type(materializer) is CaptureOnlyMaterializer
    assert materializer.backend is backend


@pytest.mark.skipif(
    not _leaf_declared("cuda_graph_cache_mode"),
    reason="ServerArgs.cuda_graph_cache_mode is not declared in this tree",
)
def test_resolve_materializer_ignores_a_config_that_says_save(tmp_path):
    # The published bag says "save"; the runner was handed no plan (or a
    # disabled one). The plan is the single source of truth, so the config
    # is never consulted and the runner stays capture-only.
    from sglang.srt.runtime_context import get_context, get_exec

    override = get_context().override_server_args(
        cuda_graph_cache_mode="save", cuda_graph_cache_dir=str(tmp_path)
    )
    override.install()
    try:
        assert get_exec().graph.cuda_graph_cache_mode == "save"
        runner = SimpleNamespace(backend=FakeBackend())
        assert type(resolve_materializer(runner, None)) is CaptureOnlyMaterializer
        disabled = _plan("save").disabled("device 'npu' is not cuda")
        assert type(resolve_materializer(runner, disabled)) is (CaptureOnlyMaterializer)
    finally:
        override.restore()


def test_materializer_module_never_reads_the_config():
    import pathlib

    source = pathlib.Path(M.__file__).read_text(encoding="utf-8")
    assert "read_plan_from_config" not in source
    assert "get_exec" not in source


def test_resolve_materializer_enabled_plan_does_not_touch_the_filesystem(tmp_path):
    cache_dir = tmp_path / "cuda_graphs"
    plan = GraphSerializationPlan(mode=CacheMode.SAVE, cache_dir=str(cache_dir))

    class DecodeCudaGraphRunner:
        def __init__(self):
            self.backend = FakeBackend()
            self.model_runner = SimpleNamespace(
                tp_group=SimpleNamespace(cpu_group="cpu-group")
            )

    runner = DecodeCudaGraphRunner()
    materializer = resolve_materializer(runner, plan)

    assert isinstance(materializer, ArtifactGraphMaterializer)
    assert materializer.plan_settings is plan
    assert materializer.backend is runner.backend
    assert materializer.runner_name == "decode"  # RunnerBundle key, not the class
    assert materializer.store.root == cache_dir
    assert materializer.store.digest == "pending"
    assert materializer.store.artifact_dir == cache_dir / "pending"
    assert materializer._cpu_group == "cpu-group"
    assert not cache_dir.exists()
    assert not tmp_path.joinpath("pending").exists()


def test_resolve_materializer_plan_argument_is_required():
    with pytest.raises(TypeError):
        resolve_materializer(SimpleNamespace(backend=FakeBackend()))


def test_keeps_raw_graphs_follows_the_plan():
    assert keeps_raw_graphs(None) is False
    assert keeps_raw_graphs(GraphSerializationPlan()) is False
    assert keeps_raw_graphs(_plan("save").disabled("device 'npu' is not cuda")) is (
        False
    )
    for mode in ("save", "load", "auto"):
        assert keeps_raw_graphs(_plan(mode)) is True


def test_shape_artifact_verdict_drives_the_load_decision():
    # Sanity on the foundation helper the verdict rests on.
    assert M.shape_artifact_verdict(_serializable_artifact(SK)) is (
        GraphVerdict.SERIALIZABLE
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
