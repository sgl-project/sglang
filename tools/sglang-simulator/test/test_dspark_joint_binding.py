"""Explicit joint construction, with declared source facts and no timing rows.

The native checkpoint serializer and original-material loader are separate
boundaries here. Original-source reconstruction is also checked on retained
source-only receipts; these fixtures do not attest native execution.
"""

import copy
import hashlib
import json
import sys
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
from sglang_simulator.simulation.manager.config import ConfigManager
from sglang_simulator.simulation.sglang.utils import resolve_scheduler_config
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.time_predictor import infercast_dspark as owner
from sglang_simulator.time_predictor.base import PredictorError
from test_dspark_config import factory as config_factory  # noqa: F401 - pytest fixture
from test_dspark_hybrid import joint_case as joint_case  # noqa: F401 - pytest fixture
from test_dspark_joint_execution import (  # noqa: F401 - pytest fixture
    case as joint_geometry,
)
from test_dspark_joint_execution import (
    prepare,
    resolve,
)
from test_dspark_native_routing import context as context  # noqa: F401 - pytest fixture


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


@pytest.fixture
def factory(context, joint_geometry, tmp_path, monkeypatch):
    import infercast
    from infercast.sdk.models.umd import build_model_description
    from infercast.sdk.perf_database.dsv41_dspark_proposal import MODEL_REVISION

    model_id = "deepseek-ai/DeepSeek-V4.1-Flash"
    geometry = build_model_description(model_id, backend="sglang").geometry
    model = NS(
        **{
            name: geometry[key]
            for name, key in {
                "hidden_size": "hidden_size",
                "num_hidden_layers": "num_layers",
                "num_attention_heads": "num_attention_heads",
                "num_key_value_heads": "num_key_value_heads",
                "head_dim": "head_dim",
                "vocab_size": "vocab_size",
                "qk_rope_head_dim": "qk_rope_head_dim",
                "indexer_head_dim": "index_head_dim",
                "window_size": "sliding_window",
                "compression_ratios": "compression_ratios_source",
            }.items()
        }
    )
    monkeypatch.setattr(infercast, "__revision__", "b" * 40)
    monkeypatch.setattr(
        owner,
        "_bind_local_model_config",
        lambda *a, **kw: {
            "sdk_canonical_sha256": context["model_config_sha256"],
            "structural_fixture_not_native_checkpoint": True,
        },
    )
    source_calls = []

    def source(binding, runtime, revision):
        source_calls.append((copy.deepcopy(binding), runtime, revision))
        assert runtime == context["runtime_profile_json"] and revision == MODEL_REVISION
        return (
            context["source_policy"],
            context["profiles"],
            {"structural_source_fixture": True},
        )

    monkeypatch.setattr(owner, "_native_joint_source_from_binding", source)

    def asset(name, value):
        path = tmp_path / name
        raw = value.encode() if type(value) is str else canonical(value)
        path.write_bytes(raw)
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest())

    options = dict(
        model_id=model_id,
        model_revision=MODEL_REVISION,
        provider_revision="b" * 40,
        system="mi355x",
        systems_root=str(tmp_path / "systems"),
        runtime_profile=asset("runtime.json", context["runtime_profile_json"]),
        progress_kind="source_golden351",
        model_seed=42,
        cost_mode="HYBRID",
        native_joint_source_binding={"root": "structural-only"},
        native_analytic_model={
            mode: asset(mode + ".json", value.to_dict())
            for mode, value in context["analytic_models"].items()
        },
    )
    config = SchedulerConfig(tp_size=2, dp_size=2, backend_version="0.5.22")
    return NS(
        context=context,
        geometry=joint_geometry,
        model=model,
        options=options,
        config=config,
        source_calls=source_calls,
    )


def create(factory, **changes):
    return owner.InferCastDSparkTimePredictor(
        factory.model,
        NS(name="mi355x"),
        factory.config,
        **{**factory.options, **changes},
    )


def test_joint_shared_ledger_constructor_remains_exclusive(factory, tmp_path):
    """The original direct constructor reproduces the actual xb conflict."""
    path = tmp_path / "shared.jsonl"
    first = create(factory, native_cost_evidence_path=str(path))
    try:
        with pytest.raises(
            PredictorError, match="cannot create native cost evidence"
        ) as caught:
            create(factory, native_cost_evidence_path=str(path))
        assert isinstance(caught.value.__cause__, FileExistsError)
        assert path.read_bytes() == b""
    finally:
        first.dspark_runtime.session.cost_model._cost_sink.close()


def test_joint_config_creates_two_real_exclusive_writers(
    factory, tmp_path, monkeypatch
):
    """Real predictor/SDK/UMD constructors, declared source, zero timing rows."""
    raw = {
        "platform": {"accelerator": {"name": "mi355x"}},
        "predictor": {
            "name": "infercast_dspark",
            **{key: value for key, value in factory.options.items() if key != "system"},
            "native_cost_evidence_path": str(tmp_path / "costs.jsonl"),
        },
    }
    original = copy.deepcopy(raw)
    config_path = tmp_path / "sim.json"
    config_path.write_text(json.dumps(raw), encoding="utf-8")
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(ConfigManager, "_raw_config", raw)
    predictors = []
    try:
        for rank in (0, 1):
            predictors.append(
                ConfigManager.get_inference_time_predictor(
                    factory.model,
                    NS(name="mi355x"),
                    factory.config,
                    attention_dp_rank=rank,
                )
            )
        for rank, predictor in enumerate(predictors):
            path = tmp_path / f"costs.dp{rank}.jsonl"
            metrics = predictor.get_metrics()["infercast_dspark"]
            assert path.read_bytes() == b""
            assert metrics["cost_grain"] == "whole_worker"
            assert metrics["hybrid_cost_queries"]["by_source"] == {}
            assert metrics["hybrid_cost_queries"]["ledger"]["path"] == str(path)
            assert metrics["hybrid_cost_queries"]["ledger"]["bytes"] == 0
            with pytest.raises(
                PredictorError, match="cannot create native cost evidence"
            ) as caught:
                ConfigManager.get_inference_time_predictor(
                    factory.model,
                    NS(name="mi355x"),
                    factory.config,
                    attention_dp_rank=rank,
                )
            assert isinstance(caught.value.__cause__, FileExistsError)
        assert not (tmp_path / "costs.jsonl").exists()
        assert raw == original
    finally:
        for predictor in predictors:
            predictor.dspark_runtime.session.cost_model._cost_sink.close()


@pytest.mark.parametrize("mode", ["EXTEND", "DECODE"])
def test_joint_constructor_reaches_real_umd_typed_miss_sol(factory, mode):
    predictor = create(factory)
    assert len(factory.source_calls) == 1
    # Existing real SDK projection and UMD, not a regional ADP1 projector.
    geometry = copy.copy(factory.geometry)
    geometry.resolver = predictor.dspark_runtime._joint_execution_resolver
    step = resolve(geometry, prepare(geometry, (1, 2), mode))
    timing = predictor.dspark_runtime.session.cost_model.predict_step(step)
    assert timing.seconds > 0 and len(timing.phases) == 1
    assert timing.phases[0].phase == "whole_worker"
    metrics = predictor.get_metrics()["infercast_dspark"]
    assert metrics["cost_grain"] == "whole_worker"
    assert metrics["topology"] == dict(tp=2, dp=2, pp=1, ep=1, cp=1)
    assert metrics["accuracy_admitted"] is False
    assert predictor.dspark_runtime.session._history == {}


@pytest.mark.parametrize(
    "changes",
    [
        {"cost_mode": "SILICON", "native_analytic_model": None},
        {"native_segment_profile_id": "c" * 64},
        {"native_source_binding": {}},
        {"native_initial_draft_buffer": {}},
        {"native_source_work_policy": "any"},
    ],
)
def test_joint_never_borrows_regional_or_empirical_selectors(factory, changes):
    with pytest.raises(PredictorError):
        create(factory, **changes)
    assert not factory.source_calls


def test_joint_missing_model_phase_and_foreign_model_profile_fail_before_sink(
    factory, tmp_path
):
    sink = tmp_path / "not-created.jsonl"
    models = factory.options["native_analytic_model"]
    for bad in (
        {"DECODE": models["DECODE"]},
        {"EXTEND": models["DECODE"], "DECODE": models["DECODE"]},
    ):
        with pytest.raises(PredictorError):
            create(
                factory, native_analytic_model=bad, native_cost_evidence_path=str(sink)
            )
        assert not sink.exists()


def native():
    return dict(
        tp_size=2,
        dp_size=1,
        attn_dp_size=2,
        pp_size=1,
        attn_cp_size=1,
        ep_size=1,
        nnodes=1,
        dtype="bfloat16",
        kv_cache_dtype="auto",
    )


def test_both_native_config_owners_map_only_explicit_joint_without_mutation(
    config_factory, monkeypatch
):
    _, raw, _ = config_factory
    raw["predictor"].update(
        native_joint_source_binding={"root": "facts"},
        cost_mode="HYBRID",
        progress_kind="source_golden351",
    )
    original, args = copy.deepcopy(raw), native()
    version = ModuleType("sglang.version")
    version.__version__ = "0.5.22"
    monkeypatch.setitem(sys.modules, version.__name__, version)
    parsed = ConfigManager._parse_server_args(args, "sglang")
    resolved = resolve_scheduler_config(NS(**args))
    for value in (parsed, resolved):
        assert (
            value.tp_size,
            value.attn_dp_size,
            value.attn_tp_size,
            value.moe_tp_size,
        ) == (2, 2, 1, 2)
    assert args == native() and raw == original
    for name in ("infercast", "aiconfigurator", "infercast_dspark"):
        raw["predictor"]["name"] = name
        if name == "infercast_dspark":
            raw["predictor"].pop("native_joint_source_binding")
        assert ConfigManager.modeled_attention_dp_size(args) == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("dp_size", 2),
        ("attn_dp_size", 1),
        ("tp_size", 4),
        ("ep_size", 2),
        ("nnodes", True),
    ],
)
def test_joint_mapping_rejects_native_replica_or_wrong_chip_topology(
    config_factory, key, value
):
    _, raw, _ = config_factory
    raw["predictor"].update(
        native_joint_source_binding={},
        cost_mode="HYBRID",
        progress_kind="source_golden351",
    )
    with pytest.raises(ValueError, match="native DP1"):
        ConfigManager.modeled_attention_dp_size({**native(), key: value})


def test_joint_config_resolves_only_outer_root_and_two_model_assets(config_factory):
    _, raw, folder = config_factory
    raw["predictor"].update(
        native_joint_source_binding={
            "root": "originals",
            "startup": [{"path": "_meta/original.json", "sha256": "e" * 64}],
        },
        native_analytic_model={
            mode: dict(path=mode + ".json", sha256="f" * 64)
            for mode in ("EXTEND", "DECODE")
        },
    )
    original = copy.deepcopy(raw)
    result = ConfigManager.get_inference_time_predictor(None, None, None)
    assert result.native_joint_source_binding["root"] == str(
        (folder / "originals").absolute()
    )
    assert (
        result.native_joint_source_binding["startup"]
        == original["predictor"]["native_joint_source_binding"]["startup"]
    )
    assert result.native_analytic_model["EXTEND"]["path"] == str(folder / "EXTEND.json")
    assert raw == original


@pytest.mark.parametrize(
    "path", ["../outside.json", "_meta/sub/escape.json", "_meta\\bad.json"]
)
def test_joint_original_reader_rejects_noncanonical_path_before_source_use(
    tmp_path, path
):
    pytest.importorskip("infercast.sdk.native_dp_progress")
    root = tmp_path / "source"
    (root / "_meta").mkdir(parents=True)
    binding = dict(
        root=str(root),
        source_runtime={},
        container={},
        profiles={},
        supplemental_files={},
    )
    for name, count in (
        ("startup", 4),
        ("proposal", 2),
        ("windows", 2),
        ("captures", 4),
        ("allocated", 2),
    ):
        binding[name] = [dict(path=path, sha256="a" * 64)] * count
    with pytest.raises(ValueError, match="canonical _meta"):
        owner._native_joint_source_from_binding(binding, "{}", "b" * 40)


def test_joint_original_reader_checks_bytes_before_parsing(tmp_path):
    pytest.importorskip("infercast.sdk.native_dp_progress")
    (tmp_path / "_meta").mkdir()
    (tmp_path / "_meta" / "startup.json").write_bytes(b"{}")
    ref = dict(path="_meta/startup.json", sha256="a" * 64)
    binding = dict(
        root=str(tmp_path),
        source_runtime={},
        container={},
        profiles={},
        supplemental_files={},
    )
    for name, count in (
        ("startup", 4),
        ("proposal", 2),
        ("windows", 2),
        ("captures", 4),
        ("allocated", 2),
    ):
        binding[name] = [ref] * count
    with pytest.raises(PredictorError, match="original bytes differ"):
        owner._native_joint_source_from_binding(binding, "{}", "b" * 40)
