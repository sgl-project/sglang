"""Executable DSPark factory identity/coverage contracts, no silicon evidence."""

import copy
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
from sglang_simulator.simulation.sglang.dspark import DSparkSession
from sglang_simulator.simulation.sglang.dspark_runtime import DSparkNativeRuntime
from sglang_simulator.simulation.types import SchedulerConfig
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.base import PredictorError
from sglang_simulator.time_predictor.infercast_dspark import (
    InferCastDSparkTimePredictor,
)


def raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def mock_config_link(monkeypatch, source, target):
    """Model link resolution deterministically on hosts without symlink privilege."""
    original_resolve, original_lstat, original_readlink = (
        Path.resolve,
        Path.lstat,
        Path.readlink,
    )
    state = {"target": target}

    def resolve(path, *args, **kwargs):
        if path == source:
            return original_resolve(state["target"], *args, **kwargs)
        return original_resolve(path, *args, **kwargs)

    def lstat(path, *args, **kwargs):
        result = original_lstat(path, *args, **kwargs)
        if path != source:
            return result
        fields = {
            name: getattr(result, name)
            for name in (
                "st_dev",
                "st_ino",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        }
        return NS(**fields, st_mode=stat.S_IFLNK | 0o777)

    monkeypatch.setattr(Path, "resolve", resolve)
    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(
        Path,
        "readlink",
        lambda path: (
            Path(os.path.relpath(state["target"], source.parent))
            if path == source
            else original_readlink(path)
        ),
    )
    return state


@pytest.fixture
def factory(tmp_path, monkeypatch):
    from sglang_simulator.time_predictor import dspark_progress

    model = ModelInfo(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=32,
        qk_rope_head_dim=2,
        indexer_head_dim=4,
        window_size=8,
        compression_ratios=[0, 4, 0],
        attention_arch="MHA",
    )
    geometry = {
        "hidden_size": 16,
        "num_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "vocab_size": 32,
        "qk_rope_head_dim": 2,
        "index_head_dim": 4,
        "sliding_window": 8,
        "compression_ratios_source": (0, 4, 0),
    }
    authoritative = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "dtype": "bfloat16",
        "transformers_version": "source fixture provenance",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "image_token_id": 27,
        "quantization_config": {"quant_method": "fp8", "expert_dtype": "fp4"},
        "text_config": {
            "model_type": "deepseek_v41_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "compress_ratios": [0, 4, 0],
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "candidate_topk_blocks": 2,
            "candidate_block_size": 4,
            "rope_theta": 10000,
            "rope_scaling": {"rope_type": "yarn", "factor": 16},
        },
        "vision_config": {"num_hidden_layers": 2, "hidden_size": 12},
    }
    # Execute the actual native V4.1 mapping and RoPE compatibility helpers.
    # Only the unavailable Transformers serializer is a CPU protocol seam;
    # this does not claim a pinned-image HF integration run.
    native_mode = {"v5_rope": False}

    class HFConfig:
        def __init__(self, **kwargs):
            vars(self).update(kwargs)

        def to_dict(self):
            value = copy.deepcopy(vars(self))
            if native_mode["v5_rope"]:
                rope = value.pop("rope_scaling")
                rope["rope_theta"] = value.pop("rope_theta")
                value["rope_parameters"] = rope
            return value

    transformers = ModuleType("transformers")
    transformers.DeepseekV3Config = HFConfig
    transformers.PretrainedConfig = HFConfig
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    def source_module(name, relative):
        path = Path(__file__).resolve().parents[3] / relative
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    native_adapter = source_module(
        "sglang.srt.configs.deepseek_v41",
        "python/sglang/srt/configs/deepseek_v41.py",
    )
    native_patches = source_module(
        "sglang.srt.utils.hf_transformers_patches",
        "python/sglang/srt/utils/hf_transformers_patches.py",
    )

    def native_config():
        reference = native_adapter.DeepseekV41Config(**copy.deepcopy(authoritative))
        native_patches.normalize_rope_scaling_compat(reference)
        value = reference.to_dict()
        value["transformers_version"] = "CPU protocol serializer provenance"
        return value

    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    native_config_path = checkpoint / "config.json"
    native_config_path.write_text(json.dumps(authoritative, indent=2), encoding="utf-8")
    model.model_path = str(checkpoint)
    model.hf_config = native_config()
    description = NS(
        model="deepseek-ai/DeepSeek-V4.1-Flash", backend="sglang", geometry=geometry
    )
    stack = tmp_path / "stack.json"
    stack.write_bytes(raw({"fixture": "CPU contract only"}))
    database = NS(slice_ref=NS(slice_dir=tmp_path / "slice", stack_path=stack))
    calls = {"open": [], "describe": [], "progress": [], "config": []}
    package = ModuleType("infercast")
    package.__path__ = []
    package.__revision__ = "b" * 40
    package.__version__ = "CPU fixture"
    sdk = ModuleType("infercast.sdk")
    sdk.__path__ = []
    models = ModuleType("infercast.sdk.models")
    models.__path__ = []
    umd = ModuleType("infercast.sdk.models.umd")
    model_config = ModuleType("infercast.sdk.models.model_config")
    perf_database = ModuleType("infercast.sdk.perf_database")
    perf_database.__path__ = []
    manifest = ModuleType("infercast.sdk.perf_database.slice_manifest")
    proposal_owner = ModuleType("infercast.sdk.perf_database.dsv41_dspark_proposal")
    proposal_owner.MODEL_REVISION = "a" * 40

    def read_artifact(*args, **kwargs):
        pytest.fail("initial prefill must not read a draft graph artifact")

    manifest.read_verified_slice_artifact = read_artifact

    def open_fidb(*args, **kwargs):
        calls["open"].append((args, kwargs))
        return database

    def describe(*args, **kwargs):
        calls["describe"].append((args, kwargs))
        return description

    def load_config(*args, **kwargs):
        calls["config"].append((args, kwargs))
        return copy.deepcopy(authoritative)

    sdk.PerfDatabase = NS(open_fidb=open_fidb)
    umd.build_model_description = describe
    model_config.load_model_config = load_config
    for module in (
        package,
        sdk,
        models,
        umd,
        model_config,
        perf_database,
        manifest,
        proposal_owner,
    ):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    class Progress:
        topology = (2, 1, 1)

        def __init__(self, path, **kwargs):
            calls["progress"].append((path, kwargs))
            self.model_ref = "sha256:" + kwargs["expected_sha256"]

        def predict_progress(self, step):
            pytest.fail("missing worker coverage must fail before progress sampling")

    monkeypatch.setattr(dspark_progress, "EmpiricalDSparkProgress", Progress)

    def asset(name, content):
        path = tmp_path / name
        path.write_bytes(content)
        return {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}

    kwargs = {
        "model_id": description.model,
        "model_revision": "a" * 40,
        "provider_revision": package.__revision__,
        "system": "mi355x",
        "systems_root": str(tmp_path),
        "runtime_profile": asset("runtime.json", raw({"source": "CPU fixture"})),
        "progress_artifact": asset(
            "progress.json", raw({"fixture": "CPU reader seam"})
        ),
        "proposal_profile_id": "c" * 64,
    }
    config = SchedulerConfig(tp_size=2, backend_version="0.5.22")

    def build(**changes):
        return InferCastDSparkTimePredictor(
            model, NS(name="mi355x"), config, **{**kwargs, **changes}
        )

    return NS(
        build=build,
        kwargs=kwargs,
        model=model,
        config=config,
        sdk=package,
        geometry=geometry,
        description=description,
        database=database,
        calls=calls,
        asset=asset,
        progress=Progress,
        authoritative=authoritative,
        native_config_path=native_config_path,
        native_config=native_config,
        native_mode=native_mode,
    )


def test_factory_assembles_real_runtime_and_exposes_incomplete_costs(factory):
    from sglang_simulator.time_predictor.dspark_umd import UMDDSparkCostModel

    predictor = factory.build()
    assert predictor.supports_dspark
    runtime = predictor.dspark_runtime
    assert isinstance(runtime, DSparkNativeRuntime)
    assert isinstance(runtime.session, DSparkSession)
    assert isinstance(runtime.session.cost_model, UMDDSparkCostModel)
    assert isinstance(runtime.session.progress_model, factory.progress)
    assert set(runtime.session.cost_model.preparers) == {"proposal"}
    original = Path(factory.kwargs["runtime_profile"]["path"]).read_bytes()
    assert runtime.runtime_profile_json.encode() == original
    assert factory.calls["progress"][0][1]["runtime_profile_json"].encode() == original
    assert factory.calls["open"] == [
        (
            ("mi355x", "sglang", "0.5.22"),
            {
                "systems_root": factory.kwargs["systems_root"],
                "database_mode": "SILICON",
            },
        )
    ]
    assert factory.calls["describe"][0][1] == {
        "backend": "sglang",
        "revision": "a" * 40,
        "allow_download": False,
    }
    assert factory.calls["config"] == [
        ((factory.kwargs["model_id"],), {"allow_download": False, "revision": "a" * 40})
    ]
    before = predictor.get_metrics()
    metrics = before["infercast_dspark"]
    assert metrics["source_resolution"] == "proposal_capture_only"
    assert "target_graph_key" in metrics["unresolved_source_facts"]
    assert (
        metrics["native_model_config"]["sha256"]
        == hashlib.sha256(factory.native_config_path.read_bytes()).hexdigest()
    )
    assert metrics["phase_preparers"]["decode"]["available"] == ["proposal"]
    assert metrics["phase_preparers"]["prefill"]["available"] == []
    assert metrics["phase_preparers"]["prefill"]["missing"] == [
        "worker_preamble",
        "target_prefill",
        "draft_kv_inject",
        "worker_tail",
    ]
    assert (
        metrics["progress_model_ref"]
        == "sha256:" + factory.kwargs["progress_artifact"]["sha256"]
    )
    assert (
        metrics["runtime_profile"]["sha256"]
        == factory.kwargs["runtime_profile"]["sha256"]
    )
    assert not any(
        metrics[name]
        for name in ("complete_worker_costs", "timing_qualified", "accuracy_admitted")
    )
    predictor.reset_metrics()
    assert predictor.get_metrics() == before
    before["infercast_dspark"]["slice"]["path"] = "tampered returned copy"
    assert (
        predictor.get_metrics()["infercast_dspark"]["slice"]["path"]
        != "tampered returned copy"
    )


def test_first_native_prefill_fails_complete_cost_gate_before_progress_or_result(
    factory, monkeypatch
):
    from sglang_simulator.simulation.sglang import dspark_runtime

    predictor = factory.build()
    runtime = predictor.dspark_runtime
    monkeypatch.setattr(
        dspark_runtime,
        "sampling_snapshot",
        lambda _: '{"ignore_eos":true,"is_normalized":true,"max_new_tokens":10}',
    )
    request = NS(
        rid="a",
        output_ids=[],
        grammar=None,
        sampling_params=object(),
        inflight_middle_chunks=0,
    )
    batch = NS(
        device="cpu",
        forward_mode=NS(name="EXTEND"),
        reqs=[request],
        prefix_lens=[0],
        extend_lens=[8],
        seq_lens_cpu=[8],
    )
    published = []
    with pytest.raises(
        PredictorError, match="full worker costs are incomplete"
    ) as failure:
        runtime.forward_batch_generation(
            worker=NS(query_token_num=5, verify_num_draft_tokens=6),
            batch=batch,
            on_publish=published.append,
        )
    assert failure.value.details["missing_worker_phases"] == [
        "worker_preamble",
        "target_prefill",
        "draft_kv_inject",
        "worker_tail",
    ]
    assert not published and not runtime.session._history and runtime._forward_id == 0


def test_generic_forward_never_substitutes_an_ordinary_model_cost(factory):
    with pytest.raises(PredictorError, match="attached native DSPark"):
        factory.build().predict_infer_time(object())


@pytest.mark.parametrize(
    "field", ["phase_preparers", "phase_costs", "callback", "database_mode", "typo"]
)
def test_unknown_fields_fail_before_sdk_reader_seams(factory, field):
    with pytest.raises(PredictorError, match="unknown DSPark predictor"):
        factory.build(**{field: "unsupported"})
    assert not factory.calls["open"] and not factory.calls["describe"]


@pytest.mark.parametrize(
    "changes",
    [
        {"model_id": None},
        {"model_id": "deepseek-ai/DeepSeek-R1"},
        {"model_revision": "main"},
        {"model_revision": "A" * 40},
        {"provider_revision": None},
        {"provider_revision": "B" * 40},
        {"system": None},
        {"system": "another GPU"},
        {"systems_root": None},
        {"proposal_profile_id": "short"},
        {"output_token_id": True},
        {"output_token_id": -1},
        {"output_token_id": 32},
    ],
)
def test_malformed_constructor_identity(factory, changes):
    with pytest.raises(PredictorError):
        factory.build(**changes)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tp_size", 4),
        ("tp_size", True),
        ("dp_size", 2),
        ("pp_size", 2),
        ("ep_size", 2),
        ("cp_size", 2),
        ("backend_name", "other"),
        ("backend_version", None),
    ],
)
def test_unsupported_scheduler_identity(factory, field, value):
    setattr(factory.config, field, value)
    with pytest.raises(PredictorError):
        factory.build()
    assert not factory.calls["open"]


def test_wrong_installed_provider_revision(factory):
    factory.sdk.__revision__ = "d" * 40
    with pytest.raises(PredictorError, match="installed InferCast revision"):
        factory.build()
    assert not factory.calls["open"]


def test_model_revision_must_match_the_sdk_owned_source_cohort(factory):
    with pytest.raises(PredictorError, match="SDK-owned DSPark source cohort"):
        factory.build(model_revision="d" * 40)
    assert not factory.calls["describe"] and not factory.calls["open"]


@pytest.mark.parametrize(
    "key",
    [
        "n_routed_experts",
        "num_experts_per_tok",
        "moe_intermediate_size",
        "candidate_topk_blocks",
        "candidate_block_size",
    ],
)
@pytest.mark.parametrize("owner", ["checkpoint", "native_hf"])
def test_full_config_binding_rejects_work_overrides_before_database(
    factory, key, owner
):
    if owner == "checkpoint":
        changed = copy.deepcopy(factory.authoritative)
        changed["text_config"][key] += 1
        factory.native_config_path.write_bytes(raw(changed))
    else:
        factory.model.hf_config[key] += 1
    with pytest.raises(PredictorError, match="config differs"):
        factory.build()
    assert not factory.calls["open"] and not factory.calls["progress"]


def test_full_config_binding_includes_wrapper_and_vision_content(factory):
    changed = {**factory.authoritative, "vision_config": {"hidden_size": 123}}
    factory.native_config_path.write_bytes(raw(changed))
    with pytest.raises(PredictorError, match="checkpoint config differs"):
        factory.build()
    assert not factory.calls["open"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_type", "another_family"),
        ("architectures", ["AnotherForCausalLM"]),
        ("bos_token_id", 3),
        ("eos_token_id", 3),
        ("pad_token_id", 3),
        ("image_token_id", 3),
        ("vision_n_layers", 0),
        ("vision_dim", 99),
    ],
)
def test_native_outer_and_vision_overrides_cannot_hide_behind_source_file(
    factory, field, value
):
    factory.model.hf_config[field] = value
    with pytest.raises(PredictorError, match="structural config differs"):
        factory.build()
    assert not factory.calls["describe"]
    assert not factory.calls["open"] and not factory.calls["progress"]


def test_native_source_mapping_binds_root_vision_and_rope_compatibility(factory):
    native = factory.model.hf_config
    assert native["model_type"] == "deepseek_v41"
    assert native["architectures"] == ["DeepseekV4ForCausalLM"]
    assert native["vision_n_layers"] == 2 and native["vision_dim"] == 12
    assert native["rope_scaling"]["type"] == "yarn"
    assert "type" not in factory.authoritative["text_config"]["rope_scaling"]
    assert (
        native["transformers_version"] != factory.authoritative["transformers_version"]
    )
    factory.build()


@pytest.mark.parametrize("v5_rope", [False, True])
def test_native_rope_representation_comes_from_source_owned_serializer(
    factory, v5_rope
):
    factory.native_mode["v5_rope"] = v5_rope
    factory.model.hf_config = NS(to_dict=factory.native_config)
    factory.build()


@pytest.mark.parametrize("v5_rope", [False, True])
@pytest.mark.parametrize(
    "fault", ["factor", "legacy_type", "theta", "missing", "extra_alias", "extra_value"]
)
def test_native_rope_mutations_are_not_repaired_or_ignored(factory, v5_rope, fault):
    factory.native_mode["v5_rope"] = v5_rope
    native = factory.native_config()
    key = "rope_parameters" if v5_rope else "rope_scaling"
    if fault == "factor":
        native[key]["factor"] += 1
    elif fault == "legacy_type":
        native[key]["type"] = "linear"
    elif fault == "theta":
        if v5_rope:
            native[key]["rope_theta"] += 1
        else:
            native["rope_theta"] += 1
    elif fault == "missing":
        native.pop(key)
    elif fault == "extra_alias":
        native["rope_scaling" if v5_rope else "rope_parameters"] = {}
    else:
        native[key]["unrepresented_control"] = True
    factory.model.hf_config = native
    with pytest.raises(PredictorError, match="RoPE config differs"):
        factory.build()
    assert not factory.calls["describe"]
    assert not factory.calls["open"] and not factory.calls["progress"]


@pytest.mark.parametrize("owner", ["quantization_config", "expert_dtype"])
def test_native_expert_dtype_override_cannot_hide_behind_unchanged_file(factory, owner):
    if owner == "quantization_config":
        factory.model.hf_config[owner]["expert_dtype"] = "fp8"
    else:
        factory.model.hf_config[owner] = "fp8"
    with pytest.raises(PredictorError, match="dtype differs|structural config differs"):
        factory.build()
    assert not factory.calls["open"] and not factory.calls["progress"]


def test_native_hf_to_dict_allows_extra_defaults_and_documented_dtype_form(factory):
    native = factory.model.hf_config
    native["return_dict"] = True
    native["torch_dtype"] = "torch.bfloat16"
    native.pop("dtype")
    factory.model.hf_config = NS(to_dict=lambda: copy.deepcopy(native))
    predictor = factory.build()
    assert predictor.get_metrics()["infercast_dspark"]["native_model_config"]["sha256"]


@pytest.mark.parametrize("fault", ["missing_key", "dtype", "not_dict", "nonfinite"])
def test_native_hf_config_is_not_repaired_with_source_defaults(factory, fault):
    if fault == "missing_key":
        factory.model.hf_config.pop("n_routed_experts")
    elif fault == "dtype":
        factory.model.hf_config["dtype"] = "float16"
    elif fault == "not_dict":
        factory.model.hf_config = NS(to_dict=list)
    else:
        factory.model.hf_config["moe_intermediate_size"] = float("nan")
    with pytest.raises(PredictorError):
        factory.build()
    assert not factory.calls["open"] and not factory.calls["progress"]


@pytest.mark.parametrize(
    "content", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b"[]", b"\xff"]
)
def test_local_checkpoint_config_must_be_strict_json(factory, content):
    factory.native_config_path.write_bytes(content)
    with pytest.raises(PredictorError):
        factory.build()
    assert not factory.calls["open"] and not factory.calls["progress"]


@pytest.mark.parametrize(
    "fault", ["missing_path", "not_local", "missing_config", "oversized"]
)
def test_local_checkpoint_config_is_required_and_bounded(factory, fault):
    if fault == "missing_path":
        factory.model.model_path = None
    elif fault == "not_local":
        factory.model.model_path = "deepseek-ai/DeepSeek-V4.1-Flash"
    elif fault == "missing_config":
        factory.native_config_path.unlink()
    else:
        with factory.native_config_path.open("wb") as stream:
            stream.truncate(64 * 1024**2 + 1)
    with pytest.raises(PredictorError):
        factory.build()
    assert not factory.calls["open"] and not factory.calls["progress"]


def test_default_model_root_preserves_regular_file_binding_metadata(factory):
    reference = factory.build().get_metrics()["infercast_dspark"]["native_model_config"]
    source = factory.native_config_path
    assert reference["path"] == str(source)
    assert reference["resolved_path"] == str(source.resolve())
    assert reference["root"] == str(source.parent.resolve())
    assert reference["size_bytes"] == source.stat().st_size
    assert reference["target_stat"]["st_ino"] == source.stat().st_ino
    assert reference["source_stat"] == reference["target_stat"]
    assert reference["stable_binding"] is True and reference["link_target"] is None


@pytest.mark.parametrize("value", ["", " ", False, 1, [], {}])
def test_invalid_model_config_root_fails_before_reader_seams(factory, value):
    with pytest.raises(PredictorError, match="model_config_root"):
        factory.build(model_config_root=value)
    assert not factory.calls["config"] and not factory.calls["open"]


@pytest.mark.parametrize("fault", ["missing", "file", "outside"])
def test_model_config_root_must_contain_existing_model_directory(
    factory, tmp_path, fault
):
    root = tmp_path / fault
    if fault == "file":
        root.write_bytes(b"not a directory")
    elif fault == "outside":
        root.mkdir()
    with pytest.raises(PredictorError):
        factory.build(model_config_root=str(root))
    assert not factory.calls["open"] and not factory.calls["progress"]


@pytest.mark.parametrize("explicit_root", [False, True])
def test_internal_config_link_is_supported_without_guessing_cache_layout(
    factory, monkeypatch, explicit_root
):
    source = factory.native_config_path
    target = source.with_name("source-config.json")
    target.write_bytes(source.read_bytes())
    mock_config_link(monkeypatch, source, target)
    kwargs = {"model_config_root": str(source.parent)} if explicit_root else {}
    reference = factory.build(**kwargs).get_metrics()["infercast_dspark"][
        "native_model_config"
    ]
    assert reference["path"] == str(source)
    assert reference["resolved_path"] == str(target)
    assert reference["link_target"] == "source-config.json"
    assert reference["stable_binding"] is True


@pytest.fixture
def hf_cache_link(factory, tmp_path, monkeypatch):
    cache = tmp_path / "hf-cache"
    snapshot = cache / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    blobs = cache / "blobs"
    blobs.mkdir()
    content = factory.native_config_path.read_bytes()
    target = blobs / "verified-hash"
    target.write_bytes(content)
    source = snapshot / "config.json"
    source.write_bytes(content)
    factory.model.model_path = str(snapshot)
    factory.native_config_path = source
    state = mock_config_link(monkeypatch, source, target)
    return NS(root=cache, source=source, target=target, state=state)


def test_hf_cache_link_needs_explicit_containing_root(factory, hf_cache_link):
    with pytest.raises(PredictorError, match="escapes model_config_root"):
        factory.build()
    assert not factory.calls["open"] and not factory.calls["progress"]
    reference = factory.build(model_config_root=str(hf_cache_link.root)).get_metrics()[
        "infercast_dspark"
    ]["native_model_config"]
    assert reference["path"] == str(hf_cache_link.source)
    assert reference["resolved_path"] == str(hf_cache_link.target)
    assert reference["root"] == str(hf_cache_link.root)
    assert Path(reference["link_target"]).as_posix() == "../../blobs/verified-hash"
    assert (
        reference["sha256"]
        == hashlib.sha256(hf_cache_link.target.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("fault", ["escape", "dangling", "directory", "oversized"])
def test_hf_config_link_target_must_be_contained_bounded_and_regular(
    factory, hf_cache_link, tmp_path, fault
):
    if fault == "escape":
        target = tmp_path / "outside.json"
        target.write_bytes(hf_cache_link.target.read_bytes())
        hf_cache_link.state["target"] = target
    elif fault == "dangling":
        hf_cache_link.target.unlink()
    elif fault == "directory":
        hf_cache_link.target.unlink()
        hf_cache_link.target.mkdir()
    else:
        with hf_cache_link.target.open("wb") as stream:
            stream.truncate(64 * 1024**2 + 1)
    with pytest.raises(PredictorError):
        factory.build(model_config_root=str(hf_cache_link.root))
    assert not factory.calls["open"] and not factory.calls["progress"]


@pytest.mark.parametrize("fault", ["retarget", "replace", "modify"])
def test_config_link_and_target_must_remain_stable_through_native_binding(
    factory, hf_cache_link, fault
):
    target = hf_cache_link.target
    replacement = target.with_name("replacement")
    replacement.write_bytes(target.read_bytes())
    native = copy.deepcopy(factory.model.hf_config)

    def native_config():
        if fault == "retarget":
            hf_cache_link.state["target"] = replacement
        elif fault == "replace":
            replacement.replace(target)
        else:
            target.write_bytes(target.read_bytes() + b"\n")
        return native

    factory.model.hf_config = NS(to_dict=native_config)
    with pytest.raises(PredictorError, match="changed during native binding"):
        factory.build(model_config_root=str(hf_cache_link.root))
    assert not factory.calls["open"] and not factory.calls["progress"]


@pytest.mark.parametrize("fault", ["opened_another_file", "changed_during_read"])
def test_open_descriptor_identity_and_read_stability_are_checked(
    factory, monkeypatch, fault
):
    original_fstat = os.fstat
    calls = []

    def fstat(descriptor):
        value = original_fstat(descriptor)
        calls.append(descriptor)
        fields = {
            name: getattr(value, name)
            for name in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        }
        if fault == "opened_another_file" and len(calls) == 1:
            fields["st_ino"] += 1
        elif fault == "changed_during_read" and len(calls) == 2:
            fields["st_size"] += 1
        return NS(**fields)

    monkeypatch.setattr(os, "fstat", fstat)
    with pytest.raises(
        PredictorError, match="changed before read|changed during bounded"
    ):
        factory.build()
    assert not factory.calls["open"] and not factory.calls["progress"]


@pytest.mark.parametrize("layout", ["internal", "hf_cache"])
def test_real_model_config_symlink_layout(factory, tmp_path, layout):
    source = factory.native_config_path
    target = (
        source.with_name("config-source.json")
        if layout == "internal"
        else tmp_path / "blobs" / "config-source"
    )
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(source.read_bytes())
    source.unlink()
    try:
        source.symlink_to(os.path.relpath(target, source.parent))
    except OSError:
        pytest.skip("symlink creation unavailable")
    root = None if layout == "internal" else str(tmp_path)
    reference = factory.build(model_config_root=root).get_metrics()["infercast_dspark"][
        "native_model_config"
    ]
    assert reference["resolved_path"] == str(target.resolve())
    assert reference["path"] == str(source) and reference["stable_binding"]


@pytest.mark.parametrize(
    "field",
    [
        "hidden_size",
        "num_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "qk_rope_head_dim",
        "index_head_dim",
        "sliding_window",
        "compression_ratios_source",
    ],
)
def test_model_geometry_mismatch_fails_before_database_or_progress(factory, field):
    factory.geometry[field] = (0, 0, 0) if field == "compression_ratios_source" else 999
    with pytest.raises(PredictorError, match="SDK geometry differs"):
        factory.build()
    assert not factory.calls["open"] and not factory.calls["progress"]


def test_progress_topology_must_match_modeled_topology(factory):
    factory.progress.topology = (4, 1, 1)
    with pytest.raises(PredictorError, match="progress originals"):
        factory.build()


@pytest.mark.parametrize("name", ["runtime_profile", "progress_artifact"])
@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "digest",
        "bad_digest",
        "directory",
        "missing_file",
        "oversized",
    ],
)
def test_strict_bounded_asset_descriptors(factory, name, fault, tmp_path):
    descriptor = dict(factory.kwargs[name])
    if fault == "missing":
        descriptor.pop("sha256")
    elif fault == "extra":
        descriptor["extra"] = True
    elif fault == "digest":
        descriptor["sha256"] = "0" * 64
    elif fault == "bad_digest":
        descriptor["sha256"] = "A" * 64
    elif fault == "directory":
        descriptor["path"] = str(tmp_path)
    elif fault == "missing_file":
        descriptor["path"] = str(tmp_path / "missing")
    else:
        path = tmp_path / "too-large"
        with path.open("wb") as stream:
            stream.truncate(64 * 1024**2 + 1)
        descriptor["path"] = str(path)
    with pytest.raises(PredictorError):
        factory.build(**{name: descriptor})
    assert not factory.calls["open"]


def test_symlink_asset_is_rejected(factory, tmp_path):
    link = tmp_path / "linked-runtime.json"
    try:
        link.symlink_to(factory.kwargs["runtime_profile"]["path"])
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(PredictorError, match="symlink"):
        factory.build(
            runtime_profile={**factory.kwargs["runtime_profile"], "path": str(link)}
        )


@pytest.mark.parametrize(
    "content",
    [
        b"{}",
        b"[]",
        b"null",
        b'{"a":1,"a":2}',
        b'{"a":NaN}',
        b'{"a":Infinity}',
        b'{"z":1,"a":2}',
        b'{ "a":1}',
        b'{"a":1}\n',
        b"\xff",
    ],
)
def test_runtime_json_must_be_canonical_nonempty_finite_and_unique(factory, content):
    with pytest.raises(PredictorError):
        factory.build(runtime_profile=factory.asset("invalid-runtime.json", content))
    assert not factory.calls["open"]


def test_profile_optional_proposal_is_visible_as_absent(factory):
    predictor = factory.build(proposal_profile_id=None)
    assert (
        predictor.get_metrics()["infercast_dspark"]["phase_preparers"]["decode"][
            "available"
        ]
        == []
    )


def test_plain_factory_import_has_no_sdk_or_device_dependency():
    src = Path(__file__).resolve().parents[1] / "src"
    script = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'infercast', 'torch', 'sglang', 'aiter', 'aiconfigurator'}:
            raise AssertionError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, Block())
from sglang_simulator.time_predictor.infercast_dspark import InferCastDSparkTimePredictor
assert InferCastDSparkTimePredictor.supports_dspark
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": str(src)},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
