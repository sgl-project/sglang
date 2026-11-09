"""Public SDK factory wiring; source/cost and progress controls remain separate."""

import hashlib
import json
from types import SimpleNamespace as NS

import pytest
from sglang_simulator.time_predictor.base import PredictorError
from sglang_simulator.time_predictor.simopus_dspark import SimOpusDSparkTimePredictor
from test_native_public_sdk import public_adapter as public_adapter


def inputs(tmp_path):
    return dict(
        model=NS(vocab_size=999),
        hw=NS(name="mi355x"),
        config=NS(
            tp_size=2,
            dp_size=1,
            pp_size=1,
            ep_size=1,
            cp_size=1,
            backend_name="sglang",
            backend_version="0.5.22",
        ),
        model_id="deepseek-ai/DeepSeek-V4.1-Flash",
        model_revision="a" * 40,
        provider_revision="b" * 40,
        system="mi355x",
        systems_root=str(tmp_path),
        native_segment_profile_id="c" * 64,
        native_capture_profile={},
        progress_kind="empirical",
        progress_artifact={},
    )


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"model_id": "other"}, "unsupported_model"),
        ({"provider_revision": "main"}, "provider_initialization_failed"),
        ({"model_revision": "main"}, "provider_initialization_failed"),
        ({"database_mode": "SOL"}, "provider_initialization_failed"),
        ({"database_mode": "HYBRID"}, "provider_initialization_failed"),
        ({"progress_kind": "implicit"}, "provider_initialization_failed"),
        ({"given_acceptance": {}}, "provider_initialization_failed"),
        ({"model_seed": 1}, "provider_initialization_failed"),
        ({"output_token_id": True}, "provider_initialization_failed"),
    ],
)
def test_invalid_factory_inputs_refuse_before_loading_assets(tmp_path, changes, code):
    args = inputs(tmp_path)
    args.update(changes)
    with pytest.raises(PredictorError) as error:
        SimOpusDSparkTimePredictor(**args)
    assert error.value.code == code


def test_factory_loads_real_native_assets_and_public_umd(
    public_adapter, tmp_path, monkeypatch
):
    import simopus
    import simopus.sdk as sdk
    from sglang_simulator.time_predictor import (
        dspark_progress,
        simopus_dspark,
    )

    adapter, database, options = public_adapter
    args = inputs(tmp_path)
    args["config"].tp_size = adapter.native_source_policy.tp_size
    args["provider_revision"] = simopus.__revision__
    args["model_revision"] = "main"
    # An exact pin is still required; the packaged model resource accepts its
    # declared checkpoint rather than arbitrary main for actual initialization.
    from simopus.sdk.perf_database.dsv41_dspark_proposal import MODEL_REVISION

    args["model_revision"] = MODEL_REVISION
    raw = json.dumps(options["execution_profile"].to_dict()).encode()
    capture = tmp_path / "capture.json"
    capture.write_bytes(raw)
    progress = tmp_path / "progress.json"
    progress.write_text("{}")
    args.update(
        native_capture_profile=dict(
            path=str(capture), sha256=hashlib.sha256(raw).hexdigest()
        ),
        native_segment_profile_id=options["native_segment_profile_id"],
        native_source_binding=options["source_binding"],
        progress_artifact=dict(
            path=str(progress), sha256=hashlib.sha256(b"{}").hexdigest()
        ),
    )
    monkeypatch.setattr(sdk.PerfDatabase, "open_fidb", lambda *a, **k: database)
    monkeypatch.setattr(
        simopus_dspark,
        "_bind_local_model_config",
        lambda *a, **k: {"scope": "checkpoint binding independently tested"},
    )
    monkeypatch.setattr(simopus_dspark, "_geometry", lambda *a: None)
    monkeypatch.setattr(
        dspark_progress,
        "EmpiricalDSparkProgress",
        lambda *a, **k: NS(
            topology=(2, 1, 1), model_ref="explicit CPU progress double"
        ),
    )
    if args["config"].tp_size == 4:
        with pytest.raises(PredictorError, match="natural TP2"):
            SimOpusDSparkTimePredictor(**args)
        return
    result = SimOpusDSparkTimePredictor(**args)
    assert (
        result.dspark_runtime.session.cost_model.assets.execution_profile_id
        == options["native_segment_profile_id"]
    )
    assert result.dspark_runtime._draft_buffer_capacity == 0
    assert result.get_metrics()["simopus"]["committed_steps"] == 0
    with pytest.raises(PredictorError, match="worker stage"):
        result.predict_infer_time(None)
    result.reset_metrics()
    assert result.dspark_runtime._forward_id == 0


def test_native_config_rejects_excluded_warm_and_joint_routes(monkeypatch):
    from sglang_simulator.simulation.manager.config import ConfigManager

    for field in (
        "native_initial_draft_buffer",
        "native_joint_source_binding",
        "proposal_profile_id",
        "cost_mode",
    ):
        monkeypatch.setattr(
            ConfigManager,
            "_raw_config",
            dict(
                predictor=dict(
                    name="simopus", native_segment_profile_id="a" * 64, **{field: {}}
                )
            ),
        )
        with pytest.raises(ValueError, match="unknown native SimOpus"):
            ConfigManager.get_inference_time_predictor(None, None, None)
