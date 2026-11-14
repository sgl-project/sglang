"""Initialize the bounded native ADP1 route through the public SimOpus SDK.

Costs, source projection and FIT remain SDK-owned. Progress is an explicit
simulator policy; neither a successful initialization nor a functional test
admits timing accuracy. A fresh run starts with a cold modeled draft buffer.
"""

import copy
import json
import re

from sglang_simulator.time_predictor.base import InferTimePredictor, PredictorError
from sglang_simulator.time_predictor.native_checkpoint import (
    _asset,
    _bind_local_model_config,
    _geometry,
    _require,
)
from sglang_simulator.utils.evidence import metrics_cpu_detail_enabled


class SimOpusDSparkTimePredictor(InferTimePredictor):
    supports_dspark = True

    def __init__(
        self,
        model,
        hw,
        config,
        *,
        model_id,
        model_revision,
        provider_revision,
        system,
        systems_root,
        native_segment_profile_id,
        native_capture_profile,
        progress_kind,
        progress_artifact=None,
        given_acceptance=None,
        model_seed=None,
        database_mode="SILICON",
        native_source_binding=None,
        native_analytic_model=None,
        native_source_work_policy=None,
        native_cost_evidence_path=None,
        output_token_id=1,
        model_config_root=None,
    ):
        super().__init__(model, hw, config)
        _require(
            model_id == "deepseek-ai/DeepSeek-V4.1-Flash",
            "native route requires V4.1 Flash",
            code="unsupported_model",
        )
        for name, value in (
            ("model_revision", model_revision),
            ("provider_revision", provider_revision),
        ):
            _require(
                type(value) is str and re.fullmatch("[0-9a-f]{40}", value),
                name + " must be a full lowercase commit",
            )
        _require(
            type(config.tp_size) is int and config.tp_size in (2, 4),
            "native route requires TP2/TP4",
            code="unsupported_topology",
        )
        _require(
            all(
                type(getattr(config, k)) is int and getattr(config, k) == 1
                for k in ("dp_size", "pp_size", "ep_size", "cp_size")
            ),
            "native route requires ADP1/PP1/EP1/CP1",
            code="unsupported_topology",
        )
        _require(
            config.backend_name == "sglang"
            and type(config.backend_version) is str
            and bool(config.backend_version),
            "native route requires an explicit SGLang version",
        )
        _require(
            type(system) is str
            and system == hw.name
            and type(systems_root) is str
            and bool(systems_root),
            "native hardware and slice must be explicit",
        )
        _require(
            database_mode in ("SILICON", "HYBRID")
            and (database_mode == "HYBRID") == (native_analytic_model is not None),
            "HYBRID requires an explicit SDK analytic asset",
        )
        _require(
            progress_kind in ("empirical", "source_golden351", "given_mean_acceptance"),
            "native progress policy must be explicit",
        )
        _require(
            (progress_kind == "empirical") == (progress_artifact is not None),
            "only empirical progress takes progress_artifact",
        )
        _require(
            (progress_kind == "given_mean_acceptance")
            == (given_acceptance is not None),
            "given acceptance must be selected explicitly",
        )
        _require(
            (model_seed is None)
            if progress_kind == "empirical"
            else (type(model_seed) is int and model_seed >= 0),
            "progress seed differs from selected policy",
        )
        _require(
            type(output_token_id) is int
            and type(model.vocab_size) is int
            and 0 <= output_token_id < model.vocab_size,
            "synthetic output token must be in the vocabulary",
        )

        from sglang_simulator.simulation.sglang.dspark import (
            DSparkSession,
            DSparkTopology,
        )
        from sglang_simulator.simulation.sglang.dspark_runtime import (
            DSparkNativeRuntime,
        )
        from sglang_simulator.time_predictor.simopus_native import (
            SimOpusNativeCostAdapter,
        )
        from sglang_simulator.time_predictor.simopus_sdk import load_sdk
        from simopus.sdk.models.model_config import load_model_config
        from simopus.sdk.models.umd import build_model_description

        package, sdk = load_sdk()
        _require(
            package.__revision__ == provider_revision,
            "installed SimOpus revision differs",
            code="incompatible_runtime",
        )
        try:
            authoritative = load_model_config(
                model_id, allow_download=False, revision=model_revision
            )
            checkpoint = _bind_local_model_config(
                model, authoritative, model_config_root
            )
            description = build_model_description(
                model_id,
                backend="sglang",
                revision=model_revision,
                allow_download=False,
            )
            _geometry(model, description, model_id)
            database = sdk.PerfDatabase.open_fidb(
                system,
                "sglang",
                config.backend_version,
                systems_root=systems_root,
                database_mode=database_mode,
            )
            _, capture_raw, capture_sha = _asset(
                native_capture_profile, "native_capture_profile"
            )
            capture = sdk.DecodeExecutionProfile.from_dict(json.loads(capture_raw))
            assets = sdk.load_native_region_assets(
                database,
                execution_profile=capture,
                native_segment_profile_id=native_segment_profile_id,
                source_binding=native_source_binding,
                analytic_model=native_analytic_model,
                source_work_policy=native_source_work_policy,
            )
            _require(
                assets.source_policy.tp_size == config.tp_size,
                "native source TP differs",
                code="unsupported_topology",
            )
            policy = assets.source_policy
            if progress_kind == "empirical":
                from sglang_simulator.time_predictor.dspark_progress import (
                    EmpiricalDSparkProgress,
                )

                _require(
                    config.tp_size == 2 and policy.golden_law is None,
                    "empirical progress requires the natural TP2 source",
                )
                progress_path, _, progress_sha = _asset(
                    progress_artifact, "progress_artifact"
                )
                progress = EmpiricalDSparkProgress(
                    progress_path,
                    expected_sha256=progress_sha,
                    runtime_profile_json=assets.runtime_profile_json,
                )
            elif progress_kind == "source_golden351":
                from sglang_simulator.time_predictor.dspark_golden_progress import (
                    SourceGoldenDSparkProgress,
                )

                progress = SourceGoldenDSparkProgress(policy, model_seed=model_seed)
            else:
                from sglang_simulator.time_predictor.dspark_given_mean_progress import (
                    GivenMeanDSparkProgress,
                )

                _, raw, _ = _asset(given_acceptance, "given_acceptance")
                progress = GivenMeanDSparkProgress(
                    policy, model_seed=model_seed, given=json.loads(raw)
                )
            _require(
                progress.topology == (config.tp_size, 1, 1),
                "progress source topology differs",
                code="unsupported_topology",
            )
            prepared = sdk.build_umd_static_model(
                database,
                model_id,
                backend="sglang",
                revision=model_revision,
                allow_download=False,
                native_region_assets=assets,
            )
            if metrics_cpu_detail_enabled():
                costs = SimOpusNativeCostAdapter(
                    prepared, assets, cost_evidence_path=native_cost_evidence_path
                )
            else:
                # Reuse this public-factory pair. Asset loading can have an
                # exclusive initialization receipt and must happen only once.
                costs = SimOpusNativeCostAdapter.from_prepared_sdk(
                    prepared,
                    assets,
                    cost_evidence_path=native_cost_evidence_path,
                )
                _require(
                    costs.assets.to_dict() == assets.to_dict(),
                    "numeric cost factory assets differ from progress source assets",
                )
            self.dspark_runtime = DSparkNativeRuntime(
                session=DSparkSession(costs, progress),
                topology=DSparkTopology(config.tp_size, 1, 1),
                runtime_profile_json=assets.runtime_profile_json,
                output_token_id=output_token_id,
                reference_events=True,
                initial_draft_buffer_capacity=0,
            )
        except PredictorError:
            raise
        except (OSError, ValueError, TypeError, KeyError, NotImplementedError) as error:
            raise PredictorError(
                getattr(error, "code", "provider_initialization_failed"),
                "native SimOpus initialization refused",
            ) from error
        self._identity = dict(
            model_id=model_id,
            model_revision=model_revision,
            provider_revision=provider_revision,
            checkpoint=checkpoint,
            framework_version=config.backend_version,
            system=system,
            native_assets=assets.to_dict(),
            capture_sha256=capture_sha,
            source_binding=copy.deepcopy(native_source_binding),
            progress_kind=progress_kind,
            progress_model_ref=progress.model_ref,
            initial_draft_buffer="cold",
            accuracy_admitted=False,
        )

    def predict_infer_time(self, batch):
        raise PredictorError(
            "unsupported_execution_profile",
            "native timing requires its worker stage result",
        )

    def get_metrics(self):
        runtime = self.dspark_runtime
        return {
            "simopus": {
                **copy.deepcopy(self._identity),
                "contract": "simopus-native-staged-latency",
                "evaluated_costs": runtime.session.cost_model.get_cost_metrics(),
                "committed_steps": runtime.session._last_forward_id + 1,
                "token_progress_modeled_by": "simulator_explicit_policy",
                "timing_qualified": False,
            }
        }

    def reset_metrics(self):
        # Keep causal session/history across profile boundaries; never reroll progress.
        pass
