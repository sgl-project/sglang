import json
from pathlib import Path

try:
    from aisimulate.sdk import models
    from aisimulate.sdk.common import (
        CommQuantMode,
        DatabaseMode,
        FMHAQuantMode,
        GEMMQuantMode,
        KVCacheQuantMode,
        MoEQuantMode,
    )
    from aisimulate.sdk.config import ModelConfig
    from aisimulate_core.sdk import ForwardPassPerfModelConfig, RustForwardPassPerfModel
except ModuleNotFoundError as exc:
    if exc.name not in {"aisimulate", "aisimulate.sdk", "aisimulate_core"}:
        raise
    raise ModuleNotFoundError(
        "Install the AISimulate SDK with: "
        "pip install -e 'tools/sglang-simulator[aisimulate]'. "
        "For Replay or ML without automatic KV-cache memory estimation, "
        "set --max-total-tokens explicitly."
    ) from exc

from sglang_simulator.simulation.sglang.batch_snapshot import batch_to_fpm_metrics
from sglang_simulator.simulation.types import (
    SchedulerConfig,
)
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.data_type import DataType
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
)
from sglang_simulator.utils import get_logger

# Map the common data types to AISimulate data types.
MAP_DTYPE_TO_GEMMQuantMode = {
    DataType.FP16: GEMMQuantMode.bfloat16,
    DataType.BF16: GEMMQuantMode.bfloat16,
    DataType.FP8: GEMMQuantMode.fp8_block,
    DataType.INT8: GEMMQuantMode.int8_wo,
    DataType.FP4: GEMMQuantMode.nvfp4,
    DataType.INT4: GEMMQuantMode.int4_wo,
    DataType.FP16_TENSOR: GEMMQuantMode.bfloat16,
    DataType.BF16_TENSOR: GEMMQuantMode.bfloat16,
    DataType.FP8_TENSOR: GEMMQuantMode.fp8,
    DataType.INT8_TENSOR: GEMMQuantMode.int8_wo,
    DataType.FP4_TENSOR: GEMMQuantMode.nvfp4,
    DataType.INT4_TENSOR: GEMMQuantMode.int4_wo,
}

MAP_DTYPE_TO_KVCacheQuantMode = {
    DataType.FP16: KVCacheQuantMode.bfloat16,
    DataType.BF16: KVCacheQuantMode.bfloat16,
    DataType.FP8: KVCacheQuantMode.fp8,
    DataType.INT8: KVCacheQuantMode.int8,
}

MAP_DTYPE_TO_FMHAQuantMode = {
    DataType.FP16: FMHAQuantMode.bfloat16,
    DataType.BF16: FMHAQuantMode.bfloat16,
    DataType.FP8: FMHAQuantMode.fp8,
}

MAP_DTYPE_TO_MoEQuantMode = {
    DataType.FP16: MoEQuantMode.bfloat16,
    DataType.BF16: MoEQuantMode.bfloat16,
    DataType.FP8: MoEQuantMode.fp8_block,
    DataType.INT8: MoEQuantMode.fp8,
    DataType.FP4: MoEQuantMode.nvfp4,
    DataType.INT4: MoEQuantMode.int4_wo,
}

MAP_DTYPE_TO_CommQuantMode = {
    DataType.FP16: CommQuantMode.half,
    DataType.BF16: CommQuantMode.half,
    DataType.FP8: CommQuantMode.fp8,
    DataType.INT8: CommQuantMode.int8,
}


logger = get_logger("sgl_simulator")


def _resolve_comm_quant_mode(sched_config: SchedulerConfig) -> CommQuantMode:
    if sched_config.comm_quant_mode_override:
        return getattr(CommQuantMode, sched_config.comm_quant_mode_override)

    if sched_config.data_type is None:
        return CommQuantMode.half

    try:
        return MAP_DTYPE_TO_CommQuantMode[sched_config.data_type]
    except KeyError:
        raise ValueError(
            "AISimulate has no communication quantization mapping for "
            f"model data type {sched_config.data_type.value}. Set "
            "comm_quant_mode_override explicitly to half, int8, or fp8."
        ) from None


def get_quant_modes(config: SchedulerConfig, *, native: bool = False) -> dict:
    """One deployment mapping for memory sizing and forward-pass prediction.

    The memory-sizing caller retains the permissive historical defaults; the
    predictor refuses the FP16->BF16 / INT8-MoE->FP8 approximations instead of
    silently estimating the wrong deployment.
    """
    dtype, kv = config.data_type, config.kv_cache_data_type
    if native:
        kv = kv or dtype
        if dtype is None or kv is None:
            raise ValueError("AISimulate needs resolved model and KV dtypes")
        if dtype.name.startswith("FP16") or kv.name.startswith("FP16"):
            raise ValueError(
                "The AIS SDK has no faithful FP16 quantization mode; this "
                "bridge refuses instead of silently approximating FP16 as "
                "BF16 the way the old static path did"
            )
        if dtype.name.startswith("INT8") and not config.moe_quant_mode_override:
            raise ValueError(
                "This bridge does not infer an INT8 deployment's MoE recipe: "
                "declare moe_quant_mode_override explicitly. Declaring it "
                "does not guarantee an estimate — KV/FMHA mappings and the "
                "systems profile must still match the deployment"
            )
        # Tensor-level cache/attention representations use the same SDK modes.
        kv = DataType[kv.name.removesuffix("_TENSOR")]
    specs = (
        ("gemm", dtype, MAP_DTYPE_TO_GEMMQuantMode, GEMMQuantMode, None),
        (
            "moe",
            dtype,
            MAP_DTYPE_TO_MoEQuantMode,
            MoEQuantMode,
            config.moe_quant_mode_override,
        ),
        ("kvcache", kv, MAP_DTYPE_TO_KVCacheQuantMode, KVCacheQuantMode, None),
        (
            "fmha",
            kv,
            MAP_DTYPE_TO_FMHAQuantMode,
            FMHAQuantMode,
            config.fmha_quant_mode_override,
        ),
    )
    result = {}
    for name, value, mapping, enum, override in specs:
        if native and not override and value not in mapping:
            raise ValueError(f"No faithful {name} mapping for {value}")
        result[name + "_quant_mode"] = (
            getattr(enum, override) if override else mapping.get(value, enum.bfloat16)
        )
    result["comm_quant_mode"] = _resolve_comm_quant_mode(config)
    return result


def get_perf_model(
    sched_config: SchedulerConfig,
    model: ModelInfo,
    workload_distribution: str = "balanced",
) -> models.BaseModel:
    """Op-level model used for memory/weight estimation, not latency."""
    model_config = ModelConfig(
        pp_size=sched_config.pp_size,
        tp_size=sched_config.attn_tp_size,
        moe_tp_size=sched_config.moe_tp_size,
        moe_ep_size=sched_config.moe_ep_size,
        attention_dp_size=sched_config.attn_dp_size,
        cp_size=sched_config.cp_size,
        cp_style=sched_config.cp_style,
        **get_quant_modes(sched_config),
        workload_distribution=workload_distribution,
    )

    logger.info(f"Model config for AISimulate: {model_config}")

    return models.get_model(
        model_path=model.model_path,
        model_config=model_config,
        backend_name=sched_config.backend_name,
    )


# Retain the static path's lower bound for applying the correction.
_IMBALANCE_MIN_RATIO = 0.4


def _prefill_imbalance_ratio(scheduled_requests: dict) -> float:
    """Actual/mean prefill attention workload ratio of a batch snapshot.

    Takes the metrics dict built by batch_to_fpm_metrics — all aggregates and
    aligned per-request arrays populated, decode rows ordinary (extend=1) —
    not arbitrary external calibration records. R uses unrounded means and is
    applied to the SDK's integer-shaped attention. Decode rows contribute
    2*p+1 each; remove their aggregate work from the aligned arrays."""
    n = scheduled_requests["num_prefill_requests"]
    if n <= 1:
        return 1.0
    total_new = scheduled_requests["sum_prefill_tokens"]
    if not total_new:
        return 1.0
    total_past = scheduled_requests["sum_prefill_kv_tokens"]
    nd = scheduled_requests["num_decode_requests"]
    total_decode_kv = scheduled_requests["sum_decode_kv_tokens"]
    extends = scheduled_requests["extend_lengths"]
    pasts = scheduled_requests["past_kv_lengths"]
    num = sum(e * (2 * p + e) for e, p in zip(extends, pasts))
    num -= 2 * total_decode_kv + nd
    return num / (total_new * (2 * total_past + total_new) / n)


class AISimulateTimePredictor(InferTimePredictor):
    """Latency predictor on the SDK's ForwardPass estimator for every mode."""

    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        *,
        database_mode: DatabaseMode | str = DatabaseMode.SILICON,
        estimation_mode: str = "op_level",
        estimator_config: dict | None = None,
        systems_paths: list[str] | None = None,
        tune_observations: str | None = None,
        worker_type: str = "aggregated",
        prefill_scale_factor: float = 1,
        decode_scale_factor: float = 1,
        prefill_min_latency: float = 0,
        enable_oom_check: bool = False,
        database_path: str | None = None,
    ):
        super().__init__(model, hw, config)

        self.prefill_scale_factor = prefill_scale_factor
        self.decode_scale_factor = decode_scale_factor
        # Millisecond floor on the (scaled) prefill estimate, matching the
        # historical static path's units.
        self.prefill_min_latency = prefill_min_latency
        if enable_oom_check:
            raise ValueError(
                "enable_oom_check=True has no ForwardPass equivalent: this "
                "adapter neither runs the old InferenceSession memory "
                "summaries nor propagates an OOM state. Device memory is "
                "sized separately via get_perf_model; pass False or omit it."
            )
        if database_path is not None:
            if systems_paths is not None:
                raise ValueError(
                    "Set either database_path (legacy single systems root, "
                    "mapped to systems_paths=[path]) or systems_paths, not both"
                )
            systems_paths = [database_path]

        if estimator_config is None:
            estimator_config = {}
        if not isinstance(estimator_config, dict):
            raise ValueError("estimator_config must be a JSON object or omitted")
        # The SDK enables correction implicitly; this predictor stays
        # uncorrected unless correction.enabled is set explicitly. Explicit
        # values — including null or a non-dict — are left for the SDK to
        # reject rather than masked as false.
        if "correction" not in estimator_config:
            estimator_config = {**estimator_config, "correction": {"enabled": False}}
        else:
            correction = estimator_config["correction"]
            if isinstance(correction, dict) and "enabled" not in correction:
                estimator_config = {
                    **estimator_config,
                    "correction": {**correction, "enabled": False},
                }
        if (
            config.pp_size != 1
            or config.dp_size != 1
            or config.cp_size != 1
            or config.cp_style != "none"
        ):
            raise ValueError(
                "The simulator's ForwardPass bridge is built from one "
                "scheduler snapshot and is only validated for PP=DP=CP=1; "
                "TP and EP are forwarded as deployment parameters. This is "
                "a bridge scope limit, not an AIS SDK capability limit."
            )
        quant = {k: v.name for k, v in get_quant_modes(config, native=True).items()}
        self._fpm = RustForwardPassPerfModel.best_available(
            ForwardPassPerfModelConfig(
                model=model.model_path,
                system=hw.name,
                backend=config.backend_name,
                backend_version=config.backend_version,
                tp=config.attn_tp_size,
                pp=1,
                attention_dp=1,
                moe_tp_size=config.moe_tp_size,
                moe_ep_size=config.moe_ep_size,
                worker_type=worker_type,
                estimation_mode=estimation_mode,
                database_mode=(
                    database_mode.name
                    if isinstance(database_mode, DatabaseMode)
                    else str(database_mode).upper()
                ),
                systems_paths=tuple(systems_paths or ()),
                estimator_config=estimator_config,
                fallback_policy="deny",
                **quant,
            )
        )
        # The resolved mode (auto picks here) and the validated correction
        # setting select between the two mutually exclusive calibration
        # paths in predict_infer_time: SDK correction enabled uses the SDK's
        # own result as-is, correction disabled keeps the local uneven-prefill
        # correction on the uncorrected op_level estimate.
        provenance = self._fpm.diagnostics()["provenance"]
        self._fpm_mode = provenance["selected_estimation_mode"]
        correction_config = (provenance["config"].get("estimator_config") or {}).get(
            "correction", {}
        )
        self._correction_enabled = correction_config.get("enabled") is True
        if tune_observations is not None:
            # Original observations are passed unchanged; the SDK validates
            # and trains on them itself.
            self._fpm.tune_with_fpms(json.loads(Path(tune_observations).read_text()))
        logger.info("AISimulate: %s", self._fpm.diagnostics())
        self.requires_batch_snapshot = True

    def get_metrics(self) -> dict:
        return self._fpm.diagnostics()

    def _context_attention_delta_ms(
        self, scheduled_requests: dict, ratio: float
    ) -> float:
        """Return the legacy delta at the SDK's pure or mixed attention shape.

        Pure prefill uses separate integer means. Mixed attention uses ceil(E/q)
        requests with q=max(E//n, 1). Only ordinary context_attention is adjusted."""
        n = int(scheduled_requests.get("num_prefill_requests", 0) or 0)
        total_new = int(scheduled_requests.get("sum_prefill_tokens", 0) or 0)
        total_past = int(scheduled_requests.get("sum_prefill_kv_tokens", 0) or 0)
        if n <= 0:
            return 0.0
        if int(scheduled_requests.get("num_decode_requests", 0) or 0):
            per_req_new = max(total_new // n, 1)
            attention_bs = max((total_new + per_req_new - 1) // per_req_new, 1)
            scale = max((per_req_new + total_new - 1) // max(total_new, 1), 1)
        else:
            per_req_new = total_new // n
            attention_bs, scale = n, 1
        per_req_past = total_past // n
        entries = self._fpm.static_phase_diagnostics(
            batch_size=attention_bs,
            context_length=per_req_new + per_req_past,
            prefix=per_req_past,
            prefill=True,
        )
        attention_ms = sum(
            entry.get("latency_ms") or 0.0
            for entry in entries
            if entry["name"] == "context_attention"
        )
        return (ratio - 1.0) * attention_ms / scale if attention_ms else 0.0

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        metrics = batch_to_fpm_metrics(batch)
        latency = self._fpm.estimate_forward_pass_time_ms(metrics)
        if latency is None:
            raise RuntimeError(
                "AISimulate has no ready estimate for this workload; provide matching calibration observations"
            )
        # SDK correction and local imbalance adjustment are mutually
        # exclusive, even when the SDK is untuned or its factor is 1.
        if self._fpm_mode == "op_level" and not self._correction_enabled:
            scheduled = metrics["scheduled_requests"]
            ratio = _prefill_imbalance_ratio(scheduled)
            if ratio != 1.0 and ratio >= _IMBALANCE_MIN_RATIO:
                latency += self._context_attention_delta_ms(scheduled, ratio)
        # Explicit user post-processing, in the historical order and units
        # (ms): factor first, then the prefill-side floor. This is not the
        # SDK's learned correction; it only runs when configured non-neutral.
        # Mixed steps keep the historical batch-level rule — the whole forward
        # is scaled as prefill; no per-phase split is attempted.
        if batch.is_decode():
            latency *= self.decode_scale_factor
        else:
            latency *= self.prefill_scale_factor
            if latency > 0:
                latency = max(latency, self.prefill_min_latency)
        return latency / 1000.0

    def predict_infer_latency_dict(self, batch: ScheduleBatch) -> dict:
        """Per-operator op-level debug evidence in milliseconds.

        Reuses the same ForwardPass estimator via static_phase_diagnostics:
        no second estimator, no InferenceSession. The values are the SDK's
        raw static evidence at its internal even shape (floored per-request
        averages — the shape the estimate itself uses), before learned
        correction and before the imbalance delta and post-processing.
        Pure extend/decode snapshots only; mixed steps have no honest
        per-op decomposition and are rejected.
        """
        if batch.forward_mode not in {"extend", "decode"}:
            raise ValueError(
                "Per-operator latency requires a pure extend or decode snapshot"
            )
        if batch.batch_size < 1:
            raise ValueError("Per-operator latency requires a non-empty snapshot")
        size = batch.batch_size
        total_new = sum(r.extend_length for r in batch.reqs)
        total_past = sum(r.past_kv_length for r in batch.reqs)
        if batch.forward_mode == "extend":
            per_req_past = total_past // size
            entries = self._fpm.static_phase_diagnostics(
                batch_size=size,
                context_length=total_new // size + per_req_past,
                prefix=per_req_past,
                prefill=True,
            )
        else:
            # The SDK evaluates one decode step at context_length + 1; the
            # snapshot's prior KV already excludes the current query token.
            entries = self._fpm.static_phase_diagnostics(
                batch_size=size,
                context_length=total_past // size,
                prefill=False,
            )
        # The SDK list can repeat an operator name (e.g. per layer); the
        # debug dict keeps one entry per name, summing its latency_ms.
        ops: dict = {}
        for entry in entries:
            ops[entry["name"]] = ops.get(entry["name"], 0.0) + entry["latency_ms"]
        return ops
