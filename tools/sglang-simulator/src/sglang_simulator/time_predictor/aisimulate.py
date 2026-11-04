import json
from pathlib import Path
from statistics import pvariance
from typing import Optional

import numpy as np

try:
    from aisimulate.sdk import models
    from aisimulate.sdk.backends.factory import get_backend
    from aisimulate.sdk.common import (
        CommQuantMode,
        DatabaseMode,
        FMHAQuantMode,
        GEMMQuantMode,
        KVCacheQuantMode,
        MoEQuantMode,
    )
    from aisimulate.sdk.config import ModelConfig, RuntimeConfig
    from aisimulate.sdk.inference_session import InferenceSession
    from aisimulate.sdk.perf_database import get_database, get_systems_paths
except ModuleNotFoundError as exc:
    if exc.name not in {"aisimulate", "aisimulate.sdk"}:
        raise
    raise ModuleNotFoundError(
        "Install the AISimulate SDK with: "
        "pip install -e 'tools/sglang-simulator[aisimulate]'"
    ) from exc

from sglang_simulator.simulation.types import (
    SchedulerConfig,
)
from sglang_simulator.spec.accelerator import AcceleratorInfo
from sglang_simulator.spec.data_type import DataType
from sglang_simulator.spec.model import ModelInfo
from sglang_simulator.time_predictor.base import (
    InferTimePredictor,
    ScheduleBatch,
    ScheduleRequest,
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
    """One deployment mapping for memory, legacy prediction and native FPM.

    The legacy path retains its historical defaults. Native prediction refuses
    the legacy FP16->BF16 / INT8-MoE->FP8 approximations.
    """
    dtype, kv = config.data_type, config.kv_cache_data_type
    if native:
        kv = kv or dtype
        if dtype is None or kv is None:
            raise ValueError("Native AISimulate needs resolved model and KV dtypes")
        if dtype.name.startswith("FP16") or kv.name.startswith("FP16"):
            raise ValueError("AISimulate has no faithful FP16 native quantization mode")
        if dtype.name.startswith("INT8") and not config.moe_quant_mode_override:
            raise ValueError("Declare moe_quant_mode_override for an INT8 deployment")
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


class AISimulateTimePredictor(InferTimePredictor):
    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        database_path: Optional[str] = None,
        database_mode: DatabaseMode | str = DatabaseMode.SILICON,
        prefill_scale_factor: float = 1,
        decode_scale_factor: float = 1,
        prefill_min_latency: float = 0,
        workload_distribution: str = "balanced",
        enable_oom_check: bool = False,
        *,
        estimation_mode: str | None = None,
        estimator_config: dict | None = None,
        systems_paths: list[str] | None = None,
        tune_observations: str | None = None,
        worker_type: str = "aggregated",
    ):
        super().__init__(model, hw, config)

        self._fpm = None
        self.requires_batch_snapshot = estimation_mode is not None
        if self.requires_batch_snapshot:
            from aisimulate_core.sdk import (
                ForwardPassPerfModelConfig,
                RustForwardPassPerfModel,
            )

            if (
                (prefill_scale_factor, decode_scale_factor, prefill_min_latency)
                != (1, 1, 0)
                or enable_oom_check
                or workload_distribution != "balanced"
                or database_path is not None
            ):
                raise ValueError(
                    "Native modes use estimator_config and systems_paths; legacy scaling, OOM and database_path options are not supported"
                )
            if (
                config.pp_size != 1
                or config.dp_size != 1
                or config.cp_size != 1
                or config.cp_style != "none"
            ):
                raise ValueError(
                    "Native AISimulate currently supports TP/EP with PP=DP=CP=1"
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
                    database_mode=database_mode.name
                    if isinstance(database_mode, DatabaseMode)
                    else database_mode,
                    systems_paths=tuple(systems_paths or ()),
                    estimator_config=estimator_config or {},
                    fallback_policy="deny",
                    **quant,
                )
            )
            if tune_observations is not None:
                self._fpm.tune_with_fpms(
                    json.loads(Path(tune_observations).read_text())
                )
            logger.info("AISimulate: %s", self._fpm.diagnostics())
            return
        if (
            estimator_config is not None
            or systems_paths is not None
            or tune_observations is not None
            or worker_type != "aggregated"
        ):
            raise ValueError("Set estimation_mode when using native AISimulate options")

        self.prefill_scale_factor = prefill_scale_factor
        self.decode_scale_factor = decode_scale_factor
        self.prefill_min_latency = prefill_min_latency
        if isinstance(database_mode, str):
            database_mode = self._get_database_mode(database_mode)

        database = get_database(
            system=hw.name,
            backend=config.backend_name,
            version=config.backend_version,
            systems_paths=(
                [database_path] if database_path is not None else get_systems_paths()
            ),
        )

        if database is None:
            raise ValueError("Failed to initialize the database.")

        database.set_default_database_mode(database_mode)
        logger.info(f"AISimulate database mode: {database_mode}")

        self._session = InferenceSession(
            model=get_perf_model(config, model, workload_distribution),
            backend=get_backend(self.config.backend_name),
            database=database,
        )

        self.enable_oom_check = enable_oom_check
        self._is_oom = False

    def _get_database_mode(self, mode: str) -> DatabaseMode:
        return {
            "SILICON": DatabaseMode.SILICON,
            "HYBRID": DatabaseMode.HYBRID,
            "EMPIRICAL": DatabaseMode.EMPIRICAL,
            "SOL": DatabaseMode.SOL,
            "SOL_FULL": DatabaseMode.SOL_FULL,
        }.get(mode.upper(), DatabaseMode.SILICON)

    def ctx_attn_flops_ratio_with_avg(self, reqs: list[ScheduleRequest]) -> float:
        if len(reqs) == 1:
            return 1.0
        mean_past = np.mean([req.past_kv_length for req in reqs])
        mean_input = np.mean([req.extend_length for req in reqs])
        avg_flops = (mean_past + mean_past + mean_input) * mean_input / 2 * len(reqs)

        actual_flops = 0
        for req in reqs:
            actual_flops += (
                (req.past_kv_length + req.past_kv_length + req.extend_length)
                * req.extend_length
                / 2
            )

        return actual_flops / avg_flops

    def predict_infer_latency_dict(self, batch: ScheduleBatch) -> dict:
        # A calibrated whole-forward estimate has no per-operator attribution.
        if self._fpm is not None:
            raise ValueError("Operator breakdown is only available on the legacy path")
        # Returns latency details for debugging operators.
        if batch.is_decode():
            # Decode: output sequence length (osl) = 2, input sequence length (isl) = mean(past_kv_length)
            isl = int(np.mean([req.past_kv_length for req in batch.reqs]))
            runtime_config = RuntimeConfig(batch_size=batch.batch_size, isl=isl, osl=2)
            if self.enable_oom_check:
                summary = self._session.run_static(runtime_config, mode="static_gen")
                latency_dict = summary.get_generation_latency_dict()
            else:
                # faster path
                results = self._session._backend._run_static_breakdown(
                    self._session._model,
                    self._session._database,
                    runtime_config,
                    mode="static_gen",
                )
                latency_dict = results[2]
        else:
            # Prefill: output sequence length (osl) = 1, input sequence length (isl) = mean(past_kv + input), prefix = mean(past_kv)
            mean_past = np.mean([req.past_kv_length for req in batch.reqs])
            mean_input = np.mean([req.extend_length for req in batch.reqs])
            isl = int(mean_past + mean_input)
            prefix = int(mean_past)
            runtime_config = RuntimeConfig(
                batch_size=batch.batch_size, isl=isl, prefix=prefix, osl=1
            )

            seq_imbalance_correction_scale = self.ctx_attn_flops_ratio_with_avg(
                batch.reqs
            )
            if seq_imbalance_correction_scale >= 0.4:
                runtime_config = RuntimeConfig(
                    batch_size=batch.batch_size,
                    isl=isl,
                    prefix=prefix,
                    osl=1,
                    seq_imbalance_correction_scale=seq_imbalance_correction_scale,
                )
            else:
                runtime_config = RuntimeConfig(
                    batch_size=batch.batch_size, isl=isl, prefix=prefix, osl=1
                )

            if self.enable_oom_check:
                summary = self._session.run_static(runtime_config, mode="static_ctx")
                latency_dict = summary.get_context_latency_dict()
            else:
                # faster path
                results = self._session._backend._run_static_breakdown(
                    self._session._model,
                    self._session._database,
                    runtime_config,
                    mode="static_ctx",
                )
                latency_dict = results[0]
        return latency_dict

    def get_metrics(self) -> dict:
        return self._fpm.diagnostics() if self._fpm is not None else {}

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        if self._fpm is not None:
            latency = self._fpm.estimate_forward_pass_time_ms(
                batch_to_fpm_metrics(batch)
            )
            if latency is None:
                raise RuntimeError(
                    "AISimulate has no ready estimate for this workload; provide matching calibration observations"
                )
            return latency / 1000.0
        latency_dict = self.predict_infer_latency_dict(batch)
        infer_time = sum(latency_dict.values())

        if self._is_oom:
            logger.warning("Out of memory detected during estimation.")
            infer_time = -infer_time
        if batch.is_decode():
            infer_time *= self.decode_scale_factor
        else:
            infer_time *= self.prefill_scale_factor

        if not batch.is_decode():
            infer_time = (
                max(infer_time, self.prefill_min_latency)
                if infer_time > 0
                else infer_time
            )

        return infer_time / 1e3


def batch_to_fpm_metrics(batch: ScheduleBatch) -> dict:
    """Translate scheduled work to native FPM features (prior KV, not total length)."""
    prefill, decode = [], []
    if batch.forward_mode not in {"extend", "decode", "mixed"}:
        raise ValueError("Native prediction requires an extend/decode/mixed snapshot")
    for req in batch.reqs:
        phase = (
            req.forward_mode if batch.forward_mode == "mixed" else batch.forward_mode
        )
        if phase not in {"extend", "decode"}:
            raise ValueError("Mixed prediction requires per-request scheduler phases")
        (prefill if phase == "extend" else decode).append(req)
    lengths = [r.extend_length for r in prefill]
    decode_kv = [r.past_kv_length for r in decode]
    return {
        "scheduled_requests": {
            "num_prefill_requests": len(prefill),
            "sum_prefill_tokens": sum(lengths),
            "sum_prefill_kv_tokens": sum(r.past_kv_length for r in prefill),
            "var_prefill_length": pvariance(lengths) if lengths else 0.0,
            "num_decode_requests": len(decode),
            "sum_decode_kv_tokens": sum(decode_kv),
            "var_decode_kv_tokens": pvariance(decode_kv) if decode_kv else 0.0,
        }
    }
