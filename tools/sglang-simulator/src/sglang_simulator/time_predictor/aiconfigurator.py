from typing import Optional

import numpy as np
from aiconfigurator.sdk import models
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.common import (
    CommQuantMode,
    DatabaseMode,
    FMHAQuantMode,
    GEMMQuantMode,
    KVCacheQuantMode,
    MoEQuantMode,
)
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import InferenceSession
from aiconfigurator.sdk.perf_database import get_database, get_systems_paths
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

# Map the common data types to AIConfigurator data types.
MAP_DTYPE_TO_GEMMQuantMode = {
    DataType.FP16: GEMMQuantMode.float16,
    DataType.BF16: GEMMQuantMode.float16,
    DataType.FP8: GEMMQuantMode.fp8_block,
    DataType.INT8: GEMMQuantMode.int8_wo,
    DataType.FP4: GEMMQuantMode.nvfp4,
    DataType.INT4: GEMMQuantMode.int4_wo,
    DataType.FP16_TENSOR: GEMMQuantMode.float16,
    DataType.BF16_TENSOR: GEMMQuantMode.float16,
    DataType.FP8_TENSOR: GEMMQuantMode.fp8,
    DataType.INT8_TENSOR: GEMMQuantMode.int8_wo,
    DataType.FP4_TENSOR: GEMMQuantMode.nvfp4,
    DataType.INT4_TENSOR: GEMMQuantMode.int4_wo,
}

MAP_DTYPE_TO_KVCacheQuantMode = {
    DataType.FP16: KVCacheQuantMode.float16,
    DataType.BF16: KVCacheQuantMode.float16,
    DataType.FP8: KVCacheQuantMode.fp8,
    DataType.INT8: KVCacheQuantMode.int8,
}

MAP_DTYPE_TO_FMHAQuantMode = {
    DataType.FP16: FMHAQuantMode.float16,
    DataType.BF16: FMHAQuantMode.float16,
    DataType.FP8: FMHAQuantMode.fp8,
}

MAP_DTYPE_TO_MoEQuantMode = {
    DataType.FP16: MoEQuantMode.float16,
    DataType.BF16: MoEQuantMode.float16,
    DataType.FP8: MoEQuantMode.fp8,
    DataType.INT8: MoEQuantMode.fp8,
    DataType.FP4: MoEQuantMode.nvfp4,
    DataType.INT4: MoEQuantMode.int4_wo,
}

MAP_DTYPE_TO_CommQunatMode = {
    DataType.FP16: CommQuantMode.half,
    DataType.BF16: CommQuantMode.half,
    DataType.FP8: CommQuantMode.fp8,
    DataType.INT8: CommQuantMode.int8,
}


logger = get_logger("sgl_simulator")


def get_perf_model(sched_config: SchedulerConfig, model: ModelInfo) -> models.BaseModel:
    model_config = ModelConfig(
        pp_size=sched_config.pp_size,
        tp_size=sched_config.tp_size,
        moe_tp_size=sched_config.tp_size // sched_config.ep_size,
        moe_ep_size=sched_config.ep_size,
        attention_dp_size=sched_config.tp_size // sched_config.dp_size,
        gemm_quant_mode=MAP_DTYPE_TO_GEMMQuantMode.get(
            sched_config.data_type, GEMMQuantMode.float16
        ),
        moe_quant_mode=MAP_DTYPE_TO_MoEQuantMode.get(
            sched_config.data_type, MoEQuantMode.float16
        ),
        kvcache_quant_mode=MAP_DTYPE_TO_KVCacheQuantMode.get(
            sched_config.kv_cache_data_type, KVCacheQuantMode.float16
        ),
        fmha_quant_mode=MAP_DTYPE_TO_FMHAQuantMode.get(
            sched_config.kv_cache_data_type, FMHAQuantMode.float16
        ),
        comm_quant_mode=MAP_DTYPE_TO_CommQunatMode.get(
            sched_config.data_type, CommQuantMode.half
        ),
        workload_distribution="power_law_1.2",
    )

    return models.get_model(
        model_path=model.model_path,
        model_config=model_config,
        backend_name=sched_config.backend_name,
    )


class AIConfiguratorTimePredictor(InferTimePredictor):
    def __init__(
        self,
        model: ModelInfo,
        hw: AcceleratorInfo,
        config: SchedulerConfig,
        database_path: Optional[str] = None,
        database_mode: DatabaseMode | str = DatabaseMode.SILICON,
        prefill_scale_factor: float = 1,
        decode_scale_factor: float = 1,
    ):
        super().__init__(model, hw, config)

        self.prefill_scale_factor = prefill_scale_factor
        self.decode_scale_factor = decode_scale_factor

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

        # --- Replace the original function to support more flexible request input. --- #

        db_nearest_1d_point_helper = database._nearest_1d_point_helper

        def wrapped_nearest_1d_point_helper(
            x: int, values: list[int], inner_only: bool = False
        ):
            # Disable the inner_only by default
            return db_nearest_1d_point_helper(x, values, inner_only)

        database._nearest_1d_point_helper = wrapped_nearest_1d_point_helper

        # --- End --- #

        self._session = InferenceSession(
            model=get_perf_model(config, model),
            backend=get_backend(self.config.backend_name),
            database=database,
        )

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

    def predict_infer_time(self, batch: ScheduleBatch) -> float:
        infer_time = 0
        if batch.is_decode():
            # Decode: output sequence length (osl) = 2, input sequence length (isl) = mean(past_kv_length)
            isl = int(np.mean([req.past_kv_length for req in batch.reqs]))
            runtime_config = RuntimeConfig(batch_size=batch.batch_size, isl=isl, osl=2)
            summary = self._session.run_static(runtime_config, mode="static_gen")
            latency_dict = summary.get_generation_latency_dict()

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

            summary = self._session.run_static(runtime_config, mode="static_ctx")
            latency_dict = summary.get_context_latency_dict()
        infer_time = sum(latency_dict.values())
        if summary.check_oom():
            logger.warning("Out of memory detected during estimation.")
            infer_time = -infer_time
        if batch.is_decode():
            infer_time *= self.decode_scale_factor
        else:
            infer_time *= self.prefill_scale_factor
        return infer_time / 1e3
