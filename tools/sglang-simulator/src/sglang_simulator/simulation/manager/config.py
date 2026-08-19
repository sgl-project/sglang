import json
from pathlib import Path
from typing import Optional

from sglang_simulator.simulation.manager.env import Envs
from sglang_simulator.simulation.types import PlatformConfig, SchedulerConfig
from sglang_simulator.simulation.utils import (
    calc_kv_cache_cell_elems,
    calc_kv_cache_per_layer_elems,
)
from sglang_simulator.spec import AcceleratorInfo, DataType, ModelInfo
from sglang_simulator.time_predictor import (
    AIConfiguratorTimePredictor,
    InferTimePredictor,
)
from sglang_simulator.utils import get_logger

logger = get_logger()


class ConfigManager:
    """Centralized configuration manager with caching."""

    _model_info: Optional[ModelInfo] = None
    _platform_config: Optional[PlatformConfig] = None
    _scheduler_config: Optional[SchedulerConfig] = None
    _raw_config: Optional[dict] = None

    @classmethod
    def _get_raw_config(cls) -> dict:
        if cls._raw_config is None:
            with open(Envs.config_path()) as f:
                cls._raw_config = json.load(f)
        return cls._raw_config

    @classmethod
    def resolve_config_relative_path(cls, path: str | None) -> str | None:
        """Resolve predictor assets without depending on the process cwd."""
        if not path or Path(path).is_absolute():
            return path

        cwd_candidate = Path(path)
        if cwd_candidate.exists():
            return str(cwd_candidate.resolve())

        config_path = Path(Envs.config_path()).resolve()
        for parent in config_path.parents:
            candidate = parent / path
            if candidate.exists():
                return str(candidate)

        # Keep the original value so predictor-specific errors remain clear.
        return path

    @classmethod
    def reset_config_cache(cls):
        cls._raw_config = None
        cls._model_info = None
        cls._platform_config = None
        cls._scheduler_config = None

    @classmethod
    def set_model_info(cls, model: ModelInfo):
        cls._model_info = model

    @classmethod
    def get_model_info(cls) -> ModelInfo | None:
        return cls._model_info

    @classmethod
    def get_accelerator_info(cls) -> AcceleratorInfo:
        config = cls._get_raw_config()
        platform_config = config.get("platform", {})
        acc_info = platform_config.get("accelerator", {})
        hw = AcceleratorInfo.find_by_hw_name(acc_info.get("name"))
        if hw is None:
            logger.debug(
                f"Failed to initialize device info with {acc_info.get('name')}. All available devices are: {AcceleratorInfo.list_all_hws().keys()}"
            )
            hw = AcceleratorInfo(
                name=acc_info.get("name"),
                vendor=acc_info.get("vendor"),
                hbm_bandwidth_gb=acc_info.get("hbm_bandwidth_gb"),
                hbm_capacity_gb=acc_info.get("hbm_capacity_gb"),
                inter_node_bandwidth_gb=acc_info.get("inter_node_bandwidth_gb"),
                intra_node_bandwidth_gb=acc_info.get("intra_node_bandwidth_gb"),
                tflops=acc_info.get("tflops"),
            )
        else:
            logger.info(f"Device info initialized: {hw}")
        return hw

    @classmethod
    def get_platform_config(cls) -> PlatformConfig:
        if cls._platform_config is None:
            hw = cls.get_accelerator_info()
            config = cls._get_raw_config()
            platform_config = config.get("platform", {})
            cls._platform_config = PlatformConfig(
                device=hw,
                disk_read_bandwidth_gb=platform_config.get("disk_read_bandwidth_gb"),
                disk_write_bandwidth_gb=platform_config.get("disk_write_bandwidth_gb"),
                memory_read_bandwidth_gb=platform_config.get(
                    "memory_read_bandwidth_gb"
                ),
                memory_write_bandwidth_gb=platform_config.get(
                    "memory_write_bandwidth_gb"
                ),
                num_device_per_node=platform_config.get("num_device_per_node"),
            )

            logger.info(
                f"Platform configuration initialized successfully. {cls._platform_config}"
            )

        return cls._platform_config

    @classmethod
    def set_scheduler_config(cls, config: SchedulerConfig):
        # The configuration from the external config file has higher priority.
        external_config = cls._get_raw_config().get("scheduler", {})
        for field_name in [
            "tp_size",
            "dp_size",
            "ep_size",
            "pp_size",
            "cp_size",
            "cp_style",
            "backend_name",
            "backend_version",
            "kv_bytes_per_token_per_gpu",
            "hicache_ratio",
            "moe_quant_mode_override",
            "fmha_quant_mode_override",
            "comm_quant_mode_override",
        ]:
            field_value = external_config.get(field_name)
            if field_value is not None:
                setattr(config, field_name, field_value)

        for field_name in ["data_type", "kv_cache_data_type"]:
            field_value = external_config.get(field_name)
            if field_value is not None:
                setattr(config, field_name, DataType(field_value))

        cls._scheduler_config = config

    @classmethod
    def get_kv_cache_bytes(cls) -> int:
        model = cls._model_info
        scheduler_config = cls._scheduler_config
        return (
            calc_kv_cache_cell_elems(
                model, scheduler_config.tp_size, scheduler_config.pp_size
            )
            * scheduler_config.kv_cache_data_type.bytes
        )

    @classmethod
    def get_kv_cache_bytes_per_layer(cls) -> int:
        model = cls._model_info
        scheduler_config = cls._scheduler_config
        return (
            calc_kv_cache_per_layer_elems(
                model, scheduler_config.tp_size, scheduler_config.pp_size
            )
            * scheduler_config.kv_cache_data_type.bytes
        )

    @classmethod
    def get_scheduler_config(cls):
        return cls._scheduler_config

    @classmethod
    def _parse_server_args(cls, server_args: dict, backend: str) -> SchedulerConfig:
        if backend == "sglang":
            return SchedulerConfig(
                tp_size=server_args.get("tp_size", 1),
                ep_size=server_args.get("ep_size", 1),
                dp_size=server_args.get("dp_size", 1),
                pp_size=server_args.get("pp_size", 1),
                cp_size=server_args.get("attn_cp_size", 1),
                cp_style=server_args.get("cp_style", "none"),
                mem_fraction_static=server_args.get("mem_fraction_static"),
                backend_name="sglang",
            )
        else:
            raise RuntimeError(f"Unsupported backend[{backend}] server args parser.")

    @classmethod
    def get_inference_time_predictor(
        cls, model: ModelInfo, hw: AcceleratorInfo, sched_config: SchedulerConfig
    ) -> InferTimePredictor:
        config = cls._get_raw_config()
        predictor_config = config.get("predictor", {})
        if predictor_config.get("name") == "aiconfigurator":
            database_mode = predictor_config.get("database_mode", "SILICON")
            prefill_scale_factor = predictor_config.get("prefill_scale_factor", 1)
            decode_scale_factor = predictor_config.get("decode_scale_factor", 1)
            prefill_min_latency = predictor_config.get("prefill_min_latency", 0)
            workload_distribution = predictor_config.get(
                "workload_distribution", "balanced"
            )
            enable_oom_check = predictor_config.get("enable_oom_check", False)

            return AIConfiguratorTimePredictor(
                model,
                hw=hw,
                config=sched_config,
                database_path=cls.resolve_config_relative_path(
                    predictor_config.get("database_path")
                ),
                database_mode=database_mode,
                prefill_scale_factor=prefill_scale_factor,
                decode_scale_factor=decode_scale_factor,
                prefill_min_latency=prefill_min_latency,
                workload_distribution=workload_distribution,
                enable_oom_check=enable_oom_check,
            )
        elif predictor_config.get("name") == "ml":
            from sglang_simulator.time_predictor.ml import MLTimePredictor

            return MLTimePredictor(
                model,
                hw=hw,
                config=sched_config,
                database_path=cls.resolve_config_relative_path(
                    predictor_config.get("database_path")
                ),
                latency_scale=predictor_config.get("latency_scale", 1.0),
            )
        elif predictor_config.get("name") == "replay":
            from sglang_simulator.time_predictor.replay import ReplayTimePredictor

            return ReplayTimePredictor(
                model,
                hw=hw,
                config=sched_config,
                database_path=cls.resolve_config_relative_path(
                    predictor_config.get("database_path")
                ),
                miss_fallback_seconds=predictor_config.get(
                    "miss_fallback_seconds", 0.0
                ),
                miss_strategy=predictor_config.get("miss_strategy", "zero"),
                miss_knn_k=predictor_config.get("miss_knn_k", 3),
            )
        else:
            raise ValueError(
                f"Unknown predictor name: {predictor_config.get('name')}. "
                f"Supported: aiconfigurator, ml, replay"
            )
