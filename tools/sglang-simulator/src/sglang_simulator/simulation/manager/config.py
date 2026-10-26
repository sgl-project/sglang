import json
import os
from dataclasses import replace
from pathlib import Path

from sglang_simulator.simulation.manager.env import Envs
from sglang_simulator.simulation.types import PlatformConfig, SchedulerConfig
from sglang_simulator.simulation.utils import (
    calc_kv_cache_cell_elems,
    calc_kv_cache_per_layer_elems,
)
from sglang_simulator.spec import AcceleratorInfo, DataType, ModelInfo
from sglang_simulator.time_predictor import InferTimePredictor
from sglang_simulator.utils import get_logger

logger = get_logger()


class ConfigManager:
    """Centralized configuration manager with caching."""

    _model_info: ModelInfo | None = None
    _platform_config: PlatformConfig | None = None
    _scheduler_config: SchedulerConfig | None = None
    _raw_config: dict | None = None
    _memory_budget_metrics: dict | None = None

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
    def resolve_infercast_path(cls, path: str | None) -> str | None:
        """Resolve the InferCast systems directory relative to its config file."""
        if not path:
            return path
        path = Path(os.path.expandvars(path)).expanduser()
        if path.is_absolute():
            return str(path)
        config_path = Path(Envs.config_path()).resolve()
        return str((config_path.parent / path).resolve())

    @classmethod
    def reset_config_cache(cls):
        cls._raw_config = None
        cls._model_info = None
        cls._platform_config = None
        cls._scheduler_config = None
        cls._memory_budget_metrics = None

    @classmethod
    def get_memory_budget_config(cls):
        return cls._get_raw_config().get("memory_budget")

    @classmethod
    def set_memory_budget_metrics(cls, metrics: dict | None):
        cls._memory_budget_metrics = metrics

    @classmethod
    def get_memory_budget_metrics(cls):
        return cls._memory_budget_metrics

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
        cls._apply_scheduler_overrides(config)
        cls._scheduler_config = config

    @classmethod
    def get_target_scheduler_config(cls, config: SchedulerConfig) -> SchedulerConfig:
        target = replace(config)
        cls._apply_scheduler_overrides(target)
        return target

    @classmethod
    def _apply_scheduler_overrides(cls, config: SchedulerConfig):
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
    def modeled_attention_dp_size(cls, native: dict):
        """Translate only the explicitly source-bound joint simulator domain.

        Native DP1 still means one TP2 worker, not two model replicas. The
        modeled field represents its two verified singleton attention groups.
        The factory independently validates the retained source topology.
        """
        ordinary = native.get("dp_size", 1)
        if cls._raw_config is None and not os.getenv("SGLANG_SIMULATOR_CONFIG_PATH"):
            return ordinary
        predictor = cls._get_raw_config().get("predictor", {})
        if (
            predictor.get("name") != "infercast_dspark"
            or predictor.get("native_joint_source_binding") is None
        ):
            return ordinary
        expected = {
            "tp_size": 2,
            "dp_size": 1,
            "attn_dp_size": 2,
            "pp_size": 1,
            "attn_cp_size": 1,
            "ep_size": 1,
            "nnodes": 1,
        }
        if (
            predictor.get("cost_mode", "SILICON") != "HYBRID"
            or predictor.get("progress_kind", "empirical") != "source_golden351"
            or any(
                type(native.get(key)) is not int or native[key] != value
                for key, value in expected.items()
            )
        ):
            raise ValueError(
                "joint DSPark requires native DP1/TP2/attention-DP2/PP1/CP1/EP1/one node"
            )
        return 2

    @classmethod
    def _parse_server_args(cls, server_args: dict, backend: str) -> SchedulerConfig:
        if backend == "sglang":
            return SchedulerConfig(
                tp_size=server_args.get("tp_size", 1),
                ep_size=server_args.get("ep_size", 1),
                dp_size=cls.modeled_attention_dp_size(server_args),
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
        cls,
        model: ModelInfo,
        hw: AcceleratorInfo,
        sched_config: SchedulerConfig,
        *,
        attention_dp_rank=None,
    ) -> InferTimePredictor:
        config = cls._get_raw_config()
        predictor_config = config.get("predictor", {})
        predictor_name = predictor_config.get("name")
        if predictor_name == "aiconfigurator":
            from sglang_simulator.time_predictor.aiconfigurator import (
                AIConfiguratorTimePredictor,
            )

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
        elif predictor_name == "ml":
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
        elif predictor_name == "replay":
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
        elif predictor_name == "infercast_dspark":
            from sglang_simulator.time_predictor.infercast_dspark import (
                InferCastDSparkTimePredictor,
            )

            supported_fields = {
                "name",
                "model_id",
                "model_revision",
                "provider_revision",
                "model_config_root",
                "systems_root",
                "runtime_profile",
                "progress_artifact",
                "progress_kind",
                "model_seed",
                "proposal_profile_id",
                "native_segment_profile_id",
                "native_capture_profile",
                "native_initial_draft_buffer",
                "native_source_work_policy",
                "cost_mode",
                "native_analytic_model",
                "native_cost_evidence_path",
                "native_source_binding",
                "native_joint_source_binding",
                "output_token_id",
            }
            unknown_fields = sorted(predictor_config.keys() - supported_fields)
            if unknown_fields:
                raise ValueError(
                    "unknown InferCast DSPark predictor configuration fields: "
                    + ", ".join(unknown_fields)
                )
            options = {
                key: value for key, value in predictor_config.items() if key != "name"
            }
            options["systems_root"] = cls.resolve_infercast_path(
                options.get("systems_root")
            )
            if (
                type(options.get("model_config_root")) is str
                and options["model_config_root"].strip()
            ):
                root = Path(
                    os.path.expandvars(options["model_config_root"])
                ).expanduser()
                if not root.is_absolute():
                    root = Path(Envs.config_path()).resolve().parent / root
                # Preserve the requested root's link identity. The factory
                # resolves and checks it across native model binding.
                options["model_config_root"] = str(root.absolute())
            for key in (
                "runtime_profile",
                "progress_artifact",
                "native_capture_profile",
                "native_initial_draft_buffer",
                "native_analytic_model",
            ):
                ref = options.get(key)
                if isinstance(ref, dict) and type(ref.get("path")) is str:
                    options[key] = {
                        **ref,
                        "path": cls.resolve_infercast_path(ref["path"]),
                    }
            if options.get("native_joint_source_binding") is not None:
                models = options.get("native_analytic_model")
                if type(models) is dict:
                    options["native_analytic_model"] = {
                        mode: {**ref, "path": cls.resolve_infercast_path(ref["path"])}
                        if type(ref) is dict and type(ref.get("path")) is str
                        else ref
                        for mode, ref in models.items()
                    }
            joint_ledger = (
                options.get("native_joint_source_binding") is not None
                and options.get("native_cost_evidence_path") is not None
            )
            if joint_ledger and (
                type(attention_dp_rank) is not int or attention_dp_rank not in (0, 1)
            ):
                raise ValueError(
                    "joint cost evidence requires attention-DP rank 0 or 1"
                )
            if type(options.get("native_cost_evidence_path")) is str:
                explicit_path = options["native_cost_evidence_path"]
                options["native_cost_evidence_path"] = cls.resolve_infercast_path(
                    explicit_path
                )
                if joint_ledger and explicit_path.strip():
                    path = Path(options["native_cost_evidence_path"])
                    # Leave directory inputs to the existing writer's rejection.
                    if not path.is_dir():
                        options["native_cost_evidence_path"] = str(
                            path.with_name(
                                f"{path.stem}.dp{attention_dp_rank}{path.suffix}"
                            )
                        )
            for name in ("native_source_binding", "native_joint_source_binding"):
                binding = options.get(name)
                if (
                    type(binding) is dict
                    and type(binding.get("root")) is str
                    and binding["root"].strip()
                ):
                    root = Path(os.path.expandvars(binding["root"])).expanduser()
                    if not root.is_absolute():
                        root = Path(Envs.config_path()).resolve().parent / root
                    # Keep link identity for the factory's safe original reader.
                    options[name] = {**binding, "root": str(root.absolute())}
            return InferCastDSparkTimePredictor(
                model,
                hw=hw,
                config=sched_config,
                system=config.get("platform", {}).get("accelerator", {}).get("name"),
                **options,
            )
        elif predictor_name == "infercast":
            from sglang_simulator.time_predictor.infercast import (
                InferCastTimePredictor,
            )

            obsolete_fields = sorted(
                {"contract_version", "reduction_policy"} & predictor_config.keys()
            )
            if obsolete_fields:
                raise ValueError(
                    "obsolete InferCast predictor configuration fields: "
                    + ", ".join(obsolete_fields)
                )

            supported_fields = {
                "name",
                "model_id",
                "systems_root",
                "database_mode",
                "attn_kernel_impl",
                "decode_attn_kernel_impl",
                "attn_dtype",
                "kv_cache_dtype",
                "model_revision",
                "provider_revision",
                "execution_profile",
                "decode_execution_profile",
            }
            unknown_fields = sorted(predictor_config.keys() - supported_fields)
            if unknown_fields:
                raise ValueError(
                    "unknown InferCast predictor configuration fields: "
                    + ", ".join(unknown_fields)
                )

            return InferCastTimePredictor(
                model,
                hw=hw,
                config=sched_config,
                model_id=predictor_config.get("model_id"),
                system=config.get("platform", {}).get("accelerator", {}).get("name"),
                systems_root=cls.resolve_infercast_path(
                    predictor_config.get("systems_root")
                ),
                database_mode=predictor_config.get("database_mode", "SILICON"),
                attn_kernel_impl=predictor_config.get("attn_kernel_impl"),
                decode_attn_kernel_impl=predictor_config.get("decode_attn_kernel_impl"),
                attn_dtype=predictor_config.get("attn_dtype"),
                kv_cache_dtype=predictor_config.get("kv_cache_dtype"),
                model_revision=predictor_config.get("model_revision"),
                provider_revision=predictor_config.get("provider_revision"),
                execution_profile=predictor_config.get("execution_profile"),
                decode_execution_profile=predictor_config.get(
                    "decode_execution_profile"
                ),
            )
        else:
            raise ValueError(
                f"Unknown predictor name: {predictor_name}. "
                f"Supported: aiconfigurator, ml, replay, infercast, infercast_dspark"
            )
