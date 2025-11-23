from __future__ import annotations

import argparse
import logging
import sys
from importlib import util as importlib_util
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List

from sglang.tune.types import AutoTuneConfig, ComponentResult, ComponentTuner

LOGGER = logging.getLogger(__name__)


class FusedMoeTritonTuner(ComponentTuner):
    name = "fused_moe_triton"

    def __init__(self) -> None:
        self._module: ModuleType | None = None

    def is_available(self) -> bool:
        return self._dependency_check()

    def run(self, config: AutoTuneConfig) -> ComponentResult:
        module = self._load_module()
        component_dir = config.results_dir / self.name
        component_dir.mkdir(parents=True, exist_ok=True)

        output_files: List[Path] = []
        metadata: Dict[str, Any] = {"script_path": str(self._script_path())}

        original_get_filename = getattr(module, "get_filename")
        original_save_configs = getattr(module, "save_configs")

        def patched_get_filename(*args: Any, **kwargs: Any) -> str:
            filename = original_get_filename(*args, **kwargs)
            target_path = component_dir / filename
            target_path.parent.mkdir(parents=True, exist_ok=True)
            return str(target_path)

        def patched_save_configs(configs: Dict[int, Any], filename: str) -> None:
            output_files.append(Path(filename))
            original_save_configs(configs, filename)

        setattr(module, "get_filename", patched_get_filename)
        setattr(module, "save_configs", patched_save_configs)

        try:
            args_namespace = argparse.Namespace(
                model=config.model_path,
                tp_size=config.tensor_parallel_size,
                ep_size=config.expert_parallel_size,
                dtype=config.dtype,
                per_channel_quant=config.per_channel_quant,
                seed=config.seed,
                batch_size=config.batch_size,
                tune=True,
                disable_shared_experts_fusion=config.disable_shared_experts_fusion,
            )
            module.main(args_namespace)
            status = "success"
            details = None
        except Exception as exc:  # pylint: disable=broad-except
            status = "failed"
            details = str(exc)
            metadata["exception_type"] = type(exc).__name__
            LOGGER.exception("Triton fused MoE tuning failed.")
        finally:
            setattr(module, "get_filename", original_get_filename)
            setattr(module, "save_configs", original_save_configs)

        return ComponentResult(
            name=self.name,
            status=status,
            details=details,
            output_files=tuple(output_files),
            metadata=metadata,
        )

    @staticmethod
    def _dependency_check() -> bool:
        try:
            import ray  # noqa: F401
            import torch  # noqa: F401
            import triton  # noqa: F401
        except ModuleNotFoundError as error:
            LOGGER.debug("Dependency check failed for fused_moe_triton: %s", error)
            return False
        return True

    def _load_module(self) -> ModuleType:
        if self._module is not None:
            return self._module

        script_path = self._script_path()
        spec = importlib_util.spec_from_file_location(
            "sglang_benchmark_fused_moe_triton", str(script_path)
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to create module spec for {script_path}.")
        module = importlib_util.module_from_spec(spec)
        sys.modules.setdefault(spec.name, module)
        spec.loader.exec_module(module)
        self._module = module
        return module

    @staticmethod
    def _script_path() -> Path:
        return (
            Path(__file__).resolve().parents[2]
            / "benchmark"
            / "kernels"
            / "fused_moe_triton"
            / "tuning_fused_moe_triton.py"
        )
