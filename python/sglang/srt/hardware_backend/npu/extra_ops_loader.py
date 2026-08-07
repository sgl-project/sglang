from __future__ import annotations

import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class OpLibSpec:
    """Configuration for a standalone operator library loaded into ``torch.ops``."""

    name: str  # human-readable id, used in logs/errors
    so_env: str  # env var that points to the standalone .so path
    namespace: str  # torch.ops.<namespace> the operators register into
    required_ops: tuple[str, ...]
    pre_load_imports: tuple[str, ...] = ()  # modules to import before loading


class TorchOpLoader:
    """Load a standalone .so into ``torch.ops`` and validate its operators."""

    def __init__(self, spec: OpLibSpec) -> None:
        self._spec = spec
        self._loaded_library: Optional[Path] = None

    def _missing_ops(self) -> list[str]:
        namespace = getattr(torch.ops, self._spec.namespace, None)
        if namespace is None:
            return list(self._spec.required_ops)
        return [op for op in self._spec.required_ops if not hasattr(namespace, op)]

    def registered(self) -> bool:
        """Return whether the required operators are already registered."""
        return not self._missing_ops()

    def _resolve_so_path(self) -> Path:
        explicit = os.environ.get(self._spec.so_env)
        if not explicit:
            raise RuntimeError(
                f"The {self._spec.name} operators are not registered. Set "
                f"{self._spec.so_env} to the standalone .so library path."
            )
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"{self._spec.so_env} points to a missing file: {path}")
        return path

    def _validate_python_abi(self, library_path: Path) -> None:
        abi_match = re.search(r"\.cpython-(\d+)-", library_path.name)
        current_abi = f"{sys.version_info.major}{sys.version_info.minor}"
        if abi_match is not None and abi_match.group(1) != current_abi:
            raise RuntimeError(
                f"{library_path} was built for CPython {abi_match.group(1)}, "
                f"but SGLang is running CPython {current_abi}. Rebuild the "
                "extension with the SGLang Python/Torch/torch-npu environment."
            )

    def initialize(self) -> Optional[Path]:
        """Register the operators before backend execution.

        Idempotent: returns ``None`` if the operators are already registered
        (e.g. by another package). Otherwise loads the standalone .so pointed
        to by ``so_env`` into ``torch.ops`` and validates the required operators.

        Returns the loaded library path when this call loaded it, else ``None``.
        """
        if self.registered():
            return None
        if self._loaded_library is not None:
            missing = self._missing_ops()
            raise RuntimeError(
                f"Loaded {self._loaded_library}, but required "
                f"{self._spec.namespace} operators are missing: {missing}."
            )

        for module in self._spec.pre_load_imports:
            __import__(module)  # noqa: F401  side-effect imports (e.g. torch_npu)

        library_path = self._resolve_so_path()
        self._validate_python_abi(library_path)
        try:
            torch.ops.load_library(str(library_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load the {self._spec.name} operator library "
                f"{library_path}. Ensure its dependent CANN/custom-op libraries "
                "are visible through LD_LIBRARY_PATH and the Ascend OPP setup."
            ) from exc

        missing = self._missing_ops()
        if missing:
            raise RuntimeError(
                f"Loaded {library_path}, but required "
                f"{self._spec.namespace} operators are missing: {missing}."
            )

        self._loaded_library = library_path
        logger.info("Registered %s operators from %s", self._spec.name, library_path)
        return library_path


def initialize_dspark_sparse_attn_ops() -> Optional[Path]:
    """Register the DSpark sparse-attention ops before backend execution."""
    spec = OpLibSpec(
        name="DSpark sparse-attention",
        so_env="SGLANG_DSPARK_EXTRA_OPS_SO",
        namespace="_C_ascend",
        required_ops=(
            "npu_sparse_attn_sharedkv_metadata",
            "npu_sparse_attn_sharedkv",
        ),
        pre_load_imports=("torch_npu",),
    )
    return TorchOpLoader(spec).initialize()
