"""Export runtime interfaces for platform-selected graph execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from sglang.srt.compilation.export_artifact import ExportArtifactSpec


class ExportRuntime(ABC):
    """Runtime backend for a captured ``torch.export.ExportedProgram``."""

    export_format = "torch"

    @abstractmethod
    def prepare_runtime(
        self,
        exported_program,
        artifact: ExportArtifactSpec,
        compile_config,
    ) -> Callable:
        """Convert/load artifacts and return the runtime callable."""


class TorchExportRuntime(ExportRuntime):
    """Default runtime that executes the exported PyTorch module directly."""

    export_format = "torch"

    def prepare_runtime(self, exported_program, artifact, compile_config):
        artifact.write_metadata()
        return exported_program.module()


__all__ = [
    "ExportRuntime",
    "TorchExportRuntime",
]
