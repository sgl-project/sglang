"""Export runtime interfaces for platform-selected graph execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from sglang.srt.compilation.export_artifact import ExportArtifactSpec
from sglang.srt.compilation.export_context import DistributedExportContext


@dataclass(frozen=True)
class ExportRuntimeCapabilities:
    supports_cuda_graph_capture: bool = False
    supports_tp: bool = True
    supports_pp: bool = True
    supports_dp: bool = True
    supports_ep: bool = True
    supports_non_alias_outputs: bool = True


class ExportRuntime(ABC):
    """Runtime backend for a captured ``torch.export.ExportedProgram``."""

    export_format = "torch"

    def capabilities(
        self,
        compile_config,
        context: DistributedExportContext,
    ) -> ExportRuntimeCapabilities:
        return ExportRuntimeCapabilities()

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

    def capabilities(self, compile_config, context):
        return ExportRuntimeCapabilities(
            supports_cuda_graph_capture=True,
            supports_tp=True,
            supports_pp=True,
            supports_dp=True,
            supports_ep=True,
            supports_non_alias_outputs=True,
        )

    def prepare_runtime(self, exported_program, artifact, compile_config):
        artifact.write_metadata()
        return exported_program.module()


__all__ = [
    "ExportRuntime",
    "ExportRuntimeCapabilities",
    "TorchExportRuntime",
]
