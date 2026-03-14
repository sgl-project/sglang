# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ExpertOffloadWrapperMethod: wraps a GPU FusedMoEMethod with UVM expert offloading.

UVM design
----------
Expert weights are stored in CUDA Unified Memory (cudaMallocManaged).
Resident experts are advised PREFER_GPU (pages stay in VRAM).
Offloaded experts are advised PREFER_CPU + ACCESSED_BY_GPU (pages live in CPU
DRAM; the GPU reads them via PCIe read-through without a page fault interrupt).

No static/dynamic split.  No ID remapping.  No assembly buffer.  No LRU cache.
UVM is fully transparent to CUDA graphs -- kernels are recorded during capture,
not executed, so page placement is irrelevant at capture time.  During replay the
GPU accesses each expert at whichever bandwidth tier its pages are currently in.

Mutually exclusive with KTransformers (--kt-weight-path).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.moe.expert_offload.config import ExpertOffloadConfig
from sglang.srt.layers.moe.expert_offload.manager import ExpertOffloadManager
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)


class ExpertOffloadWrapperMethod(FusedMoEMethodBase):
    """Wraps a GPU FusedMoEMethodBase with UVM-based expert weight offloading.

    Expert weights are stored in CUDA Unified Memory:
      - Resident experts  -> PREFER_GPU + ACCESSED_BY_GPU (VRAM, full bandwidth)
      - Offloaded experts -> PREFER_CPU + ACCESSED_BY_GPU (CPU DRAM, PCIe read-through)

    The MoE kernel indexes the full managed tensor directly -- no ID remapping,
    no assembly buffer, no LRU cache.  UVM handles correctness transparently.

    Mutually exclusive with KTransformers (KTEPWrapperMethod).
    """

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        config: ExpertOffloadConfig,
    ):
        super().__init__()
        self.gpu_method = gpu_method
        self.config = config
        self.manager: Optional[ExpertOffloadManager] = None

    # ------------------------------------------------------------------
    # FusedMoEMethodBase interface
    # ------------------------------------------------------------------

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Delegate weight creation to the wrapped GPU method."""
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Initialise UVM expert offloading after checkpoint weights are loaded.

        1. Calls gpu_method.process_weights_after_loading (e.g. FP8 scale folding).
        2. Creates ExpertOffloadManager and migrates expert tensors to managed memory.
           Resident experts: PREFER_GPU + ACCESSED_BY_GPU (stay in VRAM).
           Offloaded experts: PREFER_CPU + ACCESSED_BY_GPU (PCIe read-through).
        3. layer.num_local_experts is NOT modified -- the kernel indexes the full tensor.
        """
        # First, let the wrapped method do its own post-load processing.
        if hasattr(self.gpu_method, "process_weights_after_loading"):
            self.gpu_method.process_weights_after_loading(layer)

        # Determine the device for UVM operations.
        device = next(
            (p.device for p in layer.parameters() if p.device.type == "cuda"), None
        )
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())

        # Create manager and migrate weights to UVM.
        self.manager = ExpertOffloadManager(self.config, device)
        self.manager.initialize_from_weights(layer)

        logger.info(
            f"[ExpertOffload] Layer {self.config.layer_idx}: "
            f"{self.config.num_resident_experts}/{self.config.num_local_experts} resident "
            f"({self.config.num_offloaded_experts} offloaded via UVM)"
        )

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config) -> None:
        """Delegate runner creation to the wrapped GPU method.

        With UVM, layer.num_local_experts is never overridden -- the full managed
        tensor is always indexed directly by the kernel.  No override needed here.
        """
        self.gpu_method.create_moe_runner(layer, moe_runner_config)

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "DispatchOutput",
    ) -> "CombineInput":
        """Apply the MoE layer with UVM-backed expert weights.

        UVM handles expert access transparently -- no static/dynamic split,
        no ID remapping, no assembly buffer.  Works identically during:
          - Prefill (eager): offloaded experts read via PCIe (~64 GB/s).
          - Decode (CUDA graph capture): kernels are recorded, not executed.
          - Decode (CUDA graph replay): resident -> VRAM, offloaded -> PCIe.
        Full correctness in all phases.
        """
        return self.gpu_method.apply(layer, dispatch_output)

    # ------------------------------------------------------------------
    # Transparent attribute delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        """Delegate unknown attribute access to the wrapped GPU method."""
        if name in ("gpu_method", "config", "manager"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self.gpu_method, name)
