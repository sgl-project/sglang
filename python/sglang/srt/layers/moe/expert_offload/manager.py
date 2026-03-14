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
"""UVM-based expert weight offloading manager.

Design
------
Expert weights are stored in CUDA Unified Memory (``cudaMallocManaged``).

  Resident experts  -> ``cudaMemAdvise(PREFER_GPU | ACCESSED_BY_GPU)``
                      Pages stay in GPU VRAM; accessed at VRAM bandwidth.

  Offloaded experts -> ``cudaMemAdvise(PREFER_CPU | ACCESSED_BY_GPU)``
                      Pages live in CPU DRAM.  The GPU reads them via PCIe
                      without a page-fault interrupt (ACCESSED_BY_GPU sets up
                      a direct mapping in the GPU's page table).

The MoE kernel indexes ``layer.w_param[topk_id]`` directly -- no ID remapping,
no assembly buffer, no LRU cache.  UVM is fully transparent to CUDA graphs:
kernels are recorded (not executed) during capture, so page placement during
capture is irrelevant.  During replay the GPU reads each expert at whichever
bandwidth tier its pages are currently in.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sglang.srt.layers.moe.expert_offload.config import ExpertOffloadConfig
from sglang.srt.layers.moe.expert_offload.uvm import (
    ADVISE_SET_ACCESSED_BY,
    ADVISE_SET_PREFERRED_LOCATION,
    CUDA_CPU_DEVICE,
    uvm_advise,
    uvm_copy_from_tensor,
    uvm_prefetch_async,
)

logger = logging.getLogger(__name__)


class ExpertOffloadManager:
    """Manages UVM-based expert offloading for a single FusedMoE layer."""

    def __init__(self, config: ExpertOffloadConfig, device: torch.device):
        self.config = config
        self.device = device
        self.device_id: int = device.index if device.index is not None else 0

        # Expert ID lists (populated in initialize_from_weights).
        self.resident_expert_ids: List[int] = []
        self.offloaded_expert_ids: List[int] = []

        # Names of expert-indexed parameters found on the layer.
        self._expert_param_names: List[str] = []

        # Dedicated stream for async prefetch operations.
        self.prefetch_stream: torch.cuda.Stream = torch.cuda.Stream(device=device)

        # Per-layer adaptive resident selection state.
        self._expert_freq: Optional[torch.Tensor] = None
        self._warmup_tokens: int = 0
        self._warmup_done: bool = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_from_weights(self, layer: torch.nn.Module) -> None:
        """Replace GPU expert tensors with UVM-backed tensors.

        For each expert-indexed parameter on ``layer``:
          1. Allocate managed memory (same shape/dtype).
          2. Copy GPU tensor -> managed tensor (device-to-device, on GPU).
          3. Apply ``cudaMemAdvise`` to resident and offloaded slices.
          4. Prefetch resident slice to GPU, offloaded slice to CPU.
          5. Free the original GPU tensor.
          6. Set ``layer.<param>`` = managed tensor (full shape unchanged).

        ``layer.num_local_experts`` is NOT modified -- the kernel uses the
        original routing indices and indexes the full managed tensor.
        """
        cfg = self.config
        num_local = cfg.num_local_experts

        # 1. Choose resident expert IDs.
        self.resident_expert_ids = self._select_resident_ids(cfg)
        resident_set = set(self.resident_expert_ids)
        self.offloaded_expert_ids = [
            i for i in range(num_local) if i not in resident_set
        ]

        logger.debug(
            f"[ExpertOffload] Layer {cfg.layer_idx}: "
            f"{len(self.resident_expert_ids)} resident, "
            f"{len(self.offloaded_expert_ids)} offloaded (UVM)"
        )

        # 2. Find all expert-indexed parameters.
        self._expert_param_names = self._find_expert_param_names(layer, num_local)

        # 3. Process each parameter (one at a time to keep peak GPU usage low).
        for param_name in self._expert_param_names:
            self._migrate_param_to_uvm(layer, param_name)

        # 4. Wait for all prefetch operations to complete before continuing.
        torch.cuda.synchronize(self.device)

        # 5. Register offloaded bytes with SGLang's memory reporting.
        #
        # On PCIe-attached discrete GPUs, cudaMemPrefetchAsync(->CPU) may not
        # immediately return physical GPU pages to the free pool -- the UVM
        # driver keeps them cached until a regular cudaMalloc triggers eviction.
        # This causes torch.cuda.mem_get_info() to under-report available memory,
        # breaking SGLang's init_memory_pool check.
        #
        # Instead of forcing a slow pressure-based eviction here, we register
        # the offloaded bytes with get_available_gpu_memory() so that the check
        # sees the correct "effective" available memory.  The actual eviction
        # happens on-demand when the KV-cache allocator issues cudaMalloc calls.
        if self.offloaded_expert_ids:
            self._register_offloaded_bytes_for_memory_reporting(layer)

    def _register_offloaded_bytes_for_memory_reporting(
        self, layer: torch.nn.Module
    ) -> None:
        """Register this layer's offloaded UVM bytes with get_available_gpu_memory.

        Each call accumulates bytes from one MoE layer.  After all layers are
        processed, the total registered bytes equals the full UVM evictable
        footprint, and SGLang's memory profiling reflects the correct available
        GPU memory.
        """
        from sglang.srt.utils.common import register_uvm_evictable_memory

        total_bytes = 0
        for pname in self._expert_param_names:
            managed: torch.Tensor = getattr(layer, pname)
            bytes_per_expert = managed.nbytes // self.config.num_local_experts
            total_bytes += bytes_per_expert * len(self.offloaded_expert_ids)

        if total_bytes > 0:
            register_uvm_evictable_memory(self.device_id, total_bytes)
            logger.debug(
                f"[ExpertOffload] Layer {self.config.layer_idx}: "
                f"registered {total_bytes / (1 << 20):.1f} MiB offloaded UVM "
                f"as evictable for memory reporting (device {self.device_id})"
            )

    def _migrate_param_to_uvm(self, layer: torch.nn.Module, param_name: str) -> None:
        """Migrate one expert-indexed parameter to managed memory."""
        cfg = self.config

        full_tensor: torch.Tensor = getattr(layer, param_name)
        assert (
            full_tensor.is_contiguous()
        ), f"Expert param {param_name!r} must be contiguous before UVM migration"

        # Step 1: Allocate managed memory and copy GPU data into it.
        managed = uvm_copy_from_tensor(full_tensor)

        # Step 2: Apply memory advice to each expert's slice.
        #   Resident: prefer GPU VRAM (hot path).
        #   Offloaded: prefer CPU + allow GPU to read-through via PCIe.
        for eid in self.resident_expert_ids:
            expert_slice = managed[eid]
            uvm_advise(expert_slice, ADVISE_SET_PREFERRED_LOCATION, self.device_id)
            uvm_advise(expert_slice, ADVISE_SET_ACCESSED_BY, self.device_id)

        for eid in self.offloaded_expert_ids:
            expert_slice = managed[eid]
            uvm_advise(expert_slice, ADVISE_SET_PREFERRED_LOCATION, CUDA_CPU_DEVICE)
            uvm_advise(expert_slice, ADVISE_SET_ACCESSED_BY, self.device_id)

        # Step 3: Prefetch resident pages to GPU, offloaded pages to CPU.
        if self.resident_expert_ids:
            # Prefetch contiguous resident slices efficiently.
            # For first_n selection, resident experts are 0..num_resident-1:
            # a single contiguous prefetch covers the whole range.
            # For manual selection, we prefetch individually.
            self._prefetch_experts_to_device(
                managed, self.resident_expert_ids, self.device_id
            )

        if self.offloaded_expert_ids:
            self._prefetch_experts_to_device(
                managed, self.offloaded_expert_ids, CUDA_CPU_DEVICE
            )

        # Step 4: Free the original GPU tensor to reclaim VRAM.
        if hasattr(layer, "_parameters") and param_name in layer._parameters:
            del layer._parameters[param_name]
        else:
            try:
                delattr(layer, param_name)
            except AttributeError:
                pass

        del full_tensor
        torch.cuda.empty_cache()

        # Step 5: Set managed tensor on the layer (full shape, no slicing).
        object.__setattr__(layer, param_name, managed)

    def _prefetch_experts_to_device(
        self,
        managed: torch.Tensor,
        expert_ids: List[int],
        device_id: int,
    ) -> None:
        """Issue cudaMemPrefetchAsync for a list of expert IDs.

        Tries to coalesce contiguous ID ranges into a single prefetch call
        to reduce CUDA API overhead.
        """
        if not expert_ids:
            return

        # Coalesce into contiguous ranges.
        sorted_ids = sorted(expert_ids)
        ranges: List[tuple] = []
        start = sorted_ids[0]
        end = sorted_ids[0]
        for eid in sorted_ids[1:]:
            if eid == end + 1:
                end = eid
            else:
                ranges.append((start, end))
                start = end = eid
        ranges.append((start, end))

        for lo, hi in ranges:
            # managed[lo : hi+1] is a contiguous slice along dim 0.
            uvm_prefetch_async(
                managed[lo : hi + 1],
                device_id,
                stream=self.prefetch_stream,
            )

    # ------------------------------------------------------------------
    # Per-layer adaptive resident selection (warmup-then-readvise)
    # ------------------------------------------------------------------

    # Minimum batch size to consider a forward pass as real prefill traffic.
    # Pre-capture warmup and decode use small batch sizes with dummy/uniform
    # routing -- their data would pollute the frequency distribution.
    _MIN_TOKENS_FOR_FREQUENCY = 64

    def record_expert_usage(
        self, topk_ids: torch.Tensor, layer: torch.nn.Module
    ) -> None:
        """Record expert routing frequencies from real prefill passes.

        Only counts passes with >= ``_MIN_TOKENS_FOR_FREQUENCY`` tokens to
        filter out dummy pre-capture warmup runs and single-token decode.
        After accumulating ``config.warmup_tokens`` routed tokens, computes
        the optimal per-layer resident set and calls ``cudaMemAdvise``.

        Args:
            topk_ids: shape ``[num_tokens, top_k]``, expert indices chosen by
                the router for the current forward pass.
            layer: the ``FusedMoE`` layer module (needed to access managed
                tensors for ``cudaMemAdvise``/``cudaMemPrefetchAsync``).
        """
        if self._warmup_done or self.config.resident_selection != "frequency":
            return

        # Skip during CUDA graph capture -- .cpu() is not permitted.
        if torch.cuda.is_current_stream_capturing():
            return

        num_tokens = topk_ids.shape[0]

        # Ignore small batches (dummy warmup / decode leak).
        if num_tokens < self._MIN_TOKENS_FOR_FREQUENCY:
            return

        # Lazy-init the frequency tensor on first call.
        if self._expert_freq is None:
            self._expert_freq = torch.zeros(
                self.config.num_local_experts, dtype=torch.int64, device="cpu"
            )

        # Accumulate frequencies on CPU.
        ids_cpu = topk_ids.detach().reshape(-1).cpu()
        counts = torch.bincount(ids_cpu, minlength=self.config.num_local_experts)
        self._expert_freq += counts

        self._warmup_tokens += num_tokens * topk_ids.shape[1]
        if self._warmup_tokens >= self.config.warmup_tokens:
            self._readvise_from_frequency(layer)

    def _readvise_from_frequency(self, layer: torch.nn.Module) -> None:
        """Recompute resident set from frequency data and re-advise UVM pages."""
        assert self._expert_freq is not None
        self._warmup_done = True

        num_resident = self.config.num_resident_experts
        # Top-K most frequently routed experts become the new resident set.
        _, top_ids = self._expert_freq.topk(num_resident)
        new_resident_ids = sorted(top_ids.tolist())
        new_resident_set = set(new_resident_ids)

        old_resident_set = set(self.resident_expert_ids)

        promoted = new_resident_set - old_resident_set
        demoted = old_resident_set - new_resident_set

        if not promoted and not demoted:
            logger.info(
                f"[ExpertOffload] Layer {self.config.layer_idx}: "
                f"warmup complete, resident set unchanged"
            )
            # Free frequency tensor.
            self._expert_freq = None
            return

        logger.info(
            f"[ExpertOffload] Layer {self.config.layer_idx}: "
            f"readvised {len(promoted)} promoted, {len(demoted)} demoted "
            f"(new residents: {new_resident_ids})"
        )

        # Apply cudaMemAdvise + prefetch for each managed parameter.
        for param_name in self._expert_param_names:
            managed: torch.Tensor = getattr(layer, param_name)

            # Promote: prefer GPU
            for eid in promoted:
                expert_slice = managed[eid]
                uvm_advise(expert_slice, ADVISE_SET_PREFERRED_LOCATION, self.device_id)

            # Demote: prefer CPU
            for eid in demoted:
                expert_slice = managed[eid]
                uvm_advise(expert_slice, ADVISE_SET_PREFERRED_LOCATION, CUDA_CPU_DEVICE)

            # Prefetch promoted experts to GPU.
            if promoted:
                self._prefetch_experts_to_device(
                    managed, sorted(promoted), self.device_id
                )
            # Prefetch demoted experts to CPU.
            if demoted:
                self._prefetch_experts_to_device(
                    managed, sorted(demoted), CUDA_CPU_DEVICE
                )

        # Wait for prefetch to complete.
        self.prefetch_stream.synchronize()

        # Update ID lists.
        self.resident_expert_ids = new_resident_ids
        self.offloaded_expert_ids = [
            i for i in range(self.config.num_local_experts) if i not in new_resident_set
        ]

        # Free frequency tensor.
        self._expert_freq = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _select_resident_ids(cfg: ExpertOffloadConfig) -> List[int]:
        num_resident = cfg.num_resident_experts
        if cfg.resident_selection == "first_n":
            return list(range(num_resident))
        elif cfg.resident_selection == "manual" and cfg.resident_expert_ids is not None:
            ids = list(dict.fromkeys(cfg.resident_expert_ids))  # deduplicate
            ids = ids[:num_resident]
            existing = set(ids)
            for i in range(cfg.num_local_experts):
                if len(ids) >= num_resident:
                    break
                if i not in existing:
                    ids.append(i)
                    existing.add(i)
            return ids[:num_resident]
        else:
            return list(range(num_resident))

    @staticmethod
    def _find_expert_param_names(
        layer: torch.nn.Module, num_local_experts: int
    ) -> List[str]:
        """Find parameter names whose first dimension equals num_local_experts."""
        names = []
        for name, param in layer.named_parameters(recurse=False):
            if (
                param is not None
                and param.dim() >= 1
                and param.shape[0] == num_local_experts
            ):
                names.append(name)
        return names
