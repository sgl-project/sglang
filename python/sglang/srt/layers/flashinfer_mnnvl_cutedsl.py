"""Process-local access to FlashInfer's MNNVL CuTe DSL fusion workspace."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import get_spec

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


def _import_kernel_backend():
    # Imported here rather than at module scope: the CuTe DSL backend drags in
    # CUDA-only dependencies that CPU-side importers of this module never need.
    from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion
    from flashinfer.comm.mnnvl_cutedsl import DEFAULT_CONFIG
    from flashinfer.comm.mnnvl_cutedsl_ar import MNNVLCuteDSLAllReduceFusionWorkspace

    return (
        MNNVLCuteDSLAllReduceFusionWorkspace,
        allreduce_fusion,
        AllReduceFusionPattern,
        DEFAULT_CONFIG,
    )


# Mirrors the constants the FlashInfer HT device kernel derives its shard split
# from: warp size, and bf16 elements per 16-byte vector.
_WARP_SIZE = 32
_VEC_BF16 = 8

# The HT kernel spends two warps plus the reduction warps on non-consumer roles.
_HT_MAX_CONSUMER_THREADS = 1024 - 3 * _WARP_SIZE


def _ht_shard_split(hidden_size: int) -> tuple[int, int] | None:
    """``(consumer_threads, vectors_per_thread)`` for the HT persistent kernel.

    The kernel shards a token into consumer_threads * 8 * vectors_per_thread
    elements, and consumer_threads must divide its 16-byte vector count.
    """
    packs = hidden_size // _VEC_BF16
    limit = min(_HT_MAX_CONSUMER_THREADS, packs // 2)
    for consumer_threads in range(
        limit - limit % _WARP_SIZE, _WARP_SIZE - 1, -_WARP_SIZE
    ):
        if packs % consumer_threads == 0:
            return consumer_threads, packs // consumer_threads
    return None


def _ht_reduction_warps(hidden_size: int, tp_size: int, preferred: int) -> int | None:
    """Reduction warps that evenly cover one rank's slice of a token.

    The HT reduction shard is hidden / 8 / tp vectors wide and must divide across
    reduction_warps * 32 threads. Only ever steps down from ``preferred``.
    """
    packs_per_shard = (hidden_size // _VEC_BF16) // tp_size
    for warps in (8, 4, 2, 1):
        if warps <= preferred and packs_per_shard % (warps * _WARP_SIZE) == 0:
            return warps
    return None


def _ht_tunings(hidden_size: int, tp_size: int):
    """Re-target the GB300 HT presets at ``hidden_size``; None when unreachable."""
    from flashinfer.comm.mnnvl_cutedsl.kernel_ht import (
        HTAllReduceTuning,
        HTFinalizeTuning,
    )

    split = _ht_shard_split(hidden_size)
    if split is None:
        return None
    consumer_threads, vectors_per_thread = split

    finalize_warps = _ht_reduction_warps(
        hidden_size, tp_size, HTFinalizeTuning().reduction_warps
    )
    all_reduce_warps = _ht_reduction_warps(
        hidden_size, tp_size, HTAllReduceTuning().reduction_warps
    )
    if finalize_warps is None or all_reduce_warps is None:
        return None

    return (
        HTFinalizeTuning(
            consumer_threads=consumer_threads,
            vectors_per_thread=vectors_per_thread,
            reduction_warps=finalize_warps,
        ),
        HTAllReduceTuning(
            consumer_threads=consumer_threads,
            vectors_per_thread=vectors_per_thread,
            reduction_warps=all_reduce_warps,
        ),
    )


def _retargeted_config(tp_size: int, hidden_size: int, top_k: int):
    """A single-profile routing config for a shape FlashInfer does not ship.

    Reuses the shipped presets and their GB300 crossovers, recomputing only the
    kernel-shape parameters; the crossovers were measured at H=8192 and are
    approximate elsewhere. None when the hidden size admits no HT shard split.
    """
    from flashinfer.comm.mnnvl_cutedsl import (
        KernelTarget,
        MNNVLCuteDSLConfig,
        MRangeDispatch,
        ProtocolKind,
        StaticProfile,
    )
    from flashinfer.comm.mnnvl_cutedsl.kernel_bt import (
        BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0,
        BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1,
        BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0,
        BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1,
    )
    from flashinfer.comm.mnnvl_cutedsl.kernel_ll import (
        LL_ALL_REDUCE_GB300_TP8_H8192,
        LL_ALL_REDUCE_GB300_TP16_H8192,
        LL_FINALIZE_GB300_TP8_H8192_K10,
        LL_FINALIZE_GB300_TP16_H8192_K10,
    )

    tunings = _ht_tunings(hidden_size, tp_size)
    if tunings is None:
        logger.warning(
            "MNNVL CuTe DSL: hidden_size=%d admits no HT shard split at "
            "tp_size=%d; the fusion cannot serve this model.",
            hidden_size,
            tp_size,
        )
        return None
    ht_finalize, ht_all_reduce = tunings

    wide_tp = tp_size >= 16
    ll_finalize = (
        LL_FINALIZE_GB300_TP16_H8192_K10 if wide_tp else LL_FINALIZE_GB300_TP8_H8192_K10
    )
    ll_all_reduce = (
        LL_ALL_REDUCE_GB300_TP16_H8192 if wide_tp else LL_ALL_REDUCE_GB300_TP8_H8192
    )
    # FlashInfer's measured GB300 crossovers; TP16 shifts LL's window down
    # because each rank publishes a smaller slice.
    finalize_bounds = (7, 52, 703, None) if wide_tp else (23, 48, 703, None)
    all_reduce_bounds = (5, 512, 959, None) if wide_tp else (15, 256, 1024, None)

    def target(protocol, preset):
        return KernelTarget(protocol=protocol, preset=preset)

    profile = StaticProfile(
        tp_size=tp_size,
        hidden_size=hidden_size,
        top_k=top_k,
        dtype=torch.bfloat16,
        finalize_routes=MRangeDispatch(
            upper_bounds=finalize_bounds,
            targets=(
                target(ProtocolKind.LL, ll_finalize),
                target(ProtocolKind.BT, BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0),
                target(ProtocolKind.BT, BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1),
                target(ProtocolKind.HT, ht_finalize),
            ),
        ),
        all_reduce_routes=MRangeDispatch(
            upper_bounds=all_reduce_bounds,
            targets=(
                target(ProtocolKind.LL, ll_all_reduce),
                target(ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0),
                target(ProtocolKind.BT, BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1),
                target(ProtocolKind.HT, ht_all_reduce),
            ),
        ),
    )
    return MNNVLCuteDSLConfig(profiles=(profile,))


def _config_for_shape(default_config, *, tp_size: int, hidden_size: int, top_k: int):
    """The shipped config when it covers this shape, else one rebuilt for it.

    MNNVLCuteDSLConfig.resolve matches (tp, hidden, top_k, dtype) exactly and
    DEFAULT_CONFIG ships GB300 H=8192/K=10 only, so every other shape needs one.
    """
    for profile in default_config.profiles:
        if profile.matches(
            tp_size=tp_size,
            hidden_size=hidden_size,
            top_k=top_k,
            dtype=torch.bfloat16,
        ):
            return default_config
    logger.info(
        "MNNVL CuTe DSL: no shipped profile for tp=%d hidden=%d top_k=%d; "
        "re-targeting the GB300 presets at this shape.",
        tp_size,
        hidden_size,
        top_k,
    )
    return _retargeted_config(tp_size, hidden_size, top_k)


def _with_early_finalize_shared_load(config):
    profiles = []
    updated_presets = 0
    for profile in config.profiles:
        targets = []
        for target in profile.finalize_routes.targets:
            preset = target.preset
            if hasattr(preset, "load_shared_expert_before_pdl"):
                preset = replace(preset, load_shared_expert_before_pdl=True)
                target = replace(target, preset=preset)
                updated_presets += 1
            targets.append(target)
        profiles.append(
            replace(
                profile,
                finalize_routes=replace(
                    profile.finalize_routes,
                    targets=tuple(targets),
                ),
            )
        )

    if updated_presets == 0:
        raise RuntimeError(
            "FlashInfer MNNVL config does not expose the finalize shared-load "
            "PDL ordering option"
        )
    return replace(config, profiles=tuple(profiles))


@dataclass(frozen=True, slots=True)
class _WorkspaceSignature:
    hidden_size: int
    top_k: int
    rms_epsilon: float
    weight_bias: float
    max_m: int
    device_index: int
    process_group_identity: int


class FlashInferMNNVLCuteDSLARFusion:
    """One graph-stable workspace serving both supported fusion patterns."""

    def __init__(
        self,
        *,
        hidden_size: int,
        top_k: int,
        max_m: int,
        rms_epsilon: float,
        weight_bias: float,
        process_group: ProcessGroup,
        device: torch.device,
    ) -> None:
        if hidden_size <= 0 or top_k <= 0 or max_m <= 0:
            raise ValueError("hidden_size, top_k, and max_m must be positive")
        if device.type != "cuda":
            raise ValueError(f"MNNVL CuTe DSL fusion requires CUDA, got {device}")

        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)
        self.max_m = int(max_m)
        self.rms_epsilon = float(rms_epsilon)
        self.weight_bias = float(weight_bias)
        self.process_group = process_group
        self.device = torch.device(device)
        self._destroyed = False

        with torch.cuda.device(self.device):
            self.device = torch.device("cuda", torch.cuda.current_device())
            # CuTe DSL obtains NVLS storage through PyTorch symmetric memory, whose
            # process-local backend must be selected before workspace construction.
            import torch.distributed._symmetric_memory as symm_mem

            symmetric_memory_backend = symm_mem.get_backend(self.device)
            if symmetric_memory_backend is None:
                symm_mem.set_backend("NCCL")
                symmetric_memory_backend = symm_mem.get_backend(self.device)
            if symmetric_memory_backend is None:
                raise RuntimeError(
                    "PyTorch symmetric memory has no backend for the current device"
                )
            logger.info(
                "Using PyTorch symmetric-memory backend %s for %s",
                symmetric_memory_backend,
                self.device,
            )

            (
                workspace_type,
                self._allreduce_fusion,
                self._patterns,
                default_config,
            ) = _import_kernel_backend()
            tp_size = dist.get_world_size(process_group)
            shaped_config = _config_for_shape(
                default_config,
                tp_size=tp_size,
                hidden_size=self.hidden_size,
                top_k=self.top_k,
            )
            if shaped_config is None:
                raise RuntimeError(
                    "MNNVL CuTe DSL fusion has no kernel routing profile for "
                    f"tp_size={tp_size} hidden_size={self.hidden_size} "
                    f"top_k={self.top_k}"
                )
            # Only fused finalize launches have a completed shared-expert handoff;
            # standalone AllReduce kernels retain the safe load ordering.

            if get_spec().speculative_algorithm is None:
                self.workspace_config = _with_early_finalize_shared_load(shaped_config)
            else:
                # Early shared load is safe only for a single looping decode graph;
                # alternating draft/verify replays can read an unfinished buffer.
                logger.info(
                    "Speculative decoding active: keeping the FlashInfer MNNVL "
                    "CuTe DSL finalize presets on the safe (non-early-load) "
                    "ordering."
                )
                self.workspace_config = shaped_config
            self.workspace = workspace_type(
                tp_size=tp_size,
                tp_rank=dist.get_rank(process_group),
                max_token_num=self.max_m,
                hidden_dim=self.hidden_size,
                dtype=torch.bfloat16,
                group=process_group,
                top_k=self.top_k,
                rms_eps=self.rms_epsilon,
                routed_scaling_factor=1.0,
                weight_bias=self.weight_bias,
                include_shared_expert=True,
                add_residual=True,
                write_residual_output=True,
                config=self.workspace_config,
            )

            # Publish only after the mailbox barrier; without it the ranks
            # would desynchronize their Lamport stages.
            torch.cuda.synchronize(self.device)
            dist.barrier(group=process_group)

    def supports(self, m: int) -> bool:
        if self._destroyed or not 1 <= int(m) <= self.max_m:
            return False
        return self.workspace.is_buffer_size_sufficient(
            tp_size=dist.get_world_size(self.process_group),
            num_tokens=int(m),
            hidden_dim=self.hidden_size,
            dtype=torch.bfloat16,
        )

    def moe_finalize_all_reduce_rms_norm(
        self,
        *,
        routed_output: torch.Tensor,
        expert_weights: torch.Tensor,
        permuted_indices: torch.Tensor,
        gated_shared_output: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        norm_output: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = int(permuted_indices.shape[0])
        if not self.supports(m):
            raise ValueError(f"workspace does not support M={m}")
        shape = (m, self.hidden_size)
        if norm_output is None:
            norm_output = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        if residual_output is None:
            residual_output = torch.empty(
                shape, dtype=torch.bfloat16, device=self.device
            )

        pattern = self._patterns.kMoEFinalizeARResidualRMSNorm
        self._allreduce_fusion(
            input=routed_output,
            workspace=self.workspace,
            pattern=pattern,
            # The public API carries the caller's PDL intent. The backend's
            # routing profile owns the compiled choice and validates it.
            launch_with_pdl=True,
            residual_in=residual,
            residual_out=residual_output,
            norm_out=norm_output,
            rms_gamma=gamma,
            rms_eps=self.rms_epsilon,
            weight_bias=self.weight_bias,
            expanded_idx_to_permuted_idx=permuted_indices,
            expert_scale_factor=expert_weights,
            shared_expert_output=gated_shared_output,
        )
        return norm_output, residual_output

    def all_reduce_residual_rms_norm(
        self,
        *,
        local_contribution: torch.Tensor,
        residual: torch.Tensor,
        gamma: torch.Tensor,
        norm_output: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = int(local_contribution.shape[0])
        if not self.supports(m):
            raise ValueError(f"workspace does not support M={m}")
        if norm_output is None:
            norm_output = torch.empty_like(local_contribution)
        if residual_output is None:
            residual_output = torch.empty_like(local_contribution)

        pattern = self._patterns.kARResidualRMSNorm
        self._allreduce_fusion(
            input=local_contribution,
            workspace=self.workspace,
            pattern=pattern,
            launch_with_pdl=True,
            residual_in=residual,
            residual_out=residual_output,
            norm_out=norm_output,
            rms_gamma=gamma,
            rms_eps=self.rms_epsilon,
            weight_bias=self.weight_bias,
        )
        return norm_output, residual_output

    def destroy(self) -> None:
        if self._destroyed:
            return
        self.workspace.destroy()
        self._destroyed = True


_WORKSPACES: dict[_WorkspaceSignature, FlashInferMNNVLCuteDSLARFusion] = {}
_WORKSPACES_LOCK = threading.RLock()


def get_flashinfer_mnnvl_cutedsl_ar_fusion(
    *,
    hidden_size: int,
    top_k: int,
    max_m: int,
    rms_epsilon: float,
    weight_bias: float,
) -> FlashInferMNNVLCuteDSLARFusion:
    """Look up, or before graph capture create, the process-local workspace."""
    if not torch.cuda.is_available():
        raise RuntimeError("MNNVL CuTe DSL fusion requires CUDA")

    from sglang.srt.distributed.parallel_state import get_tp_group

    device = torch.device("cuda", torch.cuda.current_device())
    process_group = get_tp_group().device_group
    domain = (
        int(hidden_size),
        int(top_k),
        float(rms_epsilon),
        float(weight_bias),
        int(device.index),
        id(process_group),
    )

    with _WORKSPACES_LOCK:
        compatible = [
            (signature.max_m, instance)
            for signature, instance in _WORKSPACES.items()
            if (
                signature.hidden_size,
                signature.top_k,
                signature.rms_epsilon,
                signature.weight_bias,
                signature.device_index,
                signature.process_group_identity,
            )
            == domain
            and signature.max_m >= int(max_m)
        ]
        if compatible:
            return min(compatible, key=lambda item: item[0])[1]

        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "creating an MNNVL CuTe DSL fusion workspace during CUDA Graph "
                "capture is forbidden"
            )
        if _WORKSPACES:
            # Each workspace rendezvouses its own NVLS symmetric-memory region.
            raise RuntimeError(
                "a second MNNVL CuTe DSL fusion workspace was requested; one "
                "model configuration per process"
            )

        signature = _WorkspaceSignature(
            hidden_size=int(hidden_size),
            top_k=int(top_k),
            rms_epsilon=float(rms_epsilon),
            weight_bias=float(weight_bias),
            max_m=int(max_m),
            device_index=int(device.index),
            process_group_identity=id(process_group),
        )
        instance = FlashInferMNNVLCuteDSLARFusion(
            hidden_size=hidden_size,
            top_k=top_k,
            max_m=max_m,
            rms_epsilon=rms_epsilon,
            weight_bias=weight_bias,
            process_group=process_group,
            device=device,
        )
        _WORKSPACES[signature] = instance
        return instance
