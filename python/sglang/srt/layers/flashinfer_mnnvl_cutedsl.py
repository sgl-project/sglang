"""Process-local access to FlashInfer's MNNVL CuTe DSL fusion workspace."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, replace
from functools import lru_cache
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


def _config_for_shape(config, *, tp_size: int, hidden_size: int, top_k: int):
    """Return a config that has a profile for this static shape.

    FlashInfer ships profiles only for the shapes it has tuned. A model whose
    (tp_size, hidden_size, top_k) is not among them can still run the kernels:
    the routes carry launch tunings, and the shape reaches the kernel through
    the workspace constructor. Re-stamp the nearest tuned profile so the
    dispatch tables are reused rather than invented.
    """
    shape = dict(
        tp_size=tp_size, hidden_size=hidden_size, top_k=top_k, dtype=torch.bfloat16
    )
    if any(profile.matches(**shape) for profile in config.profiles):
        return config

    if not config.profiles:
        raise RuntimeError("FlashInfer MNNVL config carries no profiles")
    # Nearest by TP size, then hidden size, then top-k: the route boundaries are
    # M ranges, and those track how much work one rank publishes per token, of
    # which top-k is the smallest term -- it multiplies only the finalize
    # kernel's gather.
    donor = min(
        config.profiles,
        key=lambda profile: (
            abs(profile.tp_size - tp_size),
            abs(profile.hidden_size - hidden_size),
            abs(profile.top_k - top_k),
        ),
    )
    logger.info(
        "No tuned FlashInfer MNNVL profile for tp_size=%d hidden_size=%d top_k=%d; "
        "reusing the dispatch tables of the tp_size=%d hidden_size=%d top_k=%d profile",
        tp_size,
        hidden_size,
        top_k,
        donor.tp_size,
        donor.hidden_size,
        donor.top_k,
    )
    return replace(
        config,
        profiles=(
            replace(
                donor,
                tp_size=tp_size,
                hidden_size=hidden_size,
                top_k=top_k,
                finalize_routes=_refit_routes(donor.finalize_routes, hidden_size),
                all_reduce_routes=_refit_routes(donor.all_reduce_routes, hidden_size),
            ),
        ),
    )


def _refit_routes(routes, hidden_size: int):
    """Shrink presets whose thread shard does not tile the donor's hidden size.

    The HT presets shard a token across ``consumer_threads * 8 *
    vectors_per_thread`` elements and reject a hidden size that does not divide
    by it. Halving the per-thread vector count, then the thread count, keeps the
    preset's shape while making the shard fit; everything else it encodes --
    stage depth, warp roles, RMS grouping -- is carried over untouched.
    """
    return replace(
        routes,
        targets=tuple(
            replace(target, preset=_refit_preset(target.preset, hidden_size))
            for target in routes.targets
        ),
    )


def _refit_preset(preset, hidden_size: int):
    from flashinfer.comm.mnnvl_cutedsl.kernel_ht import (
        HTAllReduceTuning,
        HTFinalizeTuning,
    )

    if not isinstance(preset, (HTAllReduceTuning, HTFinalizeTuning)):
        return preset

    threads = preset.consumer_threads
    vectors = preset.vectors_per_thread
    groups = preset.rms_token_groups
    while hidden_size % (threads * 8 * vectors) or threads * 8 * vectors > hidden_size:
        if vectors > 1:
            vectors //= 2
        elif (
            threads // 2 >= 32
            and not (threads // 2) % 32
            and not (threads // 2) % groups
        ):
            threads //= 2
        else:
            raise RuntimeError(
                f"cannot refit MNNVL preset {preset} to hidden_size={hidden_size}"
            )
    if (threads, vectors) == (preset.consumer_threads, preset.vectors_per_thread):
        return preset
    logger.info(
        "Refit MNNVL HT preset for hidden_size=%d: consumer_threads %d -> %d, "
        "vectors_per_thread %d -> %d",
        hidden_size,
        preset.consumer_threads,
        threads,
        preset.vectors_per_thread,
        vectors,
    )
    return replace(preset, consumer_threads=threads, vectors_per_thread=vectors)


@dataclass(frozen=True, slots=True)
class _WorkspaceSignature:
    hidden_size: int
    top_k: int
    rms_epsilon: float
    weight_bias: float
    max_m: int
    fuse_residual: bool
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
        fuse_residual: bool = True,
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
        # Models whose residual stream is not a plain add (mHC hyper-connections,
        # for one) take the reduced value back and combine it themselves.
        self.fuse_residual = bool(fuse_residual)
        self.process_group = process_group
        self.tp_size = dist.get_world_size(process_group)
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
            default_config = _config_for_shape(
                default_config,
                tp_size=self.tp_size,
                hidden_size=self.hidden_size,
                top_k=self.top_k,
            )
            # Only fused finalize launches have a completed shared-expert handoff;
            # standalone AllReduce kernels retain the safe load ordering.
            if get_spec().speculative_algorithm is None:
                self.workspace_config = _with_early_finalize_shared_load(default_config)
            else:
                # Early shared load is safe only for a single looping decode graph;
                # alternating draft/verify replays can read an unfinished buffer.
                logger.info(
                    "Speculative decoding active: keeping the FlashInfer MNNVL "
                    "CuTe DSL finalize presets on the safe (non-early-load) "
                    "ordering."
                )
                self.workspace_config = default_config
            self.workspace = workspace_type(
                tp_size=self.tp_size,
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
                add_residual=self.fuse_residual,
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
            tp_size=self.tp_size,
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
        if not self.fuse_residual:
            raise RuntimeError("workspace was compiled without the residual add")
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
        if not self.fuse_residual:
            raise RuntimeError("workspace was compiled without the residual add")
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

    def all_reduce(
        self,
        *,
        local_contribution: torch.Tensor,
        gamma: torch.Tensor,
        norm_scratch: torch.Tensor,
    ) -> torch.Tensor:
        """Reduce across TP ranks and hand the sum back unnormalized.

        The compiled kernel always produces the normalized value as well; a
        caller whose residual stream is not a plain add reads the pre-norm
        output and lets ``norm_scratch`` absorb the rest.
        """
        if self.fuse_residual:
            raise RuntimeError("workspace was compiled with the residual add")
        m = int(local_contribution.shape[0])
        if not self.supports(m):
            raise ValueError(f"workspace does not support M={m}")
        output = torch.empty_like(local_contribution)
        self._allreduce_fusion(
            input=local_contribution,
            workspace=self.workspace,
            pattern=self._patterns.kARResidualRMSNorm,
            launch_with_pdl=True,
            residual_out=output,
            norm_out=norm_scratch,
            rms_gamma=gamma,
            rms_eps=self.rms_epsilon,
            weight_bias=self.weight_bias,
        )
        return output

    def destroy(self) -> None:
        if self._destroyed:
            return
        self.workspace.destroy()
        self._destroyed = True


_WORKSPACES: dict[_WorkspaceSignature, FlashInferMNNVLCuteDSLARFusion] = {}
_WORKSPACES_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def _max_workspace_instances() -> int:
    from sglang.srt.environ import envs

    value = int(envs.SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION_MAX_INSTANCES.get())
    if value < 1:
        raise ValueError(
            "SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION_MAX_INSTANCES must be positive"
        )
    return value


def get_flashinfer_mnnvl_cutedsl_ar_fusion(
    *,
    hidden_size: int | None = None,
    top_k: int | None = None,
    max_m: int | None = None,
    rms_epsilon: float | None = None,
    weight_bias: float | None = None,
    fuse_residual: bool = True,
) -> FlashInferMNNVLCuteDSLARFusion:
    """Lookup, or before graph capture create, the process-local workspace."""
    supplied = (hidden_size, top_k, max_m, rms_epsilon, weight_bias)
    if all(value is None for value in supplied):
        with _WORKSPACES_LOCK:
            if len(_WORKSPACES) == 1:
                return next(iter(_WORKSPACES.values()))
            if not _WORKSPACES:
                raise RuntimeError(
                    "MNNVL CuTe DSL fusion workspace was not initialized before use"
                )
            raise RuntimeError(
                "multiple MNNVL CuTe DSL fusion workspaces exist; configuration "
                "arguments are required"
            )
    if any(value is None for value in supplied):
        raise TypeError(
            "hidden_size, top_k, max_m, rms_epsilon, and weight_bias must be "
            "supplied together"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("MNNVL CuTe DSL fusion requires CUDA")

    assert hidden_size is not None
    assert top_k is not None
    assert max_m is not None
    assert rms_epsilon is not None
    assert weight_bias is not None
    from sglang.srt.distributed.parallel_state import get_tp_group

    device = torch.device("cuda", torch.cuda.current_device())
    process_group = get_tp_group().device_group
    domain = (
        int(hidden_size),
        int(top_k),
        float(rms_epsilon),
        float(weight_bias),
        bool(fuse_residual),
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
                signature.fuse_residual,
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
        if len(_WORKSPACES) >= _max_workspace_instances():
            raise RuntimeError(
                "MNNVL CuTe DSL fusion workspace instance limit exceeded; "
                "increase SGLANG_FLASHINFER_MNNVL_CUTEDSL_AR_FUSION_MAX_INSTANCES "
                "only when multiple model configurations intentionally coexist"
            )

        signature = _WorkspaceSignature(
            hidden_size=int(hidden_size),
            top_k=int(top_k),
            rms_epsilon=float(rms_epsilon),
            weight_bias=float(weight_bias),
            max_m=int(max_m),
            fuse_residual=bool(fuse_residual),
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
            fuse_residual=fuse_residual,
        )
        _WORKSPACES[signature] = instance
        return instance
