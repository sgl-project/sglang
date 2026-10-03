"""Process-local access to FlashInfer's MNNVL CuTe DSL fusion workspace."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from sglang.srt.runtime_context import get_spec

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)


def _import_kernel_backend():
    # Deferred: the backend drags in CUDA-only deps CPU importers never need.
    from flashinfer.comm import AllReduceFusionPattern, allreduce_fusion
    from flashinfer.comm.mnnvl_cutedsl import DEFAULT_CONFIG
    from flashinfer.comm.mnnvl_cutedsl_ar import MNNVLCuteDSLAllReduceFusionWorkspace

    return (
        MNNVLCuteDSLAllReduceFusionWorkspace,
        allreduce_fusion,
        AllReduceFusionPattern,
        DEFAULT_CONFIG,
    )


# Mirrors the FlashInfer HT device kernel: warp size, bf16 elements per 16B vector.
_WARP_SIZE = 32
_VEC_BF16 = 8

# Kernel bound, with two non-consumer warps beside the reduction warps:
# consumer_threads + (2 + reduction_warps) * WARP_SIZE <= _CUDA_BLOCK_THREADS.
_CUDA_BLOCK_THREADS = 1024
_HT_NON_REDUCTION_WARPS = 2
_HT_REDUCTION_WARP_CHOICES = (1, 2, 4, 8)

_SUPPORTED_TP_SIZES = (2, 4, 8, 16)


def _ht_shard_split(
    hidden_size: int, max_consumer_threads: int
) -> tuple[int, int] | None:
    # The kernel shards a token into consumer_threads * 8 * vectors_per_thread
    # elements, and consumer_threads must divide its 16-byte vector count.
    packs = hidden_size // _VEC_BF16
    # packs // 2 keeps vectors_per_thread >= 2, as in the GB300 presets.
    limit = min(max_consumer_threads, packs // 2)
    for consumer_threads in range(
        limit - limit % _WARP_SIZE, _WARP_SIZE - 1, -_WARP_SIZE
    ):
        if packs % consumer_threads == 0:
            return consumer_threads, packs // consumer_threads
    return None


def _ht_reduction_warp_order(preferred: int) -> tuple[int, ...]:
    # Preset's value first, then nearest, fewer before more: each warp costs consumers.
    return tuple(
        sorted(
            _HT_REDUCTION_WARP_CHOICES,
            key=lambda warps: (warps > preferred, abs(warps - preferred)),
        )
    )


def _ht_shard_major_is_legal(
    consumer_threads: int, rms_token_groups: int, tp_size: int
) -> bool:
    # Shard-major RMS needs an integer number of reduction shards per RMS warp.
    rms_warps_per_token = (consumer_threads // rms_token_groups) // _WARP_SIZE
    return (
        rms_warps_per_token > 0
        and tp_size >= rms_warps_per_token
        and tp_size % rms_warps_per_token == 0
    )


def _ht_retarget(preset, *, hidden_size: int, tp_size: int):
    """``preset`` re-aimed at this shape, or None when no legal split exists.

    Only shape-dependent fields move, except ``rms_shard_major``, which the
    kernel rejects unless tp is a multiple of the RMS warps per token.
    """
    packs = hidden_size // _VEC_BF16
    if packs % tp_size:
        # Kernel: "hidden vector count must be divisible by tp".
        return None
    packs_per_shard = packs // tp_size
    for reduction_warps in _ht_reduction_warp_order(preset.reduction_warps):
        if packs_per_shard % (reduction_warps * _WARP_SIZE):
            continue
        split = _ht_shard_split(
            hidden_size,
            _CUDA_BLOCK_THREADS
            - (_HT_NON_REDUCTION_WARPS + reduction_warps) * _WARP_SIZE,
        )
        if split is None:
            continue
        consumer_threads, vectors_per_thread = split
        return replace(
            preset,
            consumer_threads=consumer_threads,
            vectors_per_thread=vectors_per_thread,
            reduction_warps=reduction_warps,
            rms_shard_major=preset.rms_shard_major
            and _ht_shard_major_is_legal(
                consumer_threads, preset.rms_token_groups, tp_size
            ),
        )
    return None


def _routes(bounds, ll_target, bt_targets, ht_target):
    # Without HT the widest BT range takes the unbounded slot, so the profile
    # still covers the whole workspace capacity.
    from flashinfer.comm.mnnvl_cutedsl import MRangeDispatch

    if ht_target is not None:
        return MRangeDispatch(
            upper_bounds=bounds,
            targets=(ll_target, *bt_targets, ht_target),
        )
    return MRangeDispatch(
        upper_bounds=(*bounds[: len(bt_targets)], None),
        targets=(ll_target, *bt_targets),
    )


def _gb300_presets(*, wide_tp: bool):
    """The shipped GB300 (bounds, LL, BT pair, HT) per operation, TP16 or TP8."""
    from flashinfer.comm.mnnvl_cutedsl.kernel_bt import (
        BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0,
        BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1,
        BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_0,
        BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_1,
        BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0,
        BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1,
        BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_0,
        BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_1,
    )
    from flashinfer.comm.mnnvl_cutedsl.kernel_ht import (
        HT_ALL_REDUCE_GB300_TP8_H8192,
        HT_ALL_REDUCE_GB300_TP16_H8192,
        HT_FINALIZE_GB300_TP8_H8192_K10,
        HT_FINALIZE_GB300_TP16_H8192_K10,
    )
    from flashinfer.comm.mnnvl_cutedsl.kernel_ll import (
        LL_ALL_REDUCE_GB300_TP8_H8192,
        LL_ALL_REDUCE_GB300_TP16_H8192,
        LL_FINALIZE_GB300_TP8_H8192_K10,
        LL_FINALIZE_GB300_TP16_H8192_K10,
    )

    # FlashInfer's measured GB300 crossovers.
    if wide_tp:
        return {
            "finalize": (
                (7, 52, 703, None),
                LL_FINALIZE_GB300_TP16_H8192_K10,
                (
                    BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_0,
                    BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_1,
                ),
                HT_FINALIZE_GB300_TP16_H8192_K10,
            ),
            "all_reduce": (
                (5, 512, 959, None),
                LL_ALL_REDUCE_GB300_TP16_H8192,
                (
                    BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_0,
                    BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_1,
                ),
                HT_ALL_REDUCE_GB300_TP16_H8192,
            ),
        }
    return {
        "finalize": (
            (23, 48, 703, None),
            LL_FINALIZE_GB300_TP8_H8192_K10,
            (
                BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0,
                BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1,
            ),
            HT_FINALIZE_GB300_TP8_H8192_K10,
        ),
        "all_reduce": (
            (15, 256, 1024, None),
            LL_ALL_REDUCE_GB300_TP8_H8192,
            (
                BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0,
                BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1,
            ),
            HT_ALL_REDUCE_GB300_TP8_H8192,
        ),
    }


def _retargeted_config(tp_size: int, hidden_size: int, top_k: int):
    """A single-profile routing config for a shape FlashInfer does not ship.

    Reuses the shipped GB300 crossovers, which were measured at tp=8/16
    hidden=8192 top_k=10 and are unmeasured elsewhere.
    """
    from flashinfer.comm.mnnvl_cutedsl import (
        KernelTarget,
        MNNVLCuteDSLConfig,
        ProtocolKind,
        StaticProfile,
    )

    shipped = _gb300_presets(wide_tp=tp_size >= 16)
    ht = {
        op: _ht_retarget(presets[3], hidden_size=hidden_size, tp_size=tp_size)
        for op, presets in shipped.items()
    }
    if any(preset is None for preset in ht.values()):
        # Both operations share the HT protocol state, so HT goes for both.
        ht = dict.fromkeys(ht)
        logger.warning(
            "MNNVL CuTe DSL: hidden_size=%d admits no HT shard split at "
            "tp_size=%d; serving this shape with the LL and BT routes only, "
            "which costs throughput at large token counts.",
            hidden_size,
            tp_size,
        )

    def routes(op):
        bounds, ll, bt, _ = shipped[op]
        return _routes(
            bounds,
            KernelTarget(protocol=ProtocolKind.LL, preset=ll),
            tuple(KernelTarget(protocol=ProtocolKind.BT, preset=p) for p in bt),
            None
            if ht[op] is None
            else KernelTarget(protocol=ProtocolKind.HT, preset=ht[op]),
        )

    profile = StaticProfile(
        tp_size=tp_size,
        hidden_size=hidden_size,
        top_k=top_k,
        dtype=torch.bfloat16,
        finalize_routes=routes("finalize"),
        all_reduce_routes=routes("all_reduce"),
    )
    return MNNVLCuteDSLConfig(profiles=(profile,))


def _config_for_shape(default_config, *, tp_size: int, hidden_size: int, top_k: int):
    # MNNVLCuteDSLConfig.resolve matches (tp, hidden, top_k, dtype) exactly and
    # DEFAULT_CONFIG ships GB300 H=8192/K=10 only, so other shapes need one.
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
        "re-targeting the GB300 presets at this shape. Their M crossovers were "
        "measured at tp=8/16 hidden=8192 top_k=10, so benchmark against "
        "--flashinfer-allreduce-fusion-backend mnnvl before deploying a shape "
        "that matters.",
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
        # Cached: supports() runs per layer and must not re-enter c10d.
        self.tp_size = dist.get_world_size(process_group)
        if self.tp_size not in _SUPPORTED_TP_SIZES:
            raise ValueError(
                f"MNNVL CuTe DSL fusion supports tp_size in {_SUPPORTED_TP_SIZES}, "
                f"got {self.tp_size}"
            )
        self.device = torch.device(device)

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
            shaped_config = _config_for_shape(
                default_config,
                tp_size=self.tp_size,
                hidden_size=self.hidden_size,
                top_k=self.top_k,
            )
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
                add_residual=True,
                write_residual_output=True,
                config=self.workspace_config,
            )

            # Publish only after the mailbox barrier; without it the ranks
            # would desynchronize their Lamport stages.
            torch.cuda.synchronize(self.device)
            dist.barrier(group=process_group)

    def supports(self, m: int) -> bool:
        if not 1 <= int(m) <= self.max_m:
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = int(permuted_indices.shape[0])
        if not self.supports(m):
            raise ValueError(f"workspace does not support M={m}")
        shape = (m, self.hidden_size)
        norm_output = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        residual_output = torch.empty(shape, dtype=torch.bfloat16, device=self.device)

        pattern = self._patterns.kMoEFinalizeARResidualRMSNorm
        self._allreduce_fusion(
            input=routed_output,
            workspace=self.workspace,
            pattern=pattern,
            # Caller intent only; the routing profile owns the compiled choice.
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        m = int(local_contribution.shape[0])
        if not self.supports(m):
            raise ValueError(f"workspace does not support M={m}")
        norm_output = torch.empty_like(local_contribution)
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


_WORKSPACE: FlashInferMNNVLCuteDSLARFusion | None = None


def get_flashinfer_mnnvl_cutedsl_ar_fusion(
    *,
    hidden_size: int,
    top_k: int,
    max_m: int,
    rms_epsilon: float,
    weight_bias: float,
) -> FlashInferMNNVLCuteDSLARFusion:
    """Build the process-local workspace. Must run before graph capture."""
    global _WORKSPACE
    if _WORKSPACE is not None:
        raise RuntimeError(
            "a second MNNVL CuTe DSL fusion workspace was requested; each one "
            "rendezvouses its own NVLS region, and a process serves one model"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("MNNVL CuTe DSL fusion requires CUDA")

    from sglang.srt.runtime_context import get_parallel

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "creating an MNNVL CuTe DSL fusion workspace during CUDA Graph "
            "capture is forbidden"
        )

    _WORKSPACE = FlashInferMNNVLCuteDSLARFusion(
        hidden_size=hidden_size,
        top_k=top_k,
        max_m=max_m,
        rms_epsilon=rms_epsilon,
        weight_bias=weight_bias,
        process_group=get_parallel().tp_group.device_group,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    return _WORKSPACE
