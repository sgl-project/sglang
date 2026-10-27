"""Experimental Qwen 8A{2,4,8}F and GLM-5 8A8F/S2 bridge to DeepGEMM's M2N component.

Only balanced decode at the configured logical stage width uses this plane.
Prefill, tails and empty lanes keep the ordinary AFD path and its expert weights.
"""

from __future__ import annotations

import importlib
import threading
from datetime import timedelta
from typing import Any

from .contracts import AFDError, AFDRole


def m2n_model_spec(adapter: Any) -> dict:
    """Explicit model geometries; this does not admit arbitrary MoE checkpoints."""
    config = adapter.inner.config
    if (
        adapter.hidden_size == 4096
        and adapter.num_layers == 94
        and getattr(config, "num_experts", None) == 128
        and config.moe_intermediate_size == 1536
        and config.num_experts_per_tok == 8
    ):
        return {
            "hidden": 4096,
            "intermediate": 1536,
            "experts": 128,
            "first_layer": 0,
            "output_scale": 1.0,
            "identity": "qwen",
        }
    if (
        getattr(config, "model_type", None) == "glm_moe_dsa"
        and adapter.hidden_size == 6144
        and adapter.num_layers == 78
        and config.n_routed_experts == 256
        and config.moe_intermediate_size == 2048
        and config.num_experts_per_tok == 8
        and config.first_k_dense_replace == 3
        and config.moe_layer_freq == 1
        and config.n_shared_experts == 1
        and config.routed_scaling_factor == 2.5
        and not getattr(config, "num_hash_layers", 0)
    ):
        return {
            "hidden": 6144,
            "intermediate": 2048,
            "experts": 256,
            "first_layer": 3,
            "output_scale": 2.5,
            "identity": "glm5",
        }
    raise AFDError("AFD_M2N_MODEL_LAYOUT_UNSUPPORTED")


def joint_rank(role: AFDRole, lane: int) -> int:
    # The existing control world is F-first; M2N requires AG ranks first.
    return lane if role == AFDRole.ATTENTION else 8 + lane


def _packed_scale(weight: Any, scale: Any) -> Any:
    import torch

    from sglang.srt.layers.quantization.fp8_utils import transform_scale_ue8m0

    if scale.dtype == torch.int32 and getattr(scale, "format_ue8m0", False):
        expected = (weight.shape[0], weight.shape[1], (weight.shape[2] + 511) // 512)
        if tuple(scale.shape) != expected:
            raise AFDError("AFD_M2N_SCALE_LAYOUT_UNSUPPORTED", str(scale.shape))
        return scale.detach()
    expected = (weight.shape[0], weight.shape[1] // 128, weight.shape[2] // 128)
    if scale.dtype != torch.float32 or tuple(scale.shape) != expected:
        raise AFDError("AFD_M2N_SCALE_LAYOUT_UNSUPPORTED", str(scale.shape))
    # The common recipe owns requantization. Never round only the scales here.
    bits = scale.view(torch.int32)
    if not bool((torch.isfinite(scale) & (scale > 0) & ((bits & 0x7FFFFF) == 0)).all()):
        raise AFDError("AFD_M2N_COMMON_UE8M0_RECIPE_REQUIRED")
    return transform_scale_ue8m0(scale, mn=weight.shape[1])


def _prepare_m2n_weights(l1_weights: tuple, l2_weights: tuple) -> tuple:
    from sglang.srt.layers.moe.mega_moe import (
        _interleave_mega_moe_l1_weights,
        _mega_moe_kernel_scale,
        _transpose_mega_moe_sf_for_utccp,
    )

    w2, sf2 = l2_weights
    mw13, msf13 = _interleave_mega_moe_l1_weights(l1_weights, "fp8xfp4")
    return (
        (mw13, _transpose_mega_moe_sf_for_utccp(_mega_moe_kernel_scale(mw13, msf13))),
        (w2, _transpose_mega_moe_sf_for_utccp(_mega_moe_kernel_scale(w2, sf2))),
    )


def prepare_layer_weights(
    experts: Any,
    *,
    num_experts_per_rank: int = 16,
    hidden_size: int = 4096,
    intermediate_size: int = 1536,
) -> tuple:
    """Prepare separate M2N views without touching the prefill parameters."""
    import torch

    w13, w2 = experts.w13_weight.detach(), experts.w2_weight.detach()
    if (
        w13.dtype != torch.float8_e4m3fn
        or w2.dtype != torch.float8_e4m3fn
        or num_experts_per_rank not in (16, 32, 64)
        or experts.num_local_experts != num_experts_per_rank
        or (hidden_size, intermediate_size) not in ((4096, 1536), (6144, 2048))
        or tuple(w13.shape)
        != (num_experts_per_rank, 2 * intermediate_size, hidden_size)
        or tuple(w2.shape) != (num_experts_per_rank, hidden_size, intermediate_size)
    ):
        raise AFDError("AFD_M2N_FP8_EXPERT_LAYOUT_REQUIRED")
    scales = tuple(
        getattr(experts, name, None)
        for name in ("w13_weight_scale_inv", "w2_weight_scale_inv")
    )
    if any(scale is None for scale in scales):
        raise AFDError("AFD_M2N_COMMON_BLOCK_FP8_RECIPE_REQUIRED")
    sf13, sf2 = (_packed_scale(w, s) for w, s in zip((w13, w2), scales))
    # This helper allocates the interleaved W13 copy. W2's canonical FP8 bytes
    # can be shared; transformed scale tensors remain separately owned here.
    return _prepare_m2n_weights((w13, sf13), (w2, sf2))


class AFDM2NDataPlane:
    needs_warmup_barrier = True
    communicators = ()
    # Existing accounting; private allocations still belong to the component.
    hbm_bytes = 0

    def __init__(self, *, transport: Any, adapter: Any, config: Any, device: Any):
        import torch

        from sglang.srt.utils.common import init_custom_process_group

        # Torch NCCL eager initialization requires an explicit CUDA ordinal.
        # Attention runners may supply device="cuda"; FFN already uses cuda:N.
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self._torch = torch
        self._stage_rows = config.experimental_m2n_stage_rows
        self._attention_lanes = config.attention_lane_count
        self._ffn_lanes = config.lanes
        self._spec = m2n_model_spec(adapter)
        if self._spec["identity"] == "glm5" and self._ffn_lanes != 8:
            raise AFDError("AFD_M2N_GLM5_8A8F_REQUIRED")
        self.first_layer = self._spec["first_layer"]
        self._num_experts_per_rank = self._spec["experts"] // self._ffn_lanes
        self._role = transport.role
        self._device = device
        self._closed = False
        self._startup_barrier_pending = True
        self._component = importlib.import_module("deep_gemm.mega.m2n")
        capability = torch.cuda.get_device_capability(device)
        if capability not in ((10, 0), (10, 3)):
            raise AFDError("AFD_M2N_SM100_SM103_REQUIRED", str(capability))
        self._group = init_custom_process_group(
            backend="nccl",
            store=torch.distributed.PrefixStore("afd-m2n-r1", transport._control.store),
            world_size=self._attention_lanes + self._ffn_lanes,
            rank=joint_rank(transport.role, transport.lane),
            group_name="afd_m2n_r1",
            timeout=timedelta(seconds=config.rendezvous_timeout_seconds),
            device_id=torch.device(device),
        )
        self._weights = ()
        self._weight_descs = None
        if self._role == AFDRole.FFN:
            self._weights = tuple(
                prepare_layer_weights(
                    layer.mlp.experts,
                    num_experts_per_rank=self._num_experts_per_rank,
                    hidden_size=self._spec["hidden"],
                    intermediate_size=self._spec["intermediate"],
                )
                for layer in adapter.inner.layers[self.first_layer :]
            )
        self._buffers = tuple(
            self._component.get_symm_buffer_for_mega_moe_m2n(
                self._group,
                ag_size=self._attention_lanes,
                eg_size=self._ffn_lanes,
                num_experts=self._spec["experts"],
                num_max_tokens_per_rank=self._stage_rows,
                num_topk=8,
                hidden=self._spec["hidden"],
                intermediate_hidden=self._spec["intermediate"],
                mma_type="fp8xfp4",
            )
            for _ in range(2)
        )
        if self._role == AFDRole.FFN:
            self._weight_descs = torch.stack(
                [
                    self._component.make_m2n_weight_descs(
                        w13,
                        w2,
                        num_experts_per_rank=self._num_experts_per_rank,
                        hidden=self._spec["hidden"],
                        intermediate_hidden=self._spec["intermediate"],
                        mma_type="fp8xfp4",
                    )
                    for w13, w2 in self._weights
                ]
            ).to(device=device)
        # Use the configured communication budget across two stages.
        self._streams = (
            tuple(torch.cuda.Stream(device=device) for _ in range(2))
            if self._role == AFDRole.ATTENTION
            else ()
        )

    def _start(self) -> None:
        if self._startup_barrier_pending:
            self._group.barrier().wait()
            self._startup_barrier_pending = False

    def dispatch(self, *, hidden: Any, ids: Any, weights: Any, stage: int, output: Any):
        if (
            self._role != AFDRole.ATTENTION
            or tuple(hidden.shape) != (self._stage_rows, self._spec["hidden"])
            or hidden.shape != output.shape
            or tuple(ids.shape) != (self._stage_rows, 8)
            or tuple(weights.shape) != (self._stage_rows, 8)
            or stage not in (0, 1)
            or hidden.dtype != self._torch.bfloat16
            or output.dtype != self._torch.bfloat16
        ):
            raise AFDError("AFD_M2N_AG_INPUT_INVALID")
        self._start()
        torch = self._torch
        stream = self._streams[stage]
        hidden, ids, weights = (
            hidden.contiguous(),
            ids.contiguous(),
            weights.float().contiguous(),
        )
        # Conversion kernels are producers too; the AG side stream must wait
        # for them, not merely for the original router/attention outputs.
        stream.wait_stream(torch.cuda.current_stream(self._device))
        for tensor in (hidden, ids, weights, output):
            tensor.record_stream(stream)
        with torch.cuda.stream(stream):
            self._component.mega_moe_m2n_ag(
                output,
                hidden,
                ids,
                weights,
                self._buffers[stage],
                output_scale=self._spec["output_scale"],
                num_sms=12,
            )
            event = torch.cuda.Event()
            event.record(stream)
        return event

    def run_ffn_step(self) -> tuple:
        if self._role != AFDRole.FFN:
            raise AFDError("AFD_M2N_EG_ROLE_REQUIRED")
        self._start()
        self._component.mega_moe_m2n_eg(
            self._weight_descs,
            self._buffers,
            expected_num_tokens_per_rank=self._stage_rows,
            num_experts_per_rank=self._num_experts_per_rank,
            use_fp8_weights=True,
            activation_clamp=None,
            fast_math=True,
            num_sms=0,
        )
        # EG writes the layer2 activation workspaces. They remain alive with the
        # symmetric allocation and satisfy the existing graph output contract.
        return tuple((buffer.l2_acts,) for buffer in self._buffers)

    def finish_warmup(self) -> None:
        self._torch.cuda.synchronize(self._device)
        self._group.barrier().wait()

    def close(self, *, seconds: int) -> None:
        if self._closed:
            return
        self._closed = True
        failure = []

        def release():
            try:
                for buffer in self._buffers:
                    buffer.destroy()
                self._torch.distributed.destroy_process_group(self._group)
                self._weights = ()
                self._weight_descs = None
                self._group = None
            except BaseException as exc:  # noqa: BLE001 - report cleanup thread failure to caller
                failure.append(exc)

        thread = threading.Thread(target=release, daemon=True, name="afd-m2n-close")
        thread.start()
        thread.join(seconds)
        if thread.is_alive():
            raise AFDError("AFD_M2N_CLOSE_TIMEOUT")
        if failure:
            raise AFDError("AFD_M2N_CLOSE_FAILED") from failure[0]
