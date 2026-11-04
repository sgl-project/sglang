"""SAIL KDA attention backend: the PLA operators behind the KDA interfaces.

The community ``KDAAttnBackend`` keeps its Triton / CuTe DSL / FlashInfer
implementations; this module only overrides the pieces that are served by the
PLA operator library on PPU. ``SGLANG_SAIL_PLA`` gates every override, so a
PLA-less install falls back to the inherited implementations.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.sail.attention.kda_flashkda import (
    SailFlashKDAKernel,
)
from sglang.srt.hardware_backend.sail.attention.kda_triton import SailTritonKDAKernel
from sglang.srt.hardware_backend.sail.kernels import kda as sail_kda
from sglang.srt.layers.attention.linear.kda_backend import (
    KDAAttnBackend,
    KDAKernelDispatcher,
)
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnBackends,
    LinearAttnKernelBackend,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.utils import is_sail


def sail_pla_prefill_default(model_config) -> Optional[str]:
    """This runner's SAIL KDA prefill default: the PLA kernel.

    Returned as a *per-runner default* (fed into
    ``resolve_linear_attn_backends(prefill_default=...)``) rather than written
    into the process-wide ``--linear-attn-prefill-backend`` value, because a
    target and its speculative draft coexist in one process and can want
    different prefill kernels. KDA target models opt in here; a draft runner
    whose model is GDN keeps its own backend (GDNKernelDispatcher has no
    FlashKDA prefill kernel and rejects it). An explicit
    ``--linear-attn-prefill-backend`` still wins over this default.
    """
    if not is_sail() or not envs.SGLANG_SAIL_PLA.get():
        return None
    try:
        hf_config = model_config.hf_config
    except Exception:  # noqa: BLE001 - resolution must not fail on this probe
        return None
    from sglang.srt.configs.model_config import uses_kda_attention

    if not uses_kda_attention(hf_config):
        return None
    return "flashkda"


def _pla_prefill_backend(
    prefill_backend: LinearAttnKernelBackend,
) -> LinearAttnKernelBackend:
    if prefill_backend.is_flashkda():
        return prefill_backend
    return LinearAttnKernelBackend.TRITON


class SailKDAKernelDispatcher(KDAKernelDispatcher):
    """KDA kernel dispatcher that binds the PLA-backed kernel slots.

    On SAIL only fused decode and the FlashKDA prefill/prefill-graph interface
    exist; decode, verify and any other prefill backend stay on Triton because
    that is the only implementation PLA covers.
    """

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
        verify_backend: LinearAttnKernelBackend,
    ):
        prefill_backend = _pla_prefill_backend(prefill_backend)
        super().__init__(
            LinearAttnKernelBackend.TRITON,
            prefill_backend,
            LinearAttnKernelBackend.TRITON,
        )
        sail_triton = SailTritonKDAKernel()
        self.decode_kernel = sail_triton
        self.verify_kernel = sail_triton
        if prefill_backend.is_flashkda():
            self.extend_kernel = SailFlashKDAKernel()


class SailKDAAttnBackend(KDAAttnBackend):
    """KDAAttnBackend whose fused slots run on the PLA operators."""

    @staticmethod
    def _effective_backends(backends: LinearAttnBackends) -> LinearAttnBackends:
        # The kernels SailKDAKernelDispatcher binds, so checks keyed on the
        # verify backend (e.g. the FlashInfer fused-accept path) see Triton.
        return LinearAttnBackends(
            decode=LinearAttnKernelBackend.TRITON,
            prefill=_pla_prefill_backend(backends.prefill),
            verify=LinearAttnKernelBackend.TRITON,
        )

    def _create_kernel_dispatcher(self, backends) -> KDAKernelDispatcher:
        return SailKDAKernelDispatcher(
            backends.decode, backends.prefill, backends.verify
        )

    def _fused_decode_covered(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        onorm_gate: torch.Tensor,
    ) -> bool:
        if not sail_kda.pla_enabled():
            return super()._fused_decode_covered(
                mixed_qkv, a, b, conv_states, ssm_states, cache_indices, onorm_gate
            )
        return sail_kda.fused_decode_covered(
            mixed_qkv, a, b, conv_states, ssm_states, cache_indices, onorm_gate
        )

    def _run_fused_decode(
        self,
        *,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        fused_static: tuple,
        layer: RadixLinearAttention,
        forward_batch,
        onorm_gate: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        if not sail_kda.pla_enabled():
            return super()._run_fused_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                conv_states=conv_states,
                fused_static=fused_static,
                layer=layer,
                forward_batch=forward_batch,
                onorm_gate=onorm_gate,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
            )
        _, _, _, conv_bias, a_log, onorm_w, onorm_eps = fused_static
        return sail_kda.fused_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            conv_states=conv_states,
            conv_bias=conv_bias,
            A_log=a_log,
            dt_bias=layer.dt_bias,
            onorm_g=onorm_gate,
            onorm_weight=onorm_w,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            onorm_eps=onorm_eps,
            fused_weight=getattr(layer, "_k3_fused_decode_weight", None),
            actual_batch_size=getattr(forward_batch, "_original_batch_size", None),
            lower_bound=layer.lower_bound,
        )

    def _can_run_dspark_cutedsl_mtp(
        self,
        *,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        draft_token_num: int,
        ragged_layout,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        intermediate_state_cache: Optional[torch.Tensor],
        intermediate_conv_window_cache: torch.Tensor,
        retrieve_parent_token: Optional[torch.Tensor],
        replayssm_rawv: Optional[torch.Tensor],
    ) -> bool:
        """Whether the PLA verify inputs are supported for this batch."""
        if not sail_kda.pla_enabled():
            return super()._can_run_dspark_cutedsl_mtp(
                layer=layer,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                draft_token_num=draft_token_num,
                ragged_layout=ragged_layout,
                conv_states=conv_states,
                ssm_states=ssm_states,
                intermediate_state_cache=intermediate_state_cache,
                intermediate_conv_window_cache=intermediate_conv_window_cache,
                retrieve_parent_token=retrieve_parent_token,
                replayssm_rawv=replayssm_rawv,
            )
        # PLA still indexes its ring by Mamba slot; a request-level ring keeps
        # the Triton path.
        pool = self.state_pool
        if pool is not None and pool.replayssm_request_ring:
            return False
        if ragged_layout is not None or retrieve_parent_token is not None:
            return False
        # draft_token_num = 1 bonus + dspark block size; the operator is
        # specialized per block size and capped at 8 by shared-memory growth.
        if not 2 <= draft_token_num <= 8:
            return False
        if layer.bias is not None or layer.lower_bound is None:
            return False
        if (
            layer.head_q_dim != 128
            or layer.head_k_dim != 128
            or layer.head_v_dim != 128
        ):
            return False
        if (
            layer.num_q_heads != layer.num_k_heads
            or layer.num_k_heads != layer.num_v_heads
        ):
            return False
        if (
            mixed_qkv.dtype != torch.bfloat16
            or a.dtype != torch.bfloat16
            or b.dtype != torch.bfloat16
        ):
            return False
        if layer.conv_weights is None or tuple(layer.conv_weights.shape) != (
            layer.q_dim + layer.k_dim + layer.v_dim,
            4,
        ):
            return False
        if layer.conv_weights.dtype != torch.float32:
            return False
        if (
            ssm_states.dtype != torch.float32
            or tuple(ssm_states.shape[-3:])
            != (layer.num_v_heads, layer.head_v_dim, layer.head_k_dim)
            or tuple(ssm_states.stride()[-3:])
            != (
                layer.head_v_dim * layer.head_k_dim,
                layer.head_k_dim,
                1,
            )
            or ssm_states.stride(0) % 4 != 0
            or ssm_states.storage_offset() % 4 != 0
        ):
            return False
        if conv_states.shape[-2] != 3 or intermediate_conv_window_cache.shape[-2] != 3:
            return False
        # PLA verify writes raw-input rings, not per-token SSM snapshots.
        # Unsupported serving layouts keep the Triton path.
        if intermediate_state_cache is not None or replayssm_rawv is None:
            return False
        cache = self.state_layer_cache(layer.layer_id)
        rings = (
            replayssm_rawv,
            cache.replayssm_rawk,
            cache.replayssm_g,
            cache.replayssm_beta,
        )
        dtypes = (torch.bfloat16, torch.bfloat16, torch.float32, torch.float32)
        if any(t is None or t.dtype != dtype for t, dtype in zip(rings, dtypes)):
            return False
        if layer.A_log.dtype != torch.float32 or layer.dt_bias.dtype != torch.float32:
            return False
        if (
            conv_states.dtype != torch.bfloat16
            or intermediate_conv_window_cache.dtype != torch.bfloat16
        ):
            return False
        seq_len, dim = mixed_qkv.shape[0], layer.num_v_heads * 128
        if (
            mixed_qkv.shape[1] != 3 * dim
            or mixed_qkv.stride(-1) != 1
            or a.shape
            not in (
                (seq_len, dim),
                (1, seq_len, dim),
                (1, seq_len, layer.num_v_heads, 128),
            )
            or b.shape
            not in ((seq_len, layer.num_v_heads), (1, seq_len, layer.num_v_heads))
            or a.stride(-1) != 1
            or b.stride(-1) != 1
            or (a.ndim == 4 and a.stride(-2) != 128)
        ):
            return False
        # PLA reads/writes conv windows by stride; commit accepts either dense
        # axis order for conv state.
        if not intermediate_conv_window_cache.is_contiguous() and any(
            s <= 0 for s in intermediate_conv_window_cache.stride()
        ):
            return False
        return (
            conv_states.is_contiguous() or conv_states.transpose(-1, -2).is_contiguous()
        ) and all(t.is_contiguous() for t in (ssm_states, *rings))

    @staticmethod
    def _run_dspark_cutedsl_mtp(
        *,
        layer: RadixLinearAttention,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        intermediate_state_cache: torch.Tensor,
        intermediate_conv_window_cache: torch.Tensor,
        intermediate_state_indices: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        replayssm_rawv: Optional[torch.Tensor] = None,
        replayssm_rawk: Optional[torch.Tensor] = None,
        replayssm_g: Optional[torch.Tensor] = None,
        replayssm_beta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not sail_kda.pla_enabled():
            return KDAAttnBackend._run_dspark_cutedsl_mtp(
                layer=layer,
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                conv_states=conv_states,
                ssm_states=ssm_states,
                intermediate_state_cache=intermediate_state_cache,
                intermediate_conv_window_cache=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                replayssm_rawv=replayssm_rawv,
                replayssm_rawk=replayssm_rawk,
                replayssm_g=replayssm_g,
                replayssm_beta=replayssm_beta,
            )
        return sail_kda.decode_mtp(
            layer=layer,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            conv_states=conv_states,
            ssm_states=ssm_states,
            intermediate_state_cache=intermediate_state_cache,
            intermediate_conv_window_cache=intermediate_conv_window_cache,
            intermediate_state_indices=intermediate_state_indices,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            replayssm_rawv=replayssm_rawv,
            replayssm_rawk=replayssm_rawk,
            replayssm_g=replayssm_g,
            replayssm_beta=replayssm_beta,
        )
