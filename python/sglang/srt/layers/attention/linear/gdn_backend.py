from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_prefill import (
    GDNQKVShape,
    split_gdn_prefill_qkv,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

if TYPE_CHECKING:
    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        FlashInferGDNExtendPrep,
    )

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


class _Unbuilt:
    __slots__ = ()


_UNBUILT = _Unbuilt()


@dataclass(frozen=True, slots=True, eq=False, kw_only=True)
class GDNExtendMetadata:
    """Immutable values shared by every GDN layer in one extend forward."""

    has_initial_states: torch.Tensor
    no_prefix: bool
    needs_state_gather: bool
    source_cache_indices: torch.Tensor
    state_cache_indices: torch.Tensor
    source_state_pool_size: int
    state_pool_size: int
    flashinfer_prep: Optional[FlashInferGDNExtendPrep]


def maybe_set_default_flashinfer_gdn_prefill(model_runner: ModelRunner) -> None:
    """Use FlashInfer for the narrow SM100 GDN prefill domain we validated."""
    args = model_runner.server_args
    if (
        args.linear_attn_prefill_backend is not None
        or args.linear_attn_backend != "triton"
        or args.enable_page_major_kv_layout
        or not is_cuda()
        or torch.cuda.get_device_capability()[0] != 10
    ):
        return

    # Extra-buffer strategies need intermediate state checkpoints.
    if args.uses_mamba_radix_cache and args.mamba_radix_cache_strategy != "no_buffer":
        return

    cuda_version = torch.version.cuda
    chunk_size = args.chunked_prefill_size
    config = model_runner.hybrid_gdn_config
    if (
        cuda_version is None
        or int(cuda_version.split(".", 1)[0]) < 13
        or args.enable_dynamic_chunking
        or chunk_size is None
        or not 1 <= chunk_size <= 8192
        or getattr(config, "linear_key_head_dim", None) != 128
        or getattr(config, "linear_value_head_dim", None) != 128
        or model_runner.req_to_token_pool.mamba_pool.mamba_cache.temporal.dtype
        != torch.bfloat16
    ):
        return

    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        is_flashinfer_gdn_prefill_available,
    )

    if is_flashinfer_gdn_prefill_available():
        args.linear_attn_prefill_backend = "flashinfer"
        rank0_log("Defaulting SM100 GDN prefill backend to FlashInfer.")


def _qkv_shape_from_layer(layer: RadixLinearAttention) -> GDNQKVShape:
    return GDNQKVShape(
        num_q_heads=layer.num_q_heads,
        num_k_heads=layer.num_k_heads,
        num_v_heads=layer.num_v_heads,
        head_q_dim=layer.head_q_dim,
        head_k_dim=layer.head_k_dim,
        head_v_dim=layer.head_v_dim,
    )


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()
        self.tree_verify_kernel = triton_kernel

        cutedsl_kernel = None
        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            cutedsl_kernel = CuteDSLGDNKernel()
            self.decode_kernel = cutedsl_kernel
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            # Reuse the CuteDSL kernel if already created for decode
            if cutedsl_kernel is None:
                from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                    CuteDSLGDNKernel,
                )

                cutedsl_kernel = CuteDSLGDNKernel()
            # The CuteDSL prefill kernel only exists on SM100+ (Blackwell).
            # On SM90 (Hopper) fall back to Triton so users can pick
            # `cutedsl` uniformly across hardware.
            if cutedsl_kernel.supports_prefill:
                self.extend_kernel = cutedsl_kernel
            else:
                rank0_log(
                    "CuTe DSL GDN prefill is not supported on this GPU "
                    "(requires SM100+). Falling back to Triton for prefill."
                )
                self.extend_kernel = triton_kernel
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        self.uses_flashinfer_prefill = prefill_backend.is_flashinfer()

        # Verify kernel: use FlashInfer when the selected FlashInfer kernel
        # supports MTP verify. SM90 uses the fp32-state path; SM100 uses the
        # bf16-state adapter in FlashInferGDNKernel.
        if (
            decode_backend.is_flashinfer() or prefill_backend.is_flashinfer()
        ) and flashinfer_kernel.supports_target_verify:
            self.verify_kernel = flashinfer_kernel
        else:
            self.verify_kernel = triton_kernel

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if
        the decode kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def build_flashinfer_extend_prep(
        self,
        *,
        cache_indices: torch.Tensor,
        state_pool_size: int,
    ) -> FlashInferGDNExtendPrep:
        assert self.uses_flashinfer_prefill
        return self.extend_kernel.build_extend_prep(
            cache_indices=cache_indices,
            state_pool_size=state_pool_size,
        )

    def extend_prefill_from_packed(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        shape: GDNQKVShape,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        flashinfer_prep: Optional[FlashInferGDNExtendPrep] = None,
        no_prefix: bool = False,
    ) -> tuple:
        """Run GDN prefill without exposing backend tensor representations."""
        if self.uses_flashinfer_prefill and self.extend_kernel.can_use_fused_prefill(
            mixed_qkv, shape
        ):
            return self.extend_kernel.extend_fused(
                mixed_qkv,
                a,
                b,
                shape=shape,
                A_log=A_log,
                dt_bias=dt_bias,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                out=out,
                prep=flashinfer_prep,
                no_prefix=no_prefix,
            )

        query, key, value = split_gdn_prefill_qkv(mixed_qkv, shape)
        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
        return self.extend(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # FlashInfer verify supports a linear MTP chain. Tree-shaped drafts
        # carry parent indices and must use Triton even when decode/prefill use
        # FlashInfer.
        verify_kernel = (
            self.tree_verify_kernel
            if kwargs.get("retrieve_parent_token") is not None
            else self.verify_kernel
        )
        return verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert (
                self.conv_states_shape[-1] < FLA_CHUNK_SIZE
            ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)
        self._extend_metadata: GDNExtendMetadata | _Unbuilt = _UNBUILT
        self.verify_intermediate_state_indices = torch.arange(
            self.req_to_token_pool.size, dtype=torch.int32, device=model_runner.device
        )

    def _reset_extend_metadata(self) -> None:
        self._extend_metadata = _UNBUILT

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        self._reset_extend_metadata()
        if self.forward_metadata.has_mamba_track_mask:
            self.forward_metadata.mamba_track_mask_indices = (
                forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
            )
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[
                    self.forward_metadata.mamba_track_mask_indices
                ]
            )

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        super().init_forward_metadata_out_graph(forward_batch, in_capture)
        self._reset_extend_metadata()

    def init_forward_metadata_capture_cpu_graph(self, *args, **kwargs):
        super().init_forward_metadata_capture_cpu_graph(*args, **kwargs)
        self._reset_extend_metadata()

    def _get_or_build_extend_metadata(
        self,
        forward_batch: ForwardBatch,
        *,
        needs_state_gather: bool,
        cache_indices: torch.Tensor,
        source_state_pool_size: int,
    ) -> GDNExtendMetadata:
        metadata = self._extend_metadata
        state_pool_size = (
            cache_indices.shape[0] if needs_state_gather else source_state_pool_size
        )
        if metadata is _UNBUILT:
            state_cache_indices = (
                torch.arange(
                    cache_indices.shape[0],
                    device=cache_indices.device,
                    dtype=cache_indices.dtype,
                )
                if needs_state_gather
                else cache_indices
            )
            flashinfer_prep = (
                self.kernel_dispatcher.build_flashinfer_extend_prep(
                    cache_indices=state_cache_indices,
                    state_pool_size=state_pool_size,
                )
                if self.kernel_dispatcher.uses_flashinfer_prefill
                else None
            )
            metadata = GDNExtendMetadata(
                has_initial_states=forward_batch.extend_prefix_lens > 0,
                no_prefix=(
                    forward_batch.extend_prefix_lens_cpu is not None
                    and not any(forward_batch.extend_prefix_lens_cpu)
                ),
                needs_state_gather=needs_state_gather,
                source_cache_indices=cache_indices,
                state_cache_indices=state_cache_indices,
                source_state_pool_size=source_state_pool_size,
                state_pool_size=state_pool_size,
                flashinfer_prep=flashinfer_prep,
            )
            self._extend_metadata = metadata
            return metadata

        assert isinstance(metadata, GDNExtendMetadata)
        assert metadata.needs_state_gather == needs_state_gather
        assert metadata.source_cache_indices.shape == cache_indices.shape
        assert metadata.source_cache_indices.dtype == cache_indices.dtype
        assert metadata.source_cache_indices.device == cache_indices.device
        assert metadata.source_cache_indices.data_ptr() == cache_indices.data_ptr()
        assert metadata.source_state_pool_size == source_state_pool_size
        assert metadata.state_pool_size == state_pool_size
        return metadata

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices
        # GDN ReplaySSM (slice 1a): per-layer ring slices + the once-per-forward
        # per-row write cursor. All None unless --enable-linear-replayssm, so the
        # legacy dispatch below is byte-identical when the flag is off.
        replayssm_write_pos = self.forward_metadata.replayssm_write_pos
        # GDN ReplaySSM (slice 2b): per-row force-flush at radix track
        # boundaries (None unless --enable-linear-replayssm). When present the
        # kernel folds the ring into temporal[slot] on the snapshot steps.
        replayssm_force_flush = self.forward_metadata.replayssm_force_flush
        replayssm_d = layer_cache.replayssm_d
        replayssm_k = layer_cache.replayssm_k
        replayssm_g = layer_cache.replayssm_g

        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )

        # Skip split + reshape + separate gating kernel by consuming
        # the packed mixed_qkv directly in a single fused Triton kernel.
        if self.kernel_dispatcher.supports_packed_decode:
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
                replayssm_d=replayssm_d,
                replayssm_k=replayssm_k,
                replayssm_g=replayssm_g,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices
            )
            return core_attn_out

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        if forward_batch.forward_mode.is_target_verify():
            return self._forward_target_verify(
                layer=layer, forward_batch=forward_batch, mixed_qkv=mixed_qkv, a=a, b=b
            )
        return self._forward_extend_tokens(
            layer=layer,
            forward_batch=forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            out=out,
        )

    def _forward_target_verify(
        self,
        *,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ):
        seq_len = mixed_qkv.shape[0]
        forward_metadata = self.forward_metadata
        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
        conv_states = mamba_cache_params.conv[0]
        intermediate_state_indices = self.verify_intermediate_state_indices

        batch_size = seq_len // forward_batch.spec_info.draft_token_num
        mixed_qkv_reshaped = mixed_qkv.view(
            batch_size, forward_batch.spec_info.draft_token_num, -1
        ).transpose(1, 2)
        mixed_qkv_processed = causal_conv1d_update(
            mixed_qkv_reshaped,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices[:batch_size],
            intermediate_conv_window=mamba_cache_params.intermediate_conv_window[0],
            intermediate_state_indices=intermediate_state_indices[:batch_size],
            retrieve_next_token=forward_metadata.retrieve_next_token,
            retrieve_next_sibling=forward_metadata.retrieve_next_sibling,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
        )
        mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)

        query, key, value = split_gdn_prefill_qkv(
            mixed_qkv, _qkv_shape_from_layer(layer)
        )
        return self.kernel_dispatcher.target_verify(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            ssm_states=mamba_cache_params.temporal,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            intermediate_states_buffer=mamba_cache_params.intermediate_ssm,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=forward_batch.spec_info.draft_token_num,
            retrieve_parent_token=forward_metadata.retrieve_parent_token,
        )

    def _forward_extend_tokens(
        self,
        *,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        out: Optional[torch.Tensor],
    ):
        seq_len = mixed_qkv.shape[0]
        forward_metadata = self.forward_metadata
        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal

        # Page-major envelope: the prefill kernels (CUDA causal_conv1d_fwd,
        # chunk_gated_delta_rule) write state back in place assuming a contiguous
        # slot layout, so they silently drop the write to the strided envelope
        # pool. Run them on contiguous per-sequence copies (identity-indexed) and
        # scatter the result back. No-op for the default contiguous pool.
        # TODO(ch-wan): drop these .contiguous() copies by making the prefill conv
        # and chunk_gated_delta_rule kernels honor the pool's real slot stride +
        # int64 indexing, like packed_decode / causal_conv1d_update already do.
        needs_state_gather = (
            not conv_states.is_contiguous() or not ssm_states.is_contiguous()
        )
        extend_metadata = self._get_or_build_extend_metadata(
            forward_batch,
            needs_state_gather=needs_state_gather,
            cache_indices=cache_indices,
            source_state_pool_size=ssm_states.shape[0],
        )
        state_cache_indices = extend_metadata.state_cache_indices
        if needs_state_gather:
            conv_states_contig = conv_states[cache_indices].contiguous()
            ssm_states_contig = ssm_states[cache_indices].contiguous()
        else:
            conv_states_contig = conv_states
            ssm_states_contig = ssm_states

        mixed_qkv = mixed_qkv.transpose(0, 1)
        if forward_metadata.has_mamba_track_mask:
            mixed_qkv_to_track = mixed_qkv[
                :, forward_metadata.track_conv_indices
            ].transpose(0, 1)
            conv_states[forward_metadata.conv_states_mask_indices] = mixed_qkv_to_track

        mixed_qkv = causal_conv1d_fn(
            mixed_qkv,
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states_contig,
            has_initial_state=extend_metadata.has_initial_states,
            cache_indices=state_cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)[:seq_len]

        qkv_shape = _qkv_shape_from_layer(layer)

        core_attn_out, last_recurrent_state, h = (
            self.kernel_dispatcher.extend_prefill_from_packed(
                mixed_qkv,
                a,
                b,
                shape=qkv_shape,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                ssm_states=ssm_states_contig,
                cache_indices=state_cache_indices,
                query_start_loc=query_start_loc,
                out=out,
                flashinfer_prep=extend_metadata.flashinfer_prep,
                no_prefix=extend_metadata.no_prefix,
            )
        )

        if is_npu() and last_recurrent_state is not None:
            last_recurrent_state = last_recurrent_state.to(ssm_states.dtype, copy=False)
            ssm_states[cache_indices] = last_recurrent_state

        if needs_state_gather:
            # Scatter the in-place-updated contiguous copies back to the strided
            # envelope pool (advanced indexing handles the strides).
            conv_states[cache_indices] = conv_states_contig
            ssm_states[cache_indices] = ssm_states_contig

        if h is not None:
            self._track_mamba_state_extend(
                forward_batch, h, ssm_states, forward_metadata
            )

        return core_attn_out
