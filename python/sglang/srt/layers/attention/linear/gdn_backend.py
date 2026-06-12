from typing import Dict, Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
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
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.utils import is_cpu, is_cuda, is_hip, is_npu
from sglang.srt.utils.common import rank0_log

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda() or is_hip():
    from sglang.jit_kernel.triton.gdn_fused_proj import fused_qkv_split_gdn_prefill

MAX_FUSED_QKV_SPLIT_DIM = 8192

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


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()

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

        # Replay after target verify can require multi-step varlen state updates.
        # Keep this path on Triton even if decode/verify use FlashInfer.
        self.replay_kernel = triton_kernel

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
        query_start_loc: Optional[torch.Tensor],
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
        return self.verify_kernel.target_verify(
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

    def state_update(
        self,
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
        input_token_indices: Optional[torch.Tensor] = None,
        input_sequence_indices: Optional[torch.Tensor] = None,
        input_sequence_lengths: Optional[torch.Tensor] = None,
        initial_state_indices: Optional[torch.Tensor] = None,
        input_token_start: int = 0,
        input_token_stride: int = 0,
        **kwargs,
    ) -> None:
        self.replay_kernel.state_update(
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            input_token_indices=input_token_indices,
            input_sequence_indices=input_sequence_indices,
            input_sequence_lengths=input_sequence_lengths,
            initial_state_indices=initial_state_indices,
            input_token_start=input_token_start,
            input_token_stride=input_token_stride,
            **kwargs,
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    supports_dflash_mamba_replay = True

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
        self.verify_intermediate_state_indices = torch.arange(
            self.req_to_token_pool.size, dtype=torch.int32, device=model_runner.device
        )
        self._verify_replay_inputs = {}
        self._verify_replay_inputs_by_graph_key = {}
        self._verify_replay_inputs_graph_static = False
        self._verify_replay_graph_key = None
        self._replay_req_indices_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}
        self._replay_tail_lengths_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}
        self._replay_invalid_mask_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}
        self._replay_tmp_mask_cache: Dict[Tuple[str, int, int], torch.Tensor] = {}

    @staticmethod
    def _replay_buffer_cache_key(
        device: torch.device, size: int
    ) -> Tuple[str, int, int]:
        return (device.type, -1 if device.index is None else device.index, size)

    def _get_replay_req_indices(self, size: int, device: torch.device) -> torch.Tensor:
        key = self._replay_buffer_cache_key(device, size)
        req_indices = self._replay_req_indices_cache.get(key)
        if req_indices is None:
            req_indices = torch.arange(size, dtype=torch.int64, device=device)
            self._replay_req_indices_cache[key] = req_indices
        return req_indices

    def _get_replay_tail_lengths(self, size: int, device: torch.device) -> torch.Tensor:
        key = self._replay_buffer_cache_key(device, size)
        tail_lengths = self._replay_tail_lengths_cache.get(key)
        if tail_lengths is None:
            tail_lengths = torch.empty((size,), dtype=torch.int32, device=device)
            self._replay_tail_lengths_cache[key] = tail_lengths
        return tail_lengths

    def _get_replay_invalid_mask(self, size: int, device: torch.device) -> torch.Tensor:
        key = self._replay_buffer_cache_key(device, size)
        invalid_mask = self._replay_invalid_mask_cache.get(key)
        if invalid_mask is None:
            invalid_mask = torch.empty((size,), dtype=torch.bool, device=device)
            self._replay_invalid_mask_cache[key] = invalid_mask
        return invalid_mask

    def _get_replay_tmp_mask(self, size: int, device: torch.device) -> torch.Tensor:
        key = self._replay_buffer_cache_key(device, size)
        tmp_mask = self._replay_tmp_mask_cache.get(key)
        if tmp_mask is None:
            tmp_mask = torch.empty((size,), dtype=torch.bool, device=device)
            self._replay_tmp_mask_cache[key] = tmp_mask
        return tmp_mask

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        self._verify_replay_graph_key = None
        if (
            forward_batch.forward_mode.is_target_verify()
            and not self._verify_replay_inputs_graph_static
        ):
            self._verify_replay_inputs.clear()
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
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        self._verify_replay_graph_key = None
        if not forward_batch.forward_mode.is_target_verify():
            return
        if in_capture:
            return
        if self._verify_replay_inputs_graph_static:
            self._verify_replay_graph_key = getattr(
                forward_batch, "cuda_graph_key", forward_batch.batch_size
            )
        else:
            self._verify_replay_inputs.clear()

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        super().init_forward_metadata_capture_cuda_graph(
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )
        self._verify_replay_graph_key = None

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        super().init_forward_metadata_replay_cuda_graph(
            bs,
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            encoder_lens,
            forward_mode,
            spec_info,
            seq_lens_cpu,
        )
        self._verify_replay_graph_key = bs if forward_mode.is_target_verify() else None

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
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            intermediate_state_indices = self.verify_intermediate_state_indices
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        actual_seq_len = mixed_qkv.shape[0]
        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
        if (is_cuda() or is_hip()) and qkv_dim <= MAX_FUSED_QKV_SPLIT_DIM:
            query, key, value = fused_qkv_split_gdn_prefill(
                mixed_qkv,
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [layer.q_dim, layer.k_dim, layer.v_dim],
                dim=-1,
            )
            query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
            key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
            value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            mamba_cache_steps = (
                forward_batch.spec_info.draft_token_num
                if getattr(forward_batch.spec_info, "mamba_cache_steps", None) is None
                else forward_batch.spec_info.mamba_cache_steps
            )
            if (
                getattr(forward_batch.spec_info, "mamba_replay", False)
                and mamba_cache_steps < forward_batch.spec_info.draft_token_num
            ):
                replay_inputs = (
                    key,
                    value,
                    a,
                    b,
                    layer.A_log,
                    layer.dt_bias,
                    forward_batch.spec_info.draft_token_num,
                )
                if (
                    torch.cuda.is_available()
                    and torch.cuda.is_current_stream_capturing()
                ):
                    self._verify_replay_inputs_graph_static = True
                    graph_key = getattr(forward_batch, "cuda_graph_key", batch_size)
                    self._verify_replay_inputs_by_graph_key.setdefault(graph_key, {})[
                        layer.layer_id
                    ] = replay_inputs
                else:
                    self._verify_replay_inputs[layer.layer_id] = replay_inputs
            cache_intermediate_states = int(mamba_cache_steps) > 0
            effective_cache_steps = mamba_cache_steps
            if not cache_intermediate_states and retrieve_parent_token is not None:
                raise RuntimeError(
                    "DFLASH Mamba cache_steps=0 replay does not support tree "
                    "target verify because parent recurrent states are not cached."
                )

            core_attn_out = self.kernel_dispatcher.target_verify(
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=(
                    intermediate_state_cache if cache_intermediate_states else None
                ),
                intermediate_state_indices=(
                    intermediate_state_indices if cache_intermediate_states else None
                ),
                cache_steps=effective_cache_steps,
                retrieve_parent_token=retrieve_parent_token,
            )
        else:
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )

            if (is_npu() or is_cpu()) and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        return core_attn_out

    def replay_mamba_state_after_verify(
        self,
        *,
        target_steps: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
        destination_state_indices: torch.Tensor,
        mamba_cache_steps: int,
        replay_all_requests: bool = False,
        replay_raw_requests: bool = False,
        initial_state_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """Replay GDN states from cached step K-1 or the pre-verify state.

        If K > 0, the caller must first scatter intermediate_ssm[K-1] into
        destination_state_indices for requests whose target step is >= K. If K == 0,
        replay starts from draft step 0. By default the destination slots are also
        the initial state slots; callers may pass initial_state_indices when replay
        should read from one cache slot and write the final state to another.
        """
        replay_inputs_by_layer = self._verify_replay_inputs
        if self._verify_replay_graph_key is not None:
            replay_inputs_by_layer = self._verify_replay_inputs_by_graph_key.get(
                self._verify_replay_graph_key, {}
            )

        if not replay_inputs_by_layer:
            raise RuntimeError(
                "DFLASH Mamba replay was requested, but no GDN replay inputs "
                "were captured during target verify."
            )

        _, _, _, _, _, _, draft_token_num = next(iter(replay_inputs_by_layer.values()))
        max_tail_len = int(draft_token_num) - int(mamba_cache_steps)
        if replay_all_requests or replay_raw_requests:
            request_number = target_steps.shape[0]
            req_indices = (
                self._get_replay_req_indices(request_number, target_steps.device)
                if replay_raw_requests
                else None
            )
            if (
                target_lengths is not None
                and int(mamba_cache_steps) == 0
                and not replay_raw_requests
            ):
                if target_lengths.shape[0] != request_number:
                    raise ValueError(
                        "target_lengths must match target_steps in GDN replay: "
                        f"{target_lengths.shape=} {target_steps.shape=}"
                    )
                tail_lengths = target_lengths
                if tail_lengths.dtype != torch.int32:
                    tail_lengths = tail_lengths.to(torch.int32)
                if not tail_lengths.is_contiguous():
                    tail_lengths = tail_lengths.contiguous()
            elif int(mamba_cache_steps) == 1 and not replay_raw_requests:
                # Dense K=1 replay starts after cached step 0, so the replay
                # length is exactly the final accepted step index.
                tail_lengths = target_steps
                if tail_lengths.dtype != torch.int32:
                    tail_lengths = tail_lengths.to(torch.int32)
                if not tail_lengths.is_contiguous():
                    tail_lengths = tail_lengths.contiguous()
            else:
                tail_lengths = self._get_replay_tail_lengths(
                    request_number, target_steps.device
                )
                torch.add(target_steps, 1 - int(mamba_cache_steps), out=tail_lengths)
                tail_lengths.clamp_(min=0, max=max_tail_len)
            if replay_raw_requests:
                invalid_mask = self._get_replay_invalid_mask(
                    request_number, target_steps.device
                )
                tmp_mask = self._get_replay_tmp_mask(
                    request_number, target_steps.device
                )
                torch.lt(target_steps, int(mamba_cache_steps), out=invalid_mask)
                torch.ge(target_steps, int(draft_token_num), out=tmp_mask)
                invalid_mask.logical_or_(tmp_mask)
                if initial_state_indices is not None:
                    torch.lt(initial_state_indices, 0, out=tmp_mask)
                    invalid_mask.logical_or_(tmp_mask)
                torch.lt(destination_state_indices, 0, out=tmp_mask)
                invalid_mask.logical_or_(tmp_mask)
                tail_lengths.masked_fill_(invalid_mask, 0)
            replay_state_indices = destination_state_indices
            replay_initial_state_indices = (
                destination_state_indices
                if initial_state_indices is None
                else initial_state_indices
            )
        else:
            tail_mask = (
                (target_steps >= int(mamba_cache_steps))
                & (target_steps < int(draft_token_num))
                & (destination_state_indices >= 0)
            )
            if initial_state_indices is not None:
                tail_mask = tail_mask & (initial_state_indices >= 0)
            req_indices = tail_mask.nonzero(as_tuple=True)[0]
            if req_indices.numel() == 0:
                return

            tail_lengths = (
                target_steps[req_indices].to(torch.int32) - int(mamba_cache_steps) + 1
            )
            replay_state_indices = destination_state_indices[req_indices]
            replay_initial_state_indices = (
                replay_state_indices
                if initial_state_indices is None
                else initial_state_indices[req_indices]
            )
        for layer_id, replay_inputs in replay_inputs_by_layer.items():
            key, value, a, b, A_log, dt_bias, draft_token_num = replay_inputs

            layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer_id)
            self.kernel_dispatcher.state_update(
                k=key,
                v=value,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                ssm_states=layer_cache.temporal,
                cache_indices=replay_state_indices,
                query_start_loc=None,
                input_sequence_indices=req_indices,
                input_sequence_lengths=tail_lengths,
                initial_state_indices=replay_initial_state_indices,
                input_token_start=int(mamba_cache_steps),
                input_token_stride=int(draft_token_num),
            )

    def clear_mamba_replay_inputs(self) -> None:
        if self._verify_replay_inputs_graph_static:
            self._verify_replay_graph_key = None
            return
        self._verify_replay_inputs.clear()
