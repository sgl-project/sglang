from dataclasses import astuple, dataclass
from functools import lru_cache
from typing import Optional, Union

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.fused_recurrent import (
    fused_recurrent_gated_delta_rule_update,
)
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen3_next import Qwen3HybridLinearDecoderLayer, fused_gdn_gating
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput


@dataclass
class ForwardMetadata:
    query_start_loc: Optional[torch.Tensor]
    mamba_cache_indices: torch.Tensor


class MambaAttnBackend(AttentionBackend):
    """Attention backend using Mamba kernel."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.pad_slot_id = -1  # Default pad slot id
        self.device = model_runner.device
        self.req_to_token_pool: HybridReqToTokenPool = model_runner.req_to_token_pool
        self.forward_metadata: ForwardMetadata = None
        self.state_indices_list = []
        self.query_start_loc_list = []

    @classmethod
    @lru_cache(maxsize=128)
    def _get_cached_arange(cls, bs: int, device_str: str) -> torch.Tensor:
        """Cache torch.arange tensors for common batch sizes to avoid repeated allocation."""
        device = torch.device(device_str)
        return torch.arange(0, bs + 1, dtype=torch.int32, device=device)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        if forward_batch.forward_mode.is_decode_or_idle():
            query_start_loc = self._get_cached_arange(bs, str(self.device))
        elif forward_batch.forward_mode.is_extend():
            if forward_batch.forward_mode.is_target_verify():
                query_start_loc = torch.arange(
                    0,
                    forward_batch.input_ids.shape[0] + 1,
                    step=forward_batch.spec_info.draft_token_num,
                    dtype=torch.int32,
                    device=forward_batch.input_ids.device,
                )
            else:
                query_start_loc = torch.empty(
                    (bs + 1,), dtype=torch.int32, device=self.device
                )
                query_start_loc[:bs] = forward_batch.extend_start_loc
                query_start_loc[bs] = (
                    forward_batch.extend_start_loc[-1]
                    + forward_batch.extend_seq_lens[-1]
                )
        else:
            raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode=}")
        mamba_cache_indices = self.req_to_token_pool.get_mamba_indices(
            forward_batch.req_pool_indices
        )
        self.forward_metadata = ForwardMetadata(
            query_start_loc=query_start_loc,
            mamba_cache_indices=mamba_cache_indices,
        )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for i in range(max_bs):
            self.state_indices_list.append(
                torch.full((i + 1,), self.pad_slot_id, dtype=torch.int32, device="cuda")
            )
            self.query_start_loc_list.append(
                torch.empty((i + 2,), dtype=torch.int32, device="cuda")
            )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(self._get_cached_arange(bs, "cuda"))
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                torch.arange(
                    0,
                    bs * spec_info.draft_token_num + 1,
                    step=spec_info.draft_token_num,
                    dtype=torch.int32,
                    device="cuda",
                )
            )
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        self.forward_metadata = ForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_cache_indices=self.state_indices_list[bs - 1],
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        num_padding = torch.count_nonzero(
            seq_lens_cpu == self.get_cuda_graph_seq_len_fill_value()
        )
        # Make sure forward metadata is correctly handled for padding reqs
        req_pool_indices[bs - num_padding :] = 0
        mamba_indices = self.req_to_token_pool.get_mamba_indices(req_pool_indices)
        mamba_indices[bs - num_padding :] = -1
        self.state_indices_list[bs - 1][: len(mamba_indices)].copy_(mamba_indices)
        if forward_mode.is_decode_or_idle():
            self.query_start_loc_list[bs - 1].copy_(self._get_cached_arange(bs, "cuda"))
            if num_padding > 0:
                self.query_start_loc_list[bs - 1][bs - num_padding :] = bs - num_padding
        elif forward_mode.is_target_verify():
            self.query_start_loc_list[bs - 1].copy_(
                torch.arange(
                    0,
                    bs * spec_info.draft_token_num + 1,
                    step=spec_info.draft_token_num,
                    dtype=torch.int32,
                    device="cuda",
                )
            )
            if num_padding > 0:
                self.query_start_loc_list[bs - 1][bs - num_padding :] = (
                    bs - num_padding
                ) * spec_info.draft_token_num
        else:
            raise ValueError(f"Invalid forward mode: {forward_mode=}")

        self.forward_metadata = ForwardMetadata(
            query_start_loc=self.query_start_loc_list[bs - 1],
            mamba_cache_indices=self.state_indices_list[bs - 1],
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1  # Mamba attn does not use seq lens to index kv cache

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]

        conv_states, ssm_states, *rest = self.req_to_token_pool.get_mamba_params(
            layer_id
        )
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            conv_weights,
            bias,
            activation,
            conv_state_indices=cache_indices,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim // attn_tp_size,
                key_dim // attn_tp_size,
                value_dim // attn_tp_size,
            ],
            dim=-1,
        )
        # Reshape from [l, h*d] to [1, l, h, d]
        seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        query = query.view(1, seq_len, num_heads, head_k_dim)
        key = key.view(1, seq_len, num_heads, head_k_dim)
        value = value.view(1, seq_len, value.shape[1] // head_v_dim, head_v_dim)

        core_attn_out = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )

        return core_attn_out

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        mixed_qkv = kwargs["mixed_qkv"]
        conv_weights = kwargs["conv_weights"]
        bias = kwargs["bias"]
        activation = kwargs["activation"]
        key_dim = kwargs["key_dim"]
        value_dim = kwargs["value_dim"]
        attn_tp_size = kwargs["attention_tp_size"]
        head_k_dim = kwargs["head_k_dim"]
        head_v_dim = kwargs["head_v_dim"]
        a = kwargs["a"]
        b = kwargs["b"]
        A_log = kwargs["A_log"]
        dt_bias = kwargs["dt_bias"]
        layer_id = kwargs["layer_id"]
        seq_len = kwargs["seq_len"]

        is_target_verify = forward_batch.forward_mode.is_target_verify()

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        if is_target_verify:
            (
                conv_states,
                ssm_states,
                intermediate_state_cache,
                intermediate_conv_window_cache,
            ) = self.req_to_token_pool.get_mamba_params(layer_id)
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
            conv_states_to_use = conv_states.clone()
        else:
            conv_states, ssm_states, *rest = self.req_to_token_pool.get_mamba_params(
                layer_id
            )
            has_initial_states = forward_batch.extend_prefix_lens > 0
            conv_states_to_use = conv_states

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = (
                mixed_qkv.view(batch_size, draft_token_num, -1)
                .transpose(1, 2)
                .contiguous()
            )
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states_to_use,
                conv_weights,
                bias,
                activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
            )
            mixed_qkv = (
                mixed_qkv_processed.transpose(1, 2).contiguous().view(seq_len, -1)
            )
        else:
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv.transpose(0, 1),
                conv_weights,
                bias,
                activation=activation,
                conv_states=conv_states_to_use,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            ).transpose(0, 1)[:seq_len]

        key_split_dim = key_dim // attn_tp_size
        value_split_dim = value_dim // attn_tp_size

        query, key, value = torch.split(
            mixed_qkv,
            [key_split_dim, key_split_dim, value_split_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        num_value_heads = value.shape[1] // head_v_dim

        query = query.view(1, actual_seq_len, num_heads, head_k_dim)
        key = key.view(1, actual_seq_len, num_heads, head_k_dim)
        value = value.view(1, actual_seq_len, num_value_heads, head_v_dim)

        beta = b.sigmoid()
        g = fused_gdn_gating(A_log, a, dt_bias)

        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)

        if is_target_verify:
            core_attn_out = fused_recurrent_gated_delta_rule_update(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                cu_seqlens=query_start_loc,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                intermediate_states_buffer=intermediate_state_cache,
                cache_steps=forward_batch.spec_info.draft_token_num,
            )
        else:
            recurrent_state = ssm_states[cache_indices]
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                cu_seqlens=query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            last_recurrent_state = last_recurrent_state.to(ssm_states.dtype, copy=False)
            ssm_states[cache_indices] = last_recurrent_state

        return core_attn_out


class HybridLinearAttnBackend(AttentionBackend):
    """Support different backends for prefill and decode."""

    def __init__(
        self,
        full_attn_backend: AttentionBackend,
        linear_attn_backend: AttentionBackend,
        full_attn_layers: list[int],
    ):
        self.full_attn_layers = full_attn_layers
        self.attn_backend_list = [full_attn_backend, linear_attn_backend]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        for attn_backend in self.attn_backend_list:
            attn_backend.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.attn_backend_list[0].get_cuda_graph_seq_len_fill_value()

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.attn_backend_list[0].forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.attn_backend_list[1].forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        layer_id = layer.layer_id if layer else kwargs["layer_id"]
        if layer_id in self.full_attn_layers:
            return self.attn_backend_list[0].forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        return self.attn_backend_list[1].forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Run forward on an attention layer."""
        if forward_batch.forward_mode.is_idle():
            if layer is None:
                return torch.empty_like(kwargs["z"])
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

    def update_mamba_state_after_mtp_verify(self, accepted_length, model):
        request_number = accepted_length.shape[0]

        state_indices_tensor = self.attn_backend_list[
            1
        ].forward_metadata.mamba_cache_indices[:request_number]

        mamba_caches = self.attn_backend_list[
            1
        ].req_to_token_pool.get_mamba_params_all_layers()

        (
            conv_states,
            ssm_states,
            intermediate_state_cache,
            intermediate_conv_window_cache,
        ) = mamba_caches

        # SSM state updates (chunked to reduce peak memory)
        valid_mask = accepted_length > 0

        # Compute common indices once to avoid duplication
        last_steps_all = (accepted_length - 1).to(torch.int64)
        valid_state_indices = state_indices_tensor[valid_mask].to(torch.int64)
        last_steps = last_steps_all[valid_mask].to(torch.int64)

        if valid_state_indices.numel() > 0:
            chunk = 256
            num_valid = valid_state_indices.numel()

            # SSM state updates
            for i in range(0, num_valid, chunk):
                idx = valid_state_indices[i : i + chunk]
                steps = last_steps[i : i + chunk]
                # per (cache line, step)
                for j in range(idx.numel()):
                    ci = idx[j].item()
                    st = steps[j].item()
                    ssm_states[:, ci, :].copy_(
                        intermediate_state_cache[:, ci, st].to(
                            ssm_states.dtype, copy=False
                        )
                    )

            # Conv window updates
            for i in range(0, num_valid, chunk):
                idx = valid_state_indices[i : i + chunk]
                steps = last_steps[i : i + chunk]
                for j in range(idx.numel()):
                    ci = idx[j].item()
                    st = steps[j].item()
                    conv_states[:, ci, :, :].copy_(
                        intermediate_conv_window_cache[:, ci, st].to(
                            conv_states.dtype, copy=False
                        )
                    )
