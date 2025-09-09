from dataclasses import astuple, dataclass
from typing import Optional, Union

from einops import rearrange
import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule_update
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.models.qwen3_hybrid_moe import fused_gdn_gating
from sglang.srt.layers.attention.mamba.ops.causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
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
        self.req_to_token_pool: HybridReqToTokenPool = (
            model_runner.req_to_token_pool
        )
        self.forward_metadata: ForwardMetadata = None
        self.state_indices_list = []
        self.query_start_loc_list = []

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        query_start_loc = torch.zeros(
            (bs + 1,), dtype=torch.int32, device=self.device
        )
        if forward_batch.forward_mode.is_decode_or_idle():
            query_start_loc = torch.arange(
                0, bs + 1, dtype=torch.int32, device=self.device
            )
        elif forward_batch.forward_mode.is_prefill():
            query_start_loc[1 : bs + 1] = torch.cumsum(
                forward_batch.extend_seq_lens, dim=0
            )
        else:
            raise NotImplementedError("Only prefill/decode/idle modes are supported.")
        mamba_cache_indices = (
            self.req_to_token_pool.get_mamba_indices(
                forward_batch.req_pool_indices
            )
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
        mamba_indices = self.req_to_token_pool.get_mamba_indices(
            req_pool_indices
        )
        self.state_indices_list[bs - 1][:len(mamba_indices)].copy_(mamba_indices)
        self.query_start_loc_list[bs - 1].copy_(
            torch.arange(0, bs + 1, dtype=torch.int32, device="cuda")
        )
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
        mamba_indices = self.req_to_token_pool.get_mamba_indices(
            req_pool_indices
        )
        self.state_indices_list[bs - 1][:len(mamba_indices)].copy_(mamba_indices)
        self.query_start_loc_list[bs - 1].copy_(
            torch.arange(0, bs + 1, dtype=torch.int32, device="cuda")
        )
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

        conv_states, ssm_states = self.req_to_token_pool.get_mamba_params(layer_id)
        query_start_loc, cache_indices = astuple(self.forward_metadata)

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
        query, key = map(
            lambda x: rearrange(x, "l (h d) -> 1 l h d", d=head_k_dim),
            (query, key),
        )
        value = rearrange(value, "l (h d) -> 1 l h d", d=head_v_dim)
        beta = b.sigmoid()
        g = fused_gdn_gating(A_log, a, dt_bias)
        g, beta = map(lambda x: rearrange(x, "l  d -> 1 l d"), (g, beta))
        core_attn_out = fused_recurrent_gated_delta_rule_update(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            # head_first=False,
            use_qk_l2norm_in_kernel=True,
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

        conv_states, ssm_states = self.req_to_token_pool.get_mamba_params(layer_id)
        has_initial_states = forward_batch.extend_prefix_lens > 0

        query_start_loc, cache_indices = astuple(self.forward_metadata)

        mixed_qkv = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            conv_weights,
            bias,
            activation=activation,
            conv_states=conv_states,
            has_initial_state=has_initial_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        ).transpose(0, 1)[:seq_len]
        
        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim // attn_tp_size,
                key_dim // attn_tp_size,
                value_dim // attn_tp_size,
            ],
            dim=-1,
        )
        query, key = map(
            lambda x: rearrange(x, "l (h d) -> 1 l h d", d=head_k_dim),
            (query, key),
        )
        value = rearrange(value, "l (h d) -> 1 l h d", d=head_v_dim)
        beta = b.sigmoid()
        g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
        g, beta = map(lambda x: rearrange(x, "l  d -> 1 l d"), (g, beta))
        recurrent_state = ssm_states[self.forward_metadata.mamba_cache_indices]
        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            initial_state=None,  # NOTE: In my view, the initial state is not used in training and prefill stage.
            output_final_state=recurrent_state is not None,
            cu_seqlens=query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
        )

        last_recurrent_state = last_recurrent_state.to(ssm_states.dtype, copy=False)
        ssm_states[self.forward_metadata.mamba_cache_indices] = last_recurrent_state
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
            if q is None:
                mixed_qkv = kwargs["mixed_qkv"]
                key_dim = kwargs["key_dim"]
                value_dim = kwargs["value_dim"]
                attn_tp_size = kwargs["attention_tp_size"]
                _, _, value = torch.split(
                    mixed_qkv,
                    [
                        key_dim // attn_tp_size,
                        key_dim // attn_tp_size,
                        value_dim // attn_tp_size,
                    ],
                    dim=-1,
                )
                return torch.empty_like(value)
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

