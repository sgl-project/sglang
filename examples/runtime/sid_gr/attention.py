"""Reference attention using native SGLang KV pools and SID arithmetic.

Qualified for sequential requests, one beam group per batch, BF16 and page size 1.
This eager reference implementation intentionally gathers logical KV histories.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.srt.arg_groups.choices import (
    ATTENTION_BACKEND_CHOICES,
    add_attention_backend_choices,
)
from sglang.srt.layers.attention.attention_registry import register_attention_backend
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.runtime_context import get_parallel

from .gr_kernel import beam_decode_with_reference_splits, load_gr_kernel

BACKEND = "sid_gr_reference"


@dataclass
class ReferenceMetadata:
    context: int
    steps: int = 0
    indices: torch.Tensor | None = None


class SIDReferenceAttentionBackend(AttentionBackend):
    def __init__(self, runner):
        self.req_to_token_pool = runner.req_to_token_pool
        self.token_to_kv_pool = runner.token_to_kv_pool
        self.data_type = runner.dtype
        self.kv_cache_dtype = runner.kv_cache_dtype
        if self.data_type != torch.bfloat16 or self.kv_cache_dtype != torch.bfloat16:
            raise ValueError("SID reference attention requires BF16 model and KV")
        self._contexts = {}
        self.beam_decode_attn = load_gr_kernel()
        self.tp_size = get_parallel().attn_tp_size
        torch.backends.cuda.matmul.allow_tf32 = False

    def init_forward_metadata(self, batch):
        if batch.forward_mode.is_extend():
            if batch.extend_prefix_lens_cpu != [0]:
                raise ValueError("SID reference requires one complete uncached prompt")
            context = batch.extend_seq_lens_cpu[0]
            if context < 1:
                raise ValueError("SID reference requires a nonempty prompt")
            # Use the first live KV slot as a prompt identity. Slot reuse updates
            # this bounded map, including after cancelled/completed requests.
            self._contexts[int(batch.out_cache_loc[0].item())] = context
            self.forward_metadata = ReferenceMetadata(context)
        elif batch.forward_mode.is_decode():
            lengths = batch.seq_lens_cpu
            if lengths is None:
                lengths = batch.seq_lens.cpu()
            lengths = lengths.tolist()
            if not lengths or len(set(lengths)) != 1:
                raise ValueError("SID reference requires one equal-length beam group")
            indices = self.req_to_token_pool.req_to_token[
                batch.req_pool_indices.long(), : lengths[0]
            ].long()
            first_slots = indices[:, 0]
            if not torch.equal(first_slots, first_slots[:1].expand_as(first_slots)):
                raise ValueError("SID reference supports only one beam group per batch")
            context = self._contexts[int(first_slots[0].item())]
            steps = lengths[0] - context
            if steps < 1:
                raise ValueError("Decode must follow a recorded prompt")
            if not torch.equal(
                indices[:, :context], indices[:1, :context].expand(len(lengths), -1)
            ):
                raise ValueError("Beam rows must share the same prompt KV")
            self.forward_metadata = ReferenceMetadata(context, steps, indices)
        else:
            raise ValueError("SID reference supports eager prefill and decode only")

    def _write_kv(self, k, v, layer, batch, save_kv_cache):
        if k is None or v is None or layer.is_cross_attention or layer.logit_cap:
            raise ValueError("SID reference requires ordinary self-attention Q/K/V")
        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc.for_layer(batch, layer), k, v
            )

    def forward_extend(self, q, k, v, layer, batch, save_kv_cache=True, **kwargs):
        self._write_kv(k, v, layer, batch, save_kv_cache)
        head_dim = layer.head_dim
        query = q.reshape(-1, layer.tp_q_head_num, head_dim).transpose(0, 1)[None]
        key = k.reshape(-1, layer.tp_k_head_num, head_dim).transpose(0, 1)[None]
        value = v.reshape(-1, layer.tp_v_head_num, head_dim).transpose(0, 1)[None]
        groups = layer.tp_q_head_num // layer.tp_k_head_num
        with sdpa_kernel(SDPBackend.MATH):
            output = F.scaled_dot_product_attention(
                query,
                key.repeat_interleave(groups, dim=1),
                value.repeat_interleave(groups, dim=1),
                dropout_p=0.0,
                is_causal=True,
                scale=layer.scaling,
            )
        return output.squeeze(0).transpose(0, 1).reshape_as(q)

    def forward_decode(self, q, k, v, layer, batch, save_kv_cache=True, **kwargs):
        self._write_kv(k, v, layer, batch, save_kv_cache)
        meta = self.forward_metadata
        indices = meta.indices
        width = indices.shape[0]
        heads, kv_heads, dim = layer.tp_q_head_num, layer.tp_k_head_num, layer.head_dim
        key_pool, value_pool = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        context_indices = indices[:1, : meta.context]
        generated_indices = indices[:, meta.context :]
        key_context = key_pool[context_indices].contiguous()
        value_context = value_pool[context_indices].contiguous()

        # Repack selected ancestor histories in step-major order. Native SGLang
        # owns selection/reparenting; the kernel sees an identity path in this
        # gathered view, even when several beams share the same physical slots.
        def gather(pool):
            return (
                pool[generated_indices]
                .transpose(0, 1)
                .reshape(1, meta.steps * width, kv_heads, dim)
                .contiguous()
            )

        paths = (
            torch.arange(meta.steps * width, device=q.device, dtype=torch.int32)
            .reshape(1, 1, 1, meta.steps, width)
            .expand(1, 1, heads, meta.steps, width)
            .contiguous()
        )
        if self.tp_size == 1:
            kernel, options = self.beam_decode_attn, {"backend": "dsl"}
        else:
            kernel = beam_decode_with_reference_splits
            options = {"reference_heads": heads * self.tp_size}
        output = kernel(
            q.reshape(1, 1, width, heads, dim).contiguous(),
            key_context,
            value_context,
            gather(key_pool),
            gather(value_pool),
            paths,
            meta.steps,
            **options,
        )
        if isinstance(output, tuple):
            output = output[0]
        return output.reshape_as(q)


def register_reference_backend():
    if BACKEND not in ATTENTION_BACKEND_CHOICES:
        add_attention_backend_choices([BACKEND])
    register_attention_backend(BACKEND)(SIDReferenceAttentionBackend)
