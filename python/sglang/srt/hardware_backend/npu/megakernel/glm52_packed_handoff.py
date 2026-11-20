"""Bind SGLang-owned packed MLA/indexer storage to BlockRT's GLM-5.2 ABI."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from sglang.srt.model_executor.forward_context import (
    get_req_to_token_pool,
    get_token_to_kv_pool,
)


@dataclass
class _SharedFp8Cache:
    k_cache: list[torch.Tensor]
    index_k_buffer: list[torch.Tensor]
    index_k_scale_buffer: list[torch.Tensor]


@dataclass
class _DecodeBuffers:
    input_ids: torch.Tensor
    positions: torch.Tensor
    block_tables: torch.Tensor
    slot_mapping: torch.Tensor
    kv_seq_len: torch.Tensor
    q_seq_len: torch.Tensor
    mla_cos_cache: torch.Tensor
    mla_sin_cache: torch.Tensor
    previous_hidden_states: torch.Tensor | None = None
    prev_dsa_topk_indices: torch.Tensor | None = None


class Glm52PackedDecodeHandoff:
    def __init__(self, model):
        self.model = model
        self._shared_cache = None
        self._shared_pool_id = None
        self._buffers_by_signature = {}

    def _get_first_attention(self):
        inner_model = self.model.model
        if hasattr(inner_model, "decoder"):
            attention = inner_model.decoder.self_attn
            if attention.rotary_emb is None or not hasattr(attention, "indexer"):
                raise RuntimeError("Both MLA and DSA rotary caches are required.")
            return attention

        expected_layers = int(self.model.config.num_hidden_layers)
        if inner_model.start_layer != 0 or inner_model.end_layer != expected_layers:
            raise RuntimeError(
                "GLM-5.2 MegaKernel packed handoff requires pp_size=1: "
                f"start_layer={inner_model.start_layer}, "
                f"end_layer={inner_model.end_layer}, expected={expected_layers}."
            )
        attention = inner_model.layers[0].self_attn
        if attention.rotary_emb is None or not hasattr(attention, "indexer"):
            raise RuntimeError("Both MLA and DSA rotary caches are required.")
        return attention

    @staticmethod
    def _expanded_cos_sin(rotary_emb, positions: torch.Tensor):
        raw = rotary_emb.cos_sin_cache.index_select(0, positions.reshape(-1))
        cos, sin = raw.chunk(2, dim=-1)
        # BlockRT's GLM-5.2 MLA RoPE consumes interleaved input pairs but
        # writes the rotated even and odd lanes into separate output halves.
        # Its cache ABI is therefore [c0..c31, c0..c31], independent of the
        # logical interleaved layout used by SGLang.  The DSA indexer has its
        # own raw cache path and must not share this expanded MLA cache.
        cos = cos.repeat(1, 2)
        sin = sin.repeat(1, 2)
        return cos.to(torch.bfloat16), sin.to(torch.bfloat16)

    @staticmethod
    def _indexer_rope_cache(attention) -> torch.Tensor:
        """Return BlockRT's raw [position, cos(32)+sin(32)] RoPE table."""

        cache = attention.indexer.rotary_emb.cos_sin_cache
        if cache.ndim != 2 or cache.shape[1] != 64:
            raise RuntimeError(
                "MegaKernel indexer_rope requires SGLang's raw [positions, 64] "
                "cache containing [cos(32), sin(32)]; got "
                f"shape={tuple(cache.shape)}."
            )
        if cache.dtype != torch.bfloat16:
            raise RuntimeError(
                "MegaKernel indexer_rope requires a BF16 RoPE cache; refusing "
                f"to allocate a converted copy from dtype={cache.dtype}."
            )
        return cache

    @staticmethod
    def _validate_pool(pool, expected_layers: int) -> None:
        dimensions = (
            int(pool.kv_lora_rank),
            int(pool.qk_rope_head_dim),
            int(pool.index_head_dim) if pool.index_head_dim is not None else None,
        )
        if dimensions != (512, 64, 128):
            raise RuntimeError(
                f"GLM-5.2 cache dimensions must be (512, 64, 128), got {dimensions}."
            )
        # 512 FP8 values, four FP32 scales and 64 BF16 rotary values.
        expected_cache_dim = 512 + 4 * 4 + 64 * 2
        if (
            pool.page_size != 128
            or pool.index_page_size != pool.page_size
            or pool.index_size != pool.size
            or pool.index_kpool != 1
            or pool.use_fia_nz
        ):
            raise RuntimeError(
                "The provider requires matching physical MLA and indexer pages"
            )
        if not pool.dsa_kv_cache_store_fp8:
            raise RuntimeError(
                "MegaKernel requires SGLang's packed FP8 DSA KV cache. Start "
                "the server with --kv-cache-dtype=fp8_e4m3 on an Ascend A5 "
                "device so npu_quant_lightning_indexer is enabled."
            )
        if pool.dtype != torch.float8_e4m3fn:
            raise RuntimeError(
                f"MegaKernel requires FP8 E4M3 logical KV dtype, got {pool.dtype}."
            )
        if (
            pool.store_dtype != torch.float8_e4m3fn
            or pool.kv_cache_dim != expected_cache_dim
            or pool.kr_cache_dim != 0
        ):
            raise RuntimeError(
                "Unexpected SGLang packed DSA KV layout: "
                f"k_store_dtype={pool.store_dtype}, "
                f"kv_cache_dim={pool.kv_cache_dim}, "
                f"kr_cache_dim={pool.kr_cache_dim}, "
                f"expected_dim={expected_cache_dim}."
            )
        if pool.index_k_scale_buffer is None:
            raise RuntimeError(
                "MegaKernel requires the SGLang FP32 Indexer scale cache."
            )
        if pool.layer_num != expected_layers:
            raise RuntimeError(
                "The KV pool does not contain the complete decoder: "
                f"start_layer={pool.start_layer}, layer_num={pool.layer_num}, "
                f"expected={expected_layers}."
            )

    @staticmethod
    def _indexer_owner_layer(
        layer_id: int,
        physical_layer_ids: tuple[int, ...],
    ) -> int:
        """Return the preceding physical Indexer layer for a skipped layer."""

        owners = [owner for owner in physical_layer_ids if owner <= layer_id]
        if not owners:
            raise RuntimeError(
                f"Layer {layer_id} has no preceding physical Indexer cache; "
                f"physical layers are {physical_layer_ids}."
            )
        return owners[-1]

    @staticmethod
    def _zero_copy_flatten(
        tensor: torch.Tensor,
        *,
        width: int,
        dtype: torch.dtype,
        name: str,
    ) -> torch.Tensor:
        """Adapt a pool tensor to BlockRT shape without allocating storage."""

        if tensor.dtype != dtype:
            raise RuntimeError(
                f"{name} must keep SGLang's {dtype} storage, got {tensor.dtype}."
            )
        if tensor.shape[-1] != width:
            raise RuntimeError(
                f"{name} must have width {width}, got shape {tuple(tensor.shape)}."
            )
        if not tensor.is_contiguous():
            raise RuntimeError(
                f"{name} is not contiguous; refusing to call contiguous() "
                "because that would allocate a second KV cache."
            )
        view = tensor.view(-1, width)
        if view.data_ptr() != tensor.data_ptr():
            raise RuntimeError(f"{name} reshape unexpectedly copied storage.")
        return view

    def _get_shared_cache(self, pool) -> _SharedFp8Cache:
        """Build no-copy per-layer views over SGLang-owned FP8 storage."""

        if self._shared_cache is not None:
            if self._shared_pool_id != id(pool):
                raise RuntimeError(
                    "The SGLang KV pool changed after MegaKernel graph capture."
                )
            return self._shared_cache

        layer_ids = tuple(range(pool.start_layer, pool.start_layer + pool.layer_num))
        physical_indexer_layers = tuple(pool.indexer_layer_ids)
        k_cache = []
        index_k_buffer = []
        index_k_scale_buffer = []
        for layer_id in layer_ids:
            layer_k_cache = pool.get_key_buffer(layer_id)
            if (
                layer_k_cache.dtype != torch.float8_e4m3fn
                or layer_k_cache.ndim != 4
                or tuple(layer_k_cache.shape[-2:]) != (1, pool.kv_cache_dim)
                or not layer_k_cache.is_contiguous()
            ):
                raise RuntimeError(
                    "SGLang MLA cache cannot be handed to MegaKernel without "
                    "copying: "
                    f"layer={layer_id}, shape={tuple(layer_k_cache.shape)}, "
                    f"dtype={layer_k_cache.dtype}, "
                    f"contiguous={layer_k_cache.is_contiguous()}."
                )
            # Keep the exact SGLang tensor.  In particular, do not call to(),
            # clone(), contiguous(), or repack its 656-byte rows here.
            k_cache.append(layer_k_cache)

            owner = (
                layer_id
                if layer_id in pool.indexer_layer_id_to_slot
                else self._indexer_owner_layer(layer_id, physical_indexer_layers)
            )
            index_k_buffer.append(
                self._zero_copy_flatten(
                    pool.get_index_k_buffer(owner),
                    width=pool.index_head_dim,
                    dtype=torch.float8_e4m3fn,
                    name=f"index_k_cache[{layer_id}] (owner={owner})",
                )
            )
            index_k_scale_buffer.append(
                self._zero_copy_flatten(
                    pool.get_index_k_scale_buffer(owner),
                    width=1,
                    dtype=torch.float32,
                    name=f"index_k_scale[{layer_id}] (owner={owner})",
                )
            )

        self._shared_cache = _SharedFp8Cache(
            k_cache=k_cache,
            index_k_buffer=index_k_buffer,
            index_k_scale_buffer=index_k_scale_buffer,
        )
        self._shared_pool_id = id(pool)
        return self._shared_cache

    def _get_phase_buffers(
        self,
        *,
        phase: str,
        batch_size: int,
        num_tokens: int,
        block_table_width: int,
        device: torch.device,
    ) -> _DecodeBuffers:
        signature = (phase, batch_size, num_tokens, block_table_width)
        buffers = self._buffers_by_signature.get(signature)
        if buffers is not None:
            return buffers
        buffers = _DecodeBuffers(
            input_ids=torch.empty(num_tokens, dtype=torch.int32, device=device),
            positions=torch.empty(num_tokens, dtype=torch.int64, device=device),
            block_tables=torch.empty(
                (batch_size, block_table_width),
                dtype=torch.int32,
                device=device,
            ),
            slot_mapping=torch.empty((num_tokens, 1), dtype=torch.int64, device=device),
            kv_seq_len=torch.empty(batch_size, dtype=torch.int32, device=device),
            q_seq_len=torch.empty(batch_size, dtype=torch.int32, device=device),
            mla_cos_cache=torch.empty(
                (num_tokens, 64), dtype=torch.bfloat16, device=device
            ),
            mla_sin_cache=torch.empty(
                (num_tokens, 64), dtype=torch.bfloat16, device=device
            ),
        )
        self._buffers_by_signature[signature] = buffers
        return buffers

    @staticmethod
    def _fill_block_tables(buffers, forward_batch, block_size, *, kv_seq_len, **kwargs):
        # Slice the page starts before gathering the row. This also bounds reads
        # when the request table is narrower than the graph's 128 columns.
        table = get_req_to_token_pool().req_to_token[:, : 128 * block_size : block_size]
        slots = table.index_select(0, forward_batch.req_pool_indices.long())
        pages = slots.div(block_size, rounding_mode="floor")
        offsets = torch.arange(pages.shape[1], device=pages.device) * block_size
        buffers.block_tables.fill_(-1)
        buffers.block_tables[:, : pages.shape[1]].copy_(
            pages.masked_fill(offsets >= kv_seq_len.view(-1, 1), -1)
        )

    def _prepare(self, batch, ids, positions, phase, *, mtp=False):
        from blockrt.models.glm_5_2.model_args import ForwardBatchInfo

        pool = get_token_to_kv_pool()
        cache = self._get_shared_cache(pool)
        rows = ids.numel()
        if (
            batch.batch_size != 1
            or positions.shape != (rows,)
            or batch.out_cache_loc.numel() != rows
            or batch.req_pool_indices.numel() != 1
        ):
            raise ValueError(
                "Expected one request with one physical write slot per query"
            )
        b = self._get_phase_buffers(
            phase=phase,
            batch_size=1,
            num_tokens=rows,
            block_table_width=128,
            device=ids.device,
        )
        b.input_ids.copy_(ids)
        b.positions.copy_(positions)
        b.q_seq_len.fill_(rows)
        b.kv_seq_len.copy_(positions[-1:] + 1)
        b.slot_mapping.copy_(batch.out_cache_loc.reshape(-1, 1))
        self._fill_block_tables(b, batch, pool.page_size, kv_seq_len=b.kv_seq_len)
        attn = self._get_first_attention()
        cos, sin = self._expanded_cos_sin(attn.rotary_emb, b.positions)
        b.mla_cos_cache.copy_(cos)
        b.mla_sin_cache.copy_(sin)
        index_rope = self._indexer_rope_cache(attn)
        # BlockRT addresses the single NextN layer by its checkpoint layer ID.
        prefix = [None] * self.model.config.num_hidden_layers if mtp else []
        info = ForwardBatchInfo(
            sin_cos_cache=index_rope,
            index_cos_sin_cache=index_rope,
            k_cache=prefix + cache.k_cache,
            v_cache=prefix + [None] if mtp else [],
            index_k_buffer=prefix + cache.index_k_buffer,
            index_k_scale_buffer=prefix + cache.index_k_scale_buffer,
            num_blocks=cache.k_cache[0].shape[0],
            block_tables=b.block_tables,
            slot_mapping=b.slot_mapping,
            kv_seq_len=b.kv_seq_len,
            q_seq_len=b.q_seq_len,
            num_tokens=rows,
            mask=getattr(batch.spec_info, "custom_mask", None),
            index_rope_positions=b.positions,
            mla_cos_cache=b.mla_cos_cache,
            mla_sin_cache=b.mla_sin_cache,
        )
        return info, b

    def build(self, forward_batch, input_ids, positions):
        info, buffers = self._prepare(forward_batch, input_ids, positions, "decode")
        return info, buffers.input_ids, buffers.positions

    def build_target_verify(self, forward_batch, input_ids, positions):
        info, buffers = self._prepare(forward_batch, input_ids, positions, "verify")
        return info, buffers.input_ids, buffers.positions


class Glm52PackedMTPHandoff(Glm52PackedDecodeHandoff):
    @staticmethod
    def phase_name(batch):
        if batch.forward_mode.is_decode():
            return "mtp_decode"
        raise ValueError("Unexpected NextN forward mode")

    def build(
        self,
        batch,
        ids,
        positions,
        hidden,
        prev_dsa_topk_indices,
        *,
        buffer_phase,
        buffer_model_inputs=True,
    ):
        phase = self.phase_name(batch)
        info, buffers = self._prepare(batch, ids, positions, buffer_phase, mtp=True)
        if prev_dsa_topk_indices is not None:
            raise ValueError("BlockRT owns the suffix indexer state")
        if buffer_model_inputs:
            if buffers.previous_hidden_states is None:
                buffers.previous_hidden_states = torch.empty_like(hidden)
            buffers.previous_hidden_states.copy_(hidden)
            hidden = buffers.previous_hidden_states
        return (
            phase,
            info,
            buffers.input_ids,
            buffers.positions,
            hidden,
            None,
            ids.numel(),
        )
