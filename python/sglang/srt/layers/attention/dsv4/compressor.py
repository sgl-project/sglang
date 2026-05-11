from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, List, Literal, NamedTuple, Optional, Union

import torch
import torch.nn as nn

from sglang.jit_kernel.deepseek_v4 import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_forward,
    compress_fused_norm_rope_inplace,
    linear_bf16_fp32,
    torch_create_paged_compress_data,
    triton_create_paged_compress_data,
)
from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsv4.quant_k_cache import (
    quant_to_nope_fp8_rope_bf16_pack_triton,
)
from sglang.srt.layers.attention.nsa.triton_kernel import act_quant
from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp
from sglang.srt.layers.dp_attention import get_attention_cp_size
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.utils.cp_utils import cp_all_gather_rerange_output
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
    KVAndScore,
    KVAndScoreSeparate,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.utils import add_prefix, cpu_has_amx_support, is_cpu

_is_cpu = is_cpu()
_cpu_amx = cpu_has_amx_support()
if TYPE_CHECKING:
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if _is_cpu and _cpu_amx:

    def apply_rotary_emb_cpu(
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        inverse: bool = False,
    ) -> torch.Tensor:
        return torch.ops.sgl_kernel.apply_rotary_emb_interleaved_cpu(
            x, freqs_cis, inverse, positions
        )

    apply_rotary_emb_triton = apply_rotary_emb_cpu


class FusedCompressMetadata(NamedTuple):
    write_loc: torch.Tensor
    extra_data: Optional[torch.Tensor]
    plan: Union[CompressorDecodePlan, CompressorPrefillPlan]

    def copy_(self, other: FusedCompressMetadata) -> None:
        from .metadata import maybe_copy_inplace

        self.write_loc.copy_(other.write_loc)
        maybe_copy_inplace(self.extra_data, src=other.extra_data)
        self.plan.copy_(other.plan)


class CompressorBackendMixin:
    def get_paged_compress_metadata(self, compress_ratio: int) -> FusedCompressMetadata:
        attr_name = f"c{compress_ratio}_compress_metadata"
        metadata = getattr(self.forward_metadata, attr_name)
        assert isinstance(metadata, FusedCompressMetadata)
        return metadata

    def forward_compress(
        self,
        *,
        kv_score_buffer: torch.Tensor,
        kv_score_input: torch.Tensor,
        ape: torch.Tensor,
        head_dim: int,
        norm: RMSNorm,
        freqs_cis_cache: torch.Tensor,
        rotate: bool,
        forward_batch: ForwardBatch,
        compress_ratio: int,
        is_paged: bool = False,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation

        assert compress_ratio in (
            4,
            128,
        ), f"DSV4 supports CSA(4x) and HCA(128x) only, got {compress_ratio=}"
        if is_paged:
            metadata = self.get_paged_compress_metadata(compress_ratio)
            coff = 2 if is_overlap_compress(compress_ratio) else 1
            if compress_ratio == 128 and envs.SGLANG_OPT_USE_ONLINE_COMPRESS.get():
                kv_score_buffer = kv_score_buffer.view(-1, 1, head_dim * 3)
            else:
                last_dim = 2 * head_dim * coff
                assert kv_score_buffer.shape[-1] == last_dim
                kv_score_buffer = kv_score_buffer.view(-1, compress_ratio, last_dim)
        else:
            plan = make_compressor_plan(compress_ratio, forward_batch)
            metadata = (forward_batch.req_pool_indices.to(torch.int32), None, plan)
        indices, extra_data, plan = metadata

        kv_compressed = compress_forward(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score_input,
            ape=ape,
            indices=indices,
            plan=plan,
            compress_ratio=compress_ratio,
            head_dim=head_dim,
            extra_data=extra_data,
        )
        compress_fused_norm_rope_inplace(
            kv_compressed,
            norm.weight,
            norm.variance_epsilon,
            freqs_cis_cache,
            plan,
        )
        return rotate_activation(kv_compressed) if rotate else kv_compressed

    def forward_core_compressor(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        # PREP_IN_CG lazy upgrade: the concrete backend (DeepseekV4AttnBackend)
        # owns this helper. MQALayer._forward_prepare calls us before
        # attn_backend.forward(), so Raw -> DSV4Metadata must happen here too
        # (e.g. 1.6T layer 0 has compress_ratio=128 and needs cX_compress_metadata).
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        new_compressed_kv = compressor(x, forward_batch)
        core_metadata = self.forward_metadata.core_metadata
        out_loc = (
            core_metadata.c4_out_loc
            if compressor.ratio == 4
            else core_metadata.c128_out_loc
        )
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            token_to_kv_pool.set_extra_key_buffer_fused(
                layer_id=layer_id,
                loc=out_loc,
                cache_k=new_compressed_kv,
            )
        else:
            if _is_cpu and _cpu_amx:
                from sglang.srt.layers.attention.dsv4.index_buf_accessor import (
                    NopeFp8RopeBf16Pack,
                )

                pack = NopeFp8RopeBf16Pack(
                    *torch.ops.sgl_kernel.quant_to_nope_fp8_rope_bf16_pack_cpu(
                        new_compressed_kv.bfloat16()
                    )
                )
            else:
                pack = quant_to_nope_fp8_rope_bf16_pack_triton(
                    new_compressed_kv.bfloat16()
                )
            token_to_kv_pool.set_extra_key_buffer(layer_id, out_loc, pack)

    def forward_indexer_compressor(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor: Compressor,
    ) -> None:
        assert is_overlap_compress(compressor.ratio)
        # PREP_IN_CG lazy upgrade (see forward_core_compressor for rationale).
        self._maybe_upgrade_forward_metadata()
        token_to_kv_pool = forward_batch.token_to_kv_pool
        if TYPE_CHECKING:
            assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)

        new_compressed_kv = compressor(x, forward_batch)
        if envs.SGLANG_OPT_USE_FUSED_STORE_CACHE.get():
            token_to_kv_pool.set_index_k_fused(
                layer_id=layer_id,
                loc=self.forward_metadata.core_metadata.c4_out_loc,
                cache_k=new_compressed_kv,
            )
        else:
            if _is_cpu and _cpu_amx:
                new_compressed_kv_fp8, new_compressed_kv_scale = (
                    torch.ops.sgl_kernel.act_quant_cpu(new_compressed_kv)
                )
            else:
                new_compressed_kv_fp8, new_compressed_kv_scale = act_quant(
                    new_compressed_kv
                )
            token_to_kv_pool.set_index_k_scale_buffer(
                layer_id=layer_id,
                loc=self.forward_metadata.core_metadata.c4_out_loc,
                index_k=new_compressed_kv_fp8,
                index_k_scale=new_compressed_kv_scale,
            )


def is_overlap_compress(compress_ratio: int) -> bool:
    return compress_ratio == 4


def make_compressor_plan(
    compress_ratio: Literal[4, 128],
    forward_batch: ForwardBatch,
) -> Union[CompressorDecodePlan, CompressorPrefillPlan]:
    if forward_batch.forward_mode.is_decode():
        seq_lens_32 = forward_batch.seq_lens.to(torch.int32)
        return CompressorDecodePlan(compress_ratio, seq_lens_32)
    if forward_batch.forward_mode.is_prefill():
        assert not forward_batch.forward_mode.is_target_verify()
        extend_lens_list = forward_batch.extend_seq_lens_cpu
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert extend_lens_list is not None and seq_lens_cpu is not None
        return CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            num_q_tokens=sum(extend_lens_list),
            seq_lens=seq_lens_cpu,
            extend_lens=torch.tensor(extend_lens_list),
            device=forward_batch.seq_lens.device,
        )
    elif forward_batch.forward_mode.is_target_verify():
        raise NotImplementedError("target verify mode to be implemented")
    else:
        raise NotImplementedError(f"unsupported mode {forward_batch.forward_mode=}")


def create_paged_compressor_data(
    compress_ratio: Literal[4, 128],
    *,
    is_prefill: bool,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: Optional[torch.Tensor] = None,
    seq_lens_cpu: Optional[List[int]] = None,
    extend_lens_cpu: Optional[List[int]] = None,
    use_prefill_cuda_graph: bool = False,
    num_q_tokens: Optional[int] = None,
) -> FusedCompressMetadata:
    swa_page_size = token_to_kv_pool.swa_page_size
    ring_size = token_to_kv_pool.get_ring_size(compress_ratio=compress_ratio)
    # assert ring_size % compress_ratio == 0

    def clip_down(positions: torch.Tensor) -> torch.Tensor:
        return positions // compress_ratio * compress_ratio

    def get_raw_loc(positions: torch.Tensor) -> torch.Tensor:
        positions = positions.masked_fill(positions < 0, 0)
        loc = req_to_token[req_pool_indices, positions]
        swa_loc = token_to_kv_pool.translate_loc_from_full_to_swa(loc)
        swa_pages = swa_loc // swa_page_size
        state_loc = swa_pages * ring_size + swa_loc % ring_size
        return (state_loc // compress_ratio).to(torch.int32)

    is_overlap = is_overlap_compress(compress_ratio)

    if is_prefill:
        assert extend_lens is not None
        create_paged_compress_data_func = (
            triton_create_paged_compress_data
            if not _is_cpu
            else torch_create_paged_compress_data
        )
        write_loc, extra_data = create_paged_compress_data_func(
            compress_ratio=compress_ratio,
            is_overlap=is_overlap,
            swa_page_size=swa_page_size,
            ring_size=ring_size,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            extend_seq_lens=extend_lens,
            req_to_token=req_to_token,
            full_to_swa_index_mapping=token_to_kv_pool.full_to_swa_index_mapping,
        )

        plan_kwargs: dict
        if seq_lens_cpu is None:
            assert num_q_tokens is not None
            plan_kwargs = dict(
                num_q_tokens=num_q_tokens,
                seq_lens=seq_lens,
                extend_lens=extend_lens,
            )
        else:
            assert extend_lens_cpu is not None
            plan_kwargs = dict(
                num_q_tokens=sum(extend_lens_cpu),
                seq_lens=torch.tensor(seq_lens_cpu),
                extend_lens=torch.tensor(extend_lens_cpu),
            )
        plan = CompressorPrefillPlan.generate(
            compress_ratio=compress_ratio,
            device=seq_lens.device,
            use_cuda_graph=use_prefill_cuda_graph,
            **plan_kwargs,
        )
    else:
        write_positions = clip_down(seq_lens - 1)
        write_loc = get_raw_loc(write_positions)
        if is_overlap:
            write_overlap_loc = get_raw_loc(write_positions - compress_ratio)
            extra_data = write_overlap_loc.view(-1, 1)
        else:
            extra_data = None
        plan = CompressorDecodePlan(compress_ratio, seq_lens.to(torch.int32))

    return FusedCompressMetadata(write_loc=write_loc, extra_data=extra_data, plan=plan)


class Compressor(nn.Module):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        is_in_indexer: bool,
        freqs_cis: torch.Tensor,
        compress_ratio: Literal[0, 4, 128],
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_in_indexer = is_in_indexer
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        assert compress_ratio != 0, "compress_ratio should not be 0"
        self.ratio = compress_ratio
        self.overlap = self.ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap
        self.coff = coff = 1 + self.overlap

        self.ape = nn.Parameter(
            torch.empty(self.ratio, coff * self.head_dim, dtype=torch.float32)
        )
        wkv_gate_dtype = torch.bfloat16
        self.wkv_gate = ReplicatedLinear(
            self.dim,
            2 * coff * self.head_dim,
            bias=False,
            quant_config=None,
            prefix=add_prefix("wkv_gate", prefix),
            params_dtype=wkv_gate_dtype,
        )
        self.norm = RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, weight_dtype=torch.float32
        )
        self.freqs_cis = freqs_cis

        self.ape_converted = False

    def apply_ape_hotfix(self):
        assert not self.ape_converted
        self.ape_converted = True

        if self.overlap:
            ape = torch.chunk(self.ape.data, 2, dim=-1)
            ape = torch.cat([ape[0], ape[1]], dim=0)
            self.ape.data.copy_(ape.view(self.ratio, -1))

    @cached_property
    def use_fused_compress(self) -> bool:
        if envs.SGLANG_CPU_USE_COMPRESS_SEPARATE.get():
            return False
        return True

    def _get_states(
        self, forward_batch: ForwardBatch
    ) -> "KVAndScore | KVAndScoreSeparate | CompressStatePool":
        """Return the per-layer compress-state for this Compressor.

        When the radix path is on this is a paged ``CompressStatePool``;
        otherwise it is a ``KVAndScore`` / ``KVAndScoreSeparate`` view of the
        per-request non-paged buffer (used by the old-compressor fallback).
        """
        token_to_kv_pool = forward_batch.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            return token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        return token_to_kv_pool.get_attention_compress_states(self.layer_id)

    def _get_state_pool(self, forward_batch: ForwardBatch) -> CompressStatePool:
        ret = self._get_states(forward_batch)
        assert isinstance(ret, CompressStatePool)
        return ret
    def overlap_transform(self, tensor: torch.Tensor, fill_value: Any) -> torch.Tensor:
        assert tensor.dim() == 3
        assert tensor.shape[1:] == (self.ratio, 2 * self.head_dim)

        s, r, d = tensor.size(0), self.ratio, self.head_dim
        new_tensor = tensor.new_full((s, 2 * r, d), fill_value)
        new_tensor[:, r:] = tensor[:, :, d:]
        new_tensor[1:, :r] = tensor[:-1, :, :d]
        return new_tensor

    def overlap_transform_decode(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.dim() == 3
        assert tensor.shape[1:] == (2 * self.ratio, 2 * self.head_dim)
        r, d = self.ratio, self.head_dim
        ret = torch.cat((tensor[:, :r, :d], tensor[:, r:, d:]), dim=1)
        return ret

    @staticmethod
    def compute_state_len(seq_len: int, ratio: int):
        return seq_len % ratio + (ratio == 4) * ratio

    def forward(self, x: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)

        kv_score = linear_bf16_fp32(x, self.wkv_gate.weight)
        if nsa_use_prefill_cp(forward_batch):
            kv_score = cp_all_gather_rerange_output(
                kv_score,
                get_attention_cp_size(),
                forward_batch,
                torch.cuda.current_stream(),
            )
        return self.compress_dispatch(kv_score, forward_batch)

    def compress_fused(
        self,
        kv_score: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        backend = forward_batch.attn_backend
        if TYPE_CHECKING:
            assert isinstance(backend, DeepseekV4AttnBackend)
        kv_score_buffer = self._get_state_pool(forward_batch)
        kv_score_buffer = kv_score_buffer.kv_score_buffer.kv_score
        return backend.forward_compress(
            kv_score_buffer=kv_score_buffer,
            kv_score_input=kv_score,
            ape=self.ape.view(-1, self.head_dim),
            head_dim=self.head_dim,
            norm=self.norm,
            freqs_cis_cache=self.freqs_cis,
            rotate=self.rotate,
            compress_ratio=self.ratio,
            forward_batch=forward_batch,
            is_paged=True,
        )

    def compress_decode_separate(
        self,
        kv_and_scores: "KVAndScoreSeparate",
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation

        """
        Reads from non-paged ``DeepSeekV4CompressState`` buffers.
        """
        
        assert self.ape_converted
        seq_lens = forward_batch.seq_lens
        pool = self._get_states(forward_batch)
        assert isinstance(pool, KVAndScoreSeparate)
        req_pool_indices = forward_batch.req_pool_indices

        bs = kv_and_scores.kv.size(0)
        write_pos = (seq_lens - 1) % self.ratio + self.overlap * self.ratio
        pool[req_pool_indices, write_pos] = kv_and_scores

        # NOTE: copy out before modifying overlap states
        kv_and_score_to_compress = pool[req_pool_indices]

        if self.overlap:
            should_shift = (seq_lens % self.ratio == 0)[:, None, None]
            pool[req_pool_indices, : self.ratio] = KVAndScoreSeparate(
                kv=torch.where(
                    should_shift,
                    kv_and_score_to_compress.kv[:, self.ratio :],
                    kv_and_score_to_compress.kv[:, : self.ratio],
                ),
                score=torch.where(
                    should_shift,
                    kv_and_score_to_compress.score[:, self.ratio :],
                    kv_and_score_to_compress.score[:, : self.ratio],
                ),
            )

        # shape: [bs * coff, ratio, coff * head_dim]
        kv_and_score_to_compress = kv_and_score_to_compress.view(
            -1, self.ratio, self.coff * self.head_dim
        )
        kv_and_score_to_compress.score = (
            kv_and_score_to_compress.score + self.ape.unsqueeze(0)
        )

        if self.overlap:
            # shape: [bs, coff * ratio, coff * head_dim]
            kv_and_score_to_compress = kv_and_score_to_compress.view(
                bs, self.coff * self.ratio, self.coff * self.head_dim
            )
            kv_and_score_to_compress.kv = self.overlap_transform_decode(
                kv_and_score_to_compress.kv
            )
            kv_and_score_to_compress.score = self.overlap_transform_decode(
                kv_and_score_to_compress.score
            )

        # kv_to_compress: [bs, ratio * coff, head_dim]
        kv_and_score_to_compress = kv_and_score_to_compress.view(
            bs, self.ratio * self.coff, self.head_dim
        )
        kv_compressed = (
            kv_and_score_to_compress.kv * kv_and_score_to_compress.score.softmax(dim=1)
        ).sum(dim=1)
        kv_compressed = self.norm(kv_compressed)
        freqs_cis = self.freqs_cis[(seq_lens - 1) // self.ratio * self.ratio]
        apply_rotary_emb_triton(kv_compressed[..., -self.rope_head_dim :], freqs_cis)
        if self.rotate:
            kv_compressed = rotate_activation(kv_compressed)
        return kv_compressed

    def compress_extend_separate(
        self,
        kv_and_scores: "KVAndScoreSeparate",
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation

        """
        Reads from non-paged ``DeepSeekV4CompressState`` buffers.
        """
        assert self.ape_converted

        kv_and_score_states = self._get_states(forward_batch)
        assert isinstance(kv_and_score_states, KVAndScoreSeparate)
        _, _, head_dim_times_coff = kv_and_score_states.kv.shape

        prefix_lens = forward_batch.extend_prefix_lens_cpu
        extend_lens = forward_batch.extend_seq_lens_cpu
        req_pool_indices = forward_batch.req_pool_indices
        assert extend_lens is not None and prefix_lens is not None

        max_buffer_size = 2 * kv_and_score_states.shape[1] + kv_and_scores.shape[0]
        temp_buffer_shape = [max_buffer_size, head_dim_times_coff]
        temp_buffer = KVAndScoreSeparate.empty_like(temp_buffer_shape, sep=kv_and_scores)

        assert kv_and_scores.kv.shape[-1] == self.head_dim * self.coff
        compressed_kv_output = torch.full(
            (kv_and_scores.kv.size(0), self.head_dim),
            fill_value=10000.0,
            dtype=kv_and_scores.kv.dtype,
            device=kv_and_scores.kv.device,
        )

        bs = forward_batch.batch_size
        pt = 0
        for i in range(bs):
            kv_and_score = kv_and_scores[pt : pt + extend_lens[i]]
            kv_and_score_state = kv_and_score_states[req_pool_indices[i]]
            if prefix_lens[i] == 0:
                # Pad with default values for overlap.
                kv_and_score_state.clear()

            pre_state_len = self.compute_state_len(
                seq_len=prefix_lens[i], ratio=self.ratio
            )
            valid_kv_len = pre_state_len + extend_lens[i]
            kv_and_score_buffer = temp_buffer[:valid_kv_len]
            kv_and_score_buffer[:pre_state_len] = kv_and_score_state[:pre_state_len]
            kv_and_score_buffer[pre_state_len:valid_kv_len] = kv_and_score

            post_state_len = self.compute_state_len(
                seq_len=valid_kv_len, ratio=self.ratio
            )
            kv_and_score_state[:post_state_len] = kv_and_score_buffer[
                valid_kv_len - post_state_len : valid_kv_len
            ]

            compress_len = valid_kv_len // self.ratio * self.ratio
            if compress_len == 0:
                pt += extend_lens[i]
                continue

            kv_and_score_to_compress = kv_and_score_buffer[:compress_len].view(
                compress_len // self.ratio, self.ratio, -1
            )
            kv_and_score_to_compress.score = (
                kv_and_score_to_compress.score + self.ape.unsqueeze(0)
            )

            if self.overlap:
                kv_and_score_to_compress.kv = self.overlap_transform(
                    kv_and_score_to_compress.kv, 0
                )
                kv_and_score_to_compress.score = self.overlap_transform(
                    kv_and_score_to_compress.score, float("-inf")
                )
                # Drop the leading window before compression.
                kv_and_score_to_compress = kv_and_score_to_compress[1:]
                if kv_and_score_to_compress.kv.size(0) == 0:
                    pt += extend_lens[i]
                    continue

            kv_compressed = (
                kv_and_score_to_compress.kv
                * kv_and_score_to_compress.score.softmax(dim=1)
            ).sum(dim=1)
            assert kv_compressed.dtype == torch.float32
            kv_compressed = self.norm(kv_compressed)

            beg_idx = prefix_lens[i] // self.ratio * self.ratio
            end_idx = (prefix_lens[i] + extend_lens[i]) // self.ratio * self.ratio
            freqs_cis = self.freqs_cis[beg_idx : end_idx : self.ratio]
            assert freqs_cis.size(0) == kv_compressed.size(
                0
            ), f"{freqs_cis.shape=} {kv_compressed.shape=}"
            apply_rotary_emb_triton(
                kv_compressed[..., -self.rope_head_dim :], freqs_cis
            )

            if self.rotate:
                kv_compressed = rotate_activation(kv_compressed)

            start = prefix_lens[i]
            start = start + self.ratio - 1 - start % self.ratio
            indices_in_seq = torch.arange(
                start,
                prefix_lens[i] + extend_lens[i],
                self.ratio,
                device=kv_and_scores.kv.device,
            )
            assert indices_in_seq.size(0) == kv_compressed.size(0)
            compressed_kv_output[indices_in_seq - prefix_lens[i] + pt] = kv_compressed

            pt += extend_lens[i]

        return compressed_kv_output
    def _get_freqs_cis_real(self):
        """Return freqs_cis as real float32 [N, rope_dim] for CPU kernel."""
        if not hasattr(self, "_freqs_cis_real"):
            fc = self.freqs_cis
            if fc.is_complex():
                self._freqs_cis_real = torch.view_as_real(fc).contiguous().reshape(
                    fc.size(0), -1
                )
            else:
                self._freqs_cis_real = fc.contiguous()
        return self._freqs_cis_real
    def compress_dispatch(
        self,
        kv_score: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if self.use_fused_compress:
            return self.compress_fused(kv_score, forward_batch)
        self.compress_decode = self.compress_decode_separate
        self.compress_extend = self.compress_extend_separate
        kv = kv_score[:, : self.coff * self.head_dim]
        score = kv_score[:, self.coff * self.head_dim :]
        kv_and_scores = KVAndScoreSeparate(kv=kv, score=score)
        forward_mode = forward_batch.forward_mode
        if forward_mode.is_decode() or forward_mode.is_target_verify():
            if _cpu_amx:
                pool = self._get_states(forward_batch)
                assert isinstance(pool, KVAndScoreSeparate)
                freqs_real = self._get_freqs_cis_real()
                norm_weight = self.norm.weight.float()

                forward_mode = forward_batch.forward_mode
                return torch.ops.sgl_kernel.compress_decode_cpu(
                        pool.kv,
                        pool.score,
                        kv,
                        score,
                        forward_batch.seq_lens.to(torch.int64),
                        forward_batch.req_pool_indices.to(torch.int64),
                        self.ape,
                        norm_weight,
                        freqs_real,
                        self.ratio,
                        self.head_dim,
                        self.rope_head_dim,
                        self.overlap,
                        self.rotate,
                        self.norm.variance_epsilon,
                    )
            return self.compress_decode(kv_and_scores, forward_batch)
        if forward_mode.is_extend():
            return self.compress_extend(kv_and_scores, forward_batch)
        raise NotImplementedError(
            f"Forward mode {forward_mode} not supported in KVAndScoreSeparate compressor."
        )
