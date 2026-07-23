from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from sglang.srt.layers.attention.dsv4.compressor import Compressor as _CompressorBase
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
    KVAndScoreSeparate,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.utils import cpu_has_amx_support

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
_cpu_amx = cpu_has_amx_support()


def apply_rotary_emb_cpu(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    inverse: bool = False,
) -> torch.Tensor:
    return torch.ops.sgl_kernel.apply_rotary_emb_interleaved_cpu(
        x, freqs_cis, inverse, positions
    )


class CompressorCPU(_CompressorBase):
    """CPU specific Compressor implementation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _get_states(
        self,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> KVAndScoreSeparate | CompressStatePool:
        """Return the per-layer compress-state for this Compressor.

        When the radix path is on this is a paged ``CompressStatePool``;
        otherwise it is a ``KVAndScore`` / ``KVAndScoreSeparate`` view of the
        per-request non-paged buffer.
        """
        token_to_kv_pool = attn_backend.token_to_kv_pool
        assert isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool)
        if self.is_in_indexer:
            return token_to_kv_pool.get_indexer_compress_states(self.layer_id)
        return token_to_kv_pool.get_attention_compress_states(self.layer_id)

    def get_state_pool(
        self, forward_batch: ForwardBatch, attn_backend: AttentionBackend
    ) -> CompressStatePool:
        ret = self._get_states(forward_batch, attn_backend)
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

    def forward(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            assert x.shape[0] == 0
            return x.new_empty(0, self.head_dim)

        kv_score = self.compute_kv_score(x, forward_batch)
        return self.compress_dispatch(
            kv_score, forward_batch, attn_backend=attn_backend
        )

    def compress_decode_separate(
        self,
        kv_and_scores: KVAndScoreSeparate,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation

        """
        Reads from non-paged ``CompressStateSeparate`` buffers.
        """

        assert self.ape_converted
        seq_lens = forward_batch.seq_lens
        pool = self._get_states(forward_batch, attn_backend)
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
        apply_rotary_emb_cpu(kv_compressed[..., -self.rope_head_dim :], freqs_cis)
        if self.rotate:
            kv_compressed = rotate_activation(kv_compressed)
        return kv_compressed

    def compress_extend_separate(
        self,
        kv_and_scores: KVAndScoreSeparate,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> torch.Tensor:
        from sglang.srt.layers.attention.nsa.nsa_indexer import rotate_activation

        """
        Reads from non-paged ``CompressStateSeparate`` buffers.
        """
        assert self.ape_converted

        kv_and_score_states = self._get_states(forward_batch, attn_backend)
        assert isinstance(kv_and_score_states, KVAndScoreSeparate)
        _, _, head_dim_times_coff = kv_and_score_states.kv.shape

        prefix_lens = forward_batch.extend_prefix_lens_cpu
        extend_lens = forward_batch.extend_seq_lens_cpu
        req_pool_indices = forward_batch.req_pool_indices
        assert extend_lens is not None and prefix_lens is not None

        max_buffer_size = 2 * kv_and_score_states.shape[1] + kv_and_scores.shape[0]
        temp_buffer_shape = [max_buffer_size, head_dim_times_coff]
        temp_buffer = KVAndScoreSeparate.empty_like(
            temp_buffer_shape, sep=kv_and_scores
        )

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
            apply_rotary_emb_cpu(kv_compressed[..., -self.rope_head_dim :], freqs_cis)

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

    def compress_dispatch(
        self,
        kv_score: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend: AttentionBackend,
    ) -> torch.Tensor:
        self.compress_decode = self.compress_decode_separate
        self.compress_extend = self.compress_extend_separate
        kv = kv_score[:, : self.coff * self.head_dim]
        score = kv_score[:, self.coff * self.head_dim :]
        kv_and_scores = KVAndScoreSeparate(kv=kv, score=score)
        forward_mode = forward_batch.forward_mode
        if forward_mode.is_decode() or forward_mode.is_target_verify():
            if _cpu_amx:
                pool = self._get_states(forward_batch, attn_backend)
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
            return self.compress_decode(kv_and_scores, forward_batch, attn_backend)
        if forward_mode.is_extend():
            return self.compress_extend(kv_and_scores, forward_batch, attn_backend)
        raise NotImplementedError(
            f"Forward mode {forward_mode} not supported in KVAndScoreSeparate compressor."
        )

    def _get_freqs_cis_real(self):
        """Return freqs_cis as real float32 [N, rope_dim] for CPU kernel."""
        if not hasattr(self, "_freqs_cis_real"):
            fc = self.freqs_cis
            if fc.is_complex():
                self._freqs_cis_real = (
                    torch.view_as_real(fc).contiguous().reshape(fc.size(0), -1)
                )
            else:
                self._freqs_cis_real = fc.contiguous()
        return self._freqs_cis_real
