# SPDX-License-Identifier: Apache-2.0
"""FlashInfer backend for LLaDA-Image conditioning masks."""

from __future__ import annotations

import torch
from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    PrefillMetadata,
)
from sglang.srt.layers.attention.llada2_attention_utils import (
    build_llada_image_custom_mask,
)


class LLaDA2CFGFlashInferAttnBackend(FlashInferAttnBackend):
    """Stock FlashInfer plus LLaDA-specific ragged attention masks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._llada_image_conditioning_mask_active = False
        self._cfg_prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend="fa2"
        )

    @property
    def conditioning_mask_active(self) -> bool:
        return self._llada_image_conditioning_mask_active

    def init_forward_metadata(self, forward_batch):
        self._llada_image_conditioning_mask_active = False
        text_lens = getattr(
            forward_batch, "llada_image_conditioning_text_lens_cpu", None
        )
        if text_lens is None:
            return super().init_forward_metadata(forward_batch)

        if not forward_batch.forward_mode.is_extend_without_speculative():
            raise RuntimeError(
                "LLaDA-Image conditioning mask requires a one-shot extend"
            )
        if self.num_wrappers != 1:
            raise RuntimeError(
                "LLaDA-Image conditioning requires one FlashInfer wrapper"
            )

        seq_lens = forward_batch.seq_lens
        prefix_lens = forward_batch.extend_prefix_lens
        if bool(torch.any(prefix_lens != 0).item()):
            raise RuntimeError(
                "LLaDA-Image conditioning does not support cached prefixes"
            )
        custom_mask = build_llada_image_custom_mask(
            text_lens,
            seq_lens.tolist(),
            seq_lens.device,
        )
        qo_indptr = torch.zeros(
            seq_lens.numel() + 1,
            dtype=torch.int32,
            device=seq_lens.device,
        )
        qo_indptr[1:] = torch.cumsum(seq_lens, dim=0)
        prefill_indices_updater = self.indices_updater_prefill
        self._cfg_prefill_wrapper_ragged.begin_forward(
            qo_indptr,
            qo_indptr,
            prefill_indices_updater.num_qo_heads,
            prefill_indices_updater.num_kv_heads,
            prefill_indices_updater.head_dim,
            custom_mask=custom_mask,
            causal=False,
            q_data_type=prefill_indices_updater.q_data_type,
            kv_data_type=prefill_indices_updater.data_type,
            non_blocking=True,
            fixed_split_size=self.prefill_split_tile_size,
        )
        self._llada_image_conditioning_mask_active = True
        self.forward_metadata = PrefillMetadata(
            self.prefill_wrappers_paged,
            use_ragged=True,
            extend_no_prefix=True,
        )

    def forward_extend(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
    ):
        if not self._llada_image_conditioning_mask_active:
            return super().forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )

        if k is None or v is None:
            raise RuntimeError("LLaDA ragged attention requires explicit K/V")

        q_view = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
        k_view = k.view(-1, layer.tp_k_head_num, layer.head_dim)
        v_view = v.view(-1, layer.tp_v_head_num, layer.head_dim)
        attention_output = self._cfg_prefill_wrapper_ragged.forward(
            q_view,
            k_view,
            v_view,
            causal=False,
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
        )
        return attention_output.view(-1, layer.tp_q_head_num * layer.head_dim)
