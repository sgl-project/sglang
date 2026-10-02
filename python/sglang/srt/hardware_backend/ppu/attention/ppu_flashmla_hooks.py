"""PPU FlashMLA ops hooks.

Uses ``@plugin_hook`` to inject PPU-specific behavior into
``sglang.srt.layers.attention.flashmla_backend`` without modifying the
community source code.

The community ``FlashMLABackend.init_forward_metadata`` already handles
``seq_lens_cpu is None`` (via ``eager_max_k`` fallback to
``self.max_context_len``) and calls ``get_mla_metadata``, which is
hooked separately (item 1 below).  Therefore no REPLACE hook is needed
for ``init_forward_metadata`` — the community method runs as-is and
automatically uses the PPU ``get_mla_metadata`` implementation.

Registered hooks
~~~~~~~~~~~~~~~~

1. ``get_mla_metadata`` (REPLACE) — Replaces the community
   ``sgl_kernel.flash_mla.get_mla_metadata`` with the PPU implementation.

2. ``flash_mla_with_kvcache`` (REPLACE) — Same, for the decode kernel.

3. ``flash_mla_sparse_fwd`` (REPLACE) — Same, for the sparse fwd kernel.

4. ``FlashMLABackend.init_cuda_graph_state`` (AROUND) — Caps the
   ``cuda_graph_mla_metadata`` buffer to PPU's hard-limit of 320 sm_parts.

5. ``FlashMLABackend._apply_decode_target_verify_metadata`` (REPLACE) —
   Replaces the method body on PPU with the PPU-specific logic:
   ``scheduler_metadata, _ = get_mla_metadata(...)``, the
   ``if is_ppu(): raise`` guard, and the ``.tile_scheduler_metadata`` /
   ``.num_splits`` attribute pattern.  Handles DRAFT_EXTEND_V2 with the
   correct seq_lens (no double-add) and cuda_graph_draft_extend_seq_lens_k.

6. ``FlashMLABackend.init_forward_metadata_in_graph`` (REPLACE) —
   Resets ``flashmla_metadata.have_initialized`` before CUDA Graph
   capture so PPU's lazy init runs inside the capture scope.
"""

import torch

from sglang.srt.plugins.hook_registry import HookType, plugin_hook

_PPU_MAX_SM_PARTS = 320  # hard limit on num_sm_parts returned by PPU flash_mla


# ---------------------------------------------------------------------------
# 1, 2 & 3. Replace the module-level ops with PPU implementations
# ---------------------------------------------------------------------------


@plugin_hook(
    "sgl_kernel.flash_mla.flash_mla_with_kvcache",
    type=HookType.REPLACE,
)
def _ppu_flash_mla_with_kvcache(*args, **kwargs):
    from sglang.srt.hardware_backend.ppu.attention.flash_mla import (
        flash_mla_with_kvcache,
    )

    return flash_mla_with_kvcache(*args, **kwargs)


@plugin_hook(
    "sgl_kernel.flash_mla.flash_mla_sparse_fwd",
    type=HookType.REPLACE,
)
def _ppu_flash_mla_sparse_fwd(*args, **kwargs):
    from sglang.srt.hardware_backend.ppu.attention.flash_mla import (
        flash_mla_sparse_fwd,
    )

    return flash_mla_sparse_fwd(*args, **kwargs)


@plugin_hook(
    "sgl_kernel.flash_mla.get_mla_metadata",
    type=HookType.REPLACE,
)
def _ppu_get_mla_metadata(*args, **kwargs):
    from sglang.srt.hardware_backend.ppu.attention.flash_mla import get_mla_metadata

    return get_mla_metadata(*args, **kwargs)


# ---------------------------------------------------------------------------
# 4. init_cuda_graph_state — cap cuda_graph_mla_metadata buffer
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.layers.attention.flashmla_backend.FlashMLABackend.init_cuda_graph_state",
    type=HookType.AROUND,
)
def _ppu_flashmla_cuda_graph_state(original_fn, self, *args, **kwargs):
    original_fn(self, *args, **kwargs)
    buf = self.cuda_graph_mla_metadata
    if buf is not None and buf.shape[0] > _PPU_MAX_SM_PARTS:
        self.cuda_graph_mla_metadata = torch.empty(
            (_PPU_MAX_SM_PARTS, 8),
            dtype=torch.int32,
            device=buf.device,
        )


# ---------------------------------------------------------------------------
# 5. _apply_decode_target_verify_metadata — PPU metadata handling
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.layers.attention.flashmla_backend.FlashMLABackend._apply_decode_target_verify_metadata",
    type=HookType.REPLACE,
)
def _ppu_flashmla_apply_decode_target_verify_metadata(
    self, bs, req_pool_indices, seq_lens, seq_lens_cpu, forward_mode
):
    """Replace _apply_decode_target_verify_metadata on PPU.

    Key differences from the community version:
    * Uses ``scheduler_metadata, _ = get_mla_metadata(...)`` (discards
      second return value) instead of ``mla_metadata, num_splits = ...``.
    * Contains ``if is_ppu(): raise`` guard inside the fp8 kvcache path.
    * Sets ``.tile_scheduler_metadata`` and ``.num_splits`` attributes on
      the scheduler_metadata object for CUDA graph buffer views.
    * Creates ``FlashMLADecodeMetadata`` with sched-meta as flashmla_metadata
      and None as num_splits (no separate num_splits).
    """
    import logging

    import triton

    from sglang.kernels.ops.attention.utils import (
        create_flashmla_kv_indices_triton,
        get_num_kv_index_blocks_flashmla,
    )
    from sglang.srt.hardware_backend.ppu.attention.flash_mla import get_mla_metadata
    from sglang.srt.layers.attention.flashmla_backend import (
        PAGE_SIZE,
        FlashMLADecodeMetadata,
    )
    from sglang.srt.utils import is_ppu

    _logger = logging.getLogger(__name__)

    if True:
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs] if seq_lens_cpu is not None else None

        if forward_mode.is_target_verify():
            seq_lens = seq_lens + self.num_draft_tokens
            if seq_lens_cpu is not None:
                seq_lens_cpu = seq_lens_cpu + self.num_draft_tokens

        seq_max = (
            seq_lens_cpu.max().item()
            if seq_lens_cpu is not None
            else seq_lens.max().item()
        )
        max_seqlen_pad = triton.cdiv(seq_max, PAGE_SIZE)

        create_flashmla_kv_indices_triton[
            (
                bs,
                get_num_kv_index_blocks_flashmla(
                    self.cuda_graph_kv_indices.stride(0), PAGE_SIZE
                ),
            )
        ](
            self.req_to_token,
            req_pool_indices[:bs],
            seq_lens,
            None,
            self.cuda_graph_kv_indices,
            self.req_to_token.stride(0),
            self.cuda_graph_kv_indices.stride(0),
        )

        q_head_mult = (
            self.num_draft_tokens
            if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()
            else 1
        )
        scheduler_metadata, _ = get_mla_metadata(
            seq_lens.to(torch.int32),
            q_head_mult * self.num_q_heads,
            1,
            is_fp8_kvcache=self.is_fp8_kvcache,
        )
        if self.is_fp8_kvcache:
            if is_ppu():
                raise
            mla_metadata, num_splits = get_mla_metadata(
                seq_lens.to(torch.int32),
                q_head_mult * self.num_q_heads,
                1,
                is_fp8_kvcache=self.is_fp8_kvcache,
            )

            actual_num_sm_parts = mla_metadata.shape[0]
            assert actual_num_sm_parts <= self.cuda_graph_mla_metadata.shape[0], (
                f"num_sm_parts {actual_num_sm_parts} exceeds preallocated max "
                f"{self.cuda_graph_mla_metadata.shape[0]}"
            )

            if (
                self.cuda_graph_mla_metadata_view is None
                or actual_num_sm_parts != self.cuda_graph_mla_metadata_view.shape[0]
            ):
                if self.cuda_graph_mla_metadata_view is not None:
                    _logger.warning(
                        f"num_sm_parts mismatch in CUDA Graph replay: "
                        f"capture={self.cuda_graph_mla_metadata_view.shape[0]}, "
                        f"replay={actual_num_sm_parts}. "
                        f"This may indicate batch size changed between capture and replay."
                    )
                self.cuda_graph_mla_metadata_view = self.cuda_graph_mla_metadata[
                    :actual_num_sm_parts
                ]
            # num_splits has shape (bs+1,) -- always update for the current bs.
            self.cuda_graph_num_splits_view = self.cuda_graph_num_splits[: bs + 1]

            self.cuda_graph_mla_metadata[:actual_num_sm_parts].copy_(mla_metadata)
            self.cuda_graph_num_splits[: bs + 1].copy_(num_splits)
            scheduler_metadata.tile_scheduler_metadata = (
                self.cuda_graph_mla_metadata_view
            )
            scheduler_metadata.num_splits = self.cuda_graph_num_splits_view

        seq_lens_k = None
        if forward_mode.is_draft_extend_v2():
            # The graph kernel binds this static buffer; refresh per replay.
            self.cuda_graph_draft_extend_seq_lens_k[:bs].copy_(seq_lens)
            seq_lens_k = self.cuda_graph_draft_extend_seq_lens_k[:bs]

        self.forward_metadata = FlashMLADecodeMetadata(
            scheduler_metadata,
            None,
            self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            seq_lens_k if seq_lens_k is not None else seq_lens.to(torch.int32),
        )


# ---------------------------------------------------------------------------
# 6. init_forward_metadata_in_graph — reset flashmla_metadata
# ---------------------------------------------------------------------------


@plugin_hook(
    "sglang.srt.layers.attention.flashmla_backend.FlashMLABackend.init_forward_metadata_in_graph",
    type=HookType.REPLACE,
)
def _ppu_flashmla_init_forward_metadata_in_graph(self, forward_batch):
    """Replace init_forward_metadata_in_graph on PPU.

    For PPU FlashMLA, `get_mla_metadata` runs only on the first
    `flash_mla_with_kvcache` call. To capture this init logic in
    the CUDA Graph, reset flashmla_metadata before starting capture.
    This forces metadata generation to occur within the capture scope.
    """
    from sglang.srt.layers.attention.flashmla_backend import FlashMLADecodeMetadata

    if isinstance(self.forward_metadata, FlashMLADecodeMetadata):
        assert self.forward_metadata.flashmla_metadata is not None
        self.forward_metadata.flashmla_metadata.have_initialized = False
