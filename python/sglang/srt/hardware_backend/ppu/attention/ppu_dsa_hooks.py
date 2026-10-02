"""PPU DSA backend hooks.

Uses ``@plugin_hook`` to inject PPU-specific behavior into DSA backend
methods without modifying the community source code.

Registered hooks:

1. ``DSAFlashMLAMetadata.copy_`` (REPLACE) — Skips the copy on PPU because
   PPU metadata tensors are cached/immutable and must not be overwritten
   during CUDA graph replay.

2. ``DSAFlashMLAMetadata.slice`` (REPLACE) — Returns ``num_splits`` without
   slicing on PPU because the tensor shape semantics differ from CUDA.

3. ``DeepseekSparseAttnBackend.init_forward_metadata_in_graph`` (REPLACE) —
   Resets ``flashmla_metadata.have_initialized`` before CUDA graph capture
   so that PPU metadata generation occurs within the capture scope.

4. ``DeepseekSparseAttnBackend.init_forward_metadata_replay_cuda_graph_from_precomputed``
   (AROUND) — Skips flashmla_metadata copy on PPU by temporarily clearing
   and restoring the precomputed metadata.
"""

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.attention.dsa_backend.DSAFlashMLAMetadata.copy_",
    type=HookType.REPLACE,
)
def _ppu_dsa_flashmla_copy(self, other):
    """Skip copy on PPU — metadata is cached/immutable."""
    return


@plugin_hook(
    "sglang.srt.layers.attention.dsa_backend.DSAFlashMLAMetadata.slice",
    type=HookType.REPLACE,
)
def _ppu_dsa_flashmla_slice(self, sli):
    """Skip num_splits slicing on PPU — shape semantics differ from CUDA."""
    from sglang.srt.layers.attention.dsa_backend import DSAFlashMLAMetadata

    return DSAFlashMLAMetadata(
        flashmla_metadata=self.flashmla_metadata,
        num_splits=self.num_splits,
    )


@plugin_hook(
    "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.init_forward_metadata_in_graph",
    type=HookType.REPLACE,
)
def _ppu_dsa_init_forward_metadata_in_graph(self, forward_batch):
    """Replace init_forward_metadata_in_graph on PPU.

    For PPU FlashMLA, `get_mla_metadata` runs only on the first
    `flash_mla_with_kvcache` call. To capture this init logic in
    the CUDA Graph, reset flashmla_metadata before starting capture.
    This forces metadata generation to occur within the capture scope.
    """
    if self.dsa_decode_impl == "flashmla_kv":
        assert self.forward_metadata.flashmla_metadata is not None
        assert self.forward_metadata.flashmla_metadata.flashmla_metadata is not None
        self.forward_metadata.flashmla_metadata.flashmla_metadata.have_initialized = (
            False
        )


@plugin_hook(
    "sglang.srt.layers.attention.dsa_backend.DeepseekSparseAttnBackend.init_forward_metadata_replay_cuda_graph_from_precomputed",
    type=HookType.AROUND,
)
def _ppu_dsa_init_forward_metadata_replay_cuda_graph_from_precomputed(
    original_fn, self, bs, precomputed, forward_mode
):
    """Skip flashmla_metadata copy on PPU."""
    flashmla_metadata = precomputed.flashmla_metadata
    precomputed.flashmla_metadata = None
    original_fn(self, bs, precomputed, forward_mode)
    precomputed.flashmla_metadata = flashmla_metadata
