"""PPU FA3 ops hooks.

Uses ``@plugin_hook`` with ``HookType.REPLACE`` to replace the module-level
FA3 functions (``flash_attn_varlen_func``, ``flash_attn_with_kvcache``) in
the ``flashattention_backend`` module, and ``HookType.AROUND`` on
``FlashAttentionBackend.__init__`` to replace the instance-level
``_get_scheduler_metadata`` attribute.

This avoids any modification to ``flashattention_backend.py`` itself — the
PPU ops are injected purely through the plugin hook mechanism.
"""

from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.attention.flashattention_backend.flash_attn_varlen_func",
    type=HookType.REPLACE,
)
def _ppu_flash_attn_varlen_func(*args, **kwargs):
    from sglang.srt.hardware_backend.ppu.attention.flash_attention import (
        flash_attn_varlen_func,
    )

    return flash_attn_varlen_func(*args, **kwargs)


@plugin_hook(
    "sglang.srt.layers.attention.flashattention_backend.flash_attn_with_kvcache",
    type=HookType.REPLACE,
)
def _ppu_flash_attn_with_kvcache(*args, **kwargs):
    from sglang.srt.hardware_backend.ppu.attention.flash_attention import (
        flash_attn_with_kvcache,
    )

    return flash_attn_with_kvcache(*args, **kwargs)


@plugin_hook(
    "sglang.srt.layers.attention.flashattention_backend.FlashAttentionBackend.__init__",
    type=HookType.AROUND,
)
def _ppu_fa3_init(original_fn, self, *args, **kwargs):
    original_fn(self, *args, **kwargs)
    if self.fa_impl_ver == 3:
        from sglang.srt.hardware_backend.ppu.attention.flash_attention import (
            get_scheduler_metadata,
        )

        self._get_scheduler_metadata = get_scheduler_metadata
