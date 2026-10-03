"""PPU DeepEP hooks.

Uses ``@plugin_hook`` to inject PPU-specific behavior into the DeepEP token
dispatcher without modifying the community source code.

Registered hooks
~~~~~~~~~~~~~~~~

1. ``_is_mnnvl_fabric_supported`` (REPLACE) — Reads
   ``SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED`` so PPU deployments can explicitly
   enable or disable MNNVL FABRIC handles.
"""

from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import HookType, plugin_hook


@plugin_hook(
    "sglang.srt.layers.moe.token_dispatcher.deepep._is_mnnvl_fabric_supported",
    type=HookType.REPLACE,
)
def _ppu_is_mnnvl_fabric_supported() -> bool:
    return envs.SGLANG_SAIL_MNNVL_FABRIC_SUPPORTED.get()
