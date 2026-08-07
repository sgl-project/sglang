"""Compatibility imports for the multimodal MNNVL FABRIC transport.

New code should import from :mod:`sglang.srt.multimodal.transport.fabric`.
"""

from sglang.srt.multimodal.transport.fabric import (
    FabricMmFeatureMemoryPool,
    FabricTensorTransportProxy,
)

__all__ = ["FabricMmFeatureMemoryPool", "FabricTensorTransportProxy"]
