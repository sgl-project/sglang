from sglang.srt.layers.quantization.scalar_type import scalar_types

WNA16_SUPPORTED_TYPES_MAP = {4: scalar_types.uint4b8, 8: scalar_types.uint8b128}
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())
