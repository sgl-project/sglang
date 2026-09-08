"""Portable Direct INT4 format and codec, independent of the serving runtime."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.mem_cache import kvbit_dsv4_layout_constants as abi


@dataclass(frozen=True)
class DSV4KVBitLayout:
    layout_id: str = "aos_368"

    def __post_init__(self) -> None:
        if self.layout_id not in ("aos_368", "aos_384"):
            raise ValueError(f"Unsupported DSV4 INT4 layout: {self.layout_id!r}")

    nope_dim = abi.NOPE_DIM
    rope_dim = abi.ROPE_DIM
    group_size = abi.GROUP_SIZE
    bits = abi.BITS
    code_bytes = abi.CODE_BYTES
    header_bytes = abi.HEADER_BYTES
    header_offset = abi.HEADER_OFFSET
    rope_offset = abi.ROPE_OFFSET
    rope_bytes = abi.ROPE_BYTES
    payload_bytes = abi.PAYLOAD_BYTES
    format_id = abi.FORMAT_ID
    version = abi.FORMAT_VERSION

    @property
    def row_bytes(self) -> int:
        return (
            abi.COMPACT_ROW_BYTES
            if self.layout_id == "aos_368"
            else abi.ALIGNED_ROW_BYTES
        )

    def offsets(self) -> dict[str, tuple[int, int]]:
        offsets = {
            "codes": (0, self.code_bytes),
            "header": (self.header_offset, self.rope_offset),
            "rope": (self.rope_offset, self.payload_bytes),
        }
        if self.row_bytes != self.payload_bytes:
            offsets["padding"] = (self.payload_bytes, self.row_bytes)
        return offsets


DSV4_INT4_LAYOUT = DSV4KVBitLayout()
DSV4_INT4_ALIGNED_LAYOUT = DSV4KVBitLayout("aos_384")


def layout_for_row_bytes(row_bytes: int) -> DSV4KVBitLayout:
    for layout in (DSV4_INT4_LAYOUT, DSV4_INT4_ALIGNED_LAYOUT):
        if row_bytes == layout.row_bytes:
            return layout
    raise ValueError(f"packed row width must be 368 or 384, got {row_bytes}")


def validate_dsv4_int4_geometry(nope_dim: int, rope_dim: int) -> None:
    if (nope_dim, rope_dim) != (abi.NOPE_DIM, abi.ROPE_DIM):
        raise ValueError(
            "DSV4 INT4 supports only the built-in 448-nope/64-rope "
            f"layout, got {nope_dim}-nope/{rope_dim}-rope."
        )


def validate_dsv4_int4_attention(
    *,
    num_attention_heads: int,
    attn_tp_size: int,
    nope_dim: int,
    rope_dim: int,
    head_dim_v: int,
    kv_heads: int,
    sparse_width: int,
) -> None:
    validate_dsv4_int4_geometry(nope_dim, rope_dim)
    if (
        attn_tp_size <= 0
        or num_attention_heads <= 0
        or num_attention_heads % attn_tp_size != 0
        or num_attention_heads // attn_tp_size > 64
        or (attn_tp_size == 1 and num_attention_heads != 64)
    ):
        raise ValueError(
            "DSV4 INT4 requires 64 local query heads after model TP padding; "
            f"num_attention_heads={num_attention_heads}, attn_tp_size={attn_tp_size}."
        )
    if head_dim_v != 512 or kv_heads != 1:
        raise ValueError(
            "DSV4 INT4 requires MQA with head_dim_v=512 and kv_heads=1; "
            f"got head_dim_v={head_dim_v}, kv_heads={kv_heads}."
        )
    if sparse_width <= 0 or sparse_width % 64:
        raise ValueError(
            f"DSV4 INT4 requires a positive sparse width divisible by 64, got {sparse_width}."
        )


def _require_cpu_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be a CPU tensor")


def encode_dsv4_int4_reference(
    kv: torch.Tensor, *, layout: DSV4KVBitLayout = DSV4_INT4_LAYOUT
) -> torch.Tensor:
    """Encode G32 signed INT4/E4M3-nearest NoPE and raw BF16 RoPE."""
    _require_cpu_tensor(kv, name="kv")
    expected_dim = layout.nope_dim + layout.rope_dim
    if kv.ndim < 1 or kv.shape[-1] != expected_dim:
        raise ValueError(f"kv last dimension must be {expected_dim}, got {kv.shape}")
    if not kv.is_floating_point():
        raise TypeError(f"kv must be floating point, got {kv.dtype}")

    leading_shape = kv.shape[:-1]
    rows = kv.reshape(-1, expected_dim)
    num_rows = rows.shape[0]
    num_groups = layout.nope_dim // layout.group_size
    nope = (
        rows[:, : layout.nope_dim]
        .float()
        .reshape(num_rows, num_groups, layout.group_size)
    )
    max_abs = nope.abs().amax(dim=-1)
    stored_step = (max_abs / 7.0).clamp_max(torch.finfo(torch.float8_e4m3fn).max)
    stored_step = stored_step.to(torch.float8_e4m3fn)
    quant_step = stored_step.float()
    has_step = quant_step > 0
    safe_step = torch.where(has_step, quant_step, torch.ones_like(quant_step))
    codes = torch.round(nope / safe_step.unsqueeze(-1)).clamp_(-7, 7).to(torch.int8)
    codes = torch.where(has_step.unsqueeze(-1), codes, torch.zeros_like(codes))
    unsigned_codes = codes.to(torch.uint8) & 0x0F
    packed_codes = (
        unsigned_codes[..., 0::2] | (unsigned_codes[..., 1::2] << 4)
    ).reshape(num_rows, layout.code_bytes)
    packed = torch.zeros((num_rows, layout.row_bytes), dtype=torch.uint8)
    packed[:, : layout.code_bytes] = packed_codes
    packed[:, layout.header_offset : layout.header_offset + num_groups] = (
        stored_step.contiguous().view(torch.uint8).reshape(num_rows, num_groups)
    )
    packed[:, layout.rope_offset : layout.payload_bytes] = (
        rows[:, layout.nope_dim :]
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .reshape(num_rows, layout.rope_bytes)
    )
    return packed.reshape(*leading_shape, layout.row_bytes)


def decode_dsv4_int4_reference(packed: torch.Tensor) -> torch.Tensor:
    """Decode either physical stride; trailing alignment padding is ignored."""
    _require_cpu_tensor(packed, name="packed")
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed must have dtype torch.uint8, got {packed.dtype}")
    if packed.ndim < 1:
        raise ValueError("packed must have a row dimension")
    layout = layout_for_row_bytes(packed.shape[-1])
    leading_shape = packed.shape[:-1]
    rows = packed.reshape(-1, layout.row_bytes)
    num_rows = rows.shape[0]
    num_groups = layout.nope_dim // layout.group_size
    packed_codes = rows[:, : layout.code_bytes]
    codes = torch.stack((packed_codes & 0x0F, packed_codes >> 4), dim=-1)
    codes = codes.reshape(num_rows, num_groups, layout.group_size).to(torch.int8)
    codes = torch.where(codes >= 8, codes - 16, codes).float()
    steps = (
        rows[:, layout.header_offset : layout.header_offset + num_groups]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    nope = (codes * steps.unsqueeze(-1)).reshape(num_rows, layout.nope_dim)
    rope = (
        rows[:, layout.rope_offset : layout.payload_bytes]
        .contiguous()
        .view(torch.bfloat16)
        .reshape(num_rows, layout.rope_dim)
    )
    decoded = torch.cat((nope.to(torch.bfloat16), rope), dim=-1)
    return decoded.reshape(*leading_shape, layout.nope_dim + layout.rope_dim)


def repack_dsv4_int4_reference(
    packed: torch.Tensor, *, layout: DSV4KVBitLayout
) -> torch.Tensor:
    """Copy the encoded payload to a new stride without requantization."""
    _require_cpu_tensor(packed, name="packed")
    if packed.dtype != torch.uint8 or packed.ndim < 1:
        raise ValueError("packed must be a torch.uint8 tensor with a row dimension")
    source = layout_for_row_bytes(packed.shape[-1])
    result = torch.zeros((*packed.shape[:-1], layout.row_bytes), dtype=torch.uint8)
    result[..., : source.payload_bytes] = packed[..., : source.payload_bytes]
    return result
