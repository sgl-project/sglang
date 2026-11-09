"""Page2D NeoX RoPE application kernels (Triton, bf16 bit-exact with PyTorch)."""

from __future__ import annotations

import torch


def _rope2d_torch(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, head_dim: int
) -> torch.Tensor:
    t = x.shape[0]
    v = x.view(t, -1, head_dim)
    x1 = v[..., : head_dim // 2]
    x2 = v[..., head_dim // 2 :]
    rot = torch.cat((-x2, x1), dim=-1)
    # Same dtype as training: cos/sin are built in fp32 and cast down before the
    # multiply, so the rotation itself runs in the model dtype.
    out = v * cos[:, None, :] + rot * sin[:, None, :]
    return out.view_as(x)


_rope_kernel = None
_rope_kernel_pair = None


def _get_rope_kernel():
    """Lazily built so this module still imports with torch alone (CPU parity tool)."""
    global _rope_kernel
    if _rope_kernel is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _k(
            x_ptr,
            o_ptr,
            cos_ptr,
            sin_ptr,
            n_heads,
            sx,
            sc,
            HD: tl.constexpr,
            HALF: tl.constexpr,
        ):
            pid = tl.program_id(0)
            row = pid // n_heads
            head = pid % n_heads
            base = row * sx + head * HD
            d = tl.arange(0, HD)
            xp = x_ptr + base + d
            x = tl.load(xp)
            # rotate_half: rot = cat(-x2, x1)
            other = tl.where(d < HALF, d + HALF, d - HALF)
            xo = tl.load(x_ptr + base + other)
            rot = tl.where(d < HALF, -xo, xo)
            c = tl.load(cos_ptr + row * sc + d)
            s = tl.load(sin_ptr + row * sc + d)
            # Match PyTorch bf16 by rounding each multiply before the add.
            # Integer-domain RTNE prevents truncation and fp32 fusion.
            v = x.to(tl.float32) * c.to(tl.float32)
            u = v.to(tl.uint32, bitcast=True)
            u = u + 0x7FFF + ((u >> 16) & 1)
            t1 = (u & 0xFFFF0000).to(tl.float32, bitcast=True)
            v = rot.to(tl.float32) * s.to(tl.float32)
            u = v.to(tl.uint32, bitcast=True)
            u = u + 0x7FFF + ((u >> 16) & 1)
            t2 = (u & 0xFFFF0000).to(tl.float32, bitcast=True)
            v = t1 + t2
            u = v.to(tl.uint32, bitcast=True)
            u = u + 0x7FFF + ((u >> 16) & 1)
            # The low 16 bits are already zero, so this last cast is exact.
            o = (u & 0xFFFF0000).to(tl.float32, bitcast=True).to(x.dtype)
            tl.store(o_ptr + base + d, o)

        _rope_kernel = _k
    return _rope_kernel


def apply_page2d_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, head_dim: int
) -> torch.Tensor:
    """Apply NeoX RoPE to ``[T, H * head_dim]``.

    Eligible bf16 layouts use a bit-identical fused Triton kernel; other dtypes
    and layouts use the PyTorch path.
    """
    t = x.shape[0]
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and t > 0
        and x.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
        and cos.dtype == x.dtype
        and sin.dtype == x.dtype
        and cos.shape == (t, head_dim)
        and sin.shape == (t, head_dim)
        and head_dim >= 2
        and head_dim & (head_dim - 1) == 0
        and x.shape[1] % head_dim == 0
    ):
        return _rope2d_torch(x, cos, sin, head_dim)
    n_heads = x.shape[1] // head_dim
    out = torch.empty_like(x)
    _get_rope_kernel()[(t * n_heads,)](
        x,
        out,
        cos,
        sin,
        n_heads,
        x.stride(0),
        cos.stride(0),
        HD=head_dim,
        HALF=head_dim // 2,
    )
    return out


def _get_rope_kernel_pair():
    """Two tensors, one launch; each row runs the single-tensor kernel's body."""
    global _rope_kernel_pair
    if _rope_kernel_pair is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _row(
            x_ptr,
            o_ptr,
            base,
            cos_ptr,
            sin_ptr,
            crow,
            HD: tl.constexpr,
            HALF: tl.constexpr,
        ):
            d = tl.arange(0, HD)
            x = tl.load(x_ptr + base + d)
            other = tl.where(d < HALF, d + HALF, d - HALF)
            xo = tl.load(x_ptr + base + other)
            rot = tl.where(d < HALF, -xo, xo)
            c = tl.load(cos_ptr + crow + d)
            s = tl.load(sin_ptr + crow + d)
            # Same integer-domain rounding schedule as _get_rope_kernel -- see the
            # comment there for why it must go through uint32 to stay bit-exact.
            v = x.to(tl.float32) * c.to(tl.float32)
            u = v.to(tl.uint32, bitcast=True)
            u = u + 0x7FFF + ((u >> 16) & 1)
            t1 = (u & 0xFFFF0000).to(tl.float32, bitcast=True)
            v = rot.to(tl.float32) * s.to(tl.float32)
            u = v.to(tl.uint32, bitcast=True)
            u = u + 0x7FFF + ((u >> 16) & 1)
            t2 = (u & 0xFFFF0000).to(tl.float32, bitcast=True)
            v = t1 + t2
            u = v.to(tl.uint32, bitcast=True)
            u = u + 0x7FFF + ((u >> 16) & 1)
            o = (u & 0xFFFF0000).to(tl.float32, bitcast=True).to(x.dtype)
            tl.store(o_ptr + base + d, o)

        @triton.jit
        def _k2(
            x_ptr,
            ox_ptr,
            y_ptr,
            oy_ptr,
            cos_ptr,
            sin_ptr,
            nx,
            ny,
            sx,
            sy,
            sc,
            HD: tl.constexpr,
            HALF: tl.constexpr,
        ):
            pid = tl.program_id(0)
            total = nx + ny
            row = pid // total
            h = pid % total
            crow = row * sc
            if h < nx:
                _row(x_ptr, ox_ptr, row * sx + h * HD, cos_ptr, sin_ptr, crow, HD, HALF)
            else:
                _row(
                    y_ptr,
                    oy_ptr,
                    row * sy + (h - nx) * HD,
                    cos_ptr,
                    sin_ptr,
                    crow,
                    HD,
                    HALF,
                )

        _rope_kernel_pair = _k2
    return _rope_kernel_pair


def _rope_pair_ok(x, y, cos, sin, head_dim, t) -> bool:
    return (
        x.is_cuda
        and y.is_cuda
        and x.dtype == torch.bfloat16
        and y.dtype == torch.bfloat16
        and t > 0
        and y.shape[0] == t
        and x.is_contiguous()
        and y.is_contiguous()
        and cos.is_contiguous()
        and sin.is_contiguous()
        and cos.dtype == x.dtype
        and sin.dtype == x.dtype
        and cos.shape == (t, head_dim)
        and sin.shape == (t, head_dim)
        and head_dim >= 2
        and head_dim & (head_dim - 1) == 0
        and x.shape[1] % head_dim == 0
        and y.shape[1] % head_dim == 0
    )


def apply_page2d_rope_pair(
    x: torch.Tensor,
    y: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply ``apply_page2d_rope`` to both tensors, fusing when eligible."""
    t = x.shape[0]
    if not _rope_pair_ok(x, y, cos, sin, head_dim, t):
        return (
            apply_page2d_rope(x, cos, sin, head_dim),
            apply_page2d_rope(y, cos, sin, head_dim),
        )
    nx = x.shape[1] // head_dim
    ny = y.shape[1] // head_dim
    ox = torch.empty_like(x)
    oy = torch.empty_like(y)
    _get_rope_kernel_pair()[(t * (nx + ny),)](
        x,
        ox,
        y,
        oy,
        cos,
        sin,
        nx,
        ny,
        x.stride(0),
        y.stride(0),
        cos.stride(0),
        HD=head_dim,
        HALF=head_dim // 2,
    )
    return ox, oy
