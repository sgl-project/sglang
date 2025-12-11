from typing import Optional
import torch

# Detect available kernels (old and new names).
_HAS_COS_SIN = hasattr(torch.ops.sgl_kernel, "rotary_embedding_cos_sin")
_HAS_GENERIC = hasattr(torch.ops.sgl_kernel, "rotary_embedding")

print(f"HAS_COS_SIN: {_HAS_COS_SIN}")
print(f"HAS_GENERIC: {_HAS_GENERIC}")

def apply_rotary_embedding(
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    head_size: int = 0,
    interleaved: Optional[bool] = None,
    *,
    is_neox: Optional[bool] = None,
) -> None:
    """Apply rotary embedding with precomputed cos/sin.

    - `interleaved=True` (NeoX) layout: `[x0, y0, x1, y1, ...]`
    - `interleaved=False` (GPT-J/LLaMA) layout: `[x0, x1, ..., y0, y1, ...]`
    - `is_neox` is kept as alias to stay consistent with other call-sites.
    """
    if interleaved is not None and is_neox is not None and interleaved != is_neox:
        raise ValueError(
            f"is_neox({is_neox}) and interleaved({interleaved}) mismatch; keep only one or make them equal."
        )

    if is_neox is not None:
        effective_interleaved = is_neox
    elif interleaved is not None:
        effective_interleaved = interleaved
    else:
        effective_interleaved = True  # default NeoX for backward compatibility


    # Legacy explicit name.
    if _HAS_COS_SIN:
        torch.ops.sgl_kernel.rotary_embedding_cos_sin(
            cos,
            sin,
            query,
            key if key is not None else None,
            head_size,
            effective_interleaved,
        )
        return

    # Some older builds overload `rotary_embedding` directly with cos/sin signature.
    if _HAS_GENERIC:
        try:
            torch.ops.sgl_kernel.rotary_embedding(
                cos,
                sin,
                query,
                key if key is not None else None,
                head_size,
                effective_interleaved,
            )
            return
        except Exception:
            pass

    # Fallback: pack cos/sin into cache and call positions+cache variant.
    positions = torch.arange(cos.size(0), device=cos.device, dtype=torch.int64)
    cos_sin_cache = torch.cat([cos, sin], dim=1)
    # Final fallback to legacy positions kernel name.
    if _HAS_GENERIC:
        torch.ops.sgl_kernel.rotary_embedding(
            positions=positions,
            query=query,
            key=key if key is not None else None,
            head_size=head_size,
            cos_sin_cache=cos_sin_cache,
            is_neox=effective_interleaved,
        )
        return

    raise RuntimeError("No rotary embedding kernel is available in torch.ops.sgl_kernel")


# Backward-compatible aliases
def rotary_embedding_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    head_size: int = 0,
    interleaved: Optional[bool] = None,
    *,
    is_neox: Optional[bool] = None,
) -> None:
    apply_rotary_embedding(
        cos,
        sin,
        query,
        key=key,
        head_size=head_size,
        interleaved=interleaved,
        is_neox=is_neox,
    )


def rotary_embedding(
    cos: torch.Tensor,
    sin: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor] = None,
    head_size: int = 0,
    interleaved: Optional[bool] = None,
    *,
    is_neox: Optional[bool] = None,
) -> None:
    apply_rotary_embedding(
        cos,
        sin,
        query,
        key=key,
        head_size=head_size,
        interleaved=interleaved,
        is_neox=is_neox,
    )