# pyright: reportMissingImports=false

import torch
import triton


def _compute_cos_sin_cache_half(
    max_seq_len: int,
    rotary_dim: int,
    base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def sglang_aot_rotary_positions(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    interleaved: bool,
    cos_sin_cache: torch.Tensor,
) -> None:
    # NOTE: `sgl_kernel` positions-kernel uses `is_neox` flag internally:
    # - is_neox=True  => x_index=i, y_index=embed_dim+i   (split halves)
    # - is_neox=False => x_index=2*i, y_index=2*i+1       (interleaved pairs)
    from sgl_kernel.rotary_embedding import rotary_embedding

    rotary_embedding(
        positions=positions,
        query=q,
        key=k,
        head_size=head_size,
        is_neox=not interleaved,
        cos_sin_cache=cos_sin_cache,
    )


def sglang_jit_rotary_cos_sin(
    cos: torch.Tensor,
    sin: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    interleaved: bool,
) -> None:
    from sglang.jit_kernel.rotary_embedding import rotary_embedding_cos_sin

    rotary_embedding_cos_sin(cos, sin, q, k, head_size, interleaved)


@torch.no_grad()
def torch_impl_rotary_fp32(
    cos: torch.Tensor,
    sin: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    head_size: int,
    interleaved: bool,
) -> None:
    orig_dtype = q.dtype
    if interleaved and cos.shape[1] == head_size:
        half = head_size // 2
        cos = cos.view(cos.shape[0], half, 2).select(2, 0).contiguous()
        sin = sin.view(sin.shape[0], half, 2).select(2, 1).contiguous()

    cos_f, sin_f = cos.float(), sin.float()
    q_f, k_f = q.float(), k.float()

    if interleaved:
        embed_dim = int(cos_f.shape[1])
        rot_dim = embed_dim * 2
        cos_b = cos_f[:, None, :embed_dim]
        sin_b = sin_f[:, None, :embed_dim]

        def _apply(x: torch.Tensor) -> None:
            xr = x[..., :rot_dim]
            xr2 = xr.view(xr.shape[0], xr.shape[1], embed_dim, 2)
            x0 = xr2[..., 0].clone()
            x1 = xr2[..., 1].clone()
            xr2[..., 0].copy_(x0 * cos_b - x1 * sin_b)
            xr2[..., 1].copy_(x1 * cos_b + x0 * sin_b)

    else:
        if cos_f.shape[1] == head_size // 2:
            embed_dim = int(cos_f.shape[1])
            rot_dim = embed_dim * 2
            cos_x, sin_x = cos_f[:, None, :], sin_f[:, None, :]
            cos_y, sin_y = cos_x, sin_x
        else:
            embed_dim = int(cos_f.shape[1]) // 2
            rot_dim = embed_dim * 2
            cos_x, sin_x = cos_f[:, None, :embed_dim], sin_f[:, None, :embed_dim]
            cos_y, sin_y = (
                cos_f[:, None, embed_dim:rot_dim],
                sin_f[:, None, embed_dim:rot_dim],
            )

        def _apply(x: torch.Tensor) -> None:
            xr = x[..., :rot_dim]
            x0 = xr[..., :embed_dim].clone()
            x1 = xr[..., embed_dim:rot_dim].clone()
            xr[..., :embed_dim].copy_(x0 * cos_x - x1 * sin_x)
            xr[..., embed_dim:rot_dim].copy_(x1 * cos_y + x0 * sin_y)

    _apply(q_f)
    _apply(k_f)
    q.copy_(q_f.to(orig_dtype))
    k.copy_(k_f.to(orig_dtype))


def main():
    DEVICE = "cuda"
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    NUM_Q_HEADS = 32
    NUM_KV_HEADS = 8
    BS_LIST = [2**n for n in range(0, 13)]
    BS_LIST += [x + 1 + i for i, x in enumerate(BS_LIST)]

    for DTYPE in [torch.bfloat16, torch.float16]:
        for HEAD_SIZE in [64, 80, 96, 128, 256, 320]:
            rotary_dim = HEAD_SIZE
            max_seq_len = 8192
            cos_cache, sin_cache = _compute_cos_sin_cache_half(
                max_seq_len, rotary_dim, dtype=DTYPE
            )
            cos_cache, sin_cache = cos_cache.to(DEVICE), sin_cache.to(DEVICE)

            for INTERLEAVED in [True, False]:
                aot_cos_sin_cache = torch.cat(
                    [cos_cache, sin_cache], dim=-1
                ).contiguous()

                for BS in BS_LIST:
                    positions = (
                        torch.arange(BS, device=DEVICE, dtype=torch.int64) % max_seq_len
                    )
                    cos_half = cos_cache[positions].contiguous()
                    sin_half = sin_cache[positions].contiguous()

                    if INTERLEAVED:
                        cos, sin = cos_half, sin_half
                    else:
                        cos = torch.cat([cos_half, cos_half], dim=-1).contiguous()
                        sin = torch.cat([sin_half, sin_half], dim=-1).contiguous()

                    q = torch.randn(
                        BS, NUM_Q_HEADS, HEAD_SIZE, device=DEVICE, dtype=DTYPE
                    )
                    k = torch.randn(
                        BS, NUM_KV_HEADS, HEAD_SIZE, device=DEVICE, dtype=DTYPE
                    )

                    q_ref_fp32, k_ref_fp32 = q.clone(), k.clone()
                    torch_impl_rotary_fp32(
                        cos, sin, q_ref_fp32, k_ref_fp32, HEAD_SIZE, INTERLEAVED
                    )

                    q_k_aot = (q.clone(), k.clone())
                    q_k_jit = (q.clone(), k.clone())
                    sglang_aot_rotary_positions(
                        positions,
                        q_k_aot[0],
                        q_k_aot[1],
                        HEAD_SIZE,
                        INTERLEAVED,
                        aot_cos_sin_cache,
                    )
                    sglang_jit_rotary_cos_sin(
                        cos, sin, q_k_jit[0], q_k_jit[1], HEAD_SIZE, INTERLEAVED
                    )

                    ref_atol = 2e-2 if DTYPE == torch.bfloat16 else 2e-3
                    ref_rtol = 2e-2 if DTYPE == torch.bfloat16 else 2e-3

                    triton.testing.assert_close(
                        q_ref_fp32, q_k_aot[0], atol=ref_atol, rtol=ref_rtol
                    )
                    triton.testing.assert_close(
                        k_ref_fp32, q_k_aot[1], atol=ref_atol, rtol=ref_rtol
                    )
                    triton.testing.assert_close(
                        q_ref_fp32, q_k_jit[0], atol=ref_atol, rtol=ref_rtol
                    )
                    triton.testing.assert_close(
                        k_ref_fp32, q_k_jit[1], atol=ref_atol, rtol=ref_rtol
                    )

                    print(
                        f"HEAD_SIZE={HEAD_SIZE} interleaved={INTERLEAVED} dtype={DTYPE} passed."
                    )


if __name__ == "__main__":
    main()
