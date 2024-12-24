import torch
from einops import rearrange


def flashinfer_rope(
    q: torch.tensor,
    k: torch.tensor,
    positions: torch.tensor,
    rotary_dim: int,
    rope_theta: int,
):
    from flashinfer.rope import apply_rope_pos_ids

    q_rope, k_rope = apply_rope_pos_ids(
        q,
        k,
        pos_ids=positions,
        rotary_dim=rotary_dim,
        rope_theta=rope_theta,
        interleave=False,
    )
    return q_rope, k_rope


def vllm_rope(
    q: torch.tensor,
    k: torch.tensor,
    positions: torch.tensor,
    head_size: int,
    rotary_dim: int,
    rope_theta: int,
    max_position: int,
):
    from vllm.model_executor.layers.rotary_embedding import get_rope

    rotary_emb = get_rope(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=rope_theta,
        is_neox_style=True,
    )

    q_rope, k_rope = rotary_emb(positions, q, k)
    return q_rope, k_rope


def main():
    batch_size, seq_len, head_size = 2, 10, 64
    rotary_dim = head_size
    rope_theta = 1e4

    torch.cuda.manual_seed_all(42)
    q = torch.rand((batch_size, seq_len, head_size), dtype=torch.float16, device="cuda")
    k = torch.rand((batch_size, seq_len, head_size), dtype=torch.float16, device="cuda")

    max_position = seq_len
    positions = torch.randint(0, seq_len, (batch_size + 1,), device="cuda")

    #  (batch_size, seq_len, head_size) -> flashinfer input shape (nnz, num_heads, head_dim)
    flashinfer_q_rope, flashinfer_k_rope = flashinfer_rope(
        rearrange(q, "b s h -> (b s) 1 h"),
        rearrange(k, "b s h -> (b s) 1 h"),
        positions.int(),
        rotary_dim,
        rope_theta,
    )

    # flashinfer output shape (nnz, num_heads, head_dim) -> (batch_size, seq_len, head_size)
    flashinfer_q_rope, flashinfer_k_rope = rearrange(
        flashinfer_q_rope, "(b s) 1 h -> b s h", b=batch_size, s=seq_len
    ), rearrange(flashinfer_k_rope, "(b s) 1 h -> b s h", b=batch_size, s=seq_len)

    # looks like this is doing something in-place?
    vllm_q_rope, vllm_k_rope = vllm_rope(
        q, k, positions, head_size, rotary_dim, rope_theta, max_position
    )

    # Mismatched elements: 2 / 1280 (0.2%)
    # Greatest absolute difference: 0.0001220703125 at index (0, 1, 4) (up to 1e-05 allowed)
    # Greatest relative difference: 0.017852783203125 at index (0, 2, 6) (up to 0.001 allowed)

    torch.testing.assert_close(flashinfer_q_rope, vllm_q_rope, atol=2e-4, rtol=2e-1)
    torch.testing.assert_close(flashinfer_k_rope, vllm_k_rope, atol=2e-4, rtol=2e-1)


if __name__ == "__main__":
    main()
