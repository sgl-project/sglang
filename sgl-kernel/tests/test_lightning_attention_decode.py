import torch
from sgl_kernel import lightning_attention_decode


def naive_lightning_attention_decode(q, k, v, past_kv, slope):
    """Naive implementation of lightning attention decode"""
    original_dtype = q.dtype
    ratio = torch.exp(-slope)  # [h, 1, 1]

    kv = past_kv
    b, h, n, d = q.shape

    output = []
    for i in range(n):
        kv = ratio * kv.to(torch.float32) + torch.einsum(
            "... n d, ... n e -> ... d e",
            k[:, :, i : i + 1],
            v[:, :, i : i + 1],
        )
        qkv = torch.einsum(
            "... n e, ... e d -> ... n d",
            q[:, :, i : i + 1].to(torch.float32),
            kv.to(torch.float32),
        )
        output.append(qkv)
    output = torch.concat(output, dim=-2)

    return output.to(original_dtype), kv


def test_lightning_attention_decode():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda")
    configs = [
        # (batch_size, num_heads, dim, embed_dim)
        (1, 8, 64, 64),
        (2, 8, 64, 64),
        (1, 32, 32, 64),
        (2, 32, 32, 64),
        (4, 32, 64, 64),
        (4, 32, 64, 64),
        (16, 64, 96, 96),
        (64, 64, 96, 96),
    ]

    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    for dtype in dtypes:
        for batch_size, num_heads, dim, embed_dim in configs:
            print(
                f"Testing dtype={dtype}, batch_size={batch_size}, "
                f"num_heads={num_heads}, dim={dim}, embed_dim={embed_dim}"
            )

            q = torch.randn(batch_size, num_heads, 1, dim, device=device, dtype=dtype)
            k = torch.randn(batch_size, num_heads, 1, dim, device=device, dtype=dtype)
            v = torch.randn(
                batch_size, num_heads, 1, embed_dim, device=device, dtype=dtype
            )
            past_kv = torch.randn(batch_size, num_heads, dim, embed_dim, device=device)
            slope = torch.randn(num_heads, 1, 1, device=device)

            ref_output, ref_new_kv = naive_lightning_attention_decode(
                q, k, v, past_kv, slope
            )

            output = torch.empty_like(ref_output)
            new_kv = torch.empty_like(ref_new_kv)
            lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv)

            rtol = 1e-2
            atol = 1e-2

            print("output.sum(): ", output.sum())
            print("ref_output.sum(): ", ref_output.sum())
            torch.testing.assert_close(
                output,
                ref_output,
                rtol=rtol,
                atol=atol,
                msg=f"Output mismatch for batch_size={batch_size}, num_heads={num_heads}, "
                f"dim={dim}, embed_dim={embed_dim}, dtype={dtype}",
            )

            print("new_kv.sum(): ", new_kv.sum())
            print("ref_new_kv.sum(): ", ref_new_kv.sum())
            torch.testing.assert_close(
                new_kv,
                ref_new_kv,
                rtol=rtol,
                atol=atol,
                msg=f"New KV mismatch for batch_size={batch_size}, num_heads={num_heads}, "
                f"dim={dim}, embed_dim={embed_dim}, dtype={dtype}",
            )

            print("Pass!")
            print("-" * 50)


if __name__ == "__main__":
    test_lightning_attention_decode()
    print("All tests passed!")
