import torch

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
    NOPE_ROPE_BYTES,
    PADDED_SCALE_PER_TOKEN,
    dequantize_k_cache_paged,
    dequantize_k_cache_paged_ref,
)

if __name__ == "__main__":
    assert torch.cuda.is_available(), "this self-test needs a CUDA device"
    torch.manual_seed(0)
    device = "cuda"

    page_size = 64
    num_pages = 8
    num_tokens = 333
    raw_bytes = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = (
        (raw_bytes + NOPE_ROPE_BYTES - 1) // NOPE_ROPE_BYTES
    ) * NOPE_ROPE_BYTES

    quant_k_cache = torch.randint(
        0, 256, (num_pages, bytes_per_page), dtype=torch.uint8, device=device
    )
    page_table = torch.randint(
        0, num_pages * page_size, (num_tokens,), dtype=torch.int32, device=device
    )

    out_kernel = dequantize_k_cache_paged(quant_k_cache, page_table, page_size)
    out_ref = dequantize_k_cache_paged_ref(quant_k_cache, page_table, page_size)

    torch.testing.assert_close(out_kernel, out_ref, atol=0, rtol=0, equal_nan=True)
    print(
        f"OK: kernel matches torch ref for {num_tokens} tokens "
        f"(page_size={page_size}, bytes_per_page={bytes_per_page})"
    )
