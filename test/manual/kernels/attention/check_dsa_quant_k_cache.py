import torch
import triton

from sglang.kernels.ops.attention.dsa.quant_k_cache import (
    _quantize_k_cache_fast_wrapped,
    _quantize_k_cache_ref,
    quantize_k_cache,
    quantize_k_cache_separate,
)

if __name__ == "__main__":
    import dequant_k_cache

    for num_blocks, block_size in [
        (1, 1),
        (10, 64),
    ]:
        dim_nope_and_rope = 512 + 64

        input_k_cache = torch.randn(
            (num_blocks, block_size, 1, dim_nope_and_rope),
            dtype=torch.bfloat16,
            device="cuda",
        )

        ref_quant = _quantize_k_cache_ref(input_k_cache)
        actual_quant = _quantize_k_cache_fast_wrapped(input_k_cache)

        ref_ref_dequant = dequant_k_cache._dequantize_k_cache_slow(ref_quant)
        ref_actual_dequant = dequant_k_cache._dequantize_k_cache_fast_wrapped(ref_quant)
        actual_actual_dequant = dequant_k_cache._dequantize_k_cache_fast_wrapped(
            actual_quant
        )

        print(f"{ref_ref_dequant=}")
        print(f"{actual_actual_dequant=}")
        print(f"{actual_actual_dequant - ref_ref_dequant=}")
        print(f"{torch.mean(ref_ref_dequant - actual_actual_dequant)=}")

        # TODO too different?
        torch.testing.assert_close(
            ref_ref_dequant, ref_actual_dequant, atol=0.2, rtol=0.2
        )
        torch.testing.assert_close(
            ref_ref_dequant, actual_actual_dequant, atol=0.2, rtol=0.2
        )

        # test dequant_k_cache_paged
        page_table_1 = torch.arange(
            num_blocks * block_size, dtype=torch.int32, device="cuda"
        )
        actual_dequant_paged = dequant_k_cache.dequantize_k_cache_paged(
            actual_quant, page_table_1
        ).reshape(actual_actual_dequant.shape)
        print(f"{torch.mean(actual_actual_dequant - actual_dequant_paged)=}")
        torch.testing.assert_close(
            ref_ref_dequant, actual_dequant_paged, atol=0.2, rtol=0.2
        )

    print("Passed")

    # Test quantize_k_cache_separate: verify output matches concat path
    print("\nTesting quantize_k_cache_separate...")
    for num_tokens in [64, 100]:
        dim_nope = 512
        dim_rope = 64

        k_nope = torch.randn(
            num_tokens, 1, dim_nope, dtype=torch.bfloat16, device="cuda"
        )
        k_rope = torch.randn(
            num_tokens, 1, dim_rope, dtype=torch.bfloat16, device="cuda"
        )

        # Old path: concat then quantize
        k_concat = torch.cat([k_nope, k_rope], dim=-1).squeeze(1)  # (num_tokens, 576)
        old_output = quantize_k_cache(k_concat.unsqueeze(1).unsqueeze(1))  # 4D input
        old_output = old_output.squeeze(1).squeeze(1)  # Back to (num_tokens, 656)

        # New path: quantize separately
        nope_part, rope_part = quantize_k_cache_separate(k_nope, k_rope)
        new_bytes = torch.cat([nope_part.squeeze(1), rope_part.squeeze(1)], dim=-1)

        # Compare byte-level equality
        old_bytes = old_output.view(torch.uint8)

        if old_bytes.shape != new_bytes.shape:
            raise RuntimeError(
                f"Shape mismatch: {old_bytes.shape} vs {new_bytes.shape}"
            )

        diff_bytes = (old_bytes != new_bytes).sum().item()
        if diff_bytes > 0:
            max_diff = (old_bytes.float() - new_bytes.float()).abs().max().item()
            raise RuntimeError(
                f"quantize_k_cache_separate output doesn't match concat path: "
                f"{diff_bytes} differing bytes, max_diff={max_diff}"
            )

        print(f"  num_tokens={num_tokens}: PASSED (outputs match byte-wise)")

    print("quantize_k_cache_separate tests passed!")

    print("\nDo benchmark...")

    for num_blocks, block_size in [
        (1, 64),
        (64, 64),
        (128, 64),
        (256, 64),
        (512, 64),
        (1024, 64),
        (2048, 64),
    ]:
        dim_nope_and_rope = 512 + 64

        input_k_cache = torch.randn(
            (num_blocks, block_size, 1, dim_nope_and_rope),
            dtype=torch.bfloat16,
            device="cuda",
        )

        actual_quant = _quantize_k_cache_fast_wrapped(input_k_cache)

        page_table_1 = torch.arange(
            num_blocks * block_size, dtype=torch.int32, device="cuda"
        )

        def run_ans():
            return dequant_k_cache.dequantize_k_cache_paged(actual_quant, page_table_1)

        ans_time: float = triton.testing.do_bench(run_ans, warmup=10, rep=20) / 1000  # type: ignore
        print(f"seq_kv: {num_blocks * block_size}, time: {ans_time * 1e6: 4.0f} us")
