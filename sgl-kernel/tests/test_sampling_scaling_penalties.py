import torch
from sgl_kernel import sampling_scaling_penalties

def test_sampling_scaling_penalties():
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    vocab_sizes = [2048, 4096, 8192, 16384, 32768]
    dtypes = [torch.float32, torch.half, torch.bfloat16]
    device = torch.device("cuda")

    for dtype in dtypes:
        rtol = 1e-3
        atol = 1e-3
        
        for bs in batch_sizes:
            for vocab_size in vocab_sizes:
                logits = torch.randn(bs, vocab_size, device=device, dtype=dtype)
                scaling_penalties = torch.rand(bs, vocab_size, device=device, dtype=dtype) + 0.5

                ref_output = torch.where(
                    logits > 0,
                    logits / scaling_penalties,
                    logits * scaling_penalties
                )

                kernel_output = sampling_scaling_penalties(logits, scaling_penalties)

                torch.testing.assert_close(
                    kernel_output,
                    ref_output,
                    rtol=rtol,
                    atol=atol,
                    msg=f"Failed for batch_size={bs}, vocab_size={vocab_size}, dtype={dtype}"
                )

if __name__ == "__main__":
    test_sampling_scaling_penalties()
    print("All tests passed!")
