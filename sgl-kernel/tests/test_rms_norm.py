# The implementation of this RMS Norm is primarily for testing the proper use of CUB in the single-group kernel.
import torch
from sgl_kernel import rms_norm


def rms_norm_reference(hidden_states, weight, epsilon):
    # Save input dtype
    input_dtype = hidden_states.dtype
    # Convert to float32 for better precision
    hidden_states = hidden_states.to(torch.float32)
    # Compute variance
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    # Normalize
    hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
    # Apply weight and restore dtype
    return (weight * hidden_states).to(input_dtype)


def test_rms_norm():
    tokens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    hidden_sizes = [768, 1024, 2048, 4096]
    dtypes = [torch.float32, torch.half, torch.bfloat16]
    device = torch.device("cuda")

    for dtype in dtypes:
        if dtype == torch.bfloat16 or dtype == torch.half:
            rtol = 1e-2
            atol = 1e-3
        else:
            rtol = 1e-5
            atol = 1e-7

        for num_tokens in tokens:
            for hidden_size in hidden_sizes:
                input = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
                weight = torch.randn(hidden_size, device=device, dtype=dtype)
                epsilon = 1e-6

                # Reference implementation
                ref_output = rms_norm_reference(input, weight, epsilon)

                # Kernel implementation
                kernel_output = torch.empty_like(input)
                rms_norm(kernel_output, input, weight, epsilon)

                torch.testing.assert_close(
                    kernel_output,
                    ref_output,
                    rtol=rtol,
                    atol=atol,
                    msg=f"Failed for num_tokens={num_tokens}, hidden_size={hidden_size}, dtype={dtype}",
                )


if __name__ == "__main__":
    test_rms_norm()
    print("RMSNorm all tests passed!")
