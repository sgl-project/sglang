import os
import random
import unittest
from typing import Any, Tuple

import deep_gemm
import torch
from deep_gemm import calc_diff, ceil_div, get_col_major_tma_aligned_tensor, jit

"""
fork deepgemm/tests/test_core.py
"""


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(
        m, n
    ), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0), x_view.size(2)
    )


def construct(m: int, k: int, n: int) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    ref_out = x @ y.t()

    x_fp8, y_fp8 = per_token_cast_to_fp8(x), per_block_cast_to_fp8(y)
    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


def construct_grouped(
    num_groups: int, m: int, k: int, n: int, is_masked: bool
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    x = torch.randn((num_groups, m, k), device="cuda", dtype=torch.bfloat16)
    y = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((num_groups, m, n), device="cuda", dtype=torch.bfloat16)
    ref_out = torch.einsum("gmk,gnk->gmn", x, y)

    assert m % 4 == 0, f"TMA alignment error: {m}"
    x_fp8 = (
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        torch.empty((num_groups, m, k // 128), device="cuda", dtype=torch.float),
    )
    y_fp8 = (
        torch.empty_like(y, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, (n + 127) // 128, k // 128), device="cuda", dtype=torch.float
        ),
    )
    for i in range(num_groups):
        x_fp8[0][i], x_fp8[1][i] = per_token_cast_to_fp8(x[i])
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i])

    # For non-masked input, we must merge the group and M dims
    if not is_masked:
        x_fp8 = (x_fp8[0].view(-1, k), per_token_cast_to_fp8(x.view(-1, k))[1])
        out, ref_out = out.view(-1, n), ref_out.view(-1, n)

    # Transpose earlier so that the testing will not trigger transposing kernels
    x_fp8 = (x_fp8[0], get_col_major_tma_aligned_tensor(x_fp8[1]))
    return x_fp8, y_fp8, out, ref_out


class TestDeepGemmCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(0)
        random.seed(0)

        print("Library path:")
        print(f" > {deep_gemm.__path__}\n")

    def test_gemm(self):
        print("Testing GEMM:")
        for m in (64, 128, 4096):
            for k, n in [
                (7168, 2112),
                (1536, 24576),
                (512, 32768),
                (16384, 7168),
                (7168, 4096),
                (2048, 7168),
            ]:
                x_fp8, y_fp8, out, ref_out = construct(m, k, n)
                deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
                diff = calc_diff(out, ref_out)
                self.assertTrue(diff < 0.001, f"{m=}, {k=}, {n=}, {diff:.5f}")

    def test_m_grouped_gemm_contiguous(self):
        print("Testing grouped contiguous GEMM:")

        for num_groups, m, k, n in (
            (4, 8192, 7168, 4096),
            (4, 8192, 2048, 7168),
            (8, 4096, 7168, 4096),
            (8, 4096, 2048, 7168),
        ):
            # TODO: make a stronger test
            x_fp8, y_fp8, out, ref_out = construct_grouped(
                num_groups, m, k, n, is_masked=False
            )
            m_indices = torch.arange(0, num_groups, device="cuda", dtype=torch.int)
            m_indices = (
                m_indices.unsqueeze(-1).expand(num_groups, m).contiguous().view(-1)
            )
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                x_fp8, y_fp8, out, m_indices
            )
            diff = calc_diff(out, ref_out)
            self.assertTrue(diff < 0.001, f"m={m * num_groups}, {k=}, {n=}, {diff:.5f}")

    def test_m_grouped_gemm_masked(self):
        print("Testing grouped masked GEMM:")

        for num_groups, m in ((1, 1024), (2, 512), (4, 256)):
            for k, n in (
                (7168, 4096),
                (2048, 7168),
            ):
                # Test correctness
                masked_m_candidates = list(
                    filter(
                        lambda candidate: candidate <= m, (64, 128, 192, 256, 320, 384)
                    )
                )
                for i in range(10):
                    x_fp8, y_fp8, out, ref_out = construct_grouped(
                        num_groups, m, k, n, is_masked=True
                    )
                    masked_m = torch.empty(
                        (num_groups,), device="cuda", dtype=torch.int
                    )
                    for j in range(num_groups):
                        masked_m[j] = random.choice(masked_m_candidates)
                    expected_m = min(int(masked_m.float().mean()) + 1, m)
                    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_masked(
                        x_fp8, y_fp8, out, masked_m, expected_m
                    )
                    for j in range(num_groups):
                        diff = calc_diff(
                            out[j, : masked_m[j].item()],
                            ref_out[j, : masked_m[j].item()],
                        )
                        self.assertTrue(
                            diff < 0.001,
                            f"{m=}, {k=}, {n=}, {j=}, masked_m={masked_m[j]}, {num_groups=}, {diff:.5f}",
                        )


"""
fork deepgemm/tests/test_jit.py
"""


class Capture:
    def __init__(self) -> None:
        self.read_fd = None
        self.write_fd = None
        self.saved_stdout = None
        self.captured = None

    def __enter__(self) -> Any:
        self.read_fd, self.write_fd = os.pipe()
        self.saved_stdout = os.dup(1)
        os.dup2(self.write_fd, 1)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        os.dup2(self.saved_stdout, 1)
        os.close(self.write_fd)
        with os.fdopen(self.read_fd, "r") as f:
            self.captured = f.read()

    def capture(self) -> str:
        return self.captured


class TestDeepGemmJIT(unittest.TestCase):
    def test_jit(self):
        # Runtime
        print(f"NVCC compiler: {jit.get_nvcc_compiler()}\n")

        # Templates
        print("Generated code:")
        args = (
            ("lhs", torch.float8_e4m3fn),
            ("rhs", torch.float8_e4m3fn),
            ("scale", torch.float),
            ("out", torch.bfloat16),
            ("enable_double_streams", bool),
            ("stream", torch.cuda.Stream),
        )
        body = "\n"
        body += "std::cout << reinterpret_cast<uint64_t>(lhs) << std::endl;\n"
        body += "std::cout << reinterpret_cast<uint64_t>(rhs) << std::endl;\n"
        body += "std::cout << reinterpret_cast<uint64_t>(scale) << std::endl;\n"
        body += "std::cout << reinterpret_cast<uint64_t>(out) << std::endl;\n"
        body += "std::cout << enable_double_streams << std::endl;\n"
        body += "std::cout << reinterpret_cast<uint64_t>(stream) << std::endl;\n"
        code = jit.generate((), args, body)
        print(code)

        # Build
        print("Building ...")
        func = jit.build("test_func", args, code)

        # Test correctness
        print("Running ...")
        fp8_tensor = torch.empty((1,), dtype=torch.float8_e4m3fn, device="cuda")
        fp32_tensor = torch.empty((1,), dtype=torch.float, device="cuda")
        bf16_tensor = torch.empty((1,), dtype=torch.bfloat16, device="cuda")
        with Capture() as capture:
            self.assertTrue(
                func(
                    fp8_tensor,
                    fp8_tensor,
                    fp32_tensor,
                    bf16_tensor,
                    True,
                    torch.cuda.current_stream(),
                )
                == 0
            )
        output = capture.capture()
        ref_output = f"{fp8_tensor.data_ptr()}\n{fp8_tensor.data_ptr()}\n{fp32_tensor.data_ptr()}\n{bf16_tensor.data_ptr()}\n1\n{torch.cuda.current_stream().cuda_stream}\n"
        self.assertTrue(output == ref_output, f"{output=}, {ref_output=}")


if __name__ == "__main__":
    unittest.main()
