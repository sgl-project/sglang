import torch
import torch.nn.functional as F
import triton

from vllm._custom_ops import cutlass_scaled_mm as vllm_scaled_mm
from vllm._custom_ops import scaled_fp8_quant as vllm_scaled_fp8_quant
from sgl_kernel import fp8_scaled_mm as sgl_scaled_mm
from sgl_kernel import fp8_scaled_mm_profile as sgl_scaled_mm_profile
import time
def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


@triton.testing.perf_report(
        triton.testing.Benchmark(
        x_names=["batch_size"],
        x_vals=[1, 16, 64, 128, 256, 512, 1024, 2048],
        x_log=False,
        line_arg="provider",
        # line_vals=["vllm-fp8", "torch-fp8", "sglang-fp8"],
        # line_names=["vllm-fp8", "torch-fp8", "sglang-fp8"],
        line_vals=["vllm-fp8", "sglang-fp8", "sglang-fp8-profile"],
        line_names=["vllm-fp8", "sglang-fp8", "sglang-fp8-profile"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="int8 scaled matmul",
        args={},
    )
)
def benchmark(batch_size, provider):
    M, N, K = batch_size, 4096, 8192
    a = torch.ones((M, K), device="cuda") * 5.0
    b = torch.ones((N, K), device="cuda") * 5.0
    scale_a = torch.randn((M,), device="cuda", dtype=torch.float32)
    scale_b = torch.randn((N,), device="cuda", dtype=torch.float32)
    a_fp8, scale_a_fp8 = vllm_scaled_fp8_quant(a, scale_a)
    b_fp8, scale_b_fp8 = vllm_scaled_fp8_quant(b, scale_b)
    b_fp8 = b_fp8.t()
    quantiles = [0.5, 0.2, 0.8]

    if provider == "vllm-fp8":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: vllm_scaled_mm(
                a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, torch.bfloat16
            ),
            quantiles=quantiles,
        )
    if provider == "torch-fp8":
        scale_a_2d = scale_a_fp8.float().unsqueeze(1)  # [M, 1]
        scale_b_2d = scale_b_fp8.float().unsqueeze(0)  # [1, N]
        try:
            out = torch.empty(
                (a_fp8.shape[0], b_fp8.shape[0]), device="cuda", dtype=torch.bfloat16
            )
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: torch._scaled_mm(
                    a_fp8,
                    b_fp8,
                    out=out,
                    out_dtype=torch.bfloat16,
                    scale_a=scale_a_2d,
                    scale_b=scale_b_2d,
                    use_fast_accum=True,
                ),
                quantiles=quantiles,
            )
        except RuntimeError as e:
            print("Error details:", e)
            raise
    if provider == "sglang-fp8":
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: sgl_scaled_mm(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, torch.bfloat16),
            quantiles=quantiles,
        )
    if provider == "sglang-fp8-profile":
        best_configs = []
        times = []
        valid_configs = []
        best_config_info = {}  # 新增：用于存储每个输入规模的最优配置信息
        
        try:
            sgl_scaled_mm_profile(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, torch.bfloat16, bias=None, config_id=35)
        except RuntimeError as e:
            print(f"Skip config_id 35 due to error: {e}")
            
        for config_id in range(1, 7):
            try:
                torch.cuda.synchronize()
                start = time.time()
                sgl_scaled_mm_profile(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, 
                                    torch.bfloat16, bias=None, config_id=config_id)
                torch.cuda.synchronize()
                end = time.time()
                times.append(end - start)
                valid_configs.append(config_id)
                print(f"config_id: {config_id}, time: {end - start}")
            except RuntimeError as e:
                print(f"Skip config_id {config_id} due to error: {e}")
                continue
                
        if not valid_configs:
            print("No valid config found")
            return 0, 0, 0
            
        min_time = float('inf')
        best_config = None
        for i, config_id in enumerate(valid_configs):
            if times[i] < min_time:
                min_time = times[i]
                best_config = config_id
                
        # 记录当前输入规模的最优配置
        best_config_info[f"M{M}_N{N}_K{K}"] = {
            "best_config": best_config,
            "time": min_time,
            "batch_size": batch_size
        }
        
        # 将最优配置信息保存到文件
        import json
        config_file = "best_fp8_configs.json"
        try:
            with open(config_file, "r") as f:
                existing_configs = json.load(f)
        except FileNotFoundError:
            existing_configs = {}
            
        existing_configs.update(best_config_info)
        with open(config_file, "w") as f:
            json.dump(existing_configs, f, indent=4)
            
        print(f"Best config for batch_size={batch_size}: config_id={best_config}, time={min_time:.6f}s")
        
        # 使用最佳配置进行基准测试
        try:
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: sgl_scaled_mm_profile(a_fp8, b_fp8, scale_a_fp8, scale_b_fp8, torch.bfloat16, bias=None, config_id=best_config),
                quantiles=quantiles,
            )
        except RuntimeError as e:
            print("Error details:", e)
            print(f"config_id is not valid {best_config}")
            ms, min_ms, max_ms = 1, 1, 1
    gbps = lambda ms: (2 * M * N * K + M * N) * a.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


benchmark.run(print_data=True, show_plots=True, save_path="bench_int8_res")