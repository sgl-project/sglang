import argparse
import json
import os
import torch
import triton
from benchmark_sum_scale import _moe_sum_reduce_kernel_opt

TOKEN_NUMS      = [2**i for i in range(15)]          
WORKLOAD_SPECS  = [(8, 2048, torch.bfloat16)]        

BLOCK_M_VALS    = [1, 2, 4]
BLOCK_D_VALS    = [256, 512, 1024, 2048]                  
NUM_STAGES_VALS = [1, 2,3, 4]
NUM_WARPS_VALS  = [2, 4, 8, 16,32]
EVICT_POLICIES  = ["evict_first", "evict_last",None]     

def gpu_name():
    name = torch.cuda.get_device_name(0)        
    return name.replace(" ", "-").replace("_", "-")


@torch.no_grad()
def best_config_for_shape(topk: int, hidden: int, dtype: torch.dtype):
    key = f"{topk}-{hidden}-{str(dtype).split('.')[-1]}"
    print(f"Tuning {key} ...")
    result = {}                     

    for M in TOKEN_NUMS:
        best_us, best_cfg = float("inf"), None
        for bm in BLOCK_M_VALS:
            for bd in BLOCK_D_VALS:
                for ns in NUM_STAGES_VALS:
                    for nw in NUM_WARPS_VALS:
                        for evp in EVICT_POLICIES:
                            grid = (triton.cdiv(M, bm),
                                    triton.cdiv(hidden, bd))

                            x = torch.randn((M, topk, hidden), dtype=dtype, device='cuda')
                            y = torch.empty((M, hidden), dtype=dtype, device='cuda')
                            rsf = 0.3

                            def run():
                                if evp is not None:
                                    _moe_sum_reduce_kernel_opt[grid](
                                        x, x.stride(0), x.stride(1), x.stride(2),
                                        y, y.stride(0), y.stride(1),
                                        M, topk, hidden, rsf,
                                        BLOCK_M=bm, BLOCK_DIM=bd,
                                        NUM_STAGE=ns, num_warps=nw,
                                        EVICTION_POLICY=evp)
                                else:
                                    _moe_sum_reduce_kernel_opt[grid](
                                        x, x.stride(0), x.stride(1), x.stride(2),
                                        y, y.stride(0), y.stride(1),
                                        M, topk, hidden, rsf,
                                        BLOCK_M=bm, BLOCK_DIM=bd,
                                        NUM_STAGE=ns, num_warps=nw,
                                        )
                            t_ms = triton.testing.do_bench(run,
                                                         rep=400,
                                                         warmup=50,
                                                         )  
                            t_us = t_ms * 1000
                            if t_us < best_us:
                                best_us, best_cfg = t_us, {
                                    "BLOCK_M": bm, "BLOCK_DIM": bd,
                                    "NUM_STAGE": ns, "num_warps": nw,
                                    "evict_policy": evp}

        result[str(M)] = best_cfg
        print(f"  M={M:5d}  best={best_cfg}  t={best_us:7.2f} µs")
    return result
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="auto")     
    args = parser.parse_args()
    dev_tag = gpu_name() 
    for topk, hidden, dtype in WORKLOAD_SPECS:
        if args.out == "auto":
            out_file = f"moe_sum_reduce_{dev_tag}_t{topk}_h{hidden}_{str(dtype).split('.')[-1]}.json"
        else:
            out_file = args.out          
        db = {f"{topk}-{hidden}-{str(dtype).split('.')[-1]}":
              best_config_for_shape(topk, hidden, dtype)}
        os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(db, f, indent=2)
        print("Saved ->", out_file)

if __name__ == "__main__":
    main()
