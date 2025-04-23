import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tabulate import tabulate
import glob

def merge_results(output_file: str, num_gpus: int):
    all_results = []
    
    for rank in range(num_gpus):
        tmp_file = f"{output_file}.{rank}"
        if not Path(tmp_file).exists():
            print(f"Warning: Results for rank {rank} not found")
            continue
            
        with open(tmp_file) as f:
            results = json.load(f)
            all_results.extend(results)
            
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
        
    for tmp_file in glob.glob(f"{output_file}.[0-9]*"):
        Path(tmp_file).unlink()
        
    return all_results

def analyze_results(results):
    """分析基准测试结果"""
    # 将结果转换为DataFrame
    df_rows = []
    for r in results:
        row = {
            "Model": r["sub_label"].split(",")[0].strip(),
            "Num Experts": int(r["sub_label"].split("num_experts=")[1].split(",")[0]),
            "Top-k": int(r["sub_label"].split("topk=")[1].split(",")[0]),
            "MKN": eval(r["sub_label"].split("MKN=")[1].strip("()")),
            "Implementation": r["description"],
            "Mean (ms)": r["mean"] * 1000,  # 转换为毫秒
            "Median (ms)": r["median"] * 1000,
            "Std (ms)": r["std"] * 1000
        }
        df_rows.append(row)
    
    df = pd.DataFrame(df_rows)
    
    # 按实现方式分组计算加速比
    implementations = df["Implementation"].unique()
    baseline_impl = "triton_moe"  # 使用triton_moe作为基准
    
    print("\n=== Performance Summary ===")
    
    # 按模型分组显示结果
    for model in df["Model"].unique():
        print(f"\nModel: {model}")
        model_df = df[df["Model"] == model]
        
        # 计算每种实现方式相对于基准的加速比
        summary_rows = []
        for impl in implementations:
            impl_data = model_df[model_df["Implementation"] == impl]
            baseline_data = model_df[model_df["Implementation"] == baseline_impl]
            
            if len(impl_data) == 0 or len(baseline_data) == 0:
                continue
                
            mean_speedup = baseline_data["Mean (ms)"].mean() / impl_data["Mean (ms)"].mean()
            
            summary_rows.append({
                "Implementation": impl,
                "Avg Time (ms)": f"{impl_data['Mean (ms)'].mean():.2f}",
                "Speedup vs Triton": f"{mean_speedup:.2f}x"
            })
        
        # 打印表格
        print(tabulate(summary_rows, headers="keys", tablefmt="grid"))

def main():
    parser = argparse.ArgumentParser("Analyze benchmark results")
    parser.add_argument("--input-file", type=str, default="benchmark_results.json",
                      help="Input JSON file containing benchmark results")
    parser.add_argument("--merge", action="store_true",
                      help="Merge temporary result files")
    
    args = parser.parse_args()
    
    if args.merge:
        results = merge_results(args.input_file, 8)
    else:
        with open(args.input_file) as f:
            results = json.load(f)
    
    if not results:
        print("No results found!")
        return
        
    analyze_results(results)

if __name__ == "__main__":
    main() 