#!/bin/bash
# 03_compare_single_vs_multi_gpu: 对比单GPU vs 多GPU的SGLang结果
# 用法: ./03_compare_single_vs_multi_gpu.sh

set -e

SINGLE_GPU_DIR="/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark/results"
MULTI_GPU_DIR="/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark/results_multi_gpu"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT="$MULTI_GPU_DIR/comparison_single_vs_multi_gpu_${TIMESTAMP}.md"

python3 - "$SINGLE_GPU_DIR" "$MULTI_GPU_DIR" "$OUTPUT" "$TIMESTAMP" << 'EOFPYTHON'
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

SINGLE_GPU_DIR = sys.argv[1]
MULTI_GPU_DIR = sys.argv[2]
OUTPUT = sys.argv[3]
TIMESTAMP = sys.argv[4]

def load_metrics(path):
    """加载 metrics，支持带 meta 的格式和原始格式"""
    with open(path, 'r') as f:
        data = json.load(f)
    if isinstance(data, dict) and "metrics" in data:
        return data["meta"], data["metrics"]
    return {}, data

def extract_config(filename):
    """从文件名提取配置信息"""
    # GLM-Image_sglang_w512_h512_n10_c1_random_20240205_162448.json
    # GLM-Image_multi_gpu_sglang_w512_h512_n10_c1_random_20240205_162448.json
    match = re.search(r'w(\d+)_h(\d+)_n(\d+)_c(\d+)_', filename)
    if match:
        return {
            'width': int(match.group(1)),
            'height': int(match.group(2)),
            'num_prompts': int(match.group(3)),
            'concurrency': int(match.group(4))
        }
    return None

def find_matching_pairs(single_dir, multi_dir):
    """找到所有匹配的单GPU和多GPU文件对"""
    single_dir = Path(single_dir)
    multi_dir = Path(multi_dir)
    
    single_files = list(single_dir.glob("*sglang*.json"))
    multi_files = list(multi_dir.glob("*sglang*.json"))
    
    # 按配置分组
    single_by_config = {}
    multi_by_config = {}
    
    for f in single_files:
        config = extract_config(f.name)
        if config:
            key = f"w{config['width']}_h{config['height']}_c{config['concurrency']}"
            if key not in single_by_config:
                single_by_config[key] = []
            single_by_config[key].append(f)
    
    for f in multi_files:
        config = extract_config(f.name)
        if config:
            key = f"w{config['width']}_h{config['height']}_c{config['concurrency']}"
            if key not in multi_by_config:
                multi_by_config[key] = []
            multi_by_config[key].append(f)
    
    # 找到两个目录都有的配置
    matching_pairs = []
    for key in set(single_by_config.keys()) & set(multi_by_config.keys()):
        # 取最新的文件
        single_file = max(single_by_config[key], key=lambda f: f.stat().st_mtime)
        multi_file = max(multi_by_config[key], key=lambda f: f.stat().st_mtime)
        matching_pairs.append((key, single_file, multi_file))
    
    return sorted(matching_pairs)

# 找到所有匹配的文件对
pairs = find_matching_pairs(SINGLE_GPU_DIR, MULTI_GPU_DIR)

if not pairs:
    print(f"错误: 未找到匹配的结果文件")
    print(f"单GPU目录: {SINGLE_GPU_DIR}")
    print(f"多GPU目录: {MULTI_GPU_DIR}")
    exit(1)

# 生成对比报告
with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write("# SGLang-D 单GPU vs 多GPU 性能对比报告\n\n")
    f.write(f"生成时间: {TIMESTAMP}\n\n")
    f.write("**对比目标**: 评估多GPU并行对SGLang-D性能的提升\n\n")
    
    f.write(f"## 共找到 {len(pairs)} 个配置的对比\n\n")
    
    # 汇总表
    f.write("### 汇总对比表\n\n")
    f.write("| 配置 | 延迟 Mean (s) | 延迟 P99 (s) | 吞吐量 (req/s) | 峰值内存 (GB) |\n")
    f.write("|:-----|:--------------|:-------------|:---------------|:-------------|\n")
    f.write("| | **单GPU** | **多GPU** | **单GPU** | **多GPU** | **单GPU** | **多GPU** | **单GPU** | **多GPU** |\n")
    f.write("|:-----|:-------------|:------------|:-------------|:------------|:----------|:---------|:----------|:---------|\n")
    
    for key, single_file, multi_file in pairs:
        single_meta, single_metrics = load_metrics(str(single_file))
        multi_meta, multi_metrics = load_metrics(str(multi_file))
        
        lat_single = single_metrics.get("latency_mean", 0)
        lat_multi = multi_metrics.get("latency_mean", 0)
        lat_p99_single = single_metrics.get("latency_p99", 0)
        lat_p99_multi = multi_metrics.get("latency_p99", 0)
        thr_single = single_metrics.get("throughput_qps", 0)
        thr_multi = multi_metrics.get("throughput_qps", 0)
        mem_single = single_metrics.get("peak_memory_mb_max", 0) / 1024
        mem_multi = multi_metrics.get("peak_memory_mb_max", 0) / 1024
        
        # 计算提升比例
        speedup = lat_single / lat_multi if lat_multi > 0 else 0
        speedup_p99 = lat_p99_single / lat_p99_multi if lat_p99_multi > 0 else 0
        thr_improve = thr_multi / thr_single if thr_single > 0 else 0
        
        f.write(f"| {key} | {lat_single:.2f} | {lat_multi:.2f} | {lat_p99_single:.2f} | {lat_p99_multi:.2f} | {thr_single:.4f} | {thr_multi:.4f} | {mem_single:.2f} | {mem_multi:.2f} |\n")
    
    # 详细对比
    f.write("\n## 详细对比\n\n")
    for key, single_file, multi_file in pairs:
        single_meta, single_metrics = load_metrics(str(single_file))
        multi_meta, multi_metrics = load_metrics(str(multi_file))
        
        f.write(f"### 配置: {key}\n\n")
        f.write(f"**单GPU文件**: `{single_file.name}`\n\n")
        f.write(f"**多GPU文件**: `{multi_file.name}`\n\n")
        
        f.write("| 指标 | 单GPU | 多GPU | 提升/变化 |\n")
        f.write("|:-----|:------|:------|:----------|\n")
        
        lat_single = single_metrics.get("latency_mean", 0)
        lat_multi = multi_metrics.get("latency_mean", 0)
        if lat_multi > 0:
            speedup = lat_single / lat_multi
            f.write(f"| 延迟 (平均) | {lat_single:.2f}s | {lat_multi:.2f}s | {speedup:.2f}x faster |\n")
        else:
            f.write(f"| 延迟 (平均) | {lat_single:.2f}s | {lat_multi:.2f}s | - |\n")
        
        lat_p99_single = single_metrics.get("latency_p99", 0)
        lat_p99_multi = multi_metrics.get("latency_p99", 0)
        if lat_p99_multi > 0:
            speedup_p99 = lat_p99_single / lat_p99_multi
            f.write(f"| 延迟 (P99) | {lat_p99_single:.2f}s | {lat_p99_multi:.2f}s | {speedup_p99:.2f}x faster |\n")
        else:
            f.write(f"| 延迟 (P99) | {lat_p99_single:.2f}s | {lat_p99_multi:.2f}s | - |\n")
        
        thr_single = single_metrics.get("throughput_qps", 0)
        thr_multi = multi_metrics.get("throughput_qps", 0)
        if thr_single > 0:
            thr_improve = thr_multi / thr_single
            f.write(f"| 吞吐量 | {thr_single:.4f} req/s | {thr_multi:.4f} req/s | {thr_improve:.2f}x higher |\n")
        else:
            f.write(f"| 吞吐量 | {thr_single:.4f} req/s | {thr_multi:.4f} req/s | - |\n")
        
        mem_single = single_metrics.get("peak_memory_mb_max", 0) / 1024
        mem_multi = multi_metrics.get("peak_memory_mb_max", 0) / 1024
        mem_diff = mem_multi - mem_single
        f.write(f"| 峰值内存 (最大) | {mem_single:.2f} GB | {mem_multi:.2f} GB | {mem_diff:+.2f} GB |\n")
        
        f.write("\n")

print(f"✅ 对比报告已生成: {OUTPUT}")
print(f"   共对比 {len(pairs)} 个配置")
EOFPYTHON
