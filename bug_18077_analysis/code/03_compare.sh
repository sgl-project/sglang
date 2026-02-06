#!/bin/bash
# 03: 对比 SGLang 和 Diffusers 结果
# 用法: 
#   ./03_compare.sh                    # 对比所有配置
#   ./03_compare.sh w512_h512_c1       # 对比特定配置 (width=512, height=512, concurrency=1)
#   ./03_compare.sh --pattern "w512"   # 使用 pattern 匹配

set -e

RESULT_DIR="/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/benchmark/results"

# 解析参数
PATTERN=""
if [ "$1" = "--pattern" ] && [ -n "$2" ]; then
    PATTERN="$2"
elif [ -n "$1" ]; then
    PATTERN="$1"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT="$RESULT_DIR/comparison_${TIMESTAMP}.md"

python3 - "$RESULT_DIR" "$PATTERN" "$OUTPUT" "$TIMESTAMP" << 'EOFPYTHON'
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

RESULT_DIR = sys.argv[1]
PATTERN = sys.argv[2]
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
    match = re.search(r'w(\d+)_h(\d+)_n(\d+)_c(\d+)_', filename)
    if match:
        return {
            'width': int(match.group(1)),
            'height': int(match.group(2)),
            'num_prompts': int(match.group(3)),
            'concurrency': int(match.group(4))
        }
    return None

def find_matching_pairs(result_dir, pattern=""):
    """找到所有匹配的 sglang 和 diffusers 文件对"""
    result_dir = Path(result_dir)
    sglang_files = list(result_dir.glob("*sglang*.json"))
    diffusers_files = list(result_dir.glob("*diffusers*.json"))
    
    # 按配置分组
    sglang_by_config = {}
    diffusers_by_config = {}
    
    for f in sglang_files:
        config = extract_config(f.name)
        if config:
            key = f"w{config['width']}_h{config['height']}_c{config['concurrency']}"
            if pattern and pattern not in key:
                continue
            if key not in sglang_by_config:
                sglang_by_config[key] = []
            sglang_by_config[key].append(f)
    
    for f in diffusers_files:
        config = extract_config(f.name)
        if config:
            key = f"w{config['width']}_h{config['height']}_c{config['concurrency']}"
            if pattern and pattern not in key:
                continue
            if key not in diffusers_by_config:
                diffusers_by_config[key] = []
            diffusers_by_config[key].append(f)
    
    # 找到两个后端都有的配置
    matching_pairs = []
    for key in set(sglang_by_config.keys()) & set(diffusers_by_config.keys()):
        # 取最新的文件
        sglang_file = max(sglang_by_config[key], key=lambda f: f.stat().st_mtime)
        diffusers_file = max(diffusers_by_config[key], key=lambda f: f.stat().st_mtime)
        matching_pairs.append((key, sglang_file, diffusers_file))
    
    return sorted(matching_pairs)

# 找到所有匹配的文件对
pairs = find_matching_pairs(RESULT_DIR, PATTERN)

if not pairs:
    print(f"错误: 未找到匹配的结果文件")
    if PATTERN:
        print(f"Pattern: {PATTERN}")
    exit(1)

# 生成对比报告
with open(OUTPUT, 'w', encoding='utf-8') as f:
    f.write("# GLM-Image 性能对比报告\n\n")
    f.write(f"生成时间: {TIMESTAMP}\n\n")
    
    if len(pairs) == 1:
        # 单个配置对比
        key, sglang_file, diffusers_file = pairs[0]
        sglang_meta, sglang_metrics = load_metrics(str(sglang_file))
        diffusers_meta, diffusers_metrics = load_metrics(str(diffusers_file))
        
        f.write(f"## 配置: {key}\n\n")
        f.write(f"**SGLang 文件**: `{sglang_file.name}`\n\n")
        f.write(f"**Diffusers 文件**: `{diffusers_file.name}`\n\n")
        
        f.write("| 指标 | SGLang | Diffusers | 差距 |\n")
        f.write("|:-----|:-------|:----------|:-----|\n")
        
        lat_s = sglang_metrics.get("latency_mean", 0)
        lat_d = diffusers_metrics.get("latency_mean", 0)
        if lat_s > 0:
            f.write(f"| 延迟 (平均) | {lat_s:.2f}s | {lat_d:.2f}s | {lat_d/lat_s:.2f}x |\n")
        else:
            f.write(f"| 延迟 (平均) | {lat_s:.2f}s | {lat_d:.2f}s | - |\n")
        
        lat_p99_s = sglang_metrics.get("latency_p99", 0)
        lat_p99_d = diffusers_metrics.get("latency_p99", 0)
        if lat_p99_s > 0:
            f.write(f"| 延迟 (P99) | {lat_p99_s:.2f}s | {lat_p99_d:.2f}s | {lat_p99_d/lat_p99_s:.2f}x |\n")
        else:
            f.write(f"| 延迟 (P99) | {lat_p99_s:.2f}s | {lat_p99_d:.2f}s | - |\n")
        
        thr_s = sglang_metrics.get("throughput_qps", 0)
        thr_d = diffusers_metrics.get("throughput_qps", 0)
        if thr_s > 0:
            f.write(f"| 吞吐量 | {thr_s:.4f} req/s | {thr_d:.4f} req/s | {thr_d/thr_s:.2f}x |\n")
        else:
            f.write(f"| 吞吐量 | {thr_s:.4f} req/s | {thr_d:.4f} req/s | - |\n")
        
        mem_s = sglang_metrics.get("peak_memory_mb_max", 0) / 1024
        mem_d = diffusers_metrics.get("peak_memory_mb_max", 0) / 1024
        f.write(f"| 峰值内存 (最大) | {mem_s:.2f} GB | {mem_d:.2f} GB | {mem_d-mem_s:+.2f} GB |\n")
        
        mem_mean_s = sglang_metrics.get("peak_memory_mb_mean", 0) / 1024
        mem_mean_d = diffusers_metrics.get("peak_memory_mb_mean", 0) / 1024
        f.write(f"| 峰值内存 (平均) | {mem_mean_s:.2f} GB | {mem_mean_d:.2f} GB | {mem_mean_d-mem_mean_s:+.2f} GB |\n")
        
        completed_s = sglang_metrics.get("completed_requests", 0)
        completed_d = diffusers_metrics.get("completed_requests", 0)
        failed_s = sglang_metrics.get("failed_requests", 0)
        failed_d = diffusers_metrics.get("failed_requests", 0)
        f.write(f"| 成功请求 | {completed_s} | {completed_d} | - |\n")
        f.write(f"| 失败请求 | {failed_s} | {failed_d} | - |\n")
        
    else:
        # 多个配置对比
        f.write(f"## 共找到 {len(pairs)} 个配置的对比\n\n")
        
        # 汇总表
        f.write("### 汇总对比表\n\n")
        f.write("| 配置 | 延迟 (平均) | 吞吐量 | 峰值内存 (GB) |\n")
        f.write("|:-----|:------------|:-------|:-------------|\n")
        f.write("| | SGLang | Diffusers | SGLang | Diffusers | SGLang | Diffusers |\n")
        f.write("|:-----|:-----------|:---------|:-----------|:-----------|:--------|:----------|\n")
        
        for key, sglang_file, diffusers_file in pairs:
            sglang_meta, sglang_metrics = load_metrics(str(sglang_file))
            diffusers_meta, diffusers_metrics = load_metrics(str(diffusers_file))
            
            lat_s = sglang_metrics.get("latency_mean", 0)
            lat_d = diffusers_metrics.get("latency_mean", 0)
            thr_s = sglang_metrics.get("throughput_qps", 0)
            thr_d = diffusers_metrics.get("throughput_qps", 0)
            mem_s = sglang_metrics.get("peak_memory_mb_max", 0) / 1024
            mem_d = diffusers_metrics.get("peak_memory_mb_max", 0) / 1024
            
            f.write(f"| {key} | {lat_s:.2f}s | {lat_d:.2f}s | {thr_s:.4f} | {thr_d:.4f} | {mem_s:.2f} | {mem_d:.2f} |\n")
        
        # 详细对比
        f.write("\n## 详细对比\n\n")
        for key, sglang_file, diffusers_file in pairs:
            sglang_meta, sglang_metrics = load_metrics(str(sglang_file))
            diffusers_meta, diffusers_metrics = load_metrics(str(diffusers_file))
            
            f.write(f"### 配置: {key}\n\n")
            f.write(f"**SGLang 文件**: `{sglang_file.name}`\n\n")
            f.write(f"**Diffusers 文件**: `{diffusers_file.name}`\n\n")
            
            f.write("| 指标 | SGLang | Diffusers | 差距 |\n")
            f.write("|:-----|:-------|:----------|:-----|\n")
            
            lat_s = sglang_metrics.get("latency_mean", 0)
            lat_d = diffusers_metrics.get("latency_mean", 0)
            if lat_s > 0:
                f.write(f"| 延迟 (平均) | {lat_s:.2f}s | {lat_d:.2f}s | {lat_d/lat_s:.2f}x |\n")
            else:
                f.write(f"| 延迟 (平均) | {lat_s:.2f}s | {lat_d:.2f}s | - |\n")
            
            lat_p99_s = sglang_metrics.get("latency_p99", 0)
            lat_p99_d = diffusers_metrics.get("latency_p99", 0)
            if lat_p99_s > 0:
                f.write(f"| 延迟 (P99) | {lat_p99_s:.2f}s | {lat_p99_d:.2f}s | {lat_p99_d/lat_p99_s:.2f}x |\n")
            else:
                f.write(f"| 延迟 (P99) | {lat_p99_s:.2f}s | {lat_p99_d:.2f}s | - |\n")
            
            thr_s = sglang_metrics.get("throughput_qps", 0)
            thr_d = diffusers_metrics.get("throughput_qps", 0)
            if thr_s > 0:
                f.write(f"| 吞吐量 | {thr_s:.4f} req/s | {thr_d:.4f} req/s | {thr_d/thr_s:.2f}x |\n")
            else:
                f.write(f"| 吞吐量 | {thr_s:.4f} req/s | {thr_d:.4f} req/s | - |\n")
            
            mem_s = sglang_metrics.get("peak_memory_mb_max", 0) / 1024
            mem_d = diffusers_metrics.get("peak_memory_mb_max", 0) / 1024
            f.write(f"| 峰值内存 (最大) | {mem_s:.2f} GB | {mem_d:.2f} GB | {mem_d-mem_s:+.2f} GB |\n")
            
            f.write("\n")

print(f"✅ 对比报告已生成: {OUTPUT}")
print(f"   共对比 {len(pairs)} 个配置")
EOFPYTHON
