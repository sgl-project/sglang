#!/usr/bin/env python3
"""
内存分析脚本 - 从日志文件中读取内存数据并生成变化曲线
使用方法: python memory_analyzer.py <log_file_path>
"""

import argparse
import sys
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_memory_log(log_file):
    """解析内存日志文件"""
    try:
        # 读取CSV文件
        df = pd.read_csv(log_file)

        # 转换时间戳为datetime对象
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 转换内存值为MB单位
        df["memory_allocated_mb"] = df["memory_allocated"] / (1024 * 1024)
        df["memory_reserved_mb"] = df["memory_reserved"] / (1024 * 1024)

        return df
    except Exception as e:
        print(f"错误: 无法解析日志文件 {log_file}: {e}")
        return None


def create_memory_plots(df, output_prefix=None):
    """创建内存变化曲线图"""
    if df is None or df.empty:
        print("错误: 没有有效的数据用于绘图")
        return

    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 创建三个子图
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle("GPU Memory Usage Over Time", fontsize=16, fontweight="bold")

    # 第一个图: Memory Allocated
    ax1.plot(
        df["timestamp"],
        df["memory_allocated_mb"],
        color="blue",
        linewidth=2,
        label="Allocated Memory",
    )
    ax1.set_ylabel("Memory Allocated (MB)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_title("GPU Memory Allocated Over Time")

    # 第二个图: Memory Reserved
    ax2.plot(
        df["timestamp"],
        df["memory_reserved_mb"],
        color="red",
        linewidth=2,
        label="Reserved Memory",
    )
    ax2.set_ylabel("Memory Reserved (MB)", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_title("GPU Memory Reserved Over Time")

    # 第三个图: Memory Allocated vs Reserved 对比
    ax3.plot(
        df["timestamp"],
        df["memory_allocated_mb"],
        color="blue",
        linewidth=2,
        label="Allocated",
        alpha=0.8,
    )
    ax3.plot(
        df["timestamp"],
        df["memory_reserved_mb"],
        color="red",
        linewidth=2,
        label="Reserved",
        alpha=0.8,
    )
    ax3.fill_between(
        df["timestamp"], df["memory_allocated_mb"], alpha=0.3, color="blue"
    )
    ax3.fill_between(df["timestamp"], df["memory_reserved_mb"], alpha=0.3, color="red")
    ax3.set_ylabel("Memory Usage (MB)", fontsize=12)
    ax3.set_xlabel("Time", fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_title("GPU Memory Allocated vs Reserved Comparison")

    # 格式化时间轴
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # 保存图像
    if output_prefix:
        output_file = f"{output_prefix}_memory_analysis.png"
    else:
        output_file = "memory_analysis.png"

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"图像已保存到: {output_file}")

    # 显示图像
    plt.show()


def print_memory_stats(df):
    """打印内存统计信息"""
    if df is None or df.empty:
        return

    print("\n=== 内存使用统计 ===")
    print(f"数据记录数: {len(df)}")
    print(
        f"监控时长: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds():.1f} 秒"
    )

    print(f"\n分配内存 (MB):")
    print(f"  最小值: {df['memory_allocated_mb'].min():.2f}")
    print(f"  最大值: {df['memory_allocated_mb'].max():.2f}")
    print(f"  平均值: {df['memory_allocated_mb'].mean():.2f}")
    print(f"  标准差: {df['memory_allocated_mb'].std():.2f}")

    print(f"\n保留内存 (MB):")
    print(f"  最小值: {df['memory_reserved_mb'].min():.2f}")
    print(f"  最大值: {df['memory_reserved_mb'].max():.2f}")
    print(f"  平均值: {df['memory_reserved_mb'].mean():.2f}")
    print(f"  标准差: {df['memory_reserved_mb'].std():.2f}")

    # 计算内存利用率
    utilization = (df["memory_allocated_mb"] / df["memory_reserved_mb"]) * 100
    print(f"\n内存利用率 (%):")
    print(f"  最小值: {utilization.min():.2f}")
    print(f"  最大值: {utilization.max():.2f}")
    print(f"  平均值: {utilization.mean():.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="分析GPU内存使用日志文件并生成变化曲线"
    )
    parser.add_argument("log_file", help="内存日志文件路径")
    parser.add_argument("--output", "-o", help="输出图像文件前缀")
    parser.add_argument("--stats", "-s", action="store_true", help="显示统计信息")

    args = parser.parse_args()

    if not args.log_file:
        print("错误: 请提供日志文件路径")
        sys.exit(1)

    # 解析日志文件
    print(f"正在解析日志文件: {args.log_file}")
    df = parse_memory_log(args.log_file)

    if df is None:
        sys.exit(1)

    print(f"成功读取 {len(df)} 条记录")

    # 打印统计信息
    if args.stats:
        print_memory_stats(df)

    # 创建图表
    create_memory_plots(df, args.output)


if __name__ == "__main__":
    main()
