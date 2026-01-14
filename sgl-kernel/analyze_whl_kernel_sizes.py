import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def extract_whl(whl_file, extract_dir):
    with zipfile.ZipFile(whl_file, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def find_binary_files(extract_dir):
    binary_files = []
    extract_path = Path(extract_dir)

    for so_file in extract_path.rglob("*.so"):
        binary_files.append(str(so_file))

    for cubin_file in extract_path.rglob("*.cubin"):
        binary_files.append(str(cubin_file))

    return sorted(binary_files)


def run_cubloaty(binary_file):
    result = subprocess.run(
        ["cubloaty", binary_file, "--format", "json"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        if (
            "No CUDA binary sections found" in result.stderr
            or "does not contain device code" in result.stderr
        ):
            return {}
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )

    return json.loads(result.stdout)


def analyze_whl(whl_file):
    temp_dir = tempfile.mkdtemp(prefix="sgl_kernel_analysis_")

    try:
        extract_whl(whl_file, temp_dir)

        binary_files = find_binary_files(temp_dir)
        if not binary_files:
            print(f"No .so or .cubin files found in {whl_file}")
            return []

        all_kernels = []

        for binary_file in binary_files:
            file_name = os.path.basename(binary_file)
            data = run_cubloaty(binary_file)

            if not data or "kernels" not in data:
                continue
            for kernel in data["kernels"]:
                all_kernels.append(
                    {
                        "file": file_name,
                        "name": kernel.get("name", "unknown"),
                        "size": kernel.get("size", 0),
                        "size_kb": kernel.get("size", 0) / 1024,
                        "size_mb": kernel.get("size", 0) / 1024 / 1024,
                    }
                )
        return all_kernels

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def extract_kernel_prefix(kernel_name):
    if "<" in kernel_name:
        return kernel_name.split("<")[0]
    return kernel_name


def generate_report(all_kernels, output_file):
    if not all_kernels:
        print("No kernels found")
        return

    sorted_kernels = sorted(all_kernels, key=lambda x: x["size"], reverse=True)
    total_size = sum(k["size"] for k in all_kernels)
    total_size_mb = total_size / 1024 / 1024

    from collections import defaultdict

    kernel_groups = defaultdict(lambda: {"size": 0, "count": 0})
    for kernel in all_kernels:
        prefix = extract_kernel_prefix(kernel["name"])
        kernel_groups[prefix]["size"] += kernel["size"]
        kernel_groups[prefix]["count"] += 1

    sorted_groups = sorted(
        kernel_groups.items(), key=lambda x: x[1]["size"], reverse=True
    )

    lines = []
    lines.append("=" * 140)
    lines.append("CUDA Kernel Size Analysis")
    lines.append("=" * 140)
    lines.append("")
    lines.append(f"Total kernels: {len(all_kernels)}")
    lines.append(f"Total size: {total_size_mb:.2f} MB ({total_size:,} bytes)")
    lines.append(f"Average kernel size: {total_size / len(all_kernels) / 1024:.2f} KB")
    lines.append("")

    lines.append("=" * 140)
    lines.append("Kernel Groups (by name prefix) - Top 20")
    lines.append("=" * 140)
    lines.append(
        f"{'Rank':<6} {'Kernel Prefix':<80} {'Count':<8} {'Total (MB)':<12} {'%':<8}"
    )
    lines.append("-" * 140)

    TOP_N = 20
    for i, (prefix, stats) in enumerate(sorted_groups[:TOP_N], 1):
        percentage = (stats["size"] / total_size * 100) if total_size > 0 else 0
        size_mb = stats["size"] / 1024 / 1024

        display_prefix = prefix
        if len(display_prefix) > 77:
            display_prefix = display_prefix[:74] + "..."

        lines.append(
            f"{i:<6} {display_prefix:<80} {stats['count']:<8} {size_mb:<12.2f} {percentage:<8.2f}"
        )

    if len(sorted_groups) > TOP_N:
        other_size = sum(stats["size"] for _, stats in sorted_groups[TOP_N:])
        other_count = sum(stats["count"] for _, stats in sorted_groups[TOP_N:])
        other_percentage = (other_size / total_size * 100) if total_size > 0 else 0
        other_size_mb = other_size / 1024 / 1024

        lines.append(
            f"{'Other':<6} {'(remaining ' + str(len(sorted_groups) - TOP_N) + ' kernel groups)':<80} "
            f"{other_count:<8} {other_size_mb:<12.2f} {other_percentage:<8.2f}"
        )

    lines.append("")
    lines.append("=" * 140)
    lines.append("Individual Kernels (sorted by size) - Top 20")
    lines.append("=" * 140)
    lines.append(
        f"{'Rank':<6} {'File':<40} {'Kernel Name':<70} {'Size (KB)':<12} {'Size (MB)':<12} {'%':<8}"
    )
    lines.append("-" * 140)

    for i, kernel in enumerate(sorted_kernels[:TOP_N], 1):
        percentage = (kernel["size"] / total_size * 100) if total_size > 0 else 0
        kernel_name = kernel["name"]
        if len(kernel_name) > 67:
            kernel_name = kernel_name[:64] + "..."

        file_name = kernel["file"]
        if len(file_name) > 37:
            file_name = file_name[:34] + "..."

        lines.append(
            f"{i:<6} {file_name:<40} {kernel_name:<70} "
            f"{kernel['size_kb']:<12.2f} {kernel['size_mb']:<12.4f} {percentage:<8.2f}"
        )

    if len(sorted_kernels) > TOP_N:
        other_size = sum(k["size"] for k in sorted_kernels[TOP_N:])
        other_count = len(sorted_kernels) - TOP_N
        other_percentage = (other_size / total_size * 100) if total_size > 0 else 0
        other_size_kb = other_size / 1024
        other_size_mb = other_size / 1024 / 1024

        lines.append(
            f"{'Other':<6} {'(remaining ' + str(other_count) + ' kernels)':<40} "
            f"{'':<70} {other_size_kb:<12.2f} {other_size_mb:<12.4f} {other_percentage:<8.2f}"
        )

    report_text = "\n".join(lines)

    with open(output_file, "w") as f:
        f.write(report_text)
    print(f"Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CUDA kernel sizes in sgl-kernel whl file"
    )
    parser.add_argument("whl", type=str, help="Path to whl file")
    parser.add_argument(
        "--output", type=str, default="kernel_analysis.txt", help="Output report file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.whl):
        print(f"Error: {args.whl} not found")
        sys.exit(1)

    all_kernels = analyze_whl(args.whl)

    if all_kernels:
        generate_report(all_kernels, args.output)
    else:
        print("No kernel information extracted")


if __name__ == "__main__":
    main()
