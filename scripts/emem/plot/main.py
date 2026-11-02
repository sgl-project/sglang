import os
import re
import statistics
from datetime import datetime

import matplotlib.pyplot as plt


def parse_log_file(file_path):
    """Parse log file and extract token usage, running requests, throughput, and timing data"""
    full_token_usage = []
    swa_token_usage = []
    running_req = []
    gen_throughput = []
    timestamps = []

    # Data for timing measurements
    timing_timestamps = []
    timing_values = []
    timing_labels = []

    with open(file_path, "r") as f:
        start_time = None
        for line in f:
            # Extract timestamp from the beginning of the line
            timestamp_match = re.match(r"\[([0-9-]+ [0-9:]+)\]", line)
            actual_timestamp = None
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                actual_timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                if start_time is None:
                    start_time = actual_timestamp
            else:
                continue

            # Look for prefill or decode batch lines
            if "Prefill batch" in line or "Decode batch" in line:
                # Extract full token usage and swa token usage
                full_match = re.search(r"full token usage: ([0-9.]+)", line)
                swa_match = re.search(r"swa token usage: ([0-9.]+)", line)
                req_match = re.search(r"#running-req: ([0-9]+)", line)

                # Extract generation throughput for decode batches
                gen_throughput_match = re.search(
                    r"gen throughput \(token/s\): ([0-9.]+)", line
                )

                if full_match and swa_match and req_match:
                    full_token_usage.append(float(full_match.group(1)))
                    swa_token_usage.append(float(swa_match.group(1)))
                    running_req.append(int(req_match.group(1)))

                    # Add generation throughput if available, otherwise 0
                    gen_throughput.append(
                        float(gen_throughput_match.group(1))
                        if gen_throughput_match
                        else 0.0
                    )

                    # Calculate time intervals in seconds from the first timestamp
                    interval = (actual_timestamp - start_time).total_seconds()
                    timestamps.append(interval)

            # Extract timing information (took xxx ms)
            took_match = re.search(r"took ([0-9.]+) ms", line)
            if took_match:
                timing_value = float(took_match.group(1))
                timing_values.append(timing_value)

                # Extract the operation name before "took"
                operation_match = re.search(r"\] ([^,]*?) took", line)
                if operation_match:
                    operation = operation_match.group(1).strip()
                else:
                    operation = "operation"
                timing_labels.append(operation)

                # Store actual timestamp for timing data
                if actual_timestamp:
                    interval = (actual_timestamp - start_time).total_seconds()
                    timing_timestamps.append(interval)

    return (
        timestamps,
        full_token_usage,
        swa_token_usage,
        running_req,
        gen_throughput,
        timing_timestamps,
        timing_values,
        timing_labels,
    )


def plot_data(
    ax_token,
    ax_perf,
    ax_timing,
    timestamps,
    full,
    swa,
    req,
    gen_throughput,
    timing_timestamps,
    timing_values,
    timing_labels,
    title_prefix,
):
    """Plot data on given subplots"""
    # Token usage plot
    ax_token.plot(timestamps, full, label="Full Token Usage", marker="o", markersize=3)
    ax_token.plot(timestamps, swa, label="SWA Token Usage", marker="s", markersize=3)
    ax_token.set_title(f"{title_prefix} - Token Usage vs Time")
    ax_token.set_ylabel("Token Usage")
    # With shared x-axis, only the bottom subplot needs x-label
    ax_token.grid(True)
    ax_token.legend()

    # Performance metrics plot
    ax_perf.plot(
        timestamps,
        req,
        label="#Running Requests",
        color="red",
        marker="^",
        markersize=3,
    )

    # Calculate mean for running requests
    req_mean = statistics.mean(req) if req else 0
    ax_perf.axhline(
        y=req_mean,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Mean Running Requests: {req_mean:.2f}",
    )

    ax_perf_twin = ax_perf.twinx()
    ax_perf_twin.fill_between(
        timestamps,
        gen_throughput,
        alpha=0.3,
        label="Gen Throughput (token/s)",
        color="olive",
    )

    # Calculate mean for gen throughput
    gen_throughput_mean = (
        statistics.mean([v for v in gen_throughput if v > 0]) if gen_throughput else 0
    )
    ax_perf_twin.axhline(
        y=gen_throughput_mean,
        color="olive",
        linestyle="--",
        alpha=0.7,
        label=f"Mean Gen Throughput: {gen_throughput_mean:.2f} token/s",
    )

    ax_perf.set_title(f"{title_prefix} - Performance Metrics vs Time")
    ax_perf.set_ylabel("#Running Requests", color="red")
    ax_perf.tick_params(axis="y", labelcolor="red")
    ax_perf_twin.set_ylabel("Gen Throughput (token/s)", color="olive")
    ax_perf_twin.tick_params(axis="y", labelcolor="olive")
    # With shared x-axis, only the bottom subplot needs x-label
    ax_perf.grid(True)

    # Combined legend
    lines1, labels1 = ax_perf.get_legend_handles_labels()
    lines2, labels2 = ax_perf_twin.get_legend_handles_labels()
    ax_perf.legend(lines1 + lines2, labels1 + labels2)

    # Timing plot
    if timing_values:
        ax_timing.scatter(
            timing_timestamps,
            timing_values,
            label="Operation Time (ms)",
            alpha=0.6,
            s=20,
        )
        # Add labels to some points to avoid overcrowding
        for i in range(
            0, len(timing_values), max(1, len(timing_values) // 10)
        ):  # Label every 10th point or so
            ax_timing.annotate(
                f"{timing_labels[i]}",
                (timing_timestamps[i], timing_values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )
    else:
        # Create an empty plot with the label so legend works
        ax_timing.scatter([], [], label="Operation Time (ms)", alpha=0.6, s=20)
    ax_timing.set_title(f"{title_prefix} - Operation Times (ms)")
    ax_timing.set_ylabel("Time (ms)")
    # With shared x-axis, only the bottom subplot needs x-label
    ax_timing.set_xlabel("Time (seconds)")
    ax_timing.grid(True)
    ax_timing.legend()


def plot_token_usage():
    """Plot token usage from log files"""
    # File paths
    file_false = "nohup.emem.0.out"
    file_true = "nohup.emem.1.out"

    # Create figure with subplots (3 rows x 2 columns) with shared x-axis
    fig, axes = plt.subplots(3, 2, figsize=(20, 18), sharex=True)
    fig.suptitle("Token Usage and Performance Analysis", fontsize=16)

    # Process and plot emem=false data
    if os.path.exists(file_false):
        (
            timestamps1,
            full1,
            swa1,
            req1,
            gen_throughput1,
            timing_timestamps1,
            timing_values1,
            timing_labels1,
        ) = parse_log_file(file_false)
        if timestamps1:  # Check if timestamps exist
            plot_data(
                axes[0, 0],
                axes[1, 0],
                axes[2, 0],
                timestamps1,
                full1,
                swa1,
                req1,
                gen_throughput1,
                timing_timestamps1,
                timing_values1,
                timing_labels1,
                "emem=false",
            )

    # Process and plot emem=true data
    if os.path.exists(file_true):
        (
            timestamps2,
            full2,
            swa2,
            req2,
            gen_throughput2,
            timing_timestamps2,
            timing_values2,
            timing_labels2,
        ) = parse_log_file(file_true)
        if timestamps2:  # Check if timestamps exist
            plot_data(
                axes[0, 1],
                axes[1, 1],
                axes[2, 1],
                timestamps2,
                full2,
                swa2,
                req2,
                gen_throughput2,
                timing_timestamps2,
                timing_values2,
                timing_labels2,
                "emem=true",
            )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("token_usage_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_token_usage()
