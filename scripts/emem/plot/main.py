import os
import re

import matplotlib.pyplot as plt


def parse_log_file(file_path, label_prefix):
    """Parse log file and extract full token usage, swa token usage, running requests, and generation throughput data"""
    full_token_usage = []
    swa_token_usage = []
    running_req = []
    gen_throughput = []  # New data for generation throughput
    timestamps = []

    with open(file_path, "r") as f:
        for line in f:
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
                    if gen_throughput_match:
                        gen_throughput.append(float(gen_throughput_match.group(1)))
                    else:
                        gen_throughput.append(0.0)

                    timestamps.append(len(timestamps))

    return timestamps, full_token_usage, swa_token_usage, running_req, gen_throughput


def plot_token_usage():
    """Plot token usage from log files"""
    # File paths
    file_false_06 = "nohup.emem.false.ratio.0.6.out"
    file_false_10 = "nohup.emem.false.ratio.1.0.out"
    file_true_10 = "nohup.emem.true.ratio.1.0.v4.out"

    # Create figure with subplots (2 columns x 3 rows)
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    fig.suptitle("Token Usage and Performance Analysis", fontsize=16)

    # Collect all data for consistent y-axis scaling
    all_full_data = []
    all_swa_data = []
    all_req_data = []
    all_gen_throughput_data = []

    # Plot for emem=false, ratio=0.6
    if os.path.exists(file_false_06):
        timestamps1, full1, swa1, req1, gen_throughput1 = parse_log_file(
            file_false_06, "emem=false,ratio=0.6"
        )
        all_full_data.extend(full1)
        all_swa_data.extend(swa1)
        all_req_data.extend(req1)
        all_gen_throughput_data.extend(gen_throughput1)

    # Plot for emem=false, ratio=1.0
    if os.path.exists(file_false_10):
        timestamps2, full2, swa2, req2, gen_throughput2 = parse_log_file(
            file_false_10, "emem=false,ratio=1.0"
        )
        all_full_data.extend(full2)
        all_swa_data.extend(swa2)
        all_req_data.extend(req2)
        all_gen_throughput_data.extend(gen_throughput2)

    # Plot for emem=true, ratio=1.0
    if os.path.exists(file_true_10):
        timestamps3, full3, swa3, req3, gen_throughput3 = parse_log_file(
            file_true_10, "emem=true,ratio=1.0"
        )
        all_full_data.extend(full3)
        all_swa_data.extend(swa3)
        all_req_data.extend(req3)
        all_gen_throughput_data.extend(gen_throughput3)

    # Determine shared y-axis limits
    token_min = (
        min(min(all_full_data), min(all_swa_data))
        if all_full_data and all_swa_data
        else 0
    )
    token_max = (
        max(max(all_full_data), max(all_swa_data))
        if all_full_data and all_swa_data
        else 1
    )
    req_min = min(all_req_data) if all_req_data else 0
    req_max = max(all_req_data) if all_req_data else 1
    gen_throughput_min = min(all_gen_throughput_data) if all_gen_throughput_data else 0
    gen_throughput_max = max(all_gen_throughput_data) if all_gen_throughput_data else 1

    # Add some padding
    token_padding = (token_max - token_min) * 0.1
    req_padding = (req_max - req_min) * 0.1
    gen_throughput_padding = (gen_throughput_max - gen_throughput_min) * 0.1

    # Plot for emem=false, ratio=0.6
    if os.path.exists(file_false_06):
        timestamps1, full1, swa1, req1, gen_throughput1 = parse_log_file(
            file_false_06, "emem=false,ratio=0.6"
        )

        # Left column - Token usage
        ax1_left = axes[0, 0]
        ax1_left.plot(
            timestamps1, full1, label="Full Token Usage", marker="o", markersize=3
        )
        ax1_left.plot(
            timestamps1, swa1, label="SWA Token Usage", marker="s", markersize=3
        )
        ax1_left.set_title("emem=false, ratio=0.6 - Token Usage")
        ax1_left.set_ylabel("Token Usage")
        ax1_left.grid(True)
        ax1_left.set_ylim(token_min - token_padding, token_max + token_padding)
        ax1_left.legend(loc="upper left")

        # Right column - Running requests and generation throughput
        ax1_right = axes[0, 1]
        ax1_right.plot(
            timestamps1,
            req1,
            label="#Running Requests",
            color="red",
            marker="^",
            markersize=3,
            zorder=2,
        )
        ax1_right_twin = ax1_right.twinx()
        ax1_right_twin.fill_between(
            timestamps1,
            gen_throughput1,
            alpha=0.3,
            label="Gen Throughput (token/s)",
            color="olive",
            zorder=1,
        )
        ax1_right.set_title("emem=false, ratio=0.6 - Performance Metrics")
        ax1_right.set_ylabel("#Running Requests", color="red")
        ax1_right.tick_params(axis="y", labelcolor="red")
        ax1_right_twin.set_ylabel("Gen Throughput (token/s)", color="olive")
        ax1_right_twin.tick_params(axis="y", labelcolor="olive")
        ax1_right.set_ylim(req_min - req_padding, req_max + req_padding)
        ax1_right_twin.set_ylim(
            gen_throughput_min - gen_throughput_padding,
            gen_throughput_max + gen_throughput_padding,
        )
        ax1_right.grid(True)
        ax1_right_twin.grid(True)

        # Combined legend
        lines1, labels1 = ax1_right.get_legend_handles_labels()
        lines2, labels2 = ax1_right_twin.get_legend_handles_labels()
        ax1_right.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Plot for emem=false, ratio=1.0
    if os.path.exists(file_false_10):
        timestamps2, full2, swa2, req2, gen_throughput2 = parse_log_file(
            file_false_10, "emem=false,ratio=1.0"
        )

        # Left column - Token usage
        ax2_left = axes[1, 0]
        ax2_left.plot(
            timestamps2, full2, label="Full Token Usage", marker="o", markersize=3
        )
        ax2_left.plot(
            timestamps2, swa2, label="SWA Token Usage", marker="s", markersize=3
        )
        ax2_left.set_title("emem=false, ratio=1.0 - Token Usage")
        ax2_left.set_ylabel("Token Usage")
        ax2_left.grid(True)
        ax2_left.set_ylim(token_min - token_padding, token_max + token_padding)
        ax2_left.legend(loc="upper left")

        # Right column - Running requests and generation throughput
        ax2_right = axes[1, 1]
        ax2_right.plot(
            timestamps2,
            req2,
            label="#Running Requests",
            color="red",
            marker="^",
            markersize=3,
            zorder=2,
        )
        ax2_right_twin = ax2_right.twinx()
        ax2_right_twin.fill_between(
            timestamps2,
            gen_throughput2,
            alpha=0.3,
            label="Gen Throughput (token/s)",
            color="olive",
            zorder=1,
        )
        ax2_right.set_title("emem=false, ratio=1.0 - Performance Metrics")
        ax2_right.set_ylabel("#Running Requests", color="red")
        ax2_right.tick_params(axis="y", labelcolor="red")
        ax2_right_twin.set_ylabel("Gen Throughput (token/s)", color="olive")
        ax2_right_twin.tick_params(axis="y", labelcolor="olive")
        ax2_right.set_ylim(req_min - req_padding, req_max + req_padding)
        ax2_right_twin.set_ylim(
            gen_throughput_min - gen_throughput_padding,
            gen_throughput_max + gen_throughput_padding,
        )
        ax2_right.grid(True)
        ax2_right_twin.grid(True)

        # Combined legend
        lines1, labels1 = ax2_right.get_legend_handles_labels()
        lines2, labels2 = ax2_right_twin.get_legend_handles_labels()
        ax2_right.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Plot for emem=true, ratio=1.0
    if os.path.exists(file_true_10):
        timestamps3, full3, swa3, req3, gen_throughput3 = parse_log_file(
            file_true_10, "emem=true,ratio=1.0"
        )

        # Left column - Token usage
        ax3_left = axes[2, 0]
        ax3_left.plot(
            timestamps3, full3, label="Full Token Usage", marker="o", markersize=3
        )
        ax3_left.plot(
            timestamps3, swa3, label="SWA Token Usage", marker="s", markersize=3
        )
        ax3_left.set_title("emem=true, ratio=1.0 - Token Usage")
        ax3_left.set_ylabel("Token Usage")
        ax3_left.set_xlabel("Time Steps")
        ax3_left.grid(True)
        ax3_left.set_ylim(token_min - token_padding, token_max + token_padding)
        ax3_left.legend(loc="upper left")

        # Right column - Running requests and generation throughput
        ax3_right = axes[2, 1]
        ax3_right.plot(
            timestamps3,
            req3,
            label="#Running Requests",
            color="red",
            marker="^",
            markersize=3,
            zorder=2,
        )
        ax3_right_twin = ax3_right.twinx()
        ax3_right_twin.fill_between(
            timestamps3,
            gen_throughput3,
            alpha=0.3,
            label="Gen Throughput (token/s)",
            color="olive",
            zorder=1,
        )
        ax3_right.set_title("emem=true, ratio=1.0 - Performance Metrics")
        ax3_right.set_ylabel("#Running Requests", color="red")
        ax3_right.tick_params(axis="y", labelcolor="red")
        ax3_right_twin.set_ylabel("Gen Throughput (token/s)", color="olive")
        ax3_right_twin.tick_params(axis="y", labelcolor="olive")
        ax3_right.set_xlabel("Time Steps")
        ax3_right.set_ylim(req_min - req_padding, req_max + req_padding)
        ax3_right_twin.set_ylim(
            gen_throughput_min - gen_throughput_padding,
            gen_throughput_max + gen_throughput_padding,
        )
        ax3_right.grid(True)
        ax3_right_twin.grid(True)

        # Combined legend
        lines1, labels1 = ax3_right.get_legend_handles_labels()
        lines2, labels2 = ax3_right_twin.get_legend_handles_labels()
        ax3_right.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("token_usage_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_token_usage()
