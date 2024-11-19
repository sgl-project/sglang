import re

import matplotlib.pyplot as plt


def parse_log_file(filename):
    # Dictionary to store count of workloads per rank
    rank_counts = {}

    # Regular expression pattern to match any line with dp_rank
    pattern = r"Turn\s+\d+:\s*hit rate=[\d.]+%,\s*dp_rank=(\d+)"

    with open(filename, "r") as file:
        content = file.read()

        # Find all matches
        matches = re.finditer(pattern, content)

        # Count occurrences of each rank
        for match in matches:
            rank = int(match.group(1))
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

    return rank_counts


def create_bar_chart(rank_counts):
    # Create lists for x and y values
    ranks = sorted(rank_counts.keys())
    counts = [rank_counts[rank] for rank in ranks]

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(ranks, counts)

    # Customize the chart
    plt.title("Total Workload Distribution Across Ranks\n(All Turns and Hit Rates)")
    plt.xlabel("Rank")
    plt.ylabel("Number of Workloads")

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    # Set x-axis ticks to show all ranks
    plt.xticks(ranks)

    # Add grid for better readability
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Save the plot
    plt.savefig("workload_distribution.png", dpi=300, bbox_inches="tight")
    print("Chart has been saved as 'workload_distribution.png'")


def main():
    # Replace 'paste.txt' with your log file name
    filename = "./client_random.log"

    try:
        # Parse the log file
        rank_counts = parse_log_file(filename)

        # Create and save the bar chart
        create_bar_chart(rank_counts)

        # Print the counts
        print("\nTotal workload counts per rank:")
        for rank in sorted(rank_counts.keys()):
            print(f"Rank {rank}: {rank_counts[rank]} workloads")

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
