import argparse
import json
import os

import pandas as pd
from tabulate import tabulate

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Parse JSONL benchmark and summarize.")
parser.add_argument("input_file", type=str, help="Path to input JSONL file")
parser.add_argument(
    "--md",
    action="store_true",
    help="If set, print the summary table in Markdown format (GitHub style)",
)
args = parser.parse_args()

input_file = args.input_file
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_file = f"{base_name}_summary.csv"

fields = [
    "max_concurrency",
    "input_throughput",
    "output_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
]

# Read JSONL and parse
results = []
with open(input_file, "r") as f:
    for line in f:
        data = json.loads(line)
        row = {field: data.get(field, None) for field in fields}
        max_conc = data.get("max_concurrency")
        out_tp = data.get("output_throughput")
        row["per_user_throughput"] = out_tp / max_conc if max_conc else None
        results.append(row)

# Convert to DataFrame
df = pd.DataFrame(results)

# Save to CSV
df.to_csv(output_file, index=False)
print(f"\nSaved summary to: {output_file}\n")

if args.md:
    # Print Markdown table
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".3f"))
else:
    # Print ASCII table
    print(tabulate(df, headers="keys", tablefmt="grid", floatfmt=".3f"))
