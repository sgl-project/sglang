import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Any

# Ensure the script can find the 'scripts' module
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from scripts.upload_performance_metrics import upload_metrics

# --- Test Case Configuration ---

# Define a list of performance test cases.
# Each test case is a dictionary with a descriptive name, the command to execute,
# and the W&B group name for organizing results.
TEST_CASES: List[Dict[str, Any]] = [
    {
        "name": "Wan2.1-1.3B-T2V-Performance-Test",
        "command": [
            "sgl-diffusion", "generate",
            "--model-path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "--num-gpus", "2",
            "--use-fsdp-inference",
            "--text-encoder-cpu-offload",
            "--pin-cpu-memory",
            "--prompt", "A curious raccoon",
            "--save-output",
            "--log-level", "info",
            "--sp-size", "-1",
            "--num-inference-steps=5"
        ],
        "wandb_group": "perf-test-wan-2.1-1.3b-t2v",
    },
    # --- Add new test cases here ---
    # Example of a future test case:
    # {
    #     "name": "Another-Model-Performance-Test",
    #     "command": ["sgl-diffusion", "generate", "--model-path", "...", ...],
    #     "wandb_group": "perf-test-another-model",
    # },
]

# --- W&B General Configuration ---
WANDB_PROJECT = "sgl_diffusion-performance"

# --- Log File Configuration ---
LOG_DIR = os.path.join(project_root, "logs")
RAW_PERF_LOG_FILE = os.path.join(LOG_DIR, "performance.log")
PROCESSED_LOG_FILE = os.path.join(LOG_DIR, "performance.log.processed")

def run_test_case(test_case: Dict[str, Any]):
    """
    Executes a single performance test case.
    """
    test_name = test_case["name"]
    command = test_case["command"]
    wandb_group = test_case["wandb_group"]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"🚀 Starting performance test: {test_name} (Run ID: {run_id})")

    # 1. Prepare log directory and file
    print("🧹 Preparing log file...")
    os.makedirs(LOG_DIR, exist_ok=True)
    if os.path.exists(RAW_PERF_LOG_FILE):
        os.remove(RAW_PERF_LOG_FILE)

    # 2. Run the generation command
    print("🎨 Running video generation command...")
    print(f"   Command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, text=True, capture_output=False)  # stream output
        print("✅ Video generation finished.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ Error during video generation for '{test_name}': {e}", file=sys.stderr)
        # print("\n--- STDOUT ---")
        # print(e.stdout)
        # print("\n--- STDERR ---")
        # print(e.stderr)
        return  # Stop this test case

    # 3. Upload performance metrics to W&B
    if os.path.exists(RAW_PERF_LOG_FILE):
        print("📈 Uploading performance metrics to W&B...")
        print(f"   Project: {WANDB_PROJECT}, Group: {wandb_group}")
        try:
            upload_metrics(
                log_file=RAW_PERF_LOG_FILE,
                wandb_project=WANDB_PROJECT,
                wandb_group=wandb_group,
                processed_log_file=PROCESSED_LOG_FILE,
                command=" ".join(command),
            )
            print("✅ Upload complete.")
        except Exception as e:
            print(f"❌ Failed to upload metrics for '{test_name}': {e}", file=sys.stderr)

        # 4. Archive the log file for this run
        archive_log_file = os.path.join(LOG_DIR, f"perf-test-{test_name}-{run_id}.log")
        print(f"🗄️ Archiving log file to {archive_log_file}")
        os.rename(RAW_PERF_LOG_FILE, archive_log_file)
    else:
        print("⚠️ No performance log file found to upload or archive.")


    print(f"🎉 Performance test '{test_name}' completed successfully!")
    print("-" * 80)


def main():
    """
    Main function to run all defined performance test cases.
    """
    print("=" * 80)
    print("Starting All Performance Tests")
    print("=" * 80)

    for test_case in TEST_CASES:
        run_test_case(test_case)

    print("All performance tests finished.")


if __name__ == "__main__":
    main()
