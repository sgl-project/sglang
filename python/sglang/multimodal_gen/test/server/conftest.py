import os

import pytest

print("[CONFTEST] Loading conftest.py at import time")

# Use pytest's stash instead of global variable to avoid double-import issues
perf_results_key = pytest.StashKey[list]()
print(f"[CONFTEST] perf_results_key created: {perf_results_key}")


def add_perf_results(config, results: list):
    """Add performance results to the shared stash."""
    existing = config.stash.get(perf_results_key, [])
    existing.extend(results)
    config.stash[perf_results_key] = existing
    print(f"[CONFTEST] Added {len(results)} results, total now: {len(existing)}")


@pytest.fixture(scope="session")
def pytest_config(request):
    """Provide access to pytest config for storing perf results."""
    return request.config


def _write_github_step_summary(content: str):
    """Write content to GitHub Step Summary if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(content)


def _generate_diffusion_markdown_report(results: list) -> str:
    """Generate a markdown report for diffusion performance results."""
    if not results:
        return ""

    gpu_config = os.environ.get("GPU_CONFIG", "")
    header = "## Diffusion Performance Summary"
    if gpu_config:
        header += f" [{gpu_config}]"
    header += "\n\n"

    # Main performance table
    markdown = header
    markdown += "| Test Suite | Test Name | Modality | E2E (ms) | Avg Denoise (ms) | Median Denoise (ms) |\n"
    markdown += "| ---------- | --------- | -------- | -------- | ---------------- | ------------------- |\n"

    for entry in sorted(results, key=lambda x: (x["class_name"], x["test_name"])):
        modality = entry.get("modality", "image")
        markdown += (
            f"| {entry['class_name']} | {entry['test_name']} | {modality} | "
            f"{entry['e2e_ms']:.2f} | {entry['avg_denoise_ms']:.2f} | "
            f"{entry['median_denoise_ms']:.2f} |\n"
        )

    # Video-specific metrics table (if any video tests)
    video_results = [r for r in results if r.get("modality") == "video"]
    if video_results:
        markdown += "\n### Video Generation Metrics\n\n"
        markdown += "| Test Name | FPS | Total Frames | Avg Frame Time (ms) |\n"
        markdown += "| --------- | --- | ------------ | ------------------- |\n"
        for entry in video_results:
            fps = entry.get("frames_per_second", "N/A")
            frames = entry.get("total_frames", "N/A")
            avg_frame = entry.get("avg_frame_time_ms", "N/A")
            if isinstance(fps, float):
                fps = f"{fps:.2f}"
            if isinstance(avg_frame, float):
                avg_frame = f"{avg_frame:.2f}"
            markdown += f"| {entry['test_name']} | {fps} | {frames} | {avg_frame} |\n"

    return markdown


def pytest_sessionfinish(session):
    """
    This hook is called by pytest at the end of the entire test session.
    It prints a consolidated summary of all performance results.
    """
    # Get results from pytest's stash (shared across all import contexts)
    results = session.config.stash.get(perf_results_key, [])
    print(f"\n[DEBUG] pytest_sessionfinish called, has {len(results)} entries")
    if not results:
        print("[DEBUG] No results collected, skipping summary output")
        return

    # Print to stdout (existing behavior)
    print("\n\n" + "=" * 35 + " Performance Summary " + "=" * 35)
    print(
        f"{'Test Suite':<30} | {'Test Name':<20} | {'E2E (ms)':>12} | {'Avg Denoise (ms)':>18} | {'Median Denoise (ms)':>20}"
    )
    print(
        "-" * 30
        + "-+-"
        + "-" * 20
        + "-+-"
        + "-" * 12
        + "-+-"
        + "-" * 18
        + "-+-"
        + "-" * 20
    )

    for entry in sorted(results, key=lambda x: x["class_name"]):
        print(
            f"{entry['class_name']:<30} | {entry['test_name']:<20} | {entry['e2e_ms']:>12.2f} | "
            f"{entry['avg_denoise_ms']:>18.2f} | {entry['median_denoise_ms']:>20.2f}"
        )

    print("=" * 91)

    print("\n\n" + "=" * 36 + " Detailed Reports " + "=" * 37)
    for entry in sorted(results, key=lambda x: x["class_name"]):
        print(f"\n--- Details for {entry['class_name']} / {entry['test_name']} ---")
        stage_report = ", ".join(
            f"{name}:{duration:.2f}ms"
            for name, duration in entry.get("stage_metrics", {}).items()
        )
        if stage_report:
            print(f"    Stages: {stage_report}")

        sampled_steps = entry.get("sampled_steps") or {}
        if sampled_steps:
            step_report = ", ".join(
                f"{idx}:{duration:.2f}ms"
                for idx, duration in sorted(sampled_steps.items())
            )
            print(f"    Sampled Steps: {step_report}")
    print("=" * 91)

    # Write to GitHub Step Summary (new behavior for CI monitoring)
    markdown_report = _generate_diffusion_markdown_report(results)
    if markdown_report:
        _write_github_step_summary(markdown_report)
