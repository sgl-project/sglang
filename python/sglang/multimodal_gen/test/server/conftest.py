import json
import os

import pytest

_PERF_RESULTS = pytest.StashKey[list]()


@pytest.fixture(scope="session")
def perf_results(request):
    """Share results through pytest rather than importing this conftest module."""
    return request.config.stash.setdefault(_PERF_RESULTS, [])


def _write_github_step_summary(content: str):
    """Write content to GitHub Step Summary if available."""
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a") as f:
            f.write(content)


def _write_results_json(results: list, output_path: str = "diffusion-results.json"):
    """Write performance results to JSON file for CI artifact collection."""
    try:
        existing = []
        if os.path.exists(output_path):
            try:
                with open(output_path, encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, list):
                    existing = loaded
            except json.JSONDecodeError:
                pass

        merged = {
            (
                entry.get("class_name"),
                entry.get("test_name"),
                entry.get("request_index", 1),
            ): entry
            for entry in existing + results
        }
        with open(output_path, "w") as f:
            json.dump(list(merged.values()), f, indent=2)
        print(f"[CONFTEST] Wrote results to {output_path}")
    except (json.JSONDecodeError, OSError) as e:
        print(f"[CONFTEST] Failed to write results JSON: {e}")


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
    markdown += "| Test Suite | Test Name | Request | Modality | E2E (ms) | Avg Denoise (ms) | Median Denoise (ms) | Load Peak VRAM (MiB) | Runtime Peak VRAM (MiB) |\n"
    markdown += "| ---------- | --------- | ------- | -------- | -------- | ---------------- | ------------------- | -------------------- | ----------------------- |\n"

    for entry in sorted(results, key=lambda x: (x["class_name"], x["test_name"])):
        modality = entry.get("modality", "image")
        markdown += (
            f"| {entry['class_name']} | {entry['test_name']} | {entry.get('request_index', 1)} | {modality} | "
            f"{entry['e2e_ms']:.2f} | {entry['avg_denoise_ms']:.2f} | "
            f"{entry['median_denoise_ms']:.2f} | "
            f"{entry.get('load_peak_vram_mb', 0):.0f} | "
            f"{entry.get('runtime_peak_vram_mb', 0):.0f} |\n"
        )

    # Video-specific metrics table (if any video tests)
    video_results = [r for r in results if r.get("modality") == "video"]
    if video_results:
        markdown += "\n### Video Generation Metrics\n\n"
        markdown += (
            "| Test Name | Request | FPS | Total Frames | Avg Frame Time (ms) |\n"
        )
        markdown += (
            "| --------- | ------- | --- | ------------ | ------------------- |\n"
        )
        for entry in video_results:
            fps = entry.get("frames_per_second", "N/A")
            frames = entry.get("total_frames", "N/A")
            avg_frame = entry.get("avg_frame_time_ms", "N/A")
            if isinstance(fps, float):
                fps = f"{fps:.2f}"
            if isinstance(avg_frame, float):
                avg_frame = f"{avg_frame:.2f}"
            markdown += f"| {entry['test_name']} | {entry.get('request_index', 1)} | {fps} | {frames} | {avg_frame} |\n"

    return markdown


def pytest_sessionfinish(session):
    """
    This hook is called by pytest at the end of the entire test session.
    It prints a consolidated summary of all performance results.
    """
    results = session.config.stash.get(_PERF_RESULTS, [])
    if not results:
        return

    sorted_results = sorted(
        results,
        key=lambda x: (x["class_name"], x["test_name"], x.get("request_index", 1)),
    )

    # Print to stdout (existing behavior)
    print("\n\n" + "=" * 35 + " Performance Summary " + "=" * 35)
    print(
        f"{'Test Suite':<30} | {'Test Name':<20} | {'Request':>7} | {'E2E (ms)':>12} | {'Avg Denoise (ms)':>18} | {'Median Denoise (ms)':>20} | {'Load Peak (MiB)':>15} | {'Runtime Peak (MiB)':>18}"
    )
    print(
        "-" * 30
        + "-+-"
        + "-" * 20
        + "-+-"
        + "-" * 7
        + "-+-"
        + "-" * 12
        + "-+-"
        + "-" * 18
        + "-+-"
        + "-" * 20
        + "-+-"
        + "-" * 15
        + "-+-"
        + "-" * 18
    )

    for entry in sorted_results:
        print(
            f"{entry['class_name']:<30} | {entry['test_name']:<20} | {entry.get('request_index', 1):>7} | {entry['e2e_ms']:>12.2f} | "
            f"{entry['avg_denoise_ms']:>18.2f} | {entry['median_denoise_ms']:>20.2f} | "
            f"{entry.get('load_peak_vram_mb', 0):>15.0f} | "
            f"{entry.get('runtime_peak_vram_mb', 0):>18.0f}"
        )

    print("=" * 130)

    print("\n\n" + "=" * 36 + " Detailed Reports " + "=" * 37)
    for entry in sorted_results:
        print(
            f"\n--- Details for {entry['class_name']} / {entry['test_name']} "
            f"/ request {entry.get('request_index', 1)} ---"
        )
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

    print("\n\n" + "=" * 34 + " Performance Data JSON " + "=" * 34)
    print(json.dumps(sorted_results, indent=2, sort_keys=True))
    print("=" * 91)

    # Write to GitHub Step Summary (new behavior for CI monitoring)
    markdown_report = _generate_diffusion_markdown_report(sorted_results)
    if markdown_report:
        _write_github_step_summary(markdown_report)

    # Write results to JSON file for CI artifact collection
    _write_results_json(sorted_results)
