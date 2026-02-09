_GLOBAL_PERF_RESULTS = []


def pytest_sessionfinish(session):
    """
    This hook is called by pytest at the end of the entire test session.
    It prints a consolidated summary of all performance results.
    """
    if not _GLOBAL_PERF_RESULTS:
        return

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

    for entry in sorted(_GLOBAL_PERF_RESULTS, key=lambda x: x["class_name"]):
        print(
            f"{entry['class_name']:<30} | {entry['test_name']:<20} | {entry['e2e_ms']:>12.2f} | "
            f"{entry['avg_denoise_ms']:>18.2f} | {entry['median_denoise_ms']:>20.2f}"
        )

    print("=" * 91)

    print("\n\n" + "=" * 36 + " Detailed Reports " + "=" * 37)
    for entry in sorted(_GLOBAL_PERF_RESULTS, key=lambda x: x["class_name"]):
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
