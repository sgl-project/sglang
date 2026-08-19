import time

from sglang.multimodal_gen.runtime.utils.startup_profiler import StartupProfiler


def test_disabled_profiler_is_zero_overhead_noop():
    """When disabled, phase() must not build a tree at all (see #19087 -- the
    profiler must cost nothing when off, since it wraps hot startup code)."""
    profiler = StartupProfiler(enabled=False)
    with profiler.phase("a"):
        with profiler.phase("b"):
            pass
    assert profiler._root.children == []
    assert profiler.render() == ""


def test_nested_phases_report_percent_of_immediate_parent():
    """A child's percentage is relative to its parent's duration, not the
    grand total -- this is what makes `load_component.text_encoder: 52%`
    meaningful instead of misleading once there's more than one level."""
    profiler = StartupProfiler(enabled=True)
    with profiler.phase("outer"):
        time.sleep(0.02)
        with profiler.phase("inner_a"):
            time.sleep(0.01)
        with profiler.phase("inner_b"):
            time.sleep(0.005)

    lines = profiler.render().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("outer: ")
    assert "(100.0%)" in lines[0]  # top-level phase is 100% of itself
    assert lines[1].startswith("outer.inner_a: ")
    assert lines[2].startswith("outer.inner_b: ")

    # inner_a took ~2x inner_b (10ms vs 5ms), so its percentage should be
    # roughly double -- a derived property of the timings, not a fixed value.
    pct_a = float(lines[1].split("(")[1].rstrip("%)"))
    pct_b = float(lines[2].split("(")[1].rstrip("%)"))
    assert pct_a > pct_b


def test_sibling_phases_at_top_level_are_independent():
    """Two unrelated top-level phases (e.g. init_distributed_environment and
    build_pipeline) must not be nested under each other."""
    profiler = StartupProfiler(enabled=True)
    with profiler.phase("first"):
        pass
    with profiler.phase("second"):
        pass

    lines = profiler.render().splitlines()
    assert lines[0].startswith("first: ")
    assert lines[1].startswith("second: ")
