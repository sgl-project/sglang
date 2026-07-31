import json
import sys
from types import SimpleNamespace

import pytest
import torch
from sglang_simulator.simulation.manager.config import ConfigManager
from sglang_simulator.simulation.manager.state import StateManager
from sglang_simulator.simulation.sglang.mem_cache_allocator import (
    alloc_decode_cpu,
    alloc_extend_cpu,
)
from sglang_simulator.simulation.sglang.scheduler import (
    block_on_l2_load,
    effective_l2_load_delay,
    simulation_mode_log_message,
)
from sglang_simulator.simulation.types import RequestStats, SimulationMode
from sglang_simulator.simulation.utils import (
    calc_input_token_metrics,
    calc_iteration_metrics,
    calc_metrics,
    estimate_kv_cache_pool_capacity,
)
from sglang_simulator.spec.accelerator import AcceleratorInfo


def test_alloc_extend_cpu_preserves_page_layout_across_sequences():
    page_size = 4
    prefix_lens = torch.tensor([3, 4, 0], dtype=torch.int64)
    seq_lens = torch.tensor([6, 5, 3], dtype=torch.int64)
    last_loc = torch.tensor([10, 19, -1], dtype=torch.int64)
    free_pages = torch.tensor([7, 8, 9], dtype=torch.int64)
    out_indices = torch.empty(7, dtype=torch.int64)

    alloc_extend_cpu(
        prefix_lens,
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
        bs_upper=3,
        page_size=page_size,
    )

    # Sequence 0 reuses one slot in its current page, then uses page 7.
    # Sequence 1 uses page 8. Sequence 2 uses page 9.
    assert out_indices.tolist() == [11, 28, 29, 32, 36, 37, 38]


def test_alloc_decode_cpu_only_consumes_pages_at_page_boundaries():
    page_size = 4
    seq_lens = torch.tensor([4, 5, 8, 9], dtype=torch.int64)
    last_loc = torch.tensor([2, 15, 30, 35], dtype=torch.int64)
    free_pages = torch.tensor([7, 8], dtype=torch.int64)
    out_indices = torch.empty(4, dtype=torch.int64)

    alloc_decode_cpu(
        seq_lens,
        last_loc,
        free_pages,
        out_indices,
        bs_upper=4,
        page_size=page_size,
    )

    assert out_indices.tolist() == [3, 28, 31, 32]


def test_input_token_metrics_keep_accounting_exact():
    metrics = calc_input_token_metrics(
        total_input=1000,
        total_reused_tokens=400,
        total_dur_s=10,
    )

    assert metrics["total_new_input"] == 600
    assert metrics["new_input_write_throughput_tokens_per_s"] == 60


def test_capacity_estimation_never_falls_back_to_local_gpu():
    accelerator = AcceleratorInfo(
        name="unknown-accelerator",
        vendor=None,
        hbm_capacity_gb=None,
        hbm_bandwidth_gb=None,
    )

    # Model and scheduler config deliberately stay unset: missing simulated
    # HBM must fail before any predictor setup or host-device probing occurs.
    with pytest.raises(ValueError, match="never falls back to the local GPU"):
        estimate_kv_cache_pool_capacity(None, accelerator, None)


def test_request_metrics_preserve_prefix_and_tier_hit_ratios(monkeypatch):
    class FakeConfigManager:
        @staticmethod
        def get_model_info():
            return None

        @staticmethod
        def get_scheduler_config():
            return None

        @staticmethod
        def get_platform_config():
            return None

    monkeypatch.setitem(
        sys.modules,
        "sglang_simulator.simulation.manager",
        SimpleNamespace(ConfigManager=FakeConfigManager),
    )

    requests = [
        RequestStats(
            rid="r1",
            input_length=100,
            output_length=2,
            final_device_hit_len=60,
            final_host_hit_len=20,
            final_storage_hit_len=5,
            created_time=-0.01,
            queue_start=0.00,
            queue_end=0.01,
            last_event_time=0.20,
            gen_token_latencies=[0.10, 0.02],
        ),
        RequestStats(
            rid="r2",
            input_length=100,
            output_length=1,
            final_device_hit_len=40,
            final_host_hit_len=10,
            final_storage_hit_len=0,
            created_time=0.03,
            queue_start=0.05,
            queue_end=0.07,
            last_event_time=0.50,
            gen_token_latencies=[0.20],
        ),
    ]

    metrics = calc_metrics(requests)

    assert metrics["completed"] == 2
    assert metrics["total_input"] == 200
    assert metrics["prefix_cache_reused_ratio"] == pytest.approx(0.50)
    assert metrics["kv_cache_device_hit_ratio"] == pytest.approx(0.35)
    assert metrics["kv_cache_host_hit_ratio"] == pytest.approx(0.125)
    assert metrics["kv_cache_storage_hit_ratio"] == pytest.approx(0.025)
    assert metrics["mean_ttft_ms"] == pytest.approx(150)
    assert metrics["mean_queue_ms"] == pytest.approx(15)
    assert metrics["mean_dispatch_wait_ms"] == pytest.approx(15)
    assert metrics["mean_arrival_to_prefill_ms"] == pytest.approx(30)
    deferred_metrics = {
        "L3_write_throughput_tokens_per_s",
        "L3_write_throughput_GB_per_s",
        "l3_to_l2_tokens",
        "l3_to_l2_GB",
        "l3_to_l2_throughput_tokens_per_s",
        "l3_to_l2_throughput_GB_per_s",
        "l2_to_l1_tokens",
        "l2_to_l1_GB",
        "l2_to_l1_throughput_tokens_per_s",
        "l2_to_l1_throughput_GB_per_s",
        "l2_hicache_pool_capacity_GB",
        "memory_read_bandwidth_GBps",
        "memory_write_bandwidth_GBps",
        "page_size",
        "tp_size",
        "dp_size",
        "ep_size",
        "pp_size",
        "num_device_per_node",
        "kv_cache_kb_per_token",
        "total_new_input_GB",
        "new_input_write_throughput_GB_per_s",
    }
    assert deferred_metrics.isdisjoint(metrics)


def test_request_metrics_clamp_negative_queue_duration_to_zero():
    metrics = calc_metrics(
        [
            RequestStats(
                rid="clock-skew",
                input_length=8,
                output_length=1,
                queue_start=0.20,
                queue_end=0.10,
                last_event_time=0.30,
                gen_token_latencies=[0.05],
            )
        ]
    )

    assert metrics["mean_queue_ms"] == 0


def test_request_metrics_clamp_the_mean_not_each_queue_sample():
    metrics = calc_metrics(
        [
            RequestStats(
                rid="negative",
                input_length=8,
                output_length=1,
                queue_start=0.20,
                queue_end=0.10,
                last_event_time=0.30,
                gen_token_latencies=[0.05],
            ),
            RequestStats(
                rid="positive",
                input_length=8,
                output_length=1,
                queue_start=0.10,
                queue_end=0.30,
                last_event_time=0.40,
                gen_token_latencies=[0.05],
            ),
        ]
    )

    assert metrics["mean_queue_ms"] == pytest.approx(50)


def test_iteration_metrics_keep_stable_summary_only():
    metrics = calc_iteration_metrics(
        [
            {
                "forward_latency": 0.10,
                "l2_load_latency": 0.02,
                "cpu_overhead": 0.01,
                "l2_blocking_wall_latency": 0.018,
                "h2d_load_call_count": 1,
                "h2d_load_segment_count": 2,
                "h2d_load_units": 3,
                "h2d_load_bytes": 4096,
            },
            {
                "forward_latency": 0.20,
                "l2_load_latency": 0,
                "cpu_overhead": 0.02,
                "h2d_load_call_count": 4,
                "h2d_load_segment_count": 5,
                "h2d_load_units": 6,
                "h2d_load_bytes": 8192,
            },
        ]
    )

    assert metrics["iterations"] == 2
    assert metrics["avg_iter_latency_ms"] == pytest.approx(175)
    assert set(metrics) == {"iterations", "avg_iter_latency_ms"}


def test_state_manager_pop_and_reset_do_not_leak_between_runs():
    StateManager.reset()
    StateManager.step_global_clock(1.25)
    StateManager.set_current_inference_dur(0.10)
    StateManager.set_current_inference_dur(0.20)
    StateManager.inc_hicache_l2_load_dur(0.03)
    StateManager.inc_hicache_l2_load_stats(1, 2, 3, 4096)

    assert StateManager.get_global_clock() == pytest.approx(1.25)
    assert StateManager.get_last_inference_dur() == pytest.approx(0.10)
    assert StateManager.get_current_inference_dur() == pytest.approx(0.20)
    assert StateManager.pop_hicache_l2_load_dur() == pytest.approx(0.03)
    assert StateManager.pop_hicache_l2_load_dur() == 0


def test_l2_load_visible_delay_matches_overlap_semantics():
    assert effective_l2_load_delay(0.30, 0.20, False) == pytest.approx(0.30)
    assert effective_l2_load_delay(0.30, 0.20, True) == pytest.approx(0.10)
    assert effective_l2_load_delay(0.10, 0.20, True) == 0.0


def test_blocking_l2_load_sleeps_but_offline_does_not(monkeypatch):
    sleeps = []
    clock = iter((10.0, 10.25))
    monkeypatch.setattr(
        "sglang_simulator.simulation.sglang.scheduler.time.sleep",
        sleeps.append,
    )
    monkeypatch.setattr(
        "sglang_simulator.simulation.sglang.scheduler.time.perf_counter",
        lambda: next(clock),
    )

    assert block_on_l2_load(SimulationMode.BLOCKING, 0.20) == pytest.approx(0.25)
    assert sleeps == [0.20]
    assert block_on_l2_load(SimulationMode.OFFLINE, 0.20) == 0.0
    assert sleeps == [0.20]


@pytest.mark.parametrize("mode", [SimulationMode.OFFLINE, SimulationMode.BLOCKING])
def test_simulation_mode_log_message(mode):
    assert (
        simulation_mode_log_message(mode)
        == f"SGLang Simulator simulation mode: {mode.value}"
    )


def test_ignore_cpu_overhead_defaults_to_0714_semantics(monkeypatch, tmp_path):
    config_path = tmp_path / "simulator.json"
    config_path.write_text(json.dumps({"scheduler": {}}))
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config_path))
    ConfigManager.reset_config_cache()

    assert ConfigManager.ignore_cpu_overhead() is False


def test_ignore_cpu_overhead_can_be_enabled_and_cache_is_reset(monkeypatch, tmp_path):
    config_path = tmp_path / "simulator.json"
    config_path.write_text(json.dumps({"scheduler": {"ignore_cpu_overhead": True}}))
    monkeypatch.setenv("SGLANG_SIMULATOR_CONFIG_PATH", str(config_path))
    ConfigManager.reset_config_cache()
    assert ConfigManager.ignore_cpu_overhead() is True

    config_path.write_text(json.dumps({"scheduler": {"ignore_cpu_overhead": False}}))
    ConfigManager.reset_config_cache()
    assert ConfigManager.ignore_cpu_overhead() is False
