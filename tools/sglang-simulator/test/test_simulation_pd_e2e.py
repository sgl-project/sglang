"""Disaggregated workloads across in-process prefill and decode instances.

Runs prefill and decode instances on the bundled model config and checks that
every request survives both passes, that its chunked prefill becomes one KV wave
per chunk with a Prefill prefix hit sent ahead of them, and that the decode pass
starts at the end of its own prefill rather than at the request's arrival.
"""

import json
import os
import sys
from pathlib import Path

import pytest
from sglang_simulator.dataset import GenericRequest, SimpleDataset
from sglang_simulator.simulation.benchmark import BenchmarkConfig

# A prefill server cannot run SGLang's fake backend, and the Mooncake backend
# initialises its transfer engine at startup even though nothing is sent.
pytest.importorskip("mooncake.engine")
if not hasattr(pytest.importorskip("sglang_router.sglang_router_rs"), "PyPolicy"):
    pytest.skip(
        "routing needs sglang_router built from sgl-model-gateway",
        allow_module_level=True,
    )

ASSETS = Path(__file__).parent / "assets"
EXAMPLE_PROFILES = (
    Path(__file__).parents[1]
    / "examples"
    / "network_profiles"
    / "example_profiles.json"
)
SGLANG_ROOT = Path(__file__).parents[3]
if str(SGLANG_ROOT) not in sys.path:
    sys.path.insert(0, str(SGLANG_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

CHUNKED_PREFILL_SIZE = 512
SHARED_PREFIX = 256
INPUT_LENGTH = 3 * CHUNKED_PREFILL_SIZE
# A request that hits an earlier request's one-page prefix on its prefill
# instance sends that prefix on its own as its first batch starts, then computes
# the rest in chunks of 512, 512 and 256 tokens.
COLD = [512, 512, 512]
WARM = [SHARED_PREFIX, 512, 512, 256]


@pytest.fixture(scope="module")
def sim_config(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("sim")
    table_path = tmp_path / "replay.json"
    table_path.write_text(
        json.dumps({"[[1, 1023]]": 0.001, "[[1024, 0]]": 0.01}),
        encoding="utf-8",
    )
    config = {
        "platform": {
            "accelerator": {"name": "a100_sxm", "hbm_capacity_gb": 80},
            "disk_read_bandwidth_gb": 8,
            "disk_write_bandwidth_gb": 8,
            "memory_read_bandwidth_gb": 64,
            "memory_write_bandwidth_gb": 64,
            "num_device_per_node": 8,
        },
        "predictor": {
            "name": "replay",
            "database_path": str(table_path),
            "miss_strategy": "knn",
            "miss_knn_k": 1,
        },
        "scheduler": {"tp_size": 1, "ep_size": 1, "dp_size": 1},
        "kv_transfer_config": {
            "network": "EXAMPLE_RDMA",
            "network_profiles": str(EXAMPLE_PROFILES),
            "seed": 0,
            # Release every chunk immediately so the wave split is deterministic.
            "bootstrap_alpha_s": 0.0,
        },
    }
    config_path = tmp_path / "sim_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    # The simulator reads this once per process, so every test shares it.
    os.environ["SGLANG_SIMULATOR_CONFIG_PATH"] = str(config_path)
    return config_path


def _server_args(mode: str, index: int):
    from sglang.srt.server_args import ServerArgs

    return ServerArgs(
        model_path=str(ASSETS / "qwen3-8b"),
        load_format="dummy",
        device="cpu",
        max_total_tokens=10 * 1024,
        page_size=256,
        chunked_prefill_size=CHUNKED_PREFILL_SIZE,
        skip_tokenizer_init=True,
        disaggregation_mode=mode,
        # The prefill instance must not carry the fake transport (SGLang asserts
        # against it); the simulator swaps in a FakeKVSender at construction.
        disaggregation_transfer_backend="mooncake",
        disaggregation_bootstrap_port=18998 + index,
    )


def _dataset(count: int) -> SimpleDataset:
    return SimpleDataset(
        reqs=[
            GenericRequest(
                token_ids=[999] * SHARED_PREFIX
                + [1000 + i] * (INPUT_LENGTH - SHARED_PREFIX),
                input_length=INPUT_LENGTH,
                output_length=2,
                custom_params={"created_time": i / 10},
            )
            for i in range(count)
        ]
    )


@pytest.mark.parametrize(
    ("num_prefill", "num_decode", "expected_waves"),
    [
        # The second request hits the first one's prefix.
        (1, 1, [COLD, WARM]),
        # Round robin sends requests 0 and 2 to prefill 0 and 1 and 3 to
        # prefill 1, so only the later request on each instance hits a prefix.
        # Prefill 1 first sees a request at 0.1 s, not at 0.
        (2, 2, [COLD, COLD, WARM, WARM]),
    ],
)
def test_pd_disagg_runs_every_instance(
    sim_config, tmp_path, num_prefill, num_decode, expected_waves
):
    from benchmark.simulator.pd_bench_runner import PDDisaggBenchmarkRunner

    count = len(expected_waves)
    runner = PDDisaggBenchmarkRunner(
        [_server_args("prefill", i) for i in range(num_prefill)],
        [_server_args("decode", i) for i in range(num_decode)],
        output_dir=str(tmp_path / "out"),
    )
    try:
        dataset = _dataset(count)
        arrivals = [req.custom_params["created_time"] for req in dataset]

        metrics = runner.benchmark(
            BenchmarkConfig(ignore_request_timestamp=False), dataset
        )

        assert metrics["paired_requests"] == count
        assert metrics["completed"] == count
        assert len(metrics["prefill"]) == num_prefill
        assert len(metrics["decode"]) == num_decode
        prefill = {stat["rid"]: stat for stat in runner.get_prefill_request_stats()}
        decode = {stat["rid"]: stat for stat in runner.get_decode_request_stats()}
        combined = {stat["rid"]: stat for stat in runner.get_request_stats()}
        assert len(prefill) == len(decode) == len(combined) == count

        assert [
            combined[req.extra_args["rid"]]["kv_xfer_wave_tokens"] for req in dataset
        ] == expected_waves
        for index, (req, arrival) in enumerate(zip(dataset, arrivals)):
            rid = req.extra_args["rid"]
            assert (
                combined[rid]["prefill_instance"],
                combined[rid]["decode_instance"],
            ) == (index % num_prefill, index % num_decode)
            # Every instance reports on the workload's timeline.
            assert prefill[rid]["created_time"] == pytest.approx(arrival)
            visible = combined[rid]["kv_xfer_visible_s"]
            assert visible > 0
            # Decode cannot run before the modelled KV transfer completes.
            assert decode[rid]["queue_end"] - decode[rid]["created_time"] >= (
                visible - 1e-9
            )
            # The client-visible TTFT includes that wait.
            assert (
                combined[rid]["gen_token_latencies"][0]
                > prefill[rid]["gen_token_latencies"][0]
            )
            # Prefill takes simulated time, so every request must enter the
            # decode pass strictly later than it arrived. Equality would mean
            # the pairing silently fell back to the arrival time.
            assert decode[rid]["created_time"] > arrival
    finally:
        runner.shutdown()
