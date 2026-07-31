from dataclasses import dataclass

from sglang_simulator.compat import (
    override_server_args,
    validate_benchmark_runtime,
    validate_launch_runtime,
)
from sglang_simulator.dataset.autobench import (
    AutoBenchmarkDataset,
    register_autobench_dataset,
)


def test_current_sglang_launch_surface_is_supported():
    validate_launch_runtime()


def test_current_sglang_benchmark_surface_is_supported():
    validate_benchmark_runtime()


def test_simulator_registers_its_autobench_contract():
    from sglang.benchmark import datasets

    register_autobench_dataset()
    assert datasets.DATASET_MAPPING["autobench"] is AutoBenchmarkDataset


def test_override_server_args_uses_resolved_api_when_available():
    @dataclass
    class Args:
        disable_cuda_graph: bool = False
        call: object = None

        def override(self, source, **fields):
            self.call = (source, fields)

    args = Args()
    override_server_args(args, disable_cuda_graph=True)

    assert args.call == ("sglang-simulator", {"disable_cuda_graph": True})


def test_override_server_args_supports_mutable_legacy_api():
    @dataclass
    class Args:
        disable_cuda_graph: bool = False

    args = Args()
    override_server_args(args, disable_cuda_graph=True)
    assert args.disable_cuda_graph is True
