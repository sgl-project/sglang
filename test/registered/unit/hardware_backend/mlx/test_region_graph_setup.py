"""Phase routing and resource accounting of the shared MPS region."""

from types import SimpleNamespace
from unittest import mock

import pytest

from sglang.srt.hardware_backend.mlx.region_runner import MlxRegionRunner
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.model_executor.model_runner_components import cuda_graph_setup as setup
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_mps_ci

register_mps_ci(est_time=1, suite="stage-a-unit-test-mps")


@pytest.mark.parametrize("decode_enabled", [True, False])
def test_decode_factory_constructs_region_for_prefill_only_too(decode_enabled):
    config = CudaGraphConfig(
        prefill=PhaseConfig(backend=Backend.FULL, bs=[128]),
        decode=PhaseConfig(
            backend=Backend.FULL if decode_enabled else Backend.DISABLED, bs=[1, 2]
        ),
    )
    runner = SimpleNamespace(
        device="mps",
        gpu_id=0,
        is_draft_worker=False,
        is_generation=True,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
        req_to_token_pool=SimpleNamespace(size=8),
        _decode_cuda_graph_runner_cls=lambda: None,
    )
    region = object()
    with (
        get_context().override_server_args(cuda_graph_config=config),
        mock.patch.object(setup.current_platform, "is_out_of_tree", return_value=False),
        mock.patch.object(setup, "get_available_gpu_memory", side_effect=[8, 6]),
        mock.patch(
            "sglang.srt.hardware_backend.mlx.region_runner.MlxRegionRunner",
            return_value=region,
        ) as factory,
    ):
        capture = setup.capture_decode_graph(model_runner=runner)
    factory.assert_called_once_with(runner)
    assert capture.runner is region
    assert capture.memory_usage_gb == 2


@pytest.mark.parametrize(
    "prefill_enabled,decode_enabled",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_region_is_shared_only_by_enabled_phases(prefill_enabled, decode_enabled):
    config = CudaGraphConfig(
        prefill=PhaseConfig(
            backend=Backend.FULL if prefill_enabled else Backend.DISABLED
        ),
        decode=PhaseConfig(
            backend=Backend.FULL if decode_enabled else Backend.DISABLED
        ),
    )
    eager, region = object(), object()
    runner = SimpleNamespace(device="mps", is_draft_worker=False)
    capture = setup.GraphCapture(
        runner=region if prefill_enabled or decode_enabled else None,
        memory_phase="decode",
        memory_usage_gb=2 if prefill_enabled or decode_enabled else 0,
        capture_time=3 if prefill_enabled or decode_enabled else 0,
    )
    with (
        get_context().override_server_args(cuda_graph_config=config),
        mock.patch.object(setup.GraphSharedOutput, "create_for_model_runner"),
        mock.patch.object(setup, "EagerRunner", return_value=eager),
        mock.patch.object(setup, "refresh_deep_gemm_layout_memory_budget"),
        mock.patch.object(setup, "capture_decode_graph", return_value=capture),
    ):
        result = setup.capture_cuda_graphs(model_runner=runner, finalize=False)
    assert result.prefill.runner is (region if prefill_enabled else eager)
    assert result.decode.runner is (region if decode_enabled else None)
    assert sum(result.memory_usage.values()) == (
        2 if prefill_enabled or decode_enabled else 0
    )
    assert sum(result.time_usage.values()) == (
        3 if prefill_enabled or decode_enabled else 0
    )


@pytest.mark.parametrize(
    "prefill_enabled,decode_enabled", [(True, True), (True, False), (False, True)]
)
def test_disabled_phases_do_not_export_unused_buckets(prefill_enabled, decode_enabled):
    config = CudaGraphConfig(
        prefill=PhaseConfig(
            backend=Backend.FULL if prefill_enabled else Backend.DISABLED,
            bs=[128],
            full_prefill_max_req=2,
        ),
        decode=PhaseConfig(
            backend=Backend.FULL if decode_enabled else Backend.DISABLED, bs=[1, 2]
        ),
    )
    runner = SimpleNamespace(
        is_draft_worker=False,
        model=SimpleNamespace(logits_processor=None),
        sliding_window_size=None,
        req_to_token_pool=SimpleNamespace(size=8),
    )
    with (
        get_context().override_server_args(cuda_graph_config=config),
        mock.patch("sglang.srt.hardware_backend.mlx.region_runner.BaseRunner.__init__"),
        mock.patch(
            "sglang.srt.hardware_backend.mlx.region_runtime.validate_mlx_region_runtime"
        ),
        mock.patch(
            "sglang.srt.hardware_backend.mlx.region_runner._kernel_contract_reject_reason",
            return_value=None,
        ),
        mock.patch.object(MlxRegionRunner, "_export_at_startup"),
    ):
        region = MlxRegionRunner(runner)
    assert region._decode_batch_sizes == ((1, 2) if decode_enabled else ())
    assert region._prefill_token_buckets == ((128,) if prefill_enabled else ())
    assert region._prefill_batch_sizes == ((1, 2) if prefill_enabled else ())


@pytest.mark.parametrize("ratio", [0.5, float("nan"), float("inf")])
def test_invalid_packed_padding_policy_fails_before_capture(ratio):
    from sglang.srt.environ import envs

    with (
        envs.SGLANG_MLX_REGION_MAX_PREFILL_PADDING_RATIO.override(ratio),
        get_context().override_server_args(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(
                    backend=Backend.FULL, bs=[128, 256], full_prefill_max_req=2
                ),
                decode=PhaseConfig(backend=Backend.FULL, bs=[1, 2]),
            )
        ),
        mock.patch("sglang.srt.hardware_backend.mlx.region_runner.BaseRunner.__init__"),
        mock.patch(
            "sglang.srt.hardware_backend.mlx.region_runtime.validate_mlx_region_runtime"
        ),
        mock.patch.object(MlxRegionRunner, "_export_at_startup") as capture,
    ):
        with pytest.raises(ValueError, match="must be finite and >= 1"):
            MlxRegionRunner(SimpleNamespace(req_to_token_pool=SimpleNamespace(size=8)))
    capture.assert_not_called()
