from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_target_verify_skips_dcp_extend_metadata():
    runner = EagerRunner.__new__(EagerRunner)
    runner.enable_pdmux = False
    runner.load_batch = lambda batch, _proxy: batch

    model = Mock()
    model.forward.return_value = object()
    model_runner = SimpleNamespace(
        _extend_forward_kwargs=lambda *_args: {},
        _pp_kwargs=lambda *_args: {},
        ps=SimpleNamespace(attn_dcp_size=4),
        model=model,
        attn_backend=Mock(),
        device_timer=None,
        prefill_cuda_graph_runner=None,
    )
    runner.model_runner = model_runner
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        needs_forward_metadata_init=lambda: True,
        input_ids=object(),
        positions=object(),
    )

    runner._execute_extend(batch)

    model.prepare_context_parallel_metadata_for_dcp.assert_not_called()
    model_runner.attn_backend.init_forward_metadata.assert_called_once_with(batch)


@patch("sglang.srt.model_executor.runner.eager_runner.prepare_cp_forward")
@patch(
    "sglang.srt.model_executor.runner.eager_runner.is_cp_v2_active", return_value=True
)
def test_cp_v2_metadata_is_prepared_once(_is_cp_v2_active, prepare_cp_forward):
    runner = EagerRunner.__new__(EagerRunner)
    runner.enable_pdmux = False
    runner.load_batch = lambda batch, _proxy: batch
    runner._execute_extend_cp_v2 = Mock(return_value=object())

    model = Mock()
    model.forward.return_value = object()
    model_runner = SimpleNamespace(
        _extend_forward_kwargs=lambda *_args: {},
        _pp_kwargs=lambda *_args: {},
        ps=SimpleNamespace(attn_dcp_size=1),
        model=model,
        attn_backend=Mock(),
        device_timer=None,
        prefill_cuda_graph_runner=None,
    )
    runner.model_runner = model_runner
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        needs_forward_metadata_init=lambda: True,
        input_ids=object(),
        positions=object(),
    )

    runner._execute_extend(batch)

    prepare_cp_forward.assert_called_once_with(batch)
    model_runner.attn_backend.init_forward_metadata.assert_called_once_with(batch)
    runner._execute_extend_cp_v2.assert_called_once()
