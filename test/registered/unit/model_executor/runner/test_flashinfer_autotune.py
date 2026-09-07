import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner import base_runner, flashinfer_autotune
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "mode,error",
    [
        ("prefill", "_dummy_run needs a static buffer"),
        ("decode", "ordinary or PD-prefill target EXTEND dummy"),
    ],
)
def test_packed_speculative_extend_is_limited_to_pd_prefill_target(mode, error):
    runner = SimpleNamespace(
        model_runner=SimpleNamespace(
            is_draft_worker=False,
            spec_algorithm=SimpleNamespace(is_speculative=lambda: True),
            decode_num_tokens_per_req=lambda: 6,
        )
    )
    with (
        patch.object(
            base_runner,
            "get_disagg",
            return_value=SimpleNamespace(disaggregation_mode=mode),
        ),
        patch.object(
            base_runner,
            "get_server_return_hidden_states_mode",
            return_value=CaptureHiddenMode.NULL,
        ),
        pytest.raises(AssertionError, match=error),
    ):
        base_runner.BaseRunner._dummy_run(
            runner,
            batch_size=1,
            buffers=None,
            forward_mode_override=ForwardMode.EXTEND,
            extend_num_tokens_per_req=1,
        )


def test_chunked_prefill_disabled_uses_legacy_token_ceiling():
    model_runner = SimpleNamespace(
        server_args=SimpleNamespace(),
        is_generation=True,
        is_draft_worker=False,
        spec_algorithm=SimpleNamespace(is_speculative=lambda: False),
        attn_backend=SimpleNamespace(extend_dummy_seqs_capped_by_req_pool=False),
        canary_manager=None,
    )
    runner = SimpleNamespace(
        model_runner=model_runner,
        _alloc_dummy_decode_buffers=Mock(return_value=object()),
        _dummy_run=Mock(),
    )
    with (
        patch.object(
            flashinfer_autotune.envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND,
            "get",
            return_value=True,
        ),
        patch.object(
            flashinfer_autotune,
            "get_disagg",
            return_value=SimpleNamespace(disaggregation_mode="prefill"),
        ),
        patch.object(flashinfer_autotune, "max_prefill_buffer_tokens", return_value=0),
        patch.object(
            flashinfer_autotune,
            "get_schedule",
            return_value=SimpleNamespace(max_prefill_tokens=32768),
        ),
        patch.object(flashinfer_autotune, "run_flashinfer_autotune_forward"),
        patch.object(flashinfer_autotune.torch.cuda, "empty_cache"),
    ):
        flashinfer_autotune.maybe_flashinfer_autotune_extend(
            runner, decode_num_tokens=128
        )

    runner._alloc_dummy_decode_buffers.assert_called_once_with(
        32768,
        num_tokens_per_req=1,
        allocate_logits_buffer=False,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
