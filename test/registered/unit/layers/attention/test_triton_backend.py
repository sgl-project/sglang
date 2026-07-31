import pytest

from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


@pytest.mark.parametrize("draft_width", [2, 4])
def test_target_verify_custom_mask_reserves_draft_suffix(draft_width):
    backend = object.__new__(TritonAttnBackend)
    backend.num_head = 1
    backend.max_kv_splits = 1
    backend.v_head_dim = 1
    backend.swa_v_head_dim = None
    backend.device = "cpu"
    backend.max_context_len = 16
    backend.num_draft_tokens = draft_width
    backend.skip_prefill = False
    backend.is_draft_runner = False
    backend.sliding_window_size = None
    backend.use_sliding_window_kv_pool = False
    backend._translate_kv_loc = None

    max_bs = 6
    max_num_tokens = max_bs * draft_width
    backend.init_cuda_graph_state(max_bs, max_num_tokens)

    assert backend.cuda_graph_custom_mask.numel() == max_num_tokens * (
        backend.max_context_len + draft_width
    )
