import torch

from sglang.srt.layers.attention.aiter_backend import _get_aiter_max_batch_size
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_aiter_max_batch_size_includes_padding_row_and_topk():
    req_to_token = torch.empty((112, 8), device="meta")

    assert _get_aiter_max_batch_size(req_to_token, topk=1) == 112
    assert _get_aiter_max_batch_size(req_to_token, topk=4) == 448
