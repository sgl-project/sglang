from sglang.srt.batch_overlap.two_batch_overlap import _tbo_child_token_counts
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def test_tbo_child_token_counts_and_imbalance_use_host_indices():
    assert _tbo_child_token_counts(total_tokens=17, split_token_index=8) == (8, 9, 1)
    assert _tbo_child_token_counts(total_tokens=16, split_token_index=3) == (3, 13, 10)
