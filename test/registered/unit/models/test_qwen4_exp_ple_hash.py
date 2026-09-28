import pytest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.qwen4_ple_utils import assert_host_hash_matches, make_embedding

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("ngram_size", [2, 3, 5])
def test_host_hash_matches_cpu(ngram_size):
    assert_host_hash_matches(make_embedding(ngram_size, device="cpu"))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
