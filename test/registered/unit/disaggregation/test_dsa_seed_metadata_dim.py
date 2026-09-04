import sys
from types import SimpleNamespace

import pytest

from sglang.srt.configs.model_config import is_deepseek_dsa
from sglang.srt.disaggregation.utils import get_dsa_seed_metadata_dim
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _dsa_config(*, index_topk, index_share_for_mtp_iteration):
    """Minimal stand-in for a DSA checkpoint's HF config (e.g. GLM-5.3)."""
    return SimpleNamespace(
        architectures=["GlmMoeDsaForCausalLM"],
        index_topk=index_topk,
        index_share_for_mtp_iteration=index_share_for_mtp_iteration,
    )


def test_seed_metadata_dim_zero_when_dsa_disabled_but_share_flag_set():
    """DSA disabled via index_topk=None must not assert in get_dsa_index_topk.

    Regression for https://github.com/sgl-project/sglang/issues/36598: the
    scheduler gates PD seed-metadata sizing on index_share_for_mtp_iteration
    only, so a DSA-disabled checkpoint (index_topk overridden to null) with the
    share flag still true reached get_dsa_index_topk, which asserts
    is_deepseek_dsa(). With DSA off the share flag is moot: the width is 0.
    """
    cfg = _dsa_config(index_topk=None, index_share_for_mtp_iteration=True)
    assert not is_deepseek_dsa(cfg)
    assert get_dsa_seed_metadata_dim(cfg) == 0


def test_seed_metadata_dim_with_dsa_enabled_and_share_flag():
    cfg = _dsa_config(index_topk=2048, index_share_for_mtp_iteration=True)
    assert is_deepseek_dsa(cfg)
    assert get_dsa_seed_metadata_dim(cfg) == 2048


def test_seed_metadata_dim_with_dsa_enabled_without_share_flag():
    cfg = _dsa_config(index_topk=2048, index_share_for_mtp_iteration=False)
    assert is_deepseek_dsa(cfg)
    assert get_dsa_seed_metadata_dim(cfg) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
