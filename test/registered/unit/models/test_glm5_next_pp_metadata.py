import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(
    est_time=10,
    suite="base-a-test-cpu",
    nightly=False,
    disabled=None,
)


def _make_self():
    return SimpleNamespace(
        dsa_enable_prefill_cp=False,
        mla_enable_prefill_cp=False,
        cp_rank=1,
        cp_size=2,
        use_dsa=True,
    )


def test_pp_nonfirst_rank_both_inputs_none_does_not_crash():
    hidden = torch.zeros((8, 4096))
    proxy = PPProxyTensors({"hidden_states": hidden})
    self = _make_self()
    Glm5NextForConditionalGeneration._prepare_context_parallel_metadata(
        self, None, None, SimpleNamespace(), proxy
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
