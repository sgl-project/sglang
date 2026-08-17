"""EAGLE3 tree drafting (topk > 1) at page_size > 1, flashinfer (fa3 is
Hopper-only).

page_size=4 with 32 draft tokens spreads the draft window over several pages --
the layout the unit fixture refuses to build (tree draft is pinned to
page_size=1 there, see speculative_draft_runner.py). The window-inside-one-page
regime is covered by test_spec_eagle_fa3.py page256 on the Hopper runner.
"""

import unittest

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_server_kits import (
    SpecAccuracyKit,
    SpecFeatureKit,
    SpecLogprobKit,
)
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_cuda_ci(est_time=297, stage="base-b", runner_config="1-gpu-small")


class TestEagle3Page4Topk8(Eagle3Base, SpecAccuracyKit, SpecLogprobKit, SpecFeatureKit):
    """Overlap scheduler, topk=8 tree, page_size=4."""

    page_size = 4
    spec_topk = 8
    spec_tokens = 32
    disable_overlap = False
    # Preset accept-length values are topk=1 numbers -- loose for a topk=8
    # tree; tighten once CI reports the actuals.
    gsm8k_accept_len_thres = 2.0
    # Same fp32 logits budget as the topk=16 classes (bs * spec_tokens * vocab).
    max_running_requests = 16
    cuda_graph_max_bs_decode = 5
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


if __name__ == "__main__":
    unittest.main()
