"""DSV4 baseline kv_canary e2e (manual, not registered in CUDA CI).

Runs the kv_canary baseline against ``sgl-project/DeepSeek-V4-Flash-FP8`` on
a 4-GPU host. Not registered with ``register_cuda_ci`` because no CI runner
class fits DSV4 (the model requires TP=4 minimum and there is no DSV4 tiny
variant).

Invocation:

    cd /Users/tom/main/workspaces/ws-main/worktrees/sglang-dev-a
    python -m unittest test.manual.kv_canary.test_self_e2e_baseline_dsv4 -v

The DSV4 adapter degrades to slot-lifecycle-only canary (no real-KV
fingerprint) because ``DeepSeekV4SingleKVPool`` stores 584 bytes/token which
is not 16-aligned. See ``sglang.srt.kv_canary.pool_patch.adapters.dsv4`` for
the explanation.
"""

from __future__ import annotations

import unittest
from test.registered.kv_canary.test_self_e2e_baseline import _BaselineBase

from sglang.test.kv_canary.dsv4_consts import (
    DSV4_POOL_SERVER_ARGS,
    DSV4_POOL_SERVER_ENV,
)


class TestBaselineDsv4(_BaselineBase):
    __test__ = True

    model_mode = "dsv4"
    extra_server_args = DSV4_POOL_SERVER_ARGS
    extra_env = DSV4_POOL_SERVER_ENV


if __name__ == "__main__":
    unittest.main()
