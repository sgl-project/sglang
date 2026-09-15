from __future__ import annotations

import unittest
from test.registered.kv_canary.test_self_e2e_baseline import _BaselineBase

from sglang.test.kv_canary.consts import (
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
