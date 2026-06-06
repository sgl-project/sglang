from __future__ import annotations

from typing import ClassVar

from sglang.test.kv_canary.consts import SWA_POOL_SERVER_ARGS
from sglang.test.kv_canary.e2e_base import CanaryE2EBase

PP_SIZE: int = 2


class CanaryPPFixture(CanaryE2EBase):

    model_mode: ClassVar[str] = "swa"
    workload_n_batches: ClassVar[int] = 2

    @classmethod
    def setUpClass(cls) -> None:
        cls.extra_server_args = (
            "--pp-size",
            str(PP_SIZE),
            "--disable-cuda-graph",
            *SWA_POOL_SERVER_ARGS,
            *cls.extra_server_args,
        )
        super().setUpClass()
