from __future__ import annotations

from typing import Any, ClassVar, Dict

from sglang.test.scripted_runtime.http_server import ScriptedHttpServer
from sglang.test.test_utils import CustomTestCase


class ScriptedTestCase(CustomTestCase):

    ENGINE_KWARGS: ClassVar[Dict[str, Any]] = {}

    server: ClassVar[ScriptedHttpServer]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not cls.ENGINE_KWARGS:
            raise AssertionError(
                f"{cls.__name__} must set ENGINE_KWARGS to a non-empty dict"
            )
        cls.server = ScriptedHttpServer.start(**cls.ENGINE_KWARGS)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.server.shutdown()
        finally:
            super().tearDownClass()
