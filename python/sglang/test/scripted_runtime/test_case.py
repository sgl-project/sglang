"""unittest base class wrapping :class:`ScriptedHttpServer`.

Subclass and set ``ENGINE_KWARGS`` as a class attribute. The base owns
the HTTP-server lifecycle — one server per concrete class, shared across
every ``test_*`` method. Tests dispatch ``_script_*`` sub-scripts via
``self.server.execute_script(...)``.

Convention (not enforced): each ``_script_*`` must drive every request
it starts to ``finished`` before returning. The base does not perform an
automatic state reset between tests, so leaked in-flight reqs will
pollute the next test in the same class.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict

from sglang.test.scripted_runtime.http_server import ScriptedHttpServer
from sglang.test.test_utils import CustomTestCase


class ScriptedTestCase(CustomTestCase):
    """Base TestCase that owns a class-scoped :class:`ScriptedHttpServer`."""

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
