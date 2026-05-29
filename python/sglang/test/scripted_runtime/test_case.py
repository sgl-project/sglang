"""unittest base class wrapping :class:`ScriptedRuntimeSession`.

Subclass and set ``ENGINE_KWARGS`` as a class attribute. The base owns
the Engine lifecycle — one engine per concrete class, shared across
every ``test_*`` method. Tests dispatch ``_script_*`` sub-scripts via
``self.runtime.run(...)``.

Convention (no enforcement until the wishlist abort / list_active_reqs
APIs land): each ``_script_*`` must drive every request it starts to
``finished`` (or, eventually, abort) before returning. The base does
not perform an automatic state reset between tests, so leaked
in-flight reqs will pollute the next test in the same class.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict

from sglang.test.scripted_runtime.session import ScriptedRuntimeSession
from sglang.test.test_utils import CustomTestCase


class ScriptedRuntimeTestCase(CustomTestCase):
    """Base TestCase that owns a class-scoped :class:`ScriptedRuntimeSession`."""

    ENGINE_KWARGS: ClassVar[Dict[str, Any]] = {}

    runtime: ClassVar[ScriptedRuntimeSession]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        if not cls.ENGINE_KWARGS:
            raise AssertionError(
                f"{cls.__name__} must set ENGINE_KWARGS to a non-empty dict"
            )
        cls.runtime = ScriptedRuntimeSession.start(**cls.ENGINE_KWARGS)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.runtime.shutdown()
        finally:
            super().tearDownClass()
