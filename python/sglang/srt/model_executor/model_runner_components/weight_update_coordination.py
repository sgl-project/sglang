from __future__ import annotations

import logging
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)


def begin_uncoordinated_update() -> None:
    return None


def finish_uncoordinated_update(token: Any, *, success: bool) -> None:
    del token, success


def coordinated_weight_update(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            token = self.begin_weight_update()
        except Exception as error:
            message = f"Weight update rejected: {error}"
            logger.error(message)
            return False, message

        success = False
        try:
            result = method(self, *args, **kwargs)
            success = (
                isinstance(result, tuple) and len(result) >= 1 and result[0] is True
            )
            return result
        finally:
            self.finish_weight_update(token, success=success)

    return wrapper
