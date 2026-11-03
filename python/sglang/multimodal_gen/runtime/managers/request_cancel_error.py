# SPDX-License-Identifier: Apache-2.0
"""The error of a cancelled request.

This module imports nothing, so the logging and profiling utilities can import
it at module level.
"""


class RequestCancelledError(RuntimeError):
    def __init__(self, request_ids: list[str]):
        self.request_ids = request_ids
        super().__init__(f"Request {', '.join(request_ids)} was cancelled")
