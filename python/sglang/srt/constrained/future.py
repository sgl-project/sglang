"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Base tool cache for constrained decoding tools."""

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


class FutureObject:
    def __init__(self, f: Callable[[], Any]):
        self._result = None
        self._exception = None
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(f,))
        self._thread.daemon = True
        self._thread.start()

    def _run(self, f: Callable[[], Any]):
        try:
            self._result = f()
        except Exception as e:
            logger.exception(f"Error in getting a FutureObject: {e}")
        finally:
            self._done.set()

    def get(self):
        self._done.wait()
        assert self._result is not None
        return self._result

    def is_complete(self) -> bool:
        return self._done.is_set()
