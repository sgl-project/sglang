# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Thread-safe encoder URL registry for dynamic encoder discovery in EPD mode.

When ``--language-only`` is set the prefill server embeds bootstrap endpoints
(``/register_encoder_url``, ``/unregister_encoder_url``, ``/list_encoder_urls``)
in its main HTTP server.  The :class:`EncoderURLRegistry` backing those
endpoints is created in :pymethod:`TokenizerManager.init_disaggregation` and
stored on the manager so that the HTTP layer can access it via ``_global_state``.
"""

import logging
import threading
from typing import List

logger = logging.getLogger(__name__)


class EncoderURLRegistry:
    """Thread-safe registry of encoder URLs.

    Created in ``TokenizerManager.init_disaggregation()`` when
    ``--language-only`` is set.  The HTTP endpoints in ``http_server.py``
    delegate to methods on this object.
    """

    def __init__(self):
        self._urls: List[str] = []
        self._lock = threading.Lock()

    def register(self, url: str) -> bool:
        """Add *url* if not already present.  Returns True if added."""
        with self._lock:
            if url not in self._urls:
                self._urls.append(url)
                logger.info(f"Registered encoder URL: {url}")
                return True
            logger.debug(f"Encoder URL already registered: {url}")
            return False

    def unregister(self, url: str) -> bool:
        """Remove *url* if present.  Returns True if removed."""
        with self._lock:
            if url in self._urls:
                self._urls.remove(url)
                logger.info(f"Unregistered encoder URL: {url}")
                return True
            return False

    def list_urls(self) -> List[str]:
        """Return a snapshot of all registered encoder URLs."""
        with self._lock:
            return list(self._urls)
