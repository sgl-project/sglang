from dataclasses import dataclass
from typing import Optional

import httpx


# Store worker config and the client to talk to the worker
@dataclass
class Worker:
    # static server configs
    max_running_request: Optional[int] = None
    # the server url for the DP worker
    server_url: Optional[str] = None
    # the client to interact the the DP server
    client: Optional[httpx.AsyncClient] = None
