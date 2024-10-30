from dataclasses import dataclass
from typing import Optional

import httpx


# Store worker config and the client to talk to the worker
@dataclass
class Worker:
    # the server url for the DP worker
    server_url: Optional[str] = None
    # the client to interact the the DP server
    client: Optional[httpx.AsyncClient] = None


@dataclass
class WorkerUpdateReq:
    server_url: str
