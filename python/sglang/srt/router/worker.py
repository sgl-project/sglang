from dataclasses import dataclass

# Store worker config and the client to talk to the worker
@dataclass
class Worker:
    # static server configs
    max_running_request: int
    # the server url for the DP worker
    server_url: str

    # the client to interact the the DP server
    client: httpx.AsyncClient
