import random
from enum import Enum, auto
from typing import Dict, List

import httpx
from worker import Worker


class BaseRouter:
    def __init__(self, server_urls: List[str]):
        self.worker_list: List[Worker] = []
        self.server_url_to_worker: Dict[str, Worker] = {}
        self._init_worker_list(server_urls)

    ####################
    # Public Method
    ####################
    def get_worker(self, server_url: str):
        if server_url in self.server_url_to_worker:
            return self.server_url_to_worker[server_url]
        raise ValueError(f"Worker with url {server_url} not found")

    # scale down the workers / fault happens on the worker
    def remove_worker(self, server_url: str):
        worker = self.get_worker(server_url)
        self.worker_list.remove(worker)
        del self.server_url_to_worker[server_url]

    # scale up the workers / init
    def add_worker(self, server_url: str):
        if server_url in self.server_url_to_worker:
            raise ValueError(f"Worker with url {server_url} already exists")
        worker = Worker()
        worker.server_url = server_url
        # disable timeout == setting timeout as inf
        worker.client = httpx.AsyncClient(base_url=server_url, timeout=None)
        # TODO: ensure the worker is healthy before adding to the list, maybe by sending a health check request
        self.worker_list.append(worker)
        self.server_url_to_worker[server_url] = worker

    def calc_priority(self) -> Worker:
        raise NotImplementedError

    ####################
    # Private Method
    ####################
    def _init_worker_list(self, server_urls):
        for server_url in server_urls:
            self.add_worker(server_url)


class RandomRouter(BaseRouter):
    def calc_priority(self) -> Worker:
        idx = random.choice(self.worker_list)
        return self.worker_list[idx]


class RoundRobinRouter(BaseRouter):
    def __init__(self, server_urls: List[str]):
        super().__init__(server_urls)
        self.idx = 0

    def calc_priority(self) -> Worker:
        worker = self.worker_list[self.idx]
        self.idx = (self.idx + 1) % len(self.worker_list)
        return worker


class RoutingPolicy(Enum):
    ROUND_ROBIN = auto()
    RANDOM = auto()

    @classmethod
    def from_str(cls, policy: str):
        policy = policy.upper()
        try:
            return cls[policy]
        except KeyError as exc:
            valid_options = ", ".join(member.name for member in cls)
            raise ValueError(
                f"Invalid routing policy: {policy}. The valid options are {valid_options}"
            ) from exc


def get_router_class(policy_name: str):
    policy = RoutingPolicy.from_str(policy_name)

    if policy == RoutingPolicy.ROUND_ROBIN:
        return RoundRobinRouter
    elif policy == RoutingPolicy.RANDOM:
        return RandomRouter
