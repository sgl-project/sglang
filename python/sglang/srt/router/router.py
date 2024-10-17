from fastapi import FastAPI
import uvicorn
import argparse
from fastapi.responses import Response
from typing import List, Dict
import httpx
import asyncio


class BaseRouter:
    def __init__(self, server_urls: List[str]):
        self.worker_list: List[Worker] = None
        
        self._init_worker_list(server_urls)
        
        self.server_url_to_worker: Dict[str, Worker] = {worker.url: worker for worker in self.worker_list}

    ####################
    # Public Method
    ####################
    def get_worker(self, server_url: str):
        for worker in self.worker_list:
            if worker.url == server_url:
                return worker
        raise ValueError(f"Worker with url {server_url} not found")

    # scale down the workers / fault happens on the worker
    def remove_worker(self, server_url: str):
        worker = self.get_worker(server_url)
        worker.client.close()
        self.worker_list.remove(worker)
    
    # scale up the workers / init
    def add_worker(self, server_url: str):
        worker = Worker()
        worker.server_url = server_url
        worker.client = httpx.AsyncClient(base_url=server_url)
        #TODO: make a call to fill in more worker info, but need to ensure performance
        self.worker_list.append(worker)

    def calc_priority(self) -> Worker:
        raise NotImplementedError

    ####################
    # Private Method
    ####################
    def _init_worker_list(self):
        for server_url in self.server_urls:
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


# Make a enum for routing policy
class RoutingPolicy(Enum):
    ROUND_ROBIN = auto()
    RANDOM = auto()

    @classmethod
    def from_str(cls, policy: str):
        policy = policy.upper()
        try:
            return cls[policy]
        except KeyError as exc:
            raise ValueError(f"Invalid routing policy: {policy}") from exc

def get_router_class(policy_name: str):
    policy = RoutingPolicy.from_str(policy_name)

    if policy == RoutingPolicy.ROUND_ROBIN:
        return RoundRobinRouter
    elif policy == RoutingPolicy.RANDOM:
        return RandomRouter
 