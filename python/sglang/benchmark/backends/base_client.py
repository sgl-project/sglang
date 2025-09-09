from abc import ABC, abstractmethod
from argparse import Namespace
from typing import Optional

from tqdm.asyncio import tqdm

from sglang.benchmark.datasets.common import RequestFuncInput, RequestFuncOutput


class BaseBackendClient(ABC):
    def __init__(self, args: Namespace):
        self.args = args

    @abstractmethod
    async def make_request(
        self,
        request_func_input: RequestFuncInput,
        pbar: Optional[tqdm] = None,
    ) -> RequestFuncOutput:
        pass
