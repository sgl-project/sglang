from typing import Optional

from tqdm.asyncio import tqdm

from sglang.benchmark.backends.base_client import (
    BaseBackendClient,
    RequestFuncInput,
    RequestFuncOutput,
)


class GServerBackendClient(BaseBackendClient):
    async def request(
        self,
        request_func_input: RequestFuncInput,
        pbar: Optional[tqdm] = None,
    ) -> RequestFuncOutput:
        raise NotImplementedError()
