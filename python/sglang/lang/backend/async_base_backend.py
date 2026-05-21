"""Base class for async backends"""

from typing import AsyncIterator, List, Optional, Tuple, Union

from sglang.lang.chat_template import get_chat_template
from sglang.lang.choices import ChoicesDecision, ChoicesSamplingMethod
from sglang.lang.ir import SglSamplingParams


class AsyncBaseBackend:
    """Base class for async backends using asyncio primitives"""

    def __init__(self) -> None:
        self.support_concate_and_append = False
        self.chat_template = get_chat_template("default")
        self.is_chat_model = True

    def get_model_name(self):
        raise NotImplementedError()

    def get_chat_template(self):
        return self.chat_template

    async def cache_prefix(self, prefix_str: str):
        pass

    async def uncache_prefix(self, rid: str):
        pass

    async def end_request(self, rid: Union[str, List[str]]):
        pass

    async def begin_program(self, s):
        pass

    async def end_program(self, s):
        pass

    async def commit_lazy_operations(self, s):
        pass

    async def fork_program(
        self,
        src,
        dst: List,
        position_ids_offset: Optional[List[int]] = None,
    ):
        pass

    async def fill_image(self, s):
        pass

    async def generate(
        self,
        s,
        sampling_params: SglSamplingParams,
    ) -> Tuple[str, dict]:
        """Generate text asynchronously

        Args:
            s: AsyncStreamExecutor instance
            sampling_params: Sampling parameters

        Returns:
            Tuple of (generated_text, meta_info)
        """
        raise NotImplementedError()

    async def generate_stream(
        self,
        s,
        sampling_params: SglSamplingParams,
    ) -> AsyncIterator[Tuple[str, dict]]:
        """Generate text asynchronously with streaming

        Args:
            s: AsyncStreamExecutor instance
            sampling_params: Sampling parameters

        Yields:
            Tuple of (text_chunk, meta_info)
        """
        raise NotImplementedError()

    async def select(
        self,
        s,
        choices: List[str],
        temperature: float,
        choices_method: Optional[ChoicesSamplingMethod] = None,
    ) -> ChoicesDecision:
        """Select from choices asynchronously"""
        raise NotImplementedError()

    async def concatenate_and_append(self, src_rids: List[str], dst_rid: str):
        raise NotImplementedError()

    async def shutdown(self):
        pass

    async def flush_cache(self):
        pass

    async def get_server_info(self):
        pass
