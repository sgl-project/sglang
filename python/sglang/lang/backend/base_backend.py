from typing import List, Optional, Union

from sglang.lang.chat_template import get_chat_template
from sglang.lang.choices import ChoicesDecision, ChoicesSamplingMethod
from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglSamplingParams


class BaseBackend:
    def __init__(self) -> None:
        self.support_concate_and_append = False
        self.chat_template = get_chat_template("default")

    def get_model_name(self):
        raise NotImplementedError()

    def get_chat_template(self):
        return self.chat_template

    def cache_prefix(self, prefix_str: str):
        pass

    def uncache_prefix(self, rid: str):
        pass

    def end_request(self, rid: Union[str, List[str]]):
        pass

    def begin_program(self, s: StreamExecutor):
        pass

    def end_program(self, s: Union[StreamExecutor, List[StreamExecutor]]):
        pass

    def commit_lazy_operations(self, s: StreamExecutor):
        pass

    def fork_program(
        self,
        src: StreamExecutor,
        dst: List[StreamExecutor],
        position_ids_offset: Optional[List[int]] = None,
    ):
        pass

    def fill_image(self, s: StreamExecutor):
        pass

    def generate(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        raise NotImplementedError()

    def generate_stream(
        self,
        s: StreamExecutor,
        sampling_params: SglSamplingParams,
    ):
        raise NotImplementedError()

    def select(
        self,
        s: StreamExecutor,
        choices: List[str],
        temperature: float,
        choices_method: Optional[ChoicesSamplingMethod] = None,
    ) -> ChoicesDecision:
        raise NotImplementedError()

    def concatenate_and_append(self, src_rids: List[str], dst_rid: str):
        raise NotImplementedError()

    def shutdown(self):
        pass

    def flush_cache(self):
        pass

    def get_server_info(self):
        pass
