from typing import Dict, List, Optional, Union

from sglang.srt.managers.io_struct import GenerateReqInput


class EngineBase:
    # Make it in base class to ensure API is exactly the same, as well as extracting common logic
    def generate(
            self,
            # The input prompt. It can be a single prompt or a batch of prompts.
            prompt: Optional[Union[List[str], str]] = None,
            sampling_params: Optional[Union[List[Dict], Dict]] = None,
            # The token ids for text; one can either specify text or input_ids.
            input_ids: Optional[Union[List[List[int]], List[int]]] = None,
            return_logprob: Optional[Union[List[bool], bool]] = False,
            logprob_start_len: Optional[Union[List[int], int]] = None,
            top_logprobs_num: Optional[Union[List[int], int]] = None,
            lora_path: Optional[List[Optional[str]]] = None,
            stream: bool = False,
    ):
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            lora_path=lora_path,
            stream=stream,
        )
        return self._generate_impl(obj)

    def _generate_impl(self, obj: GenerateReqInput):
        raise NotImplementedError
