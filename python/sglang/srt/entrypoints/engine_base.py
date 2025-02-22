from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch

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
        # The image input. It can be a file name, a url, or base64 encoded string.
        # See also python/sglang/srt/utils.py:load_image.
        image_data: Optional[Union[List[str], str]] = None,
        return_logprob: Optional[Union[List[bool], bool]] = False,
        logprob_start_len: Optional[Union[List[int], int]] = None,
        top_logprobs_num: Optional[Union[List[int], int]] = None,
        lora_path: Optional[List[Optional[str]]] = None,
        custom_logit_processor: Optional[Union[List[str], str]] = None,
        stream: bool = False,
    ) -> Union[Dict, Iterator[Dict]]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        modalities_list = []
        if image_data is not None:
            modalities_list.append("image")

        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            lora_path=lora_path,
            modalities=modalities_list,
            custom_logit_processor=custom_logit_processor,
            stream=stream,
        )
        return self._generate_impl(obj)

    def update_weights_from_tensor(self, named_tensors: List[Tuple[str, torch.Tensor]]):
        """Update weights from tensor."""
        raise NotImplementedError

    def _generate_impl(self, obj: GenerateReqInput):
        raise NotImplementedError
