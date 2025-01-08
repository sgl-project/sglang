import os
from typing import Dict, List, Union, Any

from sglang.srt.hf_transformers_utils import get_processor, get_tokenizer
from sglang.srt.managers.image_processor import (
    get_dummy_image_processor,
    get_image_processor,
)
from sglang.srt.managers.io_struct import (
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
)
from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    SessionParams,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs


class GenerationManager:
    def __init__(
        self,
        server_args: ServerArgs,
    ):
        self.server_args = server_args

        # Create image processor placeholder
        self.image_processor = get_dummy_image_processor()

        # Create tokenizer
        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )
                self.tokenizer = self.processor.tokenizer
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

                # We want to parallelize the image pre-processing so we create an executor for it
                self.image_processor = get_image_processor(
                    self.model_config.hf_config, server_args, self.processor
                )
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                )

    async def tokenize_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ) -> Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput]:
        """Tokenize one request."""
        # Tokenize
        input_embeds = None
        input_text = obj.text
        if obj.input_embeds is not None:
            if not self.server_args.disable_radix_cache:
                raise ValueError(
                    "input_embeds is provided while disable_radix_cache is False. "
                    "Please add `--disable-radix-cache` when you launch the server "
                    "if you want to use input_embeds as inputs."
                )
            input_embeds = obj.input_embeds
            input_ids = obj.input_ids
        elif obj.input_ids is None:
            input_ids = self.tokenizer.encode(input_text)
        else:
            input_ids = obj.input_ids

        if self.is_generation:
            # TODO: also support getting embeddings for multimodal models
            image_inputs: Dict = await self.image_processor.process_images_async(
                obj.image_data, input_text or input_ids, obj
            )
            if image_inputs and "input_ids" in image_inputs:
                input_ids = image_inputs["input_ids"]
            return_logprob = obj.return_logprob
            logprob_start_len = obj.logprob_start_len
            top_logprobs_num = obj.top_logprobs_num
            session_params = (
                SessionParams(**obj.session_params) if obj.session_params else None
            )

        if obj.input_ids is not None and len(input_ids) >= self.context_len:
            raise ValueError(
                f"The input ({len(input_ids)} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

        # Parse sampling parameters
        sampling_params = SamplingParams(**obj.sampling_params)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify()

        # Build return object
        if isinstance(obj, GenerateReqInput):
            return TokenizedGenerateReqInput(
                obj.rid,
                input_text,
                input_ids,
                image_inputs,
                sampling_params,
                return_logprob,
                logprob_start_len,
                top_logprobs_num,
                obj.stream,
                lora_path=obj.lora_path,
                input_embeds=input_embeds,
                session_params=session_params,
            )
        elif isinstance(obj, EmbeddingReqInput):
            return TokenizedEmbeddingReqInput(
                obj.rid,
                input_text,
                input_ids,
                sampling_params,
            )
        else:
            raise NotImplementedError

    def postprocess_batch_output_item(
        self,
        recv_obj: Union[BatchStrOut, BatchEmbeddingOut, BatchTokenIDOut],
        index: int,
        rid: str,
        req_obj: Union[GenerateReqInput, EmbeddingReqInput],
    ) -> Dict[str, Any]:
        meta_info = self._compute_meta_info(index, recv_obj, req_obj, rid)

        if isinstance(recv_obj, BatchStrOut):
            out_dict = {
                "text": recv_obj.output_strs[index],
                "meta_info": meta_info,
            }
            if self.server_args.return_token_ids:
                out_dict.update(
                    {
                        "input_ids": recv_obj.origin_input_ids[index],
                        "output_ids": recv_obj.output_ids[index],
                    }
                )
            return out_dict

        elif isinstance(recv_obj, BatchTokenIDOut):
            return {
                "token_ids": recv_obj.output_ids[index],
                "meta_info": meta_info,
            }

        elif isinstance(recv_obj, BatchEmbeddingOut):
            return {
                "embedding": recv_obj.embeddings[index],
                "meta_info": meta_info,
            }

        else:
            raise NotImplementedError

    def _compute_meta_info(self, index, recv_obj, req_obj, rid):
        meta_info = {
            "id": rid,
            "finish_reason": recv_obj.finished_reasons[index],
            "prompt_tokens": recv_obj.prompt_tokens[index],
        }
        if getattr(req_obj, "return_logprob", False):
            self._convert_logprob_style(
                meta_info,
                req_obj.top_logprobs_num,
                req_obj.return_text_in_logprobs,
                recv_obj,
                index,
            )
        if not isinstance(recv_obj, BatchEmbeddingOut):
            meta_info.update(
                {
                    "completion_tokens": recv_obj.completion_tokens[index],
                    "cached_tokens": recv_obj.cached_tokens[index],
                }
            )
        return meta_info

    def _convert_logprob_style(
        self,
        meta_info: dict,
        top_logprobs_num: int,
        return_text_in_logprobs: bool,
        recv_obj: BatchStrOut,
        recv_obj_index: int,
    ):
        meta_info["input_token_logprobs"] = self._detokenize_logprob_tokens(
            recv_obj.input_token_logprobs_val[recv_obj_index],
            recv_obj.input_token_logprobs_idx[recv_obj_index],
            return_text_in_logprobs,
        )
        meta_info["output_token_logprobs"] = self._detokenize_logprob_tokens(
            recv_obj.output_token_logprobs_val[recv_obj_index],
            recv_obj.output_token_logprobs_idx[recv_obj_index],
            return_text_in_logprobs,
        )
        meta_info["normalized_prompt_logprob"] = recv_obj.normalized_prompt_logprob[
            recv_obj_index
        ]

        if top_logprobs_num > 0:
            meta_info["input_top_logprobs"] = self._detokenize_top_logprobs_tokens(
                recv_obj.input_top_logprobs_val[recv_obj_index],
                recv_obj.input_top_logprobs_idx[recv_obj_index],
                return_text_in_logprobs,
            )
            meta_info["output_top_logprobs"] = self._detokenize_top_logprobs_tokens(
                recv_obj.output_top_logprobs_val[recv_obj_index],
                recv_obj.output_top_logprobs_idx[recv_obj_index],
                return_text_in_logprobs,
            )

    def _detokenize_logprob_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        if not decode_to_text:
            return [
                (logprob, token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
            ]
        else:
            assert self.tokenizer is not None
            token_texts = self.tokenizer.batch_decode(token_logprobs_idx)
            return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))

    def _detokenize_top_logprobs_tokens(
        self,
        token_logprobs_val: List[float],
        token_logprobs_idx: List[int],
        decode_to_text: bool,
    ):
        # TODO: The current implementation only batches the detokenization for top-k tokens per single position.
        # We should batch all top-k tokens in all positions.
        ret = []
        for i in range(len(token_logprobs_val)):
            if token_logprobs_val[i]:
                ret.append(
                    self._detokenize_logprob_tokens(
                        token_logprobs_val[i], token_logprobs_idx[i], decode_to_text
                    )
                )
            else:
                ret.append(None)
        return ret
