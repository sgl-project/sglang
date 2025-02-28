import asyncio
import json
import os
import threading
import traceback
from json.decoder import JSONDecodeError

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from sglang.api import Engine
from sglang.srt.openai_api.protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    UsageInfo
)


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger

        # parse model configs
        self.model_config = json.loads(args["model_config"])
        self.triton_model_name = self.model_config["name"]

        # parse parameters
        parameters = self.model_config["parameters"]

        self.is_chat = parameters["is_chat"]["string_value"].lower() == "true"
        enable_prefix_caching = (
            parameters["enable_prefix_caching"]["string_value"].lower() == "true"
        )
        tp_size = torch.cuda.device_count()

        quant_policy_value = parameters["quant_policy"]["string_value"].lower()
        quant_policy = None if quant_policy_value == "none" else quant_policy_value

        # load engine
        model_path = os.path.join(
            args["model_repository"], args["model_version"], "weights"
        )
        self.engine = Engine(
            model_path=model_path,
            tp_size=tp_size,
            disable_radix_cache=not enable_prefix_caching,
            quantization=quant_policy,
        )
        self.tokenizer = self.engine.tokenizer_manager.tokenizer
        self.request_id = 0
        # warm up
        self.engine.generate(
            prompt="warm up",
        )
        self.event_loop = asyncio.get_event_loop()
        self.engine_thread = threading.Thread(target=self._engine_loop)
        self.shutdown_event = asyncio.Event()
        self.engine_thread.start()

        self.logger.log_info("SGLang backend started")

    def _engine_loop(self):
        self.logger.log_info("Engine loop started")
        self.event_loop.run_until_complete(self._await_shutdown())
        self.event_loop.close()
        self.logger.log_info("Engine loop closed")

    async def _await_shutdown(self):
        # await the shutdown signal
        await self.shutdown_event.wait()
        self.logger.log_info("Get shutdown signal")

        # cancel unfinished tasks
        for task in asyncio.all_tasks(loop=self.event_loop):
            if task is not asyncio.current_task(loop=self.event_loop):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    self.logger.log_info("Unfinished task is cancelled")

    def _get_optional_configs(self, request):
        optional_configs = {}
        config_names = [
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "n",
            "best_of",
            "presence_penalty",
            "frequency_penalty",
            "ignore_eos",
        ]
        for config_name in config_names:
            input_tensor = pb_utils.get_input_tensor_by_name(request, config_name)
            if input_tensor is not None:
                optional_configs[config_name] = input_tensor.as_numpy().item()

        return optional_configs

    async def _process_request(self, request_id, request):
        response_sender = request.get_response_sender()
        try:
            # parse request
            # Construct prompt
            if self.is_chat:
                messages = pb_utils.get_input_tensor_by_name(
                    request, "prompt"
                ).as_numpy()[0]
                if isinstance(messages, bytes):
                    messages = messages.decode("utf-8")
                prompt = json.loads(messages)
                prompt = self.tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                text = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0]
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
                prompt = text
            max_tokens = (
                pb_utils.get_input_tensor_by_name(request, "max_tokens")
                .as_numpy()
                .item()
            )
            stream = (
                pb_utils.get_input_tensor_by_name(request, "stream").as_numpy().item()
            )
            optional_configs = self._get_optional_configs(request)

            outputs = []
            sampling_params = {"max_new_tokens": max_tokens, **optional_configs}
            gen = await self.engine.async_generate(
                prompt=prompt,
                sampling_params=sampling_params,
                stream=stream,
            )
            if stream:
                is_first = True
                stream_buffer = ""
                async for content in gen:
                    index = content.get("index", 0)
                    finish_reason = content["meta_info"]["finish_reason"]
                    if is_first:
                        # First chunk with role
                        is_first = False
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(role="assistant", content=""),
                            finish_reason=(
                                finish_reason["type"] if finish_reason else ""
                            ),
                            matched_stop=(
                                finish_reason["matched"]
                                if finish_reason and "matched" in finish_reason
                                else None
                            ),
                        )
                        response = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            choices=[choice_data],
                            model=self.triton_model_name,
                        ).model_dump_json()
                    else:
                        text = content["text"]
                        delta = text[len(stream_buffer):]
                        stream_buffer = stream_buffer + delta
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=index,
                            delta=DeltaMessage(role="assistant", content=delta),
                            finish_reason=(
                                finish_reason["type"] if finish_reason else ""
                            ),
                            matched_stop=(
                                finish_reason["matched"]
                                if finish_reason and "matched" in finish_reason
                                else None
                            ),
                        )
                        response = ChatCompletionStreamResponse(
                            id=content["meta_info"]["id"],
                            choices=[choice_data],
                            model=self.triton_model_name,
                        ).model_dump_json()
                    # for stream mode, send the partial response one by one
                    triton_output_tensor = pb_utils.Tensor(
                        "response", np.asarray(response, dtype=np.object_)
                    )
                    resp = pb_utils.InferenceResponse(
                        output_tensors=[triton_output_tensor]
                    )
                    if finish_reason is not None:
                        response_sender.send(
                            resp, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                        )
                    else:
                        response_sender.send(resp)
            else:
                content = gen
                outputs.append(content["text"])
                prompt_tokens = content["meta_info"]["prompt_tokens"]
                completion_tokens = content["meta_info"]["completion_tokens"]
                finish_reason = content["meta_info"]["finish_reason"]
                # for non-stream mode, send concatenated response at one time
                choice = ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="".join(outputs),
                    ),
                    finish_reason=finish_reason["type"] if finish_reason else "",
                    matched_stop=(
                        finish_reason["matched"]
                        if finish_reason and "matched" in finish_reason
                        else None
                    ),
                )
                response = ChatCompletionResponse(
                    id=content["meta_info"]["id"],
                    model=self.triton_model_name,
                    choices=[choice],
                    usage=UsageInfo(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                ).model_dump_json()
                triton_output_tensor = pb_utils.Tensor(
                    "response", np.asarray(response, dtype=np.object_)
                )
                resp = pb_utils.InferenceResponse(output_tensors=[triton_output_tensor])
                response_sender.send(
                    resp, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
        except Exception as e:
            self.logger.log_info(
                f"Error when processing request: {traceback.format_exc()}"
            )
            if isinstance(e, JSONDecodeError):
                error = pb_utils.TritonError(
                    "Error when parsing prompt, please make sure correct JSON format"
                )
            else:
                error = pb_utils.TritonError(
                    f"Error when processing request: {traceback.format_exc()}"
                )
            triton_output_tensor = pb_utils.Tensor(
                "response", np.asarray(["N/A"], dtype=np.object_)
            )
            resp = pb_utils.InferenceResponse(
                output_tensors=[triton_output_tensor], error=error
            )
            response_sender.send(
                resp, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
            )
            raise e

    def execute(self, requests):
        for request in requests:
            future = asyncio.run_coroutine_threadsafe(
                self._process_request(self.request_id, request), self.event_loop
            )
            try:
                future.result(timeout=30)
            except Exception as e:
                print("Task timeout")
                self.logger.log_error(f"Request {self.request_id} failed: {str(e)}")
                future.cancel()
            finally:
                self.request_id += 1
        return None

    def finalize(self):
        self.logger.log_info("Finalizing SGLang backend")
        self.event_loop.call_soon_threadsafe(self.shutdown_event.set)
        if self.engine_thread is not None:
            self.engine_thread.join()
            self.engine_thread = None
