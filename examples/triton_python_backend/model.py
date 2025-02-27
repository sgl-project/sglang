import sys

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


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger

        # parse model configs
        self.model_config = json.loads(args["model_config"])
        self.triton_model_name = self.model_config["name"]

        # parse parameters
        parameters = self.model_config["parameters"]

        self.is_chat = parameters["is_chat"]["string_value"].lower() == "true"
        self.app_key = parameters["appkey"]["string_value"].lower()
        enable_prefix_caching = (
            parameters["enable_prefix_caching"]["string_value"].lower() == "true"
        )
        tp_size = torch.cuda.device_count()

        quant_policy = int(parameters["quant_policy"]["string_value"])

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

    def build_no_stream_json(
        self, answer, request_id, input_len, output_len, finish_reason
    ):
        # Build non-streaming JSON response
        response = {
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "index": 0,
                    "message": {"content": answer},
                }
            ],
            "model": self.triton_model_name,
            "id": request_id,
            "usage": {
                "completion_tokens": output_len,
                "prompt_tokens": input_len,
                "total_tokens": input_len + output_len,
            },
        }
        return json.dumps(response, ensure_ascii=False)

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
                    request, "messages"
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
                text = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()[0]
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
            input_token_len = 0
            finish_reason = "None"
            output_token_len = 0
            is_first = True
            is_second = False
            sampling_params = {"max_new_tokens": max_tokens, **optional_configs}
            gen = await self.engine.async_generate(
                prompt=prompt,
                sampling_params=sampling_params,
                stream=stream,
            )
            if stream:
                async for output in gen:
                    if is_first and stream:
                        is_second = True

                    if is_second and is_first is False and stream:
                        is_second = False

                    # for stream mode, send the partial response one by one
                    triton_output_tensor = pb_utils.Tensor(
                        "answer", np.asarray(output["text"], dtype=np.object_)
                    )
                    resp = pb_utils.InferenceResponse(
                        output_tensors=[triton_output_tensor]
                    )
                    if output["meta_info"]["finish_reason"] is not None:
                        response_sender.send(
                            resp, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                        )
                    else:
                        response_sender.send(resp)

                    if is_first:
                        is_first = False
            else:
                if type(gen) == list:
                    output = gen[0]
                else:
                    output = gen
                outputs.append(output["text"])
                input_token_len = output["meta_info"]["prompt_tokens"]
                output_token_len = output["meta_info"]["completion_tokens"]
                if output["meta_info"]["finish_reason"] is not None:
                    finish_reason = output["meta_info"]["finish_reason"]
                # for non-stream mode, send concatenated response at one time
                text_outputs = self.build_no_stream_json(
                    "".join(outputs),
                    request_id,
                    input_token_len,
                    output_token_len,
                    finish_reason,
                )
                triton_output_tensor = pb_utils.Tensor(
                    "answer", np.asarray(text_outputs, dtype=np.object_)
                )
                resp = pb_utils.InferenceResponse(output_tensors=[triton_output_tensor])
                response_sender.send(
                    resp, flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                )
                if is_first:
                    is_first = False
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
                "answer", np.asarray(["N/A"], dtype=np.object_)
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
