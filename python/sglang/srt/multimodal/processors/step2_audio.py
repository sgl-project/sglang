import json
import math
import re
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from transformers import BatchFeature, Qwen2TokenizerFast
from transformers.audio_utils import mel_filter_bank
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import TensorType

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.step2_audio import StepAudio2ForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


def _mel_filters(sr, n_mels: int, n_fft: int) -> torch.Tensor:
    """Load the mel filterbank matrix for projecting STFT into a Mel spectrogram,
    using mel_filter_bank"""
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    mel_filters = mel_filter_bank(
        num_frequency_bins=1 + n_fft // 2,  # n_fft//2 + 1 = 201
        num_mel_filters=n_mels,  # 80 or 128
        min_frequency=0.0,
        max_frequency=16000 // 2,  # sr/2 = 8000
        sampling_rate=16000,
        norm="slaney",
        mel_scale="slaney",
    )
    return torch.from_numpy(mel_filters.astype(np.float32)).T


stepaudio2_chat_template = """
{# ---------- tools & 首条 system ---------- #}
{% if tools %}
<|BOT|>system
{% if messages and messages[0]['role'] == 'system' -%}
{{ messages[0]['content'] }}<|EOT|>
{%- endif %}
<|BOT|>tool_json_schemas
{{ tools | tojson }}<|EOT|>
{% elif messages and messages[0]['role'] == 'system' %}
<|BOT|>system
{{ messages[0]['content'] }}<|EOT|>
{% endif %}

{# ---------- 主循环：逐条 message ---------- #}
{% for message in messages %}
  {% set role = message['role'] %}

  {# ---------- user ---------- #}
  {% if role == 'user' %}
<|BOT|>human
{%- if message['content'] is string -%}
{{ message['content'] }}<|EOT|>
{%- else -%}
{# 关键循环：触发 openai 检测 #}
{%- for content in message['content'] -%}
{%- if content['type'] == 'text' -%}{{ content['text'] }}{% endif -%}
{%- endfor -%}<|EOT|>
{%- endif %}

  {# ---------- system（非首条） ---------- #}
  {% elif role == 'system' and not loop.first %}
<|BOT|>system
{{ message['content'] }}<|EOT|>

  {# ---------- assistant ---------- #}
  {% elif role == 'assistant' %}
<|BOT|>assistant
{%- if message['tts_content'] is defined %}
{{ {
      'tts_text': message.tts_content.tts_text,
      'tts_audio': message.tts_content.tts_audio,
      'text': (message['content'] if message['content'] is string else (message['content'] | selectattr('type','equalto','text') | map(attribute='text') | list | join('')))
   } | tojson }}
{%- else -%}
  {%- if message['content'] is string -%}
{{ message['content'] }}
  {%- else -%}
{# 关键循环：触发 openai 检测 #}
{%- for content in message['content'] -%}
{%- if content['type'] == 'text' -%}{{ content['text'] }}{% endif -%}
{%- endfor -%}
  {%- endif -%}
{%- endif -%}
{%- if message['tool_calls'] is defined and message['tool_calls'] %}
{%- for tool_call in message['tool_calls'] -%}
{%- set fun = (tool_call['function'] if 'function' in tool_call else tool_call) -%}
<tool_call>function
{{ fun['name'] }}
{{ fun['arguments'] | tojson }}</tool_call>
{%- endfor -%}
{%- endif -%}<|EOT|>

  {# ---------- tool（函数输出） ---------- #}
  {% elif role == 'tool' %}
<|BOT|>
{# 从先前 assistant 的 tool_calls 里回溯出函数名 #}
{%- set ns = namespace(fname='tool') -%}
{%- if message['tool_call_id'] is defined -%}
  {%- for prev in messages -%}
    {%- if prev['role'] == 'assistant' and prev['tool_calls'] is defined and prev['tool_calls'] -%}
      {%- for tc in prev['tool_calls'] -%}
        {%- set f = (tc['function'] if 'function' in tc else tc) -%}
        {%- if tc['id'] == message['tool_call_id'] -%}{% set ns.fname = f['name'] %}{% endif -%}
      {%- endfor -%}
    {%- endif -%}
  {%- endfor -%}
{%- endif -%}
function_output
{{ ns.fname }}
{{ message['content'] }}<|EOT|>

  {# ---------- function_output（兼容你的分支名） ---------- #}
  {% elif role == 'function_output' %}
<|BOT|>input
{{ message['content'] }}<|EOT|>

  {# ---------- 兜底 ---------- #}
  {% else %}
<|BOT|>{{ role }}
{{ message['content'] }}<|EOT|>
  {% endif %}

{% endfor %}

{# ---------- 生成提示符（可选） ---------- #}
{% if add_generation_prompt %}
<|BOT|>assistant
{% endif %}
"""


class StepAudio2Tokenizer(Qwen2TokenizerFast):

    _tts_start_token: str = "<tts_start>"  # 151693
    _tts_end_token: str = "<tts_end>"  # 151694
    _first_audio_token: str = "<audio_0>"
    _tts_pad_token: str = "<tts_pad>"
    _audio_pad_token: str = "<audio_6561>"

    def max_token_id(self) -> int:
        return len(self.vocab) - 1

    def tts_start_token_id(self):
        return self.vocab.get(self._tts_start_token)

    def tts_end_token_id(self):
        return self.vocab.get(self._tts_end_token)

    def first_audio_token_id(self):
        return self.vocab.get(self._first_audio_token)

    def tts_pad_token_id(self):
        return self.vocab.get(self._tts_pad_token)

    def audio_pad_token_id(self):
        return self.vocab.get(self._audio_pad_token)

    def is_step_audio_token(self, token_id: int):
        return token_id >= self.first_audio_token_id

    def apply_chat_template_no_trans(
        self,
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
        tools: Optional[list[dict]] = None,
        documents: Optional[list[dict[str, str]]] = None,
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: Optional[dict[str]] = None,
        **kwargs,
    ) -> list[int]:
        """Convert chat messages to token IDs sequence.

        Args:
            conversation: list of chat messages
            tools: Tool configurations (optional)

        Returns:
            list[int]: Sequence of token IDs
        """
        result = []
        messages = conversation
        if continue_final_message and add_generation_prompt:
            raise ValueError(
                "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."  # noqa: E501
            )

        if tools:
            result.append("<|BOT|>system\n")
            if messages and messages[0]["role"] == "system":
                result.append(messages[0]["content"] + "<|EOT|>")
            result.append("<|BOT|>")
            result.append("tool_json_schemas\n")
            result.append(json.dumps(tools, ensure_ascii=False) + "<|EOT|>")
        elif messages and messages[0]["role"] == "system":
            result.append("<|BOT|>system\n" + messages[0]["content"] + "<|EOT|>")

        for i, message in enumerate(messages):
            if message["role"] == "user":
                result.append("<|BOT|>human\n" + message["content"] + "<|EOT|>")
            elif message["role"] == "system":
                if i != 0:
                    result.append("<|BOT|>system\n" + message["content"] + "<|EOT|>")
            elif message["role"] == "assistant":
                result.append("<|BOT|>" + message["role"] + "\n")
                if message["content"]:
                    result.append(message["content"])
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        if "function" in tool_call:
                            tool_call = tool_call["function"]
                        result.append(
                            "<tool_call>" + "function\n" + tool_call["name"] + "\n"
                        )
                        result.append(
                            json.dumps(tool_call["arguments"], ensure_ascii=False)
                        )
                        result.append("</tool_call>")
                result.append("<|EOT|>")
            elif message["role"] == "tool":
                result.append("<|BOT|>")
                function_name = "tool"
                if message.get("tool_call_id"):
                    for prev_msg in messages:
                        if prev_msg["role"] == "assistant" and prev_msg.get(
                            "tool_calls"
                        ):
                            for tool_call in prev_msg["tool_calls"]:
                                if (
                                    tool_call["id"] == message["tool_call_id"]
                                    and "function" in tool_call
                                ):  # noqa: E501
                                    function_name = tool_call["function"]["name"]
                result.append("function_output\n" + function_name + "\n")
                result.append(message["content"])
                result.append("<|EOT|>")
            elif message["role"] == "function_output":
                result.append("<|BOT|>" + "input\n" + message["content"] + "<|EOT|>")
            else:
                result.append(
                    "<|BOT|>" + message["role"] + "\n" + message["content"] + "<|EOT|>"
                )

        if add_generation_prompt:
            result.append("<|BOT|>assistant\n")

        if continue_final_message:
            final_message = message["content"]
            if isinstance(final_message, (list, tuple)):
                final_message = final_message[-1]["text"]
            final_message = final_message.strip()
            last_index = -1
            for i in range(len(result) - 1, -1, -1):
                if final_message in result[i]:
                    last_index = i
                    break  # 找到后立即退出循环
            result = result[: last_index + 1]

        return "".join(result)

    def apply_chat_template_trans_ta4(
        self,
        conversation: Union[list[dict[str, str]], list[list[dict[str, str]]]],
        tools: Optional[list[dict]] = None,
        documents: Optional[list[dict[str, str]]] = None,
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        tokenizer_kwargs: Optional[dict[str]] = None,
        tts_content: Optional[list[dict]] = None,
        **kwargs,
    ) -> list[int]:
        """Convert chat messages to token IDs sequence.

        Args:
            conversation: list of chat messages
            tools: Tool configurations (optional)

        Returns:
            list[int]: Sequence of token IDs
        """

        result = []
        messages = conversation

        if continue_final_message and add_generation_prompt:
            raise ValueError(
                "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."  # noqa: E501
            )
        if tools:
            result.append("<|BOT|>system\n")
            if messages and messages[0]["role"] == "system":
                result.append(messages[0]["content"] + "<|EOT|>")
            result.append("<|BOT|>")
            result.append("tool_json_schemas\n")
            result.append(json.dumps(tools, ensure_ascii=False) + "<|EOT|>")
        elif messages and messages[0]["role"] == "system":
            result.append("<|BOT|>system\n" + messages[0]["content"] + "<|EOT|>")

        for i, message in enumerate(messages):
            message_content = message["content"]
            if message["role"] == "user":
                if isinstance(message_content, list):
                    for content in message_content:
                        if content["type"] == "text":
                            result.append(
                                "<|BOT|>human\n" + content["text"] + "<|EOT|>"
                            )
                        elif content["type"] == "audio":
                            result.append(
                                "<|BOT|>human\n"
                                + "<audio_start><audio_patch><audio_end>"
                                + "<|EOT|>"
                            )
                else:
                    result.append("<|BOT|>human\n" + message_content + "<|EOT|>")
            elif message["role"] == "system":
                if i != 0:
                    result.append("<|BOT|>system\n" + message_content + "<|EOT|>")
            elif message["role"] == "assistant":
                result.append("<|BOT|>" + message["role"] + "\n")
                if "tts_content" not in message:
                    result.append(message_content)
                else:
                    tts_content = message["tts_content"]
                    if (
                        "tts_text" not in tts_content or "tts_audio" not in tts_content
                    ):  # noqa: E501
                        raise ValueError("tts_text/tts_audio must in tts_content keys.")
                    tts_content["text"] = message_content
                    result.append(tts_content)
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        if "function" in tool_call:
                            tool_call = tool_call["function"]
                        result.append(
                            "<tool_call>" + "function\n" + tool_call["name"] + "\n"
                        )
                        result.append(
                            json.dumps(tool_call["arguments"], ensure_ascii=False)
                        )
                        result.append("</tool_call>")
                result.append("<|EOT|>")
            elif message["role"] == "tool":
                result.append("<|BOT|>")
                function_name = "tool"
                if message.get("tool_call_id"):
                    for prev_msg in messages:
                        if prev_msg["role"] == "assistant" and prev_msg.get(
                            "tool_calls"
                        ):
                            for tool_call in prev_msg["tool_calls"]:
                                if (
                                    tool_call["id"] == message["tool_call_id"]
                                    and "function" in tool_call
                                ):  # noqa: E501
                                    function_name = tool_call["function"]["name"]
                result.append("function_output\n" + function_name + "\n")
                result.append(message_content)
                result.append("<|EOT|>")
            elif message["role"] == "function_output":
                result.append("<|BOT|>" + "input\n" + message["content"] + "<|EOT|>")
            else:
                result.append(
                    "<|BOT|>" + message["role"] + "\n" + message_content + "<|EOT|>"
                )

        if add_generation_prompt:
            result.append("<|BOT|>assistant\n")

        if continue_final_message:
            final_message = message_content
            if isinstance(final_message, (list, tuple)):
                final_message = final_message[-1]["text"]
            final_message = final_message.strip()
            last_index = -1
            for i in range(len(result) - 1, -1, -1):
                if final_message in result[i]:
                    last_index = i
                    break  # 找到后立即退出循环
            result = result[: last_index + 1]
        trans_token_ids = self.trans_text_audio_to_ta4(result)
        return trans_token_ids

    def build_tts_interleave_data(self, text_token_ids, audio_token_ids, chunk=4):

        text_token_ids_pad = text_token_ids
        chunk_nums = max(math.ceil(len(audio_token_ids) / chunk), len(text_token_ids))
        ta4_content = []
        text_token_ids_pad = text_token_ids + [self.tts_pad_token_id] * (
            chunk_nums - len(text_token_ids)
        )
        audio_token_ids_pad = audio_token_ids + [self.audio_pad_token_id] * (
            chunk_nums - len(audio_token_ids)
        )
        for idx in range(chunk_nums):
            ta4_content += text_token_ids_pad[idx : (idx + 1)]
            ta4_content += audio_token_ids_pad[idx * chunk : (idx + 1) * chunk]

        all_token_ids = (
            [self.tts_start_token_id] + ta4_content + [self.tts_end_token_id]
        )
        return all_token_ids

    def trans_text_audio_to_ta4(self, content_list: list[str]):
        result = []
        for content in content_list:
            if isinstance(content, str):
                content_tokens = self.tokenize(content)
                content_token_ids = self.convert_tokens_to_ids(content_tokens)
                result += content_token_ids

            elif isinstance(content, dict):
                tts_text_tokens = self.tokenize(content["tts_text"])
                tts_audio_tokens = self.tokenize(content["tts_audio"])
                text_tokens = self.tokenize(content["text"])

                tts_text_tokens_ids = self.convert_tokens_to_ids(tts_text_tokens)
                tts_audio_tokens_ids = self.convert_tokens_to_ids(tts_audio_tokens)
                text_tokens_ids = self.convert_tokens_to_ids(text_tokens)
                trans_token_ids = self.build_tts_interleave_data(
                    tts_text_tokens_ids, tts_audio_tokens_ids
                )
                result += trans_token_ids + text_tokens_ids

        return result


class StepAudio2Processor:

    _mel_filters_cache = {}

    def __init__(
        self,
        config,
        tokenizer,
    ) -> None:
        super().__init__()

        self.config = config
        # self.tokenizer = tokenizer
        self.tokenizer = StepAudio2Tokenizer.from_pretrained(
            "stepfun-ai/Step-Audio-2-mini",
        )
        self.tokenizer.apply_chat_template = (
            self.tokenizer.apply_chat_template_trans_ta4
        )
        self.tokenizer.chat_template = stepaudio2_chat_template
        self.audio_token = "<audio_patch>"
        self.n_mels = 128
        self.max_chunk_size = 29  # from audio encoder position embedding length equals 1500, means 29.98s audio # noqa: E501
        self.sampling_rate = 16000
        self._mel_filters = _mel_filters(
            sr=self.sampling_rate, n_mels=self.n_mels, n_fft=400
        )
        # self._mel_filters = torch.from_numpy(
        #     librosa.filters.mel(sr=self.sampling_rate,
        #                         n_fft=400,
        #                         n_mels=self.n_mels))
        self.chat_tempalte = stepaudio2_chat_template

    @property
    def audio_token_id(self) -> int:
        return self.tokenizer.get_vocab()[self.audio_token]

    def _log_mel_spectrogram(
        self,
        audio: np.ndarray,
        padding: int = 0,
    ):
        audio = F.pad(torch.from_numpy(audio.astype(np.float32)), (0, padding))
        window = torch.hann_window(400).to(audio.device)
        stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        filters = self._mel_filters
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec.t()

    def preprocess_audio(self, audio_tensor: np.ndarray) -> torch.Tensor:
        return self._log_mel_spectrogram(audio_tensor, padding=479)

    def get_num_audio_tokens(self, max_feature_len: int) -> int:
        encoder_output_dim = (
            (max_feature_len + 1) // 2 // 2
        )  # from hych: align with log-to-mel padding 479
        padding = 1
        kernel_size = 3
        stride = 2
        adapter_output_dim = (
            encoder_output_dim + 2 * padding - kernel_size
        ) // stride + 1
        return adapter_output_dim

    def _get_audio_repl(
        self,
        audio_feat_len: int,
    ) -> tuple[str, list[int]]:
        num_audio_tokens = self.get_num_audio_tokens(audio_feat_len)
        text = (
            "<audio_start>" + "<audio_patch>" * num_audio_tokens + "<audio_end>"
        )  # noqa: E501
        token_ids = (
            [self.tokenizer.convert_tokens_to_ids("<audio_start>")]
            + [self.audio_token_id] * num_audio_tokens
            + [self.tokenizer.convert_tokens_to_ids("<audio_end>")]
        )
        return text, token_ids

    def replace_placeholder(self, text: str, placeholder: str, repls: list[str]) -> str:
        parts = text.split(placeholder)
        if len(parts) - 1 != len(repls):
            raise ValueError(
                "The number of placeholders does not match the number of replacements."  # noqa: E501
            )

        result = [parts[0]]
        for i, repl in enumerate(repls):
            result.append(repl)
            result.append(parts[i + 1])

        return "".join(result)

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        audios: Union[np.ndarray, list[np.ndarray]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        if text is None:
            text = []
        if not isinstance(text, list):
            text = [text]
        if audios is None:
            audios = []
        if not isinstance(audios, list):
            audios = [audios]

        if len(audios) == 0:
            audio_inputs = {}
            text_inputs = self.tokenizer(text)
        else:
            audio_mels_lst = []
            audio_repl_str_lst = []
            audio_repl_ids_lst = []
            for audio in audios:
                audio_mels = self.preprocess_audio(audio)
                audio_mels_lst.append(audio_mels)
                audio_repl_str, audio_repl_ids = self._get_audio_repl(
                    audio_mels.shape[0]
                )
                audio_repl_str_lst.append(audio_repl_str)
                audio_repl_ids_lst.extend(audio_repl_ids)
            audio_inputs = {
                "input_features": torch.concat(audio_mels_lst),
                "audio_lens": [audio_mels.shape[0] for audio_mels in audio_mels_lst],
            }

            text = [
                self.replace_placeholder(t, self.audio_token, audio_repl_str_lst)
                for t in text
            ]
            text_inputs = self.tokenizer(text)
        return BatchFeature(
            {
                **text_inputs,
                **audio_inputs,
            },
            tensor_type=return_tensors,
        )


################################################ SGLang Processor ################################################


class StepAudio2MultimodalProcessor(BaseMultimodalProcessor):
    models = [StepAudio2ForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        tokenizer = _processor
        _processor = StepAudio2Processor(hf_config, tokenizer)
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.AUDIO_TOKEN = "<audio_start><audio_patch><audio_end>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<audio_start>(?:<audio_patch>)+<audio_end>"
        )
        # Collect special token ids
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<audio_start>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<audio_patch>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<audio_end>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        self.ATTR_NAME_TO_MODALITY.update({"feature_attention_mask": Modality.AUDIO})

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None
        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        mm_items[0].audio_feature_lens = ret["audio_lens"]

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
        }
