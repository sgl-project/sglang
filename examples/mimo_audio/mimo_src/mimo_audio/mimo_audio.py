# Copyright 2025 Xiaomi Corporation.
import os
import random
import re
import time
from typing import Union

import soundfile as sf
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from ..mimo_audio_tokenizer import MiMoAudioTokenizer
from .modeling_mimo_audio import (
    MiMoAudioArguments,
    MiMoAudioConfig,
    MiMoSampler,
    MiMoStopper,
)
from .process_speechdata import InputSegment, StreamingInputSegment
from .templates import (
    asr_en_templates,
    asr_zh_templates,
    tts_en_templates,
    tts_zh_templates,
)


def detect_language(text):
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    else:
        return "en"


class MimoAudio:

    def __init__(
        self,
        model_path: str,
        mimo_audio_tokenizer_path: str,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.path = model_path
        self.mimo_audio_tokenizer_path = mimo_audio_tokenizer_path

        self.tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
            self.path
        )
        self.padding_idx = int(self.tokenizer.pad_token_id)

        special_tokens = [
            "<|sosp|>",
            "<|eosp|>",
            "<|empty|>",
            "<|Human|>",
            "<|SpeechLM|>",
            "<|sostm|>",
            "<|eostm|>",
            "<|eot|>",
        ]
        for token in special_tokens:
            if token not in self.tokenizer.get_vocab():
                print(f"Add special tokens {token} to tokenizer.vocab")
                self.tokenizer.add_tokens([token], special_tokens=True)

        self.sosp_idx = self.tokenizer.convert_tokens_to_ids("<|sosp|>")
        self.eosp_idx = self.tokenizer.convert_tokens_to_ids("<|eosp|>")
        self.empty_token = self.tokenizer.convert_tokens_to_ids("<|empty|>")
        self.sostm_idx = self.tokenizer.convert_tokens_to_ids("<|sostm|>")
        self.eostm_idx = self.tokenizer.convert_tokens_to_ids("<|eostm|>")
        self.eot_idx = self.tokenizer.convert_tokens_to_ids("<|eot|>")
        self.im_start_idx = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_idx = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        model_args = MiMoAudioArguments(
            model_name_or_path=self.path,
            sosp_idx=self.sosp_idx,
            eosp_idx=self.eosp_idx,
            empty_idx=self.empty_token,
            sostm_idx=self.sostm_idx,
            eostm_idx=self.eostm_idx,
            eot_idx=self.eot_idx,
        )

        start_loading_time = time.monotonic()
        # self.model = MiMoAudioForCausalLM.from_pretrained(
        #     self.path,
        #     args=model_args,
        #     torch_dtype=torch.bfloat16,
        #     device_map={"": self.device},
        # )
        config = AutoConfig.from_pretrained(self.path)
        config = (
            MiMoAudioConfig(**vars(config))
            if isinstance(config, Qwen2Config)
            else config
        )
        self.config = config

        self.group_size = self.config.group_size
        self.audio_channels = self.config.audio_channels
        self.delay_pattern = self.config.delay_pattern
        self.vocab_size = self.config.vocab_size

        self.speech_zeroemb_idx = self.config.parsed_speech_empty_ids()

        # self.model.eval()
        # print(
        #     f"Model loaded in {time.monotonic() - start_loading_time:.2f} seconds, device: {self.device}"
        # )

        self.generate_kwargs = {
            "max_length": 8192,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        self.default_global_sampler = MiMoSampler(
            do_sample=True, temperature=0.6, top_k=50, top_p=0.95
        )
        self.default_local_sampler = MiMoSampler(
            do_sample=True, temperature=0.9, top_k=50, top_p=0.95
        )

        self.task_sampler_configs = {
            "asr": {
                "global": MiMoSampler(do_sample=False, temperature=1.0, top_p=1.0),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95),
            },
            "tts": {
                "global": MiMoSampler(do_sample=True, temperature=0.6, top_p=1.0),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95),
            },
            "spoken_dialogue": {
                "global": MiMoSampler(do_sample=True, temperature=0.6, top_p=0.95),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95),
            },
            "audio_understanding": {
                "global": MiMoSampler(do_sample=True, temperature=0.3, top_p=0.95),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95),
            },
            "text_chat": {
                "global": MiMoSampler(do_sample=True, temperature=0.4, top_p=0.95),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95),
            },
            "in_context_learning_s2s": {
                "global": MiMoSampler(do_sample=False, temperature=1.0, top_p=1.0),
                "local": MiMoSampler(do_sample=True, temperature=0.9, top_p=0.95),
            },
        }

        start_loading_mimo_audio_tokenizer_time = time.monotonic()
        self.mimo_audio_tokenizer = MiMoAudioTokenizer.from_pretrained(
            self.mimo_audio_tokenizer_path
        )

        self.mimo_audio_tokenizer.eval().bfloat16().to(self.device)
        print(
            f"MiMo-Audio Tokenizer loaded in {time.monotonic() - start_loading_mimo_audio_tokenizer_time:.2f} seconds, device: {self.device}"
        )

        # Initialize mel spectrogram transform for consistent processing
        self.mel_transform = MelSpectrogram(
            sample_rate=self.mimo_audio_tokenizer.config.sampling_rate,
            n_fft=self.mimo_audio_tokenizer.config.nfft,
            hop_length=self.mimo_audio_tokenizer.config.hop_length,
            win_length=self.mimo_audio_tokenizer.config.window_size,
            f_min=self.mimo_audio_tokenizer.config.fmin,
            f_max=self.mimo_audio_tokenizer.config.fmax,
            n_mels=self.mimo_audio_tokenizer.config.n_mels,
            power=1.0,
            center=True,
        ).to(self.device)

        self.history = None

    def get_task_sampler(self, task_name):
        if task_name not in self.task_sampler_configs:
            return {
                "global": self.default_global_sampler,
                "local": self.default_local_sampler,
            }
        return self.task_sampler_configs[task_name]

    def save_wav(self, path, wav):
        sf.write(
            path,
            wav.reshape(-1).detach().cpu().numpy(),
            24000,
        )

    def wav2mel(self, wav):
        spec = self.mel_transform(wav[None, :])
        return torch.log(torch.clip(spec, min=1e-7)).squeeze()

    def resample_audio_if_needed(self, wav_tensor: torch.Tensor, original_sr: int):
        target_sr = self.mimo_audio_tokenizer.config.sampling_rate
        if original_sr != target_sr:
            wav_tensor = torchaudio.functional.resample(
                wav_tensor, original_sr, target_sr
            )
        return wav_tensor

    def group_by_length(
        self, features: torch.Tensor, lengths: torch.Tensor, max_length: int
    ):
        if features.size(0) != lengths.sum().item():
            raise ValueError(
                f"Feature size mismatch: {features.size(0)} vs {lengths.sum().item()}"
            )

        split_points = []
        current_sum = 0

        for i, seq_len in enumerate(lengths):
            if current_sum + seq_len > max_length and current_sum > 0:
                split_points.append(i)
                current_sum = seq_len.item()
            else:
                current_sum += seq_len.item()

        # Convert split points to group sizes
        group_sizes = []
        prev = 0
        for point in split_points:
            group_sizes.append(point - prev)
            prev = point
        if prev < len(lengths):
            group_sizes.append(len(lengths) - prev)

        len_groups = torch.split(lengths, group_sizes)
        feature_sizes = [group.sum().item() for group in len_groups]
        feature_groups = torch.split(features, feature_sizes)

        return feature_groups, len_groups

    def encode_batch(
        self,
        input_features: torch.Tensor,
        input_lens: torch.Tensor,
        max_length: int = 256000,
    ):
        feature_groups, len_groups = self.group_by_length(
            input_features, input_lens, max_length
        )

        encoded_parts = []
        for features, lengths in zip(feature_groups, len_groups):
            with torch.no_grad():
                codes, _ = self.mimo_audio_tokenizer.encoder.encode(
                    input_features=features.to(self.device),
                    input_lens=lengths.to(self.device),
                    return_codes_only=True,
                )
                encoded_parts.append(codes)

        return torch.cat(encoded_parts, dim=-1)

    def preprocess_input(
        self,
        input: Union[None, str, torch.Tensor] = None,
    ):
        if isinstance(input, torch.Tensor) or (
            isinstance(input, str) and os.path.isfile(input)
        ):
            if isinstance(input, torch.Tensor):
                wav = input
            else:
                wav, sr = torchaudio.load(input)
                if wav.ndim == 2:
                    wav = wav.mean(dim=0)
                wav = self.resample_audio_if_needed(wav, sr)
            wav = wav.to(self.device)

            mel = self.wav2mel(wav).transpose(0, 1)  # (seq_len, n_mels)

            input_len = mel.size(0)
            segment_size = 6000
            input_len_seg = [segment_size] * (input_len // segment_size)
            if input_len % segment_size > 0:
                input_len_seg.append(input_len % segment_size)

            codes_packed = self.encode_batch(
                input_features=mel,
                input_lens=torch.tensor(input_len_seg),
            )

            codes = codes_packed.transpose(0, 1).detach().cpu()
            audio_codes = codes[:, : self.audio_channels]

            # Pad the sequence to be a multiple of group_size by repeating the last frame
            num_timesteps = audio_codes.shape[0]
            if num_timesteps % self.group_size != 0:
                padding_needed = self.group_size - (num_timesteps % self.group_size)
                last_tokens = audio_codes[-1:, :]  # Keep dim for repeat
                padding_tokens = last_tokens.repeat(padding_needed, 1)
                audio_codes = torch.cat([audio_codes, padding_tokens], dim=0)

            audio_tokenized = audio_codes.reshape(-1)

            return audio_tokenized
        else:
            text = input
            if (
                text.isupper() or text.islower()
            ):  # If the text only contains upper-case or lower-case letters, capitalize it.
                text = text.capitalize()
            return text

    def get_input_ids(self, prompt):
        input_ids = [
            seg.to_input_id(
                self.tokenizer,
                self.group_size,
                self.audio_channels,
            )
            for seg in prompt
        ]
        input_ids = torch.cat(input_ids, dim=1)
        return input_ids.to(self.device)

    def get_asr_sft_prompt(
        self,
        input: Union[None, str] = None,
    ):
        audio_tokenized = self.preprocess_input(input)

        template = random.choice(asr_zh_templates + asr_en_templates)

        lm_prompt = [
            InputSegment(
                text=f"<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                audio=audio_tokenized,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=template,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_end|>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text="<think>\n\n</think>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_tts_sft_prompt(
        self,
        input: Union[None, str] = None,
        instruct=None,
        read_text_only=True,
        prompt_speech=None,
    ):
        if prompt_speech is not None:
            assistant_prompt_audio_token = self.preprocess_input(prompt_speech)
        else:
            assistant_prompt_audio_token = None
        if not read_text_only:
            text = self.preprocess_input(input)
            if assistant_prompt_audio_token is None:
                lm_prompt = [
                    InputSegment(
                        text="<|im_start|>system\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"你需要根据指定的风格指令和文本内容来生成语音。",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_start|>user\n{text}<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_start|>assistant\n<think>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
            else:
                lm_prompt = [
                    InputSegment(
                        text="<|im_start|>system\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"你需要根据指定的风格指令和文本内容来生成和语音prompt具有相同音色的语音。你的音色应该是：",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="",
                        audio=assistant_prompt_audio_token,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_start|>user\n{text}<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_start|>assistant\n<think>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
        else:
            language = detect_language(input)
            if language == "zh":
                template = random.choice(tts_zh_templates)
            else:
                template = random.choice(tts_en_templates)

            text = self.preprocess_input(input)
            if instruct is None:
                lm_prompt = [
                    InputSegment(
                        text=f"<|im_start|>user\n{template}: {text}<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_start|>assistant\n<|sostm|>",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
            else:
                if assistant_prompt_audio_token is None:
                    lm_prompt = [
                        InputSegment(
                            text="<|im_start|>system\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text=f"你需要根据指定的风格指令和文本内容来生成语音。",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="<|im_end|>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text=f"<|im_start|>user\n{template}: {text}({instruct})<|im_end|>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text=f"<|im_start|>assistant\n<think>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                    ]
                else:
                    lm_prompt = [
                        InputSegment(
                            text="<|im_start|>system\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text=f"你需要根据指定的风格指令和文本内容来生成和语音prompt具有相同音色的语音。你的音色应该是：",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="",
                            audio=assistant_prompt_audio_token,
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="<|im_end|>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text=f"<|im_start|>user\n{template}: {text}({instruct})<|im_end|>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text=f"<|im_start|>assistant\n<think>\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                    ]

        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_audio_understanding_sft_prompt(
        self,
        input_speech,
        input_text,
        thinking=False,
    ):
        audio_tokenized = self.preprocess_input(input_speech)

        lm_prompt = [
            InputSegment(
                text=f"<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                audio=audio_tokenized,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=input_text,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_end|>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]
        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )

        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_spoken_dialogue_sft_prompt(
        self,
        input_speech,
        system_prompt=None,
        prompt_speech=None,
        add_history=False,
    ):
        audio_tokenized = self.preprocess_input(input_speech)

        lm_prompt = []

        if add_history and self.history is not None:
            lm_prompt += [
                InputSegment(
                    text=f"<|im_start|>user\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    audio=audio_tokenized,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text=f"<|im_end|>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text=f"<|im_start|>assistant\n<|sostm|>",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
            ]
        else:
            if prompt_speech:
                lm_prompt += [
                    InputSegment(
                        text="<|im_start|>system\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"Your voice should be:",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        audio=self.preprocess_input(prompt_speech),
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]

            lm_prompt += [
                InputSegment(
                    text=f"<|im_start|>user\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            ]

            if system_prompt:
                lm_prompt += [
                    InputSegment(
                        text=system_prompt,
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text="\n\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
            lm_prompt += [
                InputSegment(
                    audio=audio_tokenized,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text=f"<|im_end|>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text=f"<|im_start|>assistant\n<|sostm|>",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
            ]

        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_spoken_dialogue_sft_multiturn_prompt(
        self,
        message_list,
        system_prompt=None,
        prompt_speech=None,
    ):
        lm_prompt = []

        if prompt_speech:
            lm_prompt += [
                InputSegment(
                    text="<|im_start|>system\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text=f"Your voice should be:",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    audio=self.preprocess_input(prompt_speech),
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text="<|im_end|>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
            ]

        for i in range(len(message_list)):
            if message_list[i]["role"] == "user":
                lm_prompt += [
                    InputSegment(
                        text=f"<|im_start|>user\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    )
                ]
                if system_prompt and i == 0:
                    lm_prompt += [
                        InputSegment(
                            text=system_prompt,
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                        InputSegment(
                            text="\n\n",
                            speech_zeroemb_idx=self.speech_zeroemb_idx,
                            text_zeroemb_idx=self.empty_token,
                        ),
                    ]
                lm_prompt += [
                    InputSegment(
                        audio=self.preprocess_input(message_list[i]["content"]),
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
            elif message_list[i]["role"] == "assistant":
                lm_prompt += [
                    InputSegment(
                        text=f"<|im_start|>assistant\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    StreamingInputSegment(
                        text=message_list[i]["content"]["text"],
                        audio=self.preprocess_input(
                            message_list[i]["content"]["audio"]
                        ),
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                        tokenizer=self.tokenizer,
                        group_size=self.group_size,
                        audio_channels=self.audio_channels,
                    ),
                    InputSegment(
                        text=f"<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
            else:
                raise ValueError(f"Invalid role: {message_list[i]['role']}")

        lm_prompt += [
            InputSegment(
                text=f"<|im_start|>assistant\n<|sostm|>",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]

        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_s2t_dialogue_sft_prompt(
        self,
        input_speech,
        thinking=False,
    ):
        audio_tokenized = self.preprocess_input(input_speech)
        lm_prompt = [
            InputSegment(
                text=f"<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                audio=audio_tokenized,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_end|>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]
        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_s2t_dialogue_sft_multiturn_prompt(self, message_list, thinking=False):
        lm_prompt = []
        for i in range(len(message_list)):
            if message_list[i]["role"] == "user":
                lm_prompt += [
                    InputSegment(
                        text=f"<|im_start|>user\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        audio=self.preprocess_input(message_list[i]["content"]),
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
            elif message_list[i]["role"] == "assistant":
                lm_prompt += [
                    InputSegment(
                        text=f"<|im_start|>assistant\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=message_list[i]["content"],
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
            else:
                raise ValueError(f"Invalid role: {message_list[i]['role']}")

        lm_prompt.append(
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        )
        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_text_dialogue_sft_prompt(
        self,
        input_text,
        thinking=False,
    ):
        lm_prompt = [
            InputSegment(
                text=f"<|im_start|>user\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=input_text,
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_end|>\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]
        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_text_dialogue_sft_multiturn_prompt(
        self,
        message_list,
        thinking=False,
    ):
        lm_prompt = []
        for i in range(len(message_list)):
            if message_list[i]["role"] == "user":
                lm_prompt += [
                    InputSegment(
                        text=f"<|im_start|>user\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=message_list[i]["content"],
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
            elif message_list[i]["role"] == "assistant":
                lm_prompt += [
                    InputSegment(
                        text=f"<|im_start|>assistant\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=message_list[i]["content"],
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                    InputSegment(
                        text=f"<|im_end|>\n",
                        speech_zeroemb_idx=self.speech_zeroemb_idx,
                        text_zeroemb_idx=self.empty_token,
                    ),
                ]
            else:
                raise ValueError(f"Invalid role: {message_list[i]['role']}")

        lm_prompt.append(
            InputSegment(
                text=f"<|im_start|>assistant\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        )
        if not thinking:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n\n</think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        else:
            lm_prompt.append(
                InputSegment(
                    text="<think>\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                )
            )
        input_ids = self.get_input_ids(lm_prompt)
        return input_ids

    def get_in_context_learning_s2s_prompt(self, instruction, prompt_examples, audio):
        prompt = [
            InputSegment(
                text=f"[Int]:{instruction}\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            )
        ]

        for i in range(len(prompt_examples)):
            prompt += [
                InputSegment(
                    audio=self.preprocess_input(prompt_examples[i]["input_audio"]),
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                InputSegment(
                    text="\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
                StreamingInputSegment(
                    text=prompt_examples[i]["output_transcription"],
                    audio=self.preprocess_input(prompt_examples[i]["output_audio"]),
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                    tokenizer=self.tokenizer,
                    group_size=self.group_size,
                    audio_channels=self.audio_channels,
                ),
                InputSegment(
                    text=" \n\n",
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.empty_token,
                ),
            ]

        prompt += [
            InputSegment(
                audio=self.preprocess_input(audio),
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text="\n",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
            InputSegment(
                text="<|sostm|>",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.empty_token,
            ),
        ]
        input_ids = self.get_input_ids(prompt)
        return input_ids

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        return_audio=False,
        output_audio_path=None,
        stopping_criteria=None,
        min_new_tokens=0,
        max_new_tokens=8192,
        add_history=False,
        task_name=None,
    ):

        task_sampler = self.get_task_sampler(task_name)

        generation_kwargs = self.generate_kwargs.copy()
        generation_config = GenerationConfig(**generation_kwargs)

        input_ids = input_ids.T.reshape(1, -1)  # [B, flattened(T, audio_channels + 1)]
        if add_history and self.history is not None:
            input_ids = torch.cat([self.history, input_ids], dim=1)

        prompt_length = input_ids.shape[1] // (self.audio_channels + 1)

        max_length = prompt_length // self.group_size + max_new_tokens
        min_length = prompt_length // self.group_size + min_new_tokens

        if stopping_criteria is not None:
            for criterion in stopping_criteria:
                if isinstance(criterion, MiMoStopper):
                    criterion.max_length = max_length
                    criterion.min_length = min_length

        generated_ids = self.model.generate(
            input_ids,
            generation_config,
            stopping_criteria=stopping_criteria,
            global_sampler=task_sampler["global"],
            local_sampler=task_sampler["local"],
        )

        self.history = generated_ids
        generated_ids = (
            generated_ids.int()
            .cpu()
            .reshape(-1, self.audio_channels + 1)
            .T[:, prompt_length:]
        )

        text = generated_ids[0, :: self.group_size][:-1]
        detokenized_text = (
            self.tokenizer.decode(text, skip_special_tokens=False)
            .strip()
            .replace("<|empty|>", "")
            .replace("<|eot|>", "")
            .replace("<|eostm|>", "")
        )
        print("Text channel:\t", detokenized_text)

        if output_audio_path:
            return_audio = True

        if not return_audio:
            return detokenized_text

        sosp_idx_locations = (text == self.sostm_idx).nonzero(as_tuple=True)[0]
        eosp_idx_locations = (text == self.eostm_idx).nonzero(as_tuple=True)[0]
        if len(sosp_idx_locations) == 0:
            start_location = 0
        else:
            start_location = sosp_idx_locations[0] * self.group_size + self.group_size
        if len(eosp_idx_locations) == 0:
            end_location = text.shape[0] * self.group_size
        else:
            end_location = eosp_idx_locations[0] * self.group_size
        audio_sequence = generated_ids[
            :, start_location:end_location
        ]  # [audio_channels+1, audio_length]
        speech_sequence = audio_sequence[1:]

        mask = speech_sequence[0] != (
            self.speech_zeroemb_idx[0]
            if isinstance(self.speech_zeroemb_idx, list)
            else self.speech_zeroemb_idx
        )
        speech_sequence = speech_sequence[:, mask]

        assert (
            speech_sequence < torch.tensor(self.speech_zeroemb_idx).unsqueeze(1)
        ).all()

        speech_sequence = speech_sequence.T.flatten()

        speech_str = "".join([f"<{i}>" for i in speech_sequence])
        tokens = torch.tensor([int(num) for num in re.findall(r"(\d+)>", speech_str)])

        if tokens.numel() == 0:
            wav = torch.zeros(24000)
            self.save_wav(output_audio_path, wav)
            return detokenized_text

        codes = tokens.reshape(-1, self.audio_channels).T
        codes = codes.type(torch.LongTensor).to(self.device)

        segment_len = 1500
        wav_list = []
        for start in range(0, codes.shape[-1], segment_len):
            wav = self.mimo_audio_tokenizer.decode(
                codes[:, start : start + segment_len]
            ).float()
            wav_list.append(wav)
        wav_concat = torch.cat(wav_list, dim=-1)

        # wav = self.mimo_audio_tokenizer.decode(codes).float()
        if output_audio_path is not None:
            self.save_wav(output_audio_path, wav_concat)
            return detokenized_text
        else:
            return wav_concat

    def asr_sft(self, audio):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_asr_sft_prompt(audio)
        result = self.forward(
            input_ids, stopping_criteria=stopping_criteria, task_name="asr"
        )
        return result

    def tts_sft(
        self, text, output_path, instruct=None, read_text_only=True, prompt_speech=None
    ):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[
                    self.tokenizer.eos_token_id,
                    self.eostm_idx,
                    self.im_end_idx,
                ],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_tts_sft_prompt(
            text,
            instruct=instruct,
            read_text_only=read_text_only,
            prompt_speech=prompt_speech,
        )
        text_output = self.forward(
            input_ids,
            output_audio_path=output_path,
            stopping_criteria=stopping_criteria,
            task_name="tts",
        )
        return text_output

    def audio_understanding_sft(self, input_speech, input_text, thinking=False):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_audio_understanding_sft_prompt(
            input_speech, input_text, thinking=thinking
        )
        result = self.forward(
            input_ids,
            stopping_criteria=stopping_criteria,
            task_name="audio_understanding",
        )
        return result

    def spoken_dialogue_sft(
        self,
        input_speech,
        output_audio_path=None,
        system_prompt=None,
        prompt_speech=None,
        add_history=False,
    ):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[
                    self.tokenizer.eos_token_id,
                    self.eostm_idx,
                    self.im_end_idx,
                ],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_spoken_dialogue_sft_prompt(
            input_speech,
            system_prompt=system_prompt,
            prompt_speech=prompt_speech,
            add_history=add_history,
        )
        text = self.forward(
            input_ids,
            output_audio_path=output_audio_path,
            stopping_criteria=stopping_criteria,
            task_name="spoken_dialogue",
            add_history=add_history,
        )
        return text

    # interface for message list interaction
    def spoken_dialogue_sft_multiturn(
        self,
        message_list,
        output_audio_path=None,
        system_prompt=None,
        prompt_speech=None,
    ):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[
                    self.tokenizer.eos_token_id,
                    self.eostm_idx,
                    self.im_end_idx,
                ],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_spoken_dialogue_sft_multiturn_prompt(
            message_list, system_prompt=system_prompt, prompt_speech=prompt_speech
        )
        text = self.forward(
            input_ids,
            output_audio_path=output_audio_path,
            stopping_criteria=stopping_criteria,
            task_name="spoken_dialogue",
            add_history=False,
        )
        return text

    def speech2text_dialogue_sft(self, input_speech, thinking=False, add_history=False):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_s2t_dialogue_sft_prompt(input_speech, thinking=thinking)
        text = self.forward(
            input_ids,
            stopping_criteria=stopping_criteria,
            task_name="spoken_dialogue",
            add_history=add_history,
        )
        return text

    # interface for message list interaction
    def speech2text_dialogue_sft_multiturn(self, message_list, thinking=False):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_s2t_dialogue_sft_multiturn_prompt(
            message_list, thinking=thinking
        )
        text = self.forward(
            input_ids,
            stopping_criteria=stopping_criteria,
            task_name="spoken_dialogue",
            add_history=False,
        )
        return text

    def text_dialogue_sft(self, input_text, thinking=False, add_history=False):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_text_dialogue_sft_prompt(input_text, thinking=thinking)
        text = self.forward(
            input_ids,
            stopping_criteria=stopping_criteria,
            task_name="text_chat",
            add_history=add_history,
        )
        return text

    # interface for message list interaction
    def text_dialogue_sft_multiturn(self, message_list, thinking=False):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.im_end_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_text_dialogue_sft_multiturn_prompt(
            message_list, thinking=thinking
        )
        text = self.forward(
            input_ids,
            stopping_criteria=stopping_criteria,
            task_name="text_chat",
            add_history=False,
        )
        return text

    def clear_history(self):
        self.history = None
        print("History cleared")

    def in_context_learning_s2s(
        self,
        instruction,
        prompt_examples,
        audio,
        max_new_tokens=None,
        output_audio_path=None,
    ):
        stopping_criteria = [
            MiMoStopper(
                stop_tokens=[self.tokenizer.eos_token_id, self.eostm_idx],
                group_size=self.group_size,
                audio_channels=self.audio_channels,
            )
        ]
        input_ids = self.get_in_context_learning_s2s_prompt(
            instruction, prompt_examples, audio
        )
        self.forward(
            input_ids,
            output_audio_path=output_audio_path,
            stopping_criteria=stopping_criteria,
            max_new_tokens=max_new_tokens,
            task_name="in_context_learning_s2s",
        )
