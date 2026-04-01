#!/usr/bin/env python3
# Copyright 2025 Xiaomi Corporation.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.nn.functional as F
from typing import Tuple, Union, List


class InputSegment:

    def __init__(
        self,
        text: str = "",
        audio: torch.Tensor = None,
        tokenized_text: torch.Tensor = None,
        speech_zeroemb_idx: Union[int, List[int]] = 1024,
        text_zeroemb_idx: int = 152067,
        add_sosp_eosp=True,
    ) -> None:
        has_text = text is not None
        has_tokenized_text = tokenized_text is not None
        assert has_text or has_tokenized_text, "Text or tokenized text must be provided"

        self.audio = audio
        self.text = text
        self.tokenized_text = tokenized_text
        self.speech_zeroemb_idx = speech_zeroemb_idx
        self.text_zeroemb_idx = text_zeroemb_idx
        self.add_sosp_eosp = add_sosp_eosp

    @staticmethod
    def insert_between(tensor, i, value=-1):
        return torch.scatter(
            torch.full(
                (1, tensor.shape[1] + (tensor.shape[1] - 1) * i + i),
                value,
                dtype=tensor.dtype,
            ),
            1,
            torch.arange(0, tensor.shape[1], dtype=torch.int64)[None] * (i + 1),
            tensor,
        )

    def to_input_id(
        self,
        tokenizer,
        group_size: int,
        audio_channels: int = 8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.audio is None:
            if self.tokenized_text is None:
                tokenized_text = tokenizer(
                    self.text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=999999,
                    padding=False,
                    add_special_tokens=False,
                )["input_ids"].int()
            else:
                tokenized_text = self.tokenized_text.unsqueeze(0)

            
            if group_size > 1:
                tokenized_text = self.insert_between(
                    tokenized_text, group_size - 1, value=-100
                )
            
            
            if isinstance(self.speech_zeroemb_idx, list):
                audio_part_input_id = torch.zeros((audio_channels, tokenized_text.shape[1]), dtype=torch.int)
                for i, idx in enumerate(self.speech_zeroemb_idx):
                    audio_part_input_id[i, :] = idx
            else:
                audio_part_input_id = torch.full(
                    (audio_channels, tokenized_text.shape[1]), self.speech_zeroemb_idx, dtype=torch.int
                )
            
            
        else:
            sosp_token = (
                tokenizer.convert_tokens_to_ids("<|sosp|>")
                if self.add_sosp_eosp
                else None
            )
            eosp_token = (
                tokenizer.convert_tokens_to_ids("<|eosp|>")
                if self.add_sosp_eosp
                else None
            )
            audio_part = self.audio.reshape(-1, audio_channels).T  # [audio_channels, seqlen]

            assert (
                audio_part.shape[1] % group_size == 0
            ), f"Audio shape {audio_part.shape} is not divisible by group_size {group_size}"

            
            text_len = audio_part.shape[1] // group_size
            empty_token = self.text_zeroemb_idx
            if empty_token is None:
                empty_token = tokenizer.eod
            tokenized_text = torch.full((1, text_len), empty_token, dtype=torch.int)

            tokenized_text = (
                torch.cat(
                    [
                        torch.tensor([[sosp_token]], dtype=torch.int),
                        tokenized_text,
                        torch.tensor([[eosp_token]], dtype=torch.int),
                    ],
                    dim=1,
                )
                if self.add_sosp_eosp
                else tokenized_text
            )
            tokenized_text = self.insert_between(
                tokenized_text, group_size - 1, value=-100
            )
            
            
            if self.add_sosp_eosp:
                if isinstance(self.speech_zeroemb_idx, list):
                    sosp_part = torch.zeros((audio_channels, group_size), dtype=torch.int)
                    eosp_part = torch.zeros((audio_channels, group_size), dtype=torch.int)
                    for i, idx in enumerate(self.speech_zeroemb_idx):
                        sosp_part[i, :] = idx
                        eosp_part[i, :] = idx
                    audio_part_input_id = torch.cat([sosp_part, audio_part, eosp_part], dim=1)
                else:
                    audio_part_input_id = torch.cat(
                        [
                            torch.full((audio_channels, group_size), self.speech_zeroemb_idx, dtype=torch.int),
                            audio_part,
                            torch.full((audio_channels, group_size), self.speech_zeroemb_idx, dtype=torch.int),
                        ],
                        dim=1,
                    )
            else:
                audio_part_input_id = audio_part
            
            

        input_ids = torch.cat(
            [tokenized_text, audio_part_input_id], dim=0
        )  # [n_rvq + 1, seqlen]
        

        return input_ids


class StreamingInputSegment:
    def __init__(
        self,
        text: str = "",
        audio: torch.Tensor = None,
        tokenized_text: torch.Tensor = None,
        speech_zeroemb_idx: Union[int, List[int]] = 1024,
        text_zeroemb_idx: int = 152067,
        text_segment_size: int = 5,
        audio_segment_size: int = 5,
        tokenizer=None,
        group_size=None,
        audio_channels=None,
    ) -> None:
        has_text = text is not None
        has_tokenized_text = tokenized_text is not None
        assert has_text or has_tokenized_text, "Text or tokenized text must be provided"

        self.audio = audio
        self.text = text
        self.tokenized_text = tokenized_text
        self.speech_zeroemb_idx = speech_zeroemb_idx
        self.text_zeroemb_idx = text_zeroemb_idx
        self.text_segment_size = text_segment_size
        self.audio_segment_size = audio_segment_size
        self.tokenizer = tokenizer
        self.group_size = group_size
        self.audio_channels = audio_channels

    def to_input_id(
        self,
        tokenizer,
        group_size: int,
        audio_channels: int = 8,
    ):
        if self.tokenized_text is None:
            tokenized_text = tokenizer(
                self.text,
                return_tensors="pt",
                truncation=True,
                max_length=999999,
                padding=False,
                add_special_tokens=False,
            )["input_ids"].int()  # [1, seqlen]
        else:
            tokenized_text = self.tokenized_text.unsqueeze(0)

        tokenized_text = tokenized_text.squeeze(0)

        text_segments = tokenized_text.split(self.text_segment_size, dim=0)
        audio_segments = self.audio.split(self.audio_segment_size*group_size*audio_channels, dim=0)

        tokenized_segments = []
        tokenized_segments.append(
            InputSegment(
                text='<|sostm|>',
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.text_zeroemb_idx,
            ),
        )


        eot_tokens = tokenizer(
            "<|eot|>",
            return_tensors="pt",
            truncation=True,
            max_length=999999,
            padding=False,
            add_special_tokens=False,
        )["input_ids"][0].to(text_segments[-1])


        text_segments = text_segments[:-1] + (torch.cat([text_segments[-1], eot_tokens], dim=0),)


        length = min(len(text_segments), len(audio_segments))
        for i in range(length):
            text_segment = text_segments[i]
            audio_segment = audio_segments[i]

            tokenized_segments.append(
                InputSegment(
                    tokenized_text=text_segment,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.text_zeroemb_idx,
                ),
            )
            tokenized_segments.append(
                InputSegment(
                    audio=audio_segment,
                    add_sosp_eosp=False,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.text_zeroemb_idx,
                ),
            )
        
        for j in range(length, len(text_segments)):
            tokenized_segments.append(
                InputSegment(
                    tokenized_text=text_segments[j],
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.text_zeroemb_idx,
                ),
            )
        
        for j in range(length, len(audio_segments)):
            tokenized_segments.append(
                InputSegment(
                    audio=audio_segments[j],
                    add_sosp_eosp=False,
                    speech_zeroemb_idx=self.speech_zeroemb_idx,
                    text_zeroemb_idx=self.text_zeroemb_idx,
                ),
            )

        tokenized_segments.append(
            InputSegment(
                text="<|eostm|>",
                speech_zeroemb_idx=self.speech_zeroemb_idx,
                text_zeroemb_idx=self.text_zeroemb_idx,
            ),
        )


        input_ids = [
            seg.to_input_id(
                self.tokenizer,
                self.group_size,
                self.audio_channels,
            )
            for seg in tokenized_segments
        ]
        

        
        input_ids = torch.cat(input_ids, dim=1).type(torch.int64)  # [n_rvq + 1, seqlen]
        
        return input_ids
