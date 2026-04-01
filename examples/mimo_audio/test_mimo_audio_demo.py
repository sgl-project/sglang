import os
import re
from typing import Union, Optional, cast

import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from transformers.cache_utils import Cache

from transformers import (
    AutoTokenizer,
    GenerationConfig
)
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from mimo_src.mimo_audio_tokenizer import MiMoAudioTokenizer
from mimo_src.mimo_audio.process_speechdata import InputSegment 
from mimo_src.mimo_audio.mimo_audio import MimoAudio

import sglang as sgl
from sglang.srt.managers.io_struct import MiMoAudioMMId


class MiMoAudioInfer():
    def __init__(self, model_path: str, audio_tokenizer_path: str):
        self.device = "cuda"
        self.llm_engine = sgl.Engine(model_path=model_path, disable_radix_cache=True,
                                     disable_piecewise_cuda_graph=True, disable_cuda_graph=True, disable_overlap_schedule=True,
                                    #  attention_backend="torch_native"
                                     )
        self.mimo_audio = MimoAudio(model_path, audio_tokenizer_path, device=self.device)
        # self.llm_engine = None
    
    def audio_understanding_sft(self, audio_path: str, text: str, thinking=False):
        # input_ids = self.get_audio_understanding_sft_prompt(audio_path, text, thinking=thinking)
        input_ids = self.mimo_audio.get_audio_understanding_sft_prompt(audio_path, text)
        result = self.generate(input_ids)
        return result

    def tts_sft(self, audio_path: str, text: str):
        input_ids = self.mimo_audio.get_tts_sft_prompt(text)
        result = self.generate(input_ids, output_audio_path=audio_path)
        return result

    @torch.no_grad()              
    def generate(
        self,
        input_ids,
        return_audio=False,
        output_audio_path=None,
        max_new_tokens=4096,
    ):
        
        # task_sampler = self.get_task_sampler(task_name)
        
        # generation_kwargs = self.generate_kwargs.copy()
        # generation_config = GenerationConfig(**generation_kwargs)

        # input_ids = input_ids.T.reshape(1, -1) # [B, flattened(T, audio_channels + 1)]
        # if add_history and self.history is not None:
        #     input_ids = torch.cat([self.history, input_ids], dim=1)

        # prompt_length = input_ids.shape[1] // (self.audio_channels+1)

        # max_length = prompt_length // self.group_size + max_new_tokens
        # min_length = prompt_length // self.group_size + min_new_tokens
        
        # if stopping_criteria is not None:
        #     for criterion in stopping_criteria:
        #         if isinstance(criterion, MiMoStopper):
        #             criterion.max_length = max_length
        #             criterion.min_length = min_length
        
        sampling_params = {
            "temperature": 1,
            "max_new_tokens": max_new_tokens,
            "top_p": 1,
            "top_k": 1,
            "repetition_penalty": 1.0,
            "ignore_eos": False,
        }
        # import pdb;pdb.set_trace()
        input_ids = input_ids.transpose(-1,-2).contiguous()
        # model_input = self.prepare_inputs_for_generation(input_ids)
        # input_ids = model_input["input_ids"]
        input_ids = input_ids.view(-1, 4, 9)    # [seq_len, group_size, audio_channel + 1]
        input_ids = input_ids.to(torch.float32).cpu().detach().tolist() # shape [1, 13464] [seq_len, group_size, audio_channel + 1]
        input_ids = [MiMoAudioMMId.init_new(mm) for mm in input_ids]
        
        output = self.llm_engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            # stream=True,
        )
        
        # prev_length = 0
        # import sys
        # for chunk in output:
        #     list_mm_ids = chunk['output_ids']
        #     list_mm_ids = [torch.tensor(k.mm_ids) for k in list_mm_ids]
        #     tensor_mm_ids = torch.cat(list_mm_ids)  # shape [340, 9]
        #     # import pdb;pdb.set_trace()
        #     cur_length = tensor_mm_ids.shape[0]
        #     tensor_mm_ids = tensor_mm_ids[prev_length:, :]
        #     prev_length = cur_length
        #     generated_ids = tensor_mm_ids.transpose(-1,-2)
            
        #     # generated_ids = generated_ids.int().cpu().reshape(-1, self.audio_channels+1).T

        #     text = generated_ids[0, ::self.mimo_audio.group_size][:-1]
        #     detokenized_text = self.mimo_audio.tokenizer.decode(text, skip_special_tokens=False).strip().replace("<|empty|>", "").replace("<|eot|>", "").replace("<|eostm|>", "")
        #     # import pdb;pdb.set_trace()
        #     print(detokenized_text, end="", flush=True)
        #     sys.stdout.flush()
        # import pdb;pdb.set_trace()
        list_mm_ids = output['output_ids']
        list_mm_ids = [torch.tensor(k.mm_ids) for k in list_mm_ids]
        tensor_mm_ids = torch.cat(list_mm_ids)  # shape [340, 9]
        generated_ids = tensor_mm_ids.transpose(-1,-2)
        
        # generated_ids = generated_ids.int().cpu().reshape(-1, self.audio_channels+1).T

        text = generated_ids[0, ::self.mimo_audio.group_size][:-1]
        detokenized_text = self.mimo_audio.tokenizer.decode(text, skip_special_tokens=False).strip().replace("<|empty|>", "").replace("<|eot|>", "").replace("<|eostm|>", "")
        print("Text channel:\t", detokenized_text)

        if output_audio_path:
            return_audio = True
        
        if not return_audio:
            return detokenized_text
        
        sosp_idx_locations = (text == self.mimo_audio.sostm_idx).nonzero(as_tuple=True)[0]
        eosp_idx_locations = (text == self.mimo_audio.eostm_idx).nonzero(as_tuple=True)[0]
        if len(sosp_idx_locations) == 0:
            start_location = 0
        else:
            start_location = sosp_idx_locations[0] * self.mimo_audio.group_size + self.mimo_audio.group_size
        if len(eosp_idx_locations) == 0:
            end_location = text.shape[0] * self.mimo_audio.group_size
        else:
            end_location = eosp_idx_locations[0] * self.mimo_audio.group_size
        audio_sequence = generated_ids[:, start_location:end_location]  #[audio_channels+1, audio_length]
        speech_sequence = audio_sequence[1:]

        mask = speech_sequence[0] != (self.mimo_audio.speech_zeroemb_idx[0] if isinstance(self.mimo_audio.speech_zeroemb_idx, list) else self.mimo_audio.speech_zeroemb_idx)
        speech_sequence = speech_sequence[:, mask]

        assert (speech_sequence < torch.tensor(self.mimo_audio.speech_zeroemb_idx).unsqueeze(1)).all()
        
        speech_sequence = speech_sequence.T.flatten()
    
        speech_str = "".join([f"<{i}>" for i in speech_sequence])
        tokens = torch.tensor(
            [int(num) for num in re.findall(r"(\d+)>", speech_str)]
        )

        if tokens.numel() == 0:
            wav = torch.zeros(24000)
            self.mimo_audio.save_wav(output_audio_path, wav)
            return detokenized_text
        
        codes = tokens.reshape(-1, self.mimo_audio.audio_channels).T
        codes = codes.type(torch.LongTensor).to(self.device)
        
        segment_len = 1500
        wav_list=[]
        for start in range(0, codes.shape[-1], segment_len):
            wav = self.mimo_audio.mimo_audio_tokenizer.decode(codes[:,start:start+segment_len]).float() 
            wav_list.append(wav)
        wav_concat = torch.cat(wav_list, dim=-1)

        #wav = self.mimo_audio_tokenizer.decode(codes).float()
        if output_audio_path is not None:
            self.mimo_audio.save_wav(output_audio_path, wav_concat)
            return detokenized_text
        else:
            return wav_concat
        
if __name__ == '__main__':
    model_path = "/mnt/lustre-client/zhangzizheng/ALL_MODELS/MiMo-Audio-7B-Instruct"
    mimo_audio_tokenizer_path = "/mnt/lustre-client/zhangzizheng/ALL_MODELS/MiMo-Audio-Tokenizer"
    
    # audio understanding
    audio_path = "examples/spoken_dialogue_assistant_turn_1.wav"
    text = "Summarize the audio."
    
    # tts
    # text = "今天天气真好"
    # output_audio_path = "examples/tts.wav"

    
    mimo_audio_infer = MiMoAudioInfer(model_path, mimo_audio_tokenizer_path)
    result = mimo_audio_infer.audio_understanding_sft(audio_path, text)
    # result = mimo_audio_infer.tts_sft(output_audio_path, text)