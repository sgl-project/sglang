import logging
import re

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.fireredasr import FireRedASRForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.multimodal.processors.base_processor import (
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)


class FireRedASRMultimodalProcessor(SGLangBaseProcessor):
    models = [FireRedASRForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # Resolve tokenizer: _processor may be a bare tokenizer (no .tokenizer attr)
        if hasattr(self, "_tokenizer"):
            tokenizer = self._tokenizer
        elif hasattr(_processor, "tokenizer"):
            tokenizer = _processor.tokenizer
        else:
            tokenizer = _processor
        self._tokenizer = tokenizer

        audio_token = getattr(hf_config, "default_speech_token", "<speech>")
        escaped = re.escape(audio_token)
        self.AUDIO_TOKEN = audio_token
        self.AUDIO_TOKEN_REGEX = re.compile(rf"(?:{escaped})+")

        # Ensure the audio token exists in the tokenizer vocabulary.
        # Qwen2 base tokenizer may not include the <speech> token added during
        # FireRed fine-tuning, so we add it ourselves.
        token_id = tokenizer.convert_tokens_to_ids(audio_token)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if token_id is None or (unk_id is not None and token_id == unk_id):
            tokenizer.add_special_tokens(
                {"additional_special_tokens": [audio_token]}
            )
            token_id = tokenizer.convert_tokens_to_ids(audio_token)
            logger.info(
                "Added '%s' to tokenizer as special token (id=%d)",
                audio_token,
                token_id,
            )
        self.audio_token_id = token_id

        self.n_mels = getattr(hf_config.audio_config, "idim", 80)
        self.encoder_downsample_rate = getattr(
            hf_config, "encoder_downsample_rate", 2
        )

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        self.ATTR_NAME_TO_MODALITY.update(
            {
                "speech_lengths": Modality.AUDIO,
                "fake_token_lengths": Modality.AUDIO,
            }
        )

    def _compute_fbank(self, audio_array):
        """Compute log mel filterbank features from raw 16 kHz waveform."""
        import torchaudio

        if isinstance(audio_array, np.ndarray):
            waveform = torch.from_numpy(audio_array).float()
        else:
            waveform = torch.as_tensor(audio_array).float()

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        features = torchaudio.compliance.kaldi.fbank(
            waveform,
            sample_frequency=16000,
            num_mel_bins=self.n_mels,
        )
        return features  # [time_frames, n_mels]

    def _get_output_lengths(self, input_lengths):
        """Token count after Conv2dSubsampling + projector down-sampling."""
        padded = input_lengths + 6
        after_conv1 = (padded - 3) // 2 + 1
        encoder_frames = (after_conv1 - 3) // 2 + 1
        return encoder_frames // self.encoder_downsample_rate

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        result = dict(
            self._tokenizer(
                input_text,
                return_tensors="pt",
                add_special_tokens=True,
            )
        )

        if audios:
            features_list = [self._compute_fbank(a) for a in audios]

            max_len = max(f.shape[0] for f in features_list)
            padded = []
            for f in features_list:
                if f.shape[0] < max_len:
                    f = torch.nn.functional.pad(
                        f, (0, 0, 0, max_len - f.shape[0])
                    )
                padded.append(f)

            result["input_features"] = torch.stack(padded)
            frame_lengths = torch.tensor(
                [f.shape[0] for f in features_list]
            )
            audio_feature_lens = self._get_output_lengths(frame_lengths)
            result["audio_feature_lens"] = audio_feature_lens

            # HF processors return input_ids with the audio placeholder already
            # expanded to N copies (one per encoder output frame). Since we use
            # a bare tokenizer we must do this expansion ourselves, otherwise
            # pad_input_tokens / embed_mm_inputs will only see 1 placeholder
            # position while the encoder produces hundreds of embedding vectors.
            input_ids = result["input_ids"].flatten().tolist()
            new_input_ids = []
            audio_idx = 0
            for tid in input_ids:
                if tid == self.audio_token_id and audio_idx < len(audio_feature_lens):
                    n_tokens = int(audio_feature_lens[audio_idx].item())
                    new_input_ids.extend([self.audio_token_id] * n_tokens)
                    audio_idx += 1
                else:
                    new_input_ids.append(tid)
            result["input_ids"] = torch.tensor([new_input_ids])

        return result

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ):
        if audio_data and not self.AUDIO_TOKEN_REGEX.search(input_text):
            # Insert <speech> inside the last user turn so the audio
            # embeddings sit within the chat structure, not before it.
            marker = "<|im_start|>user\n"
            idx = input_text.rfind(marker)
            if idx >= 0:
                pos = idx + len(marker)
                input_text = input_text[:pos] + self.AUDIO_TOKEN + input_text[pos:]
            else:
                input_text = f"{self.AUDIO_TOKEN}{input_text}"

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

        if "audio_feature_lens" in ret and len(mm_items) > 0:
            mm_items[0].audio_feature_lens = ret["audio_feature_lens"]

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_token_id": self.audio_token_id,
        }
