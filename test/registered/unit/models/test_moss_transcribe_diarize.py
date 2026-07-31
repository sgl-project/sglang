from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoConfig

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.configs.moss_transcribe_diarize import MossTranscribeDiarizeConfig
from sglang.srt.models.moss_transcribe_diarize import (
    MossTranscribeDiarizeForConditionalGeneration,
    whisper_encoder_output_length,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakeWhisperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.last_batch_shape = None

    def forward(self, input_features, position_ids, forward_batch):
        self.last_batch_shape = tuple(input_features.shape)
        batch_size = input_features.shape[0]
        hidden_size = 2
        sequence_length = position_ids.numel()
        values = torch.arange(
            batch_size * sequence_length * hidden_size,
            dtype=input_features.dtype,
            device=input_features.device,
        )
        return values.reshape(batch_size, sequence_length, hidden_size)


class TestMossTranscribeDiarizeModel(CustomTestCase):
    def test_config_registry_and_round_trip(self):
        config = AutoConfig.for_model("moss_transcribe_diarize")
        self.assertIsInstance(config, MossTranscribeDiarizeConfig)

        restored = MossTranscribeDiarizeConfig.from_dict(config.to_dict())
        self.assertEqual(restored.text_config.model_type, "qwen3")
        self.assertEqual(restored.audio_config.model_type, "whisper")
        self.assertEqual(
            restored.adaptor_input_dim,
            restored.audio_config.d_model * restored.audio_merge_size,
        )

    def test_whisper_encoder_length_handles_odd_inputs(self):
        self.assertEqual(whisper_encoder_output_length(5), 3)
        self.assertEqual(whisper_encoder_output_length(6), 3)

    def test_time_merge_trims_incomplete_group(self):
        model = MossTranscribeDiarizeForConditionalGeneration.__new__(
            MossTranscribeDiarizeForConditionalGeneration
        )
        nn.Module.__init__(model)
        model.config = SimpleNamespace(audio_merge_size=2)
        features = torch.arange(1 * 5 * 2).reshape(1, 5, 2)

        merged = model.time_merge(features)

        self.assertEqual(tuple(merged.shape), (1, 2, 4))
        torch.testing.assert_close(merged, features[:, :4].reshape(1, 2, 4))

    def test_audio_chunks_are_encoded_in_one_batch_and_regrouped(self):
        model = MossTranscribeDiarizeForConditionalGeneration.__new__(
            MossTranscribeDiarizeForConditionalGeneration
        )
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            audio_merge_size=2,
            text_config=SimpleNamespace(hidden_size=4),
        )
        model.whisper_encoder = _FakeWhisperEncoder()
        model.vq_adaptor = nn.Linear(4, 4, bias=False)

        item = SimpleNamespace(
            feature=torch.arange(2 * 4 * 5, dtype=torch.float32).reshape(2, 4, 5),
            audio_feature_lengths=torch.tensor([1, 1]),
            audio_chunk_mapping=torch.tensor([0, 0]),
        )

        output = model.get_audio_feature([item], forward_batch=SimpleNamespace())

        self.assertEqual(model.whisper_encoder.last_batch_shape, (2, 4, 5))
        self.assertEqual(tuple(output.shape), (2, 4))
