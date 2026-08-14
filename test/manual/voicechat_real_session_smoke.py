"""One-frame smoke test for the complete production SGLang VoiceChat session."""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig

from examples.voicechat.online_session import SGLangVoiceChatSession
from sglang import Engine
from sglang.srt.configs.eartts import EarTTSConfig  # noqa: F401


def _load_speaker(model_dir: Path, speaker_latent_path: str | None):
    if speaker_latent_path is None:
        candidates = sorted((model_dir / "speaker_latents").glob("*.pt"))
        if not candidates:
            raise FileNotFoundError("No converted speaker latent was found.")
        speaker_latent_path = str(candidates[0])
    speaker = torch.load(speaker_latent_path, map_location="cpu", weights_only=True)
    if speaker.ndim == 3 and speaker.shape[0] == 1:
        speaker = speaker.squeeze(0)
    return speaker


def main(duplex_model: str, eartts_model: str, speaker_latent_path: str | None):
    duplex_config = AutoConfig.from_pretrained(duplex_model, trust_remote_code=False)
    speaker = _load_speaker(Path(eartts_model), speaker_latent_path)
    common = {
        "max_running_requests": 2,
        "skip_tokenizer_init": True,
        "enable_streaming_session": True,
        "log_level": "warning",
    }
    duplex = Engine(
        model_path=duplex_model,
        dtype="bfloat16",
        mem_fraction_static=0.45,
        context_length=1024,
        **common,
    )
    eartts = Engine(
        model_path=eartts_model,
        dtype="float32",
        mem_fraction_static=0.20,
        context_length=1024,
        **common,
    )
    try:
        with SGLangVoiceChatSession(duplex, eartts, capacity=2048) as session:
            session.start(
                [duplex_config.bos_token_id, duplex_config.pad_token_id],
                speaker,
                duplex_config.pad_token_id,
            )
            result = session.step(torch.randn(1, duplex_config.hidden_size))
            assert 0 <= result.text_token < duplex_config.vocab_size
            assert 0 <= result.function_token < duplex_config.vocab_size
            assert len(result.audio_codes) == 31
            print("real VoiceChat session smoke passed", result, flush=True)
    finally:
        duplex.shutdown()
        eartts.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("duplex_model")
    parser.add_argument("eartts_model")
    parser.add_argument("--speaker-latent")
    args = parser.parse_args()
    main(args.duplex_model, args.eartts_model, args.speaker_latent)
