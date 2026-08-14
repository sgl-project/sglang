"""One-frame smoke test for a converted production VoiceChat EarTTS stage."""

import argparse
from pathlib import Path

import torch
from transformers import AutoConfig

from sglang import Engine
from sglang.srt.configs.eartts import EarTTSConfig  # noqa: F401


def main(model_path: str, speaker_latent_path: str | None):
    model_dir = Path(model_path)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if speaker_latent_path is None:
        candidates = sorted((model_dir / "speaker_latents").glob("*.pt"))
        if not candidates:
            raise FileNotFoundError("No converted speaker latent was found.")
        speaker_latent_path = str(candidates[0])
    speaker = torch.load(speaker_latent_path, map_location="cpu", weights_only=True)
    if speaker.ndim == 3 and speaker.shape[0] == 1:
        speaker = speaker.squeeze(0)
    if speaker.shape[-1] != config.hidden_size:
        raise ValueError(
            f"Speaker latent has shape {tuple(speaker.shape)}; expected "
            f"[frames, {config.hidden_size}]."
        )

    engine = Engine(
        model_path=model_path,
        dtype="float32",
        mem_fraction_static=0.20,
        context_length=1024,
        max_running_requests=2,
        skip_tokenizer_init=True,
        enable_streaming_session=True,
        log_level="warning",
    )
    session_id = engine.open_session(2048, streaming=True)
    params = {
        "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
        "session_params": {"id": session_id, "rid": None},
    }
    try:
        engine.generate(
            input_ids=[0] * speaker.shape[0],
            custom_inputs={
                "is_speaker_prefill": True,
                "speaker_latent": speaker,
            },
            **params,
        )
        result = engine.generate(
            input_ids=[],
            custom_inputs={"text_token": 12, "previous_audio_codes": None},
            **params,
        )
        codes = result["meta_info"]["audio_codes"][-1]
        assert len(codes) == config.num_quantizers
        assert all(0 <= int(code) < config.codebook_size for code in codes)
        print("real EarTTS smoke passed", codes, flush=True)
    finally:
        engine.close_session(session_id)
        engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--speaker-latent")
    args = parser.parse_args()
    main(args.model_path, args.speaker_latent)
