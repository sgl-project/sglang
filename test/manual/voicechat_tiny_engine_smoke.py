"""GPU smoke test for the two-stage VoiceChat SGLang session.

Uses tiny randomly initialized configs (`load_format="dummy"`) to validate
model construction, scheduler IPC, streaming KV sessions, custom inputs, and
custom outputs without downloading the production checkpoint.
"""

import tempfile
from pathlib import Path

import torch

from examples.voicechat.online_session import SGLangVoiceChatSession
from sglang import Engine
from sglang.srt.configs.eartts import EarTTSConfig
from sglang.srt.configs.nemotron_h import NemotronHConfig


def _write_configs(root: Path):
    duplex = root / "duplex"
    eartts = root / "eartts"
    duplex.mkdir()
    eartts.mkdir()
    NemotronHConfig(
        architectures=["NemotronDuplexHForCausalLM"],
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        layers_block_type=["mamba", "attention"],
        hybrid_override_pattern=None,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        n_groups=2,
        mamba_num_heads=2,
        mamba_head_dim=16,
        mamba_n_groups=2,
        ssm_state_size=4,
        mamba_d_conv=2,
        max_position_embeddings=128,
        pad_token_id=2,
        predict_user_text=False,
        use_function_head=True,
        duplex_text_channel_weight=1.0,
        duplex_user_channel_weight=1.0,
        duplex_function_channel_weight=2.0,
    ).save_pretrained(duplex)
    EarTTSConfig(
        architectures=["EarTTSForCausalLM"],
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        sliding_window=64,
        num_quantizers=3,
        codebook_size=4,
        latent_size=4,
        num_iter=2,
        mog_low_rank=2,
        mog_num_layers=1,
        mog_num_predictions=5,
        emb_vocab_size=32,
        top_p_or_k=1.0,
    ).save_pretrained(eartts)
    return duplex, eartts


def main():
    with tempfile.TemporaryDirectory() as directory:
        duplex_path, eartts_path = _write_configs(Path(directory))
        common = dict(
            load_format="dummy",
            dtype="bfloat16",
            mem_fraction_static=0.15,
            context_length=128,
            max_running_requests=2,
            disable_cuda_graph=True,
            attention_backend="triton",
            sampling_backend="pytorch",
            skip_tokenizer_init=True,
            enable_streaming_session=True,
            log_level="warning",
        )
        duplex = Engine(model_path=str(duplex_path), **common)
        print("duplex engine ready", flush=True)
        eartts = Engine(model_path=str(eartts_path), **common)
        print("eartts engine ready", flush=True)
        for name, engine in (("duplex", duplex), ("eartts", eartts)):
            generate = engine.generate

            def traced_generate(*args, _name=name, _generate=generate, **kwargs):
                print(f"{_name} generate begin", flush=True)
                result = _generate(*args, **kwargs)
                print(f"{_name} generate end", flush=True)
                return result

            engine.generate = traced_generate
        try:
            with SGLangVoiceChatSession(duplex, eartts, capacity=256) as session:
                print("sessions open", flush=True)
                session.start([1, 3, 2], torch.randn(3, 32), pad_token_id=2)
                print("prefills complete", flush=True)
                result = session.step(torch.randn(1, 32))
                print("frame complete", flush=True)
                assert 0 <= result.text_token < 32
                assert 0 <= result.function_token < 32
                assert len(result.audio_codes) == 3
                assert all(0 <= code < 4 for code in result.audio_codes)
                print(
                    "voicechat tiny-engine smoke passed:",
                    result.text_token,
                    result.function_token,
                    result.audio_codes,
                )
        finally:
            duplex.shutdown()
            eartts.shutdown()


if __name__ == "__main__":
    main()
