"""One-frame smoke test for a converted VoiceChat Duplex checkpoint."""

import argparse

import torch
from transformers import AutoConfig

from sglang import Engine


def main(model_path: str):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    engine = Engine(
        model_path=model_path,
        dtype="bfloat16",
        mem_fraction_static=0.45,
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
        result = engine.generate(
            input_ids=[config.bos_token_id, config.pad_token_id],
            custom_inputs={
                "is_initial_prefill": True,
                "prompt_length": 1,
                "acoustic_embedding": torch.randn(1, config.hidden_size),
            },
            **params,
        )
        assert len(result["output_ids"]) == 1
        assert len(result["meta_info"]["function_tokens"]) == 1
        print("real Duplex smoke passed", result, flush=True)
    finally:
        engine.close_session(session_id)
        engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    main(parser.parse_args().model_path)
