# SPDX-License-Identifier: Apache-2.0
"""Create a small, fixed random checkpoint for the *complete* native pipeline.

This fixture exercises real Qwen2.5-VL, Nomas and PixelUMM implementations.
It cannot produce meaningful videos or establish pretrained-model quality.
No network or pretrained checkpoint is needed.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from sglang.multimodal_gen.configs.models.dits.nomas import NomasArchConfig, NomasConfig
from sglang.multimodal_gen.configs.models.vaes.nomas import (
    NomasVAEArchConfig,
    NomasVAEConfig,
)
from sglang.multimodal_gen.runtime.models.dits.nomas import NomasTransformer3DModel
from sglang.multimodal_gen.runtime.models.vaes.nomas import NomasPixelUMMDecoder


def init_single_process_model_parallel():
    """The DiT's parallel linears need a TP group even on one process."""
    import os

    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        maybe_init_distributed_environment_and_model_parallel,
        model_parallel_is_initialized,
    )
    from sglang.srt.utils.network import get_free_port_below_ephemeral

    if model_parallel_is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(get_free_port_below_ephemeral()))
    for key in ("RANK", "LOCAL_RANK"):
        os.environ.setdefault(key, "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main():
    init_single_process_model_parallel()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--dit-heads",
        type=int,
        default=2,
        help="DiT attention heads (head_dim stays 128); tp_size * sp_degree must divide it",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    for component in ("tokenizer", "text_encoder", "transformer", "vae", "scheduler"):
        (args.output / component).mkdir()

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        ["system user assistant A red car drives forward. A blue boat turns left."],
        trainers.BpeTrainer(
            vocab_size=320,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<pad>", "<unk>", "<|im_start|>", "<|im_end|>"],
        ),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<|im_end|>",
    )
    fast.chat_template = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    fast.save_pretrained(args.output / "tokenizer")

    text_config = {
        "vocab_size": len(fast),
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 3,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 4096,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 3,
        "rope_scaling": {"rope_type": "default", "mrope_section": [8, 12, 12]},
    }
    vision_config = {
        "depth": 1,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_heads": 2,
        "out_hidden_size": 128,
        "fullatt_block_indexes": [0],
    }
    qwen = (
        Qwen2_5_VLForConditionalGeneration(
            Qwen2_5_VLConfig(
                text_config=text_config,
                vision_config=vision_config,
                image_token_id=len(fast) - 1,
                video_token_id=len(fast) - 2,
                vision_start_token_id=len(fast) - 3,
                vision_end_token_id=len(fast) - 4,
            )
        )
        .eval()
        .to(torch.bfloat16)
    )
    qwen.save_pretrained(args.output / "text_encoder")
    # Native text configuration also carries the conditioning length.
    path = args.output / "text_encoder/config.json"
    encoder_json = json.loads(path.read_text())
    encoder_json["text_len"] = 64
    write_json(path, encoder_json)

    dit_arch = NomasArchConfig(
        hidden_size=128 * args.dit_heads,
        num_attention_heads=args.dit_heads,
        num_layers=2,
        text_states_dim=128,
    )
    dit = (
        NomasTransformer3DModel(NomasConfig(arch_config=dit_arch))
        .eval()
        .to(torch.bfloat16)
    )
    # Training initializes several gates and the prediction head to zero.
    # Leaving those zeros makes an untrained pipeline ignore its conditions,
    # hiding integration bugs. Exercise every branch with fixed nonzero weights.
    with torch.no_grad():
        for name, param in dit.named_parameters():
            if param.ndim == 1 and "norm" in name and name.endswith("weight"):
                param.fill_(1)
            elif param.ndim >= 2:
                torch.nn.init.xavier_uniform_(param)
            else:
                param.normal_(0, 0.02)
    write_json(
        args.output / "transformer/config.json",
        {"_class_name": "NomasTransformer3DModel", **asdict(dit_arch)},
    )
    save_file(
        {k: v.contiguous() for k, v in dit.state_dict().items()},
        args.output / "transformer/diffusion_pytorch_model.safetensors",
    )
    vae_arch = NomasVAEArchConfig(
        hidden_size=128,
        intermediate_size=256,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=64,
    )
    vae = (
        NomasPixelUMMDecoder(NomasVAEConfig(arch_config=vae_arch))
        .eval()
        .to(torch.bfloat16)
    )
    write_json(
        args.output / "vae/config.json",
        {"_class_name": "NomasPixelUMMDecoder", **asdict(vae_arch)},
    )
    save_file(
        {k: v.contiguous() for k, v in vae.state_dict().items()},
        args.output / "vae/diffusion_pytorch_model.safetensors",
    )
    write_json(
        args.output / "scheduler/scheduler_config.json",
        {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "num_train_timesteps": 1000,
            "shift": 5.0,
        },
    )
    write_json(
        args.output / "model_index.json",
        {
            "_class_name": "NomasPipeline",
            "_diffusers_version": "0.37.0",
            "tokenizer": ["transformers", "AutoTokenizer"],
            "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
            "transformer": ["diffusers", "NomasTransformer3DModel"],
            "vae": ["diffusers", "NomasPixelUMMDecoder"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        },
    )
    manifest = {
        "random_weights": True,
        "nonzero_dit_gates_and_head": True,
        "seed": args.seed,
        "parameters": {
            "text_encoder": sum(p.numel() for p in qwen.parameters()),
            "transformer": sum(p.numel() for p in dit.parameters()),
            "vae": sum(p.numel() for p in vae.parameters()),
        },
    }
    write_json(args.output / "random_fixture.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
