"""Create a deterministic CPU-only fused Qwen4 fixture without downloading assets.

Keeps GatedDeltaNet, QSA, hyper-connections, shared/routed experts and PLE.
This synthetic model supplies no task-quality or large-model performance evidence.

From the SGLang repository root, using its Qwen4-capable Transformers environment:
    PYTHONPATH=python python test/manual/models/create_tiny_qwen4_fixture.py \
        --out scratch/tiny-qwen4
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
import resource
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
ROOT = Path(__file__).resolve().parents[3]
QSA_CONFIG = ROOT / "python/sglang/srt/layers/attention/qsa/config.py"
MAX_CHECKPOINT_BYTES = 128 * 1024**2
MAX_HOST_BYTES = 1024**3


def tiny_config(source: dict) -> dict:
    config = copy.deepcopy(source)
    text = config["text_config"]
    text.update(
        vocab_size=512,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=256,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        full_attention_interval=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        hc_count=4,
        hc_lowrank=32,
        max_position_embeddings=1024,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        mamba_ssm_dtype="float32",
        output_gate_type="sigmoid",
        partial_rotary_factor=0.25,
        number_of_conv_states=3,
        ple_layer_ids=[2],
        ple_embed_dim=256,
        heads_per_ngram=4,
        ngram_size=3,
        ngram_vocab_size_base=32,
        split_ngram_parts=4,
        make_ngram_vocab_size_divisible_by=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        mtp_num_hidden_layers=0,
        mtp_use_dedicated_embeddings=False,
        dtype="bfloat16",
        norm_topk_prob=True,
        seed=1234,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    )
    text.pop("mtp", None)
    config["vision_config"].update(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=256,
        num_position_embeddings=16,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
        deepstack_visual_indexes=[],
    )
    config.update(
        language_model_only=True,
        image_token_id=503,
        video_token_id=504,
        vision_start_token_id=501,
        vision_end_token_id=502,
        architectures=["Qwen4ExpForConditionalGeneration"],
    )
    return config


def preflight(config: dict) -> None:
    """Reject any inherited large allocation before importing/constructing a model."""
    text, vision = config["text_config"], config["vision_config"]
    limits = dict(
        vocab_size=512,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=256,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        hc_count=4,
        hc_lowrank=32,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        ple_embed_dim=256,
        heads_per_ngram=4,
        ngram_size=3,
        ngram_vocab_size_base=32,
        split_ngram_parts=4,
    )
    for name, maximum in limits.items():
        if not 0 < text[name] <= maximum:
            raise ValueError(
                f"Unsafe text dimension {name}={text[name]}, limit={maximum}"
            )
    for name, maximum in dict(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=256,
        num_position_embeddings=16,
    ).items():
        if not 0 < vision[name] <= maximum:
            raise ValueError(
                f"Unsafe vision dimension {name}={vision[name]}, limit={maximum}"
            )
    if text["ple_layer_ids"] != [2] or text["layer_types"] != [
        "linear_attention"
    ] * 3 + ["full_attention"]:
        raise ValueError(
            "Tiny checkpoint must retain its fixed GDN/QSA/PLE layer structure"
        )


def engine_preflight(config: dict, qsa_config_path: Path) -> dict:
    """Execute the engine's CPU-only QSA parser without importing GPU runtime modules."""
    specification = importlib.util.spec_from_file_location(
        "p2_qsa_constraints", qsa_config_path
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    profile = module.parse_qsa_profile(
        SimpleNamespace(text_config=SimpleNamespace(**config["text_config"]))
    )
    text = config["text_config"]
    rotary_dim = int(
        text["head_dim"] * text["rope_parameters"]["partial_rotary_factor"]
    )
    if not 0 < rotary_dim <= profile.head_dim or rotary_dim % 2:
        raise ValueError(
            "QSA RoPE dimensions must be positive, even and fit the index head"
        )
    if profile.head_dim not in (64, 128, 256):
        raise ValueError(
            "Tiny fixture must use a supported fused indexer-prep head dimension"
        )
    return dict(
        config_path=str(qsa_config_path.resolve()),
        config_sha256=hashlib.sha256(qsa_config_path.read_bytes()).hexdigest(),
        n_heads=profile.n_heads,
        kv_heads=profile.kv_heads,
        head_dim=profile.head_dim,
        budget=profile.budget,
        compress_ratio=profile.compress_ratio,
        block_topk=profile.block_topk,
        rotary_dim=rotary_dim,
    )


def self_test(source: dict, qsa_config_path: Path) -> None:
    original = copy.deepcopy(source)
    config = tiny_config(source)
    preflight(config)
    assert source == original
    unsafe = copy.deepcopy(config)
    unsafe["text_config"]["ngram_vocab_size_base"] = 20_000_000
    try:
        preflight(unsafe)
    except ValueError:
        pass
    else:
        raise AssertionError("Original PLE allocation escaped size checks")
    assert engine_preflight(config, qsa_config_path)["block_topk"] == 512
    unsafe = copy.deepcopy(config)
    unsafe["text_config"]["indexer_budget"] = 32
    try:
        engine_preflight(unsafe, qsa_config_path)
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported QSA block top-k escaped engine preflight")


def write_tokenizer(out: Path) -> None:
    from tokenizers import pre_tokenizers
    from transformers import AutoTokenizer, Qwen3_5Tokenizer

    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocab.update(
        {
            token: index
            for index, token in enumerate(
                sorted(pre_tokenizers.ByteLevel.alphabet()), 4
            )
        }
    )
    vocab.update({f"t{i}": i for i in range(len(vocab), 512)})
    tokenizer = Qwen3_5Tokenizer(
        vocab=vocab,
        merges=[],
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        model_max_length=1024,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    )
    tokenizer.save_pretrained(out)
    reloaded = AutoTokenizer.from_pretrained(out, local_files_only=True)
    ids = reloaded.encode("hello world", add_special_tokens=False)
    if not ids or max(ids) >= 512 or reloaded.decode(ids) != "hello world":
        raise AssertionError("Tiny tokenizer failed automatic reload/roundtrip")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, help="New checkpoint directory, never overwritten"
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    from transformers import Qwen4ExpConfig

    source = Qwen4ExpConfig().to_dict()
    self_test(source, QSA_CONFIG)
    if args.self_test:
        print("tiny config bounds self-test passed; no model constructed")
        return
    if args.out is None:
        parser.error("--out is required unless --self-test is selected")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing checkpoint {args.out}")
    config_dict = tiny_config(source)
    preflight(config_dict)
    engine_qsa = engine_preflight(config_dict, QSA_CONFIG)
    engine_qsa.pop("config_path")

    import torch
    import transformers
    from safetensors import safe_open
    from transformers import AutoModelForImageTextToText

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = Qwen4ExpConfig(**config_dict)
    with torch.device("meta"):
        meta = AutoModelForImageTextToText.from_config(
            config, dtype=torch.bfloat16, attn_implementation="eager"
        )
    exact_bytes = sum(t.numel() * t.element_size() for t in meta.state_dict().values())
    if exact_bytes >= MAX_CHECKPOINT_BYTES:
        raise ValueError(f"Meta checkpoint size {exact_bytes} exceeds 128 MiB ceiling")
    del meta
    gc.collect()
    torch.manual_seed(args.seed)
    with torch.device("cpu"):
        model = AutoModelForImageTextToText.from_config(
            config, dtype=torch.bfloat16, attn_implementation="eager"
        )
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise AssertionError("Tiny generator must allocate CPU tensors only")
    # HF's original-format conversion unfuses MoE tensors into individual experts.
    # Preserve the real Qwen3.8 fused tensors and explicitly restore only PLE shards.
    state = model.state_dict()
    for name in list(state):
        if name.endswith(".ple_embedding.ngram_embedding.weight"):
            weight = state.pop(name)
            for index, shard in enumerate(
                weight.chunk(config.text_config.split_ngram_parts, dim=0)
            ):
                state[name.removesuffix(".weight") + f".shard_{index}.weight"] = (
                    shard.contiguous()
                )
    args.out.mkdir(parents=True)
    # Transformers may consume entries from the supplied state dict while saving.
    expected_shapes = {name: list(tensor.shape) for name, tensor in state.items()}
    model.save_pretrained(
        args.out, state_dict=state, max_shard_size="64MB", save_original_format=False
    )
    write_tokenizer(args.out)
    weights = {}
    for path in args.out.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            weights.update(
                {
                    name: list(checkpoint.get_slice(name).get_shape())
                    for name in checkpoint.keys()
                }
            )
    if weights != expected_shapes:
        raise AssertionError("Saved tensors do not cover the complete fused/PLE state")
    if not any("experts.gate_up_proj" in name for name in weights):
        raise AssertionError(
            "HF checkpoint did not preserve fused routed-expert tensors"
        )
    if not any("ple_embedding.ngram_embedding.shard_" in name for name in weights):
        raise AssertionError("HF checkpoint did not restore the PLE shard_N layout")
    checkpoint_bytes = sum(
        path.stat().st_size for path in args.out.iterdir() if path.is_file()
    )
    peak_host_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if checkpoint_bytes >= MAX_CHECKPOINT_BYTES or peak_host_bytes >= MAX_HOST_BYTES:
        raise ValueError(
            f"Budget exceeded: checkpoint={checkpoint_bytes}, peak_host={peak_host_bytes}"
        )
    evidence = dict(
        source="Bounded overrides of Transformers Qwen4ExpConfig defaults",
        seed=args.seed,
        transformers_version=transformers.__version__,
        meta_state_bytes=exact_bytes,
        checkpoint_bytes_without_manifest=checkpoint_bytes,
        engine_qsa_preflight=engine_qsa,
        weights=weights,
        purpose="Random loader/equivalence fixture; no quality claims",
    )
    (args.out / "fixture_manifest.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )
    checkpoint_bytes = sum(
        path.stat().st_size for path in args.out.iterdir() if path.is_file()
    )
    if checkpoint_bytes >= MAX_CHECKPOINT_BYTES:
        raise ValueError("Checkpoint plus manifest exceeds the 128 MiB ceiling")
    print(
        json.dumps(
            dict(
                out=str(args.out),
                checkpoint_bytes=checkpoint_bytes,
                peak_host_bytes=peak_host_bytes,
                seed=args.seed,
            )
        )
    )


if __name__ == "__main__":
    main()
