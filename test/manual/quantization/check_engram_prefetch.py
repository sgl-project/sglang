"""Check real checkpoint WKV/gate weights with compact host tables on one GPU.

This validates Engram components, not full-model accuracy or serving memory.
Only small row slices and Engram projection weights are read from the checkpoint.
"""

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.environ import envs
from sglang.srt.layers.engram import Engram, EngramLayout
from sglang.srt.layers.engram_prefetch import EngramPrefetcher
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.runtime_context import reset_context
from sglang.test.test_utils import publish_build_topology


def load_layers(checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())
    text = SimpleNamespace(**config.get("text_config", config))
    quant = config["quantization_config"]
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    original = EngramLayout.from_config(text)
    columns = (text.engram_max_ngram_size - 1) * text.engram_n_heads
    per_column = 1024
    rows = columns * per_column
    layout = EngramLayout(
        original.max_ngram_size,
        original.layer_ids,
        (rows,) * len(original.layer_ids),
        tuple(
            ((per_column,) * original.n_heads,) * (original.max_ngram_size - 1)
            for _ in original.layer_ids
        ),
        original.n_heads,
        original.head_dim,
    )
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=quant["weight_block_size"],
        scale_fmt=quant["scale_fmt"],
    )
    layers = []
    with (
        envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.override(True),
        envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.override("per_rank"),
        set_default_torch_dtype(torch.bfloat16),
    ):
        for layer_id in layout.layer_ids:
            prefix = f"layers.{layer_id}.engram."
            with torch.device("cuda"):
                layer = Engram(text, layer_id, layout, quant_config)
            with safe_open(
                checkpoint / index[prefix + "wkv.weight"], framework="pt", device="cpu"
            ) as weights:
                layer.wkv.weight.data.copy_(weights.get_tensor(prefix + "wkv.weight"))
                layer.wkv.weight_scale_inv.data.copy_(
                    weights.get_tensor(prefix + "wkv.scale").float()
                )
                layer.q_weight.data.copy_(weights.get_tensor(prefix + "q_weight"))
                layer.k_weight.data.copy_(weights.get_tensor(prefix + "k_weight"))
                for name, target in (
                    ("embed.weight", layer.embed.weight),
                    ("embed.scale", layer.embed.scale),
                ):
                    source = weights.get_slice(prefix + name)
                    sampled = torch.cat(
                        [
                            source[start : start + per_column]
                            for start in original.column_bounds(layer.layer_hash_index)[
                                :-1
                            ]
                        ]
                    )
                    layer.embed._load_rows(target, sampled)
            layer.wkv.quant_method.process_weights_after_loading(layer.wkv)
            layers.append(layer)
    return text, layers, rows, columns


def error(actual, reference):
    delta = actual.float() - reference.float()
    return {
        "max_abs": delta.abs().max().item(),
        "relative_l2": (
            delta.norm() / reference.float().norm().clamp_min(1e-12)
        ).item(),
    }


@torch.inference_mode()
def check(checkpoint, output):
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(73)
    publish_build_topology(
        tp_size=1, disable_custom_all_reduce=True, fp8_gemm_runner_backend="triton"
    )
    initialize_fp8_gemm_config()
    with tempfile.NamedTemporaryFile() as rendezvous:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="file://" + rendezvous.name,
        )
        initialize_model_parallel()
        try:
            config, layers, rows, columns = load_layers(checkpoint)
            results = {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "checkpoint": str(checkpoint),
                "dimensions": [
                    columns * config.engram_head_dim,
                    config.hidden_size * (config.hc_mult + 1),
                ],
                "numerics": {},
                "memory": {},
            }

            def inputs(tokens):
                return (
                    torch.randn(
                        tokens,
                        config.hc_mult,
                        config.hidden_size,
                        device="cuda",
                        dtype=torch.bfloat16,
                    ),
                    torch.randint(
                        rows // columns, (tokens, len(layers), columns), device="cuda"
                    )
                    + torch.arange(columns, device="cuda") * (rows // columns),
                )

            def run(source, hashes, prefetch=None):
                if prefetch is not None:
                    prefetch.begin(hashes, None)
                for layer in layers:
                    source = layer(
                        source, hashes[:, layer.layer_hash_index], prefetch=prefetch
                    )
                return source

            prefetch = EngramPrefetcher(layers, 512, 1)
            for tokens in (16, 1025):
                source, hashes = inputs(tokens)
                checks = {}
                prefetch.begin(hashes, None)
                for layer in layers:
                    ids = hashes[:, layer.layer_hash_index]
                    reference = layer.embed(ids)
                    projected, _ = layer.wkv(reference.flatten(1))
                    pieces = []
                    for block in range(prefetch.num_chunks):
                        with prefetch.chunk(layer.layer_hash_index, block) as (
                            span,
                            values,
                        ):
                            torch.testing.assert_close(
                                values, reference[span], atol=0, rtol=0
                            )
                            value, _ = layer.wkv(values.flatten(1))
                            pieces.append(value)
                    prefetch.end_layer(layer.layer_hash_index)
                    actual = torch.cat(pieces)
                    torch.testing.assert_close(actual, projected, atol=0.02, rtol=0.02)
                    checks[str(layer.layer_hash_index)] = error(actual, projected)
                expected = run(source, hashes)
                actual = run(source, hashes, prefetch)
                torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
                checks["gate"] = error(actual, expected)
                results["numerics"][str(tokens)] = checks
            del (
                source,
                hashes,
                reference,
                projected,
                pieces,
                value,
                values,
                actual,
                expected,
                prefetch,
            )
            torch.cuda.synchronize()

            source, hashes = inputs(4096)
            for enabled in (False, True):
                label = "prefetch" if enabled else "synchronous"
                common = torch.cuda.memory_allocated()
                prefetch = EngramPrefetcher(layers, 512, 1) if enabled else None
                for _ in range(2):
                    run(source, hashes, prefetch)
                torch.cuda.synchronize()
                steady = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.memory._record_memory_history(max_entries=10000)
                try:
                    for _ in range(3):
                        result = run(source, hashes, prefetch)
                        del result
                        torch.cuda.synchronize()
                        assert torch.cuda.memory_allocated() == steady
                    results["memory"][label] = {
                        "staging_bytes": prefetch.buffer_bytes if enabled else 0,
                        "peak_increment_bytes": torch.cuda.max_memory_allocated()
                        - common,
                        "allocated_after_increment_bytes": steady - common,
                    }
                    torch.cuda.memory._dump_snapshot(output / f"{label}.pickle")
                finally:
                    torch.cuda.memory._record_memory_history(enabled=None)
                del prefetch
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(results, indent=2))
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
            reset_context()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/engram-check"))
    args = parser.parse_args()
    check(args.checkpoint, args.output_dir)
