"""Create a synthetic fixed-cohort HTTP control case, not performance data."""

import argparse
import json
from pathlib import Path

from sglang_simulator.afd.adapters import create_adapter
from sglang_simulator.afd.costs import template
from sglang_simulator.afd.native_config import NativeAFDConfig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ffn-backend", choices=("fixed", "direct"), default="direct")
    p.add_argument("--rendezvous-port", type=int, default=1239)
    a = p.parse_args()
    if a.output.exists():
        p.error("output must be new")
    native = dict(
        stages=2,
        lanes=8,
        attention_lanes=8,
        ffn_backend=a.ffn_backend,
        attention_backend="trtllm_mha",
        rendezvous_host="127.0.0.1",
        rendezvous_port=a.rendezvous_port,
    )
    resolved = NativeAFDConfig.parse(native)
    adapter = create_adapter(a.source_root, "afd", routing=resolved.routing)
    adapter.verify()
    a.output.mkdir(parents=True)
    catalog = dict(
        schema_version=1,
        decode=[],
        prefill=[
            dict(
                requests=[[4, 0]] * 16,
                duration_ns=1000000,
                basis="fixture",
                source="Synthetic complete-batch prefill, not measured",
            )
        ],
    )
    for kv in (5, 6):
        c = adapter.template_config(batch_size=16, effective_kv=kv, ffn_ranks=8)
        costs = template(c, adapter.cost_mapping(c))
        name = f"decode-B16-KV{kv}.json"
        (a.output / name).write_text(json.dumps(costs, indent=2) + "\n")
        catalog["decode"].append(name)
    (a.output / "catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    (a.output / "native.json").write_text(json.dumps(native, indent=2) + "\n")
    sim = dict(
        platform=dict(
            accelerator=dict(name="a100_sxm"),
            num_device_per_node=8,
            disk_read_bandwidth_gb=8,
            disk_write_bandwidth_gb=8,
            memory_read_bandwidth_gb=64,
            memory_write_bandwidth_gb=64,
        ),
        scheduler=dict(tp_size=8, dp_size=8, ep_size=8, backend_name="sglang"),
        predictor=dict(
            name="afd",
            source_root=str(a.source_root.resolve()),
            cost_catalog="catalog.json",
            event_output_dir="f-events",
            allow_fixture=True,
        ),
    )
    (a.output / "simulation.json").write_text(json.dumps(sim, indent=2) + "\n")
    model = dict(
        architectures=["Qwen3MoeForCausalLM"],
        model_type="qwen3_moe",
        hidden_size=4096,
        num_hidden_layers=94,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=1536,
        intermediate_size=12288,
        max_position_embeddings=40960,
        vocab_size=151936,
        torch_dtype="bfloat16",
        bos_token_id=151643,
        eos_token_id=151645,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        rope_theta=1000000,
        tie_word_embeddings=False,
        use_cache=True,
    )
    (a.output / "model").mkdir()
    (a.output / "model/config.json").write_text(json.dumps(model, indent=2) + "\n")
    print("Synthetic assets: " + str(a.output.resolve()))


if __name__ == "__main__":
    main()
