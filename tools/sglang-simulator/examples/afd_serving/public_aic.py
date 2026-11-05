"""Reproduce the four conditional SM103 Direct cases with public AIC costs."""

import argparse
import json
import time
from pathlib import Path

from sglang_simulator.afd.adapters import create_adapter
from sglang_simulator.afd.runner import simulate
from sglang_simulator.aic_compat import AICSDK


def analytical(key, s):
    """Declared demo assumptions, not calibrated hardware measurements."""
    bw, link, launch = 6.4e12, 900e9, 3000
    if key == "direct_pack":
        payload = s["rows"] * (s["hidden"] * 2 + s["topk"] * 12)
        ns, eq = 6 * launch + 2 * payload / bw * 1e9, "6*3000 + 2*payload/6.4e12*1e9"
    elif key == "direct_combine":
        ns = 2 * launch + s["rows"] * s["hidden"] * (s["F"] * 2 + 10) / bw * 1e9
        eq = "2*3000 + rows*hidden*(F*2+10)/6.4e12*1e9"
    elif key == "direct_group_startup":
        ns, eq = s["alpha_ns"], "alpha_ns once per grouped exchange"
    elif key in ("direct_A2F_message", "direct_F2A_message"):
        ns, eq = s["bytes"] / s["bandwidth_Bps"] * 1e9, "bytes/bandwidth_Bps*1e9"
    elif key in ("lm_hidden_gather", "logits_exchange"):
        size = (
            s["global_bytes"] / s["world"]
            if key == "lm_hidden_gather"
            else s["per_rank_input_bytes"]
        )
        ns = (s["world"] - 1) * (10000 + size / link * 1e9)
        eq = "(world-1)*(10000+chunk_bytes/900e9*1e9)"
    elif key in ("graph_replay_overhead", "ffn_step_submit", "trt_metadata"):
        ns = 10000 if key == "trt_metadata" else 5000
        eq = "declared host budget: 10000ns metadata, 5000ns replay/submit per actor/step"
    elif key in {
        "input_norm_first",
        "input_norm_residual",
        "post_attention_norm",
        "qk_norm_rope",
        "topk",
        "embedding",
        "final_norm",
        "logits_postprocess",
        "sampling",
        "stage_join",
    }:
        ns = (
            launch
            + s.get("rows", 1) * s.get("hidden", s.get("vocab", 4096)) * 6 / bw * 1e9
        )
        eq = "3000 + rows*width*6/6.4e12*1e9; width=hidden or vocab, otherwise 4096"
    else:
        raise ValueError(f"No declared formula for {key}")
    return dict(
        shape=s,
        duration_ns=ns,
        basis="formula",
        equation=eq,
        source="Explicit unmeasured demo assumption; no scale/offset fit",
    )


def cost(sdk, db, systems, key, s):
    if key in ("qkv", "o", "router", "lm_head"):
        kind = "gemm"
        args = dict(
            m=s["m"],
            n=s["n"],
            k=s["k"],
            quant="fp8_static" if key in ("qkv", "o") else "bfloat16",
        )
    elif key == "trtllm_decode":
        kind = "generation_attention"
        args = dict(
            b=s["rows"],
            s=s["effective_kv"],
            n=64,
            n_kv=4,
            head_size=128,
            quant="fp8",
            lane_order=["trtllm_mha"],
        )
    elif key == "routed_experts_local_combine":
        kind = "moe"
        args = dict(
            num_tokens=s["rows"],
            hidden_size=4096,
            inter_size=1536,
            topk=8,
            num_experts=128,
            moe_tp_size=1,
            moe_ep_size=s["ep"],
            quant="fp8_block",
            workload_distribution="balanced",
            is_context=False,
            moe_backend="DEEPGEMM",
        )
    else:
        return analytical(key, s)
    quote = sdk.query(db, kind, systems_path=str(systems), **args)
    source = str(quote["sdk_source"]).lower()
    if source not in ("silicon", "sol", "empirical", "estimated"):
        raise ValueError(f"Unclassified SDK source: {source}")
    record = dict(shape=s, duration_ns=quote["latency_ms"] * 1e6, source=quote)
    if source == "silicon":
        # The SDK does not expose exact-hit/neighbor evidence here. Retain the
        # lookup provenance, and report measured/interpolated only as a group.
        record.update(
            basis="interpolation",
            neighbors={
                "lookup": "public SDK silicon lookup; exact-hit/neighbor detail unavailable"
            },
        )
    else:
        record.update(
            basis="formula", equation=f"public SDK analytical estimate ({source})"
        )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--systems-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter = create_adapter(args.source_root, "afd", routing="direct")
    adapter.verify()
    sdk = AICSDK("core")
    db = sdk.module("perf_database").get_database(
        "gb300",
        "sglang",
        "0.5.14",
        systems_paths=[str(args.systems_path)],
        database_mode="SILICON",
        shared_layer=False,
    )
    if db is None or db.version != "0.5.14":
        raise ValueError("The demo requires SGLang data 0.5.14")
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for kv, batch in ((8448, 256), (8448, 512), (65792, 256), (40320, 256)):
        start = time.perf_counter()
        config = adapter.template_config(batch_size=batch, effective_kv=kv, ffn_ranks=8)
        mapping = adapter.cost_mapping(config)
        document = dict(
            schema_version=1,
            fixture=False,
            deployment=config,
            costs={
                k: cost(sdk, db, args.systems_path, k, s)
                for k, s in mapping.expected_shapes().items()
            },
        )
        stem = f"kv{kv}-b{batch}"
        (args.output / f"{stem}-costs.json").write_text(
            json.dumps(document, indent=2) + "\n"
        )
        result = simulate(document, adapter=adapter)
        (args.output / f"{stem}-result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        row = dict(
            effective_kv=kv,
            batch=batch,
            step_ms=result["step_ms"],
            token_s_per_device=result["per_device_tgs"],
            wall_s=time.perf_counter() - start,
        )
        summary.append(row)
        print(json.dumps(row), flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
