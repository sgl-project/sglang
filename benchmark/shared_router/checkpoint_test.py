# SPDX-License-Identifier: Apache-2.0
"""Optional real-checkpoint qualification; activations here are synthetic.

Load only shared MLP/router tensors, never the full model. TP4 slices match the
column-parallel gate/up and row-parallel down loaders. This is complementary to,
not a replacement for, model-level GSM8K and production dispatch traces.
"""

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open


def tensor_digest(tensor):
    data = tensor.contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 18, 36])
    args = parser.parse_args()
    ref_path = args.repo / "test/registered/amd/test_shared_router_gfx950.py"
    spec = importlib.util.spec_from_file_location("shared_router_reference", ref_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    index = json.loads((args.model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    manifest = {
        "model_path": str(args.model),
        "activations": "synthetic BF16 standard normal, not captured serving inputs",
        "tp": 4,
        "device": torch.cuda.get_device_name(),
        "checkpoint_sha256": {},
        "results": [],
    }

    def load(name):
        with safe_open(
            str(args.model / index[name]), framework="pt", device="cpu"
        ) as f:
            value = f.get_tensor(name)
        manifest["checkpoint_sha256"][name] = tensor_digest(value)
        if name.endswith(".scale"):
            # The checkpoint stores UE8M0 bytes. Avoid relying on CPU arithmetic
            # support for torch.float8_e8m0fnu when decoding exact powers of two.
            assert value.dtype == torch.float8_e8m0fnu
            exponent = value.view(torch.uint8).to(device="cuda", dtype=torch.int32)
            assert bool((exponent != 255).all()), "nonfinite checkpoint scale"
            return (exponent << 23).view(torch.float32)
        return value.to("cuda")

    for layer in args.layers:
        prefix = f"layers.{layer}.ffn."
        gate = load(prefix + "shared_experts.w1.weight")
        up = load(prefix + "shared_experts.w3.weight")
        down = load(prefix + "shared_experts.w2.weight")
        gs = load(prefix + "shared_experts.w1.scale")
        us = load(prefix + "shared_experts.w3.scale")
        ds = load(prefix + "shared_experts.w2.scale")
        router = load(prefix + "gate.weight").to(torch.bfloat16).contiguous()
        bias = load(prefix + "gate.bias").to(torch.bfloat16).contiguous()
        assert gate.shape == up.shape == (2304, 5120)
        assert down.shape == (5120, 2304)
        assert gs.shape == us.shape == (72, 160) and ds.shape == (160, 72)
        for rank in range(4):
            lo, hi = rank * 576, (rank + 1) * 576
            # Concatenate the local gate and up slices, not a slice of the
            # already-concatenated global tensor. Each projection is TP-sharded.
            local_gate = torch.cat(
                [gate[lo:hi].view(torch.uint8), up[lo:hi].view(torch.uint8)]
            ).view(torch.float8_e4m3fn)
            local_gs = torch.cat([gs[lo // 32 : hi // 32], us[lo // 32 : hi // 32]])
            sw, ss = ref.prepare_mxfp8_native_weight(local_gate, local_gs, [32, 32])
            local_down = down[:, lo:hi].contiguous()
            local_ds = ds[:, lo // 32 : hi // 32].contiguous()
            packed_down = ref.prepare_mxfp8_native_weight(
                local_down, local_ds, [32, 32]
            )
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            unpacked = ref.unpack_shared_down(
                packed_down[0].view(torch.float8_e4m3fn), packed_down[1]
            )
            torch.cuda.synchronize()
            unpack_ms = 1000 * (time.perf_counter() - start)
            allocator_delta_bytes = torch.cuda.memory_allocated() - before
            resident_bytes = unpacked.numel() * unpacked.element_size()
            peak_extra_bytes = torch.cuda.max_memory_allocated() - before
            expected = (
                local_down.float()
                * local_ds.repeat_interleave(32, 0).repeat_interleave(32, 1)
            ).to(torch.bfloat16)
            torch.testing.assert_close(unpacked, expected, rtol=0, atol=0)
            # Allocator deltas include block rounding and can include deferred
            # releases. Count the retained tensor directly; report allocator
            # observations separately rather than using them as a numeric gate.
            for m in (6, 12):
                torch.manual_seed(1000 * layer + 10 * rank + m)
                x = torch.randn((m, 5120), dtype=torch.bfloat16, device="cuda")
                inputs = (x, sw, ss, router, unpacked, bias)
                actual = ref.shared_router(*inputs)
                native = ref.native(inputs, packed_down)
                ref.compare(actual, native)
                relative_rms = (
                    (actual[0].float() - native[0].float()).square().mean().sqrt()
                    / native[0].float().square().mean().sqrt().clamp_min(1e-6)
                ).item()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    ref.shared_router(*inputs)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = ref.shared_router(*inputs)
                for _ in range(3):
                    x.normal_()
                    graph.replay()
                    ref.compare(captured, ref.native(inputs, packed_down))
                manifest["results"].append(
                    dict(
                        layer=layer,
                        rank=rank,
                        m=m,
                        relative_rms=relative_rms,
                        exact_topk=True,
                        graph_changed_input_replays=3,
                        unpack_ms=unpack_ms,
                        derived_resident_bytes=resident_bytes,
                        allocator_delta_bytes=allocator_delta_bytes,
                        unpack_peak_extra_bytes=peak_extra_bytes,
                    )
                )
                print(json.dumps(manifest["results"][-1]), flush=True)
                del graph, captured, actual, native
            del unpacked, expected, packed_down
    manifest["status"] = "passed"
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
