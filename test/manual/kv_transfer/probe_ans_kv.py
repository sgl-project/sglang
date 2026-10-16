"""Compare lossless backends on in-memory Qwen3 BF16 KV; never export KV.

This Transformers capture is an algorithm probe, not SGLang serving evidence.
The online validator must separately exercise the installed SGLang attention
backend, P/D protocol, L2 reuse and target writeback.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path


def measure(layout, mode, repeats):
    import torch

    from sglang.srt.kv_compression.backend import BACKENDS
    from sglang.srt.kv_compression.types import align_bytes

    stream = torch.cuda.Stream(device=layout.device)
    stream.wait_stream(torch.cuda.current_stream(layout.device))
    backend = BACKENDS[mode](layout.device, stream)
    rows = []
    with torch.cuda.stream(stream):
        for start in range(0, len(layout.buffers[0]), 64):
            raw = layout.pack_pages(
                range(start, min(start + 64, len(layout.buffers[0])))
            )
            sources = list(raw.unbind())
            bounds = [backend.max_output_bytes(src) for src in sources]
            outputs = [
                torch.empty(n, device=layout.device, dtype=torch.uint8) for n in bounds
            ]
            restored = [torch.empty_like(src) for src in sources]
            stream.synchronize()
            for trial in range(repeats + 1):
                events = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                events[0].record(stream)
                started = time.perf_counter()
                lengths = backend.compress_batch(sources, outputs)
                encoded_seconds = time.perf_counter() - started
                events[1].record(stream)
                started = time.perf_counter()
                backend.decompress_batch(
                    [dst[:n] for dst, n in zip(outputs, lengths)], restored
                )
                decoded_seconds = time.perf_counter() - started
                events[2].record(stream)
                stream.synchronize()
                if not all(
                    torch.equal(src, dst) for src, dst in zip(sources, restored)
                ):
                    raise AssertionError(f"{mode} changed KV bytes at page {start}")
                if trial == 0:
                    continue  # Warm each allocation/window; excluded from results.
                rows.append(
                    dict(
                        start=start,
                        pages=len(sources),
                        trial=trial,
                        raw_bytes=raw.numel(),
                        actual_lengths=lengths,
                        output_bounds=bounds,
                        encoded_bytes=sum(lengths),
                        aligned_encoded_bytes=sum(align_bytes(n) for n in lengths),
                        normal_selected_bytes=sum(
                            align_bytes(n)
                            if n + 256 < layout.page_bytes
                            else layout.page_bytes
                            for n in lengths
                        ),
                        encode_wall_seconds=encoded_seconds,
                        decode_wall_seconds=decoded_seconds,
                        encode_gpu_ms=events[0].elapsed_time(events[1]),
                        decode_gpu_ms=events[1].elapsed_time(events[2]),
                        verified=True,
                    )
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Qwen3-8B model directory")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.tokens <= 8192:
        parser.error("--tokens must be between 1 and 8192")
    if args.output.exists():
        parser.error("Use an unused output path")
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sglang.srt.kv_compression.layout import KVLayoutAdapter
    from sglang.srt.kv_compression.types import NVCOMP_VERSION

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Bind exactly one CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = tokenizer.encode(args.prompt_file.read_text())
    if len(tokens) < args.tokens:
        raise ValueError("Prompt must contain enough tokens; no synthetic repetition")
    tokens = tokens[: args.tokens]
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda:0")
        .eval()
    )
    config = model.config
    if (
        config.architectures != ["Qwen3ForCausalLM"]
        or config.hidden_size != 4096
        or config.num_hidden_layers != 36
        or config.num_key_value_heads != 8
    ):
        raise ValueError("This probe is scoped to Qwen3-8B BF16")
    with torch.inference_mode():
        result = model.model(
            input_ids=torch.tensor([tokens], device="cuda:0"), use_cache=True
        )
        cache = result.past_key_values
        if hasattr(cache, "layers"):
            pairs = [(layer.keys, layer.values) for layer in cache.layers]
        elif hasattr(cache, "to_legacy_cache"):
            pairs = list(cache.to_legacy_cache())
        else:
            pairs = list(cache)
        # The same page,K-layers,V-layers representation used by the runtime.
        k = [key[0].transpose(0, 1).contiguous() for key, _ in pairs]
        v = [value[0].transpose(0, 1).contiguous() for _, value in pairs]
        layout = KVLayoutAdapter(k, v)
        if layout.page_bytes != 147456:
            raise ValueError("Unexpected Qwen3-8B KV page size")
        rows = {mode: measure(layout, mode, 3) for mode in ("lz4", "ans")}
    report = dict(
        model=args.model,
        tokens=len(tokens),
        input_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
        dtype="bf16",
        capture="Transformers SDPA; not SGLang serving",
        torch=torch.__version__,
        transformers=transformers.__version__,
        nvcomp=NVCOMP_VERSION,
        gpu=torch.cuda.get_device_name(0),
        page_bytes=layout.page_bytes,
        batch_pages=64,
        repeats=3,
        warmup_per_window=1,
        raw_baseline_bytes=len(tokens) * layout.page_bytes,
        results=rows,
        timing="Packing and verification excluded; wall times include API synchronization",
        boundary="No RDMA, protocol descriptors, L2, serving latency or model-quality claim",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")
    print(f"Verified KV bytes for both backends. Aggregate report: {args.output}")


if __name__ == "__main__":
    main()
