"""Capture one SenseNova denoise step on NPU without a serving process."""

import argparse
import gzip
import json
import os
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, choices=[1, 2, 4], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument(
        "--same-prompt",
        action="store_true",
        help="Repeat the longer B2 prompt to remove padding while retaining its maximum prefix length.",
    )
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    # Register the repository's native model classes.
    import torch
    import torch_npu
    from transformers import AutoModel, AutoTokenizer
    from validate_npu_batching import PROMPTS

    import sglang.multimodal_gen.runtime.models.sensenova_u1  # noqa: F401

    torch.npu.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = (
        AutoModel.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, local_files_only=True
        )
        .eval()
        .to("npu:0")
    )
    kwargs = dict(
        image_size=(args.size, args.size),
        batch_size=args.batch_size,
        num_steps=3,
        cfg_scale=4.0,
        timestep_shift=3.0,
        think_mode=False,
        seed=[1000 + i for i in range(args.batch_size)],
    )
    prompts = (
        [PROMPTS[1]] * args.batch_size
        if args.same_prompt
        else PROMPTS[: args.batch_size]
    )
    print(f"Prompt mode: {'same-long' if args.same_prompt else 'mixed'}", flush=True)
    print("Warmup: 3 steps", flush=True)
    with torch.inference_mode():
        warmup = model.t2i_generate(tokenizer, prompts, **kwargs)
    del warmup
    torch.npu.synchronize()
    original = model.extract_feature
    calls = 0
    active = False
    finished = False
    profiler = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=True,
        with_stack=False,
        profile_memory=False,
    )

    def extract(*a, **kw):
        nonlocal calls, active, finished
        calls += 1
        if calls == 2:
            torch.npu.synchronize()
            profiler.start()
            active = True
        elif calls == 3:
            torch.npu.synchronize()
            profiler.stop()
            active = False
            finished = True
        return original(*a, **kw)

    model.extract_feature = extract
    print("Capture: step 2 of 3", flush=True)
    try:
        with torch.inference_mode():
            result = model.t2i_generate(tokenizer, prompts, **kwargs)
        torch.npu.synchronize()
        if not torch.isfinite(result).all().item():
            raise RuntimeError("Generation returned nonfinite pixels")
    finally:
        model.extract_feature = original
        if active:
            torch.npu.synchronize()
            profiler.stop()
    if calls != 3 or not finished:
        raise RuntimeError(f"Expected exactly three denoise feature calls, got {calls}")
    # No tensorboard handler: keep CANN intermediate artifacts out of the result folder.
    with tempfile.TemporaryDirectory(prefix="sensenova-profile-") as temp:
        trace = Path(temp) / "trace.json"
        profiler.export_chrome_trace(str(trace))
        compressed = gzip.compress(trace.read_bytes(), compresslevel=9)
    metadata = json.dumps(
        dict(
            batch_size=args.batch_size,
            prompt_mode="same-long" if args.same_prompt else "mixed",
            prompts=prompts,
            size=args.size,
            steps=3,
            captured_step=2,
            cfg_scale=4.0,
            dtype="bfloat16",
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            model=args.model,
            optimizations={
                name: os.getenv(name, "1")
                for name in (
                    "SGLANG_SENSENOVA_NPU_FIA",
                    "SGLANG_SENSENOVA_NPU_FUSED_MLP",
                    "SGLANG_SENSENOVA_NPU_FUSED_NORM",
                )
            },
            compressed_trace_bytes=len(compressed),
            scope="one full denoise step; no prefill, HTTP or scheduler",
        ),
        indent=2,
    ).encode()
    if len(compressed) + len(metadata) > 14_000_000:
        raise RuntimeError(
            f"Complete compressed trace exceeds the per-run 14 MB budget ({len(compressed)} bytes). "
            "No partial trace saved. Retry with --size 1024 for BOTH B1 and B2."
        )
    (output / "trace.json.gz").write_bytes(compressed)
    (output / "metadata.json").write_bytes(metadata)
    print(f"Saved {len(compressed) + len(metadata)} bytes in {output}")


if __name__ == "__main__":
    main()
