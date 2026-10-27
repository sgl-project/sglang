# SPDX-License-Identifier: Apache-2.0
"""Compare Ascend FA and MindIE-SD attention on identical floating-point Q/K/V.

Run from an installed SGLang environment with Ascend 950 and MindIE-SD.
The A5 / D=64,128 comparison matrix is not an Ascend FA runtime limit.
Prints JSON containing native dispatch, latency, memory and error measurements.
"""

import argparse
import json
import statistics
import time
from collections import Counter
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, choices=(64, 128), default=128)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sd-head-chunk-size", type=int, default=4)
    args = parser.parse_args()
    if (
        min(args.batch, args.seq_len, args.heads, args.iterations) < 1
        or args.warmup < 1
    ):
        parser.error("shape dimensions, warmup and iterations must be positive")
    if args.sd_head_chunk_size < 0:
        parser.error("--sd-head-chunk-size must be non-negative; 0 disables chunking")
    if args.seq_len % 64:
        parser.error(
            "--seq-len must be divisible by 64 so each batch boundary is aligned "
            "for the existing Ascend FA MXFP8 baseline"
        )

    import mindiesd  # Load custom OPP paths before any NPU initialization.
    import torch
    import torch_npu
    from torch.utils._python_dispatch import TorchDispatchMode

    from sglang.multimodal_gen import envs
    from sglang.multimodal_gen.runtime.layers.attention.backends.ascend_fa import (
        AscendFAImpl,
    )
    from sglang.multimodal_gen.runtime.layers.quantization.mxfp8 import MXFP8Config
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args

    torch.npu.set_device(args.device)
    if torch.npu.get_soc_version() < 260:
        raise RuntimeError("This benchmark requires Ascend 950 (A5)")
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cpu").manual_seed(42)
    query, key, value = (
        torch.randn(
            args.batch,
            args.seq_len,
            args.heads,
            args.head_dim,
            generator=generator,
            dtype=dtype,
        ).to("npu")
        for _ in range(3)
    )
    scale = args.head_dim**-0.5
    set_global_server_args(SimpleNamespace(attention_backend_config={}))

    class DispatchTrace(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = Counter()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = func._schema.name
            if "attention" in name or "quant" in name or "flash_attn" in name:
                self.calls[name] += 1
            return func(*args, **(kwargs or {}))

    def measure(name, impl, expected_operator, reference):
        def run():
            return impl.forward(query, key, value, None)

        with DispatchTrace() as trace:
            output = run()
        if expected_operator not in trace.calls:
            raise RuntimeError(
                f"{name} did not execute {expected_operator}: {dict(trace.calls)}"
            )
        if not torch.isfinite(output).all():
            raise RuntimeError(f"{name} returned non-finite values")
        sd_executor = impl._sd_executor
        if sd_executor is not None:
            matrix = sd_executor._get_rotation(query)
            options = dict(
                precision=sd_executor.precision,
                layout="BSND",
                scale=scale,
                q_rot=matrix,
                k_rot=matrix,
            )
            if sd_executor.precision == "fp8":
                options["fp8_fa_mode"] = sd_executor.fp8_fa_mode
                direct = torch.cat(
                    [
                        mindiesd.quant_attention(
                            query[i : i + 1],
                            key[i : i + 1],
                            value[i : i + 1],
                            **options,
                        )
                        for i in range(args.batch)
                    ]
                )
            else:
                direct = mindiesd.quant_attention(query, key, value, **options)
            torch.testing.assert_close(output, direct)
            del direct
        actual = output.float().cpu()
        if reference is None:
            reference = actual
        difference = actual - reference
        metrics = {
            "provider": "mindiesd" if sd_executor is not None else "native",
            "precision": (
                sd_executor.precision
                if sd_executor is not None
                else "mxfp8"
                if impl._quant_scheme
                else "float"
            ),
            "qk_rotation": (
                sd_executor.qk_rotation
                if sd_executor is not None
                else bool(impl._quant_scheme and not impl.use_offline_qk_rotation)
            ),
            "head_chunk_size": (
                sd_executor.head_chunk_size
                if sd_executor is not None
                else impl._mxfp8_head_chunk_size
                if impl._quant_scheme
                else 0
            ),
            "relative_rmse": (
                difference.norm() / reference.norm().clamp_min(1e-12)
            ).item(),
            "max_abs_error": difference.abs().max().item(),
            "cosine_similarity": torch.nn.functional.cosine_similarity(
                actual.flatten(), reference.flatten(), dim=0
            ).item(),
        }
        del output
        for _ in range(args.warmup):
            run()
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            run()
            torch.npu.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        print(
            json.dumps(
                {
                    "case": name,
                    "native_calls": dict(trace.calls),
                    "median_ms": statistics.median(samples),
                    "min_ms": min(samples),
                    "peak_allocated_mib": torch.npu.max_memory_allocated() / 2**20,
                    **metrics,
                }
            ),
            flush=True,
        )
        return reference

    print(
        json.dumps(
            {
                "configuration": vars(args),
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "mindiesd": getattr(mindiesd, "__version__", "unknown"),
                "device": torch.npu.get_device_name(args.device),
                "rotation": "CPU FP32 QR, seed=42; Q and K share R",
            }
        ),
        flush=True,
    )
    with torch.no_grad():
        envs.SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION = False
        reference = measure(
            "ascend_fa_float",
            AscendFAImpl(args.heads, args.head_dim, False, scale),
            "npu::npu_fused_infer_attention_score",
            None,
        )
        envs.SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION = True
        for chunk in (0, 4):
            envs.SGLANG_DIFFUSION_MXFP8_FA_HEAD_CHUNK_SIZE = chunk
            impl = AscendFAImpl(
                args.heads, args.head_dim, False, scale, quant_config=MXFP8Config()
            )
            if impl._quant_scheme != "MXFP8" or impl._sd_executor is not None:
                raise RuntimeError(
                    "Native MXFP8 was not enabled; refusing a mislabeled baseline"
                )
            measure(
                f"ascend_fa_mxfp8_chunk{chunk}",
                impl,
                "npu::npu_fused_infer_attention_score_v2",
                reference,
            )
        for precision in ("fp8", "mxfp8", "mxfp4"):
            modes = (
                ["HIGH_PRECISION", "C8V16_TILING512"]
                if precision == "fp8" and args.head_dim == 128
                else ["HIGH_PRECISION"]
            )
            for mode in modes:
                for rotation in (True, False):
                    config = dict(
                        precision=precision,
                        qk_rotation=rotation,
                        head_chunk_size=args.sd_head_chunk_size,
                    )
                    if precision == "fp8":
                        config["fp8_fa_mode"] = mode
                    set_global_server_args(
                        SimpleNamespace(attention_backend_config=config)
                    )
                    impl = AscendFAImpl(args.heads, args.head_dim, False, scale)
                    assert impl._sd_executor is not None
                    expected_operator = {
                        "fp8": "mindiesd::fused_infer_attention_score_v2",
                        "mxfp8": "npu::npu_fused_infer_attention_score_v2",
                        "mxfp4": "mindiesd::quant_flash_attn",
                    }[precision]
                    measure(
                        f"sd_{precision}_{mode if precision == 'fp8' else 'default'}"
                        f"_rotation{int(rotation)}_chunk{args.sd_head_chunk_size}",
                        impl,
                        expected_operator,
                        reference,
                    )


if __name__ == "__main__":
    main()
