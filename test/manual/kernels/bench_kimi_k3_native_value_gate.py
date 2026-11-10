"""Native K3 projection/gate consumer against the FI public API on synthetic rank-local tensors.

This is a source-backed consumer-segment experiment, not a model forward or a
production native FI selection. Native BF16 and diagnostic FP32 rounding are
reported separately. No attention/DCP/HiCache operation is invoked.
"""

import argparse
import importlib.metadata as metadata
import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import flashinfer
import torch

import sglang
from sglang.kernels.ops import attention as attention_ops
from sglang.kernels.ops.elementwise import mla_output_gate
from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA
from sglang.srt.models.kimi_k3 import KimiK3MLAAttention

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--torch-memory-limit-mib", type=int, default=256)
parser.add_argument("--flashinfer-revision", default="unknown")
args = parser.parse_args()
args.output.parent.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
assert torch.cuda.device_count() == 1 and torch.cuda.get_device_capability() == (10, 3)
torch.cuda.set_per_process_memory_fraction(
    args.torch_memory_limit_mib
    * 1024**2
    / torch.cuda.get_device_properties(0).total_memory
)


def version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


receipt = {
    "scope": __doc__,
    "sglang": sglang.__file__,
    "native_class_module": __import__(
        KimiK3MLAAttention.__module__, fromlist=["x"]
    ).__file__,
    "flashinfer_loaded": {
        "file": flashinfer.__file__,
        "version": flashinfer.__version__,
    },
    "installed_distributions": {
        n: version(n)
        for n in (
            "torch",
            "nvidia-cutlass-dsl",
            "apache-tvm-ffi",
            "flashinfer-python",
            "flashinfer-cubin",
            "flashinfer-jit-cache",
        )
    },
    "flashinfer_revision": args.flashinfer_revision,
    "checks": [],
}
generator = torch.Generator(device="cuda").manual_seed(48510)


def rand(*shape, scale=0.05):
    return (torch.randn(*shape, generator=generator, device="cuda") * scale).bfloat16()


def errors(got, ref, mask):
    delta = (got[mask].float() - ref[mask].float()).abs()
    return {
        "maximum_absolute": delta.max().item(),
        "mean_absolute": delta.mean().item(),
        "different_elements": int((got[mask] != ref[mask]).sum()),
        "elements": int(delta.numel()),
        "reference_maximum_absolute": ref[mask].float().abs().max().item(),
    }


def timing(call):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(3):
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        begin.record()
        for _ in range(30):
            call()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / 30)
    return {
        "unit": "microseconds_per_complete_segment",
        "samples": samples,
        "median": statistics.median(samples),
    }


def check_native_rounding_mode(weight, rows):
    from flashinfer.cake_mla_value_gate import prepare_mla_value_gate

    try:
        prepare_mla_value_gate(weight, max_rows=rows, sigmoid_precision="bf16")
    except ValueError as error:
        return {"requested": "bf16", "supported": False, "error": str(error)}
    return {"requested": "bf16", "supported": True}


try:
    gate_weight = rand(1536, 7168)
    for rows in (8, 32):
        hidden = rand(rows, 7168)
        latent = rand(rows, 12, 512)
        weight = rand(12, 128, 512).transpose(1, 2)
        lengths = torch.full((rows,), 128, dtype=torch.int32, device="cuda")
        lengths[-1] = 0
        lengths[-2] = -1
        gate_calls = [0]

        def g_proj(value):
            gate_calls[0] += 1
            return torch.nn.functional.linear(value, gate_weight), None

        owner = SimpleNamespace(
            w_vc=weight,
            num_local_heads=12,
            v_head_dim=128,
            use_deep_gemm_bmm=False,
            _gate_alt_stream=torch.cuda.Stream(),
            _gate_bs_limit=128,
            g_proj=g_proj,
        )

        def prepare_gate(value):
            KimiK3MLAAttention._fork_output_gate(owner, value)

        def get_gate(value):
            return KimiK3MLAAttention._compute_output_gate(owner, value)

        runner = attention_ops.prepare_cake_mla_value_gate(
            weight, max_rows=rows, rows_per_request=1
        )
        receipt.setdefault("native_rounding_api", []).append(
            check_native_rounding_mode(weight, rows)
        )
        output = torch.empty((rows, 1536), dtype=torch.bfloat16, device="cuda")

        def native(precision="bf16"):
            value = hidden * 0.5
            prepare_gate(value)
            projected = DeepseekV2AttentionMLA.project_mla_values(owner, latent, None)
            gate = get_gate(value)
            if precision == "bf16":
                return mla_output_gate.kimi_k3_mla_output_gate(projected, gate)
            return (projected.float() * torch.sigmoid(gate.float())).bfloat16()

        def fi():
            value = hidden * 0.5
            prepare_gate(value)
            gate = get_gate(value)
            return runner.launch(latent, gate, lengths, output)

        # Actual production helpers choose the capture-only gate side stream.
        with (
            patch(
                "sglang.srt.models.kimi_k3.get_is_capture_mode",
                side_effect=torch.cuda.is_current_stream_capturing,
            ),
            patch(
                "sglang.srt.models.kimi_k3.is_in_breakable_cuda_graph",
                return_value=False,
            ),
        ):
            native()
            fi()
            torch.cuda.synchronize()
            gate_calls[0] = 0
            fi()
            assert gate_calls[0] == 1, (
                "gate projection repeated in one consumer segment"
            )
            graphs = {}
            for name, call in [
                ("native_bf16", native),
                ("native_fp32_diagnostic", lambda: native("fp32")),
                ("fi_fp32", fi),
            ]:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    out = call()
                graphs[name] = (g, out)
            check = {
                "rows": rows,
                "heads": 12,
                "latent_width": 512,
                "value_width": 128,
                "gate_input_width": 7168,
                "weight_stride": list(weight.stride()),
                "gate_stride": [1536, 1],
                "program": runner.program,
                "normal_production_methods": True,
                "gate_projection_calls_per_eager_segment": 1,
                "changed_input_graph": [],
            }
            for iteration in range(3):
                hidden.copy_(rand(rows, 7168))
                hidden[-2:] = float("nan")
                latent.copy_(rand(rows, 12, 512))
                lengths.fill_(64 + iteration)
                lengths[-1] = 0
                lengths[-2] = -1
                expected_native = native().clone()
                expected_fp32 = native("fp32").clone()
                expected_fi = fi().clone()
                for g, out in graphs.values():
                    g.replay()
                torch.cuda.synchronize()
                mask = lengths > 0
                assert (
                    torch.isfinite(expected_fi).all()
                    and torch.isfinite(expected_native[mask]).all()
                )
                for name, ref in [
                    ("native_bf16", expected_native),
                    ("native_fp32_diagnostic", expected_fp32),
                    ("fi_fp32", expected_fi),
                ]:
                    torch.testing.assert_close(
                        graphs[name][1], ref, rtol=0, atol=0, equal_nan=True
                    )
                assert torch.equal(
                    expected_fi[~mask], torch.zeros_like(expected_fi[~mask])
                )
                # Separate native rounding from projection/reduction differences.
                check["changed_input_graph"].append(
                    {
                        "iteration": iteration,
                        "fi_vs_native_bf16": errors(expected_fi, expected_native, mask),
                        "fi_vs_native_fp32": errors(expected_fi, expected_fp32, mask),
                        "native_fp32_vs_bf16": errors(
                            expected_fp32, expected_native, mask
                        ),
                    }
                )
                # Component tolerance, never a task/logit-accuracy claim.
                torch.testing.assert_close(
                    expected_fi[mask], expected_fp32[mask], rtol=1 / 64, atol=1 / 1024
                )
            gate = get_gate(hidden * 0.5)
            double_gated = mla_output_gate.kimi_k3_mla_output_gate(output, gate)
            assert not torch.equal(double_gated[lengths > 0], output[lengths > 0]), (
                "double-gate negative control must differ"
            )
            check["ordinary_native_gate_applied_after_fi"] = (
                "double-gate negative control differs; a native integration must bypass it"
            )
            # Changed W_UV storage requires another real public preparation.
            owner.w_vc = weight = rand(12, 128, 512).transpose(1, 2)
            runner = attention_ops.prepare_cake_mla_value_gate(
                weight, max_rows=rows, rows_per_request=1
            )
            rebound = fi().clone()
            ref = native("fp32")
            torch.cuda.synchronize()
            torch.testing.assert_close(
                rebound[lengths > 0], ref[lengths > 0], rtol=1 / 64, atol=1 / 1024
            )
            check["new_weight_storage_new_preparation"] = "passed outside graph capture"
            # Timings use matched finite inputs and all-valid rows in every arm.
            hidden.copy_(rand(rows, 7168))
            latent.copy_(rand(rows, 12, 512))
            lengths.fill_(128)
            check["timing_inputs"] = {
                "finite": True,
                "positive_lengths": True,
                "valid_rows": rows,
                "eager_mode": "complete Python-dispatched segment, gate on current stream",
                "graph_mode": "captured segment replay, actual capture-only gate side stream",
                "warmups": 5,
                "repeats": 30,
                "samples": 3,
                "shared_device": True,
            }
            rebound_graphs = {}
            for name, call in [
                ("native_bf16", native),
                ("native_fp32_diagnostic", lambda: native("fp32")),
                ("fi_fp32", fi),
            ]:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    call()
                rebound_graphs[name] = g
            check["timing_eager"] = {
                name: timing(call)
                for name, call in [
                    ("native_bf16", native),
                    ("native_fp32_diagnostic", lambda: native("fp32")),
                    ("fi_fp32", fi),
                ]
            }
            check["timing_graph"] = {
                name: timing(g.replay) for name, g in rebound_graphs.items()
            }
            receipt["checks"].append(check)
    receipt["status"] = "passed"
except BaseException as error:
    import traceback

    traceback.print_exc()
    receipt["status"] = "failed"
    receipt["error"] = type(error).__name__ + ": " + str(error)
finally:
    receipt["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    receipt["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)
raise SystemExit(0 if receipt["status"] == "passed" else 1)
