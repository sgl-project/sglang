"""Device dispatch regression without requiring torch or an accelerator.

Execute the production dispatch methods with a tensor double whose is_cuda is
True on NPU, as torch_npu.transfer_to_npu makes it. CUDA kernels are sentinels;
these tests check dispatch, not kernel numerics.
"""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4] / "python/sglang"
TORCH = NS(bfloat16="bf16", float16="fp16")


class NativeFallback(Exception):
    pass


class Tensor:
    is_cuda = True
    dtype = "bf16"

    def __init__(self, device="npu", shape=(8, 10240), tag="native"):
        self.device = NS(type=device)
        self.shape = shape
        self.tag = tag
        self.data = self

    def float(self):
        raise NativeFallback

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return True

    def to(self, dtype):
        return self.tag


def function(path, name, cls=None):
    tree = ast.parse((ROOT / path).read_text())
    body = (
        next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls).body
        if cls
        else tree.body
    )
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace = {
        "torch": TORCH,
        "_is_npu": True,
        "_log_path_once": lambda *a: None,
        "fused_hc_mix_supported": lambda *a: False,
        "_deterministic_inference": lambda: False,
        "_FUSED_MIX_MAX_ROWS": 16,
    }
    exec(compile(module, str(ROOT / path), "exec"), namespace)
    return namespace[name]


class TestDeviceDispatch(CustomTestCase):
    def test_npu_gr_constructor_does_not_compile(self):
        path = ROOT / "srt/layers/hyperconnection.py"
        tree = ast.parse(path.read_text())
        node = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "GatedResidual"
        )
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                node,
            ],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)

        class Base:
            def __init__(self, config, *args):
                self.config = config
                self.hc_count = config.hc_count
                self.hidden_size = config.hidden_size
                self.params_dtype = config.params_dtype

        config = NS(
            hc_count=4,
            hidden_size=2560,
            hc_per_branch_norm=True,
            hc_lowrank=320,
            params_dtype="bf16",
            rms_norm_eps=1e-6,
        )
        for npu in (True, False):
            compiler = Mock(side_effect=lambda fn: NS(_torchdynamo_orig_callable=fn))
            namespace = {
                "HyperConnectionBase": Base,
                "_is_npu": npu,
                "GroupedGemmaRMSNorm": lambda *a, **kw: object(),
                "nn": NS(
                    Linear=lambda *a, **kw: NS(weight=Tensor("npu" if npu else "cuda"))
                ),
                "torch": NS(
                    compile=compiler,
                    cuda=NS(is_available=lambda: False),
                    get_device_module=lambda: NS(current_device=lambda: 0),
                ),
            }
            exec(compile(module, str(path), "exec"), namespace)
            instance = namespace["GatedResidual"](config)
            with self.subTest(npu=npu):
                self.assertEqual(compiler.call_count, 0 if npu else 2)
                if npu:
                    self.assertTrue(callable(instance._mix_compute))
                    self.assertTrue(callable(instance._combine_compute))

    def test_grouped_norm_ignores_cuda_alias_on_npu(self):
        module = "sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm"
        kernel = NS(grouped_gemma_rmsnorm=lambda *a: "cuda")
        for path, cls in (
            ("srt/layers/hyperconnection.py", "GroupedGemmaRMSNorm"),
            ("srt/models/qwen4_exp.py", "Qwen4ExpPLEGroupedNorm"),
        ):
            forward = function(path, "forward", cls)
            layer = NS(
                _jit_group_size=2560, weight=object(), variance_epsilon=1e-6, eps=1e-6
            )
            with patch.dict(sys.modules, {module: kernel}):
                for device in ("npu", "cpu"):
                    with (
                        self.subTest(cls=cls, device=device),
                        self.assertRaises(NativeFallback),
                    ):
                        forward(layer, Tensor(device))
                self.assertEqual(forward(layer, Tensor("cuda")), "cuda")

    def test_combine_ignores_cuda_alias_on_npu(self):
        forward = function("srt/layers/hyperconnection.py", "combine", "GatedResidual")
        kernel = NS(hc_combine=lambda *a: "cuda", hc_combine_split=lambda *a: "cuda")
        layer = NS(
            hc_count=4,
            hidden_size=2560,
            params_dtype="bf16",
            _jit_combine_ok=True,
            _split_combine_ok=True,
            block_inject_weight=NS(weight=Tensor()),
            _combine_compute=lambda *a: Tensor(tag="native"),
        )
        with patch.dict(
            sys.modules, {"sglang.kernels.ops.elementwise.hc_combine": kernel}
        ):
            for device in ("npu", "cuda"):
                result = forward(
                    layer, Tensor(device, (8, 2560)), (Tensor(device), Tensor(device))
                )
                self.assertEqual(result, "cuda" if device == "cuda" else "native")

    def test_mix_ignores_cuda_alias_on_npu(self):
        forward = function("srt/layers/hyperconnection.py", "mix", "GatedResidual")
        kernel = NS(
            hc_mix=lambda *a: Tensor(tag="cuda"), permute_pad_up_weight=lambda *a: None
        )
        weight = Tensor()
        weight.data = weight
        layer = NS(
            hc_count=4,
            hidden_size=2560,
            params_dtype="bf16",
            config=NS(hc_per_branch_norm=True),
            hc_norm=lambda x: x,
            _jit_mix_ok=True,
            _mix_up_weight_padded=object(),
            input_mix_weight_down=NS(weight=weight),
            input_mix_weight_up=NS(weight=weight),
            _mix_compute=lambda *a: Tensor(tag="native"),
        )
        with patch.dict(sys.modules, {"sglang.kernels.ops.elementwise.hc_mix": kernel}):
            for device in ("npu", "cuda"):
                result, _ = forward(layer, Tensor(device))
                self.assertEqual(result, "cuda" if device == "cuda" else "native")

    def test_persistent_mix_rejects_npu_alias(self):
        supported = function("kernels/ops/gemm/hc_mix.py", "fused_hc_mix_supported")
        for device in ("npu", "cpu", "cuda"):
            self.assertEqual(
                supported(Tensor(device), Tensor(device), Tensor(device)),
                device == "cuda",
            )

    def test_ple_fusion_rejects_npu_before_cuda_contract_checks(self):
        # Non-CUDA inputs must short-circuit before inspecting dtype/shape.
        class AliasedNPU:
            is_cuda = True
            device = NS(type="npu")

            @property
            def dtype(self):
                raise AssertionError("NPU entered CUDA fusion contract")

        for name in (
            "ngram_hash",
            "gate_value",
            "short_conv_state",
            "gate_reduce",
            "verify_conv",
        ):
            paths = {
                "ngram_hash": "embeddings/qwen4_ngram.py",
                "gate_value": "elementwise/qwen4_gate.py",
                "short_conv_state": "mamba/qwen4_short_conv.py",
                "gate_reduce": "elementwise/qwen4_gate.py",
                "verify_conv": "mamba/qwen4_short_conv.py",
            }
            f = function("kernels/ops/" + paths[name], "can_fuse_qwen4_" + name)
            with self.subTest(name=name):
                self.assertFalse(
                    f(*[AliasedNPU() for _ in range(f.__code__.co_argcount)])
                )


if __name__ == "__main__":
    unittest.main()
