import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGDNChunkOutputDispatch(CustomTestCase):
    def test_mapped_output_bypasses_dense_only_grouped_kernel(self):
        """A gfx115-eligible shape must still honor mapped destinations and buffers."""
        source = (
            Path(__file__).resolve().parents[4]
            / "python/sglang/kernels/ops/attention/fla/chunk_o.py"
        )
        tree = ast.parse(source.read_text())
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_use_grouped_three_head", "chunk_fwd_o")
        ]
        self.assertEqual(len(functions), 2)
        module = ast.Module(body=functions, type_ignores=[])
        dtype = object()
        dense_output = object()
        generic_launch = Mock()
        grouped_launch = Mock()
        generic_kernel = MagicMock()
        generic_kernel.__getitem__ = Mock(return_value=generic_launch)
        grouped_kernel = MagicMock()
        grouped_kernel.__getitem__ = Mock(return_value=grouped_launch)
        namespace = {
            "Optional": Optional,
            "torch": SimpleNamespace(
                Tensor=object,
                LongTensor=object,
                bfloat16=dtype,
                zeros_like=Mock(return_value=dense_output),
            ),
            "triton": SimpleNamespace(
                cdiv=lambda n, d: (n + d - 1) // d,
                next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
            ),
            "use_gfx115_kernels": Mock(return_value=True),
            "prepare_chunk_indices": Mock(side_effect=AssertionError("rebuilt table")),
            "chunk_fwd_kernel_o": generic_kernel,
            "chunk_fwd_kernel_o_grouped_three_head": grouped_kernel,
        }
        exec(compile(module, str(source), "exec"), namespace)
        q = SimpleNamespace(shape=(1, 128, 16, 128), dtype=dtype)
        v = SimpleNamespace(shape=(1, 128, 48, 128), dtype=dtype)
        h = SimpleNamespace(shape=(1, 2, 48, 128, 128), dtype=dtype)
        g = SimpleNamespace(shape=(1, 128, 48))
        cu_seqlens = object()
        chunks = [(0, 0), (0, 1)]
        args = dict(
            q=q,
            k=q,
            v=v,
            h=h,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunks,
        )
        call = namespace["chunk_fwd_o"]
        self.assertIs(call(**args), dense_output)
        grouped_launch.assert_called_once()
        generic_launch.assert_not_called()
        self.assertIs(grouped_launch.call_args.args[7], chunks)

        mapped_output, indices = object(), object()
        self.assertIs(
            call(**args, out=mapped_output, output_indices=indices), mapped_output
        )
        generic_launch.assert_called_once()
        grouped_launch.assert_called_once()
        self.assertIs(generic_launch.call_args.args[5], mapped_output)
        self.assertIs(generic_launch.call_args.args[6], cu_seqlens)
        self.assertIs(generic_launch.call_args.args[7], chunks)
        self.assertIs(generic_launch.call_args.kwargs["output_indices"], indices)
        self.assertTrue(generic_launch.call_args.kwargs["HAS_OUTPUT_INDICES"])
        self.assertEqual(namespace["torch"].zeros_like.call_count, 1)
        self.assertEqual(
            generic_kernel.__getitem__.call_args.args[0]({"BV": 64}), (2, 2, 48)
        )

        namespace["use_gfx115_kernels"].return_value = False
        self.assertIs(call(**args), dense_output)
        self.assertIsNone(generic_launch.call_args.kwargs["output_indices"])
        self.assertFalse(generic_launch.call_args.kwargs["HAS_OUTPUT_INDICES"])
        grouped_launch.assert_called_once()

        for invalid in ({"out": mapped_output}, {"output_indices": indices}):
            with self.subTest(arguments=tuple(invalid)):
                with self.assertRaisesRegex(ValueError, "supplied together"):
                    call(**args, **invalid)


if __name__ == "__main__":
    unittest.main()
