"""CPU contract tests for the real FIA BSND prefill dispatch method."""

import ast
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def load_dispatch_methods():
    # Executing the actual methods avoids importing torch_npu or probing hardware.
    source = (
        Path(__file__).resolve().parents[5]
        / "python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py"
    )
    tree = ast.parse(source.read_text())
    backend = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendAttnBackend"
    )
    nodes = [
        ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0)
    ]
    nodes += [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "FULL_ATTENTION_WINDOW"
            for target in node.targets
        )
    ]
    for node in backend.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "forward_extend",
            "_is_swa_layer",
            "_can_use_tnd",
        ):
            node.decorator_list = []
            nodes.append(node)
    scope = {"torch": torch, "is_mla_preprocess_enabled": lambda: False}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(source),
            "exec",
        ),
        scope,
    )
    return scope


class TestFiaBsndPrefill(unittest.TestCase):
    def test_full_and_sliding_paged_prefill(self):
        for dtype in (torch.bfloat16, torch.float16):
            for window, with_sink in ((-1, True), (2, True), (2, False)):
                with self.subTest(dtype=dtype, window=window, sink=with_sink):
                    self.check_dispatch(dtype, window, with_sink)

    def check_dispatch(self, dtype, window, with_sink):
        methods = load_dispatch_methods()
        layer = SimpleNamespace(
            layer_id=3,
            qk_head_dim=64,
            v_head_dim=64,
            tp_q_head_num=4,
            tp_k_head_num=2,
            tp_v_head_num=2,
            sliding_window_size=window,
            scaling=0.125,
        )
        q = torch.arange(5 * 4 * 64).reshape(5, 4, 64).to(dtype)
        sinks = torch.tensor([-2.0, -1.0, 0.0, 1.0], dtype=dtype) if with_sink else None
        key = torch.arange(9 * 2 * 128).reshape(9, 2, 128).to(dtype)
        value = -key
        tables = torch.tensor([[3, 1, 7], [6, 4, 2], [8, 0, 5]], dtype=torch.int32)
        swa_tables = tables.flip(1)
        selected_tables = swa_tables if window != -1 else tables
        metadata = SimpleNamespace(block_tables=tables, block_tables_swa=swa_tables)
        mode = SimpleNamespace(
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
            is_context_parallel_extend=lambda: False,
        )
        # The middle request has cached KV but no new Q; the last request must
        # retain its block-table row while Q offsets skip the empty request.
        batch = SimpleNamespace(
            forward_mode=mode,
            extend_seq_lens_cpu=[2, 0, 3],
            seq_lens=torch.tensor([5, 4, 4]),
        )
        mask = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
        backend = SimpleNamespace(
            use_mla=False,
            is_dllm_model=False,
            use_fia=True,
            is_hybrid_swa=True,
            page_size=2,
            forward_metadata=metadata,
            fia_mask=mask,
            token_to_kv_pool=SimpleNamespace(
                get_key_buffer=lambda _: key,
                get_value_buffer=lambda _: value,
            ),
        )
        backend._is_swa_layer = MethodType(methods["_is_swa_layer"], backend)
        backend._can_use_tnd = methods["_can_use_tnd"]
        self.assertFalse(backend._can_use_tnd(layer))
        calls = []

        def fia(**kwargs):
            seq_idx, offset, q_len = ((0, 0, 2), (2, 2, 3))[len(calls)]
            calls.append(seq_idx)
            self.assertEqual(kwargs["input_layout"], "BSND")
            full = methods["FULL_ATTENTION_WINDOW"]
            self.assertEqual(kwargs["pre_tokens"], full if window == -1 else window)
            self.assertEqual(kwargs["next_tokens"], full if window == -1 else 0)
            self.assertEqual(kwargs["sparse_mode"], 3 if window == -1 else 4)
            self.assertIs(kwargs.get("learnable_sink"), sinks)
            self.assertEqual(kwargs["actual_seq_qlen"], [q_len])
            self.assertEqual(kwargs["actual_seq_kvlen"], [int(batch.seq_lens[seq_idx])])
            self.assertEqual(kwargs["block_size"], 2)
            self.assertEqual(kwargs["num_query_heads"], 4)
            self.assertEqual(kwargs["num_key_value_heads"], 2)
            self.assertEqual(kwargs["softmax_scale"], layer.scaling)
            self.assertTrue(
                torch.equal(
                    kwargs["block_table"], selected_tables[seq_idx : seq_idx + 1]
                )
            )
            self.assertTrue(torch.equal(kwargs["query"][0], q[offset : offset + q_len]))
            self.assertTrue(torch.equal(kwargs["key"], key))
            self.assertTrue(torch.equal(kwargs["value"], value))
            self.assertTrue(torch.equal(kwargs["atten_mask"], mask.unsqueeze(0)))
            # Distinct per-request results detect incorrect output offsets.
            return kwargs["query"] + seq_idx, None

        methods["forward_extend"].__globals__["torch_npu"] = SimpleNamespace(
            npu_fused_infer_attention_score_v2=fia
        )
        output = methods["forward_extend"](
            backend, q, None, None, layer, batch, save_kv_cache=False, sinks=sinks
        )
        expected = q.clone()
        expected[2:] += 2
        self.assertEqual(calls, [0, 2])
        self.assertEqual(output.dtype, dtype)
        self.assertEqual(output.shape, (5, 4 * 64))
        self.assertTrue(torch.equal(output, expected.reshape(5, -1)))


if __name__ == "__main__":
    unittest.main()
