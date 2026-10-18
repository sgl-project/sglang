"""CPU contract tests for real FIA dispatch and Ascend950 short-query layouts."""

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
            "forward_decode",
            "_is_swa_layer",
            "_can_use_tnd",
            "_fia_with_sink",
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

    def test_ascend950_short_prefill_with_prefix_and_padding(self):
        for window in (-1, 2):
            with self.subTest(window=window):
                self.check_dispatch(
                    torch.bfloat16,
                    window,
                    True,
                    is_ascend950=True,
                    q_lens=(6, 0, 1),
                    q_heads=64,
                    kv_heads=8,
                    padding=3,
                )

    def check_dispatch(
        self,
        dtype,
        window,
        with_sink,
        *,
        is_ascend950=False,
        q_lens=(2, 0, 3),
        q_heads=4,
        kv_heads=2,
        padding=0,
    ):
        methods = load_dispatch_methods()
        layer = SimpleNamespace(
            layer_id=3,
            qk_head_dim=64,
            v_head_dim=64,
            tp_q_head_num=q_heads,
            tp_k_head_num=kv_heads,
            tp_v_head_num=kv_heads,
            sliding_window_size=window,
            scaling=0.125,
        )
        real_tokens = sum(q_lens)
        q = (
            torch.arange((real_tokens + padding) * q_heads * 64)
            .reshape(real_tokens + padding, q_heads, 64)
            .to(dtype)
        )
        sinks = torch.arange(q_heads).to(dtype) if with_sink else None
        key = torch.arange(9 * 2 * kv_heads * 64).reshape(9, 2, kv_heads * 64).to(dtype)
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
            extend_seq_lens_cpu=list(q_lens),
            seq_lens=torch.tensor([length + 3 for length in q_lens]),
        )
        mask = torch.triu(torch.ones(16, 16, dtype=torch.bool), diagonal=1)
        backend = SimpleNamespace(
            use_mla=False,
            is_dllm_model=False,
            use_fia=True,
            _is_ascend950=is_ascend950,
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
        backend._fia_with_sink = MethodType(methods["_fia_with_sink"], backend)
        self.assertFalse(backend._can_use_tnd(layer))
        calls = []
        requests = [(0, 0, q_lens[0]), (2, q_lens[0], q_lens[2])]

        def fia(**kwargs):
            seq_idx, offset, q_len = requests[len(calls)]
            calls.append(seq_idx)
            workaround = is_ascend950 and q_heads == 8 * kv_heads and q_len < 8
            self.assertEqual(
                kwargs["input_layout"], "BSND_BNSD" if workaround else "BSND"
            )
            full = methods["FULL_ATTENTION_WINDOW"]
            self.assertEqual(kwargs["pre_tokens"], full if window == -1 else window)
            self.assertEqual(kwargs["next_tokens"], full if window == -1 else 0)
            self.assertEqual(kwargs["sparse_mode"], 3 if window == -1 else 4)
            self.assertIs(kwargs.get("learnable_sink"), sinks)
            self.assertEqual(kwargs["actual_seq_qlen"], [q_len])
            self.assertEqual(kwargs["actual_seq_kvlen"], [int(batch.seq_lens[seq_idx])])
            self.assertEqual(kwargs["block_size"], 2)
            self.assertEqual(kwargs["num_query_heads"], q_heads)
            self.assertEqual(kwargs["num_key_value_heads"], kv_heads)
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
            result = kwargs["query"] + seq_idx
            if workaround:
                result = result.transpose(1, 2).contiguous()
            return result, None

        methods["forward_extend"].__globals__["torch_npu"] = SimpleNamespace(
            npu_fused_infer_attention_score_v2=fia
        )
        output = methods["forward_extend"](
            backend, q, None, None, layer, batch, save_kv_cache=False, sinks=sinks
        )
        expected = q[:real_tokens].clone()
        expected[q_lens[0] :] += 2
        self.assertEqual(calls, [0, 2])
        self.assertEqual(output.dtype, dtype)
        self.assertEqual(output.shape, (real_tokens + padding, q_heads * 64))
        self.assertTrue(
            torch.equal(output[:real_tokens], expected.reshape(real_tokens, -1))
        )

    def test_ascend950_decode_preserves_batch_pages_and_sink(self):
        for window in (-1, 127):
            with self.subTest(window=window):
                methods = load_dispatch_methods()
                layer = SimpleNamespace(
                    layer_id=3,
                    qk_head_dim=64,
                    v_head_dim=64,
                    tp_q_head_num=64,
                    tp_k_head_num=8,
                    tp_v_head_num=8,
                    sliding_window_size=window,
                    scaling=0.125,
                )
                q = torch.arange(2 * 64 * 64).reshape(2, 64, 64).to(torch.bfloat16)
                key = torch.ones(3, 128, 512, dtype=q.dtype)
                value = -key
                sinks = torch.ones(64, dtype=q.dtype)
                tables = torch.tensor([[1, 2], [2, 1]], dtype=torch.int32)
                swa_tables = tables.flip(1)
                seq_lens = torch.tensor([5, 200])
                mask = torch.ones(16, 16, dtype=torch.bool)
                backend = SimpleNamespace(
                    _is_ascend950=True,
                    use_mla=False,
                    use_fia=True,
                    graph_mode=False,
                    is_hybrid_swa=True,
                    page_size=128,
                    fia_mask=mask,
                    forward_metadata=SimpleNamespace(
                        seq_lens=seq_lens,
                        seq_lens_cpu_int=seq_lens,
                        block_tables=tables,
                        block_tables_swa=swa_tables,
                    ),
                    token_to_kv_pool=SimpleNamespace(
                        get_key_buffer=lambda _: key,
                        get_value_buffer=lambda _: value,
                    ),
                )
                backend._is_swa_layer = MethodType(methods["_is_swa_layer"], backend)
                backend._fia_with_sink = MethodType(methods["_fia_with_sink"], backend)
                calls = []

                def fia(**kwargs):
                    calls.append(kwargs)
                    self.assertEqual(kwargs["input_layout"], "BSND_BNSD")
                    self.assertEqual(kwargs["actual_seq_qlen"], [1, 1])
                    self.assertEqual(kwargs["actual_seq_kvlen"], [5, 200])
                    self.assertEqual(kwargs["pre_tokens"], window)
                    self.assertEqual(kwargs["next_tokens"], 0)
                    self.assertEqual(kwargs["sparse_mode"], 0 if window == -1 else 4)
                    self.assertIs(kwargs["learnable_sink"], sinks)
                    self.assertIs(kwargs["atten_mask"], None if window == -1 else mask)
                    self.assertIs(
                        kwargs["block_table"], tables if window == -1 else swa_tables
                    )
                    self.assertTrue(torch.equal(kwargs["query"], q.unsqueeze(1)))
                    return kwargs["query"].transpose(1, 2).contiguous(), None

                methods["forward_decode"].__globals__["torch_npu"] = SimpleNamespace(
                    npu_fused_infer_attention_score_v2=fia
                )
                output = methods["forward_decode"](
                    backend,
                    q,
                    None,
                    None,
                    layer,
                    SimpleNamespace(batch_size=2),
                    save_kv_cache=False,
                    sinks=sinks,
                )
                self.assertEqual(len(calls), 1)
                self.assertTrue(torch.equal(output, q.reshape(2, -1)))


class TestFiaShortQueryLayout(unittest.TestCase):
    def test_short_query_gate(self):
        cases = (
            ({}, True),
            ({"q_len": 1}, True),
            ({"q_len": 7}, True),
            ({"q_len": 8}, False),
            ({"q_len": 128}, False),
            ({"is_ascend950": False}, False),
            ({"with_sink": False}, False),
            ({"dtype": torch.float16}, False),
            ({"dim": 128}, False),
            ({"q_heads": 32}, False),
            ({"kv_dtype": torch.int8}, False),
            ({"sink_dtype": torch.float32}, False),
        )
        for options, expected_workaround in cases:
            with self.subTest(options=options):
                self.check_helper(expected_workaround=expected_workaround, **options)

    def test_tnd_request_lengths_not_total_tokens(self):
        for lengths in (list(range(1, 129)), [2, 2, 5], torch.tensor([2, 2, 5])):
            with self.subTest(lengths=lengths):
                self.check_helper(
                    layout="TND",
                    q_len=int(lengths[-1]),
                    lengths=lengths,
                    expected_workaround=True,
                )
        for lengths in ([8], [1, 9], [0, 0], [128]):
            with self.subTest(lengths=lengths):
                self.check_helper(
                    layout="TND",
                    q_len=lengths[-1],
                    lengths=lengths,
                    expected_workaround=False,
                )

    def check_helper(
        self,
        *,
        expected_workaround,
        layout="BSND",
        q_len=6,
        lengths=None,
        is_ascend950=True,
        with_sink=True,
        dtype=torch.bfloat16,
        dim=64,
        q_heads=64,
        kv_dtype=None,
        sink_dtype=None,
    ):
        methods = load_dispatch_methods()
        backend = SimpleNamespace(_is_ascend950=is_ascend950)
        query = (
            torch.arange(q_len * q_heads * dim).reshape(q_len, q_heads, dim).to(dtype)
        )
        if layout == "BSND":
            query = query.unsqueeze(0)
        key = torch.ones(2, 128, 8 * dim, dtype=kv_dtype or dtype)
        value = -key
        sink = torch.ones(q_heads, dtype=sink_dtype or dtype) if with_sink else None
        lengths = [q_len] if lengths is None else lengths
        kwargs = dict(
            num_query_heads=q_heads,
            num_key_value_heads=8,
            actual_seq_qlen=lengths,
            actual_seq_kvlen=[int(lengths[-1]) + 128] * len(lengths),
            block_table=torch.ones(len(lengths), 2, dtype=torch.int32),
            block_size=128,
            sparse_mode=4,
            pre_tokens=127,
            next_tokens=0,
            softmax_scale=0.125,
            atten_mask=torch.ones(16, 16, dtype=torch.bool),
        )
        calls = []
        lse = object()

        def fia(*args, **actual):
            self.assertFalse(args)
            calls.append(actual["input_layout"])
            for name, expected in kwargs.items():
                self.assertIs(actual[name], expected)
            self.assertIs(actual["key"], key)
            self.assertIs(actual["value"], value)
            self.assertIs(actual["learnable_sink"], sink)
            if expected_workaround and layout == "TND":
                self.assertEqual(actual["input_layout"], "NTD_TND")
                self.assertTrue(actual["query"].is_contiguous())
                self.assertTrue(torch.equal(actual["query"], query.transpose(0, 1)))
                output = actual["query"].transpose(0, 1).contiguous()
            else:
                self.assertIs(actual["query"], query)
                output = query
                if expected_workaround:
                    self.assertEqual(actual["input_layout"], "BSND_BNSD")
                    output = output.transpose(1, 2).contiguous()
                else:
                    self.assertEqual(actual["input_layout"], layout)
            return output, lse

        methods["_fia_with_sink"].__globals__["torch_npu"] = SimpleNamespace(
            npu_fused_infer_attention_score_v2=fia
        )
        output, output_lse = methods["_fia_with_sink"](
            backend,
            query,
            key,
            value,
            input_layout=layout,
            learnable_sink=sink,
            **kwargs,
        )
        self.assertEqual(len(calls), 1)
        self.assertIs(output_lse, lse)
        self.assertEqual(output.shape, query.shape)
        self.assertEqual(output.dtype, query.dtype)
        self.assertTrue(output.is_contiguous())
        self.assertTrue(torch.equal(output, query))


if __name__ == "__main__":
    unittest.main()
