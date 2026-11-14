"""CPU checks for row-wise MoE scheduling and query lookahead ownership."""

import ast
import builtins
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.models import deepseek_v4_moe_pipeline
from sglang.srt.models.deepseek_v4_moe_pipeline import (
    QueryLookahead,
    RowPipelineConfig,
    row_bounds,
    submit_row_pipeline,
    swizzle_scale_factors,
    take_prepared_attention,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRowPartitioning(unittest.TestCase):
    def test_default_configuration_preserves_ordinary_execution(self):
        config = RowPipelineConfig()
        self.assertEqual(config.microbatches, 0)
        self.assertFalse(config.packed_gather)
        self.assertFalse(config.epilogue)
        self.assertFalse(config.query_lookahead)

    def test_invalid_configuration_is_rejected(self):
        for options in (
            {"microbatches": -1},
            {"min_rows": 0},
            {"min_rows": -32},
            {"comm_priority": 1},
        ):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    RowPipelineConfig(**options)

    def test_full_chunks_use_the_expected_cp_local_rows(self):
        for tokens, local_rows, rows_per_microbatch in (
            (65536, 16384, 4096),
            (81920, 20480, 5120),
        ):
            with self.subTest(tokens=tokens):
                self.assertEqual(tokens // 4, local_rows)
                self.assertEqual(
                    row_bounds(local_rows, 4),
                    tuple(
                        (i * rows_per_microbatch, (i + 1) * rows_per_microbatch)
                        for i in range(4)
                    ),
                )

    def test_small_rows_do_not_create_empty_microbatches(self):
        self.assertEqual(row_bounds(32, 4), ((0, 32),))
        self.assertEqual(row_bounds(1, 8), ((0, 1),))
        self.assertEqual(row_bounds(768, 4), ((0, 256), (256, 512), (512, 768)))

    def test_uneven_rows_are_covered_once_and_in_order(self):
        for rows in (257, 767, 1025, 16385):
            with self.subTest(rows=rows):
                bounds = row_bounds(rows, 4)
                self.assertEqual(bounds[0][0], 0)
                self.assertEqual(bounds[-1][1], rows)
                self.assertTrue(all(a < b for a, b in bounds))
                self.assertTrue(
                    all(left[1] == right[0] for left, right in zip(bounds, bounds[1:]))
                )
                self.assertEqual(sum(b - a for a, b in bounds), rows)

    def test_invalid_partition_inputs_are_rejected(self):
        for rows, microbatches in ((-1, 4), (1024, 0), (1024, -1)):
            with self.subTest(rows=rows, microbatches=microbatches):
                with self.assertRaises(ValueError):
                    row_bounds(rows, microbatches)


class TestRowPipelineSubmission(unittest.TestCase):
    def test_collective_order_is_identical_for_all_ranks(self):
        calls, collectives = [], []

        def record(name):
            def call(i, value=None):
                calls.append((name, i))
                if name in ("gather", "scatter"):
                    collectives.append((name, i))
                if name == "compute":
                    return f"expert-output-{i}"
                if name == "scatter":
                    self.assertEqual(value, f"expert-output-{i}")

            return call

        submit_row_pipeline(
            4,
            **{
                name: record(name)
                for name in (
                    "prepare",
                    "gather",
                    "compute",
                    "scatter",
                    "shared",
                    "finish",
                )
            },
        )
        self.assertEqual(
            collectives,
            [
                ("gather", 0),
                ("gather", 1),
                ("scatter", 0),
                ("gather", 2),
                ("scatter", 1),
                ("gather", 3),
                ("scatter", 2),
                ("scatter", 3),
            ],
        )
        finish_at = calls.index(("finish", 0))
        self.assertTrue(
            all(
                calls.index((name, i)) < finish_at
                for name in ("compute", "shared")
                for i in range(4)
            )
        )
        self.assertEqual(calls[finish_at:], [("finish", i) for i in range(4)])

    def test_single_microbatch_does_not_submit_a_second_gather(self):
        calls = []

        def callback(name):
            return lambda i, *args: calls.append((name, i))

        names = ("prepare", "gather", "compute", "scatter", "shared", "finish")
        submit_row_pipeline(1, **{name: callback(name) for name in names})
        self.assertEqual(calls, [(name, 0) for name in names])

    def test_row_pipeline_preserves_summed_experts_and_local_shared_once(self):
        # Four CP/EP ranks, uneven rows and nonuniform routing. Each rank owns
        # disjoint experts; only the selected rank's rows survive reduce-scatter.
        # Shared is replicated and must be added once after that sum.
        generator = torch.Generator().manual_seed(17)
        world, rank, rows, hidden, experts = 4, 1, 1025, 4, 8
        x = torch.randn(world, rows, hidden, generator=generator, dtype=torch.float64)
        weights = torch.randn(
            experts, hidden, hidden, generator=generator, dtype=torch.float64
        )
        shared_weight = torch.randn(
            hidden, hidden, generator=generator, dtype=torch.float64
        )
        q_weight = torch.randn(hidden, 3, generator=generator, dtype=torch.float64)
        ids = torch.arange(world * rows).reshape(world, rows) % experts

        def routed(value, route):
            partials = []
            for owner in range(world):
                partial = torch.zeros_like(value)
                for offset, scale in ((0, 0.6), (3, 0.4)):
                    selected = (route + offset) % experts
                    for expert in range(owner * 2, (owner + 1) * 2):
                        mask = selected == expert
                        partial[mask] += scale * torch.tanh(
                            value[mask] @ weights[expert]
                        )
                partials.append(partial)
            return torch.stack(partials)

        reference = routed(x.reshape(-1, hidden), ids.reshape(-1)).sum(0)
        reference = reference.reshape(world, rows, hidden)[rank]
        reference = torch.tanh(reference + x[rank] @ shared_weight + x[rank])
        reference_q = reference @ q_weight

        for count in (1, 4):
            with self.subTest(microbatches=count):
                bounds = row_bounds(rows, count)
                gathered, pending, shared = {}, {}, {}
                final = torch.full_like(x[rank], float("nan"))
                q = torch.full((rows, 3), float("nan"), dtype=torch.float64)
                closed = []

                def prepare(i):
                    a, b = bounds[i]
                    gathered[i] = (
                        x[:, a:b].reshape(-1, hidden),
                        ids[:, a:b].reshape(-1),
                    )

                def compute(i):
                    return routed(*gathered.pop(i))

                def scatter(i, partials):
                    pending[i] = partials

                def local_shared(i):
                    a, b = bounds[i]
                    shared[i] = x[rank, a:b] @ shared_weight

                def finish(i):
                    a, b = bounds[i]
                    # This is the CPU transport completion boundary. Pending
                    # later slices are not a prerequisite for this slice's Q.
                    reduced = pending.pop(i).sum(0).reshape(world, b - a, hidden)[rank]
                    final[a:b] = torch.tanh(reduced + shared.pop(i) + x[rank, a:b])
                    q[a:b] = final[a:b] @ q_weight
                    closed.append(i)

                submit_row_pipeline(
                    len(bounds),
                    prepare=prepare,
                    gather=lambda _: None,
                    compute=compute,
                    scatter=scatter,
                    shared=local_shared,
                    finish=finish,
                )
                torch.testing.assert_close(final, reference, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(q, reference_q, rtol=1e-12, atol=1e-12)
                self.assertEqual(closed, list(range(len(bounds))))
                self.assertFalse(gathered or pending or shared)


class TestQueryLookaheadOwnership(unittest.TestCase):
    @staticmethod
    def cache(batch, owner, positions):
        return QueryLookahead(
            batch=batch,
            owner=owner,
            positions=positions,
            input_rows=object(),
            qkv_a=object(),
            q_lora=object(),
            q=object(),
            read_state=object(),
        )

    def test_lookahead_cannot_cross_batch_layer_or_positions(self):
        batch, owner, positions = object(), object(), torch.arange(4)
        cache = self.cache(batch, owner, positions)
        self.assertTrue(cache.matches(batch, owner, positions))
        self.assertFalse(cache.matches(object(), owner, positions))
        self.assertFalse(cache.matches(batch, object(), positions))
        self.assertFalse(cache.matches(batch, owner, positions.clone()))

    def test_stale_lookahead_is_discarded_before_normal_prepare(self):
        batch, owner, positions = object(), object(), torch.arange(4)
        layer = SimpleNamespace(_query_lookahead=self.cache(batch, owner, positions))
        self.assertIsNone(
            take_prepared_attention(layer, object(), batch, positions, object())
        )
        self.assertIsNone(layer._query_lookahead)
        # Re-entering with the old matching identities cannot revive it.
        self.assertIsNone(
            take_prepared_attention(layer, owner, batch, positions, object())
        )


class TestScaleFactorLayout(unittest.TestCase):
    def test_scale_swizzle_preserves_bytes_and_zero_fills_padded_rows(self):
        rows, cols = 129, 5
        values = torch.randint(
            1,
            255,
            (rows, cols),
            generator=torch.Generator().manual_seed(23),
            dtype=torch.uint8,
        )
        actual = swizzle_scale_factors(values)
        reference = torch.zeros(256 * 8, dtype=torch.uint8)
        # Each 128x4 tile is stored as 32 row lanes x 4 row groups x 4 columns.
        # Address individual scales rather than repeating the reshape/permute.
        for row in range(rows):
            for col in range(cols):
                tile = (row // 128) * 2 + col // 4
                address = (
                    tile * 512 + (row % 32) * 16 + ((row % 128) // 32) * 4 + col % 4
                )
                reference[address] = values[row, col]
        self.assertEqual(actual.dtype, torch.uint8)
        self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


class TestDefaultOffDecoderIntegration(unittest.TestCase):
    def test_real_decoder_forward_keeps_baseline_boundary_protocol(self):
        # Execute the actual production method, without importing the model's
        # accelerator-only kernel stack. Boundary fakes perform arithmetic so
        # this checks both the protocol order and the returned value.
        source_path = Path(deepseek_v4_moe_pipeline.__file__).with_name(
            "deepseek_v4.py"
        )
        module = ast.parse(source_path.read_text())
        decoder = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef)
            and node.name == "DeepseekV4CudaDecoderLayer"
        )
        method = next(
            node
            for node in decoder.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        )

        for microbatches in (0, 1):
            with self.subTest(microbatches=microbatches):
                calls = []
                source = torch.arange(8.0).reshape(4, 2)
                positions = torch.arange(4)
                batch, ids, global_ids, quantized = (object() for _ in range(4))

                def layer_call(input_ids, input_ids_global, allow_cross_layer_fusion):
                    self.assertIs(input_ids, ids)
                    self.assertIs(input_ids_global, global_ids)
                    self.assertFalse(allow_cross_layer_fusion)
                    return SimpleNamespace()

                def written_residual(actual_batch):
                    self.assertIs(actual_batch, batch)
                    return SimpleNamespace(quantized=quantized)

                def no_pipeline_import(name, *args, **kwargs):
                    if name == "sglang.srt.models.deepseek_v4_moe_pipeline":
                        raise AssertionError(
                            "disabled decoder entered the row pipeline"
                        )
                    return builtins.__import__(name, *args, **kwargs)

                namespace = {
                    "hc_boundary": SimpleNamespace(LayerCall=layer_call),
                    "residual_batch": SimpleNamespace(
                        written_residual=written_residual
                    ),
                    "__builtins__": {
                        **vars(builtins),
                        "__import__": no_pipeline_import,
                    },
                }
                exec(
                    compile(
                        ast.Module(body=[method], type_ignores=[]),
                        str(source_path),
                        "exec",
                    ),
                    namespace,
                )

                def attn_prepare(value, actual_batch, *, read_context):
                    self.assertIs(actual_batch, batch)
                    calls.append("attention.prepare")
                    return value + 1

                def attention(value, actual_positions, actual_batch, actual_quantized):
                    self.assertIs(actual_positions, positions)
                    self.assertIs(actual_batch, batch)
                    self.assertIs(actual_quantized, quantized)
                    calls.append("attention.compute")
                    return value * 2

                def attn_finish(value, actual_batch):
                    self.assertIs(actual_batch, batch)
                    calls.append("attention.finish")
                    return value + 3

                def ffn_prepare(value, actual_batch, *, read_context):
                    self.assertIs(actual_batch, batch)
                    calls.append("ffn.prepare")
                    return value - 2

                def mlp(value, actual_batch, **kwargs):
                    self.assertIs(actual_batch, batch)
                    self.assertEqual(
                        kwargs,
                        {
                            "input_ids": ids,
                            "input_ids_global": global_ids,
                            "skip_shared_experts": True,
                            "return_moe_output": True,
                            "leave_output_sum": True,
                        },
                    )
                    calls.append("moe.compute")
                    return value * 5

                def ffn_finish(value):
                    calls.append("ffn.finish")
                    return value + 7

                transport = SimpleNamespace(
                    input_ids=ids,
                    input_ids_global=global_ids,
                    do_shared_local=True,
                    return_moe_output=True,
                )
                layer = SimpleNamespace(
                    _row_pipeline_config=RowPipelineConfig(microbatches=microbatches),
                    attn_boundary=SimpleNamespace(
                        prepare=attn_prepare, finish=attn_finish
                    ),
                    self_attn=SimpleNamespace(
                        maybe_use_decode_attn_tp=lambda _: nullcontext()
                    ),
                    _run_boundary_attention=attention,
                    ffn_boundary=SimpleNamespace(
                        prepare=ffn_prepare,
                        transport_context=lambda _: transport,
                        exit=lambda _: nullcontext(SimpleNamespace(finish=ffn_finish)),
                    ),
                    mlp=mlp,
                )
                actual = namespace["forward"](
                    layer,
                    positions,
                    source,
                    ids,
                    batch,
                    global_ids,
                    allow_cross_layer_fusion=False,
                    query_lookahead_layer=object(),
                )
                self.assertEqual(
                    calls,
                    [
                        "attention.prepare",
                        "attention.compute",
                        "attention.finish",
                        "ffn.prepare",
                        "moe.compute",
                        "ffn.finish",
                    ],
                )
                torch.testing.assert_close(actual, (((source + 1) * 2 + 3) - 2) * 5 + 7)


if __name__ == "__main__":
    unittest.main()
