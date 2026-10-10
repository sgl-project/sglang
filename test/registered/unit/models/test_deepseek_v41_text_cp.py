"""Pure-language V4.1 CP input, padding and DSpark state regressions."""

import unittest
from contextlib import contextmanager, nullcontext
from itertools import product
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.cp.utils import (
    cp_gather_after_forward,
    cp_shard_model_inputs,
    is_cp_active,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM, MQALayer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_cp_test_utils import cp_context, simulated_collective
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


RUNNER = "sglang.srt.model_executor.runner.eager_runner"


class TestDSV41TextCP(CustomTestCase):
    def test_cp_multistream_q_starts_before_gathers_and_is_joined(self):
        """Q must depend on q_lora, not the CP gathers or compressor. The
        cache writer must still wait for gathered KV, and attention must see
        completed Q. Emulate stream frontiers to detect missing dependencies.
        """
        module = "sglang.srt.models.deepseek_v4"
        for captured, compressor, fused, encoder_replay, has_q_out in product(
            (False, True), repeat=5
        ):
            with self.subTest(
                captured=captured,
                compressor=compressor,
                fused=fused,
                encoder_replay=encoder_replay,
                has_q_out=has_q_out,
            ):
                self._check_early_cp_q(
                    module, captured, compressor, fused, encoder_replay, has_q_out
                )

    def _check_early_cp_q(
        self, module, captured, compressor, fused, encoder_replay, has_q_out
    ):
        class Stream:
            def __init__(self):
                self.ready = set()

            def wait_stream(self, source):
                self.ready.update(source.ready)

        parent = Stream()
        workers = [Stream() for _ in range(3)]
        active = [parent]
        events = {}

        @contextmanager
        def use_stream(stream):
            previous, active[0] = active[0], stream
            try:
                yield
            finally:
                active[0] = previous

        def enqueue(name, value, *inputs):
            ready = active[0].ready
            for operand in inputs:
                self.assertIn(operand.name, ready, f"{name} reads unfinished input")
            events[name] = (active[0], ready.copy())
            ready.add(name)
            return NS(name=name, value=value)

        x = NS(name="input", value=2)
        x.contiguous = lambda: x
        parent.ready.add(x.name)
        positions = 0
        q_out = NS(value=-1) if has_q_out else None

        def compute_q_a(hidden, qkv_a=None):
            q = enqueue("q_a", 2 * hidden.value, qkv_a or hidden)
            return q, q

        def compute_q_b(q, pos, out):
            self.assertIs(pos, positions)
            self.assertIs(out, q_out)
            result = enqueue("q_b", 3 * q.value, q)
            if out is None:
                return result
            out.name, out.value = result.name, result.value
            return out

        def sources(**kwargs):
            inputs = [kwargs["precomputed_x_global"] or x]
            if kwargs["q_lora"] is not None:
                inputs.append(kwargs["q_lora"])
            return enqueue("sources", 5, *inputs)

        def gather_index_k(layer):
            # With no compressor this layer borrows an already-written cache.
            inputs = (NS(name="sources"),) if compressor else ()
            return enqueue("index_k", 7, *inputs)

        def topk(layer, q, w, *, prepared_q, prepared_dense_k):
            enqueue("topk", 0, prepared_q, prepared_dense_k, w)

        layer = NS(
            alt_streams=workers,
            fuse_wqa_wkv=fused,
            wqkv_a=lambda hidden: (enqueue("qkv_a", hidden.value, hidden), None),
            compressor=object() if compressor else None,
            indexer=NS(
                queries=lambda q, freqs: enqueue("index_q", q.value, q),
                head_weights=lambda hidden: enqueue("index_w", hidden.value, hidden),
            ),
            freqs_cis=(0,),
            _compute_q_a=compute_q_a,
            _compute_q_b=compute_q_b,
            _materialize_cp_swa_k=lambda hidden, *args, **kw: enqueue(
                "swa_gather", hidden.value, hidden
            ),
            _store_cp_swa_k=lambda kv, *args: enqueue("swa_store", kv.value, kv),
        )
        backend = NS(
            low_ratio_prefill_graph=True,
            forward_low_ratio_sources=sources,
            _low_ratio_gather_k_prefill_graph=gather_index_k,
            _low_ratio_quantize_q_prefill_graph=lambda q: enqueue(
                "index_quant", q.value, q
            ),
            _low_ratio_index_topk_captured=topk,
        )
        with (
            patch(module + ".torch.cuda.current_stream", side_effect=lambda: active[0]),
            patch(module + ".torch.cuda.stream", side_effect=use_stream),
            patch(
                module + ".cp_materialize_global_token_order",
                side_effect=lambda hidden, *args: enqueue(
                    "main_gather", hidden.value, hidden
                ),
            ),
            patch(module + ".is_in_breakable_cuda_graph", return_value=captured),
        ):
            result = MQALayer._forward_prepare_low_ratio_cp_multi_stream(
                layer,
                x,
                positions,
                NS(encoder_swa_replay=encoder_replay),
                backend,
                q_out,
            )

        self.assertEqual(result.value, 12)
        if has_q_out:
            self.assertIs(result, q_out)
        self.assertIn(result.name, parent.ready, "attention reads unfinished Q")
        q_stream, q_dependencies = events["q_b"]
        self.assertIs(q_stream, workers[0])
        self.assertIn("q_a", q_dependencies)
        self.assertTrue(
            q_dependencies.isdisjoint({"swa_gather", "main_gather", "sources"})
        )
        self.assertIs(events["swa_gather"][0], parent)
        store_stream, store_dependencies = events["swa_store"]
        self.assertIs(store_stream, workers[0])
        self.assertTrue({"swa_gather", "q_b"}.issubset(store_dependencies))
        if compressor and not encoder_replay:
            self.assertIs(events["main_gather"][0], parent)
            self.assertIn("swa_gather", events["main_gather"][1])
            self.assertIn("main_gather", events["sources"][1])
        else:
            self.assertNotIn("main_gather", events)
        if captured:
            self.assertNotIn("swa_store", events["topk"][1])

    def test_cp_multistream_joins_swa_after_indexer_before_attention(self):
        # SWA KV is not an input of logits/top-k, but must be complete before
        # attention starts. A moved/removed join silently changes that boundary.
        order = []
        worker_streams = [object(), object(), object()]
        parent = NS(
            wait_stream=lambda stream: order.append(worker_streams.index(stream))
        )
        workers = [NS(wait_stream=lambda stream: None) for _ in range(3)]
        worker_streams[:] = workers
        x = torch.zeros(4, 2)
        positions = torch.arange(4)
        indexer = NS(
            queries=lambda q, freqs: x,
            head_weights=lambda hidden: x,
        )
        layer = NS(
            alt_streams=workers,
            fuse_wqa_wkv=False,
            compressor=object(),
            indexer=indexer,
            freqs_cis=torch.zeros(4),
            _compute_q_a=lambda *args, **kwargs: (x, x),
            _materialize_cp_swa_k=lambda *args, **kwargs: x,
            _store_cp_swa_k=lambda *args: None,
            _compute_q_b=lambda *args: x,
        )
        backend = NS(
            low_ratio_prefill_graph=True,
            forward_low_ratio_sources=lambda **kwargs: None,
            _low_ratio_gather_k_prefill_graph=lambda layer: None,
            _low_ratio_quantize_q_prefill_graph=lambda q: None,
            _low_ratio_index_topk_captured=lambda *args, **kwargs: order.append("topk"),
        )
        module = "sglang.srt.models.deepseek_v4"
        with (
            patch(module + ".torch.cuda.current_stream", return_value=parent),
            patch(
                module + ".torch.cuda.stream", side_effect=lambda stream: nullcontext()
            ),
            patch(module + ".cp_materialize_global_token_order", return_value=x),
            patch(module + ".is_in_breakable_cuda_graph", return_value=True),
        ):
            result = MQALayer._forward_prepare_low_ratio_cp_multi_stream(
                layer, x, positions, NS(encoder_swa_replay=False), backend
            )
        self.assertIs(result, x)
        self.assertEqual(order, [1, 2, "topk", 0])

    def test_interleave_roundtrip_mixed_lengths_prefix_and_padding(self):
        for size in (2, 4):
            for length in (4, 5, 9, 127, 128, 129):
                for rank in range(size):
                    with (
                        self.subTest(size=size, length=length, rank=rank),
                        cp_context(size, rank, (1, length - 1), (0, 16384)) as (
                            strategy,
                            batch,
                        ),
                    ):
                        embeddings = torch.arange(
                            length * 3, dtype=torch.float32
                        ).reshape(length, 3)
                        original_ids = batch.input_ids.clone()
                        with cp_shard_model_inputs(
                            embeddings, batch.positions, batch, batch.input_ids
                        ) as (local, positions, ids):
                            count = len(embeddings[rank::size])
                            torch.testing.assert_close(
                                local[:count], embeddings[rank::size]
                            )
                            torch.testing.assert_close(
                                positions[:count], batch.positions[rank::size]
                            )
                            torch.testing.assert_close(
                                ids[:count], batch.input_ids[rank::size]
                            )
                            self.assertEqual(
                                torch.count_nonzero(local[count:]).item(), 0
                            )
                            self.assertEqual(torch.count_nonzero(ids[count:]).item(), 0)
                            with simulated_collective(strategy, batch, embeddings):
                                restored = cp_gather_after_forward(local, batch)
                            torch.testing.assert_close(
                                restored, embeddings, rtol=0, atol=0
                            )
                        torch.testing.assert_close(batch.input_ids, original_ids)
                        self.assertFalse(hasattr(batch, "input_ids_global"))

    def test_speculative_state_and_global_ids_restored_on_exception(self):
        for size in (2, 4):
            for rank in range(size):
                for had_global in (False, True):
                    with (
                        self.subTest(size=size, rank=rank, had_global=had_global),
                        cp_context(size, rank) as (_, batch),
                    ):
                        full = torch.arange(36, dtype=torch.float32).reshape(9, 4)
                        batch.spec_info = NS(hidden_states=full)
                        previous = object()
                        if had_global:
                            batch.input_ids_global = previous
                        with self.assertRaisesRegex(RuntimeError, "injected"):
                            with cp_shard_model_inputs(
                                full, batch.positions, batch, batch.input_ids
                            ):
                                n = len(full[rank::size])
                                torch.testing.assert_close(
                                    batch.spec_info.hidden_states[:n], full[rank::size]
                                )
                                # Global MoE IDs are in rank-major order, with padding.
                                physical = sum(
                                    batch.attn_cp_metadata.per_rank_actual_token
                                )
                                padded = batch.input_ids.new_zeros(physical)
                                padded[:9] = batch.input_ids
                                expected = torch.cat(
                                    [padded[r::size] for r in range(size)]
                                )
                                torch.testing.assert_close(
                                    batch.input_ids_global, expected
                                )
                                raise RuntimeError("injected")
                        self.assertIs(batch.spec_info.hidden_states, full)
                        if had_global:
                            self.assertIs(batch.input_ids_global, previous)
                        else:
                            self.assertFalse(hasattr(batch, "input_ids_global"))

    def test_short_prompt_falls_back_from_cp(self):
        with cp_context(4, 0, (1, 2), (0, 0)) as (_, batch):
            self.assertFalse(is_cp_active(batch))

    def test_runner_text_embedding_and_preembedded_paths(self):
        for preembedded in (False, True):
            for rank in range(4):
                with (
                    self.subTest(preembedded=preembedded, rank=rank),
                    cp_context(4, rank) as (strategy, batch),
                ):
                    full = torch.arange(27, dtype=torch.float32).reshape(9, 3)
                    embedding = Mock(return_value=full)
                    model = NS(
                        vision=None,
                        _prepare_mm_embeddings=Mock(
                            side_effect=AssertionError("Text must not invoke vision")
                        ),
                        get_input_embeddings=Mock(return_value=embedding),
                        pp_group=NS(is_last_rank=True),
                        lm_head=object(),
                        capture_aux_hidden_states=False,
                        logits_processor=Mock(return_value="ok"),
                    )
                    model.prepare_language_model_inputs = lambda ids, fb, emb: (
                        DeepseekV4ForCausalLM.prepare_language_model_inputs(
                            model, ids, fb, emb
                        )
                    )

                    def body(ids, positions, fb, input_embeds):
                        n = len(full[rank::4])
                        torch.testing.assert_close(ids[:n], batch.input_ids[rank::4])
                        torch.testing.assert_close(
                            positions[:n], batch.positions[rank::4]
                        )
                        torch.testing.assert_close(input_embeds[:n], full[rank::4])
                        return input_embeds

                    model.model = body
                    with (
                        simulated_collective(strategy, batch, full),
                        patch(RUNNER + ".torch.cuda.current_stream", return_value=None),
                    ):
                        result = EagerRunner._execute_extend_cp(
                            NS(model_runner=NS(model=model)),
                            batch,
                            {"input_embeds": full} if preembedded else {},
                        )
                    self.assertEqual(result, "ok")
                    if preembedded:
                        model.get_input_embeddings.assert_not_called()
                    else:
                        embedding.assert_called_once_with(batch.input_ids)
                    model._prepare_mm_embeddings.assert_not_called()
                    args = model.logits_processor.call_args.args
                    torch.testing.assert_close(args[0], batch.input_ids)
                    torch.testing.assert_close(args[1], full)

    def test_dspark_aux_tensor_and_list_gathered_without_pre_norm_override(self):
        for as_list in (False, True):
            with self.subTest(as_list=as_list), cp_context(4, 2) as (strategy, batch):
                full = torch.arange(27, dtype=torch.float32).reshape(9, 3)
                local = strategy.shard_hidden_states(full, batch)
                aux = [local.clone(), local.clone()] if as_list else local.clone()
                model = NS(
                    get_input_embeddings=lambda: lambda ids: full,
                    model=Mock(return_value=((local, local.clone()), aux)),
                    capture_aux_hidden_states=True,
                    pp_group=NS(is_last_rank=True),
                    lm_head=object(),
                    logits_processor=Mock(return_value="ok"),
                )
                with (
                    simulated_collective(strategy, batch, full),
                    patch(RUNNER + ".torch.cuda.current_stream", return_value=None),
                ):
                    EagerRunner._execute_extend_cp(
                        NS(model_runner=NS(model=model)), batch, {}
                    )
                args, kwargs = model.logits_processor.call_args
                torch.testing.assert_close(args[1], full)
                for tensor in args[4] if as_list else [args[4]]:
                    torch.testing.assert_close(tensor, full)
                self.assertNotIn("hidden_states_before_norm", kwargs)

    def test_target_hidden_states_before_norm_preserved_without_dspark_aux(self):
        with cp_context(4, 1) as (strategy, batch):
            full = torch.arange(27, dtype=torch.float32).reshape(9, 3)
            local = strategy.shard_hidden_states(full, batch)
            model = NS(
                get_input_embeddings=lambda: lambda ids: full,
                model=Mock(return_value=(local, local.clone())),
                capture_aux_hidden_states=False,
                pp_group=NS(is_last_rank=True),
                lm_head=object(),
                logits_processor=Mock(return_value="ok"),
            )
            with (
                simulated_collective(strategy, batch, full),
                patch(RUNNER + ".torch.cuda.current_stream", return_value=None),
            ):
                EagerRunner._execute_extend_cp(
                    NS(model_runner=NS(model=model)), batch, {}
                )
            torch.testing.assert_close(
                model.logits_processor.call_args.kwargs["hidden_states_before_norm"],
                full,
            )


if __name__ == "__main__":
    unittest.main()
