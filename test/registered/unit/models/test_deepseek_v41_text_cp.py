"""Pure-language V4.1 CP input, padding and DSpark state regressions."""

import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.cp.utils import (
    cp_gather_after_forward,
    cp_shard_model_inputs,
    is_cp_active,
)
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.dsv41_cp_test_utils import cp_context, simulated_collective
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


RUNNER = "sglang.srt.model_executor.runner.eager_runner"


class TestDSV41TextCP(CustomTestCase):
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
