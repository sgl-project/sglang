import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from safetensors.torch import save_file

import sglang.srt.beam_search.trie_constraint as trie_constraint_module
import sglang.srt.layers.logits_processor as logits_module
from sglang.srt.beam_search.trie_config import (
    TRIE_OUTPUT_HEAD_FILENAME,
    discover_trie_output_head_config,
    load_trie_output_head_config,
)
from sglang.srt.beam_search.trie_constraint import BeamTrieConstraint
from sglang.srt.beam_search.trie_output_head import (
    BeamTrieOutputHead,
    configure_model_trie_output_heads,
)
from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBeamTrieConstraint(CustomTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.tensors = {
            "root_token_ids": torch.tensor([0, 2], dtype=torch.int16),
            "level1_token_ids": torch.tensor([1, 3, 0], dtype=torch.int16),
            "level1_offsets": torch.tensor([0, 2, 3], dtype=torch.int32),
            "level2_offsets": torch.tensor([0, 2, 3, 6], dtype=torch.int32),
            "level2_token_ids": torch.tensor([0, 2, 1, 2, 2, 3], dtype=torch.int16),
        }
        self.metadata = {
            "format_version": "1",
            "token_start": "10",
            "codebook_size": "4",
            "num_codebooks": "3",
        }
        self._write_artifact(root)
        self.root = root

    def _write_artifact(self, root, *, tensors=None, metadata=None):
        save_file(
            tensors or self.tensors,
            str(root / TRIE_OUTPUT_HEAD_FILENAME),
            metadata=metadata or self.metadata,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def load(self):
        config = load_trie_output_head_config(
            str(self.root / TRIE_OUTPUT_HEAD_FILENAME)
        )
        return config, BeamTrieConstraint.load(config, "cpu")

    def processor(self, config):
        processor = LogitsProcessor.__new__(LogitsProcessor)
        torch.nn.Module.__init__(processor)
        processor.beam_trie_output_head = BeamTrieOutputHead(
            config, vocab_size=30, chunk_size=7
        )
        processor.vocab_size = 30
        processor.logit_scale = None
        processor.final_logit_softcapping = None
        processor.use_fp32_lm_head = False
        processor.rl_on_policy_target = None
        processor.do_tensor_parallel_all_gather = False
        processor.do_tensor_parallel_all_gather_dp_attn = False
        processor.use_attn_tp_group = False
        processor.use_tp_lm_head_all_to_all = False
        processor.defer_verify_gather = False
        return processor

    @torch.inference_mode()
    def test_logits_pipeline_preserves_trie_rounding_and_dense_buffer_path(self):
        config, _ = self.load()
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        head = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        torch.nn.Module.__init__(head)
        head.weight = torch.nn.Parameter(torch.arange(60.0).view(30, 2))
        head.quant_method = None
        dense = hidden @ head.weight.T
        for native, depths in (
            (False, None),
            (True, None),
            (True, torch.tensor([1, 2])),
        ):
            with self.subTest(native=native, depths=depths):
                processor = self.processor(config)
                if not native:
                    processor.beam_trie_output_head = None
                buffer = torch.full_like(dense, torch.nan)
                metadata = (
                    LogitsMetadata(
                        ForwardMode.DECODE,
                        next_token_logits_buffer=buffer,
                        beam_trie_levels=depths,
                    )
                    if native
                    else SimpleNamespace(next_token_logits_buffer=buffer)
                )
                with mock.patch.object(
                    processor, "_compute_lm_head", wraps=processor._compute_lm_head
                ) as project:
                    logits, normalizer = processor._get_logits_with_beam_normalizer(
                        hidden, head, metadata
                    )
                if depths is None:
                    self.assertIs(logits, buffer)
                    self.assertIsNone(normalizer)
                    self.assertIs(
                        project.call_args.kwargs["out"], None if native else buffer
                    )
                    torch.testing.assert_close(logits, dense)
                else:
                    indices = config.token_start + depths[:, None] * 4 + torch.arange(4)
                    torch.testing.assert_close(logits, dense.gather(1, indices))
                    torch.testing.assert_close(
                        normalizer.sum(dim=1), torch.logsumexp(dense, dim=-1)
                    )
                    self.assertTrue(torch.isnan(buffer).all())

    @torch.inference_mode()
    def test_fused_and_deferred_gather_keep_normalizer_protocol(self):
        """Every logits path must retain the tensor/normalizer return contract."""

        class CudaLogits(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        def softcap(shards, cap, output):
            output.copy_(cap * torch.tanh(shards.transpose(0, 1).reshape(3, 10) / cap))
            return output

        group = SimpleNamespace(
            world_size=2,
            rank_in_group=0,
            all_gather_into_tensor=lambda output, value: output.copy_(
                value.repeat(2, 1)
            ),
        )
        config, _ = self.load()
        hidden = torch.arange(6.0).view(3, 2)
        head = torch.nn.Linear(2, 5, bias=False)
        shard = hidden @ head.weight.T
        with (
            get_context().override_server_args(device="cpu", tp_size=2),
            get_parallel().override(tp_rank=0, tp_group=group),
            mock.patch.object(logits_module, "_is_cpu", False),
            mock.patch.object(
                logits_module, "gathered_softcap_logits", side_effect=softcap
            ),
        ):
            for deferred in (False, True):
                with self.subTest(deferred=deferred):
                    processor = self.processor(config)
                    processor.beam_trie_output_head = None
                    processor.vocab_size = 10
                    processor.do_tensor_parallel_all_gather = True
                    processor.final_logit_softcapping = 3.0
                    processor.defer_verify_gather = deferred
                    metadata = LogitsMetadata(ForwardMode.TARGET_VERIFY)
                    logits, normalizer = processor._get_logits_with_beam_normalizer(
                        hidden.as_subclass(CudaLogits), head, metadata
                    )
                    expected = (
                        shard
                        if deferred
                        else 3.0 * torch.tanh(shard.repeat(1, 2) / 3.0)
                    )
                    torch.testing.assert_close(
                        logits.as_subclass(torch.Tensor), expected
                    )
                    self.assertIsNone(normalizer)

    def test_dense_and_sparse_child_masks(self):
        config, dense = self.load()
        self.assertEqual(config.token_start_for_depth(2), 18)
        with mock.patch.object(
            trie_constraint_module,
            "_MAX_DENSE_LEVEL2_MASK_BYTES",
            0,
        ):
            sparse = BeamTrieConstraint.load(config, "cpu")
        self.assertIsNone(sparse.level2_masks)
        self.assertIsNotNone(sparse.level2_sparse)
        cases = (
            (torch.empty((1, 0), dtype=torch.int64), [[True, False, True, False]]),
            (
                torch.tensor([[0], [2]]),
                [[False, True, False, True], [True, False, False, False]],
            ),
            (
                torch.tensor([[0, 1], [2, 0]]),
                [[True, False, True, False], [False, False, True, True]],
            ),
        )
        for constraint in (dense, sparse):
            self.assertEqual(constraint.level_cardinalities, (2, 3, 5))
            for depth, (prefixes, expected) in enumerate(cases):
                with self.subTest(sparse=constraint is sparse, depth=depth):
                    self.assertEqual(
                        constraint.valid_child_mask(prefixes, depth).tolist(), expected
                    )
        with self.assertRaisesRegex(ValueError, "unknown AB pair"):
            sparse.valid_child_mask(torch.tensor([[0, 0]]), 2)

    def test_compact_signed_array_preserves_values_above_int32(self):
        maximum = np.iinfo(np.int32).max + 1
        compact = trie_constraint_module._compact_signed_array(
            np.array([maximum], dtype=np.int64),
            maximum,
        )

        self.assertEqual(compact.dtype, np.int64)
        self.assertEqual(compact.item(), maximum)

    def test_dense_and_compact_scoring_match(self):
        _, constraint = self.load()
        dense = torch.arange(40, dtype=torch.float32).reshape(2, 20)
        dense_values, dense_tokens = constraint.topk_logprobs(
            [dense],
            torch.tensor([[0], [2]]),
            depth=1,
            num_candidates=4,
        )
        row_max = dense.max(dim=-1).values
        row_log_sum = torch.log(torch.exp(dense - row_max[:, None]).sum(dim=-1))
        compact_values, compact_tokens = constraint.topk_logprobs(
            [dense[:, 14:18]],
            torch.tensor([[0], [2]]),
            depth=1,
            num_candidates=4,
            normalizers=[torch.stack((row_max, row_log_sum), dim=1)],
        )

        torch.testing.assert_close(compact_values, dense_values)
        torch.testing.assert_close(compact_tokens, dense_tokens)
        self.assertEqual(dense_tokens[0, :2].tolist(), [17, 15])
        self.assertTrue(torch.isneginf(dense_values[0, 2:]).all())

    def test_topk_logprob_normalizes_only_constrained_candidates(self):
        _, constraint = self.load()
        compact = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        prefixes = torch.tensor([[0], [2]])
        raw_values, expected_tokens = constraint.topk_logprobs(
            [compact],
            prefixes,
            depth=1,
            num_candidates=4,
            normalizers=[torch.zeros((2, 2))],
        )
        with envs.SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB.override(True):
            values, tokens = constraint.topk_logprobs(
                [compact],
                prefixes,
                depth=1,
                num_candidates=4,
            )

        torch.testing.assert_close(tokens, expected_tokens)
        torch.testing.assert_close(
            values,
            raw_values - torch.logsumexp(raw_values, dim=-1, keepdim=True),
        )

    def test_root_candidates_pad_wide_beam_with_score_dead_valid_tokens(self):
        _, constraint = self.load()
        values, tokens = constraint.topk_logprobs(
            [torch.arange(4, dtype=torch.float32).unsqueeze(0)],
            torch.empty((1, 0), dtype=torch.int64),
            depth=0,
            num_candidates=6,
            normalizers=[torch.zeros((1, 2))],
        )
        self.assertEqual(tokens.tolist(), [[12, 10, 10, 10, 12, 12]])
        self.assertTrue(torch.isneginf(values[0, 2:]).all())

    def test_advances_device_prefix_state(self):
        _, constraint = self.load()
        prefixes = torch.tensor([[0], [2]], dtype=torch.int64)
        advanced = constraint.advance_prefixes(
            prefixes,
            parent_idx=torch.tensor([1, 0]),
            next_tokens=torch.tensor([14, 17]),
            depth=1,
        )
        self.assertEqual(advanced.tolist(), [[2, 0], [0, 3]])

    def test_mixed_compact_lm_head_preserves_dense_scores_and_transforms(self):
        config, _ = self.load()
        processor = self.processor(config)
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        depths = torch.tensor([0, 2, 1], dtype=torch.int64)
        weight = torch.arange(60, dtype=torch.float32).reshape(30, 2)
        lm_head = VocabParallelEmbedding.__new__(VocabParallelEmbedding)
        torch.nn.Module.__init__(lm_head)
        lm_head.weight = torch.nn.Parameter(weight)
        lm_head.quant_method = None

        dense = hidden @ weight.T
        indices = (
            config.token_start
            + depths[:, None] * config.codebook_size
            + torch.arange(config.codebook_size)
        )
        self.enterContext(
            mock.patch(
                "sglang.srt.layers.logits_processor.fused_softcap",
                side_effect=lambda logits, cap: logits.copy_(
                    cap * torch.tanh(logits / cap)
                ),
            )
        )
        for scale, softcap in ((None, None), (0.5, 3.0)):
            with self.subTest(scale=scale, softcap=softcap):
                processor.logit_scale = scale
                processor.final_logit_softcapping = softcap
                actual = processor._compute_mixed_beam_trie_lm_head(
                    hidden, lm_head, depths
                )
                torch.testing.assert_close(actual.logits, dense.gather(1, indices))
                expected = dense if scale is None else dense * scale
                if softcap is not None:
                    expected = softcap * torch.tanh(expected / softcap)
                torch.testing.assert_close(
                    actual.normalizer[:, 0] + actual.normalizer[:, 1],
                    torch.logsumexp(expected, dim=-1),
                )

        self.assertIsNone(
            processor._beam_trie_output_head_for(
                lm_head,
                embedding_bias=torch.zeros(30),
            )
        )

    def test_compact_head_configuration_and_dense_fallback(self):
        config, _ = self.load()
        processor = self.processor(config)
        model = torch.nn.Sequential(processor)
        options = dict(
            model=model,
            config=config,
            device="cuda",
            tp_size=1,
            quantization=None,
            lora_enabled=False,
            rl_on_policy_target=None,
        )
        self.assertTrue(configure_model_trie_output_heads(**options))
        self.assertIsInstance(processor.beam_trie_output_head, BeamTrieOutputHead)
        for fallback in (
            {"device": "cpu"},
            {"tp_size": 2},
            {"quantization": "fp8"},
            {"lora_enabled": True},
            {"rl_on_policy_target": "test"},
            {"config": None},
        ):
            with self.subTest(fallback=fallback):
                self.assertFalse(
                    configure_model_trie_output_heads(**(options | fallback))
                )
                self.assertIsNone(processor.beam_trie_output_head)

    def test_configured_model_requires_a_logits_processor(self):
        config, _ = self.load()
        with self.assertRaisesRegex(RuntimeError, "no compatible LogitsProcessor"):
            configure_model_trie_output_heads(
                model=torch.nn.Identity(),
                config=config,
                device="cuda",
                tp_size=1,
                quantization=None,
                lora_enabled=False,
                rl_on_policy_target=None,
            )

    def test_discovers_only_model_root_artifact(self):
        model_root = self.root / "model"
        model_root.mkdir()
        self.assertIsNone(discover_trie_output_head_config(str(model_root)))
        nested_root = model_root / "nested"
        nested_root.mkdir()
        self._write_artifact(nested_root)
        self.assertIsNone(discover_trie_output_head_config(str(model_root)))
        self._write_artifact(model_root)
        config = discover_trie_output_head_config(str(model_root))
        self.assertIsNotNone(config)
        self.assertEqual(config.token_start, 10)
        self.assertEqual(
            config.tensor_path,
            str(model_root / TRIE_OUTPUT_HEAD_FILENAME),
        )

    def test_rejects_unsupported_format_version(self):
        root = self.root / "unsupported"
        root.mkdir()
        self._write_artifact(
            root,
            metadata={**self.metadata, "format_version": "2"},
        )
        with self.assertRaisesRegex(ValueError, "Unsupported.*format version"):
            load_trie_output_head_config(str(root / TRIE_OUTPUT_HEAD_FILENAME))

    def test_rejects_missing_tensor(self):
        root = self.root / "missing"
        root.mkdir()
        self._write_artifact(
            root,
            tensors={
                name: tensor
                for name, tensor in self.tensors.items()
                if name != "level2_token_ids"
            },
        )
        config = load_trie_output_head_config(str(root / TRIE_OUTPUT_HEAD_FILENAME))
        with self.assertRaisesRegex(ValueError, "level2_token_ids"):
            BeamTrieConstraint.load(config, "cpu")


if __name__ == "__main__":
    unittest.main()
