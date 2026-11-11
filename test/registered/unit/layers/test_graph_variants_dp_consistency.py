"""Decode graph variants under DP attention: every rank replays the variant of the
step's agreed longest sequence (the MLP-sync bound), never reading device lengths,
and a DSA target-verify capture holds a single graph."""

import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.batch_overlap.two_batch_overlap import TboForwardBatchPreparer
from sglang.srt.layers.attention import graph_variants as gv
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_components import dp_attn
from sglang.srt.model_executor import forward_batch_info
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import CaptureHiddenMode, ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _NoDeviceRead:
    """Stands in for device lengths: select() must not touch them."""

    def __getattr__(self, name):
        raise AssertionError(f"select() read a device length tensor ({name})")


def _batch(mode, bound, seq_lens_cpu=None):
    return SimpleNamespace(
        forward_mode=mode,
        dp_decode_graph_max_seq_len=bound,
        seq_lens_cpu=_NoDeviceRead() if seq_lens_cpu is None else seq_lens_cpu,
        seq_lens=_NoDeviceRead(),
        spec_info=None,
    )


class TestDsaGraphVariantsUnderDP(CustomTestCase):
    def setUp(self):
        self.variants = gv.DsaGraphVariants(index_topk=2048)

    def test_ranks_with_different_lengths_pick_one_variant(self):
        for local in (torch.tensor([7]), torch.tensor([9000])):
            self.assertEqual(
                self.variants.select(_batch(ForwardMode.DECODE, 2048, local)),
                gv.DSA_DENSE,
            )
            self.assertEqual(
                self.variants.select(_batch(ForwardMode.DECODE, 2049, local)),
                gv.DSA_SPARSE,
            )

    def test_idle_rank_follows_the_bound(self):
        # Without the bound an idle rank always replays sparse, unlike a busy
        # short-context peer that replays dense.
        self.assertEqual(
            self.variants.select(_batch(ForwardMode.IDLE, 100)), gv.DSA_DENSE
        )
        self.assertEqual(
            self.variants.select(_batch(ForwardMode.IDLE, 0)), gv.DSA_DENSE
        )

    def test_without_sync_keeps_the_local_rule(self):
        self.assertEqual(
            self.variants.select(
                _batch(ForwardMode.DECODE, None, torch.tensor([5, 2055]))
            ),
            gv.DSA_SPARSE,
        )
        self.assertEqual(
            self.variants.select(
                _batch(ForwardMode.DECODE, None, torch.tensor([5, 9]))
            ),
            gv.DSA_DENSE,
        )

    def test_target_verify_capture_holds_one_graph(self):
        # Dense only skips the decode/idle indexer logits; verify runs the full
        # indexer in both, so a pair would only double the captures.
        with (
            patch("sglang.srt.utils.is_hip", return_value=True),
            patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True),
            patch(
                "sglang.srt.configs.model_config.get_dsa_index_kpool", return_value=1
            ),
            patch(
                "sglang.srt.configs.model_config.get_dsa_index_topk", return_value=2048
            ),
        ):
            hf_config = SimpleNamespace()
            self.assertIsNone(
                gv.create_attention_graph_variants(hf_config, ForwardMode.TARGET_VERIFY)
            )
            self.assertIsInstance(
                gv.create_attention_graph_variants(hf_config, ForwardMode.DECODE),
                gv.DsaGraphVariants,
            )
            self.assertIsInstance(
                gv.create_attention_graph_variants(hf_config), gv.DsaGraphVariants
            )


class TestLengthKeyedVariantsUnderDP(CustomTestCase):
    def test_seq_len_buckets_follow_the_bound(self):
        variants = gv.SeqLenBucketGraphVariants(
            graph_limits=(("p1", 4096), ("p4", 16384)), fallback_label="pmax"
        )
        decode = ForwardMode.DECODE
        self.assertEqual(
            variants.select(_batch(decode, 4096, torch.tensor([20000]))), "p1"
        )
        self.assertEqual(variants.select(_batch(decode, 5000, torch.tensor([7]))), "p4")
        self.assertEqual(
            variants.select(_batch(decode, 99999, torch.tensor([7]))), "pmax"
        )
        self.assertEqual(variants.select(_batch(ForwardMode.IDLE, 0)), "p1")
        self.assertEqual(variants.select(_batch(decode, None, torch.tensor([7]))), "p1")

    def test_dsv41_candidates_follow_the_bound_in_decode_only(self):
        variants = gv.Dsv41CandidateGraphVariants(
            graph_limits=(
                ("candidate_all", 2048),
                (gv.DSV41_CANDIDATE_UNFILTERED, 16384),
            ),
            capture_labels=(
                "candidate_all",
                gv.DSV41_CANDIDATE_UNFILTERED,
                gv.DSV41_CANDIDATE_FILTERED,
            ),
        )
        decode = ForwardMode.DECODE
        self.assertEqual(
            variants.select(_batch(decode, 2000, torch.tensor([30000]))),
            "candidate_all",
        )
        self.assertEqual(
            variants.select(_batch(decode, 20000, torch.tensor([7]))),
            gv.DSV41_CANDIDATE_FILTERED,
        )
        # A verify replay keeps the variant its early-staged metadata was built for.
        self.assertEqual(
            variants.select(
                _batch(ForwardMode.TARGET_VERIFY, 2000, torch.tensor([30000]))
            ),
            gv.DSV41_CANDIDATE_FILTERED,
        )


class TestMLPSyncDecodeBound(CustomTestCase):
    def _info(self, bound):
        return dp_attn.MLPSyncBatchInfo(
            num_dp_ranks=1,
            tp_size=1,
            cp_size=1,
            num_tokens=3,
            num_tokens_for_logprob=3,
            can_run_decode_cuda_graph=True,
            can_run_draft_cuda_graph=True,
            can_run_prefill_cuda_graph=False,
            is_extend_in_batch=False,
            local_can_run_tbo=False,
            local_forward_mode=ForwardMode.DECODE.value,
            decode_graph_max_seq_len=bound,
        )

    def test_sync_row_carries_the_bound(self):
        info = self._info(1234)
        local, fallback = (
            info._get_local_tensor("cpu"),
            info._get_fallback_tensor("cpu"),
        )
        self.assertEqual(local.numel(), fallback.numel())
        self.assertEqual(int(local[-1]), 1234)
        self.assertEqual(int(fallback[-1]), 0)
        info.finalize_local()
        self.assertEqual(info.global_decode_graph_max_seq_len, 1234)

    def test_gathered_bound_is_the_maximum_over_ranks(self):
        bounds = (10, 3000, 0, 512)
        info = dataclasses.replace(self._info(bounds[0]), num_dp_ranks=len(bounds))
        rows = torch.cat([self._info(b)._get_local_tensor("cpu") for b in bounds])

        def exchange(flat, local, group):
            flat.copy_(rows)

        parallel = SimpleNamespace(
            tp_group=SimpleNamespace(
                active_ranks_cpu=torch.ones(len(bounds), dtype=torch.int32)
            )
        )
        with (
            patch.object(dp_attn, "exchange_sync_rows", side_effect=exchange),
            patch.object(dp_attn, "get_parallel", return_value=parallel),
        ):
            info.all_gather(device="cpu", group=None)
        self.assertEqual(info.global_decode_graph_max_seq_len, 3000)

    def test_local_bound(self):
        local = dp_attn._local_decode_graph_max_seq_len
        none, eagle = SpeculativeAlgorithm.NONE, SpeculativeAlgorithm.EAGLE
        decode = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            batch_size=lambda: 2,
            seq_lens_cpu=torch.tensor([9, 4000]),
            reqs=[],
        )
        self.assertEqual(local(decode, eagle), 4000)
        # Without the host mirror: committed lengths plus two steps' width.
        decode.seq_lens_cpu = None
        decode.reqs = [SimpleNamespace(seqlen=100), SimpleNamespace(seqlen=3990)]
        with patch.object(dp_attn, "max_speculative_num_draft_tokens", return_value=4):
            self.assertEqual(local(decode, eagle), 3998)
        with patch.object(
            dp_attn, "max_speculative_num_draft_tokens", return_value=None
        ):
            self.assertEqual(local(decode, eagle), 3992)
        # Plain decode advances one token a step and never reads the spec config.
        with patch.object(
            dp_attn,
            "max_speculative_num_draft_tokens",
            side_effect=AssertionError("spec config read without speculation"),
        ):
            self.assertEqual(local(decode, none), 3992)
        idle = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            batch_size=lambda: 0,
            seq_lens_cpu=None,
            reqs=[],
        )
        self.assertEqual(local(idle, eagle), 0)
        self.assertEqual(local(None, eagle), 0)
        extend = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            batch_size=lambda: 2,
            seq_lens_cpu=torch.tensor([9]),
            reqs=[],
        )
        self.assertEqual(local(extend, eagle), 0)

    def test_update_gather_batch_puts_the_bound_on_the_batch(self):
        info = self._info(1234)
        info.finalize_local()
        batch = SimpleNamespace()
        dp_attn._update_gather_batch(batch, info, require_mlp_tp_gather=True)
        self.assertEqual(batch.dp_decode_graph_max_seq_len, 1234)
        dp_attn._update_gather_batch(
            batch, info, require_mlp_tp_gather=True, skip_global_metadata=True
        )
        self.assertIsNone(batch.dp_decode_graph_max_seq_len)


class TestBoundReachesTheRunner(CustomTestCase):
    """The bound rides the ScheduleBatch, its copy (what overlap scheduling hands to
    the result queue), the ForwardBatch built from it, and each TBO child."""

    def test_schedule_batch_copy_keeps_the_bound(self):
        batch = ScheduleBatch(
            reqs=[],
            req_to_token_pool=SimpleNamespace(),
            token_to_kv_pool_allocator=SimpleNamespace(state_slots=SimpleNamespace()),
            model_config=SimpleNamespace(is_encoder_decoder=False),
            spec_algorithm=SpeculativeAlgorithm.NONE,
            req_pool_indices=torch.zeros(0, dtype=torch.int64),
            req_pool_indices_cpu=torch.zeros(0, dtype=torch.int64),
            dp_decode_graph_max_seq_len=3000,
        )
        self.assertEqual(batch.copy().dp_decode_graph_max_seq_len, 3000)

    def test_forward_batch_takes_the_bound(self):
        runner = SimpleNamespace(
            device="cpu",
            is_draft_worker=False,
            model_config=SimpleNamespace(
                requires_mm_token_modalities=False, model_is_mrope=False
            ),
            runner_ids=MagicMock(),
            prefill_attention_backend_str="torch_native",
            ngram_embedding_manager=SimpleNamespace(enabled=False),
            lora_manager=None,
            attn_backend=MagicMock(),
            attn_tp_sequence_sharded=lambda _: False,
        )
        seq_lens = torch.tensor([9, 3000])
        batch = ScheduleBatch(
            reqs=[],
            device="cpu",
            model_config=SimpleNamespace(is_encoder_decoder=False, is_matryoshka=False),
            spec_algorithm=SpeculativeAlgorithm.NONE,
            forward_mode=ForwardMode.DECODE,
            input_ids=torch.zeros(2, dtype=torch.int64),
            req_pool_indices=torch.arange(2),
            req_pool_indices_cpu=torch.arange(2),
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens,
            seq_lens_sum=int(seq_lens.sum()),
            orig_seq_lens=seq_lens,
            out_cache_loc=torch.arange(2),
            dp_decode_graph_max_seq_len=3000,
        )
        # Decode positions come from a fused op whose JIT path wants GPU tensors.
        with (
            get_context().override_server_args(device="cpu"),
            patch.object(
                forward_batch_info,
                "clamp_position",
                lambda seq_lens: torch.clamp(seq_lens - 1, min=0),
            ),
        ):
            forward = ForwardBatch.init_new(
                batch,
                runner,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                return_hidden_states_before_norm=False,
            )
        self.assertEqual(forward.dp_decode_graph_max_seq_len, 3000)

    def test_tbo_children_keep_the_bound(self):
        parent = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=4,
            input_ids=torch.zeros(4, dtype=torch.long),
            positions=torch.zeros(4, dtype=torch.long),
            out_cache_loc=torch.zeros(4, dtype=torch.long),
            req_pool_indices=torch.zeros(4, dtype=torch.long),
            seq_lens=torch.full((4,), 3000, dtype=torch.int32),
            seq_lens_cpu=torch.full((4,), 3000, dtype=torch.int32),
            seq_lens_sum=12000,
            spec_info=None,
            dp_decode_graph_max_seq_len=3000,
        )
        with (
            get_context().override_server_args(
                attention_backend="fa3", moe_dense_tp_size=None
            ),
            get_parallel().override(attn_tp_size=1),
        ):
            for lo, hi in ((0, 2), (2, 4)):
                child = TboForwardBatchPreparer.filter_batch(
                    parent,
                    start_token_index=lo,
                    end_token_index=hi,
                    start_seq_index=lo,
                    end_seq_index=hi,
                    out_num_token_non_padded=torch.tensor(hi - lo),
                )
                self.assertEqual(child.dp_decode_graph_max_seq_len, 3000)


if __name__ == "__main__":
    unittest.main()
