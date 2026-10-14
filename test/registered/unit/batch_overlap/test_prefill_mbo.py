"""Two-microbatch prefill: model gating, the split point, the layer operations
and per-microbatch mHC coefficients."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.batch_overlap import prefill_mbo
from sglang.srt.batch_overlap.operations_strategy import OperationsStrategy
from sglang.srt.environ import envs
from sglang.srt.layers.layer_boundary.residual.mhc import MHCState
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch(prefix, tokens, *, mode=ForwardMode.EXTEND, batch_size=1):
    return SimpleNamespace(
        forward_mode=mode,
        batch_size=batch_size,
        extend_seq_lens_cpu=[tokens],
        extend_prefix_lens_cpu=[prefix],
        input_ids=torch.zeros(tokens, dtype=torch.int64),
    )


def _runner(supports=None, *, draft=False):
    model = SimpleNamespace()
    if supports is not None:
        model.supports_prefill_mbo = lambda: supports
    return SimpleNamespace(model=model, is_draft_worker=draft)


class TestEnabled(CustomTestCase):
    def test_only_a_supporting_target_model_takes_part(self):
        with envs.SGLANG_PREFILL_MBO.override(True):
            self.assertTrue(prefill_mbo.enabled(_runner(True)))
            self.assertFalse(prefill_mbo.enabled(_runner(False)))
            self.assertFalse(prefill_mbo.enabled(_runner()))
            self.assertFalse(prefill_mbo.enabled(_runner(True, draft=True)))

    def test_off_without_the_env(self):
        with envs.SGLANG_PREFILL_MBO.override(False):
            self.assertFalse(prefill_mbo.enabled(_runner(True)))


class TestGlm5NextGate(CustomTestCase):
    def _supports(self, *, shared_topk=False, model=None, **parallel):
        from sglang.srt.models import glm5_next

        lm = object.__new__(glm5_next.Glm5NextForConditionalGeneration)
        lm.config = SimpleNamespace(mhc=True, num_hidden_layers=4)
        lm.use_dsa = True
        lm.capture_aux_hidden_states = False
        lm.model = SimpleNamespace(first_k_dense_replace=1, end_layer=4)
        for k, v in (model or {}).items():
            setattr(lm, k, v)
        widths = dict(tp_size=4, attn_tp_size=4, pp_size=1, attn_dcp_size=1)
        widths.update(moe_ep_size=1)
        widths.update(parallel)
        with (
            patch.object(
                glm5_next, "get_parallel", return_value=SimpleNamespace(**widths)
            ),
            patch.object(
                glm5_next,
                "dsa_layer_skips_topk",
                side_effect=lambda config, layer_id: shared_topk and layer_id == 2,
            ),
        ):
            return lm.supports_prefill_mbo()

    def test_tp_only_mhc_model_is_supported(self):
        self.assertTrue(self._supports())
        # Without DSA there are no top-k indices to share.
        self.assertTrue(self._supports(shared_topk=True, model=dict(use_dsa=False)))

    def test_unsupported_configurations(self):
        cases = {
            "no mhc": dict(model=dict(config=SimpleNamespace(mhc=False))),
            "aux capture": dict(model=dict(capture_aux_hidden_states=True)),
            "shared dsa top-k": dict(shared_topk=True),
            "no layers after the dense ones": dict(
                model=dict(model=SimpleNamespace(first_k_dense_replace=4, end_layer=4))
            ),
            "pp": dict(pp_size=2),
            "dp attention": dict(attn_tp_size=2),
            "dcp": dict(attn_dcp_size=2),
            "moe ep": dict(moe_ep_size=4),
        }
        for name, kwargs in cases.items():
            with self.subTest(name):
                self.assertFalse(self._supports(**kwargs))


class TestSplitPoint(CustomTestCase):
    def setUp(self):
        patcher = patch.object(prefill_mbo, "_child_backends", [object(), object()])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_full_chunk_splits_in_half(self):
        self.assertEqual(prefill_mbo._split_tokens(_batch(8192, 8192)), 4096)

    def test_split_lands_on_an_aligned_absolute_position(self):
        split = prefill_mbo._split_tokens(_batch(300, 5000))
        self.assertEqual((300 + split) % prefill_mbo._SPLIT_ALIGN, 0)
        self.assertLessEqual(abs(split - 2500), prefill_mbo._SPLIT_ALIGN)

    def test_tracked_state_without_cpu_metadata_is_not_split(self):
        batch = _batch(0, 8192)
        batch.mamba_track_mask = torch.ones(1, dtype=torch.bool)
        batch.mamba_prefill_track_mask_cpu = None
        batch.mamba_track_seqlens_cpu = None
        self.assertFalse(prefill_mbo.maybe_split(batch))
        self.assertIsNone(prefill_mbo.active_child_backends())

    def test_ineligible_batches_are_not_split(self):
        self.assertIsNone(prefill_mbo._split_tokens(_batch(0, 1024)))
        self.assertIsNone(prefill_mbo._split_tokens(_batch(0, 8192, batch_size=2)))
        self.assertIsNone(
            prefill_mbo._split_tokens(_batch(0, 8192, mode=ForwardMode.MIXED))
        )


class Glm5NextDecoderLayer:
    def __init__(self, name):
        self.op_mbo_attn = f"{name}.attn"
        self.op_mbo_ffn = f"{name}.ffn"
        self.op_mbo_finish = f"{name}.finish"


class TestLayerOperations(CustomTestCase):
    def tearDown(self):
        prefill_mbo.end_forward()

    def test_glm5_next_runs_only_split_prefills(self):
        layers = [Glm5NextDecoderLayer("l0"), Glm5NextDecoderLayer("l1")]
        with self.assertRaises(NotImplementedError):
            OperationsStrategy.init_new_tbo(layers, ForwardMode.DECODE)

        with patch.object(prefill_mbo, "_child_backends", [object(), object()]):
            with patch.object(prefill_mbo, "_active", True):
                strategy = OperationsStrategy.init_new_tbo(layers, ForwardMode.EXTEND)
        names = [op if isinstance(op, str) else "yield" for op in strategy.operations]
        self.assertEqual(
            names,
            [
                *("l0.attn", "yield", "l0.ffn", "yield"),
                *("l1.attn", "yield", "l1.ffn", "yield"),
                "l1.finish",
            ],
        )
        self.assertEqual(strategy.tbo_delta_stages, 0)


class TestMhcCoefficientsPerMicrobatch(CustomTestCase):
    def tearDown(self):
        prefill_mbo.set_microbatch(0)

    def test_interleaved_reads_keep_their_own_coefficients(self):
        posts = []

        def hc_pre(hidden_states, *_):
            tag = hidden_states[0, 0].item()
            return hidden_states, torch.tensor([tag]), torch.tensor([10 * tag]), False

        state = MHCState(
            hc_mult=1,
            hc_attn_pre=hc_pre,
            hc_ffn_pre=hc_pre,
            hc_post=lambda x, r, h_res, h_post: posts.append((h_res, h_post)) or x,
        )
        for microbatch, tag in ((0, 1.0), (1, 2.0)):
            prefill_mbo.set_microbatch(microbatch)
            state.read_attn_input(torch.full((2, 4), tag))
        for microbatch in (0, 1):
            prefill_mbo.set_microbatch(microbatch)
            state.apply_post(torch.zeros(2, 4), torch.zeros(2, 4))

        self.assertEqual([(r.item(), p.item()) for r, p in posts], [(1, 10), (2, 20)])


if __name__ == "__main__":
    unittest.main()
