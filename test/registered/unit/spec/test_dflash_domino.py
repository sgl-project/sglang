import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.speculative.domino_utils import (
    domino_greedy_rollout,
    validate_domino_runtime,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


def _domino_config(**overrides):
    dflash_config = {
        "projector_type": "domino",
        "mask_token_id": 29,
        "shift_label": True,
        "target_layer_ids": [1, 3],
        "pure_draft_prefix_len": 1,
        "gru_hidden_dim": 4,
        "emb_dim": 5,
    }
    dflash_config.update(overrides.pop("dflash_config", {}))
    fields = {
        "num_hidden_layers": 2,
        "num_target_layers": 4,
        "block_size": 16,
        "hidden_size": 8,
        "vocab_size": 31,
        "emb_dim": 5,
        "dflash_config": dflash_config,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _projector_model(projector_type="domino"):
    model = DFlashDraftModel.__new__(DFlashDraftModel)
    nn.Module.__init__(model)
    model.projector_type = projector_type
    model.config = SimpleNamespace(hidden_size=8)
    if projector_type == "domino":
        model.prefix_gru = nn.GRU(8, 4, batch_first=True, bias=False)
        model.embed_proj = nn.Sequential(
            nn.Linear(12, 5, bias=False),
            nn.SiLU(),
            nn.Linear(5, 31, bias=False),
        )
    else:
        model.prefix_gru = None
        model.embed_proj = None
    return model


def _projector_weights(model):
    return {
        "prefix_gru.weight_ih_l0": torch.randn_like(model.prefix_gru.weight_ih_l0),
        "prefix_gru.weight_hh_l0": torch.randn_like(model.prefix_gru.weight_hh_l0),
        "embed_proj.0.weight": torch.randn_like(model.embed_proj[0].weight),
        "embed_proj.2.weight": torch.randn_like(model.embed_proj[2].weight),
    }


class TestDFlashDominoConfig(CustomTestCase):
    def test_public_config_fields(self):
        parsed = parse_dflash_draft_config(draft_hf_config=_domino_config())
        self.assertTrue(parsed.is_domino)
        self.assertTrue(parsed.shift_label)
        self.assertEqual(parsed.pure_draft_prefix_len, 1)
        self.assertEqual(parsed.gru_hidden_dim, 4)
        self.assertEqual(parsed.emb_dim, 5)
        self.assertEqual(parsed.block_size, 16)

    def test_top_level_emb_dim_fallback(self):
        config = _domino_config()
        del config.dflash_config["emb_dim"]
        self.assertEqual(parse_dflash_draft_config(draft_hf_config=config).emb_dim, 5)

    def test_plain_dflash_and_other_projectors_are_unchanged(self):
        for projector_type in (None, "linear", "dspark"):
            with self.subTest(projector_type=projector_type):
                config = _domino_config(
                    dflash_config={
                        "projector_type": projector_type,
                        "shift_label": None,
                        "pure_draft_prefix_len": None,
                        "gru_hidden_dim": None,
                        "emb_dim": None,
                    },
                    emb_dim=None,
                )
                parsed = parse_dflash_draft_config(draft_hf_config=config)
                self.assertFalse(parsed.is_domino)

    def test_invalid_domino_config_fails_fast(self):
        cases = {
            "shift_label": {"shift_label": 1},
            "pure_draft_prefix_len": {"pure_draft_prefix_len": 2},
            "gru_hidden_dim": {"gru_hidden_dim": None},
        }
        for expected, updates in cases.items():
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    parse_dflash_draft_config(
                        draft_hf_config=_domino_config(dflash_config=updates)
                    )

        config = _domino_config(dflash_config={"emb_dim": None}, emb_dim=None)
        with self.assertRaisesRegex(ValueError, "emb_dim"):
            parse_dflash_draft_config(draft_hf_config=config)

        with self.assertRaisesRegex(ValueError, "block_size > 1"):
            parse_dflash_draft_config(draft_hf_config=_domino_config(block_size=1))

    def test_conflicting_emb_dim_fails(self):
        with self.assertRaisesRegex(ValueError, "emb_dim differs"):
            parse_dflash_draft_config(draft_hf_config=_domino_config(emb_dim=6))


class TestDFlashDominoWeights(CustomTestCase):
    def test_projector_weights_load_exactly(self):
        model = _projector_model()
        weights = _projector_weights(model)
        model.load_weights(weights.items())
        for name, expected in weights.items():
            torch.testing.assert_close(
                dict(model.named_parameters())[name], expected, rtol=0, atol=0
            )

    def test_each_required_projector_weight_is_checked(self):
        for missing_name in _projector_weights(_projector_model()):
            with self.subTest(missing_name=missing_name):
                model = _projector_model()
                weights = _projector_weights(model)
                del weights[missing_name]
                with self.assertRaisesRegex(ValueError, missing_name):
                    model.load_weights(weights.items())

    def test_projector_shape_mismatch_fails(self):
        model = _projector_model()
        weights = _projector_weights(model)
        weights["embed_proj.2.weight"] = torch.empty(30, 5)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            model.load_weights(weights.items())

    def test_plain_dflash_does_not_require_projector_weights(self):
        _projector_model(projector_type=None).load_weights([])

    def test_projector_weights_require_domino_config(self):
        model = _projector_model(projector_type="domnio")
        with self.assertRaisesRegex(ValueError, "projector_type"):
            model.load_weights([("prefix_gru.weight_ih_l0", torch.empty(12, 8))])


class TestDFlashDominoRollout(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.hidden_size = 8
        self.gru_hidden_size = 4
        self.vocab_size = 31
        self.embedding = nn.Embedding(
            self.vocab_size, self.hidden_size, device=self.device, dtype=self.dtype
        )
        self.prefix_gru = nn.GRU(
            self.hidden_size,
            self.gru_hidden_size,
            batch_first=True,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )
        self.embed_proj = nn.Sequential(
            nn.Linear(
                self.hidden_size + self.gru_hidden_size,
                5,
                bias=False,
                device=self.device,
                dtype=self.dtype,
            ),
            nn.SiLU(),
            nn.Linear(
                5,
                self.vocab_size,
                bias=False,
                device=self.device,
                dtype=self.dtype,
            ),
        )
        self.lm_head_weight = torch.randn(
            self.vocab_size,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )

    def _oracle(
        self, draft_hidden, verified_ids, shift_label, candidate_pool_size=None
    ):
        num_proposals = draft_hidden.shape[1] - 1
        start = 0 if shift_label else 1
        z = draft_hidden[:, start : start + num_proposals]
        base_logits = F.linear(z, self.lm_head_weight)
        first_ids = torch.argmax(base_logits[:, 0], dim=-1)
        output = [first_ids]
        if num_proposals == 1:
            return first_ids[:, None]

        candidate_ids = None
        candidate_base = None
        candidate_weight = None
        if candidate_pool_size is not None:
            candidate_pool_size = min(candidate_pool_size, self.vocab_size)
            if 0 < candidate_pool_size < self.vocab_size:
                feedback_logits = base_logits[:, 1:]
                candidate_ids = torch.topk(
                    feedback_logits.amax(dim=1),
                    k=candidate_pool_size,
                    dim=-1,
                    sorted=False,
                ).indices
                candidate_base = torch.gather(
                    feedback_logits,
                    2,
                    candidate_ids[:, None, :].expand(-1, num_proposals - 1, -1),
                )
                candidate_weight = F.embedding(candidate_ids, self.embed_proj[2].weight)

        prefix_ids = torch.cat((verified_ids[:, None], first_ids[:, None]), dim=1)
        _, state = self.prefix_gru(self.embedding(prefix_ids))
        for index in range(1, num_proposals):
            correction_hidden = self.embed_proj[1](
                self.embed_proj[0](torch.cat((z[:, index], state[0]), dim=-1))
            )
            if candidate_ids is None:
                bias = self.embed_proj[2](correction_hidden)
                next_ids = torch.argmax(base_logits[:, index] + bias, dim=-1)
            else:
                bias = torch.bmm(
                    candidate_weight, correction_hidden.unsqueeze(-1)
                ).squeeze(-1)
                candidate_position = torch.argmax(
                    candidate_base[:, index - 1] + bias, dim=-1
                )
                next_ids = torch.gather(
                    candidate_ids, 1, candidate_position[:, None]
                ).squeeze(1)
            output.append(next_ids)
            if index + 1 < num_proposals:
                _, state = self.prefix_gru(self.embedding(next_ids[:, None]), state)
        return torch.stack(output, dim=1)

    def test_full_chain_matches_native_oracle(self):
        for block_size in (2, 16):
            for shift_label in (True, False):
                with self.subTest(block_size=block_size, shift_label=shift_label):
                    draft_hidden = torch.randn(
                        2,
                        block_size,
                        self.hidden_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    verified_ids = torch.tensor([2, 7], device=self.device)
                    actual = domino_greedy_rollout(
                        draft_hidden=draft_hidden,
                        verified_ids=verified_ids,
                        target_embedding=self.embedding,
                        lm_head_weight=self.lm_head_weight,
                        prefix_gru=self.prefix_gru,
                        embed_proj=self.embed_proj,
                        vocab_size=self.vocab_size,
                        shift_label=shift_label,
                    )
                    expected = self._oracle(
                        draft_hidden, verified_ids, shift_label=shift_label
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(actual.shape, (2, block_size - 1))

                    first_hidden = draft_hidden[:, 0 if shift_label else 1]
                    first_expected = torch.argmax(
                        F.linear(first_hidden, self.lm_head_weight), dim=-1
                    )
                    torch.testing.assert_close(
                        actual[:, 0], first_expected, rtol=0, atol=0
                    )

    def test_tp2_gathered_base_matches_full_vocab_rollout(self):
        class FakeTpGroup:
            world_size = 2

            def __init__(self, remote_logits):
                self.remote_logits = remote_logits
                self.candidate_ids = None
                self.local_top_scores = None
                self.remote_top_scores = None
                self.full_logit_gathers = 0
                self.candidate_all_reduces = 0

            def all_gather_into_tensor(self, output, local):
                output[: local.shape[0]].copy_(local)
                if local.ndim == 1:
                    remote_max, remote_pos = self.remote_logits[0, :, :15].max(-1)
                    remote = (
                        remote_pos + 16 if local.dtype == torch.long else remote_max
                    )
                elif local.shape[0] == local_vocab_size:
                    self.full_logit_gathers += 1
                    remote = self.remote_logits.reshape(-1, local_vocab_size).T
                else:
                    k = local.shape[1]
                    remote_scores, remote_pos = torch.topk(
                        self.remote_logits[1:, :, :15].amax(0),
                        k=k,
                        dim=-1,
                        sorted=False,
                    )
                    if local.dtype == torch.long:
                        remote = remote_pos + 16
                        all_scores = (
                            torch.stack(
                                (self.local_top_scores, self.remote_top_scores), dim=0
                            )
                            .permute(1, 0, 2)
                            .reshape(local.shape[0], -1)
                        )
                        all_ids = (
                            torch.stack((local, remote), dim=0)
                            .permute(1, 0, 2)
                            .reshape(local.shape[0], -1)
                        )
                        global_pos = torch.topk(
                            all_scores, k=k, dim=-1, sorted=False
                        ).indices
                        self.candidate_ids = torch.gather(all_ids, 1, global_pos)
                    else:
                        remote = remote_scores
                        self.local_top_scores = local.clone()
                        self.remote_top_scores = remote_scores
                output[local.shape[0] :].copy_(remote)

            def all_reduce(self, local):
                self.candidate_all_reduces += 1
                remote_owned = (self.candidate_ids >= 16) & (self.candidate_ids < 31)
                remote_pos = (self.candidate_ids - 16).clamp(0, 14)
                remote = torch.gather(
                    self.remote_logits[1:].transpose(0, 1),
                    2,
                    remote_pos[:, None, :].expand(
                        -1, self.remote_logits.shape[0] - 1, -1
                    ),
                )
                remote.masked_fill_(~remote_owned[:, None, :], 0)
                local.add_(remote)
                return local

        block_size = 7
        local_vocab_size = 16
        padded_weight = torch.cat(
            (
                self.lm_head_weight,
                torch.zeros(
                    1,
                    self.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                ).fill_(1000),
            )
        )
        local_weight = padded_weight[:local_vocab_size]
        remote_weight = padded_weight[local_vocab_size:]
        cases = (
            (8, 5, None, 1 << 60, False),
            (8, 5, None, 0, True),
            (1, 5, False, 1 << 60, False),
            (8, 5, True, 1 << 60, True),
            (8, 0, True, 1 << 60, False),
        )
        for (
            batch_size,
            candidate_pool_size,
            prefer_tp_candidate_pool,
            full_base_logits_max_bytes,
            expect_compact,
        ) in cases:
            verified_ids = torch.arange(batch_size, device=self.device)
            for shift_label in (True, False):
                with self.subTest(
                    batch_size=batch_size,
                    shift_label=shift_label,
                    candidate_pool_size=candidate_pool_size,
                    prefer_tp_candidate_pool=prefer_tp_candidate_pool,
                    full_base_logits_max_bytes=full_base_logits_max_bytes,
                    expect_compact=expect_compact,
                ):
                    draft_hidden = torch.randn(
                        batch_size,
                        block_size,
                        self.hidden_size,
                        device=self.device,
                        dtype=self.dtype,
                    )
                    start = 0 if shift_label else 1
                    z = draft_hidden[:, start : start + block_size - 1]
                    logits_input = (
                        z.transpose(0, 1)
                        .contiguous()
                        .view((block_size - 1) * batch_size, self.hidden_size)
                    )
                    remote_logits = F.linear(logits_input, remote_weight).view(
                        block_size - 1, batch_size, local_vocab_size
                    )
                    tp_group = FakeTpGroup(remote_logits)
                    rollout_kwargs = dict(
                        draft_hidden=draft_hidden,
                        verified_ids=verified_ids,
                        target_embedding=self.embedding,
                        lm_head_weight=local_weight,
                        prefix_gru=self.prefix_gru,
                        embed_proj=self.embed_proj,
                        vocab_size=self.vocab_size,
                        shift_label=shift_label,
                        candidate_pool_size=candidate_pool_size,
                        tp_group=tp_group,
                        lm_head_num_org=local_vocab_size,
                        lm_head_num_org_padded=local_vocab_size,
                        prefer_tp_candidate_pool=prefer_tp_candidate_pool,
                    )
                    with mock.patch(
                        "sglang.srt.speculative.domino_utils."
                        "_DOMINO_TP_FULL_BASE_LOGITS_MAX_BYTES",
                        full_base_logits_max_bytes,
                    ):
                        actual = domino_greedy_rollout(**rollout_kwargs)
                    expected = self._oracle(
                        draft_hidden,
                        verified_ids,
                        shift_label=shift_label,
                        candidate_pool_size=candidate_pool_size,
                    )
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertEqual(
                        tp_group.candidate_all_reduces, int(expect_compact)
                    )
                    self.assertEqual(
                        tp_group.full_logit_gathers, int(not expect_compact)
                    )

    def test_capture_sampler_uses_capture_bucket_policy(self):
        from sglang.srt.speculative.dflash_worker_v2 import _DominoDraftSampler

        block_size = 7
        sampler = _DominoDraftSampler(
            target_embedding=self.embedding,
            lm_head_weight=self.lm_head_weight,
            prefix_gru=self.prefix_gru,
            embed_proj=self.embed_proj,
            vocab_size=self.vocab_size,
            block_size=block_size,
            shift_label=True,
            max_bs=8,
            candidate_pool_size=5,
        )
        for batch_size, expect_compact in ((1, False), (8, True)):
            with self.subTest(capture_bucket=batch_size, expect_compact=expect_compact):
                hidden_states = torch.randn(
                    batch_size * block_size,
                    self.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                input_ids = torch.zeros(
                    batch_size * block_size,
                    device=self.device,
                    dtype=torch.long,
                )
                proposals = torch.zeros(
                    batch_size,
                    block_size - 1,
                    device=self.device,
                    dtype=torch.long,
                )
                with mock.patch(
                    "sglang.srt.speculative.dflash_worker_v2.domino_greedy_rollout",
                    return_value=proposals,
                ) as rollout:
                    sampler(hidden_states, input_ids)
                self.assertEqual(
                    rollout.call_args.kwargs["prefer_tp_candidate_pool"],
                    expect_compact,
                )

    def test_block_candidate_pool_matches_oracle(self):
        block_size = 7
        draft_hidden = torch.randn(
            3,
            block_size,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        verified_ids = torch.tensor([1, 4, 9], device=self.device)
        for shift_label in (True, False):
            with self.subTest(shift_label=shift_label):
                actual = domino_greedy_rollout(
                    draft_hidden=draft_hidden,
                    verified_ids=verified_ids,
                    target_embedding=self.embedding,
                    lm_head_weight=self.lm_head_weight,
                    prefix_gru=self.prefix_gru,
                    embed_proj=self.embed_proj,
                    vocab_size=self.vocab_size,
                    shift_label=shift_label,
                    candidate_pool_size=5,
                )
                expected = self._oracle(
                    draft_hidden,
                    verified_ids,
                    shift_label=shift_label,
                    candidate_pool_size=5,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                first_hidden = draft_hidden[:, 0 if shift_label else 1]
                first_expected = torch.argmax(
                    F.linear(first_hidden, self.lm_head_weight), dim=-1
                )
                torch.testing.assert_close(actual[:, 0], first_expected, rtol=0, atol=0)

    def test_candidate_pool_boundaries(self):
        draft_hidden = torch.randn(
            2, 7, self.hidden_size, device=self.device, dtype=self.dtype
        )
        verified_ids = torch.tensor([2, 7], device=self.device)
        expected = self._oracle(draft_hidden, verified_ids, shift_label=True)
        for candidate_pool_size in (0, self.vocab_size, self.vocab_size + 1):
            with self.subTest(candidate_pool_size=candidate_pool_size):
                actual = domino_greedy_rollout(
                    draft_hidden=draft_hidden,
                    verified_ids=verified_ids,
                    target_embedding=self.embedding,
                    lm_head_weight=self.lm_head_weight,
                    prefix_gru=self.prefix_gru,
                    embed_proj=self.embed_proj,
                    vocab_size=self.vocab_size,
                    shift_label=True,
                    candidate_pool_size=candidate_pool_size,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            domino_greedy_rollout(
                draft_hidden=draft_hidden,
                verified_ids=verified_ids,
                target_embedding=self.embedding,
                lm_head_weight=self.lm_head_weight,
                prefix_gru=self.prefix_gru,
                embed_proj=self.embed_proj,
                vocab_size=self.vocab_size,
                shift_label=True,
                candidate_pool_size=-1,
            )

    def test_batch_rollout_matches_each_individual_row(self):
        draft_hidden = torch.randn(
            3, 7, self.hidden_size, device=self.device, dtype=self.dtype
        )
        verified_ids = torch.tensor([1, 4, 9], device=self.device)
        for candidate_pool_size in (0, 5):
            with self.subTest(candidate_pool_size=candidate_pool_size):
                batched = domino_greedy_rollout(
                    draft_hidden=draft_hidden,
                    verified_ids=verified_ids,
                    target_embedding=self.embedding,
                    lm_head_weight=self.lm_head_weight,
                    prefix_gru=self.prefix_gru,
                    embed_proj=self.embed_proj,
                    vocab_size=self.vocab_size,
                    shift_label=True,
                    candidate_pool_size=candidate_pool_size,
                )
                individual = torch.cat(
                    [
                        domino_greedy_rollout(
                            draft_hidden=draft_hidden[index : index + 1],
                            verified_ids=verified_ids[index : index + 1],
                            target_embedding=self.embedding,
                            lm_head_weight=self.lm_head_weight,
                            prefix_gru=self.prefix_gru,
                            embed_proj=self.embed_proj,
                            vocab_size=self.vocab_size,
                            shift_label=True,
                            candidate_pool_size=candidate_pool_size,
                        )
                        for index in range(3)
                    ],
                    dim=0,
                )
                torch.testing.assert_close(batched, individual, rtol=0, atol=0)

    def test_capture_sampler_matches_eager_rollout(self):
        from sglang.srt.speculative.dflash_worker_v2 import _DominoDraftSampler

        block_size = 7
        batch_size = 3
        for shift_label in (True, False):
            with self.subTest(shift_label=shift_label):
                draft_hidden = torch.randn(
                    batch_size,
                    block_size,
                    self.hidden_size,
                    device=self.device,
                    dtype=self.dtype,
                )
                verified_ids = torch.tensor([1, 4, 9], device=self.device)
                block_ids = torch.zeros(
                    batch_size,
                    block_size,
                    device=self.device,
                    dtype=torch.long,
                )
                block_ids[:, 0].copy_(verified_ids)
                sampler = _DominoDraftSampler(
                    target_embedding=self.embedding,
                    lm_head_weight=self.lm_head_weight,
                    prefix_gru=self.prefix_gru,
                    embed_proj=self.embed_proj,
                    vocab_size=self.vocab_size,
                    block_size=block_size,
                    shift_label=shift_label,
                    max_bs=batch_size,
                    candidate_pool_size=5,
                )
                sampler(draft_hidden.reshape(-1, self.hidden_size), block_ids.flatten())
                expected = domino_greedy_rollout(
                    draft_hidden=draft_hidden,
                    verified_ids=verified_ids,
                    target_embedding=self.embedding,
                    lm_head_weight=self.lm_head_weight,
                    prefix_gru=self.prefix_gru,
                    embed_proj=self.embed_proj,
                    vocab_size=self.vocab_size,
                    shift_label=shift_label,
                    candidate_pool_size=5,
                )
                actual = sampler.out[: batch_size * (block_size - 1)].view(
                    batch_size, block_size - 1
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_candidate_pool_cuda_graph_replay(self):
        block_size = 7
        batch_size = 3
        static_hidden = torch.randn(
            batch_size,
            block_size,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        static_ids = torch.tensor([1, 4, 9], device=self.device)

        def rollout():
            return domino_greedy_rollout(
                draft_hidden=static_hidden,
                verified_ids=static_ids,
                target_embedding=self.embedding,
                lm_head_weight=self.lm_head_weight,
                prefix_gru=self.prefix_gru,
                embed_proj=self.embed_proj,
                vocab_size=self.vocab_size,
                shift_label=True,
                candidate_pool_size=5,
            )

        rollout()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = rollout()

        static_hidden.copy_(torch.randn_like(static_hidden))
        static_ids.copy_(torch.tensor([2, 7, 11], device=self.device))
        expected = self._oracle(
            static_hidden,
            static_ids,
            shift_label=True,
            candidate_pool_size=5,
        )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_capture_sampler_replays_with_new_inputs(self):
        from sglang.srt.speculative.dflash_worker_v2 import _DominoDraftSampler

        block_size = 4
        batch_size = 2
        sampler = _DominoDraftSampler(
            target_embedding=self.embedding,
            lm_head_weight=self.lm_head_weight,
            prefix_gru=self.prefix_gru,
            embed_proj=self.embed_proj,
            vocab_size=self.vocab_size,
            block_size=block_size,
            shift_label=True,
            max_bs=batch_size,
            candidate_pool_size=5,
        )
        static_hidden = torch.randn(
            batch_size * block_size,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        static_ids = torch.zeros(
            batch_size * block_size, device=self.device, dtype=torch.long
        )
        static_ids.view(batch_size, block_size)[:, 0].copy_(
            torch.tensor([2, 7], device=self.device)
        )

        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            sampler(static_hidden, static_ids)
        torch.cuda.current_stream().wait_stream(warmup_stream)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            sampler(static_hidden, static_ids)

        new_hidden = torch.randn_like(static_hidden)
        new_ids = torch.zeros_like(static_ids)
        new_ids.view(batch_size, block_size)[:, 0].copy_(
            torch.tensor([3, 11], device=self.device)
        )
        static_hidden.copy_(new_hidden)
        static_ids.copy_(new_ids)
        expected = domino_greedy_rollout(
            draft_hidden=new_hidden.view(batch_size, block_size, self.hidden_size),
            verified_ids=new_ids.view(batch_size, block_size)[:, 0],
            target_embedding=self.embedding,
            lm_head_weight=self.lm_head_weight,
            prefix_gru=self.prefix_gru,
            embed_proj=self.embed_proj,
            vocab_size=self.vocab_size,
            shift_label=True,
            candidate_pool_size=5,
        )
        graph.replay()
        torch.cuda.synchronize()

        actual = sampler.out[: batch_size * (block_size - 1)].view(
            batch_size, block_size - 1
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class TestDFlashDominoRuntimeValidation(CustomTestCase):
    def _modules(self, dtype=torch.bfloat16):
        embedding = nn.Embedding(31, 8, dtype=dtype)
        lm_head = nn.Linear(8, 31, bias=False, dtype=dtype)
        prefix_gru = nn.GRU(8, 4, batch_first=True, bias=False, dtype=dtype)
        embed_proj = nn.Sequential(
            nn.Linear(12, 5, bias=False, dtype=dtype),
            nn.SiLU(),
            nn.Linear(5, 31, bias=False, dtype=dtype),
        )
        return embedding, lm_head, prefix_gru, embed_proj

    def _tp2_modules(self):
        embedding, lm_head, prefix_gru, embed_proj = self._modules()
        embedding = nn.Embedding(16, 8, dtype=torch.bfloat16)
        lm_head = nn.Linear(8, 16, bias=False, dtype=torch.bfloat16)
        shard = SimpleNamespace(
            num_added_elements=0,
            org_vocab_start_index=0,
            org_vocab_end_index=16,
            num_org_elements=16,
            num_org_elements_padded=16,
        )
        for module in (embedding, lm_head):
            module.shard_indices = shard
            module.org_vocab_size = 31
            module.tp_size = 2
            module.num_added_embeddings = 0
        return embedding, lm_head, prefix_gru, embed_proj

    def _validate(self, **overrides):
        embedding, lm_head, prefix_gru, embed_proj = overrides.pop(
            "modules", self._modules()
        )
        args = {
            "device": torch.device("cuda"),
            "tp_size": 1,
            "tp_rank": 0,
            "target_vocab_size": 31,
            "draft_vocab_size": 31,
            "hidden_size": 8,
            "target_embedding": embedding,
            "lm_head": lm_head,
            "prefix_gru": prefix_gru,
            "embed_proj": embed_proj,
        }
        args.update(overrides)
        validate_domino_runtime(**args)

    def test_supported_runtime(self):
        self._validate()

    def test_tp_requires_vocab_shard_metadata(self):
        with self.assertRaisesRegex(ValueError, "lm_head shard metadata"):
            self._validate(tp_size=2)

    def test_tp2_vocab_shards_supported(self):
        self._validate(tp_size=2, modules=self._tp2_modules())

    def test_tp2_incomplete_lm_head_shard_fails(self):
        modules = self._tp2_modules()
        modules[1].shard_indices = SimpleNamespace(
            num_added_elements=0,
            num_org_elements_padded=16,
        )
        with self.assertRaisesRegex(ValueError, "shard metadata is missing"):
            self._validate(tp_size=2, modules=modules)

    def test_tp_vocab_shard_must_match_rank(self):
        modules = self._tp2_modules()
        modules[1].shard_indices.org_vocab_start_index = 1
        modules[1].shard_indices.org_vocab_end_index = 17
        with self.assertRaisesRegex(ValueError, "does not match its TP rank"):
            self._validate(tp_size=2, modules=modules)

    def test_tp1_requires_complete_vocab_shard(self):
        modules = self._modules()
        modules[1].shard_indices = SimpleNamespace(
            num_added_elements=0,
            org_vocab_start_index=0,
            org_vocab_end_index=30,
            num_org_elements=30,
            num_org_elements_padded=31,
        )
        modules[1].org_vocab_size = 31
        modules[1].tp_size = 1
        modules[1].num_added_embeddings = 0
        with self.assertRaisesRegex(ValueError, "does not match its TP rank"):
            self._validate(modules=modules)

    def test_vocab_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "identical target and draft"):
            self._validate(draft_vocab_size=30)

    def test_added_vocab_fails(self):
        modules = self._modules()
        modules[1].shard_indices = SimpleNamespace(
            num_added_elements=1,
            org_vocab_start_index=0,
            num_org_elements=31,
        )
        with self.assertRaisesRegex(ValueError, "added-vocab"):
            self._validate(modules=modules)

    def test_embedding_vocab_mismatch_fails(self):
        _, lm_head, prefix_gru, embed_proj = self._modules()
        embedding = nn.Embedding(30, 8, dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "fewer rows"):
            self._validate(modules=(embedding, lm_head, prefix_gru, embed_proj))

    def test_non_bf16_fails(self):
        with self.assertRaisesRegex(ValueError, "BF16"):
            self._validate(modules=self._modules(dtype=torch.float32))


if __name__ == "__main__":
    unittest.main()
