"""Guard: aiter refuses an MLA draft model under decode context parallelism.

An MLA draft (EAGLE, NEXTN, STANDALONE) with ``--dcp-size > 1`` crashed at
startup inside the aiter extend with an opaque ``AttributeError`` ("'NoneType'
object has no attribute 'shape'"): the MLA model code sends the draft extend
through its DCP decode branch without the step's keys. The guard fails loudly
instead. It is inert for targets, non-MLA drafts, DCP 1, and ROCm DSA drafts that
shard their KV like the target.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.configs import model_config as model_config_module
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention import aiter_backend
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_NEXTN_ARCH = "DeepseekV3ForCausalLMNextN"


def _draft_model_config(**hf_fields):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[_NEXTN_ARCH], **hf_fields)
    )


def _reject(**overrides):
    kwargs = dict(
        use_mla=True,
        dcp_world_size=2,
        is_draft_worker=True,
        model_config=_draft_model_config(),
    )
    kwargs.update(overrides)
    AiterAttnBackend._reject_mla_draft_under_dcp(**kwargs)


class _GuardReached(Exception):
    pass


def _mla_model_runner(*, is_draft_worker):
    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        is_multimodal=False,
        num_attention_heads=16,
        head_dim=576,
        get_num_kv_heads=lambda _tp_size: 1,
        attention_arch=AttentionArch.MLA,
    )
    return SimpleNamespace(
        model_config=model_config,
        device=torch.device("cpu"),
        kv_cache_dtype=torch.bfloat16,
        req_to_token_pool=SimpleNamespace(req_to_token=None),
        runner_ids=None,
        is_draft_worker=is_draft_worker,
    )


class TestAiterMlaDraftDcpGuard(CustomTestCase):
    def test_raises_for_an_mla_draft_under_dcp(self):
        with self.assertRaises(ValueError) as ctx:
            _reject()
        self.assertIn("--dcp-size 2", str(ctx.exception))
        self.assertIn("MLA draft", str(ctx.exception))

    def test_inert_for_targets_non_mla_drafts_and_dcp1(self):
        # These never read hf_config, so a model config without one still builds.
        bare = SimpleNamespace()
        _reject(is_draft_worker=False, model_config=bare)
        _reject(use_mla=False, model_config=bare)
        _reject(dcp_world_size=1, model_config=bare)

    def test_inert_only_for_a_rocm_dsa_draft_that_shards_like_the_target(self):
        with patch.object(model_config_module, "is_hip", return_value=True):
            _reject(model_config=_draft_model_config(index_topk=2048))
            # The same NextN arch without an indexer is a plain MLA draft.
            with self.assertRaises(ValueError):
                _reject()

    def test_init_hands_the_runner_role_and_dcp_size_to_the_guard(self):
        with (
            patch.object(
                aiter_backend,
                "get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1, attn_dcp_size=2),
            ),
            patch.object(
                aiter_backend,
                "get_schedule",
                return_value=SimpleNamespace(page_size=1),
            ),
            patch.object(
                aiter_backend,
                "get_spec",
                return_value=SimpleNamespace(
                    speculative_num_draft_tokens=None, speculative_num_steps=None
                ),
            ),
        ):
            for is_draft_worker in (True, False):
                runner = _mla_model_runner(is_draft_worker=is_draft_worker)
                guard = MagicMock(side_effect=_GuardReached)
                with (
                    self.subTest(is_draft_worker=is_draft_worker),
                    patch.object(
                        AiterAttnBackend, "_reject_mla_draft_under_dcp", guard
                    ),
                ):
                    with self.assertRaises(_GuardReached):
                        AiterAttnBackend(runner)
                    guard.assert_called_once_with(
                        use_mla=True,
                        dcp_world_size=2,
                        is_draft_worker=is_draft_worker,
                        model_config=runner.model_config,
                    )


if __name__ == "__main__":
    unittest.main()
