import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.speculative.ragged_verify import (
    ragged_verify_dense_scatter_indices,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestReplaySSMGDN(CustomTestCase):
    def test_resolver_selects_only_folded_kda_or_gdn(self):
        mamba_pool = SimpleNamespace(
            replayssm_spec_fold=True,
            replayssm_is_kda=False,
        )
        req_pool = SimpleNamespace(mamba_pool=mamba_pool)
        gdn_backend = object.__new__(GDNAttnBackend)
        gdn_backend.req_to_token_pool = req_pool
        other_backend = SimpleNamespace(req_to_token_pool=req_pool)

        with patch(
            "sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold."
            "commit_gdn_replayssm_fold_after_verify"
        ) as gdn_commit, patch(
            "sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode."
            "commit_kda_replayssm_after_verify"
        ) as kda_commit:
            self.assertIs(
                HybridLinearAttnBackend._resolve_replayssm_verify_commit(gdn_backend),
                gdn_commit,
            )
            mamba_pool.replayssm_is_kda = True
            with self.assertRaisesRegex(AssertionError, "GDN backend"):
                HybridLinearAttnBackend._resolve_replayssm_verify_commit(gdn_backend)
            self.assertIs(
                HybridLinearAttnBackend._resolve_replayssm_verify_commit(other_backend),
                kda_commit,
            )

        mamba_pool.replayssm_spec_fold = False
        self.assertIsNone(
            HybridLinearAttnBackend._resolve_replayssm_verify_commit(other_backend)
        )
        mamba_pool.replayssm_spec_fold = True
        mamba_pool.replayssm_is_kda = False
        self.assertIsNone(
            HybridLinearAttnBackend._resolve_replayssm_verify_commit(other_backend)
        )
        self.assertIsNone(
            HybridLinearAttnBackend._resolve_replayssm_verify_commit(SimpleNamespace())
        )

    def test_direct_verify_uses_resolved_commit(self):
        mamba_caches = object()
        req_pool = SimpleNamespace(
            get_speculative_mamba2_params_all_layers=MagicMock(
                return_value=mamba_caches
            )
        )
        linear_backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(mamba_cache_indices=torch.tensor([4])),
            req_to_token_pool=req_pool,
            needs_cpu_seq_lens=False,
        )
        full_backend = SimpleNamespace(
            token_to_kv_pool=None,
            req_to_token_pool=req_pool,
            max_context_len=None,
            needs_cpu_seq_lens=False,
        )
        commit = MagicMock()

        with patch.object(
            HybridLinearAttnBackend,
            "_resolve_replayssm_verify_commit",
            return_value=commit,
        ) as resolve:
            backend = HybridLinearAttnBackend(full_backend, linear_backend, [])
            backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=torch.tensor([2]),
                mamba_track_indices=None,
                mamba_steps_to_track=None,
                model=None,
            )

        resolve.assert_called_once_with(linear_backend)
        commit.assert_called_once()
        self.assertIs(commit.call_args.kwargs["spec_state"], mamba_caches)
        self.assertTrue(
            torch.equal(commit.call_args.kwargs["accept_lens"], torch.tensor([3]))
        )

    def test_gdn_supports_compact_ragged_layout(self):
        self.assertTrue(GDNAttnBackend.supports_ragged_verify_graph)
        indices = ragged_verify_dense_scatter_indices(
            query_start_loc=torch.tensor([0, 4, 6, 7], dtype=torch.int32),
            seq_len=8,
            draft_token_num=4,
        )
        self.assertTrue(torch.equal(indices, torch.tensor([0, 1, 2, 3, 4, 5, 8, 12])))


class TestDirectReplaySSMValidation(CustomTestCase):
    @staticmethod
    def _validate(
        *,
        family="gdn",
        spec_algorithm=SpeculativeAlgorithm.DFLASH,
        device="cuda",
        unified_memory=False,
    ):
        configurator = SimpleNamespace(
            is_draft_worker=False,
            spec_algorithm=spec_algorithm,
            hybrid_gdn_config=object() if family == "gdn" else None,
            model_config=object(),
            device=device,
        )
        with patch(
            "sglang.srt.mem_cache.kv_cache_configurator.get_exec",
            return_value=SimpleNamespace(
                mamba=SimpleNamespace(enable_linear_replayssm_spec=True)
            ),
        ), patch(
            "sglang.srt.mem_cache.kv_cache_configurator.get_memory",
            return_value=SimpleNamespace(enable_unified_memory=unified_memory),
        ), patch(
            "sglang.srt.mem_cache.kv_cache_configurator.kimi_linear_config",
            return_value=object() if family == "kda" else None,
        ):
            KVCacheConfigurator._validate_direct_replayssm_spec(configurator)

    def test_direct_verify_guards(self):
        cases = (
            ({"family": "other"}, "GDN or KDA"),
            ({"device": "npu"}, "CUDA"),
            ({"family": "kda", "unified_memory": True}, "unified-memory"),
        )
        for kwargs, message in cases:
            with self.subTest(**kwargs), self.assertRaisesRegex(ValueError, message):
                self._validate(**kwargs)

        self._validate()
        self._validate(
            family="kda",
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
        )
        self._validate(
            family="other",
            spec_algorithm=SpeculativeAlgorithm.NGRAM,
            device="npu",
            unified_memory=True,
        )


if __name__ == "__main__":
    unittest.main()
