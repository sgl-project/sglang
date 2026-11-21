"""The hybrid request pool's speculative scratch must hold a padded verify batch."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.mamba_utils import (
    Mamba2CacheParams,
    Mamba2StateDType,
    Mamba2StateShape,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.linear.utils import max_verify_batch_size
from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.qwen4_exp import (
    Qwen4ExpPLELayer,
    _commit_ple_batch,
    _prepare_ple_batch,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_CHANNELS, _STATE_LEN = 3, 4


def _configurator():
    cfg = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
    cfg.mambaish_config = SimpleNamespace(
        mamba2_cache_params=Mamba2CacheParams(
            shape=Mamba2StateShape.create(
                tp_world_size=1,
                intermediate_size=8,
                n_groups=1,
                num_heads=2,
                head_dim=4,
                state_size=4,
                conv_kernel=4,
            ),
            dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
            layers=[0],
        )
    )
    cfg.model_config = SimpleNamespace(context_len=16)
    cfg.device = "cpu"
    cfg.layer_info = SimpleNamespace(start_layer=0, end_layer=1)
    cfg.hybrid_gdn_config = None
    cfg.hybrid_kda_config = None
    cfg.allocate_intermediate_ssm = True
    return cfg


class TestHybridReqPoolVerifyRows(CustomTestCase):
    def test_ple_verify_writes_fit_padded_batch(self):
        """Without PD, attention DP pads 2 requests per worker to a 4-row verify
        batch; the PLE verify writes index the scratch by those padded rows."""
        size, draft_tokens = 2, 4
        ple_kwargs = dict(
            short_conv_layer_ids=[0],
            short_conv_state_shape=(_CHANNELS, _STATE_LEN),
            ngram_context_len=2,
            ngram_eos_token_id=0,
        )
        with (
            get_context().override_server_args(
                page_size=1,
                speculative_algorithm="NEXTN",
                speculative_num_steps=draft_tokens - 1,
                speculative_eagle_topk=1,
                speculative_num_draft_tokens=draft_tokens,
                max_mamba_cache_size=size,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_parallel",
                return_value=SimpleNamespace(attn_dp_enabled=True, attn_tp_size=4),
            ),
            patch(
                "sglang.srt.layers.cp.padding.get_cp_padding_align_size",
                return_value=1,
            ),
            patch(
                "sglang.srt.layers.dp_attention.get_cuda_graph_batch_size_alignment",
                return_value=4,
            ),
            patch.object(
                kcc.KVCacheConfigurator,
                "_get_ple_req_pool_kwargs",
                return_value=ple_kwargs,
            ),
        ):
            verify_rows = max_verify_batch_size(size)
            pool = _configurator()._build_req_to_token_pool(max_num_reqs=size)
        self.assertEqual(verify_rows, 4)

        router = SimpleNamespace(
            read=lambda layer_id, part: SimpleNamespace(
                view=pool.short_conv_pool, view_layer_id=layer_id
            ),
            wait_for_part=lambda part: None,
            part_view=lambda part: pool.ngram_pool,
        )
        backend = SimpleNamespace(
            req_to_token_pool=pool,
            token_to_kv_pool=SimpleNamespace(layer_router=router),
            # Padded rows resolve to the dummy state slot.
            forward_state_slots=lambda rows, _: SimpleNamespace(
                working=torch.zeros(rows, dtype=torch.int32)
            ),
        )
        tokens = verify_rows * draft_tokens
        fb = SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            _original_forward_mode=None,
            tbo_parent_token_range=None,
            spec_algorithm=None,
            spec_info=SimpleNamespace(topk=1, draft_token_num=draft_tokens),
            global_num_token_non_padded_cpu=None,
            extend_seq_lens=None,
            extend_seq_lens_cpu=None,
            req_pool_indices=torch.zeros(verify_rows, dtype=torch.int64),
            out_cache_loc=torch.zeros(tokens, dtype=torch.int64),
        )
        ple = SimpleNamespace(
            layer_id=0,
            conv_channels=_CHANNELS,
            short_conv_dilation=1,
            short_conv_state_len=_STATE_LEN,
            conv1d=SimpleNamespace(weight=torch.ones(_CHANNELS, 1, _STATE_LEN + 1)),
        )
        with (
            envs.SGLANG_ENABLE_PP_SPEC.override(False),
            forward_context(ForwardContext(attn_backend=backend)),
        ):
            batch = _prepare_ple_batch(
                torch.zeros(tokens, dtype=torch.int64),
                fb,
                ngram_size=3,
                ngram_eos_token_id=0,
            )
            Qwen4ExpPLELayer._short_conv(ple, torch.zeros(tokens, _CHANNELS), fb, batch)
            _commit_ple_batch(batch, fb)
        self.assertEqual(batch.lengths.shape[0], verify_rows)
        # LFM2, lightning and Inkling write the Mamba scratch by position too.
        self.assertEqual(
            pool.mamba_pool.mamba_cache.intermediate_ssm.shape[1], verify_rows + 1
        )


if __name__ == "__main__":
    unittest.main()
