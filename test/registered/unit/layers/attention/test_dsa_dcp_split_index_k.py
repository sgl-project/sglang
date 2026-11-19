import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import LlamaConfig

from sglang.srt.configs.model_config import (
    ModelConfig,
    dsa_draft_shards_under_dcp,
    glm_dsa_shards_index_k,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla import (
    DeepseekMLAForwardMixin,
)
from sglang.srt.runtime_context import override_platform
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

FORWARD_MLA = "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla"
EXTEND_MODES = [
    ForwardMode.EXTEND,
    ForwardMode.TARGET_VERIFY,
    ForwardMode.DRAFT_EXTEND_V2,
]


def _config(arch: str):
    return SimpleNamespace(architectures=[arch], index_topk=2048)


def _backend(*, shards_index_k: bool, page_table_1_flattened=None):
    backend = DeepseekSparseAttnBackend.__new__(DeepseekSparseAttnBackend)
    backend.dsa_shards_index_k = shards_index_k
    backend.device = torch.device("cpu")
    backend._arange_buf = torch.arange(0, dtype=torch.int32)
    backend.forward_metadata = SimpleNamespace(
        page_table_1_flattened=page_table_1_flattened
    )
    return backend


@pytest.mark.parametrize(
    "arch,is_sm90,dcp_size,expected",
    [
        ("GlmMoeDsaForCausalLM", True, 4, True),
        ("GlmMoeDsaForCausalLMNextN", True, 4, True),
        ("GlmMoeDsaForCausalLM", False, 4, False),
        ("GlmMoeDsaForCausalLM", True, 1, False),
        ("DeepseekV32ForCausalLM", True, 4, False),
    ],
)
def test_glm_shards_index_k_only_under_dcp_on_sm90(arch, is_sm90, dcp_size, expected):
    with override_platform(is_sm90=is_sm90, is_sm100=not is_sm90):
        assert glm_dsa_shards_index_k(_config(arch), dcp_size) is expected
        # The draft must shard exactly when its target does.
        assert dsa_draft_shards_under_dcp(_config(arch), dcp_size) is expected


def test_unset_context_length_is_capped_only_under_split_index_k():
    """The top-k merge cannot take a row as long as GLM-5.3's own context, so
    a server started without --context-length must come up under the limit."""
    model_context_len = 1 << 20
    with TemporaryDirectory() as checkpoint:
        LlamaConfig(
            architectures=["GlmMoeDsaForCausalLM"],
            max_position_embeddings=model_context_len,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_hidden_layers=2,
            vocab_size=128,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
        ).save_pretrained(checkpoint)

        def context_len(*, is_sm90=True, **server_args):
            with override_platform(is_sm90=is_sm90, is_sm100=not is_sm90):
                return ModelConfig.from_server_args(
                    ServerArgs(model_path=checkpoint, page_size=64, **server_args)
                ).context_len

        # The request table over-allocates four tokens without a draft.
        assert context_len(dcp_size=4) == model_context_len - 4
        assert context_len() == model_context_len
        assert context_len(is_sm90=False, dcp_size=4) == model_context_len
        assert context_len(dcp_size=4, context_length=4096) == 4096
        with pytest.raises(ValueError, match="--context-length 1048572"):
            context_len(dcp_size=4, context_length=model_context_len)
        # A long draft reserves more than one allocator page of the table.
        long_draft = dict(
            speculative_algorithm="NEXTN",
            speculative_num_steps=64,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=65,
        )
        assert context_len(dcp_size=2, **long_draft) == model_context_len - 257


@pytest.mark.parametrize("mode", [ForwardMode.EXTEND, ForwardMode.MIXED])
def test_flashmla_kv_prefill_top_k_is_ragged_under_split_index_k(mode):
    """A mixed batch prefills from the gathered KV like an extend, so its top-k
    must address that buffer too; a paged top-k would export widened slots as
    the draft's seed."""
    backend = _backend(shards_index_k=True)
    backend.dsa_kv_cache_store_fp8 = True
    backend.dsa_prefill_impl = "flashmla_kv"
    assert backend.get_topk_transform_method(mode) == TopkTransformMethod.RAGGED


def test_prefix_rows_index_the_ragged_gathered_kv_under_split_index_k():
    """Under split index-K the prefix KV is the buffer dcp_gather_kv_cache
    returns in position order, so its rows are 0..n-1, not the DCP-widened
    slot ids page_table_1_flattened holds."""
    widened_slots = torch.tensor([8, 9, 10, 40, 41], dtype=torch.int32)
    gathered = torch.arange(len(widened_slots)) * 10
    backend = _backend(shards_index_k=True, page_table_1_flattened=widened_slots)
    assert torch.equal(gathered[backend._ragged_prefix_kv_rows()], gathered)


def test_prefix_rows_stay_the_slot_table_without_split_index_k():
    slots = torch.tensor([3, 4, 7], dtype=torch.int32)
    backend = _backend(shards_index_k=False, page_table_1_flattened=slots)
    assert torch.equal(backend._ragged_prefix_kv_rows(), slots)


@pytest.mark.parametrize("mode", EXTEND_MODES, ids=lambda m: m.name)
def test_forward_mla_merges_exactly_the_extends_that_return_dcp_partials(mode):
    """Under split index-K an extend returns (o, lse) partials exactly when the
    backend attends its local KV shard, so forward_mla must merge across DCP
    ranks for those modes and for no other extend."""
    backend = _backend(shards_index_k=True)
    layer = SimpleNamespace(dcp_size=4, use_dsa=True)
    with patch(f"{FORWARD_MLA}.dsa_shards_index_k", return_value=True):
        merged = DeepseekMLAForwardMixin.is_dcp_mla_decode_phase(
            layer, SimpleNamespace(forward_mode=mode)
        )
    assert merged == backend._attends_local_kv_shard(mode)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
