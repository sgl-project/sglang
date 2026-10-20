"""CPU contract of the EAGLE binding: a covered step's view is the running forward's
own attention metadata and pools, read from a hybrid GDN backend; the pools are
sealed at their first read and checked after capture; every struct carries data
only; the tolerated lists name real members."""

import typing
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.megakernels import eagle_binding as eb
from sglang.srt.megakernels import runtime
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def _backend(cls, **fields):
    """A backend of cls holding only the given fields."""
    backend = cls.__new__(cls)
    backend.__dict__.update(fields)
    return backend


class KvPool:
    start_layer, layer_num = 3, 2

    def __init__(self):
        self.k = {i: torch.zeros(4, 2) for i in (3, 4)}

    def get_key_buffer(self, layer_id):
        return self.k[layer_id]

    def get_value_buffer(self, layer_id):
        return self.k[layer_id]


def _req_pool():
    tensor = torch.zeros(2)
    state = SimpleNamespace(
        conv=[tensor],
        temporal=tensor,
        intermediate_conv_window=[tensor],
        intermediate_ssm=tensor,
        **{f"replayssm_{n}": None for n in ("d", "k", "g", "rawv", "rawk", "beta")},
    )
    return SimpleNamespace(
        mamba_map={0: 0, 2: 1},
        get_speculative_mamba2_params_all_layers=lambda: state,
        mamba_pool=SimpleNamespace(
            replayssm_spec_write_pos=None,
            replayssm_cache_base=None,
            replayssm_is_flush=None,
        ),
    )


def _trtllm(pool):
    metadata = SimpleNamespace(
        page_table=torch.zeros(2, 3, dtype=torch.int32),
        cache_seqlens_int32=torch.zeros(2, dtype=torch.int32),
        cu_seqlens_q=torch.zeros(3, dtype=torch.int32),
        cu_seqlens_k=torch.zeros(3, dtype=torch.int32),
        max_seq_len_q=7,
    )
    return _backend(
        TRTLLMHAAttnBackend,
        forward_metadata=metadata,
        token_to_kv_pool=pool,
        data_type=torch.bfloat16,
    )


def _hybrid():
    pool = KvPool()
    gdn = _backend(
        GDNAttnBackend,
        forward_metadata=SimpleNamespace(
            mamba_cache_indices=torch.zeros(2, dtype=torch.int32),
            query_start_loc=torch.zeros(3, dtype=torch.int32),
            mamba_track_indices=None,
        ),
        verify_intermediate_state_indices=torch.arange(2, dtype=torch.int32),
        pad_slot_id=-1,
    )
    return _backend(
        HybridLinearAttnBackend,
        full_attn_backend=_trtllm(pool),
        linear_attn_backend=gdn,
        token_to_kv_pool=pool,
        req_to_token_pool=_req_pool(),
    )


def _view(binding, backend):
    with patch.object(eb, "get_attn_backend", return_value=backend):
        return binding.forward_view()


def _leaves(hint):
    args = typing.get_args(hint)
    if not args:
        return [hint]
    return [leaf for arg in args if arg is not Ellipsis for leaf in _leaves(arg)]


class TestEagleBinding(CustomTestCase):
    def test_a_qwen3_5_view_is_the_running_forward_and_tolerated_members_exist(self):
        """A covered step is captured once and replays SGLang's refills of the
        backends' metadata, so the view must hold those very tensors."""
        for arch in ("Qwen3_5MoeForConditionalGeneration", "Qwen3_5ForCausalLMMTP"):
            self.assertIs(runtime._binding_class([arch]), eb.EagleBinding)
        backend = _hybrid()
        view = _view(eb.EagleBinding(model=torch.nn.Module()), backend)
        attn = backend.full_attn_backend.forward_metadata
        gdn = backend.linear_attn_backend
        self.assertIs(view.attn.cache_seqlens, attn.cache_seqlens_int32)
        self.assertIs(view.attn.cu_seqlens_k, attn.cu_seqlens_k)
        self.assertIs(view.gdn.state_indices, gdn.forward_metadata.mamba_cache_indices)
        self.assertIs(
            view.gdn.intermediate_indices, gdn.verify_intermediate_state_indices
        )
        self.assertEqual(view.kv.layer_ids, (3, 4))
        self.assertIs(view.kv.k[1], backend.token_to_kv_pool.k[4])
        self.assertEqual(view.kv.k_scale, (1.0, 1.0))
        self.assertEqual(view.mamba.layer_ids, (0, 2))
        # a model without GDN layers, such as the MTP draft
        draft = _view(eb.EagleBinding(model=torch.nn.Module()), _trtllm(KvPool()))
        self.assertIsNone(draft.gdn)
        self.assertIsNone(draft.mamba)
        # a plugin pins these names: a rename here must fail first
        members = {
            "EagleBinding": set(dir(eb.EagleBinding)),
            "ForwardBatch": set(ForwardBatch.__dataclass_fields__),
        }
        for name in eb.TOLERATED_CALLABLES + eb.TOLERATED_READS:
            owner, _, attr = name.partition(".")
            if owner in members:
                self.assertIn(attr, members[owner], name)

    def test_a_reallocated_pool_fails_assert_live_by_name(self):
        backend = _hybrid()
        binding = eb.EagleBinding(model=torch.nn.Module())
        first = _view(binding, backend)
        self.assertIs(_view(binding, backend).kv, first.kv)  # sealed at the first read
        with patch.object(eb, "get_attn_backend", return_value=backend):
            binding.assert_live()
            backend.token_to_kv_pool.k[3] = torch.zeros(4, 2)
            with self.assertRaisesRegex(RuntimeError, "kv was reallocated"):
                binding.assert_live()

    def test_entry_structs_carry_data_only(self):
        structs = {
            cls
            for cls in vars(eb).values()
            if isinstance(cls, type) and issubclass(cls, msgspec.Struct)
        }
        allowed = {torch.Tensor, int, float, bool, type(None)} | structs
        for cls in structs:
            for field, hint in typing.get_type_hints(cls).items():
                for leaf in _leaves(hint):
                    self.assertIn(leaf, allowed, f"{cls.__name__}.{field}")


if __name__ == "__main__":
    unittest.main()
