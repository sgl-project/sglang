"""Real construction-level verification of the `allow_routed_experts_capture`
opt-out for MoE-bearing draft families and dense allowlist entries.

The fixture below patches SGLang's INFRASTRUCTURE boundaries (distributed
groups, `_global_server_args`, MoE expert backend class, MoE A2A backend)
without touching the family's MoE block / decoder / model under test.
The real constructor runs, builds a real `TopK` module, and the test
walks `model.modules()` for the resulting flag values.

What this catches that AST/regex cannot:

  - A kwarg dropped at any layer of the constructor chain (the real
    constructor would build a `TopK` with the default `True` flag, and
    the modules() walk would expose it).
  - A future signature change that accidentally inverts the flag default.
  - A dense entry that silently grew an MoE block (the modules() walk
    would find a `TopK` instance where none was expected).

What it cannot reach: families whose constructors require runtime kernel
selection that the CPU host cannot fake (e.g., a backend that conditions
on `torch.cuda.is_available()` before constructing any submodule). The
test suite reports each such family explicitly so a future GPU CI run
can close the gap.
"""

from __future__ import annotations

import contextlib
import unittest
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from sglang.srt.layers.moe.topk import TopK


def _fake_group() -> MagicMock:
    """Return a MagicMock that mimics a `GroupCoordinator`'s shape.

    World size 1, rank() returns 0, and nested `device_group` /
    `cpu_group` attributes are themselves world-size-1 fakes so callers
    that walk `get_*_group().device_group.rank()` work cleanly. All
    runtime collective calls return MagicMock no-ops; construction
    tests never hit forward, only the constructor reads."""
    g = MagicMock()
    g.world_size = 1
    g.rank.return_value = 0
    g.rank_in_group = 0
    g.local_rank = 0
    inner = MagicMock()
    inner.world_size = 1
    inner.size.return_value = 1
    inner.rank.return_value = 0
    g.cpu_group = inner
    g.device_group = inner
    return g


_SERVER_ARGS_ATTRS = {
    # Attributes MoE constructors read directly off `_global_server_args`.
    # New attributes can be added here without changing each test.
    "disable_shared_experts_fusion": True,
    "ep_num_redundant_experts": 0,
    "enable_deepep_waterfill": False,
    "enable_dp_lm_head": False,
    "enable_eplb": False,
    "enable_elastic_expert_backup": False,
    "disable_piecewise_cuda_graph": True,
    "enable_two_batch_overlap": False,
    "moe_a2a_backend": "none",
    "kt_weight_path": None,
    "kt_num_gpu_experts": 0,
    "kt_cpuinfer": 1,
    "kt_threadpool_count": 1,
    "chunked_prefill_size": 4096,
    "kt_method": "none",
    "kt_max_deferred_experts_per_token": 0,
    "enable_nsa_prefill_context_parallel": False,
    "moe_dense_tp_size": 1,
    "speculative_algorithm": None,
    "kv_cache_dtype": "auto",
    "attn_cp_size": 1,
    "moe_dp_size": 1,
    "device": "cpu",
    "disable_chunked_prefix_cache": True,
    "chunked_prefix_cache_threshold": 0,
    "ep_size": 1,
    "tp_size": 1,
    "flashinfer_mla_disable_ragged": True,
    "disable_flashinfer_cutlass_moe_fp4_allgather": True,
    "enable_flashinfer_cutlass_moe": False,
    "enable_flashinfer_trtllm_moe": False,
    "enable_triton_kernel_moe": False,
    "enable_triton_kernel_moe_with_router_fusion": False,
}


class _FakeExperts(nn.Module):
    """Stands in for the MoE expert backend constructed via
    `get_moe_impl_class(quant_config)(...)`. The block under test
    constructs this AFTER constructing the real `TopK`, so the fake
    only needs to swallow the kwargs and exist as an attribute."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.kwargs = kwargs
        # Some block-level code reads `.should_fuse_routed_scaling_factor_in_topk`
        # off the experts attribute before it constructs TopK.
        self.should_fuse_routed_scaling_factor_in_topk = False


class _FakeRotaryEmbedding(nn.Module):
    """Stands in for `get_rope_wrapper(...)` / `get_rope(...)` returns.

    The construction-test host runs with `_SERVER_ARGS_ATTRS["device"] =
    "cpu"`, which forces `get_rope_wrapper` into the `get_rope_cpu` branch
    at `rotary_embedding/factory.py:441`. That branch asserts
    `rope_scaling is not None` and only supports `deepseek_yarn`. Several
    decoder layers under test pass `rope_scaling=None`, which surfaces as
    a bare `AssertionError` from the constructor.

    Patching the rope factory at the fixture boundary keeps the constructor
    under test (and its `TopK` instantiation) intact while hiding an
    incidental CPU-host quirk that has nothing to do with the opt-out
    plumbing."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


@contextlib.contextmanager
def construction_fixture() -> Iterator[None]:
    """Patch the infrastructure boundaries needed to construct an MoE
    block on a CPU host. The block under test runs unmodified.
    """
    import sys

    from sglang.srt import server_args as sa_mod
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.layers import dp_attention as dp_mod
    from sglang.srt.layers.moe import utils as moe_utils
    from sglang.srt.layers.moe.ep_moe import layer as ep_layer

    saved_server_args = sa_mod._global_server_args
    sa_mod._global_server_args = SimpleNamespace(**_SERVER_ARGS_ATTRS)

    group_names = [
        "_WORLD",
        "_TP",
        "_DP",
        "_PP",
        "_MOE_EP",
        "_MOE_TP",
        "_MOE_DP",
        "_ATTN_TP",
        "_ATTN_CP",
    ]
    saved_groups = {}
    for name in group_names:
        if hasattr(ps, name):
            saved_groups[name] = getattr(ps, name)
            setattr(ps, name, _fake_group())

    # DP attention state — set the module-level globals the helpers read
    # so callers like `get_attention_dp_size()` succeed without
    # initializing real DP state. This is robust to model-local imports
    # (`from sglang.srt.layers.dp_attention import get_attention_dp_size`)
    # because the function still reads the module-level value.
    dp_globals = {
        "_ATTN_DP_SIZE": 1,
        "_ATTN_DP_RANK": 0,
        "_LOCAL_ATTN_DP_SIZE": 1,
        "_LOCAL_ATTN_DP_RANK": 0,
        "_ATTN_TP_SIZE": 1,
        "_ATTN_TP_RANK": 0,
    }
    saved_dp = {}
    for name, value in dp_globals.items():
        if hasattr(dp_mod, name):
            saved_dp[name] = getattr(dp_mod, name)
            setattr(dp_mod, name, value)

    # Patch the public getters used during construction. Some live on
    # `sglang.srt.distributed`; some are accessed via the parallel_state
    # module directly. The `get_moe_impl_class` symbol is also imported
    # at module-load time by every `sglang.srt.models.*` module that
    # constructs MoE blocks (see Codex round-4 finding #4) -- those
    # local bindings must be patched explicitly because patching only
    # the source module would leave the model-local symbol pointing at
    # the real backend, defeating the fixture.
    fake_a2a = MagicMock(is_none=lambda: True, is_deepep=lambda: False)

    def _fake_rope_factory(*args, **kwargs):
        return _FakeRotaryEmbedding()

    patches = [
        patch(
            "sglang.srt.distributed.get_tensor_model_parallel_world_size",
            return_value=1,
        ),
        patch(
            "sglang.srt.distributed.get_moe_expert_parallel_world_size",
            return_value=1,
            create=True,
        ),
        patch(
            "sglang.srt.distributed.get_moe_tensor_parallel_world_size",
            return_value=1,
            create=True,
        ),
        # DP attention helpers — used by Bailing and several other
        # families. Patching the source module makes them callable
        # without initializing real DP state.
        patch(
            "sglang.srt.layers.dp_attention.get_attention_dp_size",
            return_value=1,
        ),
        patch(
            "sglang.srt.layers.dp_attention.get_attention_dp_rank",
            return_value=0,
        ),
        patch(
            "sglang.srt.layers.dp_attention.get_attention_tp_size",
            return_value=1,
        ),
        patch(
            "sglang.srt.layers.dp_attention.get_attention_tp_rank",
            return_value=0,
        ),
        patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
            return_value=False,
        ),
        patch.object(ep_layer, "get_moe_impl_class", return_value=_FakeExperts),
        patch.object(moe_utils, "get_moe_a2a_backend", return_value=fake_a2a),
        patch(
            "sglang.srt.layers.rotary_embedding.factory.get_rope_wrapper",
            side_effect=_fake_rope_factory,
        ),
        patch(
            "sglang.srt.layers.rotary_embedding.factory.get_rope",
            side_effect=_fake_rope_factory,
        ),
    ]

    # Patch every already-loaded sglang.srt.models.* module's local
    # binding of `get_moe_impl_class` / `get_rope_wrapper` / `get_rope`
    # to point at our fakes. This makes the fixture robust to import
    # order: even when the model module is imported before
    # `construction_fixture()` is entered, its module-local symbol is
    # rebound for the lifetime of the fixture.
    model_local_patches = []
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("sglang.srt.models."):
            continue
        if hasattr(mod, "get_moe_impl_class"):
            model_local_patches.append(
                patch.object(mod, "get_moe_impl_class", return_value=_FakeExperts)
            )
        if hasattr(mod, "get_rope_wrapper"):
            model_local_patches.append(
                patch.object(mod, "get_rope_wrapper", side_effect=_fake_rope_factory)
            )
        if hasattr(mod, "get_rope"):
            model_local_patches.append(
                patch.object(mod, "get_rope", side_effect=_fake_rope_factory)
            )

    for p in patches + model_local_patches:
        p.start()
    try:
        yield
    finally:
        for p in patches + model_local_patches:
            p.stop()
        sa_mod._global_server_args = saved_server_args
        for name, value in saved_groups.items():
            setattr(ps, name, value)
        for name, value in saved_dp.items():
            setattr(dp_mod, name, value)


def _collect_topks(module: nn.Module) -> list[TopK]:
    return [m for m in module.modules() if isinstance(m, TopK)]


def _minimal_qwen2_config() -> SimpleNamespace:
    return SimpleNamespace(
        num_experts=8,
        num_experts_per_tok=2,
        hidden_size=64,
        moe_intermediate_size=64,
        norm_topk_prob=True,
        hidden_act="silu",
        shared_expert_intermediate_size=0,
        # n_shared_experts intentionally absent (Qwen2Moe block reads it
        # via `hasattr(config, "n_shared_experts")`).
    )


class Qwen2MoEConstructionTest(unittest.TestCase):
    """AC-4 real construction for the Qwen2 MoE shared block. The shared
    block is reused by Qwen3Next MTP and Qwen3.5 MTP through `is_nextn`,
    so covering it here covers three inventory entries' core invariant
    (the block's TopK opts out when `is_nextn=True`)."""

    def test_draft_path_constructs_topk_with_capture_false(self):
        from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

        cfg = _minimal_qwen2_config()
        with construction_fixture():
            blk = Qwen2MoeSparseMoeBlock(0, cfg, is_nextn=True)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1, "Qwen2MoeSparseMoeBlock builds 1 TopK")
        self.assertFalse(
            topks[0].topk_config.allow_routed_experts_capture,
            "draft block (is_nextn=True) must produce TopK with "
            "allow_routed_experts_capture=False",
        )

    def test_target_path_constructs_topk_with_capture_true(self):
        from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

        cfg = _minimal_qwen2_config()
        with construction_fixture():
            blk = Qwen2MoeSparseMoeBlock(0, cfg, is_nextn=False)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertTrue(
            topks[0].topk_config.allow_routed_experts_capture,
            "target block (is_nextn=False) must produce TopK with "
            "allow_routed_experts_capture=True (default)",
        )


class ExaoneMoEConstructionTest(unittest.TestCase):
    """AC-4 real construction for the Exaone MoE block (new-plumbing
    family). Exercises the `allow_routed_experts_capture: bool = True` kwarg
    threaded into `ExaoneMoESparseMoEBlock`."""

    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            num_experts=8,
            num_experts_per_tok=2,
            hidden_size=64,
            moe_intermediate_size=64,
            norm_topk_prob=True,
            hidden_act="silu",
            routed_scaling_factor=1.0,
            n_group=1,
            topk_group=1,
            num_shared_experts=None,
            shared_expert_intermediate_size=0,
        )

    def test_draft_path_false(self):
        from sglang.srt.models.exaone_moe import ExaoneMoESparseMoEBlock

        cfg = self._config()
        with construction_fixture():
            blk = ExaoneMoESparseMoEBlock(0, cfg, allow_routed_experts_capture=False)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.allow_routed_experts_capture)

    def test_target_default_true(self):
        from sglang.srt.models.exaone_moe import ExaoneMoESparseMoEBlock

        cfg = self._config()
        with construction_fixture():
            blk = ExaoneMoESparseMoEBlock(0, cfg)  # default allow_routed_experts_capture=True
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertTrue(topks[0].topk_config.allow_routed_experts_capture)


class HunyuanV3MoEConstructionTest(unittest.TestCase):
    """AC-4 real construction for the Hunyuan v3 MoE block."""

    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            num_experts=8,
            num_experts_per_tok=2,
            hidden_size=64,
            moe_intermediate_size=64,
            route_norm=True,
            hidden_act="silu",
            router_scaling_factor=1.0,
            num_shared_experts=0,
        )

    def test_draft_path_false(self):
        from sglang.srt.models.hunyuan_v3 import HYV3MoEFused

        cfg = self._config()
        with construction_fixture():
            blk = HYV3MoEFused(cfg, layer_id=0, allow_routed_experts_capture=False)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.allow_routed_experts_capture)

    def test_target_default_true(self):
        from sglang.srt.models.hunyuan_v3 import HYV3MoEFused

        cfg = self._config()
        with construction_fixture():
            blk = HYV3MoEFused(cfg, layer_id=0)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertTrue(topks[0].topk_config.allow_routed_experts_capture)


class Step3p5MoEConstructionTest(unittest.TestCase):
    """AC-4 real construction for the Step3p5 MoE block.

    Codex round-4 finding #3: `Step3p5MoEMLP` only constructs
    `self.router_bias` when `config.use_moe_router_bias=True`, but
    always passes `correction_bias=self.router_bias` to TopK. The
    fixture must therefore use `use_moe_router_bias=True` so the block
    constructs cleanly without monkeypatching the block itself."""

    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            moe_num_experts=8,
            moe_top_k=2,
            hidden_size=64,
            moe_intermediate_size=64,
            need_fp32_gate=False,
            moe_router_scaling_factor=1.0,
            use_moe_router_bias=True,
            swiglu_limits=[0.0] * 64,
        )

    def test_draft_path_false(self):
        from sglang.srt.models.step3p5 import Step3p5MoEMLP

        cfg = self._config()
        with construction_fixture():
            blk = Step3p5MoEMLP(cfg, layer_id=0, allow_routed_experts_capture=False)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.allow_routed_experts_capture)

    def test_target_default_true(self):
        from sglang.srt.models.step3p5 import Step3p5MoEMLP

        cfg = self._config()
        with construction_fixture():
            blk = Step3p5MoEMLP(cfg, layer_id=0)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertTrue(topks[0].topk_config.allow_routed_experts_capture)


class NemotronHMoEConstructionTest(unittest.TestCase):
    """AC-4 real construction for the NemotronH MoE block."""

    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            n_routed_experts=8,
            num_experts_per_tok=2,
            hidden_size=64,
            moe_intermediate_size=64,
            n_shared_experts=0,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            hidden_act="silu",
            mlp_hidden_act="silu",
            moe_latent_size=None,
        )

    def test_draft_path_false(self):
        from sglang.srt.models.nemotron_h import NemotronHMoE

        cfg = self._config()
        with construction_fixture():
            blk = NemotronHMoE(cfg, layer_idx=0, allow_routed_experts_capture=False)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.allow_routed_experts_capture)

    def test_target_default_true(self):
        from sglang.srt.models.nemotron_h import NemotronHMoE

        cfg = self._config()
        with construction_fixture():
            blk = NemotronHMoE(cfg, layer_idx=0)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertTrue(topks[0].topk_config.allow_routed_experts_capture)


class NewPlumbingChainTest(unittest.TestCase):
    """Chain-level construction for new-plumbing families: verify the
    `allow_routed_experts_capture` kwarg actually propagates from the wrapper
    / decoder layer down to the inner MoE block's TopK. A dropped kwarg
    between the layer and the block would fail this test.
    """

    def test_nemotron_h_mtp_decoder_subclass_propagates_false(self):
        """`NemotronHMTPMoEDecoderLayer.__init__` calls
        `super().__init__(..., allow_routed_experts_capture=False)`; the
        parent `NemotronHMoEDecoderLayer.__init__` then constructs
        `NemotronHMoE(..., allow_routed_experts_capture=allow_routed_experts_capture)`,
        which sets the flag on TopK. If the subclass forgot the
        explicit False, the block defaults to True and this test fails.
        """
        from sglang.srt.models.nemotron_h_mtp import NemotronHMTPMoEDecoderLayer

        cfg = SimpleNamespace(
            n_routed_experts=8,
            num_experts_per_tok=2,
            hidden_size=64,
            moe_intermediate_size=64,
            n_shared_experts=0,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
            hidden_act="silu",
            mlp_hidden_act="silu",
            moe_latent_size=None,
            layer_norm_epsilon=1e-6,
        )
        with construction_fixture():
                layer = NemotronHMTPMoEDecoderLayer(cfg, layer_idx=0)
        topks = _collect_topks(layer)
        self.assertEqual(len(topks), 1)
        self.assertFalse(
            topks[0].topk_config.allow_routed_experts_capture,
            "NemotronHMTPMoEDecoderLayer must propagate "
            "allow_routed_experts_capture=False through to NemotronHMoE / TopK",
        )


class DeepseekV2MoEConstructionTest(unittest.TestCase):
    """AC-4 real construction for the deepseek-family shared MoE block
    (covers `DeepseekV3ForCausalLMNextN` and `DeepseekV4ForCausalLMNextN`
    via the `is_nextn` signal)."""

    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            num_experts_per_tok=2,
            n_routed_experts=8,
            moe_intermediate_size=64,
            hidden_size=64,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            hidden_act="silu",
            n_shared_experts=None,
            scoring_func="sigmoid",
            routed_scaling_factor=1.0,
            num_hidden_layers=1,
            first_k_dense_replace=0,
            topk_method="noaux_tc",
            enable_nsa_prefill_context_parallel=False,
            vocab_size=32,
            n_hash_layers=0,
            quantization_config=None,
            ep_size=1,
            kv_lora_rank=16,
            q_lora_rank=None,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=10000.0,
            rope_scaling=None,
            rms_norm_eps=1e-6,
        )

    def test_is_nextn_true_yields_false_flag(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        cfg = self._config()
        with construction_fixture():
            blk = DeepseekV2MoE(cfg, layer_id=0, is_nextn=True)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.allow_routed_experts_capture)


class GLM4MoEConstructionTest(unittest.TestCase):
    """AC-4 real construction for `Glm4MoeForCausalLMNextN` via
    Glm4MoeSparseMoeBlock."""

    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            num_experts_per_tok=2,
            n_routed_experts=8,
            moe_intermediate_size=64,
            hidden_size=64,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            hidden_act="silu",
            n_shared_experts=0,
            routed_scaling_factor=1.0,
        )

    def test_is_nextn_true_yields_false_flag(self):
        from sglang.srt.models.glm4_moe import Glm4MoeSparseMoeBlock

        cfg = self._config()
        with construction_fixture():
            blk = Glm4MoeSparseMoeBlock(
                cfg, layer_id=0, is_nextn=True
            )
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.allow_routed_experts_capture)


class DenseInnerBlockConstructionTest(unittest.TestCase):
    """AC-5 real construction for dense inner blocks. For wrappers that
    use a non-MoE feedforward (e.g. Qwen2DecoderLayer in MiMoMTP), we
    construct the inner block and assert zero TopK modules. This proves
    the dense classification at the smallest reachable structural unit.
    """

    def _qwen2_config(self) -> SimpleNamespace:
        """Minimal Qwen2 config for `Qwen2DecoderLayer` (dense). Used
        by `mimo_mtp` and by `Qwen2ForCausalLMEagle`."""
        return SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=4,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            max_position_embeddings=128,
            rope_theta=10000.0,
            rope_scaling=None,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            vocab_size=32,
            head_dim=16,
            attention_bias=False,
            attention_dropout=0.0,
            torch_dtype=torch.float32,
            quantization_config=None,
            tie_word_embeddings=False,
        )

    def test_qwen2_decoder_layer_has_no_topk(self):
        """`mimo_mtp.py` and `qwen2_eagle.py` use `Qwen2DecoderLayer`
        for their dense draft blocks. Constructing the decoder layer
        must yield zero TopK instances."""
        from sglang.srt.models.qwen2 import Qwen2DecoderLayer

        cfg = self._qwen2_config()
        with construction_fixture():
                layer = Qwen2DecoderLayer(cfg, layer_id=0)
        self.assertEqual(
            len(_collect_topks(layer)),
            0,
            "Qwen2DecoderLayer is dense (no MoE), so MiMoMTP / "
            "Qwen2ForCausalLMEagle wrappers must contain zero TopK modules",
        )

    def test_llama_decoder_layer_has_no_topk(self):
        """`llama_eagle.py` / `mistral_eagle.py` use `LlamaDecoderLayer`
        for their always-draft EAGLE blocks. Same dense contract."""
        from sglang.srt.models.llama import LlamaDecoderLayer

        cfg = SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=4,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            max_position_embeddings=128,
            rope_theta=10000.0,
            rope_scaling=None,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            vocab_size=32,
            head_dim=16,
            attention_bias=False,
            attention_dropout=0.0,
            torch_dtype=torch.float32,
            quantization_config=None,
            tie_word_embeddings=False,
            mlp_bias=False,
        )
        with construction_fixture():
                layer = LlamaDecoderLayer(cfg, layer_id=0)
        self.assertEqual(
            len(_collect_topks(layer)),
            0,
            "LlamaDecoderLayer is dense (no MoE), so Llama / Mistral / "
            "Llama3 EAGLE wrappers must contain zero TopK modules",
        )

    def test_ernie4_dense_path_constructs_no_topk(self):
        """`ernie4_eagle.py` constructs `Ernie4DecoderLayer` with
        `is_mtp=True`, which branches to the dense MLP path at
        `ernie4.py:183` and skips `Ernie4Moe`. The constructed layer
        must contain zero TopK instances."""
        from sglang.srt.models.ernie4 import Ernie4DecoderLayer

        cfg = SimpleNamespace(
            num_attention_heads=4,
            num_key_value_heads=4,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            moe_layer_start_index=0,
            max_position_embeddings=128,
            rope_theta=10000.0,
            rope_scaling=None,
            rms_norm_eps=1e-6,
            hidden_act="silu",
            vocab_size=32,
            head_dim=16,
            attention_bias=False,
            attention_dropout=0.0,
            torch_dtype=torch.float32,
            quantization_config=None,
            tie_word_embeddings=False,
            use_qkv_bias=False,
            use_bias=False,
            qk_norm=False,
            num_shared_experts=0,
            norm_topk_prob=True,
        )
        with construction_fixture():
                layer = Ernie4DecoderLayer(cfg, layer_id=0, is_mtp=True)
        self.assertEqual(
            len(_collect_topks(layer)),
            0,
            "Ernie4DecoderLayer with is_mtp=True takes the dense branch "
            "at ernie4.py:183; zero TopK modules expected",
        )


def _attn_common() -> dict:
    """Common attention config attributes shared across most decoder
    layers. Returned as a dict so each family's `_config()` can spread it
    into the namespace alongside its own MoE-specific attributes.

    Generous: any attribute a multi-family decoder constructor commonly
    reads is included here. Per-family configs override specific values
    where needed (e.g. an MoE-specific `hidden_act`)."""
    return dict(
        num_attention_heads=4,
        num_key_value_heads=4,
        hidden_size=64,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rope_scaling=None,
        rms_norm_eps=1e-6,
        layernorm_epsilon=1e-6,
        head_dim=16,
        attention_bias=False,
        attention_dropout=0.0,
        vocab_size=32,
        torch_dtype=torch.float32,
        quantization_config=None,
        tie_word_embeddings=False,
        layer_norm_epsilon=1e-6,
        kv_cache_dtype=None,
        model_type="llama",
        use_qkv_bias=False,
        qk_norm=False,
        attention_use_bias=False,
        layer_types=["full_attention"] * 8,
        # KV-LoRA / MLA family attrs
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        kv_lora_rank=16,
        q_lora_rank=None,
        # Other attrs occasionally read by various families. SWA heads
        # default to non-None integers so MiMoV2's SWA-aware MTP layer
        # (`mimo_v2_nextn.py:84` reads `config.swa_num_attention_heads`)
        # can compute `num_heads % tp_size` without `%`-ing `NoneType`.
        swa_num_attention_heads=4,
        swa_num_key_value_heads=4,
        swa_head_dim=16,
        sliding_window=None,
        sliding_window_size=-1,
        sliding_window_pattern=None,
        mlp_bias=False,
        use_bias=False,
        rope_is_neox_style=True,
        original_max_position_embeddings=None,
        attn_logit_softcapping=None,
        final_logit_softcapping=None,
        hidden_activation="gelu_pytorch_tanh",
        rope_parameters=dict(rope_theta=10000.0, rope_type="default"),
    )


# =============================================================================
# Round 6: Bailing + Gemma4 + chain-level MoE families + target-default checks
# =============================================================================


class BailingMoEConstructionTest(unittest.TestCase):
    """AC-4 real construction for the Bailing MoE family via
    `BailingMoEBlock`. The block decides layer-sparsity from `is_nextn`
    so the first layer is sparse and constructs a `TopK`."""

    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            num_experts=8,
            num_experts_per_tok=2,
            num_shared_experts=0,
            intermediate_size=128,
            moe_intermediate_size=64,
            first_k_dense_replace=0,
            norm_topk_prob=True,
            hidden_act="silu",
            num_hidden_layers=1,
            n_group=0,
            topk_group=0,
            routed_scaling_factor=1.0,
            score_function=None,
            router_dtype=None,
            ep_num_redundant_experts=0,
            **_attn_common(),
        )

    def test_draft_path_false(self):
        from sglang.srt.models.bailing_moe import BailingMoEBlock

        cfg = self._config()
        with construction_fixture():
            blk = BailingMoEBlock(cfg, layer_id=0, is_nextn=True)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.allow_routed_experts_capture)

    def test_target_default_true(self):
        from sglang.srt.models.bailing_moe import BailingMoEBlock

        cfg = self._config()
        with construction_fixture():
            blk = BailingMoEBlock(cfg, layer_id=0, is_nextn=False)
        topks = _collect_topks(blk)
        # `first_k_dense_replace=0` makes layer 0 sparse, so the target
        # block MUST build exactly one TopK. Asserting the count first
        # closes the round-7 queued vacuous-pass guard: a future config
        # change that silently drops the MoE block would otherwise pass
        # this test by reaching zero TopKs.
        self.assertEqual(
            len(topks),
            1,
            "Bailing target block must construct exactly one TopK on the "
            "sparse path (is_nextn=False, first_k_dense_replace=0).",
        )
        self.assertTrue(topks[0].topk_config.allow_routed_experts_capture)


class Gemma4DecoderLayerConstructionTest(unittest.TestCase):
    """AC-4 chain construction for Gemma4: `Gemma4DecoderLayer` is the
    layer above `Gemma4MoE` and threads `allow_routed_experts_capture` through.
    """

    def _config(self) -> SimpleNamespace:
        common = _attn_common()
        common.pop("layer_types", None)
        return SimpleNamespace(
            num_experts=8,
            top_k_experts=2,
            num_experts_per_tok=2,
            intermediate_size=128,
            moe_intermediate_size=64,
            hidden_act="silu",
            enable_moe_block=True,
            num_hidden_layers=1,
            layer_types=["full_attention"],
            query_pre_attn_scalar=1.0,
            cache_implementation="hybrid",
            **common,
        )

    def test_draft_chain_propagates_false(self):
        from sglang.srt.models.gemma4_causal import Gemma4DecoderLayer

        cfg = self._config()
        with construction_fixture():
            layer = Gemma4DecoderLayer(
                layer_id=0, config=cfg, allow_routed_experts_capture=False
            )
        topks = _collect_topks(layer)
        # `enable_moe_block=True` forces `Gemma4MoE` construction; the
        # chain MUST yield at least one `TopK`, otherwise the test passes
        # vacuously and a future regression that drops the MoE block
        # would slip through.
        self.assertGreaterEqual(
            len(topks),
            1,
            "Gemma4 draft chain test must construct at least one TopK; "
            "got zero, indicating the MoE branch was bypassed.",
        )
        for t in topks:
            self.assertFalse(t.topk_config.allow_routed_experts_capture)

    def test_target_chain_default_true(self):
        from sglang.srt.models.gemma4_causal import Gemma4DecoderLayer

        cfg = self._config()
        with construction_fixture():
            layer = Gemma4DecoderLayer(layer_id=0, config=cfg)
        topks = _collect_topks(layer)
        self.assertGreaterEqual(
            len(topks),
            1,
            "Gemma4 target chain test must construct at least one TopK.",
        )
        for t in topks:
            self.assertTrue(t.topk_config.allow_routed_experts_capture)


class ExaoneChainConstructionTest(unittest.TestCase):
    """AC-4 chain construction for Exaone via `ExaoneMoEDecoderLayer`.
    The kwarg must propagate from the decoder layer through to the
    SparseMoEBlock; a dropped kwarg between them would fail this test."""

    def _config(self) -> SimpleNamespace:
        common = _attn_common()
        return SimpleNamespace(
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=64,
            norm_topk_prob=True,
            hidden_act="silu",
            routed_scaling_factor=1.0,
            n_group=1,
            topk_group=1,
            num_shared_experts=None,
            shared_expert_intermediate_size=0,
            is_moe_layer=[True],
            intermediate_size=128,
            num_hidden_layers=1,
            **common,
        )

    def test_decoder_chain_draft_propagates_false(self):
        from sglang.srt.models.exaone_moe import ExaoneMoEDecoderLayer

        cfg = self._config()
        with construction_fixture():
            layer = ExaoneMoEDecoderLayer(
                cfg, layer_id=0, allow_routed_experts_capture=False
            )
        topks = _collect_topks(layer)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.allow_routed_experts_capture)

    def test_decoder_chain_target_default_true(self):
        from sglang.srt.models.exaone_moe import ExaoneMoEDecoderLayer

        cfg = self._config()
        with construction_fixture():
            layer = ExaoneMoEDecoderLayer(cfg, layer_id=0)
        topks = _collect_topks(layer)
        self.assertEqual(len(topks), 1)
        self.assertTrue(topks[0].topk_config.allow_routed_experts_capture)


class HunyuanV3ChainConstructionTest(unittest.TestCase):
    """AC-4 chain construction for Hunyuan v3 via `HYV3DecoderLayer`."""

    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            num_experts=8,
            num_experts_per_tok=2,
            hidden_size=64,
            moe_intermediate_size=64,
            intermediate_size=128,
            route_norm=True,
            hidden_act="silu",
            router_scaling_factor=1.0,
            num_shared_experts=0,
            first_k_dense_replace=0,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            max_position_embeddings=128,
            rope_theta=10000.0,
            rope_scaling=None,
            rms_norm_eps=1e-6,
        )

    def test_decoder_chain_draft_propagates_false(self):
        from sglang.srt.models.hunyuan_v3 import HYV3DecoderLayer

        cfg = self._config()
        with construction_fixture():
            layer = HYV3DecoderLayer(
                cfg, layer_id=0, allow_routed_experts_capture=False
            )
        topks = _collect_topks(layer)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.allow_routed_experts_capture)

    def test_decoder_chain_target_default_true(self):
        from sglang.srt.models.hunyuan_v3 import HYV3DecoderLayer

        cfg = self._config()
        with construction_fixture():
            layer = HYV3DecoderLayer(cfg, layer_id=0)
        topks = _collect_topks(layer)
        self.assertEqual(len(topks), 1)
        self.assertTrue(topks[0].topk_config.allow_routed_experts_capture)


class Step3p5ChainConstructionTest(unittest.TestCase):
    """AC-4 chain construction for Step3p5 via `Step3p5DecoderLayer`.
    The decoder layer constructs `Step3p5MoEMLP` when the layer's
    `moe_layers_enum` includes its id; the test ensures the
    `allow_routed_experts_capture` kwarg propagates through that branch."""

    def _config(self) -> SimpleNamespace:
        common = _attn_common()
        # Step3p5 indexes `config.rope_theta[layer_id]` and
        # `config.partial_rotary_factors[layer_id]` at step3p5.py:536-541,
        # so both must be per-layer sequences aligned with `layer_types`.
        num_layers = 1
        return SimpleNamespace(
            moe_num_experts=8,
            moe_top_k=2,
            moe_intermediate_size=64,
            need_fp32_gate=False,
            moe_router_scaling_factor=1.0,
            use_moe_router_bias=True,
            swiglu_limits=[0.0] * 64,
            swiglu_limits_shared=[None] * 64,
            moe_layers_enum="0",
            layer_types=["full_attention"] * num_layers,
            yarn_only_types=set(),
            num_attention_groups=4,
            num_hidden_layers=num_layers,
            partial_rotary_factors=[1.0] * num_layers,
            share_expert_dim=128,
            use_head_wise_attn_gate=False,
            attention_other_setting={
                "num_attention_heads": 4,
                "num_attention_groups": 4,
            },
            **{
                k: v
                for k, v in common.items()
                if k not in {"layer_types", "rope_theta"}
            },
            rope_theta=[10000.0] * num_layers,
        )

    def test_decoder_chain_draft_propagates_false(self):
        from sglang.srt.models.step3p5 import Step3p5DecoderLayer

        cfg = self._config()
        with construction_fixture():
            layer = Step3p5DecoderLayer(
                cfg, layer_id=0, allow_routed_experts_capture=False
            )
        topks = _collect_topks(layer)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.allow_routed_experts_capture)


class MistralLarge3ChainConstructionTest(unittest.TestCase):
    """AC-4 chain construction for the always-draft MistralLarge3
    EAGLE wrapper. It reuses `DeepseekV2DecoderLayer` with `is_nextn=False`,
    so the wrapper itself must mark the constructed TopK modules as not
    allowed to capture."""

    def _config(self) -> SimpleNamespace:
        common = _attn_common()
        return SimpleNamespace(
            num_experts_per_tok=2,
            n_routed_experts=8,
            moe_intermediate_size=64,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            hidden_act="silu",
            n_shared_experts=None,
            scoring_func="sigmoid",
            routed_scaling_factor=1.0,
            num_hidden_layers=1,
            first_k_dense_replace=0,
            topk_method="noaux_tc",
            enable_nsa_prefill_context_parallel=False,
            n_hash_layers=0,
            ep_size=1,
            intermediate_size=128,
            moe_layer_freq=1,
            **common,
        )

    def test_eagle_wrapper_marks_reused_deepseek_topks_false(self):
        from sglang.srt.models.mistral_large_3_eagle import MistralLarge3EagleModel

        cfg = self._config()
        with construction_fixture():
            model = MistralLarge3EagleModel(cfg)
        topks = _collect_topks(model)
        # `first_k_dense_replace=0` + `moe_layer_freq=1` forces layer 0
        # sparse, so the constructor MUST build at least one `TopK` —
        # otherwise the test is vacuous and a future regression that
        # silently drops the MoE block would slip through.
        self.assertGreaterEqual(
            len(topks),
            1,
            "MistralLarge3 chain test must construct at least one TopK; "
            "got zero, suggesting the sparse branch was skipped.",
        )
        for t in topks:
            self.assertFalse(t.topk_config.allow_routed_experts_capture)


class MoETargetDefaultConstructionTest(unittest.TestCase):
    """AC-4 target-default checks for DeepSeek and GLM4.

    DeepSeek (covers V3 NextN, V4 NextN, MistralLarge3 EAGLE base path)
    must construct with `is_nextn=False` -> default `True` flag on TopK.

    GLM4 MoE (covers `Glm4MoeForCausalLMNextN` target path) must construct
    with `is_nextn=False` -> default `True` flag on TopK.
    """

    def test_deepseek_v2_moe_target_default_true(self):
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        cfg = SimpleNamespace(
            num_experts_per_tok=2,
            n_routed_experts=8,
            moe_intermediate_size=64,
            hidden_size=64,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            hidden_act="silu",
            n_shared_experts=None,
            scoring_func="sigmoid",
            routed_scaling_factor=1.0,
            num_hidden_layers=1,
            first_k_dense_replace=0,
            topk_method="noaux_tc",
            enable_nsa_prefill_context_parallel=False,
            vocab_size=32,
            n_hash_layers=0,
            quantization_config=None,
            ep_size=1,
            kv_lora_rank=16,
            q_lora_rank=None,
            qk_nope_head_dim=8,
            qk_rope_head_dim=8,
            v_head_dim=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=10000.0,
            rope_scaling=None,
            rms_norm_eps=1e-6,
        )
        with construction_fixture():
            blk = DeepseekV2MoE(cfg, layer_id=0, is_nextn=False)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertTrue(topks[0].topk_config.allow_routed_experts_capture)

    def test_glm4_moe_sparse_target_default_true(self):
        from sglang.srt.models.glm4_moe import Glm4MoeSparseMoeBlock

        cfg = SimpleNamespace(
            num_experts_per_tok=2,
            n_routed_experts=8,
            moe_intermediate_size=64,
            hidden_size=64,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            hidden_act="silu",
            n_shared_experts=0,
            routed_scaling_factor=1.0,
        )
        with construction_fixture():
            blk = Glm4MoeSparseMoeBlock(cfg, layer_id=0, is_nextn=False)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertTrue(topks[0].topk_config.allow_routed_experts_capture)


# =============================================================================
# Round 6: Dense allowlist real-construction tests using the actual inner
# classes named in `draft_inventory.py` (not base-model substitutes).
# =============================================================================


class DenseEagleInnerClassConstructionTest(unittest.TestCase):
    """AC-5 real construction for EAGLE-family dense entries using the
    wrapper-specific subclasses defined in each EAGLE file (not the
    base Llama/Qwen2 decoder layers in `sglang.srt.models.llama` /
    `qwen2`)."""

    def _llama_config(self) -> SimpleNamespace:
        common = _attn_common()
        return SimpleNamespace(
            intermediate_size=128,
            num_hidden_layers=1,
            hidden_act="silu",
            **common,
        )

    def test_llama_eagle_decoder_layer_has_no_topk(self):
        from sglang.srt.models.llama_eagle import LlamaDecoderLayer as EagleLayer

        cfg = self._llama_config()
        with construction_fixture():
            layer = EagleLayer(cfg, layer_id=0)
        self.assertEqual(len(_collect_topks(layer)), 0)

    def test_llama_eagle3_decoder_layer_has_no_topk(self):
        from sglang.srt.models.llama_eagle3 import LlamaDecoderLayer as Eagle3Layer

        cfg = self._llama_config()
        with construction_fixture():
            layer = Eagle3Layer(cfg, layer_id=0)
        self.assertEqual(len(_collect_topks(layer)), 0)

    def test_qwen2_eagle_decoder_layer_has_no_topk(self):
        from sglang.srt.models.qwen2_eagle import Qwen2DecoderLayer as EagleLayer

        cfg = SimpleNamespace(
            intermediate_size=128,
            num_hidden_layers=1,
            hidden_act="silu",
            **_attn_common(),
        )
        with construction_fixture():
            layer = EagleLayer(cfg, layer_id=0)
        self.assertEqual(len(_collect_topks(layer)), 0)

    def test_eagle3_mla_decoder_layer_has_no_topk(self):
        from sglang.srt.models.kimi_k25_eagle3 import Eagle3MLADecoderLayer

        common = _attn_common()
        # Eagle3MLA validates `q_lora_rank` is set (it's a real MLA-draft
        # requirement, not an incidental CPU-host quirk). Provide a value
        # so the constructor proceeds; the test then asserts no TopK.
        common["q_lora_rank"] = 16
        cfg = SimpleNamespace(
            intermediate_size=128,
            num_hidden_layers=1,
            hidden_act="silu",
            **common,
        )
        with construction_fixture():
            layer = Eagle3MLADecoderLayer(cfg, layer_id=0)
        self.assertEqual(len(_collect_topks(layer)), 0)


class DenseNextNInnerClassConstructionTest(unittest.TestCase):
    """AC-5 real construction for *_nextn dense entries using the
    wrapper-specific inner classes (not the base model's MoE decoder)."""

    def test_glm_ocr_glm4_decoder_layer_has_no_topk(self):
        from sglang.srt.models.glm4 import Glm4DecoderLayer

        cfg = SimpleNamespace(
            intermediate_size=128,
            num_hidden_layers=1,
            hidden_act="silu",
            **_attn_common(),
        )
        with construction_fixture():
            layer = Glm4DecoderLayer(cfg, layer_id=0)
        self.assertEqual(len(_collect_topks(layer)), 0)

    def test_longcat_flash_dense_decoder_layer_has_no_topk(self):
        from sglang.srt.models.longcat_flash_nextn import (
            LongcatFlashDenseDecoderLayer,
        )

        cfg = SimpleNamespace(
            intermediate_size=128,
            num_hidden_layers=1,
            hidden_act="silu",
            **_attn_common(),
        )
        with construction_fixture():
            layer = LongcatFlashDenseDecoderLayer(cfg, layer_id=0)
        self.assertEqual(len(_collect_topks(layer)), 0)

    def test_mimo_v2_mtp_layer_has_no_topk(self):
        from sglang.srt.models.mimo_v2_nextn import MiMoV2MTPLayer

        cfg = SimpleNamespace(
            intermediate_size=128,
            num_hidden_layers=1,
            hidden_act="silu",
            **_attn_common(),
        )
        with construction_fixture():
            layer = MiMoV2MTPLayer(cfg, layer_id=0)
        self.assertEqual(len(_collect_topks(layer)), 0)


if __name__ == "__main__":
    unittest.main()
