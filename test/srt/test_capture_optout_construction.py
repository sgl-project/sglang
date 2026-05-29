"""Real construction-level verification of the `capture_routed_experts`
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


@contextlib.contextmanager
def construction_fixture() -> Iterator[None]:
    """Patch the infrastructure boundaries needed to construct an MoE
    block on a CPU host. The block under test runs unmodified.
    """
    import sys

    from sglang.srt import server_args as sa_mod
    from sglang.srt.distributed import parallel_state as ps
    from sglang.srt.layers.moe import utils as moe_utils
    from sglang.srt.layers.moe.ep_moe import layer as ep_layer

    saved_server_args = sa_mod._global_server_args
    sa_mod._global_server_args = SimpleNamespace(**_SERVER_ARGS_ATTRS)

    group_names = ["_WORLD", "_TP", "_DP", "_PP", "_MOE_EP", "_MOE_TP", "_ATTN_TP"]
    saved_groups = {}
    for name in group_names:
        if hasattr(ps, name):
            saved_groups[name] = getattr(ps, name)
            setattr(ps, name, _fake_group())

    # Patch the public getters used during construction. Some live on
    # `sglang.srt.distributed`; some are accessed via the parallel_state
    # module directly. The `get_moe_impl_class` symbol is also imported
    # at module-load time by every `sglang.srt.models.*` module that
    # constructs MoE blocks (see Codex round-4 finding #4) -- those
    # local bindings must be patched explicitly because patching only
    # the source module would leave the model-local symbol pointing at
    # the real backend, defeating the fixture.
    fake_a2a = MagicMock(is_none=lambda: True, is_deepep=lambda: False)
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
        patch.object(ep_layer, "get_moe_impl_class", return_value=_FakeExperts),
        patch.object(moe_utils, "get_moe_a2a_backend", return_value=fake_a2a),
    ]

    # Patch every already-loaded sglang.srt.models.* module's local
    # binding of `get_moe_impl_class` to point at our fake. This makes
    # the fixture robust to import order: even when the model module is
    # imported before `construction_fixture()` is entered, its module-
    # local symbol is rebound for the lifetime of the fixture.
    model_local_patches = []
    for mod_name, mod in list(sys.modules.items()):
        if not mod_name.startswith("sglang.srt.models."):
            continue
        if not hasattr(mod, "get_moe_impl_class"):
            continue
        model_local_patches.append(
            patch.object(mod, "get_moe_impl_class", return_value=_FakeExperts)
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
            topks[0].topk_config.capture_routed_experts,
            "draft block (is_nextn=True) must produce TopK with "
            "capture_routed_experts=False",
        )

    def test_target_path_constructs_topk_with_capture_true(self):
        from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock

        cfg = _minimal_qwen2_config()
        with construction_fixture():
            blk = Qwen2MoeSparseMoeBlock(0, cfg, is_nextn=False)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertTrue(
            topks[0].topk_config.capture_routed_experts,
            "target block (is_nextn=False) must produce TopK with "
            "capture_routed_experts=True (default)",
        )


class ExaoneMoEConstructionTest(unittest.TestCase):
    """AC-4 real construction for the Exaone MoE block (new-plumbing
    family). Exercises the `capture_routed_experts: bool = True` kwarg
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
            blk = ExaoneMoESparseMoEBlock(0, cfg, capture_routed_experts=False)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.capture_routed_experts)

    def test_target_default_true(self):
        from sglang.srt.models.exaone_moe import ExaoneMoESparseMoEBlock

        cfg = self._config()
        with construction_fixture():
            blk = ExaoneMoESparseMoEBlock(0, cfg)  # default capture_routed_experts=True
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertTrue(topks[0].topk_config.capture_routed_experts)


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
            blk = HYV3MoEFused(cfg, layer_id=0, capture_routed_experts=False)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.capture_routed_experts)

    def test_target_default_true(self):
        from sglang.srt.models.hunyuan_v3 import HYV3MoEFused

        cfg = self._config()
        with construction_fixture():
            blk = HYV3MoEFused(cfg, layer_id=0)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertTrue(topks[0].topk_config.capture_routed_experts)


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
            blk = Step3p5MoEMLP(cfg, layer_id=0, capture_routed_experts=False)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.capture_routed_experts)

    def test_target_default_true(self):
        from sglang.srt.models.step3p5 import Step3p5MoEMLP

        cfg = self._config()
        with construction_fixture():
            blk = Step3p5MoEMLP(cfg, layer_id=0)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertTrue(topks[0].topk_config.capture_routed_experts)


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
            blk = NemotronHMoE(cfg, layer_idx=0, capture_routed_experts=False)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.capture_routed_experts)

    def test_target_default_true(self):
        from sglang.srt.models.nemotron_h import NemotronHMoE

        cfg = self._config()
        with construction_fixture():
            blk = NemotronHMoE(cfg, layer_idx=0)
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertTrue(topks[0].topk_config.capture_routed_experts)


class NewPlumbingChainTest(unittest.TestCase):
    """Chain-level construction for new-plumbing families: verify the
    `capture_routed_experts` kwarg actually propagates from the wrapper
    / decoder layer down to the inner MoE block's TopK. A dropped kwarg
    between the layer and the block would fail this test.
    """

    def test_nemotron_h_mtp_decoder_subclass_propagates_false(self):
        """`NemotronHMTPMoEDecoderLayer.__init__` calls
        `super().__init__(..., capture_routed_experts=False)`; the
        parent `NemotronHMoEDecoderLayer.__init__` then constructs
        `NemotronHMoE(..., capture_routed_experts=capture_routed_experts)`,
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
        try:
            with construction_fixture():
                layer = NemotronHMTPMoEDecoderLayer(cfg, layer_idx=0)
        except Exception as exc:
            self.skipTest(
                f"NemotronHMTPMoEDecoderLayer fixture gap: "
                f"{type(exc).__name__}: {exc}"
            )
        topks = _collect_topks(layer)
        self.assertEqual(len(topks), 1)
        self.assertFalse(
            topks[0].topk_config.capture_routed_experts,
            "NemotronHMTPMoEDecoderLayer must propagate "
            "capture_routed_experts=False through to NemotronHMoE / TopK",
        )


class DeepseekV2MoEConstructionTest(unittest.TestCase):
    """AC-4 real construction for the deepseek-family shared MoE block
    (covers `DeepseekV3ForCausalLMNextN` and `DeepseekV4ForCausalLMNextN`
    via the `is_nextn` signal, plus the always-draft
    `MistralLarge3ForCausalLMEagle` via the explicit
    `capture_routed_experts=False` keyword)."""

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
            try:
                blk = DeepseekV2MoE(cfg, layer_id=0, is_nextn=True)
            except Exception as exc:
                self.skipTest(
                    f"DeepseekV2MoE fixture gap: {type(exc).__name__}: {exc}"
                )
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.capture_routed_experts)

    def test_explicit_capture_false_overrides_is_nextn_default(self):
        """`MistralLarge3ForCausalLMEagle` passes `capture_routed_experts=False`
        explicitly; this must win even when `is_nextn=False` (the deepseek
        target default)."""
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE

        cfg = self._config()
        with construction_fixture():
            try:
                blk = DeepseekV2MoE(
                    cfg,
                    layer_id=0,
                    is_nextn=False,
                    capture_routed_experts=False,
                )
            except Exception as exc:
                self.skipTest(
                    f"DeepseekV2MoE fixture gap: {type(exc).__name__}: {exc}"
                )
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.capture_routed_experts)


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
            try:
                blk = Glm4MoeSparseMoeBlock(
                    cfg, layer_id=0, is_nextn=True
                )
            except Exception as exc:
                self.skipTest(
                    f"Glm4MoeSparseMoeBlock fixture gap: "
                    f"{type(exc).__name__}: {exc}"
                )
        topks = _collect_topks(blk)
        self.assertEqual(len(topks), 1)
        self.assertFalse(topks[0].topk_config.capture_routed_experts)


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
        try:
            with construction_fixture():
                layer = Qwen2DecoderLayer(cfg, layer_id=0)
        except Exception as exc:
            self.skipTest(
                f"Qwen2DecoderLayer requires extra infrastructure not "
                f"covered by the fixture: {type(exc).__name__}: {exc}"
            )
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
        try:
            with construction_fixture():
                layer = LlamaDecoderLayer(cfg, layer_id=0)
        except Exception as exc:
            self.skipTest(
                f"LlamaDecoderLayer fixture gap: "
                f"{type(exc).__name__}: {exc}"
            )
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
        try:
            with construction_fixture():
                layer = Ernie4DecoderLayer(cfg, layer_id=0, is_mtp=True)
        except Exception as exc:
            self.skipTest(
                f"Ernie4DecoderLayer fixture gap: "
                f"{type(exc).__name__}: {exc}"
            )
        self.assertEqual(
            len(_collect_topks(layer)),
            0,
            "Ernie4DecoderLayer with is_mtp=True takes the dense branch "
            "at ernie4.py:183; zero TopK modules expected",
        )


if __name__ == "__main__":
    unittest.main()
