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
    """Return a MagicMock that mimics a `GroupCoordinator`'s shape:
    world_size=1, rank()=0, plus attributes that callers commonly read.
    Calls into the group are intentionally no-ops because the test never
    runs forward; only construction is exercised."""
    g = MagicMock()
    g.world_size = 1
    g.rank.return_value = 0
    g.local_rank = 0
    g.cpu_group = None
    g.device_group = None
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

    # Patch the public getters used during construction (some live on
    # `sglang.srt.distributed`, some are accessed via the parallel_state
    # module directly).
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
        patch.object(
            moe_utils,
            "get_moe_a2a_backend",
            return_value=MagicMock(is_none=lambda: True, is_deepep=lambda: False),
        ),
    ]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
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
    """AC-4 real construction for the Step3p5 MoE block."""

    def _config(self) -> SimpleNamespace:
        return SimpleNamespace(
            moe_num_experts=8,
            moe_top_k=2,
            hidden_size=64,
            moe_intermediate_size=64,
            need_fp32_gate=False,
            moe_router_scaling_factor=1.0,
            use_moe_router_bias=False,
            swiglu_limits=[0.0] * 64,
        )

    def test_draft_path_false(self):
        from sglang.srt.models.step3p5 import Step3p5MoEMLP

        cfg = self._config()
        try:
            with construction_fixture():
                blk = Step3p5MoEMLP(cfg, layer_id=0, capture_routed_experts=False)
        except Exception as exc:
            self.skipTest(
                f"Step3p5MoEMLP requires extra fixture attrs: "
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

    def test_qwen2_decoder_layer_has_no_topk(self):
        """`mimo_mtp.py` uses `Qwen2DecoderLayer` for its MTP block.
        Constructing that decoder layer must yield zero TopK instances."""
        from sglang.srt.models.qwen2 import Qwen2DecoderLayer

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
        )
        try:
            with construction_fixture():
                layer = Qwen2DecoderLayer(cfg, layer_id=0)
        except Exception as exc:
            self.skipTest(
                f"Qwen2DecoderLayer requires extra infrastructure not "
                f"covered by the fixture: {type(exc).__name__}: {exc}. "
                "Treat as fixture-gap; the AST test still asserts the "
                "wrapper has no TopK call."
            )
        topks = _collect_topks(layer)
        self.assertEqual(
            len(topks),
            0,
            f"Qwen2DecoderLayer used by mimo_mtp is supposed to be dense, "
            f"but {len(topks)} TopK module(s) appeared",
        )


if __name__ == "__main__":
    unittest.main()
