"""Muse Spark NVFP4 expert weight updates on the Marlin backend, on CPU.

Miles streams experts one at a time under per-expert names; the engine must
write them into its checkpoint-layout stacks on the owning EP rank only and
re-derive the Marlin repack of exactly those experts in place. The Marlin
weight repack is a CUDA kernel, so it is replaced by a byte-preserving stand-in;
the scale permutation and global-scale encoding run for real.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers import muse_spark_fp4_moe
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.muse_spark_bf16_moe import MuseSpark1xBf16Experts
from sglang.srt.layers.muse_spark_fp4_moe import MuseSpark1xFp4Experts
from sglang.srt.layers.quantization import marlin_utils_fp4, modelopt_quant
from sglang.srt.models import muse_spark
from sglang.test.test_utils import CustomTestCase

NUM_EXPERTS, NUM_LOCAL, HIDDEN, INTER = 4, 2, 64, 64
BF16_PREFIX = "model.language_model.layers.0.mlp.experts"
NVFP4_PREFIX = "model.language_model.layers.1.mlp.experts"
MARLIN_NAMES = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
)


def _fake_marlin_repack(weight, *, num_experts, size_n, size_k, perm):
    """Byte-preserving, expert-local stand-in with the real Marlin output shape."""
    packed = weight.reshape(num_experts, size_n, size_k // 2).view(torch.int32)
    return packed.flip(-1).reshape(num_experts, size_k // 16, size_n * 2)


@contextmanager
def _marlin_on_cpu(repack=_fake_marlin_repack):
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                muse_spark_fp4_moe,
                "_resolve_moe_runner_backend",
                return_value=MoeRunnerBackend.MARLIN,
            )
        )
        stack.enter_context(
            patch.object(
                modelopt_quant,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.MARLIN,
            )
        )
        stack.enter_context(
            patch.object(
                marlin_utils_fp4,
                "marlin_make_workspace",
                return_value=torch.zeros(0, dtype=torch.int),
            )
        )
        stack.enter_context(
            patch.object(marlin_utils_fp4, "_repack_moe_fp4_weight_for_marlin", repack)
        )
        yield


def _layer(experts: nn.Module) -> nn.Module:
    layer = nn.Module()
    layer.post_attention_residual_gate = nn.Identity()
    layer.post_feedforward_residual_gate = nn.Identity()
    layer.mlp = nn.Module()
    layer.mlp.experts = experts
    layer.self_attn = nn.Module()
    layer.self_attn._fused_o_norm_gate = Mock()
    layer._fused_attn_boundary = Mock()
    layer._fused_ffn_boundary = None
    layer.operator_dtype = torch.bfloat16
    return layer


def _make_model(ep_rank: int, *, keep_checkpoint_layout: bool = True):
    """Layer 0 keeps BF16 experts (an excluded layer), layer 1 holds NVFP4 experts."""
    with (
        envs.SGLANG_ENABLE_MUSE_SPARK_NVFP4_WEIGHT_UPDATE.override(
            keep_checkpoint_layout
        ),
        _marlin_on_cpu(),
    ):
        fp4 = MuseSpark1xFp4Experts(
            num_experts=NUM_EXPERTS,
            num_local_experts=NUM_LOCAL,
            local_expert_offset=ep_rank * NUM_LOCAL,
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            top_k=2,
            quant_config=SimpleNamespace(group_size=16, use_per_token_activation=False),
            params_dtype=torch.bfloat16,
        )
    bf16 = MuseSpark1xBf16Experts(
        num_experts=NUM_EXPERTS,
        num_local_experts=NUM_LOCAL,
        local_expert_offset=ep_rank * NUM_LOCAL,
        expert_hidden_size=HIDDEN,
        intermediate_size=INTER,
        top_k=2,
    ).to(dtype=torch.bfloat16)
    model = object.__new__(muse_spark.MuseSpark1xForCausalLM)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        num_local_experts=NUM_EXPERTS, tie_word_embeddings=False
    )
    model.int4_checkpoint = False
    model.nvfp4_checkpoint = True
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    layers = nn.ModuleList([_layer(bf16), _layer(fp4)])
    model.model.language_model.layers = layers
    object.__setattr__(model.model, "layers", layers)
    return model, bf16, fp4


def _expert_tensors(seed: int) -> dict:
    """Per-expert tensors in Miles' quantizer_nvfp4 output format."""
    gen = torch.Generator().manual_seed(seed)
    tensors = {}
    for expert_id in range(NUM_EXPERTS):
        for projection, rows, cols in (
            ("gate_proj", INTER, HIDDEN),
            ("up_proj", INTER, HIDDEN),
            ("down_proj", HIDDEN, INTER),
        ):
            tensors[expert_id, projection] = {
                "weight": torch.randint(
                    0, 256, (rows, cols // 2), dtype=torch.uint8, generator=gen
                ),
                "weight_scale": (
                    torch.rand(rows, cols // 16, generator=gen) * 4 + 0.25
                ).to(torch.float8_e4m3fn),
                # a 0-dim decode scale, distinct for gate and up
                "weight_scale_2": torch.rand((), generator=gen) * 0.01 + 0.001,
            }
    return tensors


def _stacks(tensors: dict) -> dict:
    """The same experts as the checkpoint's stacked tensors."""

    def stack(projection, suffix):
        return torch.stack([tensors[e, projection][suffix] for e in range(NUM_EXPERTS)])

    return {
        "gate_up_proj": torch.cat(
            [stack("gate_proj", "weight"), stack("up_proj", "weight")], dim=1
        ),
        "gate_up_proj_weight_scale": torch.cat(
            [stack("gate_proj", "weight_scale"), stack("up_proj", "weight_scale")],
            dim=1,
        ),
        "gate_up_proj_weight_scale_2": torch.stack(
            [stack("gate_proj", "weight_scale_2"), stack("up_proj", "weight_scale_2")],
            dim=1,
        ),
        "gate_up_proj_input_scale": torch.ones(NUM_EXPERTS, 2),
        "down_proj": stack("down_proj", "weight"),
        "down_proj_weight_scale": stack("down_proj", "weight_scale"),
        "down_proj_weight_scale_2": stack("down_proj", "weight_scale_2"),
        "down_proj_input_scale": torch.ones(NUM_EXPERTS),
    }


def _per_expert_names(tensors: dict, expert_ids):
    return [
        (f"{NVFP4_PREFIX}.{expert_id}.{projection}.{suffix}", tensor)
        for expert_id in expert_ids
        for projection in ("gate_proj", "up_proj", "down_proj")
        for suffix, tensor in tensors[expert_id, projection].items()
    ]


def _load_stacks(model, ep_rank: int, stacks: dict) -> None:
    parallel = SimpleNamespace(moe_ep_rank=ep_rank, moe_ep_size=2)
    with patch.object(muse_spark, "get_parallel", return_value=parallel):
        model.load_weights(
            [(f"{NVFP4_PREFIX}.{name}", tensor) for name, tensor in stacks.items()]
        )


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().view(torch.uint8) if tensor.element_size() == 1 else tensor


def _assert_bitwise_equal(test, actual: torch.Tensor, expected: torch.Tensor, msg=""):
    test.assertEqual(actual.shape, expected.shape, msg)
    test.assertTrue(torch.equal(_bits(actual), _bits(expected)), msg)


def _fresh_marlin(ep_rank: int, stacks: dict) -> dict:
    """Marlin tensors of a model that loads ``stacks`` and prepares once."""
    model, _, fp4 = _make_model(ep_rank)
    _load_stacks(model, ep_rank, stacks)
    with _marlin_on_cpu():
        fp4.quant_method.process_weights_after_loading(fp4)
    return {name: getattr(fp4, name).detach().clone() for name in MARLIN_NAMES}


class TestPerExpertNvfp4Loading(CustomTestCase):
    def test_streamed_experts_fill_owning_rank_stacks(self):
        tensors = _expert_tensors(seed=0)
        stacks = _stacks(tensors)
        for ep_rank in (0, 1):
            with self.subTest(ep_rank=ep_rank):
                model, _, fp4 = _make_model(ep_rank)
                pointers = {name: p.data_ptr() for name, p in fp4.named_parameters()}
                # one call per tensor, so gate and up global scales arrive apart
                for name, tensor in _per_expert_names(tensors, (3, 0, 2, 1)):
                    model.load_weights([(name, tensor)])
                owned = slice(ep_rank * NUM_LOCAL, (ep_rank + 1) * NUM_LOCAL)
                for name, expected in stacks.items():
                    if name.endswith("_input_scale"):
                        continue
                    _assert_bitwise_equal(
                        self, getattr(fp4, name), expected[owned], msg=name
                    )
                self.assertEqual(
                    pointers,
                    {name: p.data_ptr() for name, p in fp4.named_parameters()},
                )
                self.assertEqual(fp4.nvfp4_stale_experts, {0, 1})

    def test_bf16_layer_of_nvfp4_checkpoint_takes_per_expert_names(self):
        """The checkpoint's excluded layers keep BF16 experts; their streamed
        per-expert names must load there instead of failing as unknown."""
        gen = torch.Generator().manual_seed(0)
        gate_up = torch.randn(NUM_EXPERTS, 2 * INTER, HIDDEN, generator=gen)
        down = torch.randn(NUM_EXPERTS, HIDDEN, INTER, generator=gen)
        for ep_rank in (0, 1):
            with self.subTest(ep_rank=ep_rank):
                model, bf16, fp4 = _make_model(ep_rank)
                for expert_id in range(NUM_EXPERTS):
                    for projection, tensor in (
                        ("gate_proj", gate_up[expert_id, :INTER]),
                        ("up_proj", gate_up[expert_id, INTER:]),
                        ("down_proj", down[expert_id]),
                    ):
                        model.load_weights(
                            [
                                (
                                    f"{BF16_PREFIX}.{expert_id}.{projection}.weight",
                                    tensor.to(torch.bfloat16),
                                )
                            ]
                        )
                owned = slice(ep_rank * NUM_LOCAL, (ep_rank + 1) * NUM_LOCAL)
                _assert_bitwise_equal(
                    self, bf16.gate_up_proj, gate_up[owned].to(torch.bfloat16)
                )
                _assert_bitwise_equal(
                    self, bf16.down_proj, down[owned].to(torch.bfloat16)
                )
                self.assertEqual(fp4.nvfp4_stale_experts, set())

        model, _, _ = _make_model(0)
        for suffix, tensor in (
            ("weight", torch.zeros(INTER, HIDDEN // 2, dtype=torch.uint8)),
            ("weight_scale", torch.ones(INTER, HIDDEN // 16).to(torch.float8_e4m3fn)),
        ):
            with self.subTest(suffix=suffix):
                with self.assertRaisesRegex(ValueError, "not an NVFP4 expert layer"):
                    model.load_weights(
                        [(f"{BF16_PREFIX}.0.gate_proj.{suffix}", tensor)]
                    )

    def test_wrong_dtype_or_shape_is_rejected_before_writing(self):
        model, _, fp4 = _make_model(0)
        fp4.gate_up_proj.data.fill_(7)
        good = torch.zeros(INTER, HIDDEN // 2, dtype=torch.uint8)
        for name, tensor in (
            # raw scale bytes would be cast numerically
            (f"{NVFP4_PREFIX}.0.gate_proj.weight_scale", good[:, :4]),
            (f"{NVFP4_PREFIX}.0.up_proj.weight", good[:, :1]),
            (f"{NVFP4_PREFIX}.0.up_proj.weight_scale_2", torch.ones(2)),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, "mismatch"):
                    model.load_weights([(name, tensor)])
        self.assertTrue(torch.all(fp4.gate_up_proj == 7))
        self.assertEqual(fp4.nvfp4_stale_experts, set())


class TestMarlinRefresh(CustomTestCase):
    def _prepared(self, ep_rank: int, stacks: dict):
        model, _, fp4 = _make_model(ep_rank)
        _load_stacks(model, ep_rank, stacks)
        with _marlin_on_cpu():
            fp4.quant_method.process_weights_after_loading(fp4)
        return model, fp4

    def test_load_weights_rederives_only_streamed_experts_in_place(self):
        initial, update = _expert_tensors(seed=0), _expert_tensors(seed=1)
        model, fp4 = self._prepared(1, _stacks(initial))
        # the Marlin build collapses its own copy of the gate/up global scales
        _assert_bitwise_equal(
            self,
            fp4.gate_up_proj_weight_scale_2,
            _stacks(initial)["gate_up_proj_weight_scale_2"][NUM_LOCAL:],
        )
        pointers = {name: getattr(fp4, name).data_ptr() for name in MARLIN_NAMES}
        repack = Mock(side_effect=_fake_marlin_repack)
        with _marlin_on_cpu(repack):
            model.load_weights(_per_expert_names(update, (3,)))
        # global expert 3 is local expert 1 on rank 1: one w13 and one w2 repack
        self.assertEqual(repack.call_count, 2)
        self.assertEqual(fp4.nvfp4_stale_experts, set())

        expected_tensors = dict(initial)
        expected_tensors.update(
            {key: value for key, value in update.items() if key[0] == 3}
        )
        expected = _fresh_marlin(1, _stacks(expected_tensors))
        for name in MARLIN_NAMES:
            _assert_bitwise_equal(self, getattr(fp4, name), expected[name], msg=name)
            self.assertEqual(getattr(fp4, name).data_ptr(), pointers[name], name)

    def test_stacked_reload_rederives_every_expert_once(self):
        """update_weights_from_disk loads whole stacks, then post-processes again."""
        model, fp4 = self._prepared(0, _stacks(_expert_tensors(seed=0)))
        update = _stacks(_expert_tensors(seed=1))
        repack = Mock(side_effect=_fake_marlin_repack)
        with _marlin_on_cpu(repack):
            _load_stacks(model, 0, update)
            fp4.quant_method.process_weights_after_loading(fp4)
        self.assertEqual(repack.call_count, 2 * NUM_LOCAL)
        expected = _fresh_marlin(0, update)
        for name in MARLIN_NAMES:
            _assert_bitwise_equal(self, getattr(fp4, name), expected[name], msg=name)

    def test_default_releases_checkpoint_layout_and_rejects_updates(self):
        stacks = _stacks(_expert_tensors(seed=0))
        model, _, fp4 = _make_model(0, keep_checkpoint_layout=False)
        _load_stacks(model, 0, stacks)
        with _marlin_on_cpu():
            fp4.quant_method.process_weights_after_loading(fp4)
        self.assertNotIn("gate_up_proj", dict(fp4.named_parameters()))
        self.assertIn(
            "SGLANG_ENABLE_MUSE_SPARK_NVFP4_WEIGHT_UPDATE",
            fp4._derived_weight_cache_error,
        )
        tensors = _expert_tensors(seed=1)
        for names in (
            _per_expert_names(tensors, (0,))[:1],
            [(f"{NVFP4_PREFIX}.gate_up_proj", stacks["gate_up_proj"])],
        ):
            with self.subTest(name=names[0][0]):
                with self.assertRaisesRegex(
                    ValueError, "SGLANG_ENABLE_MUSE_SPARK_NVFP4_WEIGHT_UPDATE"
                ):
                    model.load_weights(names)


if __name__ == "__main__":
    unittest.main()
