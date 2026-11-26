"""
Unit tests for the gfx11 row-stride pad on unquantized linear and lm_head weights.

The pad must change only the row stride: same shape, same values, same outputs.
It applies to linear weights and the LM head (tied or not), never to an untied
input embedding, and is judged by each rank's own K shard.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization.unquant as unquant
from sglang.srt.environ import envs
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

# 1024 bf16 elements are 2 KiB per row, so these weights alias and get one 128 B line.
_VOCAB, _HIDDEN, _PADDED_STRIDE = 256, 1024, 1024 + 64
_BF16_RTOL, _BF16_ATOL = 1.6e-2, 1e-2


def _on_gfx11():
    # _is_cpu is cleared so a CPU runner with AMX does not take the AMX packing branch.
    return patch.multiple(unquant, _is_gfx11=True, _is_cpu=False)


def _publish(*, tp_size: int, device: str = "cpu") -> None:
    reset_context()
    publish(
        ServerArgs(model_path="dummy", device=device, tp_size=tp_size),
        role="test",
        ranks=SpawnRanks(world_rank=tp_size - 1),
    )
    # amd_march has no sglang.test.parallel_groups; the embedding reads these group stubs.
    parallel = get_parallel()
    parallel.override_permanently(
        **{
            f"{name}_group": SimpleNamespace(
                rank_in_group=getattr(parallel, f"{name}_rank"),
                world_size=getattr(parallel, f"{name}_size"),
            )
            for name in ("tp", "attn_tp")
        }
    )


class _OpaqueHeadMethod(QuantizeMethodBase):
    def apply(self, layer, *args, **kwargs):
        raise NotImplementedError


class _CausalLM(torch.nn.Module):
    def __init__(self, *, tie: str):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(
            _VOCAB, _HIDDEN, params_dtype=torch.bfloat16
        )
        self.model.proj = ReplicatedLinear(
            _HIDDEN, 64, bias=False, params_dtype=torch.bfloat16
        )
        if tie == "module":
            self.lm_head = self.model.embed_tokens
        elif tie == "parameter":
            self.lm_head = ParallelLMHead(
                _VOCAB, _HIDDEN, params_dtype=torch.bfloat16
            ).tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = ParallelLMHead(_VOCAB, _HIDDEN, params_dtype=torch.bfloat16)
        for param in self.parameters():
            param.data.copy_(torch.randn_like(param))


def _postprocess(model: torch.nn.Module) -> None:
    DefaultModelLoader.postprocess_weights(model, torch.device("cpu"))


class TestGfx11WeightPadding(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        _publish(tp_size=1)
        self.addCleanup(reset_context)

    def test_pad_rule_counts_row_bytes_not_elements(self):
        cases = {
            # (K, itemsize): pad in elements
            (4096, 2): 64,
            (14336, 2): 64,
            (11008, 2): 0,
            (512, 2): 0,
            (512, 4): 32,
            (3584, 2): 0,
        }
        for (k, itemsize), pad in cases.items():
            with self.subTest(k=k, itemsize=itemsize):
                self.assertEqual(
                    unquant._gfx11_row_pad_elems(k=k, itemsize=itemsize), pad
                )

    def test_pad_changes_only_the_row_stride(self):
        weight = torch.randn(192, 4096, dtype=torch.bfloat16)
        bias = torch.randn(192, dtype=torch.bfloat16)
        x = torch.randn(3, 4096, dtype=torch.bfloat16)
        with _on_gfx11():
            padded = unquant.maybe_pad_weight_for_gfx11(weight)

        self.assertEqual(padded.shape, weight.shape)
        self.assertEqual(padded.stride(), (4096 + 64, 1))
        self.assertTrue(torch.equal(padded, weight))
        torch.testing.assert_close(
            F.linear(x, padded, bias),
            F.linear(x, weight, bias),
            rtol=_BF16_RTOL,
            atol=_BF16_ATOL,
        )

    def test_gate_leaves_weight_untouched(self):
        weight = torch.randn(64, 4096, dtype=torch.bfloat16)
        with patch.object(unquant, "_is_gfx11", False), self.subTest("not gfx11"):
            self.assertIs(unquant.maybe_pad_weight_for_gfx11(weight), weight)
        with _on_gfx11(), envs.SGLANG_DISABLE_GFX11_WEIGHT_PADDING.override(True):
            with self.subTest("opted out"):
                self.assertIs(unquant.maybe_pad_weight_for_gfx11(weight), weight)
        with _on_gfx11():
            padded = unquant.maybe_pad_weight_for_gfx11(weight)
            with self.subTest("already padded"):
                self.assertIs(unquant.maybe_pad_weight_for_gfx11(padded), padded)
            fp32 = torch.randn(64, 512, dtype=torch.float32)
            with self.subTest("fp32, a 2 KiB row but unmeasured"):
                self.assertIs(unquant.maybe_pad_weight_for_gfx11(fp32), fp32)
            transposed = torch.randn(4096, 64, dtype=torch.bfloat16).t()
            with self.subTest("non-contiguous"):
                self.assertIs(
                    unquant.maybe_pad_weight_for_gfx11(transposed), transposed
                )

    def test_untied_input_embedding_is_not_padded(self):
        model = _CausalLM(tie="none")
        with _on_gfx11():
            _postprocess(model)

        self.assertTrue(model.model.embed_tokens.weight.is_contiguous())
        self.assertEqual(model.lm_head.weight.stride(0), _PADDED_STRIDE)
        self.assertEqual(model.model.proj.weight.stride(0), _PADDED_STRIDE)

    def test_tied_head_is_padded_once_and_still_embeds(self):
        ids = torch.tensor([0, 7, _VOCAB - 1])
        for tie in ("module", "parameter"):
            with self.subTest(tie=tie):
                model = _CausalLM(tie=tie)
                expected_rows = model.model.embed_tokens.weight[ids].clone()
                with _on_gfx11():
                    _postprocess(model)
                    storage = model.lm_head.weight.data_ptr()
                    # A weight update re-runs postprocessing on the same model.
                    _postprocess(model)

                self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
                self.assertEqual(model.lm_head.weight.stride(0), _PADDED_STRIDE)
                self.assertEqual(model.lm_head.weight.data_ptr(), storage)
                torch.testing.assert_close(
                    model.model.embed_tokens(ids), expected_rows, rtol=0, atol=0
                )

    def test_quantized_head_is_not_padded(self):
        model = _CausalLM(tie="none")
        model.lm_head.quant_method = _OpaqueHeadMethod()
        with _on_gfx11():
            _postprocess(model)

        self.assertTrue(model.lm_head.weight.is_contiguous())

    def test_padding_logs_at_info_only_on_the_first_load(self):
        model = _CausalLM(tie="module")
        with _on_gfx11(), patch.object(unquant, "_gfx11_padding_logged", False):
            with self.assertLogs(unquant.logger, level="DEBUG") as logs:
                for _ in range(3):
                    _postprocess(model)

        levels = [r.levelname for r in logs.records if "gfx11" in r.getMessage()]
        self.assertEqual(levels, ["INFO", "DEBUG", "DEBUG"])

    def test_reload_writes_into_padded_storage(self):
        layer = ColumnParallelLinear(
            _HIDDEN, 64, bias=False, params_dtype=torch.bfloat16
        )
        with _on_gfx11():
            layer.quant_method.process_weights_after_loading(layer)
            storage = layer.weight.data_ptr()
            fresh = torch.randn(64, _HIDDEN, dtype=torch.bfloat16)
            layer.weight.weight_loader(layer.weight, fresh)
            layer.quant_method.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.stride(0), _PADDED_STRIDE)
        self.assertEqual(layer.weight.data_ptr(), storage)
        self.assertTrue(torch.equal(layer.weight, fresh))


class TestGfx11WeightPaddingShards(CustomTestCase):
    def setUp(self):
        _publish(tp_size=4)
        self.addCleanup(reset_context)

    def test_row_parallel_pad_follows_the_shard_k(self):
        """K = 3072 aliases (6 KiB rows), but a TP=4 row shard holds K = 768 (1.5 KiB)."""
        row = RowParallelLinear(3072, 64, bias=False, params_dtype=torch.bfloat16)
        column = ColumnParallelLinear(3072, 64, bias=False, params_dtype=torch.bfloat16)
        with _on_gfx11():
            for layer in (row, column):
                layer.quant_method.process_weights_after_loading(layer)

        self.assertEqual(tuple(row.weight.shape), (64, 768))
        self.assertTrue(row.weight.is_contiguous())
        self.assertEqual(tuple(column.weight.shape), (16, 3072))
        self.assertEqual(column.weight.stride(0), 3072 + 64)

    def test_checkpoint_loaders_reload_into_padded_storage(self):
        """Fused loaders slice rows and RowParallelLinear slices K before copying."""
        # Each shard ends up with K = 1024, a 2 KiB bf16 row.
        cases = {
            "qkv": (
                lambda: QKVParallelLinear(
                    _HIDDEN, 64, 16, 4, bias=False, params_dtype=torch.bfloat16
                ),
                {"q": (1024, _HIDDEN), "k": (256, _HIDDEN), "v": (256, _HIDDEN)},
            ),
            "merged": (
                lambda: MergedColumnParallelLinear(
                    _HIDDEN, [256, 256], bias=False, params_dtype=torch.bfloat16
                ),
                {0: (256, _HIDDEN), 1: (256, _HIDDEN)},
            ),
            "row": (
                lambda: RowParallelLinear(
                    4 * _HIDDEN, 64, bias=False, params_dtype=torch.bfloat16
                ),
                {None: (64, 4 * _HIDDEN)},
            ),
        }
        for kind, (make, shards) in cases.items():
            with self.subTest(kind=kind):
                padded, reference = make(), make()
                with _on_gfx11():
                    padded.quant_method.process_weights_after_loading(padded)
                    storage = padded.weight.data_ptr()
                    for shard_id, shape in shards.items():
                        full = torch.randn(shape, dtype=torch.bfloat16)
                        for layer in (padded, reference):
                            args = () if shard_id is None else (shard_id,)
                            layer.weight.weight_loader(layer.weight, full, *args)
                    padded.quant_method.process_weights_after_loading(padded)

                self.assertEqual(padded.weight.stride(0), _PADDED_STRIDE)
                self.assertEqual(padded.weight.data_ptr(), storage)
                self.assertTrue(torch.equal(padded.weight, reference.weight))


@unittest.skipUnless(unquant._is_gfx11, "needs an AMD gfx11 GPU")
class TestGfx11WeightPaddingOnDevice(CustomTestCase):
    """hipBLASLt and the embedding gather read the padded stride correctly."""

    def setUp(self):
        torch.manual_seed(0)
        _publish(tp_size=1, device="cuda")
        self.addCleanup(reset_context)

    @torch.inference_mode()
    def test_linear_outputs_match_unpadded(self):
        for k, n in ((4096, 6144), (4096, 4096), (14336, 4096)):
            layer = ReplicatedLinear(k, n, bias=True, params_dtype=torch.bfloat16)
            layer = layer.cuda()
            layer.weight.copy_(torch.randn_like(layer.weight) / k**0.5)
            layer.bias.copy_(torch.randn_like(layer.bias))
            reference_weight = layer.weight.clone()
            layer.quant_method.process_weights_after_loading(layer)
            self.assertEqual(layer.weight.stride(0), k + 64)

            for m in (1, 8, 300):
                with self.subTest(k=k, n=n, m=m):
                    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                    torch.testing.assert_close(
                        layer(x)[0],
                        F.linear(x, reference_weight, layer.bias),
                        rtol=_BF16_RTOL,
                        atol=_BF16_ATOL,
                    )

    @torch.inference_mode()
    def test_tied_head_embeds_and_projects(self):
        model = _CausalLM(tie="module").cuda()
        reference = model.lm_head.weight.clone()
        DefaultModelLoader.postprocess_weights(model, torch.device("cuda"))
        self.assertEqual(model.lm_head.weight.stride(0), _PADDED_STRIDE)

        ids = torch.tensor([0, 7, _VOCAB - 1], device="cuda")
        torch.testing.assert_close(
            model.model.embed_tokens(ids), reference[ids], rtol=0, atol=0
        )
        hidden = torch.randn(2, _HIDDEN, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(
            hidden @ model.lm_head.weight.T,
            hidden @ reference.T,
            rtol=_BF16_RTOL,
            atol=_BF16_ATOL,
        )


if __name__ == "__main__":
    unittest.main()
