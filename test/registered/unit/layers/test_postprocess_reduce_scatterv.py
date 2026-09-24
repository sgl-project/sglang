"""Under DP attention with reduce_scatterv, postprocess sums a FULL-mode layer's
output while scattering it back to the local tokens only when the FFN left its
all-reduce out."""

import contextlib
import itertools
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import communicator as comm
from sglang.srt.layers.communicator import (
    CommunicateSummableTensorPairFn,
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.layers.moe.utils import (
    should_skip_mlp_all_reduce,
    should_skip_post_experts_all_reduce,
)
from sglang.srt.runtime_context import get_forward
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextlib.contextmanager
def reduce_scatterv_applies():
    with (
        patch.object(comm, "should_use_dp_reduce_scatterv", return_value=True),
        patch.object(moe_utils, "should_use_dp_reduce_scatterv", return_value=True),
    ):
        yield


def postprocess_sums(*, allow_reduce_scatter, is_layer_sparse):
    calls = []
    parallel = types.SimpleNamespace(
        tp_size=2,
        attn_dp_size=2,
        tp_group=types.SimpleNamespace(
            reduce_scatterv=lambda *args, **kwargs: calls.append("reduce_scatterv")
        ),
    )
    forward_batch = types.SimpleNamespace(
        dp_padding_mode=types.SimpleNamespace(is_max_len=lambda: False)
    )
    with (
        reduce_scatterv_applies(),
        patch.object(comm, "get_parallel", return_value=parallel),
        patch.object(comm, "get_local_dp_buffer", return_value=torch.empty(1, 4)),
        patch.object(comm, "get_dp_global_num_tokens", return_value=[1, 1]),
        patch.object(
            comm, "dp_scatter", side_effect=lambda *args: calls.append("dp_scatter")
        ),
    ):
        CommunicateSummableTensorPairFn._scatter_hidden_states(
            torch.zeros(2, 4),
            None,
            forward_batch,
            context=None,
            allow_reduce_scatter=allow_reduce_scatter,
            is_layer_sparse=is_layer_sparse,
        )
    assert len(calls) == 1, calls
    return calls[0] == "reduce_scatterv"


def ffn_leaves_the_sum_out(*, allow_reduce_scatter, is_layer_sparse):
    # A layer that allows reduce-scatter publishes mlp_reduce_scatter while
    # reduce_scatterv applies; a dense MLP reads only that flag.
    with (
        reduce_scatterv_applies(),
        patch.object(
            moe_utils,
            "get_parallel",
            return_value=types.SimpleNamespace(tp_size=2, dwdp_size=1),
        ),
        patch.object(
            moe_utils,
            "get_moe_a2a_backend",
            return_value=types.SimpleNamespace(
                is_flashinfer=lambda: False,
                is_pplx=lambda: False,
                is_flashinfer_megamoe=lambda: False,
            ),
        ),
        patch.object(
            moe_utils,
            "should_use_flashinfer_cutlass_moe_fp4_allgather",
            return_value=False,
        ),
        get_forward().scoped(mlp_reduce_scatter=allow_reduce_scatter),
    ):
        if is_layer_sparse:
            return should_skip_post_experts_all_reduce(is_tp_path=True)
        return should_skip_mlp_all_reduce()


class TestPostprocessReduceScatterv(CustomTestCase):
    def test_sums_exactly_when_the_ffn_left_the_sum_out(self):
        for allow, sparse in itertools.product((False, True), repeat=2):
            case = dict(allow_reduce_scatter=allow, is_layer_sparse=sparse)
            with self.subTest(**case):
                self.assertEqual(
                    postprocess_sums(**case), ffn_leaves_the_sum_out(**case)
                )

    def test_a_dense_layer_without_reduce_scatter_is_not_summed_again(self):
        self.assertFalse(
            postprocess_sums(allow_reduce_scatter=False, is_layer_sparse=False)
        )

    def test_layer_modes_record_whether_the_layer_is_sparse(self):
        for sparse in (False, True):
            with (
                self.subTest(sparse=sparse),
                patch.object(
                    comm, "sparse_mlp_scatter_mode", return_value=ScatterMode.FULL
                ),
                patch.object(comm, "enable_moe_dense_fully_dp", return_value=False),
                patch.object(
                    comm, "_generic_prefill_cp_shards_tokens", return_value=False
                ),
            ):
                modes = LayerScatterModes.init_new(
                    layer_id=1,
                    num_layers=4,
                    is_layer_sparse=sparse,
                    is_previous_layer_sparse=sparse,
                    is_next_layer_sparse=sparse,
                )
            self.assertEqual(modes.is_layer_sparse, sparse)

    def test_postprocess_passes_the_layer_sparsity(self):
        seen = {}
        communicator = LayerCommunicator.__new__(LayerCommunicator)
        communicator._sp_variant = None
        communicator._context = None
        communicator.allow_reduce_scatter = False
        communicator.layer_scatter_modes = types.SimpleNamespace(is_layer_sparse=True)
        communicator._communicate_summable_tensor_pair_fn = lambda **kwargs: (
            seen.update(kwargs) or (None, None)
        )
        communicator.postprocess_layer(None, None, None)
        self.assertIs(seen["is_layer_sparse"], True)


class TestReduceAndRedistributeOutputStep(CustomTestCase):
    """The reduce-scatter that brings a FULL-layout output back to this rank's
    tokens, or None when only a scatter remains."""

    def step(self, *, varlen, max_len, tiles, allow, sparse):
        forward_batch = types.SimpleNamespace(
            dp_padding_mode=types.SimpleNamespace(is_max_len=lambda: max_len)
        )
        with (
            patch.object(comm, "should_use_dp_reduce_scatterv", return_value=varlen),
            patch.object(comm, "can_use_dp_reduce_scatter", return_value=tiles),
        ):
            return comm._reduce_and_redistribute_output_step(
                forward_batch, allow_reduce_scatter=allow, is_layer_sparse=sparse
            )

    def test_every_condition(self):
        for varlen, max_len, tiles, allow, sparse in itertools.product(
            (False, True), repeat=5
        ):
            if varlen and (allow or sparse):
                expected = comm._reduce_and_redistribute_output_varlen
            elif allow and max_len and tiles:
                expected = comm._reduce_and_redistribute_output_max_len
            else:
                expected = None
            case = dict(
                varlen=varlen, max_len=max_len, tiles=tiles, allow=allow, sparse=sparse
            )
            with self.subTest(**case):
                self.assertIs(self.step(**case), expected)


class TestLongcatNextnReducesItsMlp(CustomTestCase):
    """Its dense layer allows no reduce-scatter, so postprocess never sums the
    MLP output; the MLP must all-reduce it itself."""

    def test_the_dense_mlp_reduces_its_output(self):
        from sglang.srt.models import longcat_flash_nextn as nextn

        mlp_kwargs = {}
        config = types.SimpleNamespace(
            hidden_size=8,
            intermediate_size=16,
            hidden_act="silu",
            num_attention_heads=1,
            qk_nope_head_dim=1,
            qk_rope_head_dim=1,
            v_head_dim=1,
            q_lora_rank=None,
            kv_lora_rank=1,
            rope_theta=1.0,
            max_position_embeddings=8,
            rms_norm_eps=1e-6,
            num_hidden_layers=1,
        )
        with (
            patch.object(nextn, "DeepseekV2AttentionMLA"),
            patch.object(
                nextn,
                "LongcatFlashMLP",
                side_effect=lambda **kwargs: mlp_kwargs.update(kwargs),
            ),
            patch.object(nextn, "RMSNorm"),
            patch.object(nextn, "get_parallel"),
            patch.object(nextn, "LayerScatterModes"),
            patch.object(nextn, "LayerCommunicator"),
        ):
            nextn.LongcatFlashDenseDecoderLayer(config, layer_id=0)
        self.assertIs(mlp_kwargs.get("reduce_results"), True)


if __name__ == "__main__":
    unittest.main()
