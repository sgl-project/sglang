import unittest

import torch

from sglang.kernels.ops.speculative.scatter_spec_extras import scatter_spec_extras
from sglang.srt.managers.overlap_utils import _can_scatter_spec_extras
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


def _random_tensor(shape, dtype, device):
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.randint(0, 32000, shape, dtype=dtype, device=device)


def _make_pair(
    pool_size,
    batch_size,
    row_shape,
    dtype,
    device,
    *,
    source_dtype=None,
):
    destination = _random_tensor((pool_size, *row_shape), dtype, device)
    source = _random_tensor((batch_size, *row_shape), source_dtype or dtype, device)
    return destination, source


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip is None,
    "NVIDIA CUDA is required for this test.",
)
class TestScatterSpecExtras(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.device = torch.device("cuda")

    def _run_case(
        self,
        *,
        pool_size=64,
        batch_size=7,
        topk=False,
        hidden=False,
        draft_probs=False,
        dsa=False,
        hidden_dim=513,
        draft_shape=(3, 17),
        index_dtype=torch.int64,
        strided_indices=False,
    ):
        selected = torch.randperm(pool_size, device=self.device)[:batch_size].to(
            index_dtype
        )
        if strided_indices:
            index_storage = torch.empty(
                batch_size * 2, dtype=index_dtype, device=self.device
            )
            index_storage[::2] = selected
            indices = index_storage[::2]
            self.assertFalse(indices.is_contiguous())
        else:
            indices = selected

        destinations = {}
        sources = {}
        destinations["output"], sources["output"] = _make_pair(
            pool_size,
            batch_size,
            (),
            torch.int64,
            self.device,
            source_dtype=torch.int32,
        )
        if topk:
            destinations["topk_p"], sources["topk_p"] = _make_pair(
                pool_size, batch_size, (4,), torch.float32, self.device
            )
            destinations["topk_index"], sources["topk_index"] = _make_pair(
                pool_size, batch_size, (4,), torch.int64, self.device
            )
        if hidden:
            destinations["hidden"], sources["hidden"] = _make_pair(
                pool_size,
                batch_size,
                (hidden_dim,),
                torch.bfloat16,
                self.device,
            )
        if draft_probs:
            destinations["draft"], sources["draft"] = _make_pair(
                pool_size,
                batch_size,
                draft_shape,
                torch.float32,
                self.device,
            )
        if dsa:
            destinations["dsa"], sources["dsa"] = _make_pair(
                pool_size, batch_size, (16,), torch.int32, self.device
            )

        expected = {name: tensor.clone() for name, tensor in destinations.items()}
        source_snapshots = {name: tensor.clone() for name, tensor in sources.items()}
        for name, source in sources.items():
            expected[name][indices] = source.to(expected[name].dtype)

        scatter_spec_extras(
            indices,
            output_tokens_buf=destinations["output"],
            bonus_tokens=sources["output"],
            topk_p_buf=destinations.get("topk_p"),
            topk_p=sources.get("topk_p"),
            topk_index_buf=destinations.get("topk_index"),
            topk_index=sources.get("topk_index"),
            hidden_states_buf=destinations.get("hidden"),
            hidden_states=sources.get("hidden"),
            draft_probs_buf=destinations.get("draft"),
            draft_probs=sources.get("draft"),
            dsa_topk_indices_buf=destinations.get("dsa"),
            dsa_topk_indices=sources.get("dsa"),
        )

        for name, destination in destinations.items():
            torch.testing.assert_close(destination, expected[name], rtol=0, atol=0)
        for name, source in sources.items():
            torch.testing.assert_close(source, source_snapshots[name], rtol=0, atol=0)

    def test_glm_trace_specialization(self):
        # GLM-5.2 trace: top-k + BF16 hidden(6144) + DSA, without draft probs.
        self._run_case(
            pool_size=49,
            batch_size=1,
            topk=True,
            hidden=True,
            dsa=True,
            hidden_dim=6144,
        )

    def test_optional_specializations(self):
        for name, options in (
            ("bonus_only", {}),
            ("hidden_only", {"hidden": True}),
            ("topk_and_draft", {"topk": True, "draft_probs": True}),
            (
                "all_fields",
                {
                    "topk": True,
                    "hidden": True,
                    "draft_probs": True,
                    "dsa": True,
                },
            ),
        ):
            with self.subTest(name=name):
                self._run_case(**options)

    def test_index_layouts(self):
        for name, options in (
            ("int32", {"index_dtype": torch.int32}),
            (
                "strided_int64",
                {"index_dtype": torch.int64, "strided_indices": True},
            ),
        ):
            with self.subTest(name=name):
                self._run_case(topk=True, hidden=True, dsa=True, **options)

    def test_layout_contract(self):
        indices = torch.tensor([1, 3], dtype=torch.int64, device=self.device)
        destination = torch.empty((8, 2), device=self.device)
        source = torch.empty((2, 2), device=self.device)
        self.assertTrue(
            _can_scatter_spec_extras(indices, ((destination, source),), None)
        )

        storage = torch.empty((2, 4), device=self.device)
        self.assertFalse(
            _can_scatter_spec_extras(indices, ((destination, storage[:, ::2]),), None)
        )
        self.assertFalse(
            _can_scatter_spec_extras(
                indices,
                ((destination, source),),
                (destination, source.to(torch.float16)),
            )
        )

    def test_empty_indices_is_noop(self):
        destination = _random_tensor((16,), torch.int64, self.device)
        before = destination.clone()
        scatter_spec_extras(
            torch.empty(0, dtype=torch.int64, device=self.device),
            output_tokens_buf=destination,
            bonus_tokens=torch.empty(0, dtype=torch.int32, device=self.device),
        )
        torch.testing.assert_close(destination, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
