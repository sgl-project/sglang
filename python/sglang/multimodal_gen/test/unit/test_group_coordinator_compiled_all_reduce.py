"""TP all-reduce: compiled path stays one graph; custom all-reduce gating."""

import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.distributed import group_coordinator
from sglang.multimodal_gen.runtime.distributed.group_coordinator import (
    GroupCoordinator,
)
from sglang.test.test_utils import CustomTestCase


class _Group(SimpleNamespace):
    """Weak-referenceable, as the op registry holds groups by weak reference."""

    def _all_reduce_out_of_place(self, tensor):
        return GroupCoordinator._all_reduce_out_of_place(self, tensor)


def _group(name: str, *, accepts: bool) -> _Group:
    custom_ar = SimpleNamespace(
        disabled=False,
        should_custom_ar=lambda tensor: accepts,
        # Stands in for the tvm_ffi communicator call Dynamo cannot trace.
        custom_all_reduce=torch._dynamo.disable(lambda tensor: tensor * 2),
    )
    group = _Group(
        world_size=2,
        unique_name=name,
        srt_custom_allreduce=custom_ar,
        device_group=None,
    )
    group_coordinator._groups[name] = weakref.ref(group)
    return group


class TestCompiledAllReduce(CustomTestCase):
    def test_compiled_all_reduce_is_one_graph(self):
        group = _group("tp:compiled", accepts=True)
        compiled = torch.compile(
            lambda x: GroupCoordinator.all_reduce(group, x),
            fullgraph=True,
            backend="eager",
        )

        out = compiled(torch.empty(4, 8, device="meta"))

        self.assertEqual(out.shape, (4, 8))

    def test_op_leaves_its_input_untouched(self):
        schema = torch.ops.sglang.diffusion_all_reduce.default._schema
        self.assertFalse(
            any(a.alias_info and a.alias_info.is_write for a in schema.arguments)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "the op is registered for CUDA")
    def test_op_reaches_the_group_by_name(self):
        group = _group("tp:dispatch", accepts=True)
        x = torch.ones(4, device="cuda")

        out = torch.ops.sglang.diffusion_all_reduce(x, group.unique_name)

        torch.testing.assert_close(out, x * 2)
        torch.testing.assert_close(x, torch.ones(4, device="cuda"))

    def test_out_of_place_reduce_uses_the_custom_all_reduce(self):
        group = _group("tp:custom", accepts=True)
        x = torch.ones(4)

        torch.testing.assert_close(group._all_reduce_out_of_place(x), x * 2)

    def test_out_of_place_fallback_reduces_a_copy(self):
        group = _group("tp:fallback", accepts=False)
        x = torch.ones(4)

        with patch.object(
            group_coordinator.torch.distributed,
            "all_reduce",
            side_effect=lambda tensor, group: tensor.mul_(3),
        ):
            out = group._all_reduce_out_of_place(x)

        torch.testing.assert_close(out, torch.full((4,), 3.0))
        torch.testing.assert_close(x, torch.ones(4))


def _custom_ar_supported(*, rocm: bool, gfx1250: bool) -> bool:
    platform = SimpleNamespace(is_rocm=lambda: rocm, is_cuda_alike=lambda: True)
    with (
        patch.object(group_coordinator, "current_platform", platform),
        patch.object(group_coordinator, "is_gfx1250_supported", lambda: gfx1250),
    ):
        return group_coordinator._srt_custom_allreduce_supported()


class TestCustomAllReduceGate(CustomTestCase):
    def test_gfx1250_reduces_over_rccl(self):
        """The ROCm custom all-reduce returns wrong sums on gfx1250 (MI450),
        which corrupted TP2 FLUX.2 text-encoder outputs; it must stay off there."""
        self.assertFalse(_custom_ar_supported(rocm=True, gfx1250=True))

    def test_other_rocm_archs_keep_custom_all_reduce(self):
        self.assertTrue(_custom_ar_supported(rocm=True, gfx1250=False))

    def test_cuda_keeps_custom_all_reduce(self):
        self.assertTrue(_custom_ar_supported(rocm=False, gfx1250=False))


if __name__ == "__main__":
    unittest.main()
