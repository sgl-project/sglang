"""CPU-only unit tests for the weight-cache parity manifest.

The manifest is the Phase 5 correctness gate for XPU: a daemon-loaded model's
tensors are declared equal to a disk-loaded model's because these fingerprints
match. A harness that cannot fail says nothing, so each test here breaks one
property on purpose and asserts the comparison reports it, and one asserts the
case that must NOT be reported -- an XPU MoE expert weight's unwritten row
padding, which is uninitialized memory on both sides.

No GPU and no model download: a torch Module with hand-built tensors reaches
every branch, including the padded-view layout
``xpu_moe_ld_padding_elems`` produces.
"""

import unittest

import torch

from sglang.srt.layers.moe.utils import (
    XPU_MOE_LD_PADDING_BYTES,
    xpu_moe_ld_padding_elems,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.weight_cache_parity_utils import (
    collect_weight_manifest,
    compare_manifests,
    manifest_summary,
)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

EXPERTS, N_DIM, K_DIM = 4, 6, 8


def _expert_weight(values: torch.Tensor, pad_elems: int) -> torch.Tensor:
    """``values`` as an [E, N, K] tensor, optionally a view into a padded buffer.

    The pad columns are filled with a sentinel rather than left to
    ``torch.empty``, so a digest that wrongly covered them would differ between
    two calls and the test would be asserting luck. Production leaves them
    uninitialized, which is exactly why the digest must not read them.
    """
    experts, n_dim, k_dim = values.shape
    if pad_elems == 0:
        return values.clone()
    buffer = torch.full((experts, n_dim, k_dim + pad_elems), 7, dtype=values.dtype)
    buffer[:, :, :k_dim] = values
    return buffer[:, :, :k_dim]


class _Model(torch.nn.Module):
    """A dense weight, a tied pair, a buffer, and one MoE expert weight."""

    def __init__(self, *, pad_elems=0, expert_values=None, dense_values=None):
        super().__init__()
        if expert_values is None:
            expert_values = torch.arange(
                EXPERTS * N_DIM * K_DIM, dtype=torch.bfloat16
            ).reshape(EXPERTS, N_DIM, K_DIM)
        if dense_values is None:
            dense_values = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        self.w13_weight = torch.nn.Parameter(
            _expert_weight(expert_values, pad_elems), requires_grad=False
        )
        self.proj = torch.nn.Parameter(dense_values.clone(), requires_grad=False)
        # Tied: must land in the same storage group under both names.
        self.lm_head = self.proj
        self.register_buffer("cos_sin_cache", torch.zeros(5, dtype=torch.float32))


class TestParityManifestAgreement(CustomTestCase):
    def test_two_identical_loads_report_no_difference(self):
        problems = compare_manifests(
            reference=collect_weight_manifest(_Model()),
            actual=collect_weight_manifest(_Model()),
        )
        self.assertEqual(problems, [])

    def test_tied_names_share_one_storage_group(self):
        tensors = collect_weight_manifest(_Model())["tensors"]
        self.assertEqual(
            tensors["proj"]["storage_group"], tensors["lm_head"]["storage_group"]
        )
        self.assertNotEqual(
            tensors["proj"]["storage_group"],
            tensors["w13_weight"]["storage_group"],
        )

    def test_manifest_covers_parameters_and_buffers(self):
        manifest = collect_weight_manifest(_Model())
        self.assertEqual(
            set(manifest["tensors"]),
            {"w13_weight", "proj", "lm_head", "cos_sin_cache"},
        )
        self.assertTrue(manifest["tensors"]["proj"]["is_param"])
        self.assertFalse(manifest["tensors"]["cos_sin_cache"]["is_param"])
        self.assertEqual(manifest["duplicate_names"], [])
        self.assertIn("4 tensors", manifest_summary(manifest))


class TestParityManifestMoePadding(CustomTestCase):
    """The XPU MoE stride padding, which is the case the digest must get right.

    ``_empty_xpu_moe_expert_weight`` returns a narrowed view of an
    over-allocated buffer, so the tensor is non-contiguous and its storage is
    larger than its logical extent. The daemon and a disk-loading engine both
    produce that layout, and neither writes the pad, so the pad is unwritten
    memory that must stay out of the digest while the stride and storage size it
    implies must stay in the comparison.
    """

    PAD = xpu_moe_ld_padding_elems(3072, torch.bfloat16.itemsize)

    def test_padding_is_the_shape_this_test_claims_to_exercise(self):
        # Guards the premise: if the padding rule stops padding this K, the
        # cases below would silently compare two contiguous tensors.
        self.assertEqual(self.PAD, XPU_MOE_LD_PADDING_BYTES // 2)
        self.assertGreater(self.PAD, 0)

    def test_same_padding_on_both_sides_is_parity(self):
        problems = compare_manifests(
            reference=collect_weight_manifest(_Model(pad_elems=self.PAD)),
            actual=collect_weight_manifest(_Model(pad_elems=self.PAD)),
        )
        self.assertEqual(problems, [])

    def test_digest_reads_only_the_logical_elements(self):
        padded = collect_weight_manifest(_Model(pad_elems=self.PAD))["tensors"]
        plain = collect_weight_manifest(_Model(pad_elems=0))["tensors"]
        # Same logical values through different layouts: the digest must agree.
        self.assertEqual(padded["w13_weight"]["sha256"], plain["w13_weight"]["sha256"])

    def test_a_dropped_padding_is_still_reported_as_a_layout_difference(self):
        # A client that rebuilt the view contiguous would hold the same numbers
        # at a stride the MoE kernel does not expect. The digest cannot see
        # that, so stride and storage_nbytes are what must catch it.
        problems = compare_manifests(
            reference=collect_weight_manifest(_Model(pad_elems=self.PAD)),
            actual=collect_weight_manifest(_Model(pad_elems=0)),
        )
        self.assertTrue(any("differ in stride" in p for p in problems), problems)
        self.assertTrue(
            any("differ in storage_nbytes" in p for p in problems), problems
        )
        self.assertFalse(any("differ in sha256" in p for p in problems), problems)

    def test_a_view_rebased_to_the_slab_start_is_reported(self):
        # The as_strided bug this gate exists for: right shape, right stride,
        # wrong storage_offset, so every element but the first block is garbage.
        reference = collect_weight_manifest(_Model(pad_elems=self.PAD))
        model = _Model(pad_elems=self.PAD)
        padded = model.w13_weight.data
        rebased = torch.empty(0, dtype=padded.dtype).set_(
            padded.untyped_storage(),
            padded.storage_offset() + K_DIM,
            padded.shape,
            padded.stride(),
        )
        model.w13_weight = torch.nn.Parameter(rebased, requires_grad=False)
        problems = compare_manifests(
            reference=reference, actual=collect_weight_manifest(model)
        )
        self.assertTrue(
            any("differ in storage_offset" in p for p in problems), problems
        )
        self.assertTrue(any("differ in sha256" in p for p in problems), problems)


class TestParityManifestCatchesEveryDifference(CustomTestCase):
    def _problems(self, actual_model, reference_model=None):
        return compare_manifests(
            reference=collect_weight_manifest(reference_model or _Model()),
            actual=collect_weight_manifest(actual_model),
        )

    def test_wrong_bytes_at_the_right_layout_are_reported(self):
        wrong = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4) + 1
        problems = self._problems(_Model(dense_values=wrong))
        self.assertTrue(any("differ in sha256" in p for p in problems), problems)

    def test_wrong_dtype_is_reported(self):
        model = _Model()
        model.proj = torch.nn.Parameter(
            model.proj.data.to(torch.float16), requires_grad=False
        )
        model.lm_head = model.proj
        problems = self._problems(model)
        self.assertTrue(any("differ in dtype" in p for p in problems), problems)

    def test_wrong_shape_is_reported(self):
        model = _Model()
        model.proj = torch.nn.Parameter(
            model.proj.data.reshape(4, 3), requires_grad=False
        )
        model.lm_head = model.proj
        problems = self._problems(model)
        self.assertTrue(any("differ in shape" in p for p in problems), problems)

    def test_broken_tying_is_reported_even_when_the_bytes_match(self):
        # Untied weights hold equal values, so only the storage grouping can
        # tell that one storage became two.
        model = _Model()
        model.lm_head = torch.nn.Parameter(model.proj.data.clone(), requires_grad=False)
        problems = self._problems(model)
        self.assertTrue(any("differ in storage_group" in p for p in problems), problems)

    def test_a_missing_tensor_is_reported(self):
        model = _Model()
        del model.cos_sin_cache
        problems = self._problems(model)
        self.assertTrue(any("are absent from" in p for p in problems), problems)

    def test_an_extra_tensor_is_reported(self):
        model = _Model()
        model.register_buffer("surprise", torch.zeros(2))
        problems = self._problems(model)
        self.assertTrue(any("are absent from" in p for p in problems), problems)

    def test_a_parameter_demoted_to_a_buffer_is_reported(self):
        model = _Model()
        values = model.proj.data
        del model.proj
        model.register_buffer("proj", values)
        problems = self._problems(model)
        self.assertTrue(any("differ in is_param" in p for p in problems), problems)

    def test_labels_name_which_side_each_value_came_from(self):
        wrong = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4) + 1
        problems = compare_manifests(
            reference=collect_weight_manifest(_Model()),
            actual=collect_weight_manifest(_Model(dense_values=wrong)),
            reference_label="disk",
            actual_label="daemon",
        )
        joined = "\n".join(problems)
        self.assertIn("disk=", joined)
        self.assertIn("daemon=", joined)


if __name__ == "__main__":
    unittest.main()
