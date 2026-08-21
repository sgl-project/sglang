import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch.nn as nn

from sglang.multimodal_gen import registry
from sglang.multimodal_gen.runtime.layers import linear as mg_linear
from sglang.multimodal_gen.runtime.models.dits import boogu_image

TP_SIZE = 7
NUM_HEADS = 28
NUM_KV_HEADS = 7
HEAD_DIM = 8
DIM = NUM_HEADS * HEAD_DIM


class _StubAttn(nn.Module):
    """Stands in for USPAttention, which needs global server args + a backend."""

    def __init__(self, *args, **kwargs):
        super().__init__()


def _accepted_input_width(layer: mg_linear.RowParallelLinear) -> int:
    """Last-dim width the layer's forward() actually expects to be handed.

    A RowParallelLinear consumes a shard when input_is_parallel=True and a full
    replicated tensor (which it slices itself) when False.
    """
    if layer.input_is_parallel:
        return layer.input_size_per_partition
    return layer.input_size


def _build_at_tp7():
    """Build the two attention modules as if this rank were 1 of 7."""
    sentinel_group = object()
    with ExitStack() as stack:
        for target, repl in (
            ("get_tp_group", lambda: sentinel_group),
            ("get_group_rank", lambda group: 0),
            ("get_group_size", lambda group: TP_SIZE),
        ):
            stack.enter_context(patch.object(mg_linear, target, repl))
        stack.enter_context(
            patch.object(boogu_image, "get_tp_world_size", lambda: TP_SIZE)
        )
        stack.enter_context(patch.object(boogu_image, "USPAttention", _StubAttn))
        joint = boogu_image.BooguJointAttention(
            dim=DIM, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS
        )
        self_attn = boogu_image.BooguAttention(
            dim=DIM, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS
        )
    return joint, self_attn


class TestBooguAttentionTensorParallelWidths(unittest.TestCase):
    """Producer/consumer width contract for the output projections at tp=7.

    Regression guard: joint attention runs the attention output through
    processor.{instruct,img}_out (RowParallelLinear, so each already all-reduces
    to the full dim) and only then through to_out[0]. to_out[0] was declared
    input_is_parallel=True, i.e. expecting dim/tp features, so at tp=7 it was
    handed 3360 where it expected 480 and the matmul failed. tp=1 cannot catch
    this because dim/1 == dim.
    """

    def test_joint_to_out_accepts_the_width_the_processor_emits(self) -> None:
        joint, _ = _build_at_tp7()
        for producer in (joint.processor.instruct_out, joint.processor.img_out):
            self.assertEqual(producer.output_size, DIM)
            self.assertEqual(
                _accepted_input_width(joint.to_out[0]),
                producer.output_size,
            )

    def test_self_attention_to_out_still_consumes_a_head_shard(self) -> None:
        """The sibling projection must NOT be "fixed" the same way.

        BooguAttention feeds to_out[0] the flattened local heads directly, so
        there its input really is sharded.
        """
        _, self_attn = _build_at_tp7()
        self.assertTrue(self_attn.to_out[0].input_is_parallel)
        self.assertEqual(
            _accepted_input_width(self_attn.to_out[0]),
            self_attn.local_num_heads * self_attn.head_dim,
        )

    def test_output_projection_weights_stay_sharded(self) -> None:
        """Slicing the input locally must not silently replicate the weight."""
        joint, _ = _build_at_tp7()
        self.assertEqual(joint.to_out[0].input_size_per_partition, DIM // TP_SIZE)


class TestBooguImageRegistryDetector(unittest.TestCase):
    """Only the Base T2I checkpoint may claim this pipeline.

    The detector used to be `"boogu-image" in hf_id`, which also swallowed the
    Edit (TI2I), Turbo (4-step distilled, CFG 1.0) and fp8 checkpoints that this
    pipeline does not implement.

    Scope: these cases pin the detector predicate only. `Boogu-Image-0.1-Base-fp8`
    still resolves to this pipeline through the registry's partial basename match
    (`boogu-image-0.1-base` is a substring of it), which happens before detectors
    run and is framework-wide behaviour, not something this entry controls.
    """

    BASE_REPO = "Boogu/Boogu-Image-0.1-Base"

    def _boogu_detectors(self):
        model_id = registry._MODEL_HF_PATH_TO_NAME[self.BASE_REPO]
        detectors = [
            detector
            for registered_id, detector in registry._MODEL_NAME_DETECTORS
            if registered_id == model_id
        ]
        self.assertTrue(detectors, "Boogu-Image registered no detector")
        return detectors

    def test_base_repo_is_registered_under_the_official_org(self) -> None:
        self.assertIn(self.BASE_REPO, registry._MODEL_HF_PATH_TO_NAME)

    def test_detector_matches_base(self) -> None:
        detectors = self._boogu_detectors()
        self.assertTrue(any(detector(self.BASE_REPO) for detector in detectors))

    def test_detector_rejects_unsupported_variants(self) -> None:
        detectors = self._boogu_detectors()
        for hf_id in (
            "Boogu/Boogu-Image-0.1-Edit",
            "Boogu/Boogu-Image-0.1-Edit-Turbo",
            "Boogu/Boogu-Image-0.1-Edit-fp8",
            "Boogu/Boogu-Image-0.1-Turbo",
            "Boogu/Boogu-Image-0.1-Turbo-fp8",
            "Boogu/Boogu-Image-0.1-Base-fp8",
        ):
            with self.subTest(hf_id=hf_id):
                self.assertFalse(any(detector(hf_id) for detector in detectors))


if __name__ == "__main__":
    unittest.main()
