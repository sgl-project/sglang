"""Native weight updates preserve rank selection and receive complete manifests."""

import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
    _WeightUpdateSession,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    LocalSerializedTensor,
    WeightUpdater,
)
from sglang.srt.models.gemma import GemmaForCausalLM
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import MultiprocessingSerializer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class TestTensorUpdateRank(unittest.TestCase):
    def test_a_narrowed_updater_reads_the_deployment_rank_entry(self):
        model = torch.nn.Linear(2, 2, bias=False)
        # An attention-owning draft records its attention-TP rank at init.
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
        )
        per_rank = [torch.full((2, 2), float(rank)) for rank in range(2)]
        payload = LocalSerializedTensor(
            values=[MultiprocessingSerializer.serialize(t) for t in per_rank]
        )
        with (
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
            get_parallel().override(tp_rank=1),
        ):
            success, message = updater.update_weights_from_tensor(
                [("weight", payload)], load_format="direct"
            )
        self.assertTrue(success, message)
        self.assertTrue(torch.equal(model.weight.detach(), per_rank[1]))


def _cpu_runner(model):
    return SimpleNamespace(
        weight_updater=WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
        )
    )


class _GemmaNorm(torch.nn.Module):
    load_weights = GemmaForCausalLM.load_weights

    def __init__(self):
        super().__init__()
        self.norm = torch.nn.LayerNorm(2, bias=False)


class TestTensorUpdateRebuildsEachIpcTensorOnce(CustomTestCase):
    def setUp(self):
        for patcher in (
            patch("torch.distributed.barrier"),
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def _update(
        self,
        named_tensors,
        target,
        draft,
        *,
        selector="all",
        load_format="direct",
        flush_cache=None,
    ):
        """``draft=None`` is a PP stage that does not host the draft runner."""
        with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
            manager = SchedulerWeightUpdaterManager(
                tp_worker=SimpleNamespace(
                    model_runner=SimpleNamespace(device="cpu"),
                    weight_update_runners=lambda: [("target", target)],
                    deserialize_own_rank=lambda payloads: named_tensors,
                ),
                draft_worker=SimpleNamespace(
                    weight_update_runners=lambda: (
                        [] if draft is None else [("draft", draft)]
                    )
                ),
                memory_saver_adapter=None,
                flush_cache=flush_cache or (lambda **kwargs: True),
                is_fully_idle=lambda **kwargs: True,
                scheduler=SimpleNamespace(
                    record_weight_version_change=lambda new_version: None,
                    wait_for_launched_forward=lambda: None,
                ),
            )
        manager._session = _WeightUpdateSession(selector=selector)
        request = SimpleNamespace(
            serialized_named_tensors=[b""],
            load_format=load_format,
            selector=selector,
            weight_version=None,
            flush_cache=flush_cache is not None,
            torch_empty_cache=flush_cache is not None,
        )
        with (
            get_parallel().override(tp_rank=1),
            patch(
                "sglang.srt.managers.scheduler_components.weight_updater.control_request_barrier"
            ),
        ):
            return manager.update_weights_from_tensor(request)

    def test_target_and_draft_share_one_rebuild_of_each_ipc_tensor(self):
        target_model = torch.nn.Linear(2, 2, bias=False)
        draft_model = torch.nn.Linear(2, 2, bias=False)
        per_rank = [torch.full((2, 2), float(rank)) for rank in range(2)]
        payload = LocalSerializedTensor(values=[b"rank0", b"rank1"])
        with patch.object(
            LocalSerializedTensor,
            "get",
            autospec=True,
            side_effect=lambda self, rank: per_rank[rank].clone(),
        ) as get:
            output = self._update(
                [("weight", payload)],
                _cpu_runner(target_model),
                _cpu_runner(draft_model),
            )
        self.assertTrue(output.success, output.message)
        get.assert_called_once_with(payload, 1)
        self.assertTrue(torch.equal(target_model.weight.detach(), per_rank[1]))
        self.assertTrue(torch.equal(draft_model.weight.detach(), per_rank[1]))

    def test_a_stage_hosting_no_selected_runner_opens_no_ipc_tensor(self):
        """A draft-only update reaches PP stages whose GPUs cannot open the handles."""
        payload = LocalSerializedTensor(values=[b"rank0", b"rank1"])
        with patch.object(LocalSerializedTensor, "get", autospec=True) as get:
            output = self._update(
                [("weight", payload)],
                _cpu_runner(torch.nn.Linear(2, 2, bias=False)),
                None,
                selector="draft",
            )
        self.assertTrue(output.success, output.message)
        get.assert_not_called()

    def test_rebuilt_tensors_are_released_before_the_cache_flush(self):
        """torch_empty_cache returns no block a rebuilt tensor still occupies."""
        rebuilt = []
        alive_at_flush = []

        def get(self, rank):
            tensor = torch.zeros(2, 2)
            rebuilt.append(weakref.ref(tensor))
            return tensor

        def flush_cache(**kwargs):
            alive_at_flush.extend(ref() is not None for ref in rebuilt)
            return True

        payload = LocalSerializedTensor(values=[b"rank0", b"rank1"])
        with patch.object(LocalSerializedTensor, "get", autospec=True, side_effect=get):
            output = self._update(
                [("weight", payload)],
                _cpu_runner(torch.nn.Linear(2, 2, bias=False)),
                _cpu_runner(torch.nn.Linear(2, 2, bias=False)),
                flush_cache=flush_cache,
            )
        self.assertTrue(output.success, output.message)
        self.assertEqual(alive_at_flush, [False])

    def test_a_loader_that_adjusts_a_tensor_leaves_it_intact_for_the_next_runner(self):
        """Gemma serves 1 + the checkpoint norm weight; the draft must not get 2 +."""
        target_model, draft_model = _GemmaNorm(), _GemmaNorm()
        payload = LocalSerializedTensor(values=[b"rank0", b"rank1"])
        with patch.object(
            LocalSerializedTensor,
            "get",
            autospec=True,
            side_effect=lambda self, rank: torch.zeros(2),
        ):
            output = self._update(
                [("norm.weight", payload)],
                _cpu_runner(target_model),
                _cpu_runner(draft_model),
                load_format=None,
            )
        self.assertTrue(output.success, output.message)
        for model in (target_model, draft_model):
            self.assertTrue(torch.equal(model.norm.weight.detach(), torch.ones(2)))


class TestDistributedWeightManifest(CustomTestCase):
    def setUp(self):
        self.group = object()
        self.updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: None,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
            _model_update_group={"test": self.group},
        )

    def _receive(self, **changes):
        args = dict(
            names=["w1", "w2"],
            dtypes=["float32", "float32"],
            shapes=[[2], [2]],
            group_name="test",
        )
        args.update(changes)
        return self.updater.receive_weights_from_distributed(**args)

    def test_unequal_metadata_is_rejected_before_receiving_a_prefix(self):
        """A truncated zip reported a successful receive with old weights omitted."""
        cases = (
            {"names": ["w1"]},
            {"names": ["w1", "w2", "w3"]},
            {"names": []},
            {"dtypes": ["float32"]},
            {"dtypes": ["float32"] * 3},
            {"dtypes": []},
            {"shapes": [[2]]},
            {"shapes": [[2]] * 3},
            {"shapes": []},
        )
        for load_format in (None, "flattened_bucket"):
            for changes in cases:
                with (
                    self.subTest(load_format=load_format, changes=changes),
                    patch("torch.distributed.broadcast") as broadcast,
                    patch("torch.empty", wraps=torch.empty) as allocate,
                ):
                    with self.assertRaises(ValueError):
                        self._receive(load_format=load_format, **changes)
                    broadcast.assert_not_called()
                    allocate.assert_not_called()

    def test_invalid_later_entry_does_not_start_earlier_broadcasts(self):
        """An invalid second dtype/shape used to leave the first receive in flight."""
        cases = (
            {"dtypes": ["float32", "not_a_torch_dtype"]},
            {"dtypes": ["float32", "empty"]},
            {"dtypes": ["float32", None]},
            {"shapes": [[2], [-1]]},
            {"shapes": [[2], [1.5]]},
        )
        for changes in cases:
            with (
                self.subTest(changes=changes),
                patch("torch.distributed.broadcast") as broadcast,
            ):
                with self.assertRaises(
                    (AttributeError, TypeError, ValueError, RuntimeError)
                ):
                    self._receive(**changes)
                broadcast.assert_not_called()

    def test_later_allocation_failure_does_not_leave_an_earlier_receive_in_flight(self):
        """Preparing a later buffer must finish before any async receive is posted."""
        first = torch.empty(2)
        with (
            patch("torch.empty", side_effect=[first, RuntimeError("out of memory")]),
            patch("torch.distributed.broadcast") as broadcast,
        ):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                self._receive()
            broadcast.assert_not_called()

    def test_valid_scalar_empty_and_mixed_dtype_tensors_keep_their_wire_order(self):
        """Preflight must preserve both native wire formats and async completion."""
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        names = ["scalar", "empty", "vector"]
        payloads = [
            torch.tensor(7.25, dtype=torch.float64),
            torch.empty((0, 3), dtype=torch.float32),
            torch.tensor([9, 10], dtype=torch.int64),
        ]
        bucket = FlattenedTensorBucket(list(zip(names, payloads, strict=True)))
        for load_format in (None, "flattened_bucket"):
            sent = []

            def broadcast(tensor, *, src, group, async_op=False):
                self.assertEqual(src, 0)
                self.assertIs(group, self.group)
                payload = (
                    payloads[len(sent)]
                    if load_format is None
                    else bucket.get_flattened_tensor()
                )
                sent.append(tensor)
                if async_op:
                    return SimpleNamespace(wait=lambda: tensor.copy_(payload))
                tensor.copy_(payload)

            with (
                self.subTest(load_format=load_format),
                patch("torch.distributed.broadcast", side_effect=broadcast),
            ):
                received = self._receive(
                    names=names,
                    dtypes=["double", torch.float32, torch.int64],
                    shapes=[[], [0, 3], torch.Size([2])],
                    load_format=load_format,
                )
                self.assertEqual([name for name, _ in received], names)
                for (_, tensor), expected in zip(received, payloads, strict=True):
                    torch.testing.assert_close(tensor, expected)

    def test_empty_normal_manifest_remains_a_no_op(self):
        with patch("torch.distributed.broadcast") as broadcast:
            self.assertEqual(self._receive(names=[], dtypes=[], shapes=[]), [])
            broadcast.assert_not_called()


if __name__ == "__main__":
    unittest.main()
