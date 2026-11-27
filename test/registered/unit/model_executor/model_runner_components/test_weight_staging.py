"""A use_staging weight-update session: peers write the staging region while the runner keeps its live weights, and
only end_weight_update installs them, at the live weights' unchanged addresses."""

import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.model_executor.model_runner_components.weight_updater as weight_updater_mod
from sglang.srt.distributed.parallel_state import get_parallel
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.managers.io_struct import (
    BeginWeightUpdateReqInput,
    EndWeightUpdateReqInput,
)
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.model_runner_components.weight_staging import (
    WeightStaging,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trained = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3))
        # nothing a trainer sends maps onto it, so peers never write its slot
        self.untrained = torch.nn.Parameter(
            torch.tensor([7, 8, 9], dtype=torch.int16), requires_grad=False
        )
        # a second parameter over the same storage, as tied embeddings are
        self.tied = torch.nn.Parameter(self.trained.data)

    def load_weights(self, named_tensors):
        """As the models' loaders do: copy each tensor into the parameter of its name."""
        params = dict(self.named_parameters())
        for name, tensor in named_tensors:
            params[name].data.copy_(tensor)


def _runner(model, staging):
    model_runner = SimpleNamespace(
        remote_instance_weight_transporter=SimpleNamespace(
            weight_info=None if staging is None else staging.slot_info,
            staging=staging,
        )
    )
    weight_updater = WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=None,
        custom_weight_loaders={},
        get_model=lambda: model,
        update_model_fields=None,
        recapture_cuda_graph=None,
        get_model_runner=lambda: model_runner,
    )
    return SimpleNamespace(weight_updater=weight_updater)


def _peer_write(staging, name, tensor):
    """What a transport does: bytes land at the published address, with no tensor of ours involved."""
    address, numel, element_size = staging.slot_info[name]
    assert tensor.numel() * tensor.element_size() == numel * element_size
    ctypes.memmove(address, tensor.contiguous().data_ptr(), numel * element_size)


class _StagingTestBase(CustomTestCase):
    def setUp(self):
        loader = weight_updater_mod.DefaultModelLoader
        for target, attribute, replacement in (
            (loader, "restore_weights_before_loading", lambda model, device: None),
            (loader, "postprocess_weights", lambda model, device: None),
            (torch.distributed, "barrier", lambda group=None: None),
            (
                torch,
                "get_device_module",
                lambda device=None: SimpleNamespace(
                    synchronize=lambda: None, current_device=lambda: "cpu"
                ),
            ),
            # the engine-wide checks a tensor load runs first read the published server config
            (
                weight_updater_mod,
                "_unsupported_derived_weight_cache_error",
                lambda model: None,
            ),
            (WeightUpdater, "_assert_weight_cache_inactive", lambda self, op: None),
            (weight_updater_mod, "monkey_patch_torch_reductions", lambda: None),
        ):
            patcher = patch.object(target, attribute, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        # a tensor load picks this rank's entry of the payload
        rank = get_parallel().override(tp_rank=0)
        rank.__enter__()
        self.addCleanup(rank.__exit__, None, None, None)
        self.model = _Model()
        self.staging = WeightStaging.allocate(self.model)
        self.recorded_versions = []

    def _manager(self, runner):
        with get_parallel().override(tp_group=SimpleNamespace(cpu_group=None)):
            return SchedulerWeightUpdaterManager(
                tp_worker=SimpleNamespace(
                    weight_update_runners=lambda: [("target", runner)]
                ),
                draft_worker=None,
                memory_saver_adapter=None,
                flush_cache=lambda **kwargs: True,
                is_fully_idle=lambda **kwargs: True,
                scheduler=SimpleNamespace(
                    record_weight_version_change=self._record_version
                ),
            )

    def _record_version(self, new_version):
        # the version must change with the weights, not before them
        self.recorded_versions.append(
            (new_version, self.model.trained.detach().clone())
        )


class TestWeightStaging(_StagingTestBase):
    def test_slots_are_disjoint_ranges_of_one_buffer_and_tied_parameters_share_one(
        self,
    ):
        start, end = (
            self.staging.buffer.data_ptr(),
            self.staging.buffer.data_ptr() + self.staging.buffer.nbytes,
        )
        ranges = {
            name: (address, address + numel * element_size)
            for name, (address, numel, element_size) in self.staging.slot_info.items()
        }
        self.assertEqual(ranges.pop("tied"), ranges["trained"])
        for low, high in ranges.values():
            self.assertTrue(start <= low < high <= end)
        self.assertTrue(
            ranges["trained"][1] <= ranges["untrained"][0]
            or ranges["untrained"][1] <= ranges["trained"][0]
        )

    def test_the_live_weights_change_only_at_end_and_keep_their_addresses(self):
        manager = self._manager(_runner(self.model, self.staging))
        addresses = {
            name: param.data_ptr() for name, param in self.model.named_parameters()
        }
        new = torch.full((2, 3), 42.0)

        self.assertTrue(
            manager.begin_weight_update(
                BeginWeightUpdateReqInput(use_staging=True)
            ).success
        )
        _peer_write(self.staging, "trained", new)
        self.assertTrue(
            torch.equal(self.model.trained, torch.arange(6.0).reshape(2, 3))
        )
        self.assertEqual(self.recorded_versions, [])

        self.assertTrue(
            manager.end_weight_update(
                EndWeightUpdateReqInput(weight_version="5")
            ).success
        )
        self.assertTrue(torch.equal(self.model.trained, new))
        # the slot of a parameter no peer wrote held its live bytes, not uninitialized memory
        self.assertEqual(self.model.untrained.tolist(), [7, 8, 9])
        self.assertEqual(
            addresses,
            {name: param.data_ptr() for name, param in self.model.named_parameters()},
        )
        self.assertEqual([version for version, _ in self.recorded_versions], ["5"])
        self.assertTrue(torch.equal(self.recorded_versions[0][1], new))

    def test_an_aborted_session_leaves_the_live_weights_and_is_not_installed_later(
        self,
    ):
        manager = self._manager(_runner(self.model, self.staging))
        old = self.model.trained.detach().clone()

        manager.begin_weight_update(BeginWeightUpdateReqInput(use_staging=True))
        _peer_write(self.staging, "trained", torch.full((2, 3), -1.0))
        self.assertTrue(
            manager.end_weight_update(
                EndWeightUpdateReqInput(abort=True, weight_version="5")
            ).success
        )
        self.assertTrue(torch.equal(self.model.trained, old))
        self.assertEqual(self.recorded_versions, [])

        # a later session whose peers write nothing must not install what the aborted one left in the slot
        manager.begin_weight_update(BeginWeightUpdateReqInput(use_staging=True))
        manager.end_weight_update(EndWeightUpdateReqInput(weight_version="6"))
        self.assertTrue(torch.equal(self.model.trained, old))

    def test_a_plain_session_whose_peers_wrote_the_staging_region_fails_instead_of_doing_nothing(
        self,
    ):
        """The trainer believes it wrote the live weights; reporting success would leave the engine on old ones."""
        manager = self._manager(_runner(self.model, self.staging))

        manager.begin_weight_update(BeginWeightUpdateReqInput())
        output = manager.end_weight_update(EndWeightUpdateReqInput(weight_version="5"))

        self.assertFalse(output.success)
        self.assertIn("use_staging=True", output.message)
        self.assertEqual(self.recorded_versions, [])

    def test_an_in_place_load_between_staged_updates_is_not_undone_by_the_next_one(
        self,
    ):
        """The slots mirrored the weights before the in-place load; a staged update that writes only some
        parameters must not install those old bytes over the others."""
        manager = self._manager(_runner(self.model, self.staging))
        manager.begin_weight_update(BeginWeightUpdateReqInput(use_staging=True))
        manager.end_weight_update(EndWeightUpdateReqInput())

        in_place = torch.full((2, 3), 3.0)
        manager.begin_weight_update(BeginWeightUpdateReqInput())
        manager.update_weights_from_tensor(
            self._tensor_request(manager, [("trained", in_place)])
        )
        self.assertTrue(manager.end_weight_update(EndWeightUpdateReqInput()).success)
        self.assertTrue(torch.equal(self.model.trained, in_place))

        manager.begin_weight_update(BeginWeightUpdateReqInput(use_staging=True))
        _peer_write(
            self.staging, "untrained", torch.tensor([1, 2, 3], dtype=torch.int16)
        )
        manager.end_weight_update(EndWeightUpdateReqInput())

        self.assertEqual(self.model.untrained.tolist(), [1, 2, 3])
        self.assertTrue(torch.equal(self.model.trained, in_place))

    def test_every_other_in_place_change_of_the_live_weights_makes_the_next_update_mirror_them_again(
        self,
    ):
        """Disk and IPC loads and the weight checker's reset write the live weights without a session."""
        runner = _runner(self.model, self.staging)
        updater = runner.weight_updater
        refused = patch.object(
            weight_updater_mod,
            "_unsupported_derived_weight_cache_error",
            lambda model: "stop here: the load itself is not under test",
        )
        checker = SimpleNamespace(handle=lambda **kwargs: None)
        model_runner = SimpleNamespace(weight_updater=updater, _weight_checker=checker)
        for change in (
            lambda: updater.update_weights_from_disk("path", "auto"),
            lambda: updater.update_weights_from_ipc(None),
            lambda: ModelRunner.check_weights(
                model_runner, action="reset_tensors", role="target"
            ),
        ):
            self.staging.mirror_live()
            with refused:
                change()
            self.assertFalse(self.staging._mirrors_live)
        ModelRunner.check_weights(model_runner, action="compare", role="target")
        self.staging.mirror_live()
        ModelRunner.check_weights(model_runner, action="compare", role="target")
        self.assertTrue(self.staging._mirrors_live)

    def test_the_install_runs_between_the_restore_and_the_post_load_processing(self):
        """Restore the load layout, copy, then recompute what derives from the weights; an abort runs none."""
        calls = []
        new = torch.full((2, 3), 5.0)
        seen = lambda step: (
            lambda *args: calls.append(  # noqa: E731
                (step, torch.equal(self.model.trained, new))
            )
        )
        loader = weight_updater_mod.DefaultModelLoader
        for target, attribute, step in (
            (loader, "restore_weights_before_loading", "restore"),
            (weight_updater_mod, "post_load_weights", "post_load"),
            (loader, "postprocess_weights", "postprocess"),
        ):
            patcher = patch.object(target, attribute, seen(step))
            patcher.start()
            self.addCleanup(patcher.stop)
        manager = self._manager(_runner(self.model, self.staging))

        manager.begin_weight_update(BeginWeightUpdateReqInput(use_staging=True))
        manager.end_weight_update(EndWeightUpdateReqInput(abort=True))
        self.assertEqual(calls, [])

        manager.begin_weight_update(BeginWeightUpdateReqInput(use_staging=True))
        _peer_write(self.staging, "trained", new)
        manager.end_weight_update(EndWeightUpdateReqInput())
        self.assertEqual(
            calls, [("restore", False), ("post_load", True), ("postprocess", True)]
        )

    def test_parameters_that_overlap_without_sharing_their_whole_storage_get_no_staging_region(
        self,
    ):
        model = torch.nn.Module()
        model.whole = torch.nn.Parameter(torch.arange(8.0))
        model.part = torch.nn.Parameter(model.whole.data[4:])
        with self.assertRaisesRegex(AssertionError, "overlap in memory"):
            WeightStaging.allocate(model)

    def test_a_model_whose_backend_reshapes_weights_for_a_load_gets_no_staging_region(
        self,
    ):
        model = _Model()
        model.quant_method = SimpleNamespace(use_flashinfer_trtllm_moe=True)
        with self.assertRaisesRegex(AssertionError, "flashinfer TRT-LLM"):
            WeightStaging.allocate(model)

    def test_begin_is_refused_without_a_staging_region(self):
        manager = self._manager(_runner(self.model, None))

        output = manager.begin_weight_update(
            BeginWeightUpdateReqInput(use_staging=True)
        )

        self.assertFalse(output.success)
        self.assertIn("--weight-update-staging", output.message)
        self.assertIsNone(manager._session)

    def test_a_live_parameter_that_moved_fails_the_session(self):
        """The commit copies into the views taken at allocation; a rebound parameter would no longer be updated."""
        runner = _runner(self.model, self.staging)
        self.model.untrained.data = torch.zeros(3, dtype=torch.int16)

        with self.assertRaisesRegex(AssertionError, "moved published parameters"):
            runner.weight_updater.begin_weight_update(use_staging=True)

    def _tensor_request(self, manager, named_tensors):
        manager.tp_worker.deserialize_own_rank = lambda payloads: named_tensors
        return SimpleNamespace(
            serialized_named_tensors=[b""],
            load_format=None,
            selector="all",
            weight_version=None,
            flush_cache=False,
            torch_empty_cache=False,
        )

    def test_a_tensor_load_fills_the_staging_region_and_is_installed_at_end(self):
        """update_weights_from_tensor runs the model's own loader; during the session it must not reach the weights
        the engine is generating with."""
        manager = self._manager(_runner(self.model, self.staging))
        old, new = self.model.trained.detach().clone(), torch.full((2, 3), 42.0)
        live_storage = self.model.trained.data_ptr()

        manager.begin_weight_update(BeginWeightUpdateReqInput(use_staging=True))
        output = manager.update_weights_from_tensor(
            self._tensor_request(manager, [("trained", new)])
        )

        self.assertTrue(output.success, output.message)
        self.assertTrue(torch.equal(self.model.trained, old))
        self.assertEqual(self.model.trained.data_ptr(), live_storage)

        manager.end_weight_update(EndWeightUpdateReqInput(weight_version="5"))
        self.assertTrue(torch.equal(self.model.trained, new))
        self.assertEqual(self.model.trained.data_ptr(), live_storage)
        self.assertEqual(self.model.untrained.tolist(), [7, 8, 9])

    def test_a_tensor_load_by_the_name_of_a_parameter_that_shares_storage_is_staged_too(
        self,
    ):
        """`tied` is a second Parameter over the storage of `trained`. Redirecting only one of the two would let a
        load by the other name write the live weights at once, and the install would then put the old bytes back."""
        manager = self._manager(_runner(self.model, self.staging))
        old, new = self.model.trained.detach().clone(), torch.full((2, 3), 42.0)

        manager.begin_weight_update(BeginWeightUpdateReqInput(use_staging=True))
        output = manager.update_weights_from_tensor(
            self._tensor_request(manager, [("tied", new)])
        )

        self.assertTrue(output.success, output.message)
        self.assertTrue(torch.equal(self.model.trained, old))
        manager.end_weight_update(EndWeightUpdateReqInput())
        self.assertTrue(torch.equal(self.model.trained, new))
        self.assertTrue(torch.equal(self.model.tied, new))

    def test_a_tensor_load_is_refused_for_a_model_whose_loader_also_derives_state(self):
        """Such a loader would change what the engine generates with now, while the weights are still staged."""
        derived = []
        self.model.post_load_weights = lambda: derived.append("recomputed")
        manager = self._manager(_runner(self.model, self.staging))
        old = self.model.trained.detach().clone()
        manager.begin_weight_update(BeginWeightUpdateReqInput(use_staging=True))

        output = manager.update_weights_from_tensor(
            self._tensor_request(manager, [("trained", torch.zeros(2, 3))])
        )

        self.assertFalse(output.success)
        self.assertIn("recomputes state derived from its weights", output.message)
        self.assertEqual(derived, [])
        self.assertTrue(torch.equal(self.model.trained, old))

    def test_a_tensor_load_is_refused_for_a_model_with_a_gemma_norm(self):
        self.model.norm = GemmaRMSNorm(4)
        manager = self._manager(_runner(self.model, WeightStaging.allocate(self.model)))
        manager.begin_weight_update(BeginWeightUpdateReqInput(use_staging=True))

        output = manager.update_weights_from_tensor(
            self._tensor_request(manager, [("trained", torch.zeros(2, 3))])
        )

        self.assertFalse(output.success)
        self.assertIn("recomputes state derived from its weights", output.message)

    def test_a_buffer_the_loader_changes_is_put_back_and_fails_the_load(self):
        """The staging region holds parameters only: a buffer written by the loader would be live at once."""
        model = self.model
        model.register_buffer("scale", torch.tensor([1.0]))
        model.register_buffer("cache", torch.tensor([1.0]), persistent=False)

        def load_weights(named_tensors):
            state = {**dict(model.named_parameters()), **dict(model.named_buffers())}
            for name, tensor in named_tensors:
                state[name].data.copy_(tensor)

        model.load_weights = load_weights
        manager = self._manager(_runner(model, self.staging))
        manager.begin_weight_update(BeginWeightUpdateReqInput(use_staging=True))

        same = manager.update_weights_from_tensor(
            self._tensor_request(manager, [("scale", torch.tensor([1.0]))])
        )
        self.assertTrue(same.success, same.message)

        changed = manager.update_weights_from_tensor(
            self._tensor_request(manager, [("scale", torch.tensor([3.0]))])
        )
        self.assertFalse(changed.success)
        self.assertIn("['scale']", changed.message)
        self.assertEqual(model.scale.tolist(), [1.0])

        # a loader takes the buffers a checkpoint does not carry as well
        changed = manager.update_weights_from_tensor(
            self._tensor_request(manager, [("cache", torch.tensor([3.0]))])
        )
        self.assertFalse(changed.success)
        self.assertEqual(model.cache.tolist(), [1.0])

    def test_in_place_loads_are_refused_while_a_staging_session_is_open(self):
        """The install at the end of the session would overwrite what they loaded."""
        runner = _runner(self.model, self.staging)
        manager = self._manager(runner)
        manager.begin_weight_update(BeginWeightUpdateReqInput(use_staging=True))

        for load in (manager.update_weights_from_disk, manager.update_weights_from_ipc):
            output = load(SimpleNamespace())
            self.assertFalse(output.success)
            self.assertIn("open use_staging session will overwrite", output.message)

    def test_a_load_for_other_runners_than_the_session_s_is_refused(self):
        manager = self._manager(_runner(self.model, self.staging))
        manager.begin_weight_update(
            BeginWeightUpdateReqInput(use_staging=True, selector="target")
        )
        request = self._tensor_request(manager, [("trained", torch.zeros(2, 3))])

        with self.assertRaisesRegex(AssertionError, "got a load for 'all'"):
            manager.update_weights_from_tensor(request)

    def test_a_loader_that_replaces_a_parameter_cannot_fill_the_staging_region(self):
        """Its bytes would be in a tensor that is neither live nor staged, and the install would miss them."""
        manager = self._manager(_runner(self.model, self.staging))
        old = self.model.trained.detach().clone()
        self.model.load_weights = lambda named_tensors: setattr(
            self.model.trained, "data", named_tensors[0][1]
        )
        manager.begin_weight_update(BeginWeightUpdateReqInput(use_staging=True))

        with self.assertRaisesRegex(AssertionError, "rebound 1 parameters"):
            manager.update_weights_from_tensor(
                self._tensor_request(manager, [("trained", torch.zeros(2, 3))])
            )

        # the live parameter is back on its own storage, untouched
        self.assertTrue(torch.equal(self.model.trained, old))
        self.assertEqual(
            self.model.trained.data_ptr(), self.staging.live_info["trained"][0]
        )


if __name__ == "__main__":
    unittest.main()
