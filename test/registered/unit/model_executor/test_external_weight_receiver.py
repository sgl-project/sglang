import sys
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.managers.io_struct import (
    InitWeightsUpdateGroupReqInput,
    UpdateWeightsFromDistributedReqInput,
)
from sglang.srt.model_executor.model_runner_components import (
    weight_updater as weight_updater_module,
)
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.weight_sync.external_receiver import build_weight_update_receiver
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

MODULE = "fake_external_receiver_pkg"
PATH = f"{MODULE}.create_receiver"


class FakeReceiver:
    def __init__(self, context):
        self.context = context
        self.payloads = []
        self.destroyed = 0

    def receive(self, payload):
        self.payloads.append(payload)

    def destroy(self):
        self.destroyed += 1


def make_updater(allowed, model=None):
    model = model if model is not None else torch.nn.Linear(2, 2, bias=False)
    return WeightUpdater(
        tp_rank=1,
        device="cpu",
        gpu_id=0,
        model_config=None,
        custom_weight_loaders={},
        weight_update_receivers=allowed,
        get_model=lambda: model,
        update_model_fields=lambda *a, **k: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: None,
    )


def init_group(updater, group_name="g", **kwargs):
    return updater.init_weights_update_group(
        "10.0.0.1", 9999, 4, 8, group_name, **kwargs
    )


class TestExternalWeightReceiver(unittest.TestCase):
    def setUp(self):
        self.factory_calls = []
        self.factory_result = None
        self.factory_error = None

        def create_receiver(context):
            self.factory_calls.append(context)
            if self.factory_error is not None:
                raise self.factory_error
            self.factory_result = FakeReceiver(context)
            return self.factory_result

        module = types.ModuleType(MODULE)
        module.create_receiver = create_receiver
        module.not_a_receiver = lambda context: object()
        self.module_patch = patch.dict(sys.modules, {MODULE: module})
        self.module_patch.start()
        for target, value in (
            (
                "torch.distributed.is_initialized",
                lambda: True,
            ),
            (
                f"{weight_updater_module.__name__}.get_parallel",
                lambda: types.SimpleNamespace(tp_size=8),
            ),
            (
                # The write guards read the runtime-context model; tests run
                # with the weight cache off unless they re-patch it.
                f"{weight_updater_module.__name__}.get_model",
                lambda: types.SimpleNamespace(weight_cache_mode="off"),
            ),
        ):
            p = patch(target, value)
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.module_patch.stop)

    def test_not_allowlisted_is_rejected_before_import(self):
        updater = make_updater(allowed=["other.create"])
        with patch(
            "sglang.srt.weight_sync.external_receiver.dynamic_import"
        ) as dynamic_import:
            ok, message = init_group(updater, receiver=PATH)
        self.assertFalse(ok)
        self.assertIn("not listed in --weight-update-receivers", message)
        dynamic_import.assert_not_called()
        self.assertEqual(self.factory_calls, [])
        self.assertEqual(updater._external_receivers, {})

    def test_empty_allowlist_rejects(self):
        for allowed in (None, []):
            ok, _ = init_group(make_updater(allowed), receiver=PATH)
            self.assertFalse(ok)
        self.assertEqual(self.factory_calls, [])

    def test_factory_receives_generic_context(self):
        model = torch.nn.Linear(2, 2, bias=False)
        updater = make_updater([PATH], model)
        with patch.object(
            weight_updater_module, "init_custom_process_group"
        ) as init_pg:
            ok, _ = init_group(
                updater, "grp", receiver=PATH, receiver_init_payload={"plan": [1, 2]}
            )
        self.assertTrue(ok)
        # A receiver group owns no torch process group.
        init_pg.assert_not_called()
        (context,) = self.factory_calls
        self.assertIs(context.model, model)
        self.assertEqual(context.device, torch.device("cpu"))
        self.assertEqual((context.tp_rank, context.tp_size), (1, 8))
        self.assertEqual(context.group_name, "grp")
        self.assertEqual(
            (context.master_address, context.master_port), ("10.0.0.1", 9999)
        )
        self.assertEqual((context.world_size, context.rank_offset), (8, 4))
        self.assertEqual(context.init_payload, {"plan": [1, 2]})
        self.assertIs(updater._external_receivers["grp"], self.factory_result)
        self.assertEqual(updater._model_update_group, {})

    def test_receive_dispatches_payload_and_loads_nothing(self):
        updater = make_updater([PATH])
        init_group(updater, receiver=PATH)
        with patch("torch.distributed.broadcast") as broadcast:
            result = updater.receive_weights_from_distributed(
                names=[],
                dtypes=[],
                shapes=[],
                group_name="g",
                receiver_payload={"version": "7"},
            )
        broadcast.assert_not_called()
        self.assertEqual(result, [])
        self.assertEqual(self.factory_result.payloads, [{"version": "7"}])

    def test_receiver_round_rejected_when_the_weight_cache_is_active(self):
        updater = make_updater([PATH])
        init_group(updater, receiver=PATH)
        with patch(
            f"{weight_updater_module.__name__}.get_model",
            lambda: types.SimpleNamespace(weight_cache_mode="daemon"),
        ):
            with self.assertRaisesRegex(RuntimeError, "weight cache"):
                updater.receive_weights_from_distributed(
                    names=[],
                    dtypes=[],
                    shapes=[],
                    group_name="g",
                    receiver_payload={"version": "7"},
                )
        self.assertEqual(self.factory_result.payloads, [])

    def test_receiver_round_rejected_with_a_derived_weight_cache(self):
        model = torch.nn.Linear(2, 2, bias=False)
        model._derived_weight_cache_error = "derived scales require restart"
        updater = make_updater([PATH], model)
        init_group(updater, receiver=PATH)
        with self.assertRaisesRegex(RuntimeError, "derived scales require restart"):
            updater.receive_weights_from_distributed(
                names=[],
                dtypes=[],
                shapes=[],
                group_name="g",
                receiver_payload={"version": "7"},
            )
        self.assertEqual(self.factory_result.payloads, [])

    def test_named_tensors_on_a_receiver_group_are_rejected(self):
        updater = make_updater([PATH])
        init_group(updater, receiver=PATH)
        with self.assertRaisesRegex(ValueError, "must be empty"):
            updater.receive_weights_from_distributed(
                names=["w"],
                dtypes=["float32"],
                shapes=[(1,)],
                group_name="g",
                receiver_payload={"x": 1},
            )
        self.assertEqual(self.factory_result.payloads, [])

    def test_load_format_on_a_receiver_round_is_rejected(self):
        updater = make_updater([PATH])
        init_group(updater, receiver=PATH)
        with self.assertRaisesRegex(ValueError, "load_format"):
            updater.receive_weights_from_distributed(
                names=[],
                dtypes=[],
                shapes=[],
                group_name="g",
                load_format="flattened_bucket",
                receiver_payload={"version": "7"},
            )
        self.assertEqual(self.factory_result.payloads, [])

    def test_payload_without_receiver_group_is_rejected(self):
        updater = make_updater([PATH])
        updater._model_update_group["plain"] = object()
        with self.assertRaisesRegex(ValueError, "no external receiver"):
            updater.receive_weights_from_distributed(
                names=[],
                dtypes=[],
                shapes=[],
                group_name="plain",
                receiver_payload={"x": 1},
            )

    def test_default_path_unchanged(self):
        updater = make_updater([PATH])
        with patch.object(
            weight_updater_module, "init_custom_process_group", return_value="pg"
        ) as init_pg:
            ok, message = init_group(updater, backend="nccl")
        self.assertTrue(ok)
        self.assertEqual(message, "Succeeded to initialize custom process group.")
        init_pg.assert_called_once_with(
            backend="nccl",
            init_method="tcp://10.0.0.1:9999",
            world_size=8,
            rank=5,
            group_name="g",
        )
        self.assertEqual(updater._model_update_group, {"g": "pg"})
        self.assertEqual(updater._external_receivers, {})
        self.assertEqual(self.factory_calls, [])
        with patch("torch.distributed.destroy_process_group") as destroy:
            self.assertEqual(
                updater.destroy_weights_update_group("g"),
                (True, "Succeeded to destroy custom process group."),
            )
        destroy.assert_called_once_with("pg")

    def test_init_payload_requires_receiver(self):
        updater = make_updater([PATH])
        ok, message = init_group(updater, receiver_init_payload={"a": 1})
        self.assertFalse(ok)
        self.assertIn("requires receiver", message)

    def test_factory_failure_leaves_no_state_and_allows_retry(self):
        updater = make_updater([PATH])
        self.factory_error = RuntimeError("boom")
        ok, message = init_group(updater, receiver=PATH)
        self.assertFalse(ok)
        self.assertIn("boom", message)
        self.assertEqual(updater._external_receivers, {})
        self.assertEqual(updater._model_update_group, {})
        self.factory_error = None
        self.assertTrue(init_group(updater, receiver=PATH)[0])

    def test_malformed_receiver_is_destroyed_and_rejected(self):
        updater = make_updater([f"{MODULE}.not_a_receiver"])
        ok, message = init_group(updater, receiver=f"{MODULE}.not_a_receiver")
        self.assertFalse(ok)
        self.assertIn("receive(payload) and destroy()", message)
        self.assertEqual(updater._external_receivers, {})

    def test_destroy_failure_on_a_malformed_receiver_keeps_the_rejection(self):
        class BrokenDestroy:
            def destroy(self):
                raise OSError("close failed")

        path = f"{MODULE}.broken_destroy"
        sys.modules[MODULE].broken_destroy = lambda context: BrokenDestroy()
        with self.assertRaises(TypeError) as ctx:
            build_weight_update_receiver(path, [path], context=None)
        self.assertIn("receive(payload) and destroy()", str(ctx.exception))
        self.assertIsInstance(ctx.exception.__cause__, OSError)

    def test_duplicate_group_is_rejected_without_calling_factory(self):
        updater = make_updater([PATH])
        self.assertTrue(init_group(updater, receiver=PATH)[0])
        ok, message = init_group(updater, receiver=PATH)
        self.assertFalse(ok)
        self.assertIn("already exists", message)
        self.assertEqual(len(self.factory_calls), 1)

    def test_receiver_cannot_shadow_torch_group(self):
        """A receiver init on a torch-owned name must fail without calling the factory."""
        updater = make_updater([PATH])
        with patch.object(
            weight_updater_module, "init_custom_process_group", return_value="pg"
        ):
            self.assertTrue(init_group(updater)[0])
        ok, message = init_group(updater, receiver=PATH)
        self.assertFalse(ok)
        self.assertIn("already exists", message)
        self.assertEqual(self.factory_calls, [])
        self.assertEqual(updater._external_receivers, {})
        self.assertEqual(updater._model_update_group, {"g": "pg"})

    def test_torch_group_cannot_shadow_receiver(self):
        """A torch group init on a receiver-owned name must fail, not shadow."""
        updater = make_updater([PATH])
        self.assertTrue(init_group(updater, receiver=PATH)[0])
        with patch.object(
            weight_updater_module, "init_custom_process_group"
        ) as init_pg:
            ok, message = init_group(updater)
        self.assertFalse(ok)
        self.assertIn("already exists", message)
        init_pg.assert_not_called()
        self.assertEqual(updater._model_update_group, {})
        self.assertIs(updater._external_receivers["g"], self.factory_result)
        # The receiver can still be destroyed, freeing the name for either path.
        self.assertTrue(updater.destroy_weights_update_group("g")[0])
        with patch.object(
            weight_updater_module, "init_custom_process_group", return_value="pg"
        ):
            self.assertTrue(init_group(updater)[0])
        self.assertEqual(updater._model_update_group, {"g": "pg"})

    def test_destroy_calls_receiver_once_and_forgets_it(self):
        updater = make_updater([PATH])
        init_group(updater, receiver=PATH)
        receiver = self.factory_result
        self.assertTrue(updater.destroy_weights_update_group("g")[0])
        self.assertEqual(receiver.destroyed, 1)
        self.assertFalse(updater.destroy_weights_update_group("g")[0])
        self.assertEqual(receiver.destroyed, 1)

    def test_destroy_failure_is_reported_and_receiver_forgotten(self):
        updater = make_updater([PATH])
        init_group(updater, receiver=PATH)
        self.factory_result.destroy = lambda: (_ for _ in ()).throw(OSError("close"))
        ok, message = updater.destroy_weights_update_group("g")
        self.assertFalse(ok)
        # The receiver path has no process group; the message must say so.
        self.assertIn("external weight-update receiver", message)
        self.assertIn("close", message)
        self.assertEqual(updater._external_receivers, {})

    def test_request_fields_default_to_none(self):
        init = InitWeightsUpdateGroupReqInput(
            master_address="a", master_port=1, rank_offset=0, world_size=2
        )
        self.assertIsNone(init.receiver)
        self.assertIsNone(init.receiver_init_payload)
        update = UpdateWeightsFromDistributedReqInput(names=[], dtypes=[], shapes=[])
        self.assertIsNone(update.receiver_payload)


if __name__ == "__main__":
    unittest.main()
