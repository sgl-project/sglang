import io
import pickle
import unittest
from collections import OrderedDict, defaultdict, deque
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.platforms.cpu import CpuSRTPlatform
from sglang.srt.utils import serialization
from sglang.srt.utils.serialization import (
    MultiprocessingSerializer,
    SafeUnpickler,
    safe_pickle_loads,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestSafeUnpickler(CustomTestCase):
    def test_rejects_dangerous_builtin_globals(self):
        for name in ("__import__", "getattr", "eval", "exec", "compile", "open"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    RuntimeError, rf"Blocked unsafe global \(builtins\.{name}\)"
                ),
            ):
                # GLOBAL resolves the callable but does not invoke it. This exercises
                # the deserialization boundary without constructing an exploit chain.
                safe_pickle_loads(f"cbuiltins\n{name}\n.".encode())

    def test_rejects_unlisted_standard_library_globals(self):
        for module, name in (
            ("copyreg", "_reconstructor"),
            ("operator", "attrgetter"),
            ("types", "FunctionType"),
        ):
            with (
                self.subTest(module=module, name=name),
                self.assertRaisesRegex(
                    RuntimeError, rf"Blocked unsafe global \({module}\.{name}\)"
                ),
            ):
                safe_pickle_loads(f"c{module}\n{name}\n.".encode())

    def test_round_trips_safe_standard_library_types(self):
        value = SimpleNamespace(
            values=OrderedDict([("items", deque([1, 2]))]),
            factory=defaultdict(list, {"items": [3]}),
            index=slice(1, 4),
            parser=partial(int, base=10),
        )

        restored = safe_pickle_loads(
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        )

        self.assertEqual(restored.values, value.values)
        self.assertEqual(restored.factory, value.factory)
        self.assertEqual(restored.index, value.index)
        self.assertEqual(restored.parser("11"), 11)

    def test_allowlist_covers_torch_dtypes_and_tensor_classes(self):
        # A pickled tensor carries its dtype and tensor class as GLOBAL
        # opcodes, so every dtype the installed torch exposes must be listed
        # or a payload using it is refused.
        torch_globals = {
            ("torch", name)
            for name in dir(torch)
            if isinstance(getattr(torch, name), torch.dtype)
        }
        torch_globals.update(
            (cls.__module__, cls.__name__)
            for cls in (*torch._storage_classes, *torch._tensor_classes)
        )

        missing = sorted(torch_globals - SafeUnpickler.ALLOWED_GLOBALS)

        self.assertEqual(
            missing,
            [],
            f"torch {torch.__version__} globals missing from "
            f"SafeUnpickler.ALLOWED_GLOBALS: {missing}",
        )

    def test_round_trips_tensor_payload(self):
        value = [("weight", torch.arange(6).reshape(2, 3))]

        restored = MultiprocessingSerializer.deserialize(
            MultiprocessingSerializer.serialize(value)
        )

        self.assertEqual(restored[0][0], "weight")
        self.assertTrue(torch.equal(restored[0][1], value[0][1]))

    def test_modality_loads_from_its_current_and_pre_move_paths(self):
        # Senders built before the move pickle Modality under schedule_batch;
        # receivers must accept both paths during a rolling upgrade.
        from sglang.srt.multimodal.modality import Modality

        current = pickle.dumps(Modality.IMAGE, protocol=2)
        current_global = b"csglang.srt.multimodal.modality\nModality\n"
        self.assertIn(current_global, current)
        pre_move = current.replace(
            current_global, b"csglang.srt.managers.schedule_batch\nModality\n"
        )
        for payload in (current, pre_move):
            with self.subTest(payload=payload):
                self.assertIs(safe_pickle_loads(payload), Modality.IMAGE)

    def test_platform_rebuild_deserializes_tensor(self):
        """A backend tensor payload must deserialize without general-plugin loading."""

        class Platform(CpuSRTPlatform):
            def get_tensor_ipc_rebuild_fn(self, module, name):
                if (module, name) == ("example_backend.reductions", "rebuild_tensor"):
                    return torch.tensor
                return None

        # GLOBAL + REDUCE reconstructs a tensor from the arguments ((1, 2),).
        payload = b"cexample_backend.reductions\nrebuild_tensor\n((I1\nI2\nttR."
        with patch.object(serialization, "current_platform", Platform()):
            restored = safe_pickle_loads(payload)
        self.assertTrue(torch.equal(restored, torch.tensor([1, 2])))

    def test_platform_default_does_not_allow_unknown_global(self):
        """Declining a backend global must not fall back to unrestricted imports."""
        with patch.object(serialization, "current_platform", CpuSRTPlatform()):
            with self.assertRaisesRegex(RuntimeError, "Blocked unsafe global"):
                safe_pickle_loads(b"cos\nsystem\n.")

    def test_builtin_globals_and_storage_loader_take_precedence(self):
        """Backend lookup must not replace built-ins or the restricted storage loader."""

        class Platform(CpuSRTPlatform):
            def get_tensor_ipc_rebuild_fn(self, module, name):
                raise AssertionError("Built-in globals must not consult the backend")

        with patch.object(serialization, "current_platform", Platform()):
            restored = safe_pickle_loads(b"cbuiltins\nset\n((I1\nI2\nttR.")
            self.assertEqual(restored, {1, 2})
            unpickler = serialization.SafeUnpickler(io.BytesIO())
            self.assertIs(
                unpickler.find_class("torch.storage", "_load_from_bytes"),
                serialization._safe_load_torch_storage,
            )


if __name__ == "__main__":
    unittest.main()
