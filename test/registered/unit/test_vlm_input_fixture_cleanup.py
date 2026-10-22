"""CPU regressions for VLM fixture cleanup; no model weights or engine launch."""

import importlib.util
import traceback
import unittest
import weakref
from contextlib import ExitStack
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestVLMInputFixtureCleanup(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        maybe_stub_sgl_kernel()
        path = Path(__file__).resolve().parents[1] / "vlm/test_vlm_input_format.py"
        spec = importlib.util.spec_from_file_location("_vlm_cleanup_fixture", path)
        cls.vlm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.vlm)

    def test_teardown_releases_modules_before_emptying_cache(self):
        for shutdown_fails in (False, True):
            with self.subTest(shutdown_fails=shutdown_fails):
                fixture = type("Fixture", (self.vlm.VLMInputTestBase,), {})
                shutdown_error = RuntimeError("engine shutdown failed")
                events = []

                def shutdown():
                    events.append("shutdown")
                    if shutdown_fails:
                        raise shutdown_error

                fixture.engine = SimpleNamespace(shutdown=shutdown)
                fixture.vision_tower = torch.nn.Linear(2, 2)
                # Require cyclic GC, and leave the module last in the class
                # dictionary to catch a teardown loop retaining its last value.
                fixture.vision_tower.cycle = [fixture.vision_tower]
                module_ref = weakref.ref(fixture.vision_tower)
                self.assertEqual(next(reversed(vars(fixture))), "vision_tower")
                cache_states = []

                def empty_cache():
                    events.append("cache")
                    cache_states.append(
                        (
                            fixture.engine is None,
                            fixture.vision_tower is None,
                            module_ref() is None,
                        )
                    )

                with patch.object(self.vlm, "empty_gpu_cache", side_effect=empty_cache):
                    if shutdown_fails:
                        with self.assertRaises(RuntimeError) as caught:
                            fixture.tearDownClass()
                        self.assertIs(caught.exception, shutdown_error)
                    else:
                        fixture.tearDownClass()

                self.assertEqual(events, ["shutdown", "cache"])
                self.assertEqual(cache_states, [(True, True, True)])

    def test_failed_visual_initialization_releases_traceback_references(self):
        setup_classes = (
            self.vlm.TestGemmaUnderstandsImage,
            self.vlm.TestInternVLUnderstandsImage,
            self.vlm.TestMiniCPMVUnderstandsImage,
        )
        for setup_class, chain in product(
            setup_classes, ("direct", "cause", "context")
        ):
            with self.subTest(setup_class=setup_class.__name__, chain=chain):
                setup_error = RuntimeError("visual initialization failed")
                allocation_error = ValueError("module allocation failed")
                refs = []
                cache_states = []

                def allocate_and_fail(cls):
                    # The parent exists only in this failed frame. It retains
                    # the child even after tearDownClass clears the class attr.
                    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
                    cls.vision_tower = model[0]
                    refs.extend((weakref.ref(model), weakref.ref(model[0])))
                    raise setup_error if chain == "direct" else allocation_error

                def fail_initialization(cls):
                    try:
                        allocate_and_fail(cls)
                    except ValueError as exc:
                        if chain == "cause":
                            raise setup_error from exc
                        raise setup_error

                fixture = type(
                    "FailedFixture",
                    (setup_class,),
                    {"_init_visual": classmethod(fail_initialization)},
                )

                def empty_cache():
                    # safe_setUpClass suppresses teardown errors. Record only
                    # booleans here, then assert outside that exception handler.
                    cache_states.append(
                        (
                            fixture.engine is None,
                            fixture.vision_tower is None,
                            tuple(ref() is None for ref in refs),
                        )
                    )

                with ExitStack() as stack:
                    stack.enter_context(
                        patch("sglang.test.test_utils._wait_for_gpu_idle_in_ci")
                    )
                    stack.enter_context(
                        patch("sglang.test.test_utils.is_in_ci", return_value=False)
                    )
                    # All setup paths reach their real visual-initialization
                    # wrapper without network, model, or accelerator work.
                    stack.enter_context(patch.object(self.vlm, "_is_cuda", False))
                    stack.enter_context(patch.object(self.vlm, "_is_xpu", False))
                    stack.enter_context(
                        patch("torch.cuda.is_available", return_value=False)
                    )
                    stack.enter_context(
                        patch.object(
                            self.vlm.requests,
                            "get",
                            return_value=SimpleNamespace(content=b""),
                        )
                    )
                    stack.enter_context(patch.object(self.vlm.Image, "open"))
                    stack.enter_context(
                        patch.object(
                            self.vlm.AutoProcessor,
                            "from_pretrained",
                            return_value=SimpleNamespace(tokenizer=object()),
                        )
                    )
                    stack.enter_context(
                        patch("transformers.AutoTokenizer.from_pretrained")
                    )
                    stack.enter_context(
                        patch("sglang.srt.utils.hf_transformers.get_tokenizer")
                    )
                    stack.enter_context(
                        patch.object(self.vlm, "_fix_added_tokens_encoding")
                    )
                    stack.enter_context(
                        patch.object(
                            self.vlm, "empty_gpu_cache", side_effect=empty_cache
                        )
                    )
                    start_engine = stack.enter_context(
                        patch.object(fixture, "_start_engine")
                    )
                    try:
                        fixture.setUpClass()
                    except RuntimeError as exc:
                        self.assertIs(exc, setup_error)
                        self.assertEqual(str(exc), "visual initialization failed")
                        self.assertIn(
                            "fail_initialization",
                            [
                                frame.name
                                for frame in traceback.extract_tb(exc.__traceback__)
                            ],
                        )
                        if chain != "direct":
                            self.assertIs(exc.__context__, allocation_error)
                            self.assertIs(
                                exc.__cause__,
                                allocation_error if chain == "cause" else None,
                            )
                            self.assertEqual(
                                str(allocation_error), "module allocation failed"
                            )
                            self.assertIn(
                                "allocate_and_fail",
                                [
                                    frame.name
                                    for frame in traceback.extract_tb(
                                        allocation_error.__traceback__
                                    )
                                ],
                            )
                    else:
                        self.fail(
                            "setUpClass did not propagate the initialization error"
                        )

                    start_engine.assert_not_called()

                self.assertEqual(cache_states, [(True, True, (True, True))])


if __name__ == "__main__":
    unittest.main()
