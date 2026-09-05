"""Unit tests for the quantization scheme ABCs - CPU-only, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import importlib
import inspect
import pkgutil
import subprocess
import sys
import unittest

from sglang.srt.layers.quantization.awq.schemes.awq_scheme import (
    AWQLinearSchemeBase,
    AWQMoESchemeBase,
)
from sglang.srt.layers.quantization.base_scheme import BaseLinearScheme, BaseMoEScheme
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_scheme import (
    CompressedTensorsLinearScheme,
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.quantization.gptq.schemes.gptq_scheme import (
    GPTQLinearSchemeBase,
    GPTQMoESchemeBase,
)
from sglang.srt.layers.quantization.modelslim.schemes.modelslim_scheme import (
    ModelSlimLinearScheme,
    ModelSlimMoEScheme,
)
from sglang.srt.layers.quantization.quark.schemes.quark_scheme import (
    QuarkLinearScheme,
    QuarkMoEScheme,
)
from sglang.test.test_utils import CustomTestCase

SCHEME_PACKAGES = [
    "sglang.srt.layers.quantization.awq.schemes",
    "sglang.srt.layers.quantization.compressed_tensors.schemes",
    "sglang.srt.layers.quantization.gptq.schemes",
    "sglang.srt.layers.quantization.modelslim.schemes",
    "sglang.srt.layers.quantization.quark.schemes",
]

FAMILY_LINEAR_BASES = [
    AWQLinearSchemeBase,
    CompressedTensorsLinearScheme,
    GPTQLinearSchemeBase,
    ModelSlimLinearScheme,
    QuarkLinearScheme,
]

FAMILY_MOE_BASES = [
    AWQMoESchemeBase,
    CompressedTensorsMoEScheme,
    GPTQMoESchemeBase,
    ModelSlimMoEScheme,
    QuarkMoEScheme,
]

# Set by the class statement itself, so present on any empty class body.
CLASS_BODY_NOISE = {
    "__doc__",
    "__module__",
    "__qualname__",
    "__abstractmethods__",
    "__dict__",
    "__weakref__",
    "_abc_impl",
    "__firstlineno__",
    "__static_attributes__",
}


def _scheme_classes():
    """(class, defining module) for every class in every family's schemes package."""
    seen = set()
    for package_name in SCHEME_PACKAGES:
        package = importlib.import_module(package_name)
        for info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{package_name}.{info.name}")
            for obj in vars(module).values():
                if not inspect.isclass(obj):
                    continue
                if not issubclass(obj, (BaseLinearScheme, BaseMoEScheme)):
                    continue
                key = f"{obj.__module__}.{obj.__qualname__}"
                if key in seen:
                    continue
                seen.add(key)
                yield obj, key


class TestRootContract(CustomTestCase):
    """The two root ABCs carry the whole scheme contract.

    Which members are abstract is the load-bearing part: `abstractmethod` on a
    root member makes every scheme in every family that does not define it
    unconstructible, so widening these sets is a breaking change and not a
    tightening of types.
    """

    def test_linear_abstract_methods(self):
        self.assertEqual(
            sorted(BaseLinearScheme.__abstractmethods__),
            ["apply_weights", "create_weights", "process_weights_after_loading"],
        )

    def test_moe_abstract_methods(self):
        # `apply_weights` and `create_moe_runner` are deliberately absent:
        # families whose MoE method drives the MoeRunner leave both to the
        # method, and the schemes below prove that is not hypothetical.
        self.assertEqual(
            sorted(BaseMoEScheme.__abstractmethods__),
            ["create_weights", "process_weights_after_loading"],
        )

    def test_declared_but_unimplemented_members_raise(self):
        class Scheme(BaseMoEScheme):
            def create_weights(self, *args, **kwargs):
                raise AssertionError("not reached")

            def process_weights_after_loading(self, layer):
                raise AssertionError("not reached")

        scheme = Scheme()
        with self.assertRaises(NotImplementedError):
            scheme.apply_weights(layer=None, dispatch_output=None)
        with self.assertRaises(NotImplementedError):
            scheme.create_moe_runner(layer=None, moe_runner_config=None)
        with self.assertRaises(NotImplementedError):
            Scheme.get_min_capability()
        with self.assertRaises(NotImplementedError):
            BaseLinearScheme.get_min_capability()

    def test_defaults(self):
        self.assertIs(BaseLinearScheme.requires_weight_loader_v2, False)
        self.assertIs(BaseMoEScheme.load_up_proj_weight_first, False)

    def test_the_roots_do_not_import_the_moe_package(self):
        # base_scheme is imported by every family, so a runtime import of
        # sglang.srt.layers.moe here would re-enter the quantization <-> moe
        # cycle that the rest of the package works around with function-local
        # imports. The MoE types it names are TYPE_CHECKING-only.
        code = (
            "import sys;"
            "import sglang.srt.layers.quantization.base_scheme;"
            "print([m for m in sys.modules if m.startswith('sglang.srt.layers.moe')])"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
        self.assertEqual(eval(out.strip()), [])


class TestFamilyBases(CustomTestCase):
    """Each family's base exists to name the family, not to restate the contract.

    Redeclaring `create_weights` and friends on a family base is how ten
    near-identical copies accumulated; this is the check that catches the
    eleventh.
    """

    def test_linear_bases_subclass_the_root_and_add_nothing(self):
        for base in FAMILY_LINEAR_BASES:
            with self.subTest(base=base.__name__):
                self.assertTrue(issubclass(base, BaseLinearScheme))
                self.assertIsNot(base, BaseLinearScheme)
                self.assertEqual(set(vars(base)) - CLASS_BODY_NOISE, set())

    def test_moe_bases_subclass_the_root_and_add_nothing(self):
        for base in FAMILY_MOE_BASES:
            with self.subTest(base=base.__name__):
                self.assertTrue(issubclass(base, BaseMoEScheme))
                self.assertIsNot(base, BaseMoEScheme)
                self.assertEqual(set(vars(base)) - CLASS_BODY_NOISE, set())

    def test_family_bases_are_distinct_types(self):
        # The return annotations across the family configs (-> QuarkLinearScheme,
        # -> CompressedTensorsMoEScheme, ...) only say something if these are
        # not all the same object.
        bases = FAMILY_LINEAR_BASES + FAMILY_MOE_BASES
        self.assertEqual(len({id(base) for base in bases}), len(bases))

    def test_family_bases_stay_abstract(self):
        for base in FAMILY_LINEAR_BASES + FAMILY_MOE_BASES:
            with self.subTest(base=base.__name__):
                self.assertTrue(base.__abstractmethods__)


class TestConcreteSchemes(CustomTestCase):
    def test_every_concrete_scheme_is_constructible(self):
        """No scheme is left abstract by the shared contract.

        A member that is abstract on a root but unimplemented by a scheme does
        not fail at import; it fails at model load, on whichever platform owns
        that scheme.
        """
        family_bases = set(FAMILY_LINEAR_BASES) | set(FAMILY_MOE_BASES)
        roots = {BaseLinearScheme, BaseMoEScheme}
        checked = 0
        for cls, key in _scheme_classes():
            if cls in family_bases or cls in roots:
                continue
            with self.subTest(scheme=key):
                self.assertEqual(sorted(cls.__abstractmethods__), [])
            checked += 1
        self.assertGreater(checked, 20, "scheme discovery found almost nothing")

    def test_moe_schemes_may_leave_the_runner_to_their_method(self):
        """The reason `apply_weights` is declared and not abstract.

        These schemes implement neither `apply_weights` nor `create_moe_runner`;
        ModelSlim's MoE method owns the MoeRunner. Making either abstract on
        BaseMoEScheme makes all of them unconstructible.
        """
        from sglang.srt.layers.quantization.modelslim.schemes.modelslim_w8a8_int8_moe import (
            ModelSlimW8A8Int8MoE,
        )

        self.assertEqual(sorted(ModelSlimW8A8Int8MoE.__abstractmethods__), [])
        for member in ("apply_weights", "create_moe_runner"):
            with self.subTest(member=member):
                owner = next(
                    cls
                    for cls in ModelSlimW8A8Int8MoE.__mro__
                    if member in cls.__dict__
                )
                self.assertIs(owner, BaseMoEScheme)

    def test_a_scheme_may_still_override_the_loader_order(self):
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4_moe import (
            CompressedTensorsW4A4Nvfp4MoE,
        )

        self.assertIsInstance(
            CompressedTensorsW4A4Nvfp4MoE.load_up_proj_weight_first, property
        )


if __name__ == "__main__":
    unittest.main()
