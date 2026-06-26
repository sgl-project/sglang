"""Hellaswag sanity kit.

Runs hellaswag via the sgl frontend DSL bound to ``self.base_url`` and
asserts accuracy above a threshold. Catches systematic regressions that
pass every cheap single-prompt probe but tank multi-choice reasoning.

Mix into a ``CustomTestCase`` subclass exposing ``self.base_url``.
"""


class HellaswagMixin:
    """Assert hellaswag accuracy > threshold."""

    hellaswag_accuracy_threshold: float = 0.60

    def test_accuracy_floor(self):
        import sglang as sgl
        from sglang.test.test_programs import test_hellaswag_select

        sgl.set_default_backend(sgl.RuntimeEndpoint(self.base_url))
        try:
            accuracy, _ = test_hellaswag_select()
        finally:
            sgl.set_default_backend(None)
        self.assertGreater(
            accuracy,
            self.hellaswag_accuracy_threshold,
            f"hellaswag accuracy floor breached: {accuracy:.3f}",
        )
