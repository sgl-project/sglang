import unittest

from sglang.srt.true_on_policy import kernel_inventory
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKernelInventory(unittest.TestCase):
    def test_every_reference_resolves(self):
        """A bump that removes or renames a stock symbol the manifest names, or moves a gate or pin site, fails here."""
        self.assertEqual(kernel_inventory.problems(), [])

    def test_pinned_keywords_exist_on_the_stock_kernel(self):
        """A stock kernel that drops or renames a keyword TOP pins would ignore or reject the pin."""
        self.assertEqual(kernel_inventory.signature_problems(), [])

    def test_checked_in_inventory_is_current(self):
        """The inventory is generated; regenerate it after changing an op row, the manifest or a cited reason."""
        self.assertEqual(
            kernel_inventory.INVENTORY_PATH.read_text(),
            kernel_inventory.render(),
            "run: python python/sglang/srt/true_on_policy/kernel_inventory.py generate",
        )


if __name__ == "__main__":
    unittest.main()
