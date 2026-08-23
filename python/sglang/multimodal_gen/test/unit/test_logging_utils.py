# SPDX-License-Identifier: Apache-2.0

import logging
import unittest

from sglang._dependency_warnings import suppress_known_dependency_warnings


class TestSuppressNoisyDependencyLogs(unittest.TestCase):
    def test_filters_only_pytree_enum_registration_deprecation(self):
        logger = logging.getLogger("torch.utils._pytree")
        with self.assertLogs(logger, level=logging.WARNING) as captured:
            suppress_known_dependency_warnings()
            logger.warning(
                "<enum 'KernelPreference'> is an Enum subclass and is now "
                "natively supported by torch.compile as an opaque value type. "
                "Calling register_constant() on Enum subclasses is deprecated "
                "and will be an error in a future release."
            )
            logger.warning("unrelated pytree warning")

        self.assertEqual(
            captured.output,
            ["WARNING:torch.utils._pytree:unrelated pytree warning"],
        )

    def test_filters_only_torchao_checkpoint_compatibility_warning(self):
        logger = logging.getLogger("diffusers.quantizers.torchao.torchao_quantizer")
        with self.assertLogs(logger, level=logging.WARNING) as captured:
            suppress_known_dependency_warnings()
            logger.warning(
                "Unable to import `torchao` Tensor objects. This may affect "
                "loading checkpoints serialized with `torchao`"
            )
            logger.warning("unrelated torchao warning")

        self.assertEqual(
            captured.output,
            [
                "WARNING:diffusers.quantizers.torchao.torchao_quantizer:"
                "unrelated torchao warning"
            ],
        )


if __name__ == "__main__":
    unittest.main()
