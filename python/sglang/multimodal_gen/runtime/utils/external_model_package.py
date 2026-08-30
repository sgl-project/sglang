# SPDX-License-Identifier: Apache-2.0

import importlib
from functools import lru_cache

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def load_external_model_package() -> None:
    """Import the configured out-of-tree model package once per process."""
    package_name = envs.SGLANG_EXTERNAL_MODEL_PACKAGE
    if not package_name:
        return

    logger.info("Loading external model package: %s", package_name)
    importlib.import_module(package_name)
