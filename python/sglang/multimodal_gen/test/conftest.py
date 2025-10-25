# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import torch

from sgl_diffusion.runtime.distributed import (
    cleanup_dist_env_and_memory,
    maybe_init_distributed_environment_and_model_parallel,
)


@pytest.fixture(scope="function")
def distributed_setup():
    """
    Fixture to set up and tear down the distributed environment for tests.

    This ensures proper cleanup even if tests fail.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    yield

    cleanup_dist_env_and_memory()
