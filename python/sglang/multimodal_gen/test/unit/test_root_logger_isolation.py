# SPDX-License-Identifier: Apache-2.0
"""A unit test that reconfigures the root logger must not change the next one's."""

import logging

from sglang.multimodal_gen.runtime.server_args import ServerArgs

_ROOT_LEVEL_BEFORE = logging.getLogger().level


def test_server_args_reconfigures_the_root_logger():
    ServerArgs(model_path="dummy", log_level="error")
    assert logging.getLogger().level == logging.ERROR


def test_next_test_sees_the_root_logger_it_started_with():
    assert logging.getLogger().level == _ROOT_LEVEL_BEFORE
