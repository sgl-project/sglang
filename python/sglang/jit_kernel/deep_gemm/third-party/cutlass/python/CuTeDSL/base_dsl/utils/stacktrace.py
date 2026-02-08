# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# Use of this software is governed by the terms and conditions of the
# NVIDIA End User License Agreement (EULA), available at:
# https://docs.nvidia.com/cutlass/media/docs/pythonDSL/license.html
#
# Any use, reproduction, disclosure, or distribution of this software
# and related documentation outside the scope permitted by the EULA
# is strictly prohibited.

"""
 This module provides stacktrace helper functions
"""

import os
import re


def walk_to_top_module(start_path):
    """
    Walk up from the start_path to find the top-level Python module.

    :param start_path: The path to start from.
    :return: The path of the top-level module.
    """
    current_path = start_path

    while True:
        # Check if we are at the root directory
        if os.path.dirname(current_path) == current_path:
            break

        # Check for __init__.py
        init_file_path = os.path.join(current_path, "__init__.py")
        if os.path.isfile(init_file_path):
            # If __init__.py exists, move up one level
            current_path = os.path.dirname(current_path)
        else:
            # If no __init__.py, we are not in a module; stop
            break

    # If we reached the root without finding a module, return None
    if os.path.dirname(current_path) == current_path and not os.path.isfile(
        os.path.join(current_path, "__init__.py")
    ):
        return None

    # Return the path of the top-level module
    return current_path


def _filter_internal_frames(traceback, internal_path):
    """
    Filter out stack frames from the traceback that belong to the specified module path.

    This function removes stack frames from the traceback whose file paths start with
    the given prefix_path, effectively hiding internal implementation details from
    the error traceback shown to users.
    """
    iter_prev = None
    iter_tb = traceback
    while iter_tb is not None:
        if os.path.abspath(iter_tb.tb_frame.f_code.co_filename).startswith(
            internal_path
        ):
            if iter_tb.tb_next:
                if iter_prev:
                    iter_prev.tb_next = iter_tb.tb_next
                else:
                    traceback = iter_tb.tb_next
        else:
            iter_prev = iter_tb
        iter_tb = iter_tb.tb_next
    return traceback


_generated_function_names = re.compile(
    r"^(loop_body|while_region|while_before_block|while_after_block|if_region|then_block|else_block|elif_region)_\d+$"
)


def _filter_duplicated_frames(traceback):
    """
    Filter out duplicated stack frames from the traceback.
    The function filters out consecutive frames that are in the same file and have the same line number.
    In a sequence of consecutive frames, the logic prefers to keep the non-generated frame or the last frame.
    """
    iter_prev = None
    iter_tb = traceback
    while iter_tb is not None:
        skip_current = False
        skip_next = False
        if iter_tb.tb_next:
            current_filename = os.path.abspath(iter_tb.tb_frame.f_code.co_filename)
            next_filename = os.path.abspath(iter_tb.tb_next.tb_frame.f_code.co_filename)
            # if in the same file, check if the line number is the same
            if current_filename == next_filename:
                current_lineno = iter_tb.tb_lineno
                next_lineno = iter_tb.tb_next.tb_lineno
                if current_lineno == next_lineno:
                    # Same file and line number, check name, if current is generated, skip current, otherwise skip next
                    name = iter_tb.tb_frame.f_code.co_name
                    is_generated = bool(_generated_function_names.match(name))
                    if is_generated:
                        # Skip current
                        skip_current = True
                    else:
                        # Skip next if it's generated, otherwise keep both
                        next_name = iter_tb.tb_next.tb_frame.f_code.co_name
                        skip_next = bool(_generated_function_names.match(next_name))
        if skip_current:
            if iter_prev:
                iter_prev.tb_next = iter_tb.tb_next
            else:
                traceback = iter_tb.tb_next
        elif skip_next:
            # if next is last frame, don't skip
            if iter_tb.tb_next.tb_next:
                iter_tb.tb_next = iter_tb.tb_next.tb_next
            iter_prev = iter_tb
        else:
            iter_prev = iter_tb
        iter_tb = iter_tb.tb_next

    return traceback


def filter_stackframe(traceback, prefix_path):
    """
    Filter out stack frames from the traceback that belong to the specified module path.

    This function removes stack frames from the traceback whose file paths start with
    the given prefix_path, effectively hiding internal implementation details from
    the error traceback shown to users.

    :param traceback: The traceback object to filter.
    :param prefix_path: The path prefix to filter out from the traceback.
    :return: The filtered traceback with internal frames removed.
    """
    # Step 1: filter internal frames
    traceback = _filter_internal_frames(traceback, prefix_path)

    # Step 2: consolidate duplicated frames
    return _filter_duplicated_frames(traceback)


def filter_exception(value, module_dir):
    """
    Filter out internal implementation details from exception traceback.

    This function recursively processes an exception and its cause chain,
    removing stack frames that belong to the specified module directory.
    This helps to present cleaner error messages to users by hiding
    implementation details.

    :param value: The exception object to filter.
    :param module_dir: The module directory path to filter out from tracebacks.
    :return: The filtered exception with internal frames removed.
    """
    if hasattr(value, "__cause__") and value.__cause__:
        filter_exception(value.__cause__, module_dir)

    if hasattr(value, "__traceback__"):
        filter_stackframe(value.__traceback__, module_dir)
