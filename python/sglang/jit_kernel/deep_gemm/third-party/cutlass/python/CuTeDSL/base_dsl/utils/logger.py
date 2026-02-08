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
This module provides logging helper functions
"""

import logging

logger = None


def log():
    return logger


def setup_log(
    name, log_to_console=False, log_to_file=False, log_file_path=None, log_level=1
):
    """Set up and configure a logger with console and/or file handlers.

    :param name: Name of the logger to create
    :type name: str
    :param log_to_console: Whether to enable logging to console, defaults to False
    :type log_to_console: bool, optional
    :param log_to_file: Whether to enable logging to file, defaults to False
    :type log_to_file: bool, optional
    :param log_file_path: Path to the log file, required if log_to_file is True
    :type log_file_path: str, optional
    :param log_level: Logging level to set, defaults to 1
    :type log_level: int, optional
    :raises ValueError: If log_to_file is True but log_file_path is not provided
    :return: Configured logger instance
    :rtype: logging.Logger
    """
    # Create a custom logger
    global logger
    logger = logging.getLogger(name)
    if log_to_console or log_to_file:
        logger.setLevel(log_level)
    else:
        # Makes sure logging is OFF
        logger.setLevel(logging.CRITICAL + 1)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define formatter
    formatter = logging.Formatter(
        f"%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s"
    )

    # Add console handler if enabled
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Add file handler if enabled
    if log_to_file:
        if not log_file_path:
            raise ValueError("log_file_path must be provided when enable_file is True")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_log("generic")
