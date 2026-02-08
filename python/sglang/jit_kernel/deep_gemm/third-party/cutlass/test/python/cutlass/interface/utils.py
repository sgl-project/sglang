#################################################################################################
#
# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#################################################################################################

"""
Helper functions & classes for interface test
"""
class ExpectException:
    """
    Utility class to assert that an exception was raised when expected

    Example:

    .. highlight:: python
    .. code-block:: python

        with ExceptionExpected(True, 'Division by zero'):
            x = 1.0 / 0.0

    :param exception_expected: whether an exception is expected to be raised
    :type exception_expected: bool
    :param message: message to print if an exception is raised when not expected or vice versa
    :type message: str
    """
    def __init__(self, exception_expected: bool, message: str = '', verify_msg=False):
        self.exception_expected = exception_expected
        self.message = message
        self.verify_msg = verify_msg

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        exception_raised = exc_type is not None
        assert self.exception_expected == exception_raised, self.message
        if self.verify_msg:
            exc_message = f"{exc_type.__name__}: {exc_val}"
            assert exc_message == self.message, f"expect error message {self.message}, got {exc_message}"

        # Suppress the exception
        return True
