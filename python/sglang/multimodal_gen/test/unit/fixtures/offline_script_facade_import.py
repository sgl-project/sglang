# SPDX-License-Identifier: Apache-2.0

"""The supported offline-script layout: only the facade at module scope.

``spawn`` re-executes this in every child, so binding ``DiffGenerator`` here
must not import the diffusion runtime.
"""

from offline_script_runner import main

from sglang.multimodal_gen import DiffGenerator  # noqa: F401

if __name__ == "__main__":
    main()
