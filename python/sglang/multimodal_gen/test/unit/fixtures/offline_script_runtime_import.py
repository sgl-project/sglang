# SPDX-License-Identifier: Apache-2.0

"""An offline script that reaches past the facade at module scope.

The child therefore imports the runtime before it can initialize its platform.
Nothing in-tree can reorder that; the child is expected to say so.
"""

from offline_script_runner import main

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (  # noqa: F401
    DiffGenerator,
)

if __name__ == "__main__":
    main()
