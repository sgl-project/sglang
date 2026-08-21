# SPDX-License-Identifier: Apache-2.0
"""ComfyUI checkpoint specs, one module per DiT family.

Importing this package registers every spec. To support a new model, add a
module here and import it below.
"""

from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import (  # noqa: F401
    flux,
    qwen_image,
    zimage,
)
