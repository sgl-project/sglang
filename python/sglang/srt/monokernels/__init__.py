"""Full-forward dispatch for vendor monokernel plugins.

General plugins register specs through the sglang.srt.plugins entry point.
The runtime validates configuration, observes weight loading, and routes
supported steps through the selected spec; other steps use the model forward.
"""

from sglang.srt.monokernels.guards import handle_monokernel
from sglang.srt.monokernels.runtime import attach, watch_weights
from sglang.srt.monokernels.spec import API_VERSION, register

__all__ = ["API_VERSION", "attach", "handle_monokernel", "register", "watch_weights"]
