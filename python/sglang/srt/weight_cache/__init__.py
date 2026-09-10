# SPDX-License-Identifier: Apache-2.0
#
# Intentionally empty: importing any weight_cache submodule (e.g.
# ``sglang.srt.weight_cache.protocol``) executes this package __init__ first.
# Eagerly re-exporting daemon/ipc_loader here would pull in torch and the model
# loader on that cheap protocol import, re-introducing the circular-import and
# startup-cost problems the local-import refactor removed. Import the concrete
# symbols from their submodules instead, e.g.
#     from sglang.srt.weight_cache.protocol import CacheConfig
#     from sglang.srt.weight_cache.daemon import launch_weight_cache_daemons
#     from sglang.srt.weight_cache.ipc_loader import IpcModelLoader
