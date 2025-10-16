"""Launch the inference server."""

import os
import sys

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

MOVE_ENVS_WARN = """
########################################################################
# For contributors and developers:                                    #
# Please move environment variable definitions to sglang.srt.environ  #
# using the following pattern:                                        #
#     SGLANG_XXX = EnvBool(False)                                     #
#                                                                     #
########################################################################
"""

if __name__ == "__main__":
    server_args = prepare_server_args(sys.argv[1:])

    from sglang.srt.server_args import print_deprecated_warning

    print_deprecated_warning(MOVE_ENVS_WARN)

    try:
        launch_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
