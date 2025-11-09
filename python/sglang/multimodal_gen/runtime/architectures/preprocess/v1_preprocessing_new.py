# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from sglang.multimodal_gen.runtime.distributed import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.workflow.workflow_base import WorkflowBase
from sglang.multimodal_gen.utils import FlexibleArgumentParser

logger = init_logger(__name__)


def main(server_args: ServerArgs) -> None:
    maybe_init_distributed_environment_and_model_parallel(1, 1)
    preprocess_workflow_cls = WorkflowBase.get_workflow_cls(server_args)
    preprocess_workflow = preprocess_workflow_cls(server_args)
    preprocess_workflow.run()


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = ServerArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    main(server_args)
