from sgl_diffusion.runtime.distributed import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sgl_diffusion.runtime.server_args import ServerArgs
from sgl_diffusion.runtime.utils.logging_utils import init_logger
from sgl_diffusion.runtime.workflow.workflow_base import WorkflowBase
from sgl_diffusion.utils import FlexibleArgumentParser

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
