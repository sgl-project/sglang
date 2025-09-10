import argparse
import dataclasses

from sglang.srt.disaggregation.mini_lb import PrefillConfig, VisionConfig, run


@dataclasses.dataclass
class LBArgs:
    host: str = "0.0.0.0"
    port: int = 8000
    policy: str = "random"
    prefill_infos: list = dataclasses.field(default_factory=list)
    decode_infos: list = dataclasses.field(default_factory=list)
    vision_infos: list = dataclasses.field(default_factory=list)
    log_interval: int = 5
    timeout: int = 600

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--host",
            type=str,
            default=LBArgs.host,
            help=f"Host to bind the server (default: {LBArgs.host})",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=LBArgs.port,
            help=f"Port to bind the server (default: {LBArgs.port})",
        )
        parser.add_argument(
            "--policy",
            type=str,
            default=LBArgs.policy,
            choices=["random", "po2"],
            help=f"Policy to use for load balancing (default: {LBArgs.policy})",
        )
        parser.add_argument(
            "--enable-multimodal-disagg",
            action="store_true",
            help="Enable multimodal disaggregation",
        )
        parser.add_argument(
            "--vision",
            type=str,
            default=[],
            nargs="+",
            help="URLs for vision servers",
        )
        parser.add_argument(
            "--vision-bootstrap-ports",
            type=int,
            default=[8998],
            nargs="+",
            help="Bootstrap ports for vision servers",
        )
        parser.add_argument(
            "--prefill",
            type=str,
            default=[],
            nargs="+",
            help="URLs for prefill servers",
        )
        parser.add_argument(
            "--decode",
            type=str,
            default=[],
            nargs="+",
            help="URLs for decode servers",
        )
        parser.add_argument(
            "--prefill-bootstrap-ports",
            type=int,
            nargs="+",
            help="Bootstrap ports for prefill servers",
        )
        parser.add_argument(
            "--log-interval",
            type=int,
            default=LBArgs.log_interval,
            help=f"Log interval in seconds (default: {LBArgs.log_interval})",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=LBArgs.timeout,
            help=f"Timeout in seconds (default: {LBArgs.timeout})",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "LBArgs":
        vision_bootstrap_ports = args.vision_bootstrap_ports
        if vision_bootstrap_ports is None:
            vision_bootstrap_ports = [None] * len(args.vision)
        elif len(vision_bootstrap_ports) == 1:
            vision_bootstrap_ports = vision_bootstrap_ports * len(args.vision)
        else:
            if len(vision_bootstrap_ports) != len(args.vision):
                raise ValueError(
                    "Number of vision URLs must match number of bootstrap ports"
                )
        vision_infos = [
            (url, port) for url, port in zip(args.vision, vision_bootstrap_ports)
        ]

        bootstrap_ports = args.prefill_bootstrap_ports
        if bootstrap_ports is None:
            bootstrap_ports = [None] * len(args.prefill)
        elif len(bootstrap_ports) == 1:
            bootstrap_ports = bootstrap_ports * len(args.prefill)
        else:
            if len(bootstrap_ports) != len(args.prefill):
                raise ValueError(
                    "Number of prefill URLs must match number of bootstrap ports"
                )

        prefill_infos = [
            (url, port) for url, port in zip(args.prefill, bootstrap_ports)
        ]

        return cls(
            host=args.host,
            port=args.port,
            policy=args.policy,
            vision_infos=vision_infos,
            prefill_infos=prefill_infos,
            decode_infos=args.decode,
            log_interval=args.log_interval,
            timeout=args.timeout,
        )


def main():
    parser = argparse.ArgumentParser(
        description="PD Disaggregation Load Balancer Server"
    )
    LBArgs.add_cli_args(parser)
    args = parser.parse_args()
    lb_args = LBArgs.from_cli_args(args)

    vision_configs = [VisionConfig(url, port) for url, port in lb_args.vision_infos]
    prefill_configs = [PrefillConfig(url, port) for url, port in lb_args.prefill_infos]
    run(
        vision_configs,
        prefill_configs,
        lb_args.decode_infos,
        lb_args.host,
        lb_args.port,
        lb_args.timeout,
    )


if __name__ == "__main__":
    main()
