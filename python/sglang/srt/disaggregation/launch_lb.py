from sglang.srt.disaggregation.mini_lb import PrefillConfig
from sglang.srt.disaggregation.mini_lb import run as mini_lb_run


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Mini Load Balancer Server")
    parser.add_argument(
        "--prefill", type=str, default=[], nargs="+", help="URLs for prefill servers"
    )
    parser.add_argument(
        "--decode", type=str, default=[], nargs="+", help="URLs for decode servers"
    )
    parser.add_argument(
        "--prefill-bootstrap-ports",
        type=int,
        nargs="+",
        help="Bootstrap ports for prefill servers",
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind the server (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server (default: 8000)"
    )
    args = parser.parse_args()

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

    prefill_configs = [
        PrefillConfig(url, port) for url, port in zip(args.prefill, bootstrap_ports)
    ]

    mini_lb_run(prefill_configs, args.decode, args.host, args.port)


if __name__ == "__main__":
    main()
