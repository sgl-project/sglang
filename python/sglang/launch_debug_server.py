import argparse

from enochian_studio.cli import start_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the debug studio server")
    parser.add_argument(
        "--PORT",
        type=int,
        default=56765,
        help="Port to run the server on (default: 56765)",
    )

    args = parser.parse_args()
    start_server(port=args.PORT)
