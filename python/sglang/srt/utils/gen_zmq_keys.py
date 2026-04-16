"""Generate CurveZMQ keypair for SGLang cluster authentication.

Usage::

    python -m sglang.srt.utils.gen_zmq_keys --output /path/to/keys/

This produces two files inside the output directory:

* ``cluster.key``         -- public key only (safe to distribute)
* ``cluster.key_secret``  -- public + secret key (keep secure)

Pass the directory to every SGLang node via ``--zmq-curve-keys-dir`` or the
``SGLANG_ZMQ_CURVE_KEYS_DIR`` environment variable to enable CURVE
encryption and authentication on cross-machine ZMQ sockets.
"""

from __future__ import annotations

import argparse
import os
import sys


def generate_certificates(output_dir: str) -> None:
    import zmq
    import zmq.auth

    if zmq.zmq_version_info() < (4, 0):
        print(
            f"Error: libzmq >= 4.0 required for CURVE security. "
            f"Installed: {zmq.zmq_version()}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not zmq.has("curve"):
        print(
            "Error: This pyzmq/libzmq build does not include CURVE support "
            "(libsodium missing).",
            file=sys.stderr,
        )
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    zmq.auth.create_certificates(output_dir, "cluster")

    # Restrict the secret key file to owner-only so other users on the same
    # machine cannot read it and impersonate a cluster node.
    secret_file = os.path.join(output_dir, "cluster.key_secret")
    os.chmod(secret_file, 0o600)

    print(f"CurveZMQ keypair generated in: {output_dir}")
    print()
    print("Files created:")
    print(f"  {os.path.join(output_dir, 'cluster.key')}          (public key)")
    print(f"  {os.path.join(output_dir, 'cluster.key_secret')}   (secret key)")
    print()
    print("To enable CURVE authentication, pass the directory to every node:")
    print(f"  sglang serve ... --zmq-curve-keys-dir {output_dir}")
    print("  # or")
    print(f"  export SGLANG_ZMQ_CURVE_KEYS_DIR={output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CurveZMQ keypair for SGLang cluster authentication.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to write the keypair files into.",
    )
    args = parser.parse_args()
    generate_certificates(args.output)


if __name__ == "__main__":
    main()
