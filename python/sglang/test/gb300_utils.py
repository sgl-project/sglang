"""Shared configuration for tests on the dedicated GB300 runner."""

# The runner's ephemeral client-port range is 10240-65535. Keep the TCPStore
# rendezvous below it so a recently closed client connection cannot make the
# subsequent TCPStore bind fail with EADDRINUSE.
GB300_NCCL_PORT = "10000"
