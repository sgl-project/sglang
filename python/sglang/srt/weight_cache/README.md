# Weight cache protocol compatibility

The engine and weight cache daemon must use the same `protocol_version`.
This version describes the socket protocol and weight metadata, not the SGLang
release. Different SGLang releases can share a cache if their protocol versions
and `CacheConfig` values match.

Before it requests weights, the engine sends a `handshake` request with its
protocol version. The daemon checks it and replies with its own version. The
engine checks the reply, then sends `fetch_state` on the same connection.
That request also carries the version, which the daemon checks before it parses
the config or exports IPC handles. `ping` and `query_config` do not require a
handshake.

A missing, invalid, or different protocol version is an error. Version 1 does
not support peers that predate this handshake. There is no automatic downgrade
or disk fallback when a daemon uses an incompatible protocol. The existing
Torch and GPU checks in `CacheConfig` still apply.

To upgrade across a protocol change, stop the engines attached to the daemon,
upgrade both sides to compatible versions, then restart the daemon before the
engines. Do not stop a daemon while engines still use its shared weights.

Increase `PROTOCOL_VERSION` when a change makes existing protocol messages or
weight metadata incompatible. Keep the version for changes that both peers can
already handle.
