# Native gRPC health checks

The native gRPC listener enabled by `--grpc-port` exposes
`grpc.health.v1.Health/Check` on the same port as
`sglang.runtime.v1.SglangService`. The existing SGLang `HealthCheck` RPC is also
available.

Both the empty service name (overall health) and
`sglang.runtime.v1.SglangService` report runtime readiness. They return
`NOT_SERVING` during startup, when the runtime is unhealthy, or during shutdown,
and `SERVING` when the runtime is ready. Checks read runtime state without
submitting inference requests. A runtime check failure returns `NOT_SERVING`.

An unknown service name returns `NOT_FOUND` from `Check`. Streaming `Watch`
returns `UNIMPLEMENTED`; the server does not poll runtime health in the
background.

For a Kubernetes container running SGLang with `--grpc-port 50051`, add these
fields to its container spec. Native
[gRPC probes](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/#define-a-grpc-liveness-probe)
are stable in Kubernetes 1.27 and later:

```yaml
startupProbe:
  grpc:
    port: 50051
    service: sglang.runtime.v1.SglangService
  periodSeconds: 5
  timeoutSeconds: 2
  failureThreshold: 120
readinessProbe:
  grpc:
    port: 50051
    service: sglang.runtime.v1.SglangService
  periodSeconds: 5
  timeoutSeconds: 2
```

Set SGLang's `--host` to an address reachable through the Pod IP, such as
`0.0.0.0`. Adjust the startup probe's failure threshold for your model's load
and warmup time. These probes describe readiness; they do not execute a model
inference to test GPU responsiveness.
