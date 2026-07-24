# Kubernetes deployment recipes for sgl-router

Working manifest sets for two deployment shapes:

- [`plain/`](./plain/) — plain-mode routing: one pool of SGLang workers,
  router does prefix-cache-aware / power-of-two / sticky dispatch.
  Reference model: Qwen3-32B, TP=2, 2 replicas.
- [`pd/`](./pd/) — prefill/decode disaggregation: two pools (prefill +
  decode), router dual-sends the request body with an injected
  `bootstrap_room`, decode's response is client-visible.
  Reference model: DeepSeek-V3-Instruct, TP=8, 1 prefill + 1 decode.

Both variants use `sgl-router --service-discovery` against the Kubernetes
EndpointSlice API — no static worker URL list, no manual reconciliation.

Apply with:

```bash
kubectl apply -k experimental/sgl-router/deploy/k8s/plain
# or
kubectl apply -k experimental/sgl-router/deploy/k8s/pd
```

For the full walkthrough with prerequisites, tuning, observability, and
troubleshooting, see the
[Router Kubernetes deployment guide](../../../../docs_new/docs/advanced_features/router_k8s_deployment.mdx).

These manifests are documented recipes, not chart artifacts. For a
Helm-based deployment integrated into a larger platform, adapt them into
your own chart — or see downstream distributions such as clusterdOS which
package this shape as an opt-in `gitapp`.
