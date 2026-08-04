# AWS Spot benchmark topology

The base kustomization creates one `p5.4xlarge` H100 Denoiser and one
`g6.2xlarge` or `g6.4xlarge` L4 TAEHV worker. Both NodePools are Spot-only,
expire after four hours, and must still be deleted explicitly after a run.
The benchmark reuses the existing `default/s3-claim` for the donor model and
creates a read-only S3 CSI volume for the checkpoint in
`leap-world-us-west-2`. It also reuses the ECR image, GPU EC2NodeClasses, and
AWS Load Balancer Controller. It does not create or write a DynamoDB lease
table; one Gateway process owns the in-memory leases.

Before applying, render the manifests and replace `REPLACE_WITH_GIT_SHA` with a
pushed commit SHA:

```bash
SHA=$(git rev-parse HEAD)
kubectl kustomize benchmark/minwm_realtime_async_vae/k8s \
  | sed "s/REPLACE_WITH_GIT_SHA/${SHA}/g" \
  | kubectl apply -f -
```

For the synchronous TAEHV baseline, set `REALTIME_VAE_WORKER_URL` to an empty
value and restart only the Denoiser. Restore the URL for the async run. The
`l40s-vae.yaml` NodePool is an explicit fallback if L4 Spot is unavailable; it
is intentionally absent from the base kustomization.

Cleanup is mandatory:

```bash
kubectl delete -k benchmark/minwm_realtime_async_vae/k8s --wait=true --timeout=20m
kubectl delete nodepool minwm-async-vae-l40s --ignore-not-found
```
