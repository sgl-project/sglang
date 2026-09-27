# Serving a model? Read its cookbook page first

Before running, deploying, benchmarking, or reproducing a specific model, read
its page under `docs/cookbook/`:

- `docs/cookbook/autoregressive/` — LLM and VLM serving
- `docs/cookbook/diffusion/` — image, video, 3D, robotics
- `docs/cookbook/omni/`, `docs/cookbook/vla/`, `docs/cookbook/specbundle/`,
  `docs/cookbook/base/`

That page carries the deployment we recommend to users — GPU count, parallelism
degrees, quantization, and the flags that matter for that model — so it is the
answer to "how should this be served", and the baseline any tuning starts from.

Configs under `test/` are a different thing: pinned, reproducible CI setups for
correctness and regression checks, not deployment advice. Reach for them when
you need an exactly reproducible run, and say which cookbook flags you diverged
from — a measurement taken on a configuration nobody deploys describes nothing.

Keep it true in both directions: when a change alters a model's recommended
deployment, update that model's cookbook page in the same PR.
