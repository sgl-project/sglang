# SGLang Code Maintenance Model
This document describes the code maintenance model for the SGLang project.
Since SGLang is a large project involving multiple organizations and hardware platforms, we designed this model with the following goals:
- Ensure a responsive and smooth review process.
- Allow for fast iteration, so maintainers can sometimes bypass flaky CI tests for important PRs.

## Role Descriptions
There are four roles in this maintenance model. Some are custom roles, while others are predefined by GitHub.

- **Merge Oncall**: The person who drives the PR merge process. They have strong area-specific expertise and uphold a high bar for code quality.
  - Permission: Merge PRs. Bypass branch protection rules if needed.
  - Responsibility: Shepherd the merge of PRs assigned to their area. Revert or hotfix any issues related to their merge (especially if they bypass).
- **Codeowner**: The person who protects critical code. Without a bypass, each PR needs at least one Codeowner approval for each modified file protected by [CODEOWNERS](./CODEOWNERS). Please note that this role is not an honor but a significant responsibility because PRs cannot be merged without your approval (except when bypassed by a Merge Oncall).
  - Permission: Approve PRs, allowing them to be merged without a bypass.
  - Responsibility: Review PRs in a timely manner.
- **Write**: A person with write permission to the SGLang repo.
  - Permission: Merge PRs if they have passed required tests and been approved by Codeowners. This role cannot bypass branch protection rules.
  - Responsibility: Review and merge PRs in a timely manner.
- **CI Oncall**: A person who manages CI runners for specific hardware platforms.
  - Permission: Add CI runners.
  - Responsibility: Keep the CI runners up and running.

__Note__: Difference between Merge Oncall and Codeowner
- The Merge Oncall is an active role held by someone who actively tries to help merge PRs and can bypass CI if needed.
- The Codeowner is a passive protection role provided by GitHub; it prevents accidental changes to critical code.
- The list of Merge Oncalls is attached below. The list of Codeowners is in the [CODEOWNERS](./CODEOWNERS) file.

__Note__: The permissions to trigger CI tests are defined separately according to these [rules](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests).


## Pull Request Merge Process
1. The author submits a pull request (PR) and fills out the PR checklist.
2. A bot assigns this PR to a Merge Oncall and @-mentions them. At the same time, GitHub will automatically request reviews from Codeowners.
3. Someone tags the PR with a `run-ci` label ([help](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests)). Then the author can trigger CI by pushing new commits.
4. The Merge Oncall coordinates the review (e.g., asking people to review) and approves the PR; the Codeowners also approve the PR. If the assigned Merge Oncall is not responsive, the author can ping other related Merge Oncalls and Reviewers in the list below.
5. The code can now be merged:
   - **Ideal case:** For each modified file, one Codeowner has approved the PR. The PR has also passed the required CI tests. Then, anyone with write permission can merge the PR.
   - **Exception:** In cases where it is difficult to meet all requirements (due to flaky CI or slow responses), a Merge Oncall can bypass branch protection to merge the PR.

If you meet any issues during the merge, you can discuss in [slack channels](https://slack.sglang.io/): #pull-request, #ci-cd-build-release, #dev.

## The List of Merge Oncalls and Reviewers
This section lists the oncalls for each module or feature.
The format is @github-username (Slack username).

### Scheduler
[@merrymercy](https://github.com/merrymercy) (Lianmin Zheng), [@hnyls2002](https://github.com/hnyls2002) (Liangsheng Yin), [@cctry](https://github.com/cctry) (Shiyang Chen)

related files
- python/sglang/srt/managers
- python/sglang/srt/model_executor

### Diffusion
[@mickqian](https://github.com/mickqian) (Mick), [@BBuf](https://github.com/BBuf) (BBuf)

related files
- python/sglang/multimodal_gen

### PD disaggregation
[@ByronHsu](https://github.com/ByronHsu) (Byron Hsu), [@cctry](https://github.com/cctry) (Shiyang Chen), [@ShangmingCai](https://github.com/ShangmingCai) (Shangming Cai)

related files
- python/sglang/srt/disaggregation

### KV Cache
[@ispobock](https://github.com/ispobock) (Ke Bao), [@xiezhq-hermann](https://github.com/xiezhq-hermann) (Zhiqiang Xie)

related files
- python/sglang/srt/mem_cache

### Parallelism
[@ch-wan](https://github.com/ch-wan) (Cheng Wan), [@fzyzcjy](https://github.com/fzyzcjy) (Tom)

related files
- python/sglang/srt/eplb
- python/sglang/srt/distributed
- python/sglang/srt/layers/dp_attention.py

### Kernel
[@BBuf](https://github.com/BBuf) (BBuf)

related files
- python/sglang/jit_kernel
- sgl-kernel

### Speculative decoding
[@hnyls2002](https://github.com/hnyls2002) (Liangsheng Yin), [@Qiaolin-Yu](https://github.com/Qiaolin-Yu) (Qiaolin Yu)

related files
- python/sglang/srt/speculative

### NV and model-specific optimizations
[@Fridge003](https://github.com/Fridge003) (Baizhou Zhang), [@ishandhanani](https://github.com/ishandhanani) (Ishan Dhanani), [@Qiaolin-Yu](https://github.com/Qiaolin-Yu) (Qiaolin Yu)

related files
- python/sglang/srt/models
- python/sglang/srt/layers/attention

### AMD optimizations
[@HaiShaw](https://github.com/HaiShaw) (Henry HAI)

### NPU optimizations
[@iforgetmyname](https://github.com/iforgetmyname) (Even Zhou)

related files
- python/sglang/srt/hardware_backend/npu

### CI, Release, Package
[@Kangyan-Zhou](https://github.com/Kangyan-Zhou) (Kangyan Zhou), [@Fridge003](https://github.com/Fridge003) (Baizhou Zhang)

related files
- .github/workflows

### Router, API
[@slin1237](https://github.com/slin1237) (Simo Lin)

related files
- sgl-model-gateway
- python/sglang/srt/grpc
- python/sglang/srt/entrypoints

### Other Notes

Now we have many Merge Oncalls mainly because the CI is flaky and the CODEOWNERS is too coarse-grained.
In the future, we hope the CI can be improved and we only need bypass rarely. After that, most Merge Oncalls can be converted back to Write and CODEOWNERS.

This list is based on the current situation. If you or someone you know would like to take on more responsibility and are qualified, please ping [Lianmin Zheng](https://github.com/merrymercy) and [Ying Sheng](https://github.com/Ying1123) in the Slack channel. They will start a nomination and internal review process.

## The List of CI Oncalls
This section lists the oncalls for each hardware platform. The format is @github-username (Slack username).

### NVIDIA GPUs
[@Kangyan-Zhou](https://github.com/Kangyan-Zhou) (Kangyan Zhou), [@ch-wan](https://github.com/ch-wan) (Cheng Wan), [@HanHan009527](https://github.com/HanHan009527) (hanhan), [@ishandhanani](https://github.com/ishandhanani) (Ishan Dhanani), [@ShangmingCai](https://github.com/ShangmingCai) (Shangming Cai), [@alisonshao](https://github.com/alisonshao) (Alison Shao).

### AMD GPUs
[@saienduri](https://github.com/saienduri) (Sai Enduri), [@HaiShaw](https://github.com/HaiShaw) (Henry HAI)

### Intel CPU and XPU
[@mingfeima](https://github.com/mingfeima) (Mingfei Ma), [@DiweiSun](https://github.com/DiweiSun) (Diwei Sun)

### Ascend NPUs
[@iforgetmyname](https://github.com/iforgetmyname) (Even Zhou)

This list is based on the current situation. If you or someone you know would like to donate machines for CI, they can serve as the CI oncalls for their machines. Please ping [Lianmin Zheng](https://github.com/merrymercy) and [Ying Sheng](https://github.com/Ying1123) in the Slack channel. They will start a nomination and internal review process.

## CI Maintenance Mode
When the CI is unhealthy (e.g., the scheduled pr-test on `main` is broken for consecutive runs), the project enters **CI Maintenance Mode** by opening [issue #21065](https://github.com/sgl-project/sglang/issues/21065). While active:
- All PR CI runs are paused. Resources are allocated to PRs that fix the CI.
- **Merging non-CI-fix PRs is prohibited.** Only PRs that fix the CI may be merged. In severe cases, merge permissions may be revoked.

Maintenance mode ends when `pr-test.yml` is all green on `main` and the issue is closed.

## Suspending Permissions
If a Merge Oncall bypasses checks to merge a PR that breaks the `main` branch, merges a non-CI-fix PR during CI Maintenance Mode, or repeatedly breaks the CI due to various reasons, their privileges will be suspended for at least two days, depending on the severity of the incident.
