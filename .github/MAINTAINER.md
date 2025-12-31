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

If you meet any issues during the merge, you can discuss in [slack channels](https://slack.sglang.io/): #dev, #pull-request, and #ci-cd-build-release.

## The List of Merge Oncalls and Reviewers
The format is @github-username (Slack username).

TODO: fill in the list.

Now we have many Merge Oncalls mainly because the CI is flaky and the CODEOWNERS is too coarse-grained.
In the future, we hope the CI can be improved and we only need bypass rarely. After that, most Merge Oncalls can be converted back to Write and CODEOWNERS.

This list is based on the current situation. If you or someone you know would like to take on more responsibility and are qualified, please ping @Lianmin Zheng and @Ying Sheng in the Slack channel. They will start a nomination and internal review process.

## The List of CI Oncalls
The format is @github-username (Slack username).

### NVIDIA GPUs
@merrymercy (Lianmin Zheng), @Kangyan-Zhou (Kangyan Zhou), @ch-wan (Cheng Wan), @HanHan009527 (hanhan), @ishandhanani (Ishan Dhanani), @key4ng (Keyang Ru), @slin1237 (Simo Lin), @ShangmingCai (Shangming Cai)

### AMD GPUs
@saienduri (Sai Enduri), @HaiShaw (Henry HAI)

### Intel CPU and XPU
@mingfeima (Mingfei Ma), @DiweiSun (Diwei Sun)

### Ascend NPUs
@iforgetmyname (Even Zhou)

This list is based on the current situation. If you or someone you know would like to donate machines for CI, they can serve as the CI oncalls for their machines. Please ping @Lianmin Zheng and @Ying Sheng in the Slack channel. They will start a nomination and internal review process.
