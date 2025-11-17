# SGLang Code Maintenance Model
This documentation describes the code maintenance model for the SGLang project.
Since SGLang is a large project involving multiple organizations and hardware platforms, we designed this model with the following goals:
- Ensure a responsive and smooth review process.
- Allow fast iteration, so maintainers can sometimes bypass flaky CI tests for important PRs.

## Role Descriptions
There are four important roles in this maintenance model. Some are custom roles, while the others are roles predefined by GitHub.

- *Area Maintainer*: The person who drives the PR merge process. They have strong area expertise and hold a high code quality bar.
  - Permission: Merge PRs. Bypass branch protection rules if needed.
  - Responsibility: Shepherd the merge of PRs assigned to your area. Revert or hotfix any issues related to your merge (especially if you bypass).
- *Codeowner*: The person who protects critical code. Without bypass, each PR needs at least one Codeowner approval for each modified file. Please note that this role is not an honor but a great responsibility because PRs cannot be merged without your approval (except when bypassed by an Area Maintainer).
  - Permission: Approve PRs, allowing them to be merged without bypass.
  - Responsibility: Review PRs in a timely manner.
- *Write*: The person with write permission to the sglang repo.
  - Permission: Merge PRs if they have passed required tests and been approved by Codeowners. This role cannot bypass branch protection rules.
  - Responsibility: Review and merge PRs in a timely manner.
- *CI Maintainer*: The person who manages CI runners for specific hardware platforms.
  - Permission: Add CI runners.
  - Responsibility: Keep the CI runners up and running.

Note: Difference between Area Maintainer and Codeowner
- Area Maintainer is an active role who actively try to help merge PRs and can bypass CI if urgent.
- Codeowner is a passive protection rule provided by GitHub; it prevents accidental changes to critical code.
- The list of Area Maintainer is attached below. The list of Codeowner is in [CODEOWNERS](./CODEOWNERS) file.

## Pull Request Merge Process
1. Author submits a pull request (PR) and fills out the PR checklist.
2. A bot assigns this PR to an Area Maintainer and @ them. At the same time, GitHub will auto-request reviews from Codeowners.
3. The Area Maintainer coordinates the review (asking people to review) and approves the PR; the Codeowner approves the PR.
4. We can now merge the code:
   - Ideal case: For each modified file, one Codeowner approves the PR. It also passes the required CI. Then anyone with write permission can merge the code.
   - It is often very hard to meet all requirements due to flaky CI and slow responses. Area Maintainer can bypass the branch protection and merge the PR.

## The List of Area Maintainers and Reviewers
TODO

## The List of CI Maintainers
TODO
