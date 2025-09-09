### Sync Code Between OSS and Private Fork

You can use the following principles and tools to sync the code between a private fork and the OSS repo [sgl-project/sglang](https://github.com/sgl-project/sglang/tree/main).
It learns from [Copybara](https://github.com/google/copybara), a tool used at Google for maintaining open-source code synchronization.

## Principals

- The core folders (e.g., `python/sglang/srt`) are 100% mirrored between the private fork and OSS repo.
- The OSS repo is the single source of truth. If one commit changes `python/sglang/srt` in the private repo, the change should be synced to the OSS repo as soon as possible with the action B below.
- The common code (e.g., base classes, well-known techniques in the industry without private secrets) goes to `python/sglang/srt`. The private-specific code (e.g., with private-specific features, confidential info) goes to `python/sglang/private` .
- Anytime you want to make private changes to a file or class under `python/sglang/srt`, duplicate the file and move it under `python/sglang/private`. You can achieve code reuse by importing and inheriting.

## How to sync the code bidirectionally
### Action A: Copy code from OSS to private

- We can run this action: [Open A PR to Copy Code From OSS](https://github.com/sgl-project/sglang/tree/main/.github/workflows/open-pr-copy-from-oss.yml)
    - It opens a PR to copy all files under certain folders (e.g., `python/sglang/srt` , `test/srt` , `sgl-kernel` ) from the OSS main branch to the private fork.
    - Since the OSS repo is the single source of truth, this action copies files and overwrites any changes in the private fork. To prevent the private changes from being overwritten, you need to ensure all private changes are merged into the OSS repo before running this action.
- This action will be run automatically every day and can also be triggered manually.

### Action B: Copy diff from private to OSS

- We can run this action: [Open A PR to Copy Code To OSS](https://github.com/sgl-project/sglang/tree/main/.github/workflows/open-pr-copy-to-oss.yml)
    - It opens a PR to apply the diff of one specific commit of the private fork to the OSS main branch. It will only pick the changes under certain folders (e.g., `python/sglang/srt` , `test/srt` , `sgl-kernel` ) and ignore changes under private folders (e.g., `python/sglang/private` )
    - For example, you can have a PR that changes both `python/sglang/srt` and `python/sglang/private/srt`. Once you merge the PR into the private repo, `python/sglang/srt` becomes desynced between the two repos. You need to run this action on your merge commit immediately to open a PR to send your diff to the OSS repo. Then, we need to merge the OSS PR as soon as possible. Once your OSS PR is merged, we can run action A again.
    - Action A copies files directly, but Action B applies diff. This is because OSS is the source of truth; action A can just copy files. Action B cannot copy, so it uses diff instead.
- This action currently needs a manual trigger in order to prevent incidental code leaks. One can also consider making it automatic.
