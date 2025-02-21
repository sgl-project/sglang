# Contribution Guide

Welcome to **SGLang**! We appreciate your interest in contributing. This guide provides a concise overview of how to set up your environment, run tests, build documentation, and open a Pull Request (PR). Whether you’re fixing a small bug or developing a major feature, we encourage following these steps for a smooth contribution process.

## Setting Up & Building from Source

### Fork and Clone the Repository

**Note**: New contributors do **not** have the write permission to push to the official SGLang repo. Please fork the repository under your GitHub account, then clone your fork locally.

```bash
git clone https://github.com/<your_user_name>/sglang.git
```

### Install Dependencies & Build

Refer to [Install SGLang from Source](https://docs.sglang.ai/start/install.html#method-2-from-source) documentation for more details on setting up the necessary dependencies.

## Code Formatting with Pre-Commit

We use [pre-commit](https://pre-commit.com/) to maintain consistent code style checks. Before pushing your changes, please run:

```bash
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

- **`pre-commit run --all-files`** manually runs all configured checks, applying fixes if possible. If it fails the first time, re-run it to ensure lint errors are fully resolved. Make sure your code passes all checks **before** creating a Pull Request.
- **Do not commit** directly to the `main` branch. Always create a new branch (e.g., `feature/my-new-feature`), push your changes, and open a PR from that branch.

## Running Unit Tests & Adding to CI

SGLang uses Python's built-in [unittest](https://docs.python.org/3/library/unittest.html) framework. For detailed instructions on running tests and adding them to CI, please refer to [test/README.md](https://github.com/sgl-project/sglang/tree/main/test/README.md).

## Writing Documentation & Running Docs CI

We recommend new contributors start from writing documentation, which helps you quickly understand SGLang codebase. For more details, please refer to [docs/README.md](https://github.com/sgl-project/sglang/tree/main/docs/README.md).

## Tips for Newcomers

If you want to contribute but don’t have a specific idea in mind, pick issues labeled [“good first issue” or “help wanted”](https://github.com/sgl-project/sglang/issues?q=is%3Aissue+label%3A%22good+first+issue%22%2C%22help+wanted%22). These tasks typically have lower complexity and provide an excellent introduction to the codebase. Also check out this [code walk-through](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/tree/main/sglang/code-walk-through) for a deeper look into SGLang’s workflow.

If you have any questions or want to start a discussion, please feel free to ask in our [Slack channel](https://join.slack.com/t/sgl-fru7574/shared_invite/zt-2um0ad92q-LkU19KQTxCGzlCgRiOiQEw).

Thank you for your interest in SGLang—**happy coding**!
