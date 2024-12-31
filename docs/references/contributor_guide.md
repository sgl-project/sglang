# Contribution Guide

Welcome to **SGLang**! We appreciate your interest in contributing. This guide provides a concise overview of how to set up your environment, run tests, build documentation, and open a Pull Request (PR). Whether you’re fixing a small bug or developing a major feature, we encourage following these steps for a smooth contribution process.

## 1. Setting Up & Building from Source

1. **Clone the Repository**

bash
   git clone https://github.com/sgl-project/sglang.git
   cd sglang

2. **Install Dependencies & Build**
   Refer to our [Install SGLang](https://sgl-project.github.io/start/install.html) documentation for a detailed explanation of setting up the necessary dependencies and compiling the codebase locally.

## 2. Code Formatting with Pre-Commit

We use [pre-commit](https://pre-commit.com/) to maintain consistent code quality and style checks. Before pushing your changes, please run:

bash
pip3 install pre-commit
cd sglang
pre-commit install
pre-commit run --all-files


- **pre-commit install**: Installs the Git hook, which automatically runs style checks on staged files.
- **pre-commit run --all-files**: Manually runs all the configured checks on your repository, applying automatic fixes where possible.

Please ensure your code passes all these checks locally before creating a Pull Request.

## 3. Writing Documentation & Running Docs CI

All documentation files are located in the docs folder. We use make commands for building, cleaning, and previewing the documentation. Below is the typical workflow:

1. **Compile Jupyter Notebooks**

bash
   make compile

   This command processes all .ipynb files into intermediate formats.

2. **Generate HTML Documentation**

bash
   make html

   The generated static site can be found under the _build/html directory.

3. **Local Preview**

bash
   bash serve.sh

   Open your browser at the specified port to view the docs locally.

4. **Clean Notebook Outputs & Build Artifacts**

bash
   find . -name '*.ipynb' -exec nbstripout {} \;
   make clean

   - **nbstripout** removes all outputs from Jupyter notebooks.
   - **make clean** removes temporary files and _build artifacts.

5. **Modify index.rst (If Necessary)**
   Update index.rst whenever you add or remove sections from the documentation.

6. **Pass Pre-Commit and Submit a PR**
   After confirming the docs build successfully and pass all checks, open a Pull Request.

CI will automatically build and validate the documentation to ensure consistency and completeness before merging.

## 4. Running Unit Tests Locally & Adding to CI

SGLang uses Python’s built-in [unittest](https://docs.python.org/3/library/unittest.html) framework. You can run tests either individually or in suites.

### 4.1 Test Backend Runtime

bash
cd sglang/test/srt

# Run a single file
python3 test_srt_endpoint.py

# Run a single test
python3 -m unittest test_srt_endpoint.TestSRTEndpoint.test_simple_decode

# Run a suite with multiple files
python3 run_suite.py --suite minimal


### 4.2 Test Frontend Language

bash
cd sglang/test/lang
export OPENAI_API_KEY=sk-*****

# Run a single file
python3 test_openai_backend.py

# Run a single test
python3 -m unittest test_openai_backend.TestOpenAIBackend.test_few_shot_qa

# Run a suite with multiple files
python3 run_suite.py --suite minimal


### 4.3 Adding or Updating Tests in CI

- Create new test files under test/srt, ensure they are referenced in test/srt/run_suite.py, or they will not be picked up.
- In CI, all tests are executed automatically. You may modify the corresponding workflow in .github/workflows/ if you wish to include custom test groups or additional checks.

### 4.4 Writing Elegant Test Cases

- Examine existing tests in [sglang/test](https://github.com/sgl-project/sglang/tree/main/test) for practical examples.
- Keep each test function focused on a specific scenario or unit of functionality.
- Give tests descriptive names reflecting their purpose.
- Use robust assertions (e.g., assert, unittest methods) to validate outcomes.
- Clean up resources to avoid side effects and preserve test independence.

## 5. Tips for Newcomers

If you want to contribute but don’t have a specific idea in mind, start by picking up issues labeled [“good first issue” or “help wanted”](https://github.com/sgl-project/sglang/issues?q=is%3Aissue+label%3A%22good+first+issue%22%2C%22help+wanted%22). These tasks typically have lower complexity and provide an excellent introduction to the codebase. Feel free to ask questions or request clarifications in the corresponding issue threads.

Thank you again for your interest in SGLang! If you have any questions along the way, feel free to open an issue or start a thread in our [slack channel](https://join.slack.com/t/sgl-fru7574/shared_invite/zt-2um0ad92q-LkU19KQTxCGzlCgRiOiQEw). Happy coding!
