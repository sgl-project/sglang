# SGLang Code Style and Conventions

## Python Style
- **Code Formatter**: Black (configured in pyproject.toml)
- **Import Sorter**: isort with profile=black
- **Line Length**: Default Black settings (88 characters)
- **Import Style**: First-party imports grouped under `sglang`

## Pre-commit Hooks
The project uses pre-commit hooks for code quality:
- Black for Python formatting (including Jupyter notebooks)
- isort for import sorting
- Ruff for linting (specifically checking F401 - unused imports)
- Codespell for spell checking
- clang-format for C++ and CUDA code
- nbstripout for Jupyter notebooks

## Code Organization
- Classes should be clearly named and follow PascalCase
- Functions should use snake_case
- Private functions/methods should start with underscore (_)
- Constants should be in UPPER_CASE

## Type Hints
The codebase uses type hints moderately. When adding new code:
- Use type hints for function parameters and return types where it improves clarity
- For complex data structures, consider using typing module annotations

## Documentation
- Use docstrings for classes and public functions
- Keep comments concise and focused on "why" rather than "what"
- Avoid adding comments unless specifically requested