# SGLang Task Completion Checklist

When completing a coding task in SGLang, ensure you:

## 1. Code Quality Checks
- [ ] Run code formatting: `make format` or use individual tools:
  - `black <file.py>` for Python formatting
  - `isort <file.py>` for import sorting
- [ ] Ensure pre-commit hooks pass: `pre-commit run --all-files`

## 2. Testing
- [ ] Run relevant tests to verify changes work correctly
- [ ] Add new tests if implementing new features
- [ ] Common test commands:
  ```bash
  cd test
  pytest test/srt/test_*.py  # For server runtime tests
  pytest test/lang/test_*.py # For language API tests
  ```

## 3. Type Checking and Linting
- [ ] The project uses Ruff for linting (configured in pre-commit)
- [ ] If the user provides specific lint/typecheck commands, run them
- [ ] If unsure, ask the user for the correct commands and suggest adding them to CLAUDE.md

## 4. Important Reminders
- Follow existing code patterns and conventions
- Use the same libraries and frameworks already in the codebase
- Never add comments unless specifically requested
- Prefer editing existing files over creating new ones
- Only create documentation files if explicitly requested

## 5. Git Operations (only when explicitly requested)
- Never commit changes unless the user explicitly asks
- When creating commits, follow the repository's commit style
- Never push to remote unless explicitly requested