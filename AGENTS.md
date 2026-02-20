# Agent Instructions: Tracepipe Pipelines

## Purpose

Data processing pipelines for Tracepipe — Session-to-Example transformation and LLM-powered action classification.

## Project Type

**Application** — Production data pipeline service meant for deployment.

## Documentation

For platform-level documentation and cross-repository coordination, see [tracepipe](https://github.com/CRSS-AI/tracepipe) and [tracepipe-docs](https://github.com/CRSS-AI/tracepipe-docs).

## Before Writing Code

1. **Plan first** — outline approach before coding; pause if user jumps to implementation
2. **Get verification criteria** — ask "How should I verify this works?" (tests, expected output, pipeline behavior)
3. **Match existing patterns** — read relevant code to understand project style

## Code Style

**Naming**: Descriptive names; avoid type-hint-redundant suffixes (`questions_df` → `questions`)

**Type hints**: Required on all new code; modern syntax (`list[int]` not `List[int]`); Google-style docstrings

**Imports**: Never use `sys.path.append/insert` or `PYTHONPATH` hacks; use proper `pyproject.toml` config

**Formatting**: Trailing commas in multi-line collections; named arguments for multi-param calls; enforce with `*` in definitions

**Comments**: Only when providing non-obvious context; code should be self-explanatory, DRY, SOLID

**Terminology**: "model" over "LLM"

**Circular imports**: Never use `if TYPE_CHECKING` pattern; avoid circular references entirely

**Reflection**: Never use `hasattr` or `getattr`; use explicit types and classes

**Exception handling**: Route and test exceptions by type, not by error message phrasing

## Quality Checks

Run before committing:

```bash
uv run ruff check --fix .   # lint
uv run ruff format .        # format + imports
uvx ty check                # types (fix sensible issues, ignore noise)
uv run pytest               # tests
```

**Pre-commit requirement**: Run `uvx ty check` before committing — **type checking errors are unacceptable in committed code**. Fix legitimate type issues; document pre-existing ones but ensure your new code is correctly typed.

## Verification

- Verify pipeline executes without runtime errors
- **Run the pipeline and test manually with sample data** — not just automated tests
- Verify output data quality and format
- Summarize: what changed, why, verification results
- Fix failures before claiming success

## Testing

**Test types** (all required for each feature):

- **Unit tests**: Test a single class/module in isolation
- **Integration tests**: Test a single pipeline stage end-to-end with external calls mocked (APIs, DB, file system, model inference)
- **E2E tests**: Test complete pipeline execution with real or representative test data

**Requirements**:

- **pytest** with **Arrange-Act-Assert** pattern
- Fixtures for setup; mock external dependencies in integration tests
- Production-quality: type-hinted, documented, DRY, SOLID
- **80% coverage per feature** (not average) — all three test types should exist when adding features

**Pipeline-specific considerations**:

- Test data transformations with edge cases (empty inputs, malformed data, boundary values)
- Verify idempotency where applicable
- Test error handling and retry logic

## Test Maintenance

**When making breaking changes:**

- Update ALL affected tests (unit, integration, E2E) to reflect the new contract
- "Expected failures" is never acceptable — if a test fails due to your change, fix the test

**Regression test failures after your changes mean:**

1. Your change broke something unintentionally → fix your code
2. Your change intentionally altered the contract → update the tests

Never report test failures as "expected" or "intentional" without also fixing them.

## Don'ts

- Hard-code values to pass tests
- Suppress errors without fixing root cause
- Modify files outside task scope
- Claim success without verification
- Skip planning for non-trivial tasks

## Git Workflow

**Branching**: `feature/{JIRA_ticket}` from default branch; commit frequently; **never push until instructed**

**Commits**: Clear messages; single logical change per commit; **NEVER commit when tests are failing** — all tests must pass before creating a commit

**Pre-commit**: Run `uvx ty check` before committing — type checking errors are unacceptable in committed code
