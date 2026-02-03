# MLflow LlamaStack Integration - Testing Guide

This guide provides instructions for running unit tests for the MLflow LlamaStack autologging integration.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Troubleshooting](#troubleshooting)
- [Development Workflow](#development-workflow)

## Prerequisites

### Required Tools

- Python 3.10+
- `uv` package manager (recommended) or `pip`
- Git

### Installation

The tests use mocked versions of `llama-stack-client` to avoid external dependencies. No actual LlamaStack server is required.

```bash
# Clone the MLflow repository (if not already done)
git clone https://github.com/mlflow/mlflow.git
cd mlflow

# Install development dependencies
uv sync
uv pip install -r requirements/test-requirements.txt
```

## Quick Start

Run all LlamaStack tests:

```bash
# Using uv (recommended)
uv run pytest tests/llamastack/test_llamastack_autolog.py -v

# Using pytest directly
pytest tests/llamastack/test_llamastack_autolog.py -v
```

## Running Tests

### All Tests

Run the complete test suite for LlamaStack integration:

```bash
uv run pytest tests/llamastack/ -v
```

Expected output:
```
tests/llamastack/test_llamastack_autolog.py::test_chat_completions_autolog[async] PASSED
tests/llamastack/test_llamastack_autolog.py::test_chat_completions_autolog[sync] PASSED
tests/llamastack/test_llamastack_autolog.py::test_disable_autologging[async] PASSED
tests/llamastack/test_llamastack_autolog.py::test_disable_autologging[sync] PASSED
tests/llamastack/test_llamastack_autolog.py::test_log_traces_false[async] PASSED
tests/llamastack/test_llamastack_autolog.py::test_log_traces_false[sync] PASSED
tests/llamastack/test_llamastack_autolog.py::test_autolog_with_active_run[async] PASSED
tests/llamastack/test_llamastack_autolog.py::test_autolog_with_active_run[sync] PASSED

========================= 8 passed in 0.77s =========================
```

### Specific Test Cases

Run a specific test:

```bash
# Test basic autologging (both sync and async)
uv run pytest tests/llamastack/test_llamastack_autolog.py::test_chat_completions_autolog -v

# Test only async version
uv run pytest "tests/llamastack/test_llamastack_autolog.py::test_chat_completions_autolog[async]" -v

# Test only sync version
uv run pytest "tests/llamastack/test_llamastack_autolog.py::test_chat_completions_autolog[sync]" -v

# Test enable/disable functionality
uv run pytest tests/llamastack/test_llamastack_autolog.py::test_disable_autologging -v

# Test log_traces=False functionality
uv run pytest tests/llamastack/test_llamastack_autolog.py::test_log_traces_false -v

# Test integration with MLflow runs
uv run pytest tests/llamastack/test_llamastack_autolog.py::test_autolog_with_active_run -v
```

### Test with Coverage

Generate coverage report:

```bash
uv run pytest tests/llamastack/ --cov=mlflow.llamastack --cov-report=term-missing --cov-report=html
```

View HTML coverage report:

```bash
# Open coverage report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Test with Different Python Versions

```bash
# Python 3.10
uv run --python 3.10 pytest tests/llamastack/

# Python 3.11
uv run --python 3.11 pytest tests/llamastack/

# Python 3.12
uv run --python 3.12 pytest tests/llamastack/
```

## Test Coverage

The test suite covers the following functionality:

### Core Features

- ✅ **Sync Chat Completions** - `test_chat_completions_autolog[sync]`
  - Validates that synchronous chat completion calls are traced
  - Checks span attributes (name, type, inputs, message format)
  - Verifies trace status is OK

- ✅ **Async Chat Completions** - `test_chat_completions_autolog[async]`
  - Validates that asynchronous chat completion calls are traced
  - Ensures async execution doesn't break tracing
  - Checks span naming includes "Async" prefix

### Autologging Control

- ✅ **Enable/Disable** - `test_disable_autologging`
  - Tests `mlflow.llamastack.autolog(disable=True)`
  - Verifies no traces are created when disabled
  - Tests re-enabling autologging works correctly

- ✅ **Trace Logging Control** - `test_log_traces_false`
  - Tests `mlflow.llamastack.autolog(log_traces=False)`
  - Ensures traces are not logged when disabled
  - Validates the function still executes normally

### Integration Features

- ✅ **MLflow Run Integration** - `test_autolog_with_active_run`
  - Tests autologging within `mlflow.start_run()` context
  - Validates traces are associated with the active run
  - Tests both sync and async within run contexts

### Test Matrix

Each test (except integration tests) runs in two modes:

| Test | Sync | Async |
|------|------|-------|
| Basic Autologging | ✅ | ✅ |
| Disable/Enable | ✅ | ✅ |
| Log Traces False | ✅ | ✅ |
| Active Run Integration | ✅ | ✅ |

**Total**: 8 test cases

## Troubleshooting

### Common Issues

#### 1. ImportError: No module named 'llama_stack_client'

**Cause**: The tests use mocked modules, so this should not happen. If it does, the mock setup failed.

**Solution**: The tests have a `mock_llamastack_client` fixture that mocks all necessary modules. Ensure the fixture is being used.

#### 2. Tests Pass Locally but Fail in CI

**Cause**: Environment differences or missing dependencies.

**Solution**:
```bash
# Run in isolated environment
uv run pytest tests/llamastack/

# Or use tox for multi-environment testing
tox -e py310-llamastack
```

#### 3. Trace Not Captured

**Cause**: Autologging must be called before the client method is invoked.

**Solution**: Ensure `mlflow.llamastack.autolog()` is called before creating client instances:

```python
# Correct order
mlflow.llamastack.autolog()
client = LlamaStackClient(...)
client.chat.completions.create(...)

# Incorrect order
client = LlamaStackClient(...)
mlflow.llamastack.autolog()  # Too late!
client.chat.completions.create(...)
```

#### 4. Async Tests Failing

**Cause**: Event loop issues or improper async handling.

**Solution**: Tests use `asyncio.run()` for async execution. Ensure you're not mixing sync/async incorrectly.

### Debug Mode

Run tests with verbose output and logging:

```bash
# Maximum verbosity
uv run pytest tests/llamastack/ -vv -s --log-cli-level=DEBUG

# Show print statements
uv run pytest tests/llamastack/ -v -s

# Stop on first failure
uv run pytest tests/llamastack/ -x -v
```

### Checking Test Isolation

Ensure tests don't interfere with each other:

```bash
# Run tests in random order
uv run pytest tests/llamastack/ --random-order

# Run tests multiple times to catch flaky tests
uv run pytest tests/llamastack/ --count=10
```

## Development Workflow

### Making Changes

1. **Modify Code**
   ```bash
   # Edit files in mlflow/llamastack/
   vim mlflow/llamastack/autolog.py
   ```

2. **Run Tests**
   ```bash
   uv run pytest tests/llamastack/ -v
   ```

3. **Check Code Quality**
   ```bash
   # Linting
   uv run ruff check mlflow/llamastack/ --fix

   # Formatting
   uv run ruff format mlflow/llamastack/

   # Custom MLflow linting
   uv run clint mlflow/llamastack/
   ```

4. **Run Full Test Suite** (before committing)
   ```bash
   # Run all related tests
   uv run pytest tests/llamastack/ -v

   # Run with coverage
   uv run pytest tests/llamastack/ --cov=mlflow.llamastack --cov-report=term-missing
   ```

### Pre-commit Checks

Before committing changes, run:

```bash
# Install pre-commit hooks
uv run pre-commit install --install-hooks

# Run all pre-commit checks
uv run pre-commit run --all-files

# Or run on specific files
uv run pre-commit run --files mlflow/llamastack/*.py tests/llamastack/*.py
```

### Adding New Tests

When adding new test cases:

1. **Follow Naming Convention**
   - Test functions: `test_<functionality>_<scenario>`
   - Use parametrize for sync/async variants: `@pytest.fixture(params=[True, False])`

2. **Use Fixtures**
   - Mock client: `mock_llamastack_client` (auto-use fixture)
   - Async/Sync toggle: `is_async` fixture

3. **Structure**
   ```python
   def test_new_feature(is_async):
       # Import classes
       from llama_stack_client.resources.chat.completions import AsyncCompletions, Completions

       # Enable autologging
       mlflow.llamastack.autolog()

       try:
           # Test implementation
           ...

           # Assertions
           traces = get_traces()
           assert len(traces) == 1
           ...
       finally:
           # Cleanup
           mlflow.llamastack.autolog(disable=True)
   ```

4. **Run Your New Test**
   ```bash
   uv run pytest tests/llamastack/test_llamastack_autolog.py::test_new_feature -v
   ```

## Integration Testing (Optional)

To test against a real LlamaStack server (requires running server):

```bash
# Start LlamaStack server (in separate terminal)
llama stack run <distribution> --port 8321

# Run integration test
python -c "
import mlflow
import mlflow.llamastack
from llama_stack_client import LlamaStackClient

mlflow.llamastack.autolog()

client = LlamaStackClient(base_url='http://localhost:8321')
response = client.chat.completions.create(
    model='Llama3.2-3B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)

print('Response:', response.choices[0].message.content)

traces = mlflow.search_traces()
print(f'Traces captured: {len(traces)}')
print(f'Model: {traces[0].data.spans[0].model_name}')
"
```

## Contributing

When contributing tests:

1. Ensure all existing tests pass
2. Add tests for new features
3. Maintain >90% code coverage
4. Follow existing test patterns
5. Document complex test scenarios
6. Run linting and formatting checks

For questions or issues, please refer to:
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [LlamaStack Documentation](https://github.com/meta-llama/llama-stack)
- [MLflow GitHub Issues](https://github.com/mlflow/mlflow/issues)
