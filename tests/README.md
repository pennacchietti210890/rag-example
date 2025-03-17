# Testing the RAG Example Application

This directory contains tests for the RAG Example application. The tests are organized into unit tests and functional tests.

## Test Structure

- `unit/`: Contains unit tests for individual components
  - `backend/`: Tests for backend components
    - `rag/`: Tests for RAG-related functionality
    - `llm/`: Tests for LLM integration
    - `utils/`: Tests for utility functions
  - `frontend/`: Tests for frontend components
- `functional/`: Contains functional tests for end-to-end flows
- `conftest.py`: Contains shared fixtures for tests
- `pytest.ini`: Configuration for pytest

## Running Tests

### Prerequisites

Make sure you have all the required dependencies installed. Since this project uses Poetry for dependency management, you should have Poetry installed:

```bash
# If you don't have Poetry installed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install
```

### Running All Tests

To run all tests:

```bash
poetry run pytest
```

### Running Specific Test Categories

To run only unit tests:

```bash
poetry run pytest tests/unit
```

To run only functional tests:

```bash
poetry run pytest -m functional
```

To run tests for a specific component:

```bash
poetry run pytest tests/unit/backend/rag
```

### Running with Coverage

To run tests with coverage reporting:

```bash
poetry run pytest --cov=backend --cov=frontend
```

For a detailed HTML coverage report:

```bash
poetry run pytest --cov=backend --cov=frontend --cov-report=html
```

## Writing New Tests

When adding new tests:

1. Follow the existing directory structure
2. Use appropriate markers (`@pytest.mark.unit`, `@pytest.mark.functional`)
3. Use fixtures from `conftest.py` when possible
4. Mock external dependencies to ensure tests are isolated

## Continuous Integration

These tests are designed to be run in a CI/CD pipeline. The pipeline will run all tests and report any failures.

## Test Data

Test data is generated programmatically or through fixtures. No real API keys or sensitive data should be used in tests. 