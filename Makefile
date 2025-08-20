# Makefile for Twitter SOM Analysis Project

.PHONY: help install install-dev test test-verbose test-coverage lint format type-check clean run demo setup

# Default target
help:
	@echo "Available commands:"
	@echo "  setup        - Complete project setup (install dependencies, configure environment)"
	@echo "  install      - Install project dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run all tests"
	@echo "  test-verbose - Run tests with verbose output"
	@echo "  test-coverage- Run tests with coverage report"
	@echo "  lint         - Run code linting (flake8)"
	@echo "  format       - Format code (black, isort)"
	@echo "  type-check   - Run type checking (mypy)"
	@echo "  clean        - Clean up generated files"
	@echo "  run          - Run the main application"
	@echo "  demo         - Run the demonstration"

# Project setup
setup: install-dev configure-python
	@echo "✅ Project setup complete!"

# Install dependencies
install:
	uv pip install -r requirements.in

install-dev:
	uv pip install -e ".[dev]"

# Configure Python environment
configure-python:
	@echo "Configuring Python environment..."
	python -c "import sys; print(f'Python version: {sys.version}')"
	@echo "✅ Python environment configured"

# Testing
test:
	pytest

test-verbose:
	pytest -v

test-coverage:
	pytest --cov=src --cov-report=html --cov-report=term-missing

test-models:
	pytest tests/test_models.py -v

test-preprocessor:
	pytest tests/test_preprocessor.py -v

test-som:
	pytest tests/test_som_analyzer.py -v

test-visualizer:
	pytest tests/test_visualizer.py -v

# Code quality
lint:
	flake8 src/ tests/

format:
	black src/ tests/
	isort src/ tests/

type-check:
	mypy src/

# Quality check (all code quality tools)
check: lint type-check
	@echo "✅ Code quality check complete"

# Clean up
clean:
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf *.egg-info/
	rm -rf build/
	rm -rf dist/
	rm -rf .mypy_cache/
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "✅ Cleanup complete"

# Run application
run:
	python main.py

demo: run

# Development workflow
dev-setup: clean install-dev
	@echo "✅ Development environment ready"

dev-test: format lint test
	@echo "✅ Development testing complete"

# CI/CD simulation
ci: install-dev lint type-check test-coverage
	@echo "✅ CI pipeline simulation complete"

# Project information
info:
	@echo "Twitter SOM Analysis Project"
	@echo "============================"
	@echo "Python version: $(shell python --version)"
	@echo "Project structure:"
	@find . -type f -name "*.py" | head -20
	@echo "..."
	@echo "Total Python files: $(shell find . -type f -name "*.py" | wc -l)"

# Generate requirements.txt from requirements.in
requirements:
	pip-compile requirements.in

# Install specific package for development
install-package:
	@read -p "Enter package name: " package; \
	pip install $$package; \
	pip freeze | grep $$package >> requirements.in; \
	echo "Added $$package to requirements.in"

# Run specific test file
test-file:
	@read -p "Enter test file (e.g., test_models.py): " file; \
	pytest tests/$$file -v

# Profile the application
profile:
	python -m cProfile -o profile_output.prof main.py
	@echo "Profile saved to profile_output.prof"

# Generate documentation (if docs are added later)
docs:
	@echo "Documentation generation not yet implemented"

# Docker commands (if Docker is added later)
docker-build:
	@echo "Docker build not yet implemented"

docker-run:
	@echo "Docker run not yet implemented"

# Database commands (if database is added later)
migrate:
	@echo "Database migrations not yet implemented"

# Backup/restore (if needed)
backup:
	@echo "Creating backup..."
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz src/ tests/ main.py README.md requirements.in pyproject.toml

# Help for specific commands
help-test:
	@echo "Testing Commands:"
	@echo "  test         - Run all tests"
	@echo "  test-verbose - Run tests with detailed output"
	@echo "  test-coverage- Generate coverage report"
	@echo "  test-models  - Test only data models"
	@echo "  test-preprocessor - Test only preprocessor"
	@echo "  test-som     - Test only SOM analyzer"
	@echo "  test-visualizer - Test only visualizer"

help-dev:
	@echo "Development Commands:"
	@echo "  dev-setup    - Setup development environment"
	@echo "  dev-test     - Run full development test suite"
	@echo "  format       - Format code with black and isort"
	@echo "  lint         - Check code style with flake8"
	@echo "  type-check   - Check types with mypy"
	@echo "  clean        - Remove generated files"
