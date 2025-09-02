# Contributing to Policy Gradient Active Learning

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/policy-gradient-active-learning.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes: `python -m pytest tests/`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/ -v

# Check code quality
black src/
flake8 src/
```

## Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting
- Add type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and small

## Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Use descriptive test names

## Documentation

- Update README.md if needed
- Add docstrings to new functions
- Update type hints
- Include examples in docstrings

## Bug Reports

When reporting bugs, please include:
- Python version
- PyTorch version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

## Feature Requests

For new features, please:
- Check existing issues first
- Describe the use case
- Explain why it would be valuable
- Consider implementation complexity

## Pull Request Guidelines

- Keep PRs focused and small
- Write clear commit messages
- Update tests and documentation
- Ensure CI passes
- Request review from maintainers

Thank you for contributing!