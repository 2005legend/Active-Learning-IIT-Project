#!/usr/bin/env python3
"""
GitHub Setup Script for Policy Gradient Active Learning Project
Prepares the project for GitHub upload by creating necessary files and structure
"""

import os
import json
from pathlib import Path

def create_license():
    """Create MIT License file"""
    license_text = """MIT License

Copyright (c) 2024 Policy Gradient Active Learning Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    with open("LICENSE", "w") as f:
        f.write(license_text)
    print("✅ Created LICENSE file")

def create_contributing_guide():
    """Create contributing guidelines"""
    contributing_text = """# Contributing to Policy Gradient Active Learning

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/policy-gradient-active-learning.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Test your changes: `python -m pytest tests/`
6. Commit your changes: `git commit -m "Add your feature"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Create a Pull Request

## 🧪 Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/ -v

# Check code quality
black src/
flake8 src/
mypy src/
```

## 📝 Code Style

- Follow PEP 8 guidelines
- Use Black for code formatting
- Add type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and small

## 🧪 Testing

- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage
- Use descriptive test names

## 📚 Documentation

- Update README.md if needed
- Add docstrings to new functions
- Update type hints
- Include examples in docstrings

## 🐛 Bug Reports

When reporting bugs, please include:
- Python version
- PyTorch version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Error messages/stack traces

## 💡 Feature Requests

For new features, please:
- Check existing issues first
- Describe the use case
- Explain why it would be valuable
- Consider implementation complexity

## 📋 Pull Request Guidelines

- Keep PRs focused and small
- Write clear commit messages
- Update tests and documentation
- Ensure CI passes
- Request review from maintainers

## 🏷️ Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to docs
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed

Thank you for contributing! 🎉
"""
    
    with open("CONTRIBUTING.md", "w") as f:
        f.write(contributing_text)
    print("✅ Created CONTRIBUTING.md file")

def create_sample_data_info():
    """Create information about sample data"""
    sample_data_info = {
        "note": "Large datasets and model checkpoints are excluded from GitHub",
        "datasets": {
            "cats_vs_dogs": {
                "source": "https://www.kaggle.com/c/dogs-vs-cats",
                "size": "25,000 images",
                "location": "data/processed/catsdogs_128/",
                "description": "Main dataset for binary classification"
            },
            "cifar100_wildlife": {
                "source": "https://www.cs.toronto.edu/~kriz/cifar.html",
                "size": "6,000 images (10 animal classes)",
                "location": "data/processed/wildlife/",
                "description": "Multi-class animal classification"
            }
        },
        "model_checkpoints": {
            "baseline_cnn": "checkpoints/week2/resnet18_epoch5.pth",
            "active_learning_models": "checkpoints/week3/",
            "reinforce_policies": "checkpoints/week4/",
            "wildlife_classifier": "checkpoints/wildlife/best_wildlife_model.pth"
        },
        "results_included": {
            "performance_metrics": "outputs/week*/metrics.json",
            "learning_curves": "outputs/week*/curves.json",
            "visualizations": "outputs/performance_comparison.png"
        },
        "setup_instructions": [
            "1. Download datasets from provided sources",
            "2. Run preprocessing: python main_week1.py",
            "3. Train models: python main_week2.py, main_week3.py",
            "4. View results: streamlit run app.py"
        ]
    }
    
    # Create data directory structure info
    os.makedirs("data", exist_ok=True)
    with open("data/README.md", "w") as f:
        f.write("# Dataset Information\n\n")
        f.write("This directory contains the datasets used in the project.\n\n")
        f.write("## Download Instructions\n\n")
        f.write("Due to size constraints, datasets are not included in the repository.\n")
        f.write("Please download from the following sources:\n\n")
        f.write("1. **Dogs vs Cats**: https://www.kaggle.com/c/dogs-vs-cats\n")
        f.write("2. **CIFAR-100**: https://www.cs.toronto.edu/~kriz/cifar.html\n\n")
        f.write("## Structure\n\n")
        f.write("```\n")
        f.write("data/\n")
        f.write("├── raw/           # Original downloaded datasets\n")
        f.write("├── processed/     # Preprocessed images\n")
        f.write("└── splits/        # Train/val/test splits\n")
        f.write("```\n")
    
    with open("data/dataset_info.json", "w") as f:
        json.dump(sample_data_info, f, indent=2)
    
    print("✅ Created data directory and info files")

def create_github_workflows():
    """Create GitHub Actions workflows"""
    os.makedirs(".github/workflows", exist_ok=True)
    
    # CI workflow
    ci_workflow = """name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, "3.10"]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Lint with flake8
      run: |
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=src/ --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
"""
    
    with open(".github/workflows/ci.yml", "w") as f:
        f.write(ci_workflow)
    
    print("✅ Created GitHub Actions CI workflow")

def create_issue_templates():
    """Create GitHub issue templates"""
    os.makedirs(".github/ISSUE_TEMPLATE", exist_ok=True)
    
    # Bug report template
    bug_template = """---
name: Bug report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment (please complete the following information):**
 - OS: [e.g. Windows, macOS, Linux]
 - Python version: [e.g. 3.8, 3.9]
 - PyTorch version: [e.g. 2.0.0]
 - CUDA version (if applicable): [e.g. 11.8]

**Additional context**
Add any other context about the problem here.
"""
    
    with open(".github/ISSUE_TEMPLATE/bug_report.md", "w") as f:
        f.write(bug_template)
    
    # Feature request template
    feature_template = """---
name: Feature request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: enhancement
assignees: ''

---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
"""
    
    with open(".github/ISSUE_TEMPLATE/feature_request.md", "w") as f:
        f.write(feature_template)
    
    print("✅ Created GitHub issue templates")

def main():
    """Main setup function"""
    print("🚀 Setting up project for GitHub...")
    print("=" * 50)
    
    create_license()
    create_contributing_guide()
    create_sample_data_info()
    create_github_workflows()
    create_issue_templates()
    
    print("\n" + "=" * 50)
    print("✅ GitHub setup complete!")
    print("\n📋 Next steps:")
    print("1. Review and customize README_GITHUB.md")
    print("2. Update author information in files")
    print("3. Initialize git repository: git init")
    print("4. Add files: git add .")
    print("5. Commit: git commit -m 'Initial commit'")
    print("6. Create GitHub repository")
    print("7. Add remote: git remote add origin <your-repo-url>")
    print("8. Push: git push -u origin main")
    print("\n🎉 Your project is ready for GitHub!")

if __name__ == "__main__":
    main()