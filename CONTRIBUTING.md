# Contributing to Beat the Market Research

Thank you for contributing to Beat the Market Research! This document provides guidelines for contributing code, research, and documentation to the project.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Research Contributions](#research-contributions)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Review Process](#review-process)

## Code of Conduct

### Our Standards
- Be respectful and inclusive
- Provide constructive feedback
- Focus on the work, not the person
- Ask questions when unclear
- Help others learn and grow

### Unacceptable Behavior
- Harassment or discriminatory language
- Personal attacks
- Publishing others' private information
- Unprofessional conduct

## Getting Started

### Prerequisites
1. Read the [README.md](README.md) for setup instructions
2. Clone the repository and set up your environment
3. Ensure all tests pass before making changes
4. Familiarize yourself with the codebase structure

### First Contribution
1. Check existing issues for tasks labeled "good first issue"
2. Comment on the issue to indicate you're working on it
3. Follow the development workflow below
4. Ask questions if you get stuck

## Development Workflow

### 1. Create a Branch
```bash
git checkout main
git pull origin main
git checkout -b your_username_feature_name
```

### 2. Make Changes
- Follow coding standards (see below)
- Write clear, focused commits
- Test your changes thoroughly

### 3. Commit Changes
```bash
git add <files>
git commit -m "Clear, descriptive commit message

- Bullet points for details
- Reference issues if applicable

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### 4. Push and Create PR
```bash
git push -u origin your_username_feature_name
```
Then create a Pull Request on GitHub.

## Coding Standards

### Python Style Guide

#### PEP 8 Compliance
Follow [PEP 8](https://pep8.org/) style guide:
- 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- 2 blank lines between top-level functions/classes
- 1 blank line between methods

#### Naming Conventions
```python
# Variables and functions: snake_case
user_count = 10
def calculate_returns():
    pass

# Classes: PascalCase
class PortfolioOptimizer:
    pass

# Constants: UPPER_CASE
MAX_ITERATIONS = 1000
DEFAULT_RISK_FREE_RATE = 0.02

# Private methods/variables: _leading_underscore
def _internal_helper():
    pass
```

#### Type Hints
Always use type hints for function signatures:

```python
from typing import List, Dict, Optional, Tuple
import pandas as pd
import polars as pl

def process_data(
    df: pd.DataFrame,
    columns: List[str],
    threshold: Optional[float] = None
) -> pd.DataFrame:
    """Process data with optional threshold."""
    # Implementation
    return df

def get_tickers() -> pl.DataFrame:
    """Retrieve ticker data."""
    # Implementation
    pass
```

#### Docstrings
Use NumPy-style docstrings for all public functions and classes:

```python
def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Series of periodic returns
    risk_free_rate : float, default=0.0
        Annual risk-free rate
    periods_per_year : int, default=252
        Number of periods per year (252 for daily, 12 for monthly)

    Returns
    -------
    float
        Annualized Sharpe ratio

    Raises
    ------
    ValueError
        If returns series is empty or contains NaN values

    Examples
    --------
    >>> import pandas as pd
    >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
    >>> calculate_sharpe_ratio(returns)
    2.4495

    Notes
    -----
    Sharpe ratio = (mean_return - risk_free_rate) / std_return * sqrt(periods)
    """
    if returns.empty:
        raise ValueError("Returns series cannot be empty")

    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = excess_returns.mean() / excess_returns.std()
    return sharpe * (periods_per_year ** 0.5)
```

#### Code Organization
```python
"""
Module docstring describing the file's purpose.

Author: Your Name
Date: 2024-01-XX
"""

# Standard library imports
import os
import logging
from typing import List, Dict

# Third-party imports
import pandas as pd
import numpy as np

# Local imports
from src.financial_data import FinancialData
from src.s3io import S3IO

# Constants
DEFAULT_LOOKBACK_PERIOD = 252
MAX_POSITION_SIZE = 0.1

# Module-level code
logger = logging.getLogger(__name__)
```

### Code Quality Tools

#### Automated Formatting
Use these tools to maintain code quality:

```bash
# Install tools
pip install black flake8 mypy isort

# Format code automatically
black src/

# Sort imports
isort src/

# Check style
flake8 src/ --max-line-length=100

# Check type hints
mypy src/ --ignore-missing-imports
```

#### Pre-commit Configuration
Consider setting up pre-commit hooks:

```bash
pip install pre-commit
```

Create `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100]
```

### Error Handling

#### Use Specific Exceptions
```python
# GOOD
if ticker not in valid_tickers:
    raise ValueError(f"Invalid ticker: {ticker}")

# AVOID
if ticker not in valid_tickers:
    raise Exception("Error")
```

#### Provide Context
```python
try:
    data = s3.s3_read_parquet(path)
except ClientError as e:
    logger.error(f"Failed to read from S3: {path}")
    raise ValueError(f"Cannot access S3 file: {path}") from e
```

## Research Contributions

### Research Notebook Structure

Every research notebook should follow this structure:

```markdown
# Research Title

**Author:** Your Name
**Date:** 2024-01-XX
**Status:** In Progress | Complete | Archived

## Executive Summary
2-3 sentences summarizing the research and key findings.

## Objective
What are we trying to achieve or learn?

## Research Question
Specific, measurable question this research addresses.

## Hypothesis
What do we expect to find?

## Methodology

### Data Sources
- Data source 1: Description, time period
- Data source 2: Description, time period

### Data Preprocessing
Steps taken to clean and prepare data.

### Models/Algorithms
- Model 1: Description, parameters, rationale
- Model 2: Description, parameters, rationale

### Validation Approach
How we're testing the model/hypothesis.

## Implementation
[Code cells with clear explanations]

## Results

### Key Findings
1. Finding 1: [description]
2. Finding 2: [description]

### Visualizations
[Charts with clear titles, labels, and captions]

### Statistical Significance
[If applicable, include p-values, confidence intervals]

## Discussion

### Interpretation
What do these results mean?

### Limitations
What are the constraints or caveats?

### Practical Implications
How can this be used in trading/investing?

## Conclusions
Summary of what we learned.

## Next Steps
- Future research direction 1
- Future research direction 2

## References
[Papers, articles, books referenced]

## Appendix
[Additional details, supplementary analyses]
```

### Code in Notebooks

#### Abstract Complex Logic
```python
# BAD: Complex logic in notebook cells
for ticker in tickers:
    data = fetch_data(ticker)
    # 50 lines of preprocessing
    # 30 lines of feature engineering
    # 40 lines of model training

# GOOD: Abstract to separate file
from models import preprocess_data, engineer_features, train_model

for ticker in tickers:
    data = fetch_data(ticker)
    processed = preprocess_data(data)
    features = engineer_features(processed)
    model = train_model(features)
```

#### Document Assumptions
```python
# Assumption: Returns are normally distributed
# Assumption: No transaction costs
# Assumption: Prices are in USD

returns = calculate_returns(prices)
```

### Before Submitting Research

Checklist:
- [ ] Notebook runs from top to bottom without errors
- [ ] All visualizations have titles, labels, and legends
- [ ] Results are clearly explained with context
- [ ] Methodology is documented and justified
- [ ] Limitations are acknowledged
- [ ] Code is abstracted to .py files where appropriate
- [ ] All dependencies are in requirements.txt
- [ ] Temporary files and outputs are deleted
- [ ] Notebook is cleared and re-run
- [ ] Research presentation is prepared

## Testing Guidelines

### Writing Tests

Create test notebooks in `src/Research_Notebooks/Testing/`:

```python
def test_financial_data_tickers():
    """Test that ticker listing works correctly."""
    from src.financial_data import FinancialData

    fin_data = FinancialData()
    tickers = fin_data.list_tickers()

    # Test assertions
    assert len(tickers) > 0, "Should return tickers"
    assert 'Symbol' in tickers.columns, "Should have Symbol column"
    assert 'Name' in tickers.columns, "Should have Name column"

    print("✓ Ticker listing test passed")

def test_financial_data_balance_sheet():
    """Test balance sheet retrieval."""
    from src.financial_data import FinancialData

    fin_data = FinancialData()

    # Test valid ticker
    balance = fin_data.get_balance_sheet('AAPL')
    assert len(balance) > 0, "Should return data for valid ticker"

    # Test invalid ticker
    try:
        fin_data.get_balance_sheet('INVALID')
        assert False, "Should raise error for invalid ticker"
    except ValueError:
        pass

    print("✓ Balance sheet test passed")

# Run tests
test_financial_data_tickers()
test_financial_data_balance_sheet()
```

### Test Coverage

Aim to test:
- **Happy path**: Normal, expected usage
- **Edge cases**: Boundary conditions, empty inputs
- **Error conditions**: Invalid inputs, missing data
- **Integration**: Multiple components working together

## Documentation

### When to Update Documentation

Update documentation when:
- Adding new features or functions
- Changing existing behavior
- Adding new dependencies
- Modifying workflows or processes
- Discovering common issues (add to troubleshooting)

### Documentation Types

#### Code Documentation
- Docstrings for all public functions and classes
- Inline comments for complex logic
- Type hints for all functions

#### README Updates
- New features in appropriate sections
- Updated installation steps if needed
- New troubleshooting entries

#### Research Documentation
- Document methodology and findings in notebooks
- Update research summary when complete
- Cross-reference related research

## Pull Request Process

### Before Creating PR

1. **Update from main**
   ```bash
   git checkout main
   git pull origin main
   git checkout your_branch
   git merge main
   # Resolve conflicts if any
   ```

2. **Run tests**
   - Execute test notebooks
   - Verify your changes work
   - Check for unintended side effects

3. **Clean up**
   - Remove debug code
   - Delete temporary files
   - Clear notebook outputs and re-run
   - Review your own changes

4. **Update documentation**
   - Update docstrings
   - Update README if needed
   - Add/update tests

### Creating the PR

1. Push your branch: `git push origin your_branch`
2. Go to GitHub and create Pull Request
3. Fill out the PR template completely
4. Tag all team members for review
5. Add appropriate labels (feature, bugfix, research, etc.)

### PR Title Format

Use clear, descriptive titles:
- `Add momentum trading strategy backtesting`
- `Fix S3 timeout issue in financial_data.py`
- `Improve documentation for data access`
- `Research: Mean reversion in tech stocks`

## Review Process

### For Authors

**Responding to Reviews:**
- Respond to all comments
- Ask for clarification if needed
- Make requested changes or explain why not
- Mark conversations as resolved when addressed
- Re-request review after making changes
- Be open to feedback

### For Reviewers

**Reviewing Code:**
- Check correctness and logic
- Verify style guidelines are followed
- Look for potential bugs or edge cases
- Ensure documentation is adequate
- Test the changes if possible
- Provide specific, actionable feedback

**Review Checklist:**
- [ ] Code is correct and does what PR claims
- [ ] Style guidelines followed (PEP 8)
- [ ] Functions have docstrings and type hints
- [ ] No hardcoded credentials or sensitive data
- [ ] Error handling is appropriate
- [ ] Tests are included (if applicable)
- [ ] Documentation updated (if needed)
- [ ] No unnecessary code or files
- [ ] Commits are clean and messages clear

**Providing Feedback:**
```markdown
# GOOD feedback examples

"Consider extracting this logic into a separate function for reusability:
[code example]"

"This could be more efficient using polars instead of pandas:
[code example]"

"Add a docstring here to explain the parameters and return value."

# AVOID vague feedback

"This doesn't look right"
"Change this"
"I don't like this approach"
```

### Approval Requirements

PRs require:
- At least 2 approvals from team members
- All conversations resolved
- No merge conflicts
- Tests passing (if applicable)
- For research: Presentation to team and approval

## Questions?

If you have questions about contributing:
- Check the [README](README.md) first
- Search existing issues
- Ask in team standups
- Create a GitHub issue with the "question" label
- Contact maintainers directly

## Recognition

Contributors will be:
- Listed in project acknowledgments
- Credited in research papers/posts
- Recognized in team meetings

Thank you for contributing to Beat the Market Research!
