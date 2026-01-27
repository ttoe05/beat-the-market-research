# Beat the Market Research

A quantitative finance research repository for developing and testing trading strategies, portfolio optimization models, and price forecasting algorithms. This repository provides infrastructure for accessing financial data from AWS S3 and Yahoo Finance, with a focus on reproducible research workflows.

## Overview

This repository is designed to facilitate quantitative finance research by providing:
- Clean abstraction layer for accessing financial data from S3 and Yahoo Finance
- Structured workflow for conducting and documenting research
- Collaboration tools and best practices for team-based research
- Production-ready code that can be deployed to the Beat the Market dashboard

### Technology Stack
- **Python 3.9+** - Primary programming language
- **Polars** - High-performance dataframe library (primary)
- **Pandas** - Alternative dataframe library (supported)
- **AWS S3** - Cloud storage for financial data
- **Yahoo Finance API** - Real-time and historical market data
- **scikit-learn** - Machine learning models
- **Jupyter** - Interactive research notebooks

## Repository Structure

```
beat-the-market-research/
├── src/                              # Source code directory
│   ├── financial_data.py            # Main data access class (FinancialData)
│   ├── s3io.py                      # S3 interaction wrapper (S3IO)
│   ├── claude/                      # Documentation and planning
│   │   ├── readme_plan.md          # Original documentation plan
│   │   ├── readme_planII.md        # Comprehensive documentation plan
│   │   └── fin_data.md             # FinancialData class design document
│   └── Research_Notebooks/          # Research and analysis
│       ├── Testing/                 # Testing notebooks and examples
│       │   └── fin_data_test.ipynb # Example: Testing FinancialData class
│       ├── price_forecasting/       # Price prediction research
│       │   └── auto_kernel_ensemble.py # Kernel autoregressive model
│       └── [Other research areas]/  # Add your research folders here
├── claude/                          # Project documentation
│   └── readme_planII.md            # This documentation plan
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── .env.example                    # Environment variable template
└── .gitignore                      # Git ignore patterns
```

### Directory Purposes

- **src/** - All source code and research notebooks
  - **financial_data.py** - User-friendly interface for accessing financial data without exposing S3 paths
  - **s3io.py** - Low-level S3 operations wrapper
  - **Research_Notebooks/** - Research projects organized by topic
- **claude/** - Project-level documentation and planning files

### Naming Conventions
- Research folders: Use descriptive names like `price_forecasting`, `portfolio_optimization`, `risk_analysis`
- Python files: Use lowercase with underscores (snake_case)
- Notebooks: Use descriptive names ending in `.ipynb`

## Prerequisites

Before getting started, ensure you have:

### Required Software
- **Python 3.9 or higher** (tested on Python 3.9.6)
- **Git** for version control
- **AWS CLI** (optional, but recommended)

### Required Access
- **AWS Account** with S3 access
- **IAM Permissions** for S3 bucket operations:
  - `s3:GetObject` - Read files from S3
  - `s3:PutObject` - Write files to S3 (if applicable)
  - `s3:ListBucket` - List bucket contents
- **AWS Credentials** configured locally

### Recommended Tools
- **IDE**: PyCharm, VS Code, or Jupyter Lab
- **Terminal**: For running git and python commands

## Installation & Setup

### Step 1: Clone Repository

```bash
git clone git@github.com:ttoe05/beat-the-market-research.git
cd beat-the-market-research
```

### Step 2: Set Up Python Environment

Choose one of the following approaches:

#### Option A: Using venv (recommended)
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate  # On Windows
```

#### Option B: Using conda
```bash
# Create conda environment
conda create -n btm-research python=3.9

# Activate environment
conda activate btm-research
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- Data manipulation: `polars`, `pandas`, `numpy`
- AWS integration: `boto3`
- Financial APIs: `yfinance`
- Machine learning: `scikit-learn`, `scipy`
- Jupyter notebooks: `jupyter`, `notebook`, `ipykernel`

### Step 4: Configure AWS Credentials

#### Create AWS Credentials File
Create or edit `~/.aws/credentials` (macOS/Linux) or `%USERPROFILE%\.aws\credentials` (Windows):

```ini
[default]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY

[beat-the-market]
aws_access_key_id = YOUR_ACCESS_KEY_ID
aws_secret_access_key = YOUR_SECRET_ACCESS_KEY
```

#### Create AWS Config File
Create or edit `~/.aws/config`:

```ini
[default]
region = us-east-1

[profile beat-the-market]
region = us-east-1
```

**Resources:**
- [AWS SDK Credentials Setup](https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html)
- [Boto3 Credentials Guide](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)

### Step 5: Set Environment Variables

#### Create .env File
Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and set your values:

```bash
# Required: Your S3 bucket name
S3_ARB_BUCKET=your-bucket-name-here

# Required: AWS profile name from ~/.aws/credentials
S3_PROFILE=default
```

#### Load Environment Variables

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, or `~/.bash_profile`):

```bash
export S3_ARB_BUCKET="your-bucket-name"
export S3_PROFILE="default"
```

Or source them in your current session:

```bash
source .env
# OR on some systems:
export $(cat .env | xargs)
```

**Alternative: Using python-dotenv**
In your Python scripts/notebooks:

```python
from dotenv import load_dotenv
load_dotenv()  # Loads variables from .env file
```

### Step 6: Verify Installation

Test your setup with Python:

```python
import os
from src.financial_data import FinancialData

# Check environment variables
print(f"S3 Bucket: {os.environ.get('S3_ARB_BUCKET')}")
print(f"S3 Profile: {os.environ.get('S3_PROFILE')}")

# Initialize FinancialData
fin_data = FinancialData()

# List available tickers
tickers = fin_data.list_tickers()
print(f"Successfully loaded {len(tickers)} tickers!")
```

## Quick Start Guide

### Basic Data Access

```python
from src.financial_data import FinancialData

# Initialize the data client
fin_data = FinancialData()

# List all available tickers
tickers = fin_data.list_tickers()
print(f"Available tickers: {len(tickers)}")
print(tickers.head())

# Get tickers as pandas DataFrame
tickers_pd = fin_data.list_tickers(as_pandas=True)
```

### Accessing Financial Statements

```python
# Get balance sheet for Apple
balance_sheet = fin_data.get_balance_sheet('AAPL')
print(balance_sheet)

# Get income statement for Microsoft (as pandas)
income_stmt = fin_data.get_income_statement('MSFT', as_pandas=True)
print(income_stmt)

# Get cash flow statement for Google
cash_flow = fin_data.get_cash_flow('GOOGL')
print(cash_flow)
```

### Getting Historical Stock Prices

```python
# Get maximum historical daily prices for Tesla
prices = fin_data.get_historical_stock_prices('TSLA')
print(prices.head())

# Get last year of data for Amazon (as pandas)
prices_1y = fin_data.get_historical_stock_prices(
    'AMZN',
    as_pandas=True,
    period='1y'
)

# Get weekly data for the last 6 months
prices_weekly = fin_data.get_historical_stock_prices(
    'NVDA',
    period='6mo',
    interval='1wk'
)
```

### Working with S3 Directly (Advanced)

```python
from src.s3io import S3IO

# Initialize S3 client
s3 = S3IO()

# List files in a path
files = s3.s3_list('balance/')
print(files[:5])  # Show first 5 files

# Read parquet file directly
data = s3.s3_read_parquet('balance/AAPL/balance.parq')

# Write parquet file
import polars as pl
df = pl.DataFrame({'col1': [1, 2, 3]})
s3.s3_write_parquet(df, 'test/output.parq')

# Check if path exists
exists = s3.s3_is_dir('balance/')
print(f"Path exists: {exists}")
```

## Research Workflow

### Creating New Research

1. **Create a Research Folder**
   ```bash
   mkdir -p src/Research_Notebooks/your_research_topic
   cd src/Research_Notebooks/your_research_topic
   ```

2. **Create Subfolders if Needed**
   ```bash
   mkdir -p experiments
   mkdir -p models
   mkdir -p data
   ```

3. **Create Your Notebook**
   ```bash
   jupyter notebook
   # Create a new notebook in your research folder
   ```

### Research Documentation Standards

Each research notebook should include:

#### 1. Header Section
```markdown
# Research Title

**Author:** Your Name
**Date:** 2024-01-XX
**Status:** In Progress / Complete

## Objective
Brief description of research goal and hypothesis

## Research Question
What specific question are we trying to answer?
```

#### 2. Methodology Section
```markdown
## Methodology

### Data Sources
- List data sources used
- Time periods covered
- Any data preprocessing steps

### Models/Algorithms
- Description of models used
- Key parameters
- Justification for approach
```

#### 3. Results Section
```markdown
## Results

### Key Findings
- Finding 1: [description with supporting evidence]
- Finding 2: [description with supporting evidence]

### Visualizations
[Include charts with clear labels and explanations]
```

#### 4. Conclusions Section
```markdown
## Conclusions

### Summary
[Concise summary of results]

### Implications
[What do these results mean for trading/investing?]

### Next Steps
[Suggested follow-up research or improvements]

## References
[Papers, articles, or resources referenced]
```

### Code Organization Best Practices

#### Keep Notebooks Clean
- **DO**: Focus notebooks on analysis, visualization, and narrative
- **DON'T**: Put complex logic directly in notebook cells

#### Abstract Core Logic
Create separate `.py` files for reusable functions:

```python
# src/Research_Notebooks/price_forecasting/models.py
def train_model(X, y, **kwargs):
    """Reusable model training function."""
    # Complex logic here
    return trained_model

# In your notebook:
from models import train_model
model = train_model(X_train, y_train)
```

#### Use Type Hints and Docstrings
```python
from typing import Tuple
import pandas as pd

def prepare_data(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for modeling.

    Args:
        df: Input dataframe
        target: Name of target column

    Returns:
        Tuple of (features, target)
    """
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
```

### Data Management

#### Use FinancialData Class
```python
# GOOD: Using abstraction layer
from src.financial_data import FinancialData
fin_data = FinancialData()
data = fin_data.get_balance_sheet('AAPL')

# AVOID: Hardcoding S3 paths
# data = s3.s3_read_parquet('balance/AAPL/balance.parq')  # Don't do this
```

#### Cache Expensive Operations
```python
import pickle
from pathlib import Path

cache_file = Path('data/cached_results.pkl')

if cache_file.exists():
    results = pickle.load(open(cache_file, 'rb'))
else:
    # Expensive computation
    results = expensive_computation()
    pickle.dump(results, open(cache_file, 'wb'))
```

#### Clean Up Before Pushing
Before pushing research to main:
- Remove unnecessary intermediate files
- Delete test plots and temporary outputs
- Condense notebook to essential content
- Ensure notebook runs from top to bottom
- Clear all output cells and re-run

## Development Workflow

### Branch Strategy

- **main** - Only finalized, reviewed, and approved work
  - Should always be stable
  - All code must be reviewed before merging
  - Research must be presented and approved
- **Feature branches** - Individual development work
  - Create from latest main
  - Use descriptive names
  - Delete after merging

### Branch Naming Convention

```bash
# Format: <username>_<feature_description>
# OR
# Format: feature/<description>

# Examples:
git checkout -b john_portfolio_optimization
git checkout -b feature/risk-parity-model
git checkout -b sarah_data_pipeline_fix
```

### Making Changes: Step-by-Step

#### 1. Start with Latest Main
```bash
# Ensure you're on main
git checkout main

# Pull latest changes
git pull origin main
```

#### 2. Create Feature Branch
```bash
# Create and switch to new branch
git checkout -b your_username_feature_name
```

#### 3. Make Changes and Commit
```bash
# Check what changed
git status

# Stage specific files (preferred over git add -A)
git add src/financial_data.py
git add src/Research_Notebooks/new_research/

# Commit with clear message
git commit -m "Add portfolio optimization research

- Implemented mean-variance optimization
- Added Sharpe ratio calculations
- Documented results in notebook

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

#### 4. Push to Remote
```bash
# First push of new branch
git push -u origin your_username_feature_name

# Subsequent pushes
git push
```

#### 5. Open Pull Request
- Go to GitHub repository
- Click "Pull requests" > "New pull request"
- Select your branch
- Fill in PR template (see below)
- Tag all team members for review
- Submit PR

#### 6. Address Feedback
```bash
# Make requested changes
# Commit changes
git add .
git commit -m "Address PR feedback: fix data validation"
git push
```

#### 7. Merge After Approval
- Wait for approvals from reviewers
- Ensure all checks pass
- Merge via GitHub UI (or command line if preferred)
- Delete branch after merge

```bash
# After merge, update local main
git checkout main
git pull origin main

# Delete local feature branch
git branch -d your_username_feature_name
```

### Commit Message Guidelines

#### Format
```
Short summary (50 chars or less)

More detailed explanation if needed. Wrap at 72 characters.
- Bullet points for multiple changes
- Use present tense ("Add feature" not "Added feature")
- Reference issues if applicable (#123)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
```

#### Good Examples
```bash
git commit -m "Add Kelly Criterion position sizing model"

git commit -m "Fix S3 connection timeout issue

- Increase boto3 timeout to 60 seconds
- Add retry logic for transient failures
- Update error messages for clarity"

git commit -m "Refactor price forecasting models

Extract common forecasting logic into base class.
This will make it easier to add new forecasting models.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Pull Request Guidelines

#### PR Title
- Clear and descriptive
- Summarize the change in one line
- Use imperative mood

Examples:
- "Add momentum strategy backtesting framework"
- "Fix data validation in FinancialData class"
- "Improve documentation for S3IO methods"

#### PR Description Template
```markdown
## Summary
Brief description of what this PR does and why.

## Changes
- Change 1: Description
- Change 2: Description
- Change 3: Description

## Testing
How was this tested?
- [ ] Unit tests pass
- [ ] Manually tested with [describe scenario]
- [ ] Reviewed by [team member]

## Related Issues
Closes #123 (if applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Documentation updated (if needed)
- [ ] Tests added/updated (if applicable)
- [ ] All commits have clear messages
- [ ] Ready for review
```

### Code Review Process

#### For Reviewers
When reviewing PRs:
- **Check correctness**: Does the code work as intended?
- **Check clarity**: Is the code easy to understand?
- **Check performance**: Are there any obvious performance issues?
- **Check documentation**: Are docstrings and comments adequate?
- **Be constructive**: Provide specific, actionable feedback
- **Approve or request changes**: Don't leave PRs hanging

#### Review Checklist
- [ ] Code logic is correct
- [ ] Variable/function names are descriptive
- [ ] Complex logic is commented
- [ ] No hardcoded values (use constants/config)
- [ ] Error handling is appropriate
- [ ] No sensitive data exposed
- [ ] Documentation is updated
- [ ] Tests are included (if applicable)

#### For PR Authors
- Respond to all comments
- Ask questions if feedback is unclear
- Make requested changes or explain why not
- Mark conversations as resolved after addressing
- Re-request review after making changes

### Research Review Process

#### Presentation Guidelines
Research must be presented in bi-weekly standups before merging:

1. **Prepare Presentation** (10-15 minutes)
   - Slides or notebook walkthrough
   - Clear objective and methodology
   - Key findings and visualizations
   - Conclusions and implications

2. **Present to Team**
   - Explain your research question
   - Walk through methodology
   - Show results with visualizations
   - Discuss implications

3. **Answer Questions**
   - Be prepared to explain methodology choices
   - Discuss limitations and caveats
   - Address concerns about data or models

4. **Get Approval**
   - Team votes on whether research is ready
   - Address any concerns or requests for changes
   - Merge after approval

#### Before Presenting
- [ ] Notebook runs from top to bottom without errors
- [ ] All visualizations have clear labels and titles
- [ ] Results are clearly documented
- [ ] Conclusions are well-supported
- [ ] Code is clean and well-organized
- [ ] Unnecessary files removed

## Data Access & S3 Structure

### S3 Bucket Organization

```
s3://[your-bucket-name]/
├── stock_tracker/
│   └── tickers.parq              # Master list of all available tickers
│                                 # Columns: Symbol, Name, Market Cap, Country,
│                                 #          IPO Year, Sector, Industry, etc.
├── balance/
│   ├── AAPL/
│   │   └── balance.parq          # Apple Inc. balance sheet data
│   ├── MSFT/
│   │   └── balance.parq          # Microsoft Corp. balance sheet data
│   ├── GOOGL/
│   │   └── balance.parq          # Alphabet Inc. balance sheet data
│   └── [TICKER]/
│       └── balance.parq          # Balance sheet for each ticker
├── income/
│   ├── AAPL/
│   │   └── income.parq           # Apple Inc. income statement
│   ├── MSFT/
│   │   └── income.parq           # Microsoft Corp. income statement
│   └── [TICKER]/
│       └── income.parq           # Income statement for each ticker
└── cash/
    ├── AAPL/
    │   └── cash.parq             # Apple Inc. cash flow statement
    ├── MSFT/
    │   └── cash.parq             # Microsoft Corp. cash flow statement
    └── [TICKER]/
        └── cash.parq             # Cash flow statement for each ticker
```

### Data Access Patterns

#### FinancialData Class (Recommended)
The `FinancialData` class provides a clean abstraction:
- Automatically handles S3 paths
- Validates ticker symbols
- Caches ticker list for performance
- Supports both Polars and Pandas output

```python
from src.financial_data import FinancialData

fin_data = FinancialData()

# Methods available:
# - list_tickers()
# - get_balance_sheet(ticker)
# - get_income_statement(ticker)
# - get_cash_flow(ticker)
# - get_historical_stock_prices(ticker)
```

#### S3IO Class (Advanced)
Direct S3 access for custom scenarios:
- Full control over S3 operations
- Custom file paths
- Batch operations

```python
from src.s3io import S3IO

s3 = S3IO()

# Methods available:
# - s3_list(path)
# - s3_read_parquet(file_path)
# - s3_write_parquet(df, file_path)
# - s3_is_dir(path)
```

### Data Availability

#### Financial Statements
- **Source**: Stored in S3 bucket
- **Format**: Parquet files (optimized for analytics)
- **Update Frequency**: [Specify frequency]
- **Coverage**: [Specify tickers/time range]
- **Data Quality**: Filtered to current data only (`is_current = True`)

#### Historical Prices
- **Source**: Yahoo Finance API (via `yfinance`)
- **Format**: Pandas/Polars DataFrame
- **Update Frequency**: Real-time (fetched on demand)
- **Coverage**: All tickers available on Yahoo Finance
- **Data Fields**: Open, High, Low, Close, Volume, Dividends, Stock Splits

#### Ticker Metadata
- **Source**: S3 `stock_tracker/tickers.parq`
- **Update Frequency**: [Specify frequency]
- **Fields**: Symbol, Name, Market Cap, Country, IPO Year, Sector, Industry, Market Cap Name

### Data Limitations

- Historical data availability varies by ticker
- Some tickers may have incomplete financial statements
- Yahoo Finance API has rate limits
- S3 data is point-in-time (check update timestamps)

## API Reference

### FinancialData Class

The main interface for accessing financial data.

#### Initialization

```python
from src.financial_data import FinancialData

fin_data = FinancialData()
```

Uses environment variables `S3_ARB_BUCKET` and `S3_PROFILE`.

#### Methods

##### `list_tickers(as_pandas=False)`
List all available ticker symbols.

**Parameters:**
- `as_pandas` (bool): Return pandas DataFrame if True, polars if False. Default: False

**Returns:**
- `pl.DataFrame` or `pd.DataFrame`: Ticker information

**Example:**
```python
# Get as Polars DataFrame
tickers = fin_data.list_tickers()

# Get as Pandas DataFrame
tickers_pd = fin_data.list_tickers(as_pandas=True)
```

##### `get_balance_sheet(ticker, as_pandas=False)`
Retrieve balance sheet data for a ticker.

**Parameters:**
- `ticker` (str): Ticker symbol (case-insensitive)
- `as_pandas` (bool): Return pandas DataFrame if True. Default: False

**Returns:**
- `pl.DataFrame` or `pd.DataFrame`: Balance sheet data

**Raises:**
- `ValueError`: If ticker not found

**Example:**
```python
balance = fin_data.get_balance_sheet('AAPL')
balance_pd = fin_data.get_balance_sheet('msft', as_pandas=True)
```

##### `get_income_statement(ticker, as_pandas=False)`
Retrieve income statement data for a ticker.

**Parameters:**
- `ticker` (str): Ticker symbol (case-insensitive)
- `as_pandas` (bool): Return pandas DataFrame if True. Default: False

**Returns:**
- `pl.DataFrame` or `pd.DataFrame`: Income statement data

**Raises:**
- `ValueError`: If ticker not found

**Example:**
```python
income = fin_data.get_income_statement('GOOGL')
```

##### `get_cash_flow(ticker, as_pandas=False)`
Retrieve cash flow statement data for a ticker.

**Parameters:**
- `ticker` (str): Ticker symbol (case-insensitive)
- `as_pandas` (bool): Return pandas DataFrame if True. Default: False

**Returns:**
- `pl.DataFrame` or `pd.DataFrame`: Cash flow data

**Raises:**
- `ValueError`: If ticker not found

**Example:**
```python
cash_flow = fin_data.get_cash_flow('TSLA')
```

##### `get_historical_stock_prices(ticker, as_pandas=False, period='max', interval='1d')`
Retrieve historical stock prices from Yahoo Finance.

**Parameters:**
- `ticker` (str): Ticker symbol (case-insensitive)
- `as_pandas` (bool): Return pandas DataFrame if True. Default: False
- `period` (str): Time period. Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'. Default: 'max'
- `interval` (str): Data interval. Options: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'. Default: '1d'

**Returns:**
- `pl.DataFrame` or `pd.DataFrame`: Historical price data

**Raises:**
- `ValueError`: If ticker not found or no data available

**Example:**
```python
# Get all available daily data
prices = fin_data.get_historical_stock_prices('AAPL')

# Get last year of weekly data
prices_weekly = fin_data.get_historical_stock_prices(
    'AMZN',
    period='1y',
    interval='1wk',
    as_pandas=True
)
```

### S3IO Class

Low-level S3 operations wrapper.

#### Initialization

```python
from src.s3io import S3IO

s3 = S3IO()  # Uses environment variables
```

#### Methods

##### `s3_list(path)`
List objects in an S3 path.

**Parameters:**
- `path` (str): S3 path prefix

**Returns:**
- `list`: List of object keys

**Example:**
```python
files = s3.s3_list('balance/')
```

##### `s3_read_parquet(file_path)`
Read a Parquet file from S3.

**Parameters:**
- `file_path` (str): Full S3 path to file

**Returns:**
- `pl.DataFrame`: Data from Parquet file

**Example:**
```python
data = s3.s3_read_parquet('balance/AAPL/balance.parq')
```

##### `s3_write_parquet(df, file_path)`
Write a Parquet file to S3.

**Parameters:**
- `df` (pl.DataFrame): DataFrame to write
- `file_path` (str): Full S3 path for output file

**Example:**
```python
import polars as pl
df = pl.DataFrame({'col': [1, 2, 3]})
s3.s3_write_parquet(df, 'test/output.parq')
```

##### `s3_is_dir(path)`
Check if a path exists in S3.

**Parameters:**
- `path` (str): S3 path to check

**Returns:**
- `bool`: True if path exists, False otherwise

**Example:**
```python
exists = s3.s3_is_dir('balance/')
```

## Testing

### Running Tests

Example test notebook location: `src/Research_Notebooks/Testing/fin_data_test.ipynb`

To run tests:
```bash
# Start Jupyter
jupyter notebook

# Navigate to Testing folder and open test notebooks
```

### Writing Tests

When adding new features, create test notebooks:

```python
# Example test structure
def test_financial_data_class():
    """Test FinancialData class methods."""
    from src.financial_data import FinancialData

    fin_data = FinancialData()

    # Test ticker listing
    tickers = fin_data.list_tickers()
    assert len(tickers) > 0, "Should load tickers"

    # Test balance sheet retrieval
    balance = fin_data.get_balance_sheet('AAPL')
    assert len(balance) > 0, "Should load balance sheet"

    print("All tests passed!")

test_financial_data_class()
```

## Troubleshooting

### Common Issues

#### AWS Credentials Not Found
```
Error: botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Solution:**
1. Check that `~/.aws/credentials` exists and has valid credentials
2. Verify the profile name matches `S3_PROFILE` environment variable
3. Ensure credentials have S3 access permissions

```bash
# Check credentials file
cat ~/.aws/credentials

# Test AWS CLI access
aws s3 ls --profile default
```

#### Environment Variables Not Set
```
Error: KeyError: 'S3_ARB_BUCKET'
```

**Solution:**
1. Verify `.env` file exists and has correct values
2. Source environment variables in your session
3. Restart Python kernel if using Jupyter

```bash
# Check environment variables
echo $S3_ARB_BUCKET
echo $S3_PROFILE

# Set them if missing
export S3_ARB_BUCKET="your-bucket-name"
export S3_PROFILE="default"
```

#### Import Errors
```
Error: ModuleNotFoundError: No module named 'polars'
```

**Solution:**
1. Ensure virtual environment is activated
2. Reinstall requirements
3. Check Python version compatibility

```bash
# Activate virtual environment
source venv/bin/activate

# Reinstall requirements
pip install -r requirements.txt

# Verify installation
python -c "import polars; print(polars.__version__)"
```

#### S3 Access Denied
```
Error: botocore.exceptions.ClientError: An error occurred (403) when calling the GetObject operation: Forbidden
```

**Solution:**
1. Verify IAM permissions for S3 bucket
2. Check bucket name is correct
3. Ensure credentials belong to correct AWS account

Required IAM permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket",
        "s3:PutObject"
      ],
      "Resource": [
        "arn:aws:s3:::your-bucket-name/*",
        "arn:aws:s3:::your-bucket-name"
      ]
    }
  ]
}
```

#### Ticker Not Found
```
Error: ValueError: Ticker 'XYZ' not found in available tickers
```

**Solution:**
1. Verify ticker symbol spelling
2. Check if ticker exists in the S3 bucket
3. Try listing all tickers to see available symbols

```python
fin_data = FinancialData()
tickers = fin_data.list_tickers()
print(tickers['Symbol'].unique())  # See all available tickers
```

#### Package Version Conflicts
```
Error: ImportError: cannot import name 'xyz' from 'package'
```

**Solution:**
1. Create fresh virtual environment
2. Update pip and setuptools
3. Install specific package versions

```bash
# Create fresh environment
python3 -m venv venv_new
source venv_new/bin/activate

# Update pip
pip install --upgrade pip setuptools

# Install requirements
pip install -r requirements.txt
```

### Getting Help

If you encounter issues not covered here:

1. **Check existing issues** on GitHub
2. **Search documentation** in `src/claude/` folder
3. **Ask the team** in standup meetings or Slack
4. **Create a GitHub issue** with:
   - Clear description of the problem
   - Steps to reproduce
   - Error messages (full traceback)
   - Your environment (OS, Python version, etc.)

## Contributing

We welcome contributions from all team members! See below for guidelines.

### Coding Standards

#### Python Style Guide
Follow [PEP 8](https://pep8.org/) conventions:
- Use 4 spaces for indentation (not tabs)
- Max line length: 100 characters (flexible for readability)
- Use snake_case for functions and variables
- Use PascalCase for classes
- Use UPPER_CASE for constants

#### Type Hints
Use type hints for function signatures:
```python
from typing import List, Dict, Optional
import pandas as pd

def process_data(
    df: pd.DataFrame,
    columns: List[str],
    threshold: Optional[float] = None
) -> pd.DataFrame:
    """Process dataframe with specified columns."""
    # Implementation
    return df
```

#### Docstrings
Use NumPy-style docstrings:
```python
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for a return series.

    Parameters
    ----------
    returns : pd.Series
        Series of investment returns
    risk_free_rate : float, default=0.0
        Risk-free rate for Sharpe calculation

    Returns
    -------
    float
        Calculated Sharpe ratio

    Examples
    --------
    >>> returns = pd.Series([0.01, 0.02, -0.01, 0.03])
    >>> calculate_sharpe_ratio(returns)
    1.2247
    """
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std()
```

#### Code Formatting
Consider using automated formatters:
```bash
# Install formatters
pip install black flake8 mypy

# Format code
black src/

# Check style
flake8 src/

# Check types
mypy src/
```

### Testing Requirements

- Add tests for new functionality
- Ensure existing tests still pass
- Document test scenarios in notebooks
- Include edge cases and error conditions

### Documentation Requirements

When contributing:
- Update docstrings for new functions/classes
- Update README if adding major features
- Document research methodology in notebooks
- Add comments for complex logic

### Pull Request Checklist

Before submitting a PR:
- [ ] Code follows style guidelines
- [ ] Type hints added to new functions
- [ ] Docstrings added/updated
- [ ] Tests added for new functionality
- [ ] Documentation updated (if needed)
- [ ] Notebook runs from top to bottom
- [ ] Commit messages are clear
- [ ] PR description is complete
- [ ] All team members tagged for review

## Additional Resources

### Documentation
- [FinancialData Class Design](src/claude/fin_data.md) - Detailed design document
- [Original README Plan](src/claude/readme_plan.md) - Initial documentation planning
- [Comprehensive Plan](claude/readme_planII.md) - Full documentation roadmap

### AWS & S3
- [AWS SDK Credentials Setup](https://docs.aws.amazon.com/sdk-for-java/v1/developer-guide/setup-credentials.html)
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [AWS S3 User Guide](https://docs.aws.amazon.com/s3/index.html)

### Python Libraries
- [Polars Documentation](https://pola-rs.github.io/polars/py-polars/html/reference/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [yfinance Documentation](https://pypi.org/project/yfinance/)

### Finance & Trading
- [Quantitative Finance Resources](https://www.quantstart.com/)
- [Portfolio Optimization Theory](https://www.investopedia.com/terms/m/meanvariance-analysis.asp)

### Git & GitHub
- [Git Documentation](https://git-scm.com/doc)
- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [Writing Good Commit Messages](https://chris.beams.io/posts/git-commit/)

## License

[Specify license here - e.g., MIT, Apache 2.0, Proprietary]

## Team & Contact

**Repository Maintainers:**
- [List team members and roles]

**Getting Help:**
- GitHub Issues: [Create an issue](https://github.com/ttoe05/beat-the-market-research/issues)
- Team Standups: Bi-weekly meetings
- Slack/Communication Channel: [If applicable]

## Acknowledgments

This project uses several open-source libraries:
- **Polars** - High-performance DataFrame library
- **Pandas** - Data analysis library
- **Boto3** - AWS SDK for Python
- **yfinance** - Yahoo Finance API wrapper
- **scikit-learn** - Machine learning library

---

**Last Updated:** January 2026

For questions or suggestions about this documentation, please open an issue or contact the team.
