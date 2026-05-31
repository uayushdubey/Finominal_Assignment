# Portfolio Optimizer API

A professional and clean FastAPI backend designed for multi-strategy quantitative portfolio optimization and risk factor exposure analysis.

---

## Key Features

1. **Portfolio Optimization**
   Supports multiple optimization strategies to rebalance asset weights based on your target:
   * **Equal Weight (`equal_weight`)**: Distributes weights equally among all assets.
   * **Minimum Volatility (`min_volatility` / `min_variance`)**: Minimizes portfolio risk/volatility using SciPy SLSQP.
   * **Maximum Sharpe (`max_sharpe` / `mean_variance`)**: Maximizes risk-adjusted return (Sharpe Ratio).
   * **Risk Parity (`risk_parity`)**: Allocates weight so that each asset contributes equally to overall risk.
   * **Minimum Drawdown (`min_drawdown`)**: Minimizes the maximum historical drawdown of the portfolio.

2. **Smart Constraint Checking**
   Before running heavy optimization solvers, the system performs a quick analytical audit of constraints (e.g., minimum dividend yield, min/max asset weights) to see if they are mathematically feasible. If they are infeasible, it returns a clear error response immediately instead of failing silently or crashing.

3. **Factor Exposure Regression**
   Automatically runs an Ordinary Least Squares (OLS) regression using `statsmodels` after optimization to measure how sensitive the portfolio is to key risk factors (Value, Size, Momentum) and provides alpha, beta, R-squared, and significance metrics (p-values & t-stats).

---

## Project Structure

The project follows Clean Architecture principles:
* **`api/`**: Thin FastAPI routers defining the HTTP endpoints.
* **`core/`**: Centralized configurations (via Pydantic Settings), logging, and global exception handlers.
* **`domain/`**: Pure entities (Portfolio, Security, Constraints) holding business rules.
* **`schemas/`**: Pydantic models verifying request/response payloads.
* **`services/`**: Orchestration logic (Portfolio and Factor services).
* **`strategies/`**: Implementations of various optimization strategies (SciPy, custom).
* **`repositories/`**: Handles loading mock price and factor data from CSV files located in `data/`.
* **`tests/`**: A comprehensive test suite with 35 test cases validating endpoints, strategies, and constraint solvers.

---

## Installation & Setup

You can run this project locally using **Python (pip)** or **Docker**.

### Option A: Local Installation

#### 1. Prerequisites
Ensure you have **Python 3.10+** installed on your system.

#### 2. Clone the Repository
```bash
git clone https://github.com/uayushdubey/Finominal_Assignment.git
cd Finominal_Assignment
```

#### 3. Setup Virtual Environment
Create and activate a virtual environment to keep dependencies isolated:

* **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .venv\Scripts\activate
  ```
* **macOS / Linux**:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

#### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 5. Configuration
Copy the sample environment variables file to create your own configuration:
```bash
cp .env.example .env
```

#### 6. Run the Server
Start the development server using Uvicorn:
```bash
uvicorn main:app --reload
```
* **Interactive API Documentation (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **API Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

### Option B: Run with Docker

If you have Docker installed, you can containerize and run the application easily:

#### 1. Build the Docker Image
```bash
docker build -t portfolio-optimizer .
```

#### 2. Run the Container
```bash
docker run -p 8000:8000 portfolio-optimizer
```
Access the Swagger documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Running Tests

To verify that everything is working correctly, you can run the test suite using `pytest`:

```bash
pytest -v
```

---

## API Reference & Example Payloads

### POST `/api/v1/optimize`

Optimizes a list of assets based on your constraints and strategy.

#### Request Example
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/v1/optimize' \
  -H 'Content-Type: application/json' \
  -d '{
  "assets": [
    {"ticker": "IEFA", "weight": 20.0},
    {"ticker": "GLD", "weight": 20.0},
    {"ticker": "AGG", "weight": 20.0},
    {"ticker": "VEA", "weight": 20.0},
    {"ticker": "SPY", "weight": 20.0}
  ],
  "constraints": {
    "min_weight": 10.0,
    "max_weight": 50.0,
    "min_dividend_yield": 0.5
  },
  "strategy": "min_volatility"
}'
```

#### Response Example (200 OK)
```json
{
  "assets": [
    {
      "ticker": "IEFA",
      "name": "iShares Core MSCI EAFE ETF",
      "current_weight": 20.0,
      "optimized_weight": 20.0,
      "change": 0.0
    },
    {
      "ticker": "GLD",
      "name": "SPDR Gold Shares",
      "current_weight": 20.0,
      "optimized_weight": 20.0,
      "change": 0.0
    },
    {
      "ticker": "AGG",
      "name": "iShares Core US Aggregate Bond ETF",
      "current_weight": 20.0,
      "optimized_weight": 20.0,
      "change": 0.0
    },
    {
      "ticker": "VEA",
      "name": "Vanguard Developed Markets Index Fund;ETF",
      "current_weight": 20.0,
      "optimized_weight": 20.0,
      "change": 0.0
    },
    {
      "ticker": "SPY",
      "name": "State Street SPDR S&P 500 ETF Trust",
      "current_weight": 20.0,
      "optimized_weight": 20.0,
      "change": 0.0
    }
  ],
  "metrics": {
    "expected_return": 0.0244,
    "expected_volatility": 0.0118,
    "sharpe_ratio": 2.0727
  },
  "factor_exposure": {
    "alpha": 0.0246,
    "beta_momentum": 0.6694,
    "beta_value": -0.3221,
    "beta_size": -3.6536,
    "r_squared": 0.6197,
    "t_stats": {
      "const": 2.2047,
      "momentum": 0.6287,
      "value": -0.3674,
      "size": -2.6863
    },
    "p_values": {
      "const": 0.0697,
      "momentum": 0.5527,
      "value": 0.726,
      "size": 0.0362
    }
  }
}
```

---

## Example Scenarios

Below are common testing scenarios demonstrating the optimizer behavior across different strategies and constraints.

### 1. Risk Parity (Base Case)

```json
{
  "assets": [
    {"ticker": "IEFA", "weight": 20.0},
    {"ticker": "GLD", "weight": 20.0},
    {"ticker": "AGG", "weight": 20.0},
    {"ticker": "VEA", "weight": 20.0},
    {"ticker": "SPY", "weight": 20.0}
  ],
  "constraints": {
    "min_weight": 10.0,
    "max_weight": 50.0,
    "min_dividend_yield": 0.5
  },
  "strategy": "risk_parity"
}
```
Output:
<img width="1870" height="617" alt="image" src="https://github.com/user-attachments/assets/caa07b46-5af6-419e-a8f9-1ec092efcb54" />

Expected:
* AGG and VEA weights increase.
* Risk is balanced across assets.

### 2. Max Sharpe (High Value Test)

```json
{
  "assets": [
    {"ticker": "IEFA", "weight": 20.0},
    {"ticker": "GLD", "weight": 20.0},
    {"ticker": "AGG", "weight": 20.0},
    {"ticker": "VEA", "weight": 20.0},
    {"ticker": "SPY", "weight": 20.0}
  ],
  "constraints": {
    "min_weight": 5.0,
    "max_weight": 50.0
  },
  "strategy": "max_sharpe"
}
```

Output:

<img width="1886" height="744" alt="image" src="https://github.com/user-attachments/assets/dd1aa5e5-01d3-44b7-b366-071bca5b3791" />


Expected:
* Better performing assets receive more weight.
* Weight distribution is not equal.

### 3. Minimize Volatility (Safe Portfolio)

```json
{
  "assets": [
    {"ticker": "SPY", "weight": 50.0},
    {"ticker": "AGG", "weight": 30.0},
    {"ticker": "GLD", "weight": 20.0}
  ],
  "constraints": {
    "min_weight": 0.0,
    "max_weight": 70.0
  },
  "strategy": "min_volatility"
}
```
Output:

<img width="1870" height="725" alt="image" src="https://github.com/user-attachments/assets/206ba7e8-981f-4a6c-9ab1-69aaec6dccef" />


Expected:
* AGG weight increases (low risk asset).
* SPY weight decreases.


### 5. Bonus (Error Test)

```json
{
  "assets": [
    {"ticker": "SPY", "weight": 80.0},
    {"ticker": "AGG", "weight": 30.0}
  ],
  "strategy": "risk_parity"
}
```

Output:
<img width="1865" height="473" alt="image" src="https://github.com/user-attachments/assets/9f6cfd2d-d529-45a6-bdf4-520dbf953ffd" />


Expected:
* Triggers validation or constraint error due to invalid weights (e.g. sum of initial weights is not equal to 100% or fails validation).

