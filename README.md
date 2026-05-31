# Portfolio Optimizer API

A professional and clean FastAPI backend designed for multi-strategy quantitative portfolio optimization and risk factor exposure analysis.

This project is hosted on GitHub: [github.com/uayushdubey/Finominal_Assignment](https://github.com/uayushdubey/Finominal_Assignment)

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
    {"ticker": "AAPL", "weight": 25.0},
    {"ticker": "MSFT", "weight": 25.0},
    {"ticker": "GOOGL", "weight": 25.0},
    {"ticker": "AMZN", "weight": 25.0}
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
      "ticker": "AAPL",
      "name": "Apple Inc.",
      "current_weight": 25.0,
      "optimized_weight": 27.94,
      "change": 2.94
    },
    {
      "ticker": "MSFT",
      "name": "Microsoft Corporation",
      "current_weight": 25.0,
      "optimized_weight": 33.56,
      "change": 8.56
    },
    {
      "ticker": "GOOGL",
      "name": "Alphabet Inc.",
      "current_weight": 25.0,
      "optimized_weight": 25.88,
      "change": 0.88
    },
    {
      "ticker": "AMZN",
      "name": "Amazon.com Inc.",
      "current_weight": 25.0,
      "optimized_weight": 12.63,
      "change": -12.37
    }
  ],
  "metrics": {
    "expected_return": 0.0381,
    "expected_volatility": 0.0175,
    "sharpe_ratio": 2.1757
  },
  "factor_exposure": {
    "alpha": 0.0465,
    "beta_momentum": -0.1729,
    "beta_value": -1.1546,
    "beta_size": -4.9748,
    "r_squared": 0.7351,
    "t_stats": {
      "const": 3.3452,
      "momentum": -0.1306,
      "value": -1.0592,
      "size": -2.942
    },
    "p_values": {
      "const": 0.0155,
      "momentum": 0.9004,
      "value": 0.3303,
      "size": 0.0259
    }
  }
}
```
