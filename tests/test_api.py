import pytest

def test_api_health(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app_name"] == "Portfolio Optimizer API"

def test_api_optimize_success(api_client):
    payload = {
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
    }
    
    response = api_client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "assets" in data
    assert "metrics" in data
    assert "factor_exposure" in data
    
    assets = {a["ticker"]: a for a in data["assets"]}
    assert len(assets) == 5
    assert assets["IEFA"]["name"] == "iShares Core MSCI EAFE ETF"
    assert assets["IEFA"]["current_weight"] == 20.0
    
    # Assert change math
    for ticker, asset in assets.items():
        expected_change = asset["optimized_weight"] - asset["current_weight"]
        assert abs(asset["change"] - expected_change) < 1e-4

def test_api_validation_negative_weight(api_client):
    payload = {
        "assets": [
            {"ticker": "IEFA", "weight": -10.0},
            {"ticker": "GLD", "weight": 110.0}
        ],
        "strategy": "min_volatility"
    }
    
    response = api_client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "ValidationError"

def test_api_infeasible_yield_failure(api_client):
    payload = {
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
            "min_dividend_yield": 5.0  # Mathematically impossible
        },
        "strategy": "min_volatility"
    }
    
    response = api_client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "ConstraintViolationException"

def test_api_unsupported_strategy_failure(api_client):
    payload = {
        "assets": [
            {"ticker": "IEFA", "weight": 50.0},
            {"ticker": "GLD", "weight": 50.0}
        ],
        "strategy": "custom_unsupported_optimizer"
    }
    
    response = api_client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "ValidationError"

def test_api_strategy_normalization_success(api_client):
    payload = {
        "assets": [
            {"ticker": "IEFA", "weight": 50.0},
            {"ticker": "GLD", "weight": 50.0}
        ],
        "strategy": "Maximize_Sharpe"
    }
    
    response = api_client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "assets" in data

def test_api_strategy_typo_fuzzy_match_suggestion(api_client):
    payload = {
        "assets": [
            {"ticker": "IEFA", "weight": 50.0},
            {"ticker": "GLD", "weight": 50.0}
        ],
        "strategy": "min_volatilty"  # Typo: missing 'i'
    }
    
    response = api_client.post("/api/v1/optimize", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "ValidationError"
    
    # Locate the error for the strategy field
    errors = data["error"]["details"]["errors"]
    strategy_error = next(err for err in errors if "strategy" in err["field"])
    
    assert strategy_error["did_you_mean"] == "min_volatility"
    assert strategy_error["input"] == "min_volatilty"
    assert "min_volatility" in strategy_error["supported_strategies"]
