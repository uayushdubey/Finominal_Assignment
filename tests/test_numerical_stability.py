import pytest
import numpy as np
import pandas as pd
from repositories.data_repository import DataRepository
from utils.financial import (
    calculate_covariance_matrix,
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio
)
from domain.portfolio import Portfolio, Security, Constraints
from strategies.portfolio_strategies import StrategyFactory
from core.exceptions import ConstraintViolationException, InvalidInputException

def test_data_processing_and_conversion(tmp_path):
    # Create messy mock price data
    price_data = {
        "date": ["2026-05-02", "2026-05-01", "2026-05-05", "2026-05-04", "2026-05-03"],
        "AAPL": [150.0, np.nan, 155.0, 153.0, 152.0],
        "MSFT": [300.0, 298.0, np.nan, 305.0, 302.0]
    }
    df = pd.DataFrame(price_data)
    prices_file = tmp_path / "historical_prices.csv"
    df.to_csv(prices_file, index=False)
    
    # Write empty metadata and factor file to prevent dependency crashes
    metadata_file = tmp_path / "fund_metadata.csv"
    pd.DataFrame({"ticker": ["AAPL", "MSFT"], "name": ["Apple", "Microsoft"], "dividend_yield": [0.5, 0.7]}).to_csv(metadata_file, index=False)
    factors_file = tmp_path / "factor_returns.csv"
    pd.DataFrame({"date": ["2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05"], "momentum": [0.01, 0.02, 0.03, 0.04], "value": [0.0, 0.0, 0.0, 0.0], "size": [0.0, 0.0, 0.0, 0.0]}).to_csv(factors_file, index=False)
    
    repo = DataRepository(data_dir=str(tmp_path))
    
    # Get returns and verify sorting, filling, pct_change, dropna
    returns = repo.get_returns(["AAPL", "MSFT"])
    assert isinstance(returns, pd.DataFrame)
    assert not returns.isnull().any().any()
    assert returns.index.is_monotonic_increasing

def test_covariance_matrix_regularization():
    # Large dataset should not trigger regularization warning in length but let's test short dataset behavior
    returns = pd.DataFrame({
        "AAPL": [0.01, -0.01, 0.02],
        "MSFT": [-0.005, 0.01, 0.005]
    })
    # Since len(returns) is 3 (< 252), it will apply regularization (+ 1e-6 * eye)
    cov_reg = calculate_covariance_matrix(returns, annualize=False)
    
    # Calculate base cov without regularization
    cov_base = returns.cov()
    
    # Regularized cov should equal base cov plus 1e-6 on diagonal
    np.testing.assert_allclose(cov_reg.values, cov_base.values + np.eye(2) * 1e-6, atol=1e-12)

def test_covariance_matrix_validation_failures():
    # Covariance containing NaNs should raise ValueError
    invalid_returns = pd.DataFrame({
        "AAPL": [0.01, np.nan, 0.02],
        "MSFT": [-0.005, 0.01, 0.005]
    })
    with pytest.raises(ValueError, match="Covariance matrix contains NaN values"):
        calculate_covariance_matrix(invalid_returns)

def test_portfolio_metrics_vectorization():
    returns_df = pd.DataFrame({
        "AAPL": [0.01, 0.02, -0.01],
        "MSFT": [0.005, -0.005, 0.01]
    })
    factor = len(returns_df) if len(returns_df) < 30 else 252
    cov_matrix = returns_df.cov().values * factor
    
    weights = [0.6, 0.4]
    
    ret = portfolio_return(weights, returns_df)
    vol = portfolio_volatility(weights, cov_matrix)
    sr = sharpe_ratio(weights, returns_df, cov_matrix)
    
    assert isinstance(ret, float)
    assert isinstance(vol, float)
    assert isinstance(sr, float)
    
    # Zero weights
    assert portfolio_return([0.0, 0.0], returns_df) == 0.0
    assert portfolio_volatility([0.0, 0.0], cov_matrix) == 0.0
    assert sharpe_ratio([0.0, 0.0], returns_df, cov_matrix) == 0.0

def test_risk_parity_zero_volatility_safety():
    # Mock return history with 300 identical rows so no small-dataset regularization is triggered
    returns_df = pd.DataFrame({
        "IEFA": np.zeros(300),
        "GLD": np.zeros(300)
    })
    portfolio = Portfolio(securities=[Security("IEFA", 50.0), Security("GLD", 50.0)])
    strategy = StrategyFactory.get_strategy("risk_parity")
    
    with pytest.raises(ValueError, match="Portfolio volatility is zero. Optimization is unstable."):
        strategy.optimize(portfolio, returns_df)

def test_post_optimization_dividend_constraint(tmp_path):
    # Mock data repo and service pipeline to test constraint infeasibility check post-optimization
    # Let's create returns where GLD and IEFA returns are equal but metadata forces target minimum yield high
    price_data = {
        "date": ["2026-05-01", "2026-05-02", "2026-05-03"],
        "IEFA": [100.0, 101.0, 102.0],
        "GLD": [100.0, 101.0, 102.0]
    }
    metadata = {
        "ticker": ["IEFA", "GLD"],
        "name": ["iShares EAFE", "Gold Shares"],
        "dividend_yield": [3.28, 0.0]  # Matches real file yields
    }
    factors = {
        "date": ["2026-05-02", "2026-05-03"],
        "momentum": [0.01, 0.02],
        "value": [0.0, 0.0],
        "size": [0.0, 0.0]
    }
    
    pd.DataFrame(price_data).to_csv(tmp_path / "historical_prices.csv", index=False)
    pd.DataFrame(metadata).to_csv(tmp_path / "fund_metadata.csv", index=False)
    pd.DataFrame(factors).to_csv(tmp_path / "factor_returns.csv", index=False)
    
    repo = DataRepository(data_dir=str(tmp_path))
    returns_data = repo.get_returns(["IEFA", "GLD"])
    
    portfolio = Portfolio(
        securities=[Security("IEFA", 50.0), Security("GLD", 50.0)],
        # Expecting at least 5.0% dividend yield, which is impossible (since max is 3.28%)
        constraints=Constraints(min_weight=0.0, max_weight=100.0, min_dividend_yield=5.0)
    )
    
    strategy = StrategyFactory.get_strategy("min_volatility")
    # Pre-optimization constraint audit or post-optimization audit should catch this and raise ConstraintViolationException
    with pytest.raises(ConstraintViolationException):
         strategy.optimize(portfolio, returns_data)
