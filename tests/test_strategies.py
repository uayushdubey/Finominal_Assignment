import pytest
import pandas as pd
from domain.portfolio import Portfolio, Security, Constraints
from repositories.data_repository import DataRepository
from strategies.portfolio_strategies import StrategyFactory
from core.exceptions import StrategyNotSupportedException, ConstraintViolationException

@pytest.fixture(scope="module")
def returns_data():
    repo = DataRepository()
    return repo.get_returns(["IEFA", "GLD", "AGG", "VEA", "SPY"])

def test_strategy_factory_lookup():
    eq = StrategyFactory.get_strategy("equal_weight")
    assert eq.__class__.__name__ == "EqualWeightStrategy"
    
    min_vol = StrategyFactory.get_strategy("min_volatility")
    assert min_vol.__class__.__name__ == "MinVolatilityStrategy"
    
    with pytest.raises(StrategyNotSupportedException):
        StrategyFactory.get_strategy("quantum_optimizer")

def test_equal_weight_strategy(returns_data):
    tickers = ["IEFA", "GLD", "AGG", "VEA", "SPY"]
    securities = [Security(t, 0.0) for t in tickers]
    portfolio = Portfolio(securities=securities)
    
    strategy = StrategyFactory.get_strategy("equal_weight")
    res = strategy.optimize(portfolio, returns_data)
    
    assert len(res.optimized_weights) == 5
    for t in tickers:
        assert abs(res.optimized_weights[t] - 20.0) < 1e-4
    assert abs(sum(res.optimized_weights.values()) - 100.0) < 1e-4

def test_min_volatility_strategy(returns_data):
    tickers = ["IEFA", "GLD", "AGG", "VEA", "SPY"]
    securities = [Security(t, 0.0) for t in tickers]
    portfolio = Portfolio(securities=securities)
    
    strategy = StrategyFactory.get_strategy("min_volatility")
    res = strategy.optimize(portfolio, returns_data)
    
    assert abs(sum(res.optimized_weights.values()) - 100.0) < 1e-4
    assert res.expected_volatility > 0.0

def test_max_sharpe_strategy(returns_data):
    tickers = ["IEFA", "GLD", "AGG", "VEA", "SPY"]
    securities = [Security(t, 0.0) for t in tickers]
    portfolio = Portfolio(securities=securities)
    
    strategy = StrategyFactory.get_strategy("max_sharpe")
    res = strategy.optimize(portfolio, returns_data)
    
    assert abs(sum(res.optimized_weights.values()) - 100.0) < 1e-4
    assert res.sharpe_ratio > 0.0

def test_risk_parity_strategy(returns_data):
    tickers = ["IEFA", "GLD", "AGG", "VEA", "SPY"]
    securities = [Security(t, 0.0) for t in tickers]
    portfolio = Portfolio(securities=securities)
    
    strategy = StrategyFactory.get_strategy("risk_parity")
    res = strategy.optimize(portfolio, returns_data)
    
    assert abs(sum(res.optimized_weights.values()) - 100.0) < 1e-4

def test_min_drawdown_strategy(returns_data):
    tickers = ["IEFA", "GLD", "AGG", "VEA", "SPY"]
    securities = [Security(t, 0.0) for t in tickers]
    portfolio = Portfolio(securities=securities)
    
    strategy = StrategyFactory.get_strategy("min_drawdown")
    res = strategy.optimize(portfolio, returns_data)
    
    assert abs(sum(res.optimized_weights.values()) - 100.0) < 1e-4

def test_strategy_bounds_constraints(returns_data):
    tickers = ["IEFA", "GLD", "AGG", "VEA", "SPY"]
    securities = [Security(t, 0.0) for t in tickers]
    # Restrict weights between 15% and 40%
    constraints = Constraints(min_weight=15.0, max_weight=40.0)
    portfolio = Portfolio(securities=securities, constraints=constraints)
    
    for name in ["min_volatility", "max_sharpe", "risk_parity", "min_drawdown"]:
        strategy = StrategyFactory.get_strategy(name)
        res = strategy.optimize(portfolio, returns_data)
        
        assert abs(sum(res.optimized_weights.values()) - 100.0) < 1e-4
        for t, w in res.optimized_weights.items():
            assert w >= 15.0 - 1e-4
            assert w <= 40.0 + 1e-4

def test_infeasible_dividend_yield_constraint(returns_data):
    tickers = ["IEFA", "GLD", "AGG", "VEA", "SPY"]
    securities = [Security(t, 0.0) for t in tickers]
    # Set impossible dividend yield (max available in dataset is 3.97%)
    constraints = Constraints(min_weight=10.0, max_weight=50.0, min_dividend_yield=5.0)
    portfolio = Portfolio(securities=securities, constraints=constraints)
    
    strategy = StrategyFactory.get_strategy("min_volatility")
    with pytest.raises(ConstraintViolationException):
         strategy.optimize(portfolio, returns_data)
