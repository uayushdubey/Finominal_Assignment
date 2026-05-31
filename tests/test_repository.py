import pytest
import pandas as pd
from repositories.data_repository import DataRepository
from core.exceptions import NotFoundException, InvalidInputException

def test_repository_get_factor_data(repository):
    factors = repository.get_factor_data()
    assert isinstance(factors, pd.DataFrame)
    assert not factors.empty
    # Ensure no NaN values remain
    assert not factors.isnull().any().any()
    # Intercept must be datetime
    assert isinstance(factors.index, pd.DatetimeIndex)
    assert list(factors.columns) == ["momentum", "value", "size"]

def test_repository_get_dividend_yield_success(repository):
    tickers = ["AAPL", "MSFT", "GOOGL"]
    yields = repository.get_dividend_yield(tickers)
    assert isinstance(yields, dict)
    assert len(yields) == 3
    assert yields["AAPL"] == 0.52
    assert yields["MSFT"] == 0.71

def test_repository_get_dividend_yield_failures(repository):
    # Empty tickers list
    with pytest.raises(InvalidInputException):
        repository.get_dividend_yield([])
    # Non-existent ticker
    with pytest.raises(NotFoundException):
        repository.get_dividend_yield(["AAPL", "NON_EXISTENT"])

def test_repository_get_returns_success(repository):
    tickers = ["AAPL", "MSFT", "GOOGL"]
    returns = repository.get_returns(tickers)
    assert isinstance(returns, pd.DataFrame)
    assert not returns.empty
    assert not returns.isnull().any().any()
    assert list(returns.columns) == tickers
    
    # Check alignment: returns dates should exist in factor returns
    factors = repository.get_factor_data()
    assert returns.index.isin(factors.index).all()

def test_repository_get_returns_failures(repository):
    # Empty tickers list
    with pytest.raises(InvalidInputException):
        repository.get_returns([])
    # Non-existent ticker
    with pytest.raises(NotFoundException):
        repository.get_returns(["AAPL", "NON_EXISTENT"])
