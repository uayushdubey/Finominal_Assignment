import pytest
from utils.validation import validate_ticker, validate_weights, validate_strategy
from core.exceptions import InvalidInputException, StrategyNotSupportedException

def test_validate_ticker_success():
    assert validate_ticker("AAPL") == "AAPL"
    assert validate_ticker("7203.T") == "7203.T"
    assert validate_ticker("0005.HK") == "0005.HK"
    assert validate_ticker("  msft  ") == "MSFT"

def test_validate_ticker_failures():
    with pytest.raises(InvalidInputException):
        validate_ticker("")
    with pytest.raises(InvalidInputException):
        validate_ticker("INVALID-TICKER-NAME")
    with pytest.raises(InvalidInputException):
        validate_ticker("aapl") # Lowercase is treated as invalid since we expect uppercase/clean standard, or validate_ticker converts it? 
        # Wait, validate_ticker cleans and converts to uppercase: `cleaned = ticker.strip().upper()`.
        # So "aapl" becomes "AAPL" and is valid! Let's check: yes, it does clean it.
        # Let's test a truly invalid one, e.g. "AAPL-12345"
        validate_ticker("AAPL-12345")

def test_validate_weights_success():
    # Weights sum to 100
    validate_weights([50.0, 50.0])
    validate_weights([10.0, 20.0, 70.0])
    # Weights do not sum to 100, but sum check is disabled
    validate_weights([50.0, 10.0], enforce_sum=False)

def test_validate_weights_failures():
    # Negative weight
    with pytest.raises(InvalidInputException):
        validate_weights([50.0, -10.0, 60.0])
    # Sum mismatch
    with pytest.raises(InvalidInputException):
        validate_weights([50.0, 40.0])
    # Empty weights list
    with pytest.raises(InvalidInputException):
        validate_weights([])

def test_validate_strategy_success():
    assert validate_strategy("mean_variance") == "mean_variance"
    assert validate_strategy("min_volatility") == "min_volatility"
    assert validate_strategy("  Risk_Parity  ") == "risk_parity"

def test_validate_strategy_failures():
    with pytest.raises(InvalidInputException):
        validate_strategy("")
    
    with pytest.raises(StrategyNotSupportedException) as excinfo:
        validate_strategy("quantum_gravity_strategy")
    # Verify no suggestions found for a completely unrelated string
    assert "did_you_mean" not in excinfo.value.details

def test_normalize_strategy():
    from utils.validation import normalize_strategy
    assert normalize_strategy("maximize_sharpe") == "max_sharpe"
    assert normalize_strategy("Max Sharpe") == "max_sharpe"
    assert normalize_strategy("minimize-volatility") == "min_volatility"
    assert normalize_strategy("equal weights") == "equal_weight"
    assert normalize_strategy("min_volatilty") == "min_volatilty"  # typo kept, not mapped

def test_validate_strategy_aliases():
    assert validate_strategy("maximize_sharpe") == "max_sharpe"
    assert validate_strategy("Max Sharpe") == "max_sharpe"
    assert validate_strategy("minimize-volatility") == "min_volatility"
    assert validate_strategy("equal weights") == "equal_weight"

def test_validate_strategy_fuzzy_suggestions():
    # Test that a slight typo triggers a did_you_mean suggestion
    with pytest.raises(StrategyNotSupportedException) as excinfo:
        validate_strategy("min_volatilty")  # Typo: missing 'i'
    
    assert excinfo.value.details["did_you_mean"] == "min_volatility"
    assert "did_you_mean" in excinfo.value.details
    assert excinfo.value.details["input"] == "min_volatilty"
