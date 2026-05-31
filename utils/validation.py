import re
import logging
import difflib
from typing import List, Set
from core.exceptions import InvalidInputException, StrategyNotSupportedException

# Configure logger for validation module
logger = logging.getLogger("app.validation")

# Ticker regex allows 1-6 alphanumeric uppercase chars, with an optional dot-separated exchange suffix (1-4 letters)
# Examples: AAPL, 7203.T, 0005.HK, RY.TO
TICKER_PATTERN = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{1,4})?$")

# Set of canonical portfolio optimization strategies
CANONICAL_STRATEGIES: Set[str] = {
    "mean_variance",
    "max_sharpe",
    "min_variance",
    "min_volatility",
    "risk_parity",
    "min_drawdown",
    "equal_weight"
}

# For backward compatibility
SUPPORTED_STRATEGIES: Set[str] = CANONICAL_STRATEGIES

# Comprehensive dictionary mapping aliases and variations to their canonical strategy name
STRATEGY_ALIASES = {
    "equal_weights": "equal_weight",
    "equalweight": "equal_weight",
    "equalweights": "equal_weight",
    "equal": "equal_weight",
    
    "maximize_sharpe": "max_sharpe",
    "maxsharpe": "max_sharpe",
    "sharpe": "max_sharpe",
    "max_sharpe_ratio": "max_sharpe",
    "sharpe_ratio": "max_sharpe",
    
    "meanvariance": "mean_variance",
    "mean_variance_optimization": "mean_variance",
    "mvo": "mean_variance",
    
    "minimize_volatility": "min_volatility",
    "minvolatility": "min_volatility",
    "minimum_volatility": "min_volatility",
    
    "minimize_variance": "min_variance",
    "minvariance": "min_variance",
    "minimum_variance": "min_variance",
    
    "riskparity": "risk_parity",
    "risk_parity_optimization": "risk_parity",
    
    "minimize_drawdown": "min_drawdown",
    "mindrawdown": "min_drawdown",
    "minimum_drawdown": "min_drawdown",
}

def validate_ticker(ticker: str) -> str:
    """
    Validate and clean ticker string.
    Raises InvalidInputException if the format is invalid.
    """
    if not ticker:
        raise InvalidInputException("Ticker name cannot be empty.")
    
    cleaned = ticker.strip().upper()
    if not TICKER_PATTERN.match(cleaned):
        raise InvalidInputException(
            f"Invalid ticker format: '{ticker}'. Ticker must be 1-6 alphanumeric uppercase characters, "
            f"optionally followed by a dot and an exchange code suffix (1-4 letters)."
        )
    return cleaned

def validate_weights(weights: List[float], enforce_sum: bool = True) -> None:
    """
    Ensure weights are non-negative and sum to exactly 100% (within floating point error margin).
    """
    if not weights:
        raise InvalidInputException("Asset weights list cannot be empty.")
        
    for w in weights:
        if w < 0.0:
            raise InvalidInputException(f"Asset weights cannot be negative. Found negative weight: {w}")
            
    if enforce_sum:
        total = sum(weights)
        if abs(total - 100.0) > 1e-4:
            raise InvalidInputException(
                f"Asset weights must sum to exactly 100%. Current sum is {total:.6f}%"
            )

def normalize_strategy(strategy: str) -> str:
    """
    Normalize a strategy name by cleaning casing, stripping spaces, replacing
    spaces/dashes with underscores, and mapping aliases to their canonical forms.
    """
    if not strategy:
        return ""
    
    # Trim and lowercase
    cleaned = strategy.strip().lower()
    
    # Replace whitespace and hyphens with underscores
    cleaned = re.sub(r"[\s\-]+", "_", cleaned)
    
    # Remove any duplicate consecutive underscores
    cleaned = re.sub(r"_+", "_", cleaned)
    
    # Check if this cleaned version is directly an alias
    if cleaned in STRATEGY_ALIASES:
        return STRATEGY_ALIASES[cleaned]
        
    return cleaned

def validate_strategy(strategy: str) -> str:
    """
    Confirm requested optimization strategy is supported, resolving aliases.
    Raises StrategyNotSupportedException if not supported, offering suggestions.
    """
    if not strategy:
        raise InvalidInputException("Optimization strategy must be specified.")
        
    normalized = normalize_strategy(strategy)
    logger.info(f"Received strategy: {strategy} -> normalized to {normalized}")
    
    if normalized not in CANONICAL_STRATEGIES:
        suggestions = difflib.get_close_matches(normalized, list(CANONICAL_STRATEGIES), n=1, cutoff=0.4)
        did_you_mean = suggestions[0] if suggestions else None
        
        details = {
            "error": "Strategy not supported",
            "input": strategy,
            "supported_strategies": sorted(list(CANONICAL_STRATEGIES)),
        }
        if did_you_mean:
            details["did_you_mean"] = did_you_mean
            
        err_msg = f"Strategy '{strategy}' is not supported. Supported strategies: {sorted(list(CANONICAL_STRATEGIES))}"
        if did_you_mean:
            err_msg += f" Did you mean: '{did_you_mean}'?"
            
        raise StrategyNotSupportedException(err_msg, details=details)
        
    return normalized

