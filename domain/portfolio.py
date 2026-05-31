from typing import List, Optional
from dataclasses import dataclass
from utils.validation import validate_ticker
from core.exceptions import InvalidInputException, ConstraintViolationException

@dataclass
class Security:
    """
    Domain entity representing an asset security and its portfolio allocation weight.
    """
    ticker: str
    weight: float

    def __post_init__(self) -> None:
        # Validate ticker format and cast to standard uppercase format
        self.ticker = validate_ticker(self.ticker)
        
        # Enforce weight boundaries for individual security
        if self.weight < 0.0:
            raise InvalidInputException(
                f"Security weight cannot be negative. Asset: {self.ticker}, Weight: {self.weight}%"
            )
        if self.weight > 100.0:
            raise InvalidInputException(
                f"Security weight cannot exceed 100%. Asset: {self.ticker}, Weight: {self.weight}%"
            )

@dataclass
class Constraints:
    """
    Domain value object containing portfolio mathematical optimization constraints.
    """
    min_weight: float = 0.0
    max_weight: float = 100.0
    min_dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.min_weight < 0.0:
            raise ConstraintViolationException(
                f"Minimum weight constraint cannot be negative. Found: {self.min_weight}%"
            )
        if self.max_weight > 100.0:
            raise ConstraintViolationException(
                f"Maximum weight constraint cannot exceed 100%. Found: {self.max_weight}%"
            )
        if self.min_weight > self.max_weight:
            raise ConstraintViolationException(
                f"Infeasible boundaries: Minimum weight constraint ({self.min_weight}%) "
                f"cannot be greater than maximum weight constraint ({self.max_weight}%)."
            )
        if self.min_dividend_yield < 0.0:
            raise ConstraintViolationException(
                f"Minimum dividend yield constraint cannot be negative. Found: {self.min_dividend_yield}%"
            )

@dataclass
class Portfolio:
    """
    Domain aggregate representing a collection of Securities under optional Optimization Constraints.
    """
    securities: List[Security]
    constraints: Optional[Constraints] = None

    def __post_init__(self) -> None:
        if not self.securities:
            raise InvalidInputException("A portfolio must contain at least one security.")
            
        # Ensure no duplicates in the security set
        seen_tickers = set()
        for security in self.securities:
            if security.ticker in seen_tickers:
                raise InvalidInputException(f"Duplicate security ticker detected: {security.ticker}")
            seen_tickers.add(security.ticker)
            
        # Validate feasibility of constraints against portfolio asset list size
        if self.constraints:
            n_assets = len(self.securities)
            min_possible = self.constraints.min_weight * n_assets
            max_possible = self.constraints.max_weight * n_assets
            
            if min_possible > 100.0:
                raise ConstraintViolationException(
                    f"Infeasible constraints: Minimum weight limit per asset ({self.constraints.min_weight}%) "
                    f"across {n_assets} assets requires at least {min_possible}%, which exceeds 100% total."
                )
            if max_possible < 100.0:
                raise ConstraintViolationException(
                    f"Infeasible constraints: Maximum weight limit per asset ({self.constraints.max_weight}%) "
                    f"across {n_assets} assets limits total weight to {max_possible}%, falling short of 100%."
                )

    def validate_weights_sum(self) -> None:
        """
        Enforce that the weights of all assets in the portfolio sum to exactly 100%.
        """
        total_weight = sum(s.weight for s in self.securities)
        if abs(total_weight - 100.0) > 1e-4:
            raise InvalidInputException(
                f"Portfolio weights must sum to exactly 100%. Current sum: {total_weight:.6f}%"
            )
