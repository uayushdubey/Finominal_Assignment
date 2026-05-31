import numpy as np
from typing import List, Dict, Tuple, Optional
from domain.portfolio import Portfolio
from core.exceptions import ConstraintViolationException

def validate_portfolio_constraints(portfolio: Portfolio, yields: Dict[str, float]) -> None:
    """
    Perform a pre-optimization audit to verify constraint feasibility.
    Raises ConstraintViolationException if weight bounds or dividend yield targets are mathematically impossible.
    """
    tickers = [s.ticker for s in portfolio.securities]
    n = len(tickers)
    
    # 1. Extract and validate weight bounds
    min_w_pct = 0.0
    max_w_pct = 100.0
    if portfolio.constraints:
        min_w_pct = portfolio.constraints.min_weight
        max_w_pct = portfolio.constraints.max_weight
        
    min_w = min_w_pct / 100.0
    max_w = max_w_pct / 100.0
    
    # Feasibility of weights sum bounds
    if min_w * n > 1.0:
        raise ConstraintViolationException(
            f"Weight constraints are infeasible: Minimum weight per asset ({min_w_pct}%) "
            f"across {n} assets requires at least {min_w_pct * n}%, exceeding 100% total."
        )
    if max_w * n < 1.0:
        raise ConstraintViolationException(
            f"Weight constraints are infeasible: Maximum weight per asset ({max_w_pct}%) "
            f"across {n} assets limits total weight to {max_w_pct * n}%, falling short of 100%."
        )
        
    # 2. Extract and validate dividend yield target feasibility
    if portfolio.constraints and portfolio.constraints.min_dividend_yield > 0.0:
        min_yield = portfolio.constraints.min_dividend_yield
        
        # Greedy analytical solver to find maximum achievable dividend yield under weights bounds
        allocated_w = {t: min_w for t in tickers}
        remaining = 1.0 - n * min_w
        
        # Sort tickers by dividend yield in descending order
        sorted_tickers = sorted(tickers, key=lambda t: yields.get(t, 0.0), reverse=True)
        
        for t in sorted_tickers:
            if remaining <= 1e-9:
                break
            add_w = min(max_w - min_w, remaining)
            allocated_w[t] += add_w
            remaining -= add_w
            
        max_possible_yield = sum(allocated_w[t] * yields.get(t, 0.0) for t in tickers)
        
        # Verify if maximum achievable yield is at least the target minimum
        if max_possible_yield < min_yield - 1e-6:
            raise ConstraintViolationException(
                f"Infeasible dividend yield target: Minimum dividend yield requested is {min_yield:.4f}%, "
                f"but the maximum possible dividend yield achievable under current weight constraints "
                f"({min_w_pct}% to {max_w_pct}%) is {max_possible_yield:.4f}%."
            )

class SciPyConstraintBuilder:
    """
    Builder class to translate domain-level Portfolio Constraints to SciPy optimization boundaries.
    """
    def __init__(self, portfolio: Portfolio, yields: Dict[str, float]):
        self.portfolio = portfolio
        self.yields = yields
        self.tickers = [s.ticker for s in portfolio.securities]
        
    def get_bounds(self) -> List[Tuple[float, float]]:
        """
        Build bounds tuple list for individual asset weights.
        """
        if self.portfolio.constraints:
            min_w = self.portfolio.constraints.min_weight / 100.0
            max_w = self.portfolio.constraints.max_weight / 100.0
        else:
            min_w = 0.0
            max_w = 1.0
        return [(min_w, max_w) for _ in range(len(self.tickers))]
        
    def get_sum_constraint(self) -> Dict:
        """
        Build sum-to-1.0 equality constraint: sum(w) - 1.0 = 0.
        """
        return {
            "type": "eq",
            "fun": lambda w: np.sum(w) - 1.0
        }
        
    def get_dividend_yield_constraint(self) -> Optional[Dict]:
        """
        Build dividend yield inequality constraint: w^T * yields >= min_yield.
        """
        if not self.portfolio.constraints or self.portfolio.constraints.min_dividend_yield <= 0.0:
            return None
            
        min_yield = self.portfolio.constraints.min_dividend_yield
        yields_array = np.array([self.yields.get(t, 0.0) for t in self.tickers])
        
        # SciPy inequality format: g(w) >= 0
        return {
            "type": "ineq",
            "fun": lambda w: np.dot(w, yields_array) - min_yield
        }
        
    def build_all(self) -> Tuple[List[Tuple[float, float]], List[Dict]]:
        """
        Compile bounds list and all constraint dicts.
        """
        bounds = self.get_bounds()
        constraints = [self.get_sum_constraint()]
        
        dy_constraint = self.get_dividend_yield_constraint()
        if dy_constraint:
            constraints.append(dy_constraint)
            
        return bounds, constraints
