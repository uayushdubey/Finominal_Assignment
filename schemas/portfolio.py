from pydantic import BaseModel, Field, field_validator, model_validator, AliasChoices
from typing import List, Dict, Optional
from utils.validation import validate_ticker, validate_strategy, validate_weights

class SecurityInput(BaseModel):
    """
    Pydantic schema representing a security and its weight in a portfolio.
    Used for parsing and validating individual asset allocations.
    """
    ticker: str = Field(description="Alphanumeric security identifier, e.g. 'AAPL' or '0005.HK'")
    weight: float = Field(description="Percentage allocation weight (0.0 to 100.0)")

    @field_validator("ticker")
    @classmethod
    def validate_ticker_format(cls, v: str) -> str:
        try:
            return validate_ticker(v)
        except Exception as e:
            raise ValueError(str(e))

    @field_validator("weight")
    @classmethod
    def validate_weight_value(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("Asset weight cannot be negative.")
        if v > 100.0:
            raise ValueError("Asset weight cannot exceed 100.0%.")
        return v

class Constraints(BaseModel):
    """
    Pydantic schema for portfolio optimization limits.
    """
    min_weight: float = Field(default=0.0, description="Minimum weight constraint per asset (0 to 100)")
    max_weight: float = Field(default=100.0, description="Maximum weight constraint per asset (0 to 100)")
    min_dividend_yield: float = Field(default=0.0, description="Minimum portfolio-wide dividend yield (non-negative)")

    @field_validator("min_weight", "max_weight")
    @classmethod
    def validate_weight_bounds(cls, v: float) -> float:
        if v < 0.0 or v > 100.0:
            raise ValueError("Weight constraints must be between 0.0 and 100.0 inclusive.")
        return v

    @field_validator("min_dividend_yield")
    @classmethod
    def validate_yield_bound(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError("Minimum dividend yield constraint cannot be negative.")
        return v

    @model_validator(mode="after")
    def validate_bounds_relationship(self) -> "Constraints":
        if self.min_weight > self.max_weight:
            raise ValueError(
                f"Minimum weight constraint ({self.min_weight}%) cannot exceed "
                f"maximum weight constraint ({self.max_weight}%)."
            )
        return self

class PortfolioInput(BaseModel):
    """
    Pydantic schema representing a complete portfolio input.
    Enforces that weights sum exactly to 100%.
    """
    assets: List[SecurityInput] = Field(description="List of security allocations making up the portfolio")

    @field_validator("assets")
    @classmethod
    def validate_assets(cls, v: List[SecurityInput]) -> List[SecurityInput]:
        if not v:
            raise ValueError("Portfolio must contain at least one asset.")
        
        # Check for duplicate tickers
        seen = set()
        for asset in v:
            if asset.ticker in seen:
                raise ValueError(f"Duplicate security ticker detected: {asset.ticker}")
            seen.add(asset.ticker)
        return v

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "PortfolioInput":
        weights = [asset.weight for asset in self.assets]
        try:
            validate_weights(weights, enforce_sum=True)
        except Exception as e:
            raise ValueError(str(e))
        return self

class OptimizationRequest(BaseModel):
    """
    Pydantic schema for requesting portfolio optimization.
    """
    assets: List[SecurityInput] = Field(
        description="Assets to consider for optimization (weights may serve as initial guess)",
        validation_alias=AliasChoices("assets", "portfolio")
    )
    constraints: Optional[Constraints] = Field(default_factory=Constraints, description="Optimization constraints")
    strategy: str = Field(description="Optimization strategy, e.g. 'mean_variance', 'min_variance', 'risk_parity'")

    @field_validator("assets")
    @classmethod
    def validate_assets(cls, v: List[SecurityInput]) -> List[SecurityInput]:
        if not v:
            raise ValueError("Optimization request must specify at least one asset.")
        
        # Check for duplicate tickers
        seen = set()
        for asset in v:
            if asset.ticker in seen:
                raise ValueError(f"Duplicate security ticker detected: {asset.ticker}")
            seen.add(asset.ticker)
        return v

    @field_validator("strategy")
    @classmethod
    def validate_opt_strategy(cls, v: str) -> str:
        try:
            return validate_strategy(v)
        except Exception as e:
            if hasattr(e, "details") and getattr(e, "details"):
                import json
                raise ValueError(json.dumps(getattr(e, "details")))
            raise ValueError(str(e))

    @model_validator(mode="after")
    def validate_feasibility(self) -> "OptimizationRequest":
        if self.constraints:
            n_assets = len(self.assets)
            min_possible = self.constraints.min_weight * n_assets
            max_possible = self.constraints.max_weight * n_assets
            
            if min_possible > 100.0:
                raise ValueError(
                    f"Infeasible constraints: Minimum weight limit per asset ({self.constraints.min_weight}%) "
                    f"across {n_assets} assets requires at least {min_possible}%, exceeding 100% total."
                )
            if max_possible < 100.0:
                raise ValueError(
                    f"Infeasible constraints: Maximum weight limit per asset ({self.constraints.max_weight}%) "
                    f"across {n_assets} assets limits total weight to {max_possible}%, falling short of 100%."
                )
        return self

class OptimizationResponse(BaseModel):
    """
    Pydantic schema for returning optimized portfolio metrics.
    """
    optimized_weights: Dict[str, float] = Field(description="Map of security ticker to optimized weight percentage")
    expected_return: float = Field(description="Expected portfolio annualized return percentage")
    expected_volatility: float = Field(description="Expected portfolio annualized volatility percentage")
    sharpe_ratio: float = Field(description="Portfolio Sharpe ratio")

    @field_validator("optimized_weights")
    @classmethod
    def validate_optimized_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        weights = list(v.values())
        try:
            validate_weights(weights, enforce_sum=True)
        except Exception as e:
            raise ValueError(f"Optimized weights validation failed: {str(e)}")
        return v

from schemas.factor import FactorExposureResponse

class AssetAllocationDetail(BaseModel):
    """
    Detailed asset allocation response showing name, current allocation, optimized allocation, and shift.
    """
    ticker: str = Field(description="Security alphanumeric identifier")
    name: str = Field(description="Full company or fund name")
    current_weight: float = Field(description="Provided initial allocation weight percentage")
    optimized_weight: float = Field(description="Calculated optimal allocation weight percentage")
    change: float = Field(description="Net allocation weight difference (optimized - current)")

class PortfolioMetrics(BaseModel):
    """
    Annualized portfolio return, risk, and efficiency metrics.
    """
    expected_return: float = Field(description="Annualized expected return percentage")
    expected_volatility: float = Field(description="Annualized portfolio standard deviation percentage")
    sharpe_ratio: float = Field(description="Portfolio Sharpe ratio")

class PortfolioOptimizationResponse(BaseModel):
    """
    Complete response body for POST /optimize containing asset breakdown, metrics, and factor beta exposures.
    """
    assets: List[AssetAllocationDetail] = Field(description="List of security breakdowns")
    metrics: PortfolioMetrics = Field(description="Annualized risk-return statistics")
    factor_exposure: FactorExposureResponse = Field(description="Optimized portfolio OLS factor regression outputs")

