import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List
from repositories.data_repository import DataRepository
from schemas.factor import FactorExposureResponse
from core.exceptions import InvalidInputException

class FactorService:
    """
    Service layer responsible for calculating portfolio returns and performing
    OLS multivariate linear regressions to measure factor exposures (betas) 
    against benchmark risk factors (momentum, value, size).
    """
    def __init__(self, data_repository: DataRepository = None):
        """
        Inject DataRepository dependency.
        """
        self.data_repository = data_repository or DataRepository()

    def calculate_portfolio_returns(self, weights: Dict[str, float], asset_returns: pd.DataFrame) -> pd.Series:
        """
        Calculate the daily return series of a portfolio based on ticker weights and asset returns.
        """
        if not weights:
            raise InvalidInputException("Portfolio weights dictionary cannot be empty.")
            
        tickers = [t.strip().upper() for t in weights.keys()]
        
        # Verify that all weighted assets have return history
        missing = [t for t in tickers if t not in asset_returns.columns]
        if missing:
            raise InvalidInputException(f"Asset returns history not available in returns DataFrame for: {missing}")
            
        # Normalize weights to sum to 1.0 (decimal scale)
        total_w = sum(weights.values())
        if abs(total_w) < 1e-8:
            raise InvalidInputException("Sum of portfolio weights cannot be zero.")
            
        normalized_weights = {t.strip().upper(): weights[t] / total_w for t in weights}
        
        # Build aligned weights series
        w_series = pd.Series(normalized_weights)[tickers]
        
        # Dot product to calculate daily portfolio return: R_p = sum(w_i * R_i)
        port_returns = asset_returns[tickers].dot(w_series)
        port_returns.name = "portfolio_return"
        return port_returns

    def regress_factor_exposure(self, portfolio_returns: pd.Series, factor_returns: pd.DataFrame) -> FactorExposureResponse:
        """
        Align portfolio returns and factor returns, execute OLS regression, and format exposure results.
        
        Regression Formula:
        portfolio_return = alpha + beta_mom * momentum + beta_val * value + beta_size * size
        """
        # Ensure aligned dates via index intersection
        common_index = portfolio_returns.index.intersection(factor_returns.index)
        aligned_port_returns = portfolio_returns.loc[common_index]
        aligned_factor_returns = factor_returns.loc[common_index]
        
        # Ensure no NaN values exist in either
        if aligned_port_returns.isnull().any() or aligned_factor_returns.isnull().any().any():
            raise ValueError("Regressed data contains NaN values.")
            
        # Concat along dates and drop NaN rows just in case (ensuring aligned indices)
        data = pd.concat([aligned_port_returns, aligned_factor_returns], axis=1).dropna()
        
        # Check sufficient overlap
        if len(data) < 5:
            raise InvalidInputException(
                f"Insufficient date intersection between portfolio returns and factor returns. "
                f"Found only {len(data)} overlapping rows."
            )
            
        y = data["portfolio_return"]
        
        # Verify required factors exist
        factor_cols = ["momentum", "value", "size"]
        for col in factor_cols:
            if col not in data.columns:
                raise InvalidInputException(f"Required factor column '{col}' is missing in factor returns data.")
                
        X = data[factor_cols]
        X = sm.add_constant(X) # Appends intercept (const column of 1.0)
        
        # Fit Ordinary Least Squares model on unstandardized daily returns
        model = sm.OLS(y, X).fit()
        
        # Step 1: Detect dataset size for dynamic annualization factor
        factor = len(portfolio_returns) if len(portfolio_returns) < 30 else 252
        
        # Extract daily alpha and scale to annualized alpha
        daily_alpha = model.params.get("const", 0.0)
        annual_alpha = daily_alpha * factor
        
        # Beta coefficients (standardized)
        beta_mom = model.params.get("momentum", 0.0)
        beta_val = model.params.get("value", 0.0)
        beta_size = model.params.get("size", 0.0)
        
        # Extract t-statistics
        t_stats = {
            "const": float(model.tvalues.get("const", 0.0)),
            "momentum": float(model.tvalues.get("momentum", 0.0)),
            "value": float(model.tvalues.get("value", 0.0)),
            "size": float(model.tvalues.get("size", 0.0))
        }
        
        # Extract p-values (statistical significance levels)
        p_values = {
            "const": float(model.pvalues.get("const", 0.0)),
            "momentum": float(model.pvalues.get("momentum", 0.0)),
            "value": float(model.pvalues.get("value", 0.0)),
            "size": float(model.pvalues.get("size", 0.0))
        }
        
        # Round Fama-French regression outputs to 4 decimal places
        return FactorExposureResponse(
            alpha=round(float(annual_alpha), 4),
            beta_momentum=round(float(beta_mom), 4),
            beta_value=round(float(beta_val), 4),
            beta_size=round(float(beta_size), 4),
            r_squared=round(float(model.rsquared), 4),
            t_stats={k: round(v, 4) for k, v in t_stats.items()},
            p_values={k: round(v, 4) for k, v in p_values.items()}
        )

    def get_portfolio_factor_exposure(self, weights: Dict[str, float]) -> FactorExposureResponse:
        """
        Helper method to retrieve assets returns and factors, compute returns, and perform OLS.
        """
        tickers = list(weights.keys())
        asset_returns = self.data_repository.get_returns(tickers)
        factor_returns = self.data_repository.get_factor_data()
        
        port_returns = self.calculate_portfolio_returns(weights, asset_returns)
        return self.regress_factor_exposure(port_returns, factor_returns)
