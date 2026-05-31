import numpy as np
import pandas as pd
from typing import Union

def calculate_portfolio_return(
    weights: Union[np.ndarray, list], 
    expected_returns: Union[np.ndarray, list]
) -> float:
    """
    Calculate the expected portfolio return.
    
    Formula: R_p = w^T * R
    
    Parameters:
    - weights: Allocation weights (will be normalized to sum to 1.0)
    - expected_returns: Expected returns for each individual asset
    """
    w = np.array(weights, dtype=float)
    r = np.array(expected_returns, dtype=float)
    
    if len(w) != len(r):
        raise ValueError(
            f"Weights size ({len(w)}) must match expected returns size ({len(r)})."
        )
        
    total_w = np.sum(w)
    if total_w == 0.0:
        raise ValueError("Sum of portfolio weights cannot be zero.")
        
    # Normalize weights to sum to 1.0 (decimals)
    w_norm = w / total_w
    return float(np.dot(w_norm, r))

def calculate_covariance_matrix(
    returns: pd.DataFrame, 
    annualize: bool = True, 
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Calculate the sample covariance matrix from asset returns.
    Ensures numerical stability and validates no NaN/inf values.
    
    Parameters:
    - returns: Pandas DataFrame of daily percentage changes (returns)
    - annualize: Whether to scale the matrix to annual terms
    - trading_days: Number of trading days in a year (standard is 252)
    """
    if returns.empty:
        raise ValueError("Cannot calculate covariance matrix from empty returns DataFrame.")
        
    # Ensure no NaN exists in input returns
    if returns.isnull().any().any():
        raise ValueError("Covariance matrix contains NaN values.")
        
    # Step 4: Compute covariance correctly using returns.cov()
    cov_df = returns.cov()
    cov_values = cov_df.values
    
    # Validate no NaN or Inf
    if np.isnan(cov_values).any():
        raise ValueError("Covariance matrix contains NaN values.")
    if np.isinf(cov_values).any():
        raise ValueError("Covariance matrix contains Inf values.")
        
    n_assets = returns.shape[1]
    if cov_values.shape != (n_assets, n_assets):
        raise ValueError(f"Covariance matrix shape {cov_values.shape} does not match expected ({n_assets}, {n_assets}).")
        
    # Step 5: Handle small dataset instability
    # We apply regularization if the number of observations is small relative to the number of assets,
    # or if we have fewer than 252 data points.
    if len(returns) < 252 or returns.shape[0] < returns.shape[1]:
        print(f"Small dataset detected (size: {len(returns)}). Adding regularization term to covariance matrix.")
        cov_values = cov_values + np.eye(len(cov_values)) * 1e-6
        
    # Print the covariance matrix
    print("Cov matrix:\n", cov_values)
    
    # Scale if annualize
    if annualize:
        # Step 1: Detect dataset size for dynamic annualization factor
        factor = len(returns) if len(returns) < 30 else trading_days
        cov_values = cov_values * factor
        
    # Re-wrap in DataFrame to preserve index/columns
    return pd.DataFrame(cov_values, index=cov_df.index, columns=cov_df.columns)

def calculate_portfolio_volatility(
    weights: Union[np.ndarray, list], 
    cov_matrix: Union[np.ndarray, pd.DataFrame]
) -> float:
    """
    Calculate expected portfolio volatility (standard deviation of returns).
    
    Formula: sigma_p = sqrt(w^T * Sigma * w)
    
    Parameters:
    - weights: Allocation weights (will be normalized to sum to 1.0)
    - cov_matrix: Covariance matrix of asset returns (must be square, NxN)
    """
    w = np.array(weights, dtype=float)
    sigma = np.array(cov_matrix, dtype=float)
    
    if len(w) != sigma.shape[0] or len(w) != sigma.shape[1]:
        raise ValueError(
            f"Weights dimensions ({len(w)}) must match covariance matrix dimensions {sigma.shape}."
        )
        
    total_w = np.sum(w)
    if total_w == 0.0:
        raise ValueError("Sum of portfolio weights cannot be zero.")
        
    # Normalize weights to sum to 1.0 (decimals)
    w_norm = w / total_w
    
    # Quadratic form: w^T * Sigma * w
    variance = np.dot(w_norm.T, np.dot(sigma, w_norm))
    
    # Catch floating point edge cases below zero
    if variance < 0.0:
        return 0.0
        
    return float(np.sqrt(variance))

def portfolio_return(weights: Union[np.ndarray, list], returns_df: pd.DataFrame) -> float:
    """
    Calculate expected portfolio return (vectorized, annualized, prevents division by zero).
    """
    if returns_df.empty:
        raise ValueError("Returns DataFrame is empty.")
    if returns_df.isnull().any().any():
        raise ValueError("Returns DataFrame contains NaN values.")
        
    w = np.array(weights, dtype=float)
    total_w = np.sum(w)
    if total_w == 0.0:
        return 0.0
    w_norm = w / total_w
    
    # Calculate daily portfolio returns series
    portfolio_returns = returns_df.dot(w_norm)
    
    # Step 1: Detect dataset size for dynamic annualization factor
    factor = len(returns_df) if len(returns_df) < 30 else 252
    expected_return = portfolio_returns.mean() * factor
    return float(expected_return)

def portfolio_volatility(weights: Union[np.ndarray, list], cov_matrix: Union[np.ndarray, pd.DataFrame]) -> float:
    """
    Calculate expected portfolio volatility (vectorized, prevents division by zero).
    """
    w = np.array(weights, dtype=float)
    cov = np.array(cov_matrix, dtype=float)
    total_w = np.sum(w)
    if total_w == 0.0:
        return 0.0
    w_norm = w / total_w
    
    variance = np.dot(w_norm.T, np.dot(cov, w_norm))
    if variance <= 0.0:
        return 0.0
    return float(np.sqrt(variance))

def sharpe_ratio(
    weights: Union[np.ndarray, list], 
    returns_df: pd.DataFrame = None, 
    cov_matrix: Union[np.ndarray, pd.DataFrame] = None,
    risk_free_rate: float = 0.0
) -> float:
    """
    Calculate portfolio Sharpe ratio (vectorized, prevents division by zero).
    """
    if returns_df is None or cov_matrix is None:
        raise ValueError("returns_df and cov_matrix are required to calculate the Sharpe ratio.")
        
    ret = portfolio_return(weights, returns_df)
    vol = portfolio_volatility(weights, cov_matrix)
    if vol == 0.0:
        return 0.0
    return float((ret - risk_free_rate) / vol)
