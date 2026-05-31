import os
import pandas as pd
from typing import List, Dict
from core.exceptions import InvalidInputException, NotFoundException

class DataRepository:
    """
    Data repository for loading, cleaning, and aligning security returns,
    fund metadata, and factor returns datasets using Pandas.
    """
    def __init__(self, data_dir: str = None):
        """
        Initialize file paths. Defaults to searching in the project's root 'data/' directory.
        """
        if data_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.abspath(os.path.join(current_dir, "..", "data"))
        else:
            self.data_dir = data_dir
            
        self.metadata_path = os.path.join(self.data_dir, "fund_metadata.csv")
        self.prices_path = os.path.join(self.data_dir, "historical_prices.csv")
        self.factors_path = os.path.join(self.data_dir, "factor_returns.csv")

    def _load_metadata(self) -> pd.DataFrame:
        """
        Loads fund metadata from disk.
        """
        if not os.path.exists(self.metadata_path):
            raise NotFoundException(f"Metadata file not found at: {self.metadata_path}")
        df = pd.read_csv(self.metadata_path)
        df["ticker"] = df["ticker"].str.strip().str.upper()
        df = df.set_index("ticker")
        return df

    def _load_prices_and_compute_returns(self) -> pd.DataFrame:
        """
        Loads daily prices, cleans missing values, and computes percentage returns.
        """
        if not os.path.exists(self.prices_path):
            raise NotFoundException(f"Prices file not found at: {self.prices_path}")
            
        # Load and parse date index
        df = pd.read_csv(self.prices_path, parse_dates=["date"])
        df = df.set_index("date")
        df = df.sort_index() # Ensure chronological ordering
        
        # Log raw data shape
        print("Raw price data shape:", df.shape)
        
        # Check and log if missing values are found and filled
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Missing price values detected: {missing_count}. Performing forward/backward fill.")
            df = df.ffill().bfill()
            print("Cleaned price data shape:", df.shape)
        else:
            df = df.ffill().bfill()
            
        # Ensure no NaNs remain in price data
        if df.isnull().any().any():
            raise ValueError("Price dataset contains NaN values after imputation.")
            
        # Calculate daily percentage returns (decimal values) and drop the initial row (since it has no change)
        returns_df = df.pct_change().dropna()
        
        # Ensure no NaN remains in returns
        if returns_df.isnull().any().any():
            raise ValueError("Returns dataset contains NaN values.")
            
        # Format tickers to uppercase
        returns_df.columns = [col.strip().upper() for col in returns_df.columns]
        
        # Log returns shape
        print("Returns shape:", returns_df.shape)
        
        return returns_df

    def get_returns(self, tickers: List[str]) -> pd.DataFrame:
        """
        Retrieves daily returns for specific tickers, aligned on date index.
        """
        if not tickers:
            raise InvalidInputException("Ticker list for daily returns retrieval cannot be empty.")
            
        cleaned_tickers = [t.strip().upper() for t in tickers]
        returns_df = self._load_prices_and_compute_returns()
        
        # Available tickers print log
        dataset_tickers = list(returns_df.columns)
        print("Available tickers:", dataset_tickers)
        
        # Ensure all requested tickers are available
        missing_tickers = [t for t in cleaned_tickers if t not in returns_df.columns]
        if missing_tickers:
            raise NotFoundException(
                f"Ticker not found: {missing_tickers}"
            )
            
        # Subset returns DataFrame
        subset_df = returns_df[cleaned_tickers]
        
        # Diagnostic prints for returns DataFrame validation
        print("returns_df shape:", subset_df.shape)
        print("returns_df nulls:\n", subset_df.isnull().sum().to_dict())
        print("returns_df head:\n", subset_df.head())
        
        # Load factor data to align index (ensuring we return perfectly matched indices across the application)
        factors_df = self.get_factor_data()
        aligned_dates = subset_df.index.intersection(factors_df.index)
        
        # Subset to the intersection of dates and return
        return subset_df.loc[aligned_dates]

    def get_dividend_yield(self, tickers: List[str]) -> Dict[str, float]:
        """
        Retrieves annualized dividend yield values for a set of tickers.
        """
        if not tickers:
            raise InvalidInputException("Ticker list for dividend yield retrieval cannot be empty.")
            
        metadata = self._load_metadata()
        cleaned_tickers = [t.strip().upper() for t in tickers]
        
        # Verify requested tickers exist in metadata
        missing_tickers = [t for t in cleaned_tickers if t not in metadata.index]
        if missing_tickers:
            raise NotFoundException(
                f"Metadata for the following tickers was not found: {missing_tickers}"
            )
            
        yields = metadata.loc[cleaned_tickers, "dividend_yield"].to_dict()
        return yields

    def get_factor_data(self) -> pd.DataFrame:
        """
        Loads and returns daily factor returns, clean and sorted.
        """
        if not os.path.exists(self.factors_path):
            raise NotFoundException(f"Factor returns file not found at: {self.factors_path}")
            
        df = pd.read_csv(self.factors_path, parse_dates=["date"])
        df = df.set_index("date")
        df = df.sort_index()
        
        # Clean any missing values
        df = df.ffill().bfill()
        
        # Ensure no NaNs remain in factor returns
        if df.isnull().any().any():
            raise ValueError("Factor returns dataset contains NaN values after imputation.")
            
        return df

    def get_names(self, tickers: List[str]) -> Dict[str, str]:
        """
        Retrieves full names for a list of tickers.
        """
        if not tickers:
            raise InvalidInputException("Ticker list cannot be empty.")
            
        metadata = self._load_metadata()
        cleaned_tickers = [t.strip().upper() for t in tickers]
        
        missing_tickers = [t for t in cleaned_tickers if t not in metadata.index]
        if missing_tickers:
            raise NotFoundException(
                f"Metadata names for the following tickers were not found: {missing_tickers}"
            )
            
        names = metadata.loc[cleaned_tickers, "name"].to_dict()
        return names

