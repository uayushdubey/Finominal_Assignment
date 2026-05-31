import pandas as pd
from typing import Dict, List

from domain.portfolio import Portfolio, Security, Constraints
from repositories.data_repository import DataRepository
from services.factor_service import FactorService
from strategies.portfolio_strategies import StrategyFactory
from schemas.portfolio import (
    OptimizationRequest,
    PortfolioOptimizationResponse,
    AssetAllocationDetail,
    PortfolioMetrics
)

class PortfolioService:
    """
    Orchestration service that validates requests, resolves math strategies,
    triggers portfolio optimization, runs factor exposure regressions, and
    compiles detailed results.
    """
    def __init__(self, data_repository: DataRepository = None, factor_service: FactorService = None):
        """
        Inject DataRepository and FactorService dependencies.
        """
        self.data_repository = data_repository or DataRepository()
        self.factor_service = factor_service or FactorService(self.data_repository)

    def optimize_portfolio(self, request: OptimizationRequest) -> PortfolioOptimizationResponse:
        """
        Main use-case pipeline:
        1. Convert inputs to domain models (verifying domain invariants and bounds).
        2. Resolve and run optimization strategy.
        3. Perform factor OLS regression for the optimized weight distribution.
        4. Assemble detailed asset allocation and efficiency summaries.
        """
        try:
            tickers = [asset.ticker for asset in request.assets]
            current_weights = {asset.ticker: asset.weight for asset in request.assets}
            
            # 1. Instantiate Domain Models to enforce constraints and invariants
            domain_securities = [
                Security(ticker=asset.ticker, weight=asset.weight) 
                for asset in request.assets
            ]
            
            domain_constraints = None
            if request.constraints:
                domain_constraints = Constraints(
                    min_weight=request.constraints.min_weight,
                    max_weight=request.constraints.max_weight,
                    min_dividend_yield=request.constraints.min_dividend_yield
                )
                
            domain_portfolio = Portfolio(
                securities=domain_securities, 
                constraints=domain_constraints
            )
            
            # 2. Retrieve necessary data (resolves yields, names, and cleans price history)
            asset_returns = self.data_repository.get_returns(tickers)
            asset_names = self.data_repository.get_names(tickers)
            
            # 3. Resolve and run the mathematical strategy
            strategy = StrategyFactory.get_strategy(request.strategy)
            opt_res = strategy.optimize(domain_portfolio, asset_returns)
            
            # 4. Calculate factor exposure betas for the optimized portfolio weights
            factor_exp = self.factor_service.get_portfolio_factor_exposure(opt_res.optimized_weights)
            
            # 5. Format detailed asset allocations
            allocation_details = []
            for ticker in tickers:
                curr_w = current_weights[ticker]
                opt_w = opt_res.optimized_weights[ticker]
                name = asset_names.get(ticker, ticker)
                
                # Step 7: Round weights and changes to 2 decimal places
                detail = AssetAllocationDetail(
                    ticker=ticker,
                    name=name,
                    current_weight=round(curr_w, 2),
                    optimized_weight=round(opt_w, 2),
                    change=round(opt_w - curr_w, 2)
                )
                allocation_details.append(detail)
                
            # 6. Format metrics summary (Step 7: round expected_return, expected_volatility, sharpe_ratio to 4 decimal places)
            metrics = PortfolioMetrics(
                expected_return=round(opt_res.expected_return, 4),
                expected_volatility=round(opt_res.expected_volatility, 4),
                sharpe_ratio=round(opt_res.sharpe_ratio, 4)
            )
            
            return PortfolioOptimizationResponse(
                assets=allocation_details,
                metrics=metrics,
                factor_exposure=factor_exp
            )
        except Exception as e:
            from core.exceptions import AppException, InvalidInputException
            if isinstance(e, AppException):
                raise e
            raise InvalidInputException(
                "Data processing failed",
                details={"error": "Data processing failed", "details": str(e)}
            )
