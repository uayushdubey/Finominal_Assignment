import logging
from abc import ABC, abstractmethod
from typing import Dict, Type, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from domain.portfolio import Portfolio
from schemas.portfolio import OptimizationResponse
from core.exceptions import StrategyNotSupportedException, ConstraintViolationException
from utils.financial import calculate_portfolio_return, calculate_portfolio_volatility, calculate_covariance_matrix
from utils.constraints import validate_portfolio_constraints, SciPyConstraintBuilder

logger = logging.getLogger("app.strategies")

class OptimizationStrategy(ABC):
    """
    Abstract Base Class representing a portfolio optimization strategy.
    """
    @abstractmethod
    def optimize(self, portfolio: Portfolio, returns: pd.DataFrame) -> OptimizationResponse:
        """
        Executes the optimization algorithm given a portfolio and asset returns history.
        Returns an OptimizationResponse with weights and annualized metrics.
        """
        pass


class StrategyFactory:
    """
    Registry factory mapping optimization strategy names to their concrete class implementations.
    Prevents hardcoded if-else chains.
    """
    _registry: Dict[str, Type[OptimizationStrategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_cls: Type[OptimizationStrategy]) -> None:
        """
        Registers a strategy class under a given name.
        """
        cls._registry[name.lower().strip()] = strategy_cls

    @classmethod
    def get_strategy(cls, name: str) -> OptimizationStrategy:
        """
        Resolves a strategy name to an instantiated strategy object.
        Raises StrategyNotSupportedException if the strategy is unregistered.
        """
        from utils.validation import normalize_strategy
        
        normalized_name = normalize_strategy(name)
        strategy_cls = cls._registry.get(normalized_name)
        if not strategy_cls:
            import difflib
            suggestions = difflib.get_close_matches(normalized_name, list(cls._registry.keys()), n=1, cutoff=0.4)
            did_you_mean = suggestions[0] if suggestions else None
            
            details = {
                "error": "Strategy not supported",
                "input": name,
                "supported_strategies": sorted(list(cls._registry.keys())),
            }
            if did_you_mean:
                details["did_you_mean"] = did_you_mean
                
            err_msg = f"Strategy '{name}' is not supported. Supported strategies: {sorted(list(cls._registry.keys()))}"
            if did_you_mean:
                err_msg += f" Did you mean: '{did_you_mean}'?"
                
            raise StrategyNotSupportedException(err_msg, details=details)
        return strategy_cls()


class EqualWeightStrategy(OptimizationStrategy):
    """
    Equal Weight Portfolio Strategy.
    Distributes allocations equally: w_i = 1/N.
    """
    def optimize(self, portfolio: Portfolio, returns: pd.DataFrame) -> OptimizationResponse:
        tickers = [sec.ticker for sec in portfolio.securities]
        n_assets = len(tickers)
        
        # Load dividend yields from data repository
        from repositories.data_repository import DataRepository
        yields = DataRepository().get_dividend_yield(tickers)
        
        # Run pre-optimization feasibility audit
        validate_portfolio_constraints(portfolio, yields)
        
        # Calculate equal allocations
        equal_weight = 1.0 / n_assets
        weights = np.repeat(equal_weight, n_assets)
        p_weights = weights * 100.0
        
        # Explicitly verify equal allocation satisfies dividend yield constraint
        if portfolio.constraints and portfolio.constraints.min_dividend_yield > 0.0:
            port_yield = sum(equal_weight * yields.get(t, 0.0) for t in tickers)
            if port_yield < portfolio.constraints.min_dividend_yield - 1e-6:
                raise ConstraintViolationException(
                    f"Equal weight portfolio dividend yield ({port_yield:.4f}%) "
                    f"violates minimum target ({portfolio.constraints.min_dividend_yield:.4f}%)."
                )
        
        # Compute annualized expected metrics
        factor = len(returns) if len(returns) < 30 else 252
        ann_returns = returns[tickers].mean() * factor
        ann_cov = calculate_covariance_matrix(returns[tickers], annualize=True)
        
        exp_return = calculate_portfolio_return(p_weights, ann_returns)
        exp_vol = calculate_portfolio_volatility(p_weights, ann_cov)
        
        # Volatility check and error raising
        if exp_vol == 0.0:
            raise ValueError("Portfolio volatility is zero. Optimization is unstable.")
            
        sharpe = exp_return / exp_vol
        
        # Print diagnostic logs
        print("Mean daily return:", exp_return / factor)
        print("Annual return:", exp_return)
        print("Annual volatility:", exp_vol)
        print("Sharpe ratio:", sharpe)
        
        opt_weights_dict = {ticker: float(w) for ticker, w in zip(tickers, p_weights)}
        
        return OptimizationResponse(
            optimized_weights=opt_weights_dict,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe
        )

StrategyFactory.register("equal_weight", EqualWeightStrategy)


class MinVolatilityStrategy(OptimizationStrategy):
    """
    Minimum Volatility (Variance) Strategy.
    Finds weights that minimize the portfolio annualized return variance: w^T * Sigma * w.
    """
    def optimize(self, portfolio: Portfolio, returns: pd.DataFrame) -> OptimizationResponse:
        tickers = [sec.ticker for sec in portfolio.securities]
        n_assets = len(tickers)
        
        from repositories.data_repository import DataRepository
        yields = DataRepository().get_dividend_yield(tickers)
        
        # Validate feasibility
        validate_portfolio_constraints(portfolio, yields)
        
        factor = len(returns) if len(returns) < 30 else 252
        ann_returns = returns[tickers].mean() * factor
        
        # STEP 4: Covariance matrix check
        cov_matrix = returns[tickers].cov().values
        assert cov_matrix.shape == (n_assets, n_assets), f"Expected covariance matrix shape {(n_assets, n_assets)}, got {cov_matrix.shape}"
        assert not np.isnan(cov_matrix).any(), "Covariance matrix contains NaN values"
        assert not np.isinf(cov_matrix).any(), "Covariance matrix contains Inf values"
        print("Cov matrix:", cov_matrix)
        
        ann_cov = calculate_covariance_matrix(returns[tickers], annualize=True).values
        
        # Build SciPy boundaries
        builder = SciPyConstraintBuilder(portfolio, yields)
        bounds, constraints = builder.build_all()
        
        # Objective: minimize variance
        def variance_objective(w: np.ndarray) -> float:
            return float(np.dot(w.T, np.dot(ann_cov, w)))

        # Initial guess (equal weight)
        w0 = np.repeat(1.0 / n_assets, n_assets)
        
        res = minimize(
            variance_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        
        # STEP 6: SciPy Optimization debug prints
        print("Optimizer success:", res.success)
        print("Optimizer message:", res.message)
        print("Weights:", res.x)
        
        if not res.success:
            logger.error(f"Minimum volatility optimization failed: {res.message}")
            raise ConstraintViolationException(f"Minimum volatility optimization failed: {res.message}")
            
        opt_weights = res.x
        
        # Step 8: Post-optimization constraint verification
        if portfolio.constraints and portfolio.constraints.min_dividend_yield > 0.0:
            portfolio_dividend = sum(w * yields.get(t, 0.0) for w, t in zip(opt_weights, tickers))
            if portfolio_dividend < portfolio.constraints.min_dividend_yield - 1e-6:
                raise ConstraintViolationException("Constraints infeasible")
                
        p_weights = opt_weights * 100.0
        exp_return = calculate_portfolio_return(p_weights, ann_returns)
        exp_vol = calculate_portfolio_volatility(p_weights, ann_cov)
        
        # Volatility check and error raising
        if exp_vol == 0.0:
            raise ValueError("Portfolio volatility is zero. Optimization is unstable.")
            
        sharpe = exp_return / exp_vol
        
        # Print diagnostic logs
        print("Mean daily return:", exp_return / factor)
        print("Annual return:", exp_return)
        print("Annual volatility:", exp_vol)
        print("Sharpe ratio:", sharpe)
        
        opt_weights_dict = {ticker: float(w) for ticker, w in zip(tickers, p_weights)}
        
        return OptimizationResponse(
            optimized_weights=opt_weights_dict,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe
        )

StrategyFactory.register("min_volatility", MinVolatilityStrategy)
StrategyFactory.register("min_variance", MinVolatilityStrategy)


class MaxSharpeStrategy(OptimizationStrategy):
    """
    Maximum Sharpe Ratio Strategy.
    Finds weights that maximize portfolio Sharpe ratio: (R_p - R_f) / Vol_p.
    Optimizes by minimizing the negative Sharpe ratio.
    """
    def optimize(self, portfolio: Portfolio, returns: pd.DataFrame) -> OptimizationResponse:
        tickers = [sec.ticker for sec in portfolio.securities]
        n_assets = len(tickers)
        
        from repositories.data_repository import DataRepository
        yields = DataRepository().get_dividend_yield(tickers)
        
        # Validate feasibility
        validate_portfolio_constraints(portfolio, yields)
        
        factor = len(returns) if len(returns) < 30 else 252
        ann_returns = returns[tickers].mean() * factor
        
        # STEP 4: Covariance matrix check
        cov_matrix = returns[tickers].cov().values
        assert cov_matrix.shape == (n_assets, n_assets), f"Expected covariance matrix shape {(n_assets, n_assets)}, got {cov_matrix.shape}"
        assert not np.isnan(cov_matrix).any(), "Covariance matrix contains NaN values"
        assert not np.isinf(cov_matrix).any(), "Covariance matrix contains Inf values"
        print("Cov matrix:", cov_matrix)
        
        ann_cov = calculate_covariance_matrix(returns[tickers], annualize=True).values
        
        builder = SciPyConstraintBuilder(portfolio, yields)
        bounds, constraints = builder.build_all()
        
        # Objective: minimize negative Sharpe ratio (assume risk-free rate is 0.0)
        def negative_sharpe_objective(w: np.ndarray) -> float:
            port_ret = np.dot(w, ann_returns)
            port_var = np.dot(w.T, np.dot(ann_cov, w))
            port_vol = np.sqrt(port_var) if port_var > 1e-8 else 1e-4
            return -port_ret / port_vol

        w0 = np.repeat(1.0 / n_assets, n_assets)
        
        res = minimize(
            negative_sharpe_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        
        # STEP 6: SciPy Optimization debug prints
        print("Optimizer success:", res.success)
        print("Optimizer message:", res.message)
        print("Weights:", res.x)
        
        if not res.success:
            logger.error(f"Maximum Sharpe optimization failed: {res.message}")
            raise ConstraintViolationException(f"Maximum Sharpe optimization failed: {res.message}")
            
        opt_weights = res.x
        
        # Step 8: Post-optimization constraint verification
        if portfolio.constraints and portfolio.constraints.min_dividend_yield > 0.0:
            portfolio_dividend = sum(w * yields.get(t, 0.0) for w, t in zip(opt_weights, tickers))
            if portfolio_dividend < portfolio.constraints.min_dividend_yield - 1e-6:
                raise ConstraintViolationException("Constraints infeasible")
                
        p_weights = opt_weights * 100.0
        exp_return = calculate_portfolio_return(p_weights, ann_returns)
        exp_vol = calculate_portfolio_volatility(p_weights, ann_cov)
        
        # Volatility check and error raising
        if exp_vol == 0.0:
            raise ValueError("Portfolio volatility is zero. Optimization is unstable.")
            
        sharpe = exp_return / exp_vol
        
        # Print diagnostic logs
        print("Mean daily return:", exp_return / factor)
        print("Annual return:", exp_return)
        print("Annual volatility:", exp_vol)
        print("Sharpe ratio:", sharpe)
        
        opt_weights_dict = {ticker: float(w) for ticker, w in zip(tickers, p_weights)}
        
        return OptimizationResponse(
            optimized_weights=opt_weights_dict,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe
        )

StrategyFactory.register("max_sharpe", MaxSharpeStrategy)
StrategyFactory.register("mean_variance", MaxSharpeStrategy)


class RiskParityStrategy(OptimizationStrategy):
    """
    Risk Parity Strategy.
    Finds weights that equalize risk contributions across all assets.
    """
    def optimize(self, portfolio: Portfolio, returns: pd.DataFrame) -> OptimizationResponse:
        tickers = [sec.ticker for sec in portfolio.securities]
        n_assets = len(tickers)
        
        from repositories.data_repository import DataRepository
        yields = DataRepository().get_dividend_yield(tickers)
        
        # Validate feasibility
        validate_portfolio_constraints(portfolio, yields)
        
        factor = len(returns) if len(returns) < 30 else 252
        ann_returns = returns[tickers].mean() * factor
        
        # STEP 4: Covariance matrix check
        cov_matrix = returns[tickers].cov().values
        assert cov_matrix.shape == (n_assets, n_assets), f"Expected covariance matrix shape {(n_assets, n_assets)}, got {cov_matrix.shape}"
        assert not np.isnan(cov_matrix).any(), "Covariance matrix contains NaN values"
        assert not np.isinf(cov_matrix).any(), "Covariance matrix contains Inf values"
        print("Cov matrix:", cov_matrix)
        
        ann_cov = calculate_covariance_matrix(returns[tickers], annualize=True).values
        
        builder = SciPyConstraintBuilder(portfolio, yields)
        bounds, constraints = builder.build_all()
        
        # Objective: minimize sum of squared deviations of risk contributions from targets
        def risk_parity_objective(w: np.ndarray) -> float:
            variance = np.dot(w.T, np.dot(ann_cov, w))
            vol = np.sqrt(variance) if variance > 1e-8 else 1e-4
            
            mrc = np.dot(ann_cov, w)
            rc = w * mrc / vol
            target = vol / n_assets
            
            return float(np.sum((rc - target) ** 2) * 1e4)

        w0 = np.repeat(1.0 / n_assets, n_assets)
        
        res = minimize(
            risk_parity_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        
        # STEP 6: SciPy Optimization debug prints
        print("Optimizer success:", res.success)
        print("Optimizer message:", res.message)
        print("Weights:", res.x)
        
        if not res.success:
            logger.error(f"Risk parity optimization failed: {res.message}")
            raise ConstraintViolationException(f"Risk parity optimization failed: {res.message}")
            
        opt_weights = res.x
        
        # Step 7: Risk parity zero volatility safety check
        p_weights = opt_weights * 100.0
        exp_vol = calculate_portfolio_volatility(p_weights, ann_cov)
        if exp_vol == 0.0:
            raise ValueError("Portfolio volatility is zero. Optimization is unstable.")
            
        # Step 8: Post-optimization constraint verification
        if portfolio.constraints and portfolio.constraints.min_dividend_yield > 0.0:
            portfolio_dividend = sum(w * yields.get(t, 0.0) for w, t in zip(opt_weights, tickers))
            if portfolio_dividend < portfolio.constraints.min_dividend_yield - 1e-6:
                raise ConstraintViolationException("Constraints infeasible")
                
        exp_return = calculate_portfolio_return(p_weights, ann_returns)
        sharpe = exp_return / exp_vol
        
        # Print diagnostic logs
        print("Mean daily return:", exp_return / factor)
        print("Annual return:", exp_return)
        print("Annual volatility:", exp_vol)
        print("Sharpe ratio:", sharpe)
        
        opt_weights_dict = {ticker: float(w) for ticker, w in zip(tickers, p_weights)}
        
        return OptimizationResponse(
            optimized_weights=opt_weights_dict,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe
        )

StrategyFactory.register("risk_parity", RiskParityStrategy)


class MinDrawdownStrategy(OptimizationStrategy):
    """
    Minimum Drawdown Strategy.
    Finds weights that minimize the maximum historical drawdown of the portfolio.
    """
    def optimize(self, portfolio: Portfolio, returns: pd.DataFrame) -> OptimizationResponse:
        tickers = [sec.ticker for sec in portfolio.securities]
        n_assets = len(tickers)
        
        from repositories.data_repository import DataRepository
        yields = DataRepository().get_dividend_yield(tickers)
        
        # Validate feasibility
        validate_portfolio_constraints(portfolio, yields)
        
        factor = len(returns) if len(returns) < 30 else 252
        ann_returns = returns[tickers].mean() * factor
        
        # STEP 4: Covariance matrix check
        cov_matrix = returns[tickers].cov().values
        assert cov_matrix.shape == (n_assets, n_assets), f"Expected covariance matrix shape {(n_assets, n_assets)}, got {cov_matrix.shape}"
        assert not np.isnan(cov_matrix).any(), "Covariance matrix contains NaN values"
        assert not np.isinf(cov_matrix).any(), "Covariance matrix contains Inf values"
        print("Cov matrix:", cov_matrix)
        
        ann_cov = calculate_covariance_matrix(returns[tickers], annualize=True).values
        daily_returns = returns[tickers].values
        
        builder = SciPyConstraintBuilder(portfolio, yields)
        bounds, constraints = builder.build_all()
        
        # Objective: minimize maximum historical drawdown
        def max_drawdown_objective(w: np.ndarray) -> float:
            port_daily_ret = np.dot(daily_returns, w)
            wealth = np.cumprod(1.0 + port_daily_ret)
            running_max = np.maximum.accumulate(wealth)
            drawdowns = (running_max - wealth) / running_max
            return float(np.max(drawdowns))

        w0 = np.repeat(1.0 / n_assets, n_assets)
        
        res = minimize(
            max_drawdown_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )
        
        # STEP 6: SciPy Optimization debug prints
        print("Optimizer success:", res.success)
        print("Optimizer message:", res.message)
        print("Weights:", res.x)
        
        if not res.success:
            logger.error(f"Minimum drawdown optimization failed: {res.message}")
            raise ConstraintViolationException(f"Minimum drawdown optimization failed: {res.message}")
            
        opt_weights = res.x
        
        # Step 8: Post-optimization constraint verification
        if portfolio.constraints and portfolio.constraints.min_dividend_yield > 0.0:
            portfolio_dividend = sum(w * yields.get(t, 0.0) for w, t in zip(opt_weights, tickers))
            if portfolio_dividend < portfolio.constraints.min_dividend_yield - 1e-6:
                raise ConstraintViolationException("Constraints infeasible")
                
        p_weights = opt_weights * 100.0
        exp_return = calculate_portfolio_return(p_weights, ann_returns)
        exp_vol = calculate_portfolio_volatility(p_weights, ann_cov)
        
        # Volatility check and error raising
        if exp_vol == 0.0:
            raise ValueError("Portfolio volatility is zero. Optimization is unstable.")
            
        sharpe = exp_return / exp_vol
        
        # Print diagnostic logs
        print("Mean daily return:", exp_return / factor)
        print("Annual return:", exp_return)
        print("Annual volatility:", exp_vol)
        print("Sharpe ratio:", sharpe)
        
        opt_weights_dict = {ticker: float(w) for ticker, w in zip(tickers, p_weights)}
        
        return OptimizationResponse(
            optimized_weights=opt_weights_dict,
            expected_return=exp_return,
            expected_volatility=exp_vol,
            sharpe_ratio=sharpe
        )

StrategyFactory.register("min_drawdown", MinDrawdownStrategy)
