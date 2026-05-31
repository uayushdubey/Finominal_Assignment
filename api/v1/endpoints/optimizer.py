from fastapi import APIRouter, Depends
from schemas.portfolio import OptimizationRequest, PortfolioOptimizationResponse
from services.portfolio_service import PortfolioService

router = APIRouter()

def get_portfolio_service() -> PortfolioService:
    """
    Dependency injection helper to instantiate PortfolioService.
    """
    return PortfolioService()

@router.post(
    "/optimize", 
    response_model=PortfolioOptimizationResponse, 
    summary="Optimize asset allocation weights",
    description="Consumes a list of assets and constraints, executes selected math optimization model, and runs factor regression analysis."
)
def optimize_portfolio(
    request: OptimizationRequest,
    service: PortfolioService = Depends(get_portfolio_service)
) -> PortfolioOptimizationResponse:
    """
    POST /optimize controller endpoint.
    All business, math, and constraint validations run inside the service layer.
    """
    print("Incoming request:", request)
    return service.optimize_portfolio(request)
