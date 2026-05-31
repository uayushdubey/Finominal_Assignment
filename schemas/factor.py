from pydantic import BaseModel, Field
from typing import Dict

class FactorExposureResponse(BaseModel):
    """
    Pydantic schema representing the Fama-French-like risk factor exposures (betas)
    of a portfolio, calculated via OLS regression.
    """
    alpha: float = Field(description="Annualized intercept coefficient representing active return excess of factor exposures")
    beta_momentum: float = Field(description="Portfolio sensitivity to the momentum factor (MOM)")
    beta_value: float = Field(description="Portfolio sensitivity to the value factor (HML)")
    beta_size: float = Field(description="Portfolio sensitivity to the size factor (SMB)")
    r_squared: float = Field(description="Coefficient of determination (R-squared) explaining variance ratio")
    t_stats: Dict[str, float] = Field(description="T-statistics for each of the regression coefficients (const, momentum, value, size)")
    p_values: Dict[str, float] = Field(description="P-values reflecting the statistical significance of each parameter")
