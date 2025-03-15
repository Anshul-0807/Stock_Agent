import numpy as np
from typing import Dict

def calculate_risk_metrics(returns: np.ndarray, confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Calculate various risk metrics from a series of returns.
    
    Args:
        returns (np.ndarray): Array of historical returns
        confidence_level (float): Confidence level for VaR calculation (default: 0.95)
    
    Returns:
        Dict[str, float]: Dictionary containing various risk metrics
    """
    try:
        # Remove any NaN values
        returns = returns[~np.isnan(returns)]
        
        if len(returns) == 0:
            return {
                'var': 0.0,
                'cvar': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        # Calculate VaR (Value at Risk)
        var = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Calculate CVaR (Conditional VaR)
        cvar = returns[returns <= var].mean()
        
        # Calculate Volatility (annualized)
        volatility = np.std(returns) * np.sqrt(252)  # Assuming daily returns
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(returns) if np.std(returns) != 0 else 0
        
        # Calculate Sortino Ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * np.mean(excess_returns) / downside_std if downside_std != 0 else 0
        
        # Calculate Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'var': float(var),
            'cvar': float(cvar),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown': float(max_drawdown)
        }
    except Exception as e:
        print(f"Error calculating risk metrics: {str(e)}")
        return {
            'var': 0.0,
            'cvar': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0
        } 