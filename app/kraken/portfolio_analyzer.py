"""
Enhanced Portfolio Analyzer for Crypto Trading
Provides comprehensive portfolio analysis with advanced metrics
"""

import json
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️  scipy not available. Some statistical features will be limited.")

class PortfolioAnalyzer:
    """Enhanced portfolio analyzer with multi-timeframe and risk analysis"""
    
    def __init__(self, data_dir: str = "app/data"):
        self.data_dir = data_dir
        self.cache_dir = os.path.join(data_dir, "cache")
        
        # Ensure directories exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Risk-free rate (annual, for Sharpe ratio)
        self.risk_free_rate = 0.02  # 2% annual
        
        # Crypto sector categories
        self.crypto_sectors = {
            'BTC': 'Store of Value',
            'ETH': 'Smart Contract Platform',
            'ADA': 'Smart Contract Platform',
            'SOL': 'Smart Contract Platform',
            'DOT': 'Interoperability',
            'LINK': 'Oracle',
            'UNI': 'DEX',
            'AAVE': 'DeFi',
            'LTC': 'Payment',
            'XRP': 'Payment',
            'DOGE': 'Meme',
            'SHIB': 'Meme',
            'AVAX': 'Smart Contract Platform',
            'MATIC': 'Scaling',
            'ATOM': 'Interoperability',
            'FTM': 'Smart Contract Platform',
            'ALGO': 'Smart Contract Platform',
            'VET': 'Supply Chain',
            'XLM': 'Payment',
            'HBAR': 'Enterprise',
            'ICP': 'Web3',
            'FIL': 'Storage',
            'ETC': 'Smart Contract Platform',
            'THETA': 'Media',
            'TRX': 'Entertainment',
            'XMR': 'Privacy',
            'ZEC': 'Privacy',
        }
        
        # Timeframes for analysis
        self.timeframes = {
            '7d': 7,
            '30d': 30,
            '90d': 90,
            '1y': 365
        }
        
        # Benchmark assets
        self.benchmarks = ['BTC/USD', 'ETH/USD']
    
    def analyze_portfolio_comprehensive(self, portfolio_data: Dict[str, Any], 
                                      price_history: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Comprehensive portfolio analysis with all advanced metrics"""
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_overview': {},
            'multi_timeframe_analysis': {},
            'risk_metrics': {},
            'performance_metrics': {},
            'beta_analysis': {},
            'sector_analysis': {},
            'drawdown_analysis': {},
            'recommendations': []
        }
        
        try:
            # Portfolio overview
            analysis['portfolio_overview'] = self._calculate_portfolio_overview(portfolio_data)
            
            # Multi-timeframe analysis
            analysis['multi_timeframe_analysis'] = self._multi_timeframe_analysis(
                portfolio_data, price_history
            )
            
            # Risk metrics (Sharpe ratio, volatility, etc.)
            analysis['risk_metrics'] = self._calculate_risk_metrics(
                portfolio_data, price_history
            )
            
            # Performance metrics
            analysis['performance_metrics'] = self._calculate_performance_metrics(
                portfolio_data, price_history
            )
            
            # Beta analysis vs BTC/ETH
            analysis['beta_analysis'] = self._calculate_beta_analysis(
                portfolio_data, price_history
            )
            
            # Sector/category exposure
            analysis['sector_analysis'] = self._analyze_sector_exposure(portfolio_data)
            
            # Drawdown analysis
            analysis['drawdown_analysis'] = self._calculate_drawdown_analysis(
                portfolio_data, price_history
            )
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(analysis)
            
        except Exception as e:
            print(f"⚠️  Error in comprehensive portfolio analysis: {e}")
            
        return analysis
    
    def _calculate_portfolio_overview(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic portfolio overview metrics"""
        overview = {
            'total_value': 0.0,
            'total_assets': 0,
            'largest_position': {'symbol': '', 'percentage': 0.0},
            'cash_percentage': 0.0,
            'crypto_percentage': 100.0,
            'diversification_score': 0.0
        }
        
        try:
            positions = portfolio_data.get('positions', {})
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            
            overview['total_value'] = total_value
            overview['total_assets'] = len(positions)
            
            if total_value > 0:
                # Find largest position
                largest_pos = max(positions.values(), key=lambda x: x.get('value', 0))
                largest_symbol = [k for k, v in positions.items() if v == largest_pos][0]
                
                overview['largest_position'] = {
                    'symbol': largest_symbol,
                    'percentage': largest_pos.get('value', 0) / total_value * 100
                }
                
                # Calculate diversification score (inverse of concentration)
                weights = [pos.get('value', 0) / total_value for pos in positions.values()]
                herfindahl_index = sum(w ** 2 for w in weights)
                overview['diversification_score'] = (1 - herfindahl_index) * 100
                
        except Exception as e:
            print(f"⚠️  Error calculating portfolio overview: {e}")
            
        return overview
    
    def _multi_timeframe_analysis(self, portfolio_data: Dict[str, Any], 
                                price_history: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze portfolio performance across multiple timeframes"""
        
        timeframe_analysis = {}
        
        for timeframe, days in self.timeframes.items():
            try:
                analysis = {
                    'return': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'best_performer': {'symbol': '', 'return': 0.0},
                    'worst_performer': {'symbol': '', 'return': 0.0}
                }
                
                # Calculate portfolio returns for this timeframe
                portfolio_returns = self._calculate_portfolio_returns(
                    portfolio_data, price_history, days
                )
                
                if len(portfolio_returns) > 0:
                    # Total return
                    analysis['return'] = ((portfolio_returns.iloc[-1] / portfolio_returns.iloc[0]) - 1) * 100
                    
                    # Daily returns for volatility calculation
                    daily_returns = portfolio_returns.pct_change().dropna()
                    
                    if len(daily_returns) > 1:
                        # Volatility (annualized)
                        analysis['volatility'] = daily_returns.std() * np.sqrt(365) * 100
                        
                        # Sharpe ratio
                        excess_returns = daily_returns - (self.risk_free_rate / 365)
                        if excess_returns.std() > 0:
                            analysis['sharpe_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(365)
                        
                        # Max drawdown
                        rolling_max = portfolio_returns.expanding().max()
                        drawdown = (portfolio_returns / rolling_max - 1) * 100
                        analysis['max_drawdown'] = drawdown.min()
                        
                        # Win rate
                        positive_days = (daily_returns > 0).sum()
                        analysis['win_rate'] = positive_days / len(daily_returns) * 100
                
                # Individual asset performance
                asset_returns = self._calculate_individual_asset_returns(
                    portfolio_data, price_history, days
                )
                
                if asset_returns:
                    best_asset = max(asset_returns.items(), key=lambda x: x[1])
                    worst_asset = min(asset_returns.items(), key=lambda x: x[1])
                    
                    analysis['best_performer'] = {
                        'symbol': best_asset[0],
                        'return': best_asset[1]
                    }
                    analysis['worst_performer'] = {
                        'symbol': worst_asset[0],
                        'return': worst_asset[1]
                    }
                
                timeframe_analysis[timeframe] = analysis
                
            except Exception as e:
                print(f"⚠️  Error in {timeframe} analysis: {e}")
                timeframe_analysis[timeframe] = {
                    'error': str(e),
                    'return': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0
                }
        
        return timeframe_analysis
    
    def _calculate_risk_metrics(self, portfolio_data: Dict[str, Any], 
                              price_history: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        
        risk_metrics = {
            'sharpe_ratio': {
                '30d': 0.0,
                '90d': 0.0,
                '1y': 0.0
            },
            'sortino_ratio': {
                '30d': 0.0,
                '90d': 0.0,
                '1y': 0.0
            },
            'var_95': 0.0,  # Value at Risk 95%
            'var_99': 0.0,  # Value at Risk 99%
            'expected_shortfall': 0.0,
            'volatility_30d': 0.0,
            'volatility_90d': 0.0,
            'volatility_1y': 0.0,
            'correlation_with_btc': 0.0,
            'correlation_with_eth': 0.0
        }
        
        try:
            # Calculate for different timeframes
            for period_name, days in [('30d', 30), ('90d', 90), ('1y', 365)]:
                portfolio_returns = self._calculate_portfolio_returns(
                    portfolio_data, price_history, days
                )
                
                if len(portfolio_returns) > 1:
                    daily_returns = portfolio_returns.pct_change().dropna()
                    
                    if len(daily_returns) > 0:
                        # Sharpe ratio
                        excess_returns = daily_returns - (self.risk_free_rate / 365)
                        if excess_returns.std() > 0:
                            risk_metrics['sharpe_ratio'][period_name] = (
                                excess_returns.mean() / excess_returns.std() * np.sqrt(365)
                            )
                        
                        # Sortino ratio (using downside deviation)
                        downside_returns = daily_returns[daily_returns < 0]
                        if len(downside_returns) > 0 and downside_returns.std() > 0:
                            risk_metrics['sortino_ratio'][period_name] = (
                                excess_returns.mean() / downside_returns.std() * np.sqrt(365)
                            )
                        
                        # Volatility
                        volatility = daily_returns.std() * np.sqrt(365) * 100
                        risk_metrics[f'volatility_{period_name}'] = volatility
            
            # VaR and Expected Shortfall (using 30-day data)
            portfolio_returns_30d = self._calculate_portfolio_returns(
                portfolio_data, price_history, 30
            )
            
            if len(portfolio_returns_30d) > 1:
                daily_returns = portfolio_returns_30d.pct_change().dropna()
                
                if len(daily_returns) > 0:
                    # Value at Risk
                    risk_metrics['var_95'] = np.percentile(daily_returns, 5) * 100
                    risk_metrics['var_99'] = np.percentile(daily_returns, 1) * 100
                    
                    # Expected Shortfall (Conditional VaR)
                    var_95_threshold = np.percentile(daily_returns, 5)
                    tail_losses = daily_returns[daily_returns <= var_95_threshold]
                    if len(tail_losses) > 0:
                        risk_metrics['expected_shortfall'] = tail_losses.mean() * 100
            
            # Correlation with benchmarks
            risk_metrics['correlation_with_btc'] = self._calculate_benchmark_correlation(
                portfolio_data, price_history, 'BTC/USD'
            )
            risk_metrics['correlation_with_eth'] = self._calculate_benchmark_correlation(
                portfolio_data, price_history, 'ETH/USD'
            )
            
        except Exception as e:
            print(f"⚠️  Error calculating risk metrics: {e}")
            
        return risk_metrics
    
    def _calculate_performance_metrics(self, portfolio_data: Dict[str, Any], 
                                     price_history: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        
        performance = {
            'total_return_1y': 0.0,
            'annualized_return': 0.0,
            'monthly_returns': [],
            'rolling_returns': {},
            'alpha_vs_btc': 0.0,
            'alpha_vs_eth': 0.0,
            'information_ratio_btc': 0.0,
            'information_ratio_eth': 0.0,
            'calmar_ratio': 0.0,
            'sterling_ratio': 0.0
        }
        
        try:
            # 1-year performance
            portfolio_returns_1y = self._calculate_portfolio_returns(
                portfolio_data, price_history, 365
            )
            
            if len(portfolio_returns_1y) > 1:
                total_return = (portfolio_returns_1y.iloc[-1] / portfolio_returns_1y.iloc[0] - 1) * 100
                performance['total_return_1y'] = total_return
                
                # Annualized return
                days = len(portfolio_returns_1y)
                performance['annualized_return'] = ((portfolio_returns_1y.iloc[-1] / portfolio_returns_1y.iloc[0]) ** (365/days) - 1) * 100
                
                # Monthly returns
                monthly_data = portfolio_returns_1y.resample('M').last()
                monthly_returns = monthly_data.pct_change().dropna() * 100
                performance['monthly_returns'] = monthly_returns.tolist()
                
                # Rolling returns (3-month windows)
                if len(portfolio_returns_1y) >= 90:
                    rolling_3m = portfolio_returns_1y.rolling(90).apply(
                        lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100 if len(x) == 90 else np.nan
                    ).dropna()
                    
                    if len(rolling_3m) > 0:
                        performance['rolling_returns'] = {
                            'mean': rolling_3m.mean(),
                            'std': rolling_3m.std(),
                            'max': rolling_3m.max(),
                            'min': rolling_3m.min()
                        }
                    else:
                        performance['rolling_returns'] = {
                            'mean': 0.0,
                            'std': 0.0,
                            'max': 0.0,
                            'min': 0.0
                        }
                else:
                    # Not enough data for rolling returns
                    performance['rolling_returns'] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'max': 0.0,
                        'min': 0.0
                    }
                
                # Alpha and Information Ratio vs benchmarks
                performance['alpha_vs_btc'] = self._calculate_alpha(
                    portfolio_data, price_history, 'BTC/USD'
                )
                performance['alpha_vs_eth'] = self._calculate_alpha(
                    portfolio_data, price_history, 'ETH/USD'
                )
                
                performance['information_ratio_btc'] = self._calculate_information_ratio(
                    portfolio_data, price_history, 'BTC/USD'
                )
                performance['information_ratio_eth'] = self._calculate_information_ratio(
                    portfolio_data, price_history, 'ETH/USD'
                )
                
                # Calmar ratio (annual return / max drawdown)
                daily_returns = portfolio_returns_1y.pct_change().dropna()
                rolling_max = portfolio_returns_1y.expanding().max()
                drawdown = (portfolio_returns_1y / rolling_max - 1)
                max_drawdown = abs(drawdown.min())
                
                if max_drawdown > 0:
                    performance['calmar_ratio'] = performance['annualized_return'] / (max_drawdown * 100)
                
                # Sterling ratio (similar to Calmar but uses average drawdown)
                avg_drawdown = abs(drawdown[drawdown < 0].mean())
                if avg_drawdown > 0:
                    performance['sterling_ratio'] = performance['annualized_return'] / (avg_drawdown * 100)
            
        except Exception as e:
            print(f"⚠️  Error calculating performance metrics: {e}")
            
        return performance
    
    def _calculate_beta_analysis(self, portfolio_data: Dict[str, Any], 
                               price_history: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate portfolio beta vs BTC and ETH"""
        
        beta_analysis = {
            'beta_btc': {
                '30d': 0.0,
                '90d': 0.0,
                '1y': 0.0
            },
            'beta_eth': {
                '30d': 0.0,
                '90d': 0.0,
                '1y': 0.0
            },
            'beta_interpretation': {
                'btc': 'Market Neutral',
                'eth': 'Market Neutral'
            },
            'systematic_risk': 0.0,
            'idiosyncratic_risk': 0.0
        }
        
        try:
            # Calculate beta for different timeframes
            for period_name, days in [('30d', 30), ('90d', 90), ('1y', 365)]:
                # Beta vs BTC
                beta_btc = self._calculate_beta(portfolio_data, price_history, 'BTC/USD', days)
                beta_analysis['beta_btc'][period_name] = beta_btc
                
                # Beta vs ETH
                beta_eth = self._calculate_beta(portfolio_data, price_history, 'ETH/USD', days)
                beta_analysis['beta_eth'][period_name] = beta_eth
            
            # Interpret beta values (using 90-day as reference)
            btc_beta = beta_analysis['beta_btc']['90d']
            eth_beta = beta_analysis['beta_eth']['90d']
            
            # BTC beta interpretation
            if btc_beta > 1.2:
                beta_analysis['beta_interpretation']['btc'] = 'High Beta (More Volatile than BTC)'
            elif btc_beta > 0.8:
                beta_analysis['beta_interpretation']['btc'] = 'Similar Volatility to BTC'
            elif btc_beta > 0.2:
                beta_analysis['beta_interpretation']['btc'] = 'Low Beta (Less Volatile than BTC)'
            else:
                beta_analysis['beta_interpretation']['btc'] = 'Market Neutral'
            
            # ETH beta interpretation
            if eth_beta > 1.2:
                beta_analysis['beta_interpretation']['eth'] = 'High Beta (More Volatile than ETH)'
            elif eth_beta > 0.8:
                beta_analysis['beta_interpretation']['eth'] = 'Similar Volatility to ETH'
            elif eth_beta > 0.2:
                beta_analysis['beta_interpretation']['eth'] = 'Low Beta (Less Volatile than ETH)'
            else:
                beta_analysis['beta_interpretation']['eth'] = 'Market Neutral'
            
            # Systematic vs Idiosyncratic risk
            portfolio_returns = self._calculate_portfolio_returns(portfolio_data, price_history, 90)
            btc_returns = self._get_benchmark_returns('BTC/USD', price_history, 90)
            
            if len(portfolio_returns) > 1 and len(btc_returns) > 1:
                port_daily = portfolio_returns.pct_change().dropna()
                btc_daily = btc_returns.pct_change().dropna()
                
                # Align data
                min_len = min(len(port_daily), len(btc_daily))
                if min_len > 10:
                    port_aligned = port_daily.iloc[-min_len:]
                    btc_aligned = btc_daily.iloc[-min_len:]
                    
                    # Calculate R-squared to determine systematic risk
                    if SCIPY_AVAILABLE:
                        correlation = np.corrcoef(port_aligned, btc_aligned)[0, 1]
                        # Handle NaN correlation (when data is constant)
                        if np.isnan(correlation):
                            correlation = 0.0
                        r_squared = correlation ** 2
                        beta_analysis['systematic_risk'] = r_squared * 100
                        beta_analysis['idiosyncratic_risk'] = (1 - r_squared) * 100
                    else:
                        # Default values when scipy is not available
                        beta_analysis['systematic_risk'] = 50.0
                        beta_analysis['idiosyncratic_risk'] = 50.0
            
        except Exception as e:
            print(f"⚠️  Error calculating beta analysis: {e}")
            
        return beta_analysis
    
    def _analyze_sector_exposure(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze portfolio exposure by crypto sectors/categories"""
        
        sector_analysis = {
            'sector_allocation': {},
            'sector_diversification_score': 0.0,
            'largest_sector': {'name': '', 'percentage': 0.0},
            'sector_concentration_risk': 'LOW',
            'recommended_rebalancing': []
        }
        
        try:
            positions = portfolio_data.get('positions', {})
            total_value = sum(pos.get('value', 0) for pos in positions.values())
            
            if total_value > 0:
                # Calculate sector allocations
                sector_values = {}
                
                for symbol, position in positions.items():
                    # Extract base symbol (remove /USD, /USDT etc.)
                    base_symbol = symbol.split('/')[0].upper()
                    sector = self.crypto_sectors.get(base_symbol, 'Other')
                    
                    if sector not in sector_values:
                        sector_values[sector] = 0
                    sector_values[sector] += position.get('value', 0)
                
                # Convert to percentages
                sector_percentages = {
                    sector: (value / total_value) * 100 
                    for sector, value in sector_values.items()
                }
                
                sector_analysis['sector_allocation'] = sector_percentages
                
                # Find largest sector
                if sector_percentages:
                    largest_sector = max(sector_percentages.items(), key=lambda x: x[1])
                    sector_analysis['largest_sector'] = {
                        'name': largest_sector[0],
                        'percentage': largest_sector[1]
                    }
                
                # Calculate sector diversification (Herfindahl-Hirschman Index)
                weights = list(sector_percentages.values())
                hhi = sum((w/100) ** 2 for w in weights)
                sector_analysis['sector_diversification_score'] = (1 - hhi) * 100
                
                # Assess concentration risk
                max_sector_pct = max(sector_percentages.values()) if sector_percentages else 0
                if max_sector_pct > 70:
                    sector_analysis['sector_concentration_risk'] = 'HIGH'
                elif max_sector_pct > 50:
                    sector_analysis['sector_concentration_risk'] = 'MEDIUM'
                else:
                    sector_analysis['sector_concentration_risk'] = 'LOW'
                
                # Generate rebalancing recommendations
                sector_analysis['recommended_rebalancing'] = self._generate_sector_rebalancing_recommendations(
                    sector_percentages
                )
            
        except Exception as e:
            print(f"⚠️  Error in sector analysis: {e}")
            
        return sector_analysis
    
    def _calculate_drawdown_analysis(self, portfolio_data: Dict[str, Any], 
                                   price_history: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Comprehensive drawdown analysis"""
        
        drawdown_analysis = {
            'current_drawdown': 0.0,
            'max_drawdown_1y': 0.0,
            'max_drawdown_duration': 0,
            'avg_drawdown': 0.0,
            'drawdown_frequency': 0.0,
            'recovery_time_avg': 0.0,
            'underwater_periods': [],
            'drawdown_severity': 'LOW'
        }
        
        try:
            # Get 1-year portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(portfolio_data, price_history, 365)
            
            if len(portfolio_returns) > 1:
                # Calculate rolling maximum (peak values)
                rolling_max = portfolio_returns.expanding().max()
                
                # Calculate drawdown percentage
                drawdown = (portfolio_returns / rolling_max - 1) * 100
                
                # Current drawdown
                drawdown_analysis['current_drawdown'] = drawdown.iloc[-1]
                
                # Maximum drawdown
                drawdown_analysis['max_drawdown_1y'] = drawdown.min()
                
                # Average drawdown (only negative values)
                negative_drawdowns = drawdown[drawdown < 0]
                if len(negative_drawdowns) > 0:
                    drawdown_analysis['avg_drawdown'] = negative_drawdowns.mean()
                
                # Drawdown frequency (percentage of days in drawdown)
                days_in_drawdown = (drawdown < -1).sum()  # More than 1% drawdown
                drawdown_analysis['drawdown_frequency'] = (days_in_drawdown / len(drawdown)) * 100
                
                # Find underwater periods (consecutive days in drawdown)
                underwater_periods = []
                current_period = 0
                
                for dd in drawdown:
                    if dd < -1:  # In drawdown
                        current_period += 1
                    else:
                        if current_period > 0:
                            underwater_periods.append(current_period)
                            current_period = 0
                
                if current_period > 0:  # Still in drawdown
                    underwater_periods.append(current_period)
                
                if underwater_periods:
                    drawdown_analysis['max_drawdown_duration'] = max(underwater_periods)
                    drawdown_analysis['recovery_time_avg'] = np.mean(underwater_periods)
                    drawdown_analysis['underwater_periods'] = underwater_periods
                
                # Assess drawdown severity
                max_dd = abs(drawdown_analysis['max_drawdown_1y'])
                if max_dd > 50:
                    drawdown_analysis['drawdown_severity'] = 'EXTREME'
                elif max_dd > 30:
                    drawdown_analysis['drawdown_severity'] = 'HIGH'
                elif max_dd > 15:
                    drawdown_analysis['drawdown_severity'] = 'MEDIUM'
                else:
                    drawdown_analysis['drawdown_severity'] = 'LOW'
            
        except Exception as e:
            print(f"⚠️  Error in drawdown analysis: {e}")
            
        return drawdown_analysis
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate portfolio recommendations based on analysis"""
        recommendations = []
        
        try:
            # Risk-based recommendations
            risk_metrics = analysis.get('risk_metrics', {})
            sharpe_30d = risk_metrics.get('sharpe_ratio', {}).get('30d', 0)
            volatility_30d = risk_metrics.get('volatility_30d', 0)
            
            if sharpe_30d < 0.5:
                recommendations.append("IMPROVE_RISK_ADJUSTED_RETURNS: Consider reducing position sizes or adding defensive assets")
            
            if volatility_30d > 50:
                recommendations.append("REDUCE_VOLATILITY: Portfolio volatility is high, consider adding stable assets or reducing leverage")
            
            # Sector diversification recommendations
            sector_analysis = analysis.get('sector_analysis', {})
            concentration_risk = sector_analysis.get('sector_concentration_risk', 'LOW')
            
            if concentration_risk == 'HIGH':
                recommendations.append("DIVERSIFY_SECTORS: High sector concentration detected, consider diversifying across different crypto categories")
            
            # Drawdown recommendations
            drawdown_analysis = analysis.get('drawdown_analysis', {})
            current_drawdown = drawdown_analysis.get('current_drawdown', 0)
            max_drawdown = drawdown_analysis.get('max_drawdown_1y', 0)
            
            if current_drawdown < -20:
                recommendations.append("CURRENT_DRAWDOWN_HIGH: Portfolio is in significant drawdown, consider risk management measures")
            
            if max_drawdown < -40:
                recommendations.append("IMPROVE_DOWNSIDE_PROTECTION: Historical max drawdown is severe, implement better risk controls")
            
            # Beta recommendations
            beta_analysis = analysis.get('beta_analysis', {})
            btc_beta = beta_analysis.get('beta_btc', {}).get('90d', 0)
            
            if btc_beta > 1.5:
                recommendations.append("HIGH_MARKET_SENSITIVITY: Portfolio is highly sensitive to BTC movements, consider reducing correlation")
            
            # Performance recommendations
            performance = analysis.get('performance_metrics', {})
            alpha_btc = performance.get('alpha_vs_btc', 0)
            
            if alpha_btc < 0:
                recommendations.append("NEGATIVE_ALPHA: Portfolio underperforming vs BTC, review asset selection and strategy")
            
            # Timeframe analysis recommendations
            timeframe_analysis = analysis.get('multi_timeframe_analysis', {})
            if '30d' in timeframe_analysis and '90d' in timeframe_analysis:
                return_30d = timeframe_analysis['30d'].get('return', 0)
                return_90d = timeframe_analysis['90d'].get('return', 0)
                
                if return_30d < -15 and return_90d > 0:
                    recommendations.append("SHORT_TERM_WEAKNESS: Recent performance declining despite positive medium-term trend")
            
        except Exception as e:
            print(f"⚠️  Error generating recommendations: {e}")
            recommendations.append("ERROR_IN_ANALYSIS: Some analysis components failed, manual review recommended")
        
        return recommendations
    
    # Helper methods
    def _calculate_portfolio_returns(self, portfolio_data: Dict[str, Any], 
                                   price_history: Dict[str, pd.DataFrame], 
                                   days: int) -> pd.Series:
        """Calculate portfolio value time series"""
        try:
            positions = portfolio_data.get('positions', {})
            if not positions:
                return pd.Series()
            
            # Get the latest date from price history if available, otherwise use current time
            end_date = datetime.now()
            for symbol, data in price_history.items():
                if symbol in positions and 'close' in data.columns and len(data) > 0:
                    latest_date = data.index.max()
                    if latest_date > pd.Timestamp(end_date - timedelta(days=30)):  # Only use if recent-ish
                        end_date = latest_date
                    else:
                        # Use the latest date from the data
                        end_date = max(data.index.max() for data in price_history.values() 
                                     if len(data) > 0 and hasattr(data.index, 'max'))
                    break
            
            start_date = end_date - timedelta(days=days)
            
            portfolio_values = pd.Series(dtype=float)
            
            for symbol, position in positions.items():
                if symbol in price_history and 'close' in price_history[symbol].columns:
                    prices = price_history[symbol]['close']
                    
                    # Filter by date range
                    mask = (prices.index >= start_date) & (prices.index <= end_date)
                    asset_prices = prices.loc[mask]
                    
                    if len(asset_prices) > 0:
                        # Calculate position value over time
                        quantity = position.get('quantity', 0)
                        asset_values = asset_prices * quantity
                        
                        if len(portfolio_values) == 0:
                            portfolio_values = asset_values
                        else:
                            # Align series and add
                            portfolio_values = portfolio_values.add(asset_values, fill_value=0)
            
            return portfolio_values
            
        except Exception as e:
            print(f"⚠️  Error calculating portfolio returns: {e}")
            return pd.Series()
    
    def _calculate_individual_asset_returns(self, portfolio_data: Dict[str, Any], 
                                          price_history: Dict[str, pd.DataFrame], 
                                          days: int) -> Dict[str, float]:
        """Calculate individual asset returns"""
        asset_returns = {}
        
        try:
            positions = portfolio_data.get('positions', {})
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            for symbol in positions.keys():
                if symbol in price_history and 'close' in price_history[symbol].columns:
                    prices = price_history[symbol]['close']
                    
                    # Filter by date range
                    mask = (prices.index >= start_date) & (prices.index <= end_date)
                    asset_prices = prices.loc[mask]
                    
                    if len(asset_prices) > 1:
                        return_pct = ((asset_prices.iloc[-1] / asset_prices.iloc[0]) - 1) * 100
                        asset_returns[symbol] = return_pct
                        
        except Exception as e:
            print(f"⚠️  Error calculating individual asset returns: {e}")
            
        return asset_returns
    
    def _calculate_beta(self, portfolio_data: Dict[str, Any], 
                       price_history: Dict[str, pd.DataFrame], 
                       benchmark: str, days: int) -> float:
        """Calculate portfolio beta vs benchmark"""
        try:
            portfolio_returns = self._calculate_portfolio_returns(portfolio_data, price_history, days)
            benchmark_returns = self._get_benchmark_returns(benchmark, price_history, days)
            
            if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
                port_daily = portfolio_returns.pct_change().dropna()
                bench_daily = benchmark_returns.pct_change().dropna()
                
                # Align data
                min_len = min(len(port_daily), len(bench_daily))
                if min_len > 10:
                    port_aligned = port_daily.iloc[-min_len:]
                    bench_aligned = bench_daily.iloc[-min_len:]
                    
                    # Calculate beta
                    if SCIPY_AVAILABLE:
                        covariance = np.cov(port_aligned, bench_aligned)[0, 1]
                        benchmark_variance = np.var(bench_aligned)
                        if benchmark_variance > 0:
                            return covariance / benchmark_variance
            
            return 0.0
            
        except Exception as e:
            print(f"⚠️  Error calculating beta: {e}")
            return 0.0
    
    def _get_benchmark_returns(self, benchmark: str, 
                              price_history: Dict[str, pd.DataFrame], 
                              days: int) -> pd.Series:
        """Get benchmark return time series"""
        try:
            if benchmark in price_history and 'close' in price_history[benchmark].columns:
                # Use the latest date from the benchmark data
                benchmark_data = price_history[benchmark]
                if len(benchmark_data) > 0:
                    end_date = benchmark_data.index.max()
                else:
                    end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                prices = price_history[benchmark]['close']
                mask = (prices.index >= start_date) & (prices.index <= end_date)
                return prices.loc[mask]
                
            return pd.Series()
            
        except Exception as e:
            print(f"⚠️  Error getting benchmark returns: {e}")
            return pd.Series()
    
    def _calculate_benchmark_correlation(self, portfolio_data: Dict[str, Any], 
                                       price_history: Dict[str, pd.DataFrame], 
                                       benchmark: str) -> float:
        """Calculate correlation with benchmark"""
        try:
            portfolio_returns = self._calculate_portfolio_returns(portfolio_data, price_history, 90)
            benchmark_returns = self._get_benchmark_returns(benchmark, price_history, 90)
            
            if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
                port_daily = portfolio_returns.pct_change().dropna()
                bench_daily = benchmark_returns.pct_change().dropna()
                
                min_len = min(len(port_daily), len(bench_daily))
                if min_len > 10:
                    port_aligned = port_daily.iloc[-min_len:]
                    bench_aligned = bench_daily.iloc[-min_len:]
                    
                    return np.corrcoef(port_aligned, bench_aligned)[0, 1]
                    
            return 0.0
            
        except Exception as e:
            print(f"⚠️  Error calculating correlation: {e}")
            return 0.0
    
    def _calculate_alpha(self, portfolio_data: Dict[str, Any], 
                        price_history: Dict[str, pd.DataFrame], 
                        benchmark: str) -> float:
        """Calculate portfolio alpha vs benchmark"""
        try:
            portfolio_returns = self._calculate_portfolio_returns(portfolio_data, price_history, 365)
            benchmark_returns = self._get_benchmark_returns(benchmark, price_history, 365)
            
            if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
                # Calculate annualized returns
                port_annual = ((portfolio_returns.iloc[-1] / portfolio_returns.iloc[0]) ** (365/len(portfolio_returns)) - 1) * 100
                bench_annual = ((benchmark_returns.iloc[-1] / benchmark_returns.iloc[0]) ** (365/len(benchmark_returns)) - 1) * 100
                
                # Alpha = Portfolio Return - Benchmark Return
                return port_annual - bench_annual
                
            return 0.0
            
        except Exception as e:
            print(f"⚠️  Error calculating alpha: {e}")
            return 0.0
    
    def _calculate_information_ratio(self, portfolio_data: Dict[str, Any], 
                                   price_history: Dict[str, pd.DataFrame], 
                                   benchmark: str) -> float:
        """Calculate information ratio vs benchmark"""
        try:
            portfolio_returns = self._calculate_portfolio_returns(portfolio_data, price_history, 365)
            benchmark_returns = self._get_benchmark_returns(benchmark, price_history, 365)
            
            if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
                port_daily = portfolio_returns.pct_change().dropna()
                bench_daily = benchmark_returns.pct_change().dropna()
                
                min_len = min(len(port_daily), len(bench_daily))
                if min_len > 10:
                    port_aligned = port_daily.iloc[-min_len:]
                    bench_aligned = bench_daily.iloc[-min_len:]
                    
                    # Active returns
                    active_returns = port_aligned - bench_aligned
                    
                    if active_returns.std() > 0:
                        # Information Ratio = Mean Active Return / Std of Active Returns
                        return (active_returns.mean() / active_returns.std()) * np.sqrt(365)
                        
            return 0.0
            
        except Exception as e:
            print(f"⚠️  Error calculating information ratio: {e}")
            return 0.0
    
    def _generate_sector_rebalancing_recommendations(self, sector_percentages: Dict[str, float]) -> List[str]:
        """Generate sector rebalancing recommendations"""
        recommendations = []
        
        try:
            # Ideal diversified allocation targets
            target_allocations = {
                'Store of Value': 30,      # BTC
                'Smart Contract Platform': 25,  # ETH, ADA, SOL, etc.
                'DeFi': 15,               # AAVE, UNI, etc.
                'Payment': 10,            # LTC, XRP, etc.
                'Other': 20               # Various smaller sectors
            }
            
            for sector, current_pct in sector_percentages.items():
                target_pct = target_allocations.get(sector, 5)
                
                if current_pct > target_pct * 1.5:  # More than 50% over target
                    recommendations.append(f"REDUCE_{sector.upper()}: Current {current_pct:.1f}%, target ~{target_pct}%")
                elif current_pct < target_pct * 0.5:  # Less than 50% of target
                    recommendations.append(f"INCREASE_{sector.upper()}: Current {current_pct:.1f}%, target ~{target_pct}%")
                    
        except Exception as e:
            print(f"⚠️  Error generating sector recommendations: {e}")
            
        return recommendations

if __name__ == "__main__":
    # Test the portfolio analyzer
    analyzer = PortfolioAnalyzer()
    
    # Create sample portfolio data
    sample_portfolio = {
        'positions': {
            'BTC/USD': {'quantity': 0.5, 'value': 25000},
            'ETH/USD': {'quantity': 10, 'value': 15000},
            'ADA/USD': {'quantity': 1000, 'value': 500},
            'SOL/USD': {'quantity': 50, 'value': 2500}
        }
    }
    
    # Create sample price history
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    sample_price_history = {}
    
    for symbol in sample_portfolio['positions'].keys():
        sample_price_history[symbol] = pd.DataFrame({
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        }, index=dates)
    
    # Run comprehensive analysis
    analysis = analyzer.analyze_portfolio_comprehensive(sample_portfolio, sample_price_history)
    
    print("📊 Portfolio Analysis Results:")
    print(json.dumps(analysis, indent=2, default=str))