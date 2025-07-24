#!/usr/bin/env python3
"""
📊 QUANTITATIVE PORTFOLIO ADVISOR
=================================
Professional-grade quantitative analysis of your Kraken portfolio
Uses advanced mathematical models and statistical analysis to provide actionable insights
"""

import sys
import os
import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.kraken import kraken_api

class QuantitativeAdvisor:
    """Professional quantitative portfolio advisor"""
    
    def __init__(self):
        self.portfolio_data = {}
        self.market_data = {}
        self.analysis_timestamp = datetime.now()
        
        # Quant model parameters
        self.lookback_period = 252  # 1 year for volatility calculations
        self.short_ma = 20
        self.long_ma = 50
        self.rsi_period = 14
        self.var_confidence = 0.05  # 95% VaR
        
        print("📊 Quantitative Portfolio Advisor Initialized")
        print("🔬 Mathematical models loaded: Sharpe, Sortino, VaR, Beta, Correlation Analysis")

    async def analyze_portfolio(self) -> Dict:
        """Main quantitative analysis function"""
        print("\n" + "="*70)
        print("🎯 QUANTITATIVE PORTFOLIO ANALYSIS")
        print("="*70)
        
        try:
            # Step 1: Get actual portfolio data
            portfolio = await self._get_actual_portfolio()
            if not portfolio:
                print("❌ Could not retrieve portfolio data")
                return {}
            
            # Step 2: Fetch market data for all assets
            print(f"\n📥 Fetching market data for {len(portfolio)} assets...")
            market_data = await self._fetch_comprehensive_market_data(portfolio)
            
            # Step 3: Run quantitative analysis
            analysis = await self._run_quantitative_analysis(portfolio, market_data)
            
            # Step 4: Generate recommendations
            recommendations = await self._generate_quant_recommendations(analysis)
            
            # Step 5: Display results
            self._display_professional_analysis(analysis, recommendations)
            
            # Step 6: Save analysis
            self._save_analysis(analysis, recommendations)
            
            return {
                'portfolio': portfolio,
                'analysis': analysis,
                'recommendations': recommendations,
                'timestamp': self.analysis_timestamp.isoformat()
            }
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    async def _get_actual_portfolio(self) -> Dict:
        """Get actual portfolio from Kraken"""
        try:
            print("🔍 Retrieving live portfolio data from Kraken...")
            
            # Get balance
            balance_response = kraken_api.get_balance()
            if 'result' not in balance_response:
                print("⚠️  Could not get balance from Kraken")
                return {}
            
            balance = balance_response['result']
            portfolio = {}
            
            print("\n💰 CURRENT PORTFOLIO HOLDINGS:")
            print("-" * 40)
            
            total_usd_value = 0.0
            
            for asset, amount_str in balance.items():
                try:
                    amount = float(amount_str)
                    if amount > 0.01:  # Only meaningful amounts
                        
                        # Get current price
                        if asset == 'ZUSD':
                            current_price = 1.0
                            usd_value = amount
                        else:
                            current_price = await self._get_asset_price(asset)
                            usd_value = amount * current_price if current_price > 0 else 0
                        
                        if usd_value > 1:  # Only assets worth more than $1
                            portfolio[asset] = {
                                'amount': amount,
                                'current_price': current_price,
                                'usd_value': usd_value,
                                'symbol': self._convert_to_yfinance_symbol(asset)
                            }
                            total_usd_value += usd_value
                            
                            print(f"   {asset:<12} {amount:>12.6f} @ ${current_price:>8.2f} = ${usd_value:>10.2f}")
                
                except Exception as e:
                    continue
            
            print(f"\n💎 Total Portfolio Value: ${total_usd_value:,.2f}")
            print(f"📊 Number of Assets: {len(portfolio)}")
            
            # Calculate percentages
            for asset in portfolio:
                portfolio[asset]['percentage'] = (portfolio[asset]['usd_value'] / total_usd_value) * 100
            
            return portfolio
            
        except Exception as e:
            print(f"❌ Portfolio retrieval failed: {e}")
            return {}

    async def _get_asset_price(self, asset: str) -> float:
        """Get current price for an asset"""
        try:
            # Special handling for different asset types
            if asset == 'ZUSD':
                return 1.0
            
            # Try Kraken first for crypto
            kraken_symbols = ['XXBT', 'XETH', 'XXRP', 'ADA', 'SOL', 'XXDG', 'XXLM']
            if any(asset.startswith(symbol) for symbol in kraken_symbols):
                try:
                    # Map to correct Kraken format
                    symbol_map = {
                        'XXBT': 'XBTUSD',
                        'XETH': 'ETHUSD', 
                        'XXRP': 'XRPUSD',
                        'ADA': 'ADAEUR',
                        'SOL': 'SOLEUR',
                        'XXDG': 'DOGEUSD',
                        'XXLM': 'XLMUSD'
                    }
                    
                    for prefix, kraken_pair in symbol_map.items():
                        if asset.startswith(prefix):
                            price = kraken_api.get_price(kraken_pair)
                            if price > 0:
                                # Convert EUR to USD if needed
                                if 'EUR' in kraken_pair:
                                    eur_usd_rate = 1.1  # Approximate rate
                                    price = price * eur_usd_rate
                                return price
                except:
                    pass
            
            # Fallback to yfinance
            yf_symbol = self._convert_to_yfinance_symbol(asset)
            if yf_symbol:
                ticker = yf.Ticker(yf_symbol)
                hist = ticker.history(period='1d')
                if not hist.empty:
                    return float(hist['Close'].iloc[-1])
            
            return 0.0
            
        except Exception as e:
            return 0.0

    def _convert_to_yfinance_symbol(self, kraken_asset: str) -> str:
        """Convert Kraken asset to yfinance symbol"""
        symbol_map = {
            'XXBT': 'BTC-USD',
            'XETH': 'ETH-USD',
            'XXRP': 'XRP-USD',
            'ADA': 'ADA-USD',
            'SOL': 'SOL-USD',
            'XXDG': 'DOGE-USD',
            'XXLM': 'XLM-USD',
            'ZUSD': None  # USD cash
        }
        
        for kraken_prefix, yf_symbol in symbol_map.items():
            if kraken_asset.startswith(kraken_prefix):
                return yf_symbol
        
        # If no mapping found, try as-is with -USD suffix
        if kraken_asset not in ['ZUSD'] and not kraken_asset.endswith('.EQ'):
            return f"{kraken_asset}-USD"
        
        return None

    async def _fetch_comprehensive_market_data(self, portfolio: Dict) -> Dict:
        """Fetch comprehensive market data for quantitative analysis"""
        market_data = {}
        
        for asset, data in portfolio.items():
            yf_symbol = data['symbol']
            if not yf_symbol:
                continue
            
            try:
                print(f"📊 Fetching data for {asset} ({yf_symbol})...")
                ticker = yf.Ticker(yf_symbol)
                
                # Get multiple timeframes
                hist_1y = ticker.history(period='1y')
                hist_3m = ticker.history(period='3mo')
                hist_1m = ticker.history(period='1mo')
                
                if not hist_1y.empty:
                    market_data[asset] = {
                        'symbol': yf_symbol,
                        'hist_1y': hist_1y,
                        'hist_3m': hist_3m,
                        'hist_1m': hist_1m,
                        'current_price': data['current_price']
                    }
                    
            except Exception as e:
                print(f"⚠️  Failed to fetch data for {asset}: {e}")
                continue
        
        print(f"✅ Market data fetched for {len(market_data)} assets")
        return market_data

    async def _run_quantitative_analysis(self, portfolio: Dict, market_data: Dict) -> Dict:
        """Run comprehensive quantitative analysis"""
        print("\n🔬 RUNNING QUANTITATIVE MODELS...")
        print("-" * 50)
        
        analysis = {
            'portfolio_metrics': {},
            'asset_analysis': {},
            'risk_metrics': {},
            'performance_metrics': {},
            'correlation_analysis': {},
            'technical_indicators': {}
        }
        
        # Portfolio-level metrics
        analysis['portfolio_metrics'] = self._calculate_portfolio_metrics(portfolio, market_data)
        
        # Individual asset analysis
        for asset, asset_data in portfolio.items():
            if asset in market_data:
                analysis['asset_analysis'][asset] = self._analyze_individual_asset(
                    asset, asset_data, market_data[asset]
                )
        
        # Risk analysis
        analysis['risk_metrics'] = self._calculate_risk_metrics(portfolio, market_data)
        
        # Performance analysis
        analysis['performance_metrics'] = self._calculate_performance_metrics(portfolio, market_data)
        
        # Correlation analysis
        analysis['correlation_analysis'] = self._calculate_correlation_matrix(market_data)
        
        # Technical indicators
        analysis['technical_indicators'] = self._calculate_technical_indicators(market_data)
        
        return analysis

    def _calculate_portfolio_metrics(self, portfolio: Dict, market_data: Dict) -> Dict:
        """Calculate portfolio-level metrics"""
        total_value = sum(asset['usd_value'] for asset in portfolio.values())
        
        # Concentration metrics
        max_position = max(asset['percentage'] for asset in portfolio.values())
        hhi = sum((asset['percentage']/100)**2 for asset in portfolio.values())
        diversification_ratio = 1 - hhi
        
        # Asset type breakdown
        crypto_allocation = 0
        cash_allocation = 0
        
        for asset, data in portfolio.items():
            if asset == 'ZUSD':
                cash_allocation += data['percentage']
            else:
                crypto_allocation += data['percentage']
        
        return {
            'total_value': total_value,
            'number_of_positions': len(portfolio),
            'max_position_size': max_position,
            'hhi_concentration': hhi,
            'diversification_ratio': diversification_ratio,
            'crypto_allocation': crypto_allocation,
            'cash_allocation': cash_allocation,
            'concentration_risk': 'HIGH' if max_position > 40 else 'MODERATE' if max_position > 25 else 'LOW'
        }

    def _analyze_individual_asset(self, asset: str, asset_data: Dict, market_data: Dict) -> Dict:
        """Analyze individual asset using quantitative methods"""
        try:
            hist = market_data['hist_1y']
            if hist.empty:
                return {}
            
            # Calculate returns
            returns = hist['Close'].pct_change().dropna()
            
            # Risk-adjusted metrics
            annual_return = returns.mean() * 252
            annual_volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
            
            # Downside metrics
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = annual_return / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Value at Risk (5%)
            var_5 = np.percentile(returns, 5)
            
            # Beta (if we have market data - using BTC as proxy for crypto market)
            beta = 0
            try:
                if asset != 'XXBT' and 'XXBT' in [a for a in asset_data.keys()]:
                    # Calculate beta against BTC
                    market_returns = returns  # Simplified - would need actual market data
                    covariance = np.cov(returns, market_returns)[0][1]
                    market_variance = np.var(market_returns)
                    beta = covariance / market_variance if market_variance > 0 else 0
            except:
                pass
            
            return {
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'var_5': var_5,
                'beta': beta,
                'current_allocation': asset_data['percentage']
            }
            
        except Exception as e:
            return {}

    def _calculate_risk_metrics(self, portfolio: Dict, market_data: Dict) -> Dict:
        """Calculate portfolio risk metrics"""
        try:
            # Collect all returns for portfolio-level analysis
            all_returns = []
            weights = []
            
            for asset, asset_data in portfolio.items():
                if asset in market_data:
                    hist = market_data[asset]['hist_3m']
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna()
                        if len(returns) > 0:
                            all_returns.append(returns)
                            weights.append(asset_data['percentage'] / 100)
            
            if not all_returns:
                return {}
            
            # Portfolio VaR calculation (simplified)
            portfolio_var = 0
            total_volatility = 0
            
            for i, returns in enumerate(all_returns):
                weight = weights[i]
                asset_var = np.percentile(returns, 5)
                volatility = returns.std() * np.sqrt(252)
                
                portfolio_var += weight * asset_var
                total_volatility += weight * volatility
            
            return {
                'portfolio_var_5': portfolio_var,
                'portfolio_volatility': total_volatility,
                'risk_level': 'HIGH' if total_volatility > 0.4 else 'MODERATE' if total_volatility > 0.2 else 'LOW'
            }
            
        except Exception as e:
            return {}

    def _calculate_performance_metrics(self, portfolio: Dict, market_data: Dict) -> Dict:
        """Calculate performance metrics"""
        try:
            # Portfolio performance over different periods
            performance_1m = 0
            performance_3m = 0
            
            for asset, asset_data in portfolio.items():
                if asset in market_data:
                    weight = asset_data['percentage'] / 100
                    
                    # 1 month performance
                    hist_1m = market_data[asset]['hist_1m']
                    if not hist_1m.empty and len(hist_1m) >= 2:
                        perf_1m = (hist_1m['Close'].iloc[-1] - hist_1m['Close'].iloc[0]) / hist_1m['Close'].iloc[0]
                        performance_1m += weight * perf_1m
                    
                    # 3 month performance
                    hist_3m = market_data[asset]['hist_3m']
                    if not hist_3m.empty and len(hist_3m) >= 2:
                        perf_3m = (hist_3m['Close'].iloc[-1] - hist_3m['Close'].iloc[0]) / hist_3m['Close'].iloc[0]
                        performance_3m += weight * perf_3m
            
            return {
                'performance_1m': performance_1m,
                'performance_3m': performance_3m,
                'annualized_3m': performance_3m * 4,  # Rough annualization
            }
            
        except Exception as e:
            return {}

    def _calculate_correlation_matrix(self, market_data: Dict) -> Dict:
        """Calculate correlation matrix between assets"""
        try:
            if len(market_data) < 2:
                return {}
            
            # Build returns matrix
            returns_data = {}
            for asset, data in market_data.items():
                hist = data['hist_3m']
                if not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    if len(returns) > 30:  # Need sufficient data
                        returns_data[asset] = returns
            
            if len(returns_data) < 2:
                return {}
            
            # Create DataFrame and calculate correlation
            df = pd.DataFrame(returns_data).fillna(0)
            correlation_matrix = df.corr()
            
            # Find highest and lowest correlations
            corr_values = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_values.append({
                        'asset1': correlation_matrix.columns[i],
                        'asset2': correlation_matrix.columns[j],
                        'correlation': correlation_matrix.iloc[i, j]
                    })
            
            if corr_values:
                highest_corr = max(corr_values, key=lambda x: x['correlation'])
                lowest_corr = min(corr_values, key=lambda x: x['correlation'])
                avg_correlation = np.mean([c['correlation'] for c in corr_values])
                
                return {
                    'avg_correlation': avg_correlation,
                    'highest_correlation': highest_corr,
                    'lowest_correlation': lowest_corr,
                    'diversification_benefit': 'HIGH' if avg_correlation < 0.3 else 'MODERATE' if avg_correlation < 0.6 else 'LOW'
                }
            
            return {}
            
        except Exception as e:
            return {}

    def _calculate_technical_indicators(self, market_data: Dict) -> Dict:
        """Calculate technical indicators for each asset"""
        technical_analysis = {}
        
        for asset, data in market_data.items():
            try:
                hist = data['hist_3m']
                if hist.empty:
                    continue
                
                closes = hist['Close']
                
                # Moving averages
                sma_20 = closes.rolling(window=20).mean().iloc[-1] if len(closes) >= 20 else closes.mean()
                sma_50 = closes.rolling(window=50).mean().iloc[-1] if len(closes) >= 50 else closes.mean()
                current_price = closes.iloc[-1]
                
                # RSI
                delta = closes.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1] if not rsi.empty else 50
                
                # Momentum
                momentum_20 = (current_price - closes.iloc[-20]) / closes.iloc[-20] * 100 if len(closes) >= 20 else 0
                
                technical_analysis[asset] = {
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'rsi': current_rsi,
                    'momentum_20d': momentum_20,
                    'trend': 'BULLISH' if current_price > sma_20 > sma_50 else 'BEARISH' if current_price < sma_20 < sma_50 else 'NEUTRAL'
                }
                
            except Exception as e:
                continue
        
        return technical_analysis

    async def _generate_quant_recommendations(self, analysis: Dict) -> List[Dict]:
        """Generate quantitative recommendations based on analysis"""
        recommendations = []
        
        try:
            portfolio_metrics = analysis.get('portfolio_metrics', {})
            asset_analysis = analysis.get('asset_analysis', {})
            risk_metrics = analysis.get('risk_metrics', {})
            correlation_analysis = analysis.get('correlation_analysis', {})
            technical_indicators = analysis.get('technical_indicators', {})
            
            # Portfolio-level recommendations
            if portfolio_metrics.get('concentration_risk') == 'HIGH':
                recommendations.append({
                    'type': 'PORTFOLIO_STRUCTURE',
                    'priority': 'HIGH',
                    'recommendation': f"REDUCE CONCENTRATION RISK: Maximum position is {portfolio_metrics.get('max_position_size', 0):.1f}%. Consider rebalancing to reduce single-asset risk.",
                    'mathematical_basis': f"HHI Concentration Index: {portfolio_metrics.get('hhi_concentration', 0):.3f} (>0.25 indicates high concentration)",
                    'action': 'REBALANCE - Reduce largest positions to <25% each'
                })
            
            # Risk-adjusted performance recommendations
            for asset, metrics in asset_analysis.items():
                sharpe = metrics.get('sharpe_ratio', 0)
                sortino = metrics.get('sortino_ratio', 0)
                allocation = metrics.get('current_allocation', 0)
                
                if sharpe > 1.5 and allocation < 20:
                    recommendations.append({
                        'type': 'POSITION_SIZING',
                        'priority': 'MEDIUM',
                        'recommendation': f"INCREASE {asset}: Excellent risk-adjusted returns (Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}). Current allocation: {allocation:.1f}%",
                        'mathematical_basis': f"Sharpe Ratio: {sharpe:.3f} (>1.5 excellent), Sortino: {sortino:.3f}",
                        'action': f'INCREASE {asset} allocation to 15-25%'
                    })
                
                elif sharpe < 0 and allocation > 10:
                    recommendations.append({
                        'type': 'POSITION_SIZING',
                        'priority': 'HIGH',
                        'recommendation': f"REDUCE {asset}: Poor risk-adjusted returns (Sharpe: {sharpe:.2f}). Current allocation: {allocation:.1f}%",
                        'mathematical_basis': f"Negative Sharpe Ratio: {sharpe:.3f} indicates poor risk-adjusted performance",
                        'action': f'REDUCE {asset} allocation or EXIT position'
                    })
            
            # Technical recommendations
            for asset, tech_data in technical_indicators.items():
                rsi = tech_data.get('rsi', 50)
                trend = tech_data.get('trend', 'NEUTRAL')
                momentum = tech_data.get('momentum_20d', 0)
                
                if rsi > 70 and trend == 'BULLISH':
                    recommendations.append({
                        'type': 'TECHNICAL_SIGNAL',
                        'priority': 'MEDIUM',
                        'recommendation': f"{asset} OVERBOUGHT: RSI {rsi:.1f} in bullish trend. Consider taking profits.",
                        'mathematical_basis': f"RSI: {rsi:.1f} (>70 overbought), 20-day momentum: {momentum:.1f}%",
                        'action': f'TAKE PARTIAL PROFITS in {asset}'
                    })
                
                elif rsi < 30 and momentum > -10:
                    recommendations.append({
                        'type': 'TECHNICAL_SIGNAL',
                        'priority': 'MEDIUM',
                        'recommendation': f"{asset} OVERSOLD BUY OPPORTUNITY: RSI {rsi:.1f} with limited downside momentum.",
                        'mathematical_basis': f"RSI: {rsi:.1f} (<30 oversold), momentum: {momentum:.1f}%",
                        'action': f'CONSIDER BUYING {asset} on weakness'
                    })
            
            # Correlation-based recommendations
            if correlation_analysis.get('diversification_benefit') == 'LOW':
                recommendations.append({
                    'type': 'DIVERSIFICATION',
                    'priority': 'HIGH',
                    'recommendation': f"IMPROVE DIVERSIFICATION: Average correlation {correlation_analysis.get('avg_correlation', 0):.2f} is too high. Assets move together, reducing portfolio benefits.",
                    'mathematical_basis': f"Average correlation: {correlation_analysis.get('avg_correlation', 0):.3f} (>0.6 indicates poor diversification)",
                    'action': 'ADD UNCORRELATED ASSETS or reduce correlated positions'
                })
            
            # Sort by priority
            priority_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
            recommendations.sort(key=lambda x: priority_order.get(x['priority'], 0), reverse=True)
            
            return recommendations[:10]  # Top 10 recommendations
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []

    def _display_professional_analysis(self, analysis: Dict, recommendations: List[Dict]):
        """Display professional quantitative analysis results"""
        print("\n" + "="*80)
        print("📊 QUANTITATIVE ANALYSIS RESULTS")
        print("="*80)
        
        # Portfolio Overview
        portfolio_metrics = analysis.get('portfolio_metrics', {})
        print(f"\n💰 PORTFOLIO OVERVIEW:")
        print(f"   Total Value: ${portfolio_metrics.get('total_value', 0):,.2f}")
        print(f"   Number of Positions: {portfolio_metrics.get('number_of_positions', 0)}")
        print(f"   Largest Position: {portfolio_metrics.get('max_position_size', 0):.1f}%")
        print(f"   Diversification Ratio: {portfolio_metrics.get('diversification_ratio', 0):.3f}")
        print(f"   Concentration Risk: {portfolio_metrics.get('concentration_risk', 'UNKNOWN')}")
        
        # Risk Metrics
        risk_metrics = analysis.get('risk_metrics', {})
        if risk_metrics:
            print(f"\n⚠️  RISK ANALYSIS:")
            print(f"   Portfolio VaR (5%): {risk_metrics.get('portfolio_var_5', 0)*100:.2f}%")
            print(f"   Estimated Volatility: {risk_metrics.get('portfolio_volatility', 0)*100:.1f}%")
            print(f"   Risk Level: {risk_metrics.get('risk_level', 'UNKNOWN')}")
        
        # Performance
        performance = analysis.get('performance_metrics', {})
        if performance:
            print(f"\n📈 PERFORMANCE METRICS:")
            print(f"   1-Month Return: {performance.get('performance_1m', 0)*100:+.2f}%")
            print(f"   3-Month Return: {performance.get('performance_3m', 0)*100:+.2f}%")
            print(f"   Annualized (est.): {performance.get('annualized_3m', 0)*100:+.1f}%")
        
        # Asset Analysis
        asset_analysis = analysis.get('asset_analysis', {})
        if asset_analysis:
            print(f"\n🔍 INDIVIDUAL ASSET ANALYSIS:")
            print("   Asset        Allocation   Sharpe   Sortino   Max DD    Annual Vol")
            print("   " + "-"*70)
            for asset, metrics in asset_analysis.items():
                print(f"   {asset:<12} {metrics.get('current_allocation', 0):>8.1f}%   "
                      f"{metrics.get('sharpe_ratio', 0):>6.2f}   "
                      f"{metrics.get('sortino_ratio', 0):>7.2f}   "
                      f"{metrics.get('max_drawdown', 0)*100:>6.1f}%   "
                      f"{metrics.get('annual_volatility', 0)*100:>8.1f}%")
        
        # Correlation Analysis
        correlation = analysis.get('correlation_analysis', {})
        if correlation:
            print(f"\n🔗 CORRELATION ANALYSIS:")
            print(f"   Average Correlation: {correlation.get('avg_correlation', 0):.3f}")
            print(f"   Diversification Benefit: {correlation.get('diversification_benefit', 'UNKNOWN')}")
            
            highest = correlation.get('highest_correlation', {})
            if highest:
                print(f"   Highest Correlation: {highest.get('asset1', 'N/A')} vs {highest.get('asset2', 'N/A')} ({highest.get('correlation', 0):.3f})")
        
        # Top Recommendations
        print(f"\n🎯 QUANTITATIVE RECOMMENDATIONS:")
        print("-" * 50)
        
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"\n{i}. [{rec['priority']}] {rec['type']}")
                print(f"   💡 {rec['recommendation']}")
                print(f"   📊 {rec['mathematical_basis']}")
                print(f"   🎯 {rec['action']}")
        else:
            print("   No specific recommendations at this time.")
        
        print(f"\n⏰ Analysis completed: {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

    def _save_analysis(self, analysis: Dict, recommendations: List[Dict]):
        """Save analysis results"""
        timestamp = self.analysis_timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"app/data/cache/quant_analysis_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            results = {
                'timestamp': self.analysis_timestamp.isoformat(),
                'analysis': analysis,
                'recommendations': recommendations
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Analysis saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️  Could not save analysis: {e}")

async def main():
    """Main function"""
    advisor = QuantitativeAdvisor()
    await advisor.analyze_portfolio()

if __name__ == "__main__":
    asyncio.run(main()) 