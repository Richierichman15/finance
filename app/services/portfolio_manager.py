import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import asyncio

class PortfolioManager:
    def __init__(self):
        self.target_allocation = {
            "US_Equities": 20.0,      # VTI, SCHB
            "Energy": 20.0,           # XLE, AMLP  
            "Infrastructure": 15.0,   # IGF, BIPC
            "Technology": 15.0,       # QQQ, VGT
            "Real_Estate": 5.0,       # VNQ
            "Crypto": 25.0            # Speculative
        }
        
        self.etf_mapping = {
            "US_Equities": ["VTI", "SCHB"],
            "Energy": ["XLE", "AMLP"],
            "Infrastructure": ["IGF", "BIPC"], 
            "Technology": ["QQQ", "VGT"],
            "Real_Estate": ["VNQ"],
            "Crypto": ["BTC-USD", "ETH-USD"]  # For reference
        }
        
        self.risk_multipliers = {
            "conservative": 0.7,
            "moderate": 1.0,
            "aggressive": 1.3,
            "yolo": 1.6
        }

    async def analyze_portfolio(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive portfolio analysis"""
        current_value = portfolio_data.get('current_value', 0)
        current_allocations = portfolio_data.get('allocations', {})
        cash_available = portfolio_data.get('cash_available', 0)
        
        # Get market data for analysis
        market_data = await self._fetch_market_data()
        
        # Calculate current vs target allocation
        allocation_analysis = self._analyze_allocations(current_allocations)
        
        # Performance analysis
        performance_metrics = await self._calculate_performance_metrics(
            current_allocations, market_data
        )
        
        # Risk assessment
        risk_assessment = self._assess_portfolio_risk(current_allocations, market_data)
        
        # Diversification score
        diversification_score = self._calculate_diversification_score(current_allocations)
        
        return {
            "portfolio_value": current_value,
            "cash_available": cash_available,
            "current_allocation": current_allocations,
            "target_allocation": self.target_allocation,
            "allocation_analysis": allocation_analysis,
            "performance_metrics": performance_metrics,
            "risk_assessment": risk_assessment,
            "diversification_score": diversification_score,
            "market_data": market_data,
            "rebalancing_needed": self._needs_rebalancing(current_allocations),
            "profit_opportunities": self._identify_profit_opportunities(
                current_allocations, market_data
            )
        }

    def _analyze_allocations(self, current_allocations: Dict[str, float]) -> Dict[str, Any]:
        """Analyze current vs target allocations"""
        deviations = {}
        total_deviation = 0
        
        for sector, target_pct in self.target_allocation.items():
            current_pct = current_allocations.get(sector, 0)
            deviation = current_pct - target_pct
            deviations[sector] = {
                "current": current_pct,
                "target": target_pct,
                "deviation": deviation,
                "status": self._get_allocation_status(deviation)
            }
            total_deviation += abs(deviation)
        
        return {
            "sector_deviations": deviations,
            "total_deviation": total_deviation,
            "allocation_quality": self._grade_allocation_quality(total_deviation)
        }

    def _get_allocation_status(self, deviation: float) -> str:
        """Get status of allocation deviation"""
        if abs(deviation) < 2:
            return "ON_TARGET"
        elif deviation > 5:
            return "OVERWEIGHT"
        elif deviation < -5:
            return "UNDERWEIGHT"
        else:
            return "SLIGHT_DEVIATION"

    def _grade_allocation_quality(self, total_deviation: float) -> str:
        """Grade the overall allocation quality"""
        if total_deviation < 5:
            return "EXCELLENT"
        elif total_deviation < 10:
            return "GOOD"
        elif total_deviation < 20:
            return "NEEDS_IMPROVEMENT"
        else:
            return "POOR"

    async def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch current market data for all ETFs"""
        market_data = {}
        
        all_symbols = []
        for etfs in self.etf_mapping.values():
            all_symbols.extend(etfs)
        
        try:
            # Remove crypto symbols for yfinance
            stock_symbols = [s for s in all_symbols if not s.endswith('-USD')]
            
            for symbol in stock_symbols:
                ticker = yf.Ticker(symbol)
                
                # Get basic info
                hist = ticker.history(period="30d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    monthly_return = ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                    volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized
                    
                    market_data[symbol] = {
                        "current_price": float(current_price),
                        "monthly_return": float(monthly_return),
                        "volatility": float(volatility),
                        "volume": float(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0
                    }
                    
        except Exception as e:
            print(f"Error fetching market data: {e}")
            
        return market_data

    async def _calculate_performance_metrics(self, allocations: Dict[str, float], 
                                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        try:
            # Weighted returns based on allocations
            total_return = 0
            total_volatility = 0
            
            for sector, allocation_pct in allocations.items():
                if sector in self.etf_mapping:
                    # Use first ETF as representative
                    etf_symbol = self.etf_mapping[sector][0]
                    if etf_symbol in market_data:
                        etf_data = market_data[etf_symbol]
                        weight = allocation_pct / 100
                        total_return += etf_data.get('monthly_return', 0) * weight
                        total_volatility += etf_data.get('volatility', 0) * weight
            
            # Sharpe ratio approximation (using 4% risk-free rate)
            sharpe_ratio = (total_return * 12 - 4) / (total_volatility if total_volatility > 0 else 1)
            
            return {
                "monthly_return": total_return,
                "annualized_return": total_return * 12,
                "volatility": total_volatility,
                "sharpe_ratio": sharpe_ratio,
                "risk_adjusted_return": sharpe_ratio * 10,  # Scaled for presentation
                "performance_grade": self._grade_performance(total_return * 12, total_volatility)
            }
            
        except Exception as e:
            print(f"Error calculating performance: {e}")
            return {
                "monthly_return": 0,
                "annualized_return": 0,
                "volatility": 15,
                "sharpe_ratio": 0.5,
                "risk_adjusted_return": 5,
                "performance_grade": "UNKNOWN"
            }

    def _grade_performance(self, annual_return: float, volatility: float) -> str:
        """Grade portfolio performance"""
        if annual_return > 20 and volatility < 20:
            return "ELITE"
        elif annual_return > 15:
            return "EXCELLENT"
        elif annual_return > 10:
            return "GOOD"
        elif annual_return > 5:
            return "AVERAGE"
        else:
            return "POOR"

    def _assess_portfolio_risk(self, allocations: Dict[str, float], 
                             market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess portfolio risk characteristics"""
        risk_score = 0
        risk_factors = []
        
        # Concentration risk
        max_allocation = max(allocations.values()) if allocations else 0
        if max_allocation > 50:
            risk_factors.append(f"High concentration risk: {max_allocation:.1f}% in single sector")
            risk_score += 30
        
        # Sector risk assessment
        high_risk_sectors = ["Technology", "Crypto", "Energy"]
        high_risk_allocation = sum(allocations.get(sector, 0) for sector in high_risk_sectors)
        
        if high_risk_allocation > 60:
            risk_factors.append(f"High growth/volatility exposure: {high_risk_allocation:.1f}%")
            risk_score += 25
        
        # Diversification risk
        if len([a for a in allocations.values() if a > 5]) < 4:
            risk_factors.append("Insufficient diversification across sectors")
            risk_score += 20
        
        return {
            "risk_score": min(risk_score, 100),
            "risk_level": self._categorize_risk(risk_score),
            "risk_factors": risk_factors,
            "aggressive_rating": self._rate_aggressiveness(allocations)
        }

    def _categorize_risk(self, risk_score: float) -> str:
        """Categorize risk level"""
        if risk_score < 20:
            return "LOW"
        elif risk_score < 40:
            return "MODERATE"
        elif risk_score < 70:
            return "HIGH"
        else:
            return "EXTREME"

    def _rate_aggressiveness(self, allocations: Dict[str, float]) -> str:
        """Rate how aggressive the portfolio is"""
        aggressive_sectors = ["Technology", "Crypto", "Energy"]
        aggressive_allocation = sum(allocations.get(sector, 0) for sector in aggressive_sectors)
        
        if aggressive_allocation > 50:
            return "VERY_AGGRESSIVE"
        elif aggressive_allocation > 30:
            return "AGGRESSIVE"
        elif aggressive_allocation > 15:
            return "MODERATE"
        else:
            return "CONSERVATIVE"

    def _calculate_diversification_score(self, allocations: Dict[str, float]) -> float:
        """Calculate diversification score (0-100)"""
        if not allocations:
            return 0
        
        # Use Herfindahl-Hirschman Index approach
        hhi = sum((pct/100)**2 for pct in allocations.values())
        
        # Convert to 0-100 score (lower HHI = better diversification)
        diversification_score = max(0, (1 - hhi) * 100)
        
        return diversification_score

    def _needs_rebalancing(self, current_allocations: Dict[str, float]) -> bool:
        """Determine if portfolio needs rebalancing"""
        total_deviation = 0
        for sector, target_pct in self.target_allocation.items():
            current_pct = current_allocations.get(sector, 0)
            total_deviation += abs(current_pct - target_pct)
        
        return total_deviation > 10  # Rebalance if total deviation > 10%

    def _identify_profit_opportunities(self, allocations: Dict[str, float], 
                                     market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify profit opportunities"""
        opportunities = []
        
        # Underweight high-performing sectors
        for sector, target_pct in self.target_allocation.items():
            current_pct = allocations.get(sector, 0)
            if current_pct < target_pct - 5:  # Significantly underweight
                etf_symbol = self.etf_mapping.get(sector, [""])[0]
                if etf_symbol in market_data:
                    monthly_return = market_data[etf_symbol].get('monthly_return', 0)
                    if monthly_return > 3:  # Strong performance
                        opportunities.append({
                            "type": "UNDERWEIGHT_OPPORTUNITY",
                            "sector": sector,
                            "current_allocation": current_pct,
                            "target_allocation": target_pct,
                            "recommended_increase": target_pct - current_pct,
                            "performance": monthly_return,
                            "priority": "HIGH" if monthly_return > 5 else "MEDIUM"
                        })
        
        return opportunities

    async def generate_rebalance_plan(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed rebalancing plan"""
        current_value = portfolio_data.get('current_value', 0)
        current_allocations = portfolio_data.get('allocations', {})
        cash_available = portfolio_data.get('cash_available', 0)
        
        total_investable = current_value + cash_available
        
        rebalance_actions = []
        cash_needed = 0
        
        for sector, target_pct in self.target_allocation.items():
            current_pct = current_allocations.get(sector, 0)
            current_value_sector = (current_pct / 100) * current_value
            target_value_sector = (target_pct / 100) * total_investable
            
            difference = target_value_sector - current_value_sector
            
            if abs(difference) > total_investable * 0.02:  # Only if >2% of portfolio
                action_type = "BUY" if difference > 0 else "SELL"
                etf_recommendation = self.etf_mapping.get(sector, [""])[0]
                
                rebalance_actions.append({
                    "sector": sector,
                    "action": action_type,
                    "amount": abs(difference),
                    "percentage_change": (difference / total_investable) * 100,
                    "recommended_etf": etf_recommendation,
                    "current_allocation": current_pct,
                    "target_allocation": target_pct,
                    "priority": self._calculate_action_priority(difference, total_investable)
                })
                
                if difference > 0:
                    cash_needed += difference
        
        return {
            "total_portfolio_value": total_investable,
            "cash_required": max(0, cash_needed - cash_available),
            "rebalance_actions": sorted(rebalance_actions, 
                                      key=lambda x: x['priority'], reverse=True),
            "estimated_cost": len(rebalance_actions) * 7,  # $7 per trade estimate
            "execution_timeline": "IMMEDIATE",
            "expected_improvement": self._estimate_improvement(rebalance_actions)
        }

    def _calculate_action_priority(self, difference: float, total_value: float) -> int:
        """Calculate priority score for rebalancing action"""
        percentage_impact = abs(difference) / total_value * 100
        
        if percentage_impact > 10:
            return 10  # Highest priority
        elif percentage_impact > 5:
            return 7
        elif percentage_impact > 2:
            return 5
        else:
            return 3

    def _estimate_improvement(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate improvement from rebalancing"""
        total_changes = sum(abs(action['percentage_change']) for action in actions)
        
        return {
            "diversification_improvement": min(total_changes * 2, 20),
            "risk_reduction": min(total_changes * 1.5, 15),
            "expected_return_boost": f"{min(total_changes * 0.5, 3):.1f}% annually",
            "optimization_score": min(total_changes * 5, 50)
        }

    def get_aggressive_tweaks(self, current_allocations: Dict[str, float]) -> List[Dict[str, Any]]:
        """Get aggressive optimization suggestions"""
        tweaks = []
        
        # Increase growth allocations
        tech_allocation = current_allocations.get("Technology", 0)
        if tech_allocation < 20:
            tweaks.append({
                "tweak": "TECH_BOOST",
                "description": f"Increase Tech from {tech_allocation}% to 20% for explosive growth",
                "impact": "HIGH_GROWTH",
                "risk_increase": "MODERATE"
            })
        
        # Energy play
        energy_allocation = current_allocations.get("Energy", 0)
        if energy_allocation < 25:
            tweaks.append({
                "tweak": "ENERGY_AGGRESSIVE",
                "description": f"Pump Energy to 25% - oil/gas infrastructure boom incoming!",
                "impact": "SECTOR_MOMENTUM",
                "risk_increase": "MODERATE"
            })
        
        return tweaks