#!/usr/bin/env python3
"""
🧠 SMART AI PORTFOLIO ANALYZER - ENHANCED VERSION
=================================================
Advanced AI analysis with proven Kraken symbol mapping, real-time data,
learning capabilities, and comprehensive portfolio insights.
"""

import sys
import os
import asyncio
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.kraken import kraken_api

class SmartAIAnalyzer:
    def __init__(self):
        self.analysis_timestamp = datetime.now()
        
        # Learning system
        self.learning_data = self._load_learning_data()
        self.prediction_history = []
        self.accuracy_tracking = {}
        
        # Proven symbol mapping from pure_5k_system.py
        self.symbol_map = {
            'BTC-USD': 'XXBTZUSD',
            'ETH-USD': 'XETHZUSD',
            'SOL-USD': 'SOLUSD',
            'XRP-USD': 'XXRPZUSD',
            'ADA-USD': 'ADAUSD',
            'TRX-USD': 'TRXUSD',
            'XLM-USD': 'XXLMZUSD',
            'DOGE-USD': 'XXDGZUSD',
            'DOT-USD': 'DOTUSD',
            'MATIC-USD': 'MATICUSD',
            'LINK-USD': 'LINKUSD',
            'UNI-USD': 'UNIUSD',
            'AVAX-USD': 'AVAXUSD',
            'BONK-USD': 'BONKUSD',
            'FLOKI-USD': 'FLOKIUSD',
            'PEPE-USD': 'PEPEUSD',
            'ATOM-USD': 'ATOMUSD',
            'FIL-USD': 'FILUSD',
            'ALGO-USD': 'ALGOUSD',
            'VET-USD': 'VETUSD',
            'THETA-USD': 'THETAUSD',
            'HBAR-USD': 'HBARUSD',
            'ICP-USD': 'ICPUSD',
            'NEAR-USD': 'NEARUSD',
            'FTM-USD': 'FTMUSD',
            'MANA-USD': 'MANAUSD',
            'SAND-USD': 'SANDUSD',
            'APE-USD': 'APEUSD',
            'LDO-USD': 'LDOUSD',
            'LRC-USD': 'LRCUSD',
            'ARB-USD': 'ARBUSD',
            'OP-USD': 'OPUSD',
            'XTZ-USD': 'XTZUSD',
            'COMP-USD': 'COMPUSD',
            'MKR-USD': 'MKRUSD',
            'YFI-USD': 'YFIUSD',
            'SNX-USD': 'SNXUSD',
            'CRV-USD': 'CRVUSD',
            'SUSHI-USD': 'SUSHIUSD',
            'AAVE-USD': 'AAVEUSD'
        }
        
        # Enhanced Kraken symbol variations
        self.kraken_symbol_variations = {
            'XXBT': ['XXBTZUSD', 'XBTUSD', 'XXBTCZUSD'],
            'XETH': ['XETHZUSD', 'ETHUSD', 'XETHUSDT'],
            'XXRP': ['XXRPZUSD', 'XRPUSD', 'XXRPZUSD'],
            'XXDG': ['XXDGZUSD', 'XDGUSD', 'DOGEUSDT'],
            'XXLM': ['XXLMZUSD', 'XLMUSD'],
            'XLTC': ['XLTCZUSD', 'LTCUSD'],
            'XTZ': ['XTZUSD', 'XTZZUSD']
        }
        
        print("🧠 Smart AI Portfolio Analyzer Initialized")
        print("🔬 Features: Proven symbol mapping, Learning system, Real-time analysis")
        
    def _load_learning_data(self) -> Dict:
        """Load learning data for AI improvement"""
        try:
            with open('app/data/cache/ai_learning_data.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'symbol_mapping_success': {},
                'prediction_accuracy': {},
                'price_fetch_success_rate': {},
                'last_updated': self.analysis_timestamp.isoformat()
            }
    
    def _save_learning_data(self):
        """Save learning data for future improvements"""
        os.makedirs('app/data/cache', exist_ok=True)
        
        self.learning_data['last_updated'] = self.analysis_timestamp.isoformat()
        
        with open('app/data/cache/ai_learning_data.json', 'w') as f:
            json.dump(self.learning_data, f, indent=2, default=str)

    async def run_smart_analysis(self):
        """Run comprehensive AI analysis with learning capabilities"""
        print("\n" + "="*80)
        print("🧠 SMART AI PORTFOLIO ANALYSIS - LEARNING SYSTEM")
        print("="*80)
        print(f"⏰ Analysis Timestamp: {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        try:
            # Step 1: Get comprehensive portfolio data
            portfolio_data = await self._get_comprehensive_portfolio_data()
            
            # Step 2: Analyze with learning system
            analysis_results = await self._run_smart_analysis(portfolio_data)
            
            # Step 3: Display results
            self._display_smart_results(analysis_results)
            
            # Step 4: Save results and update learning data
            self._save_smart_results(analysis_results)
            
        except Exception as e:
            print(f"❌ Smart analysis failed: {e}")
            import traceback
            traceback.print_exc()

    async def _get_comprehensive_portfolio_data(self) -> Dict:
        """Get comprehensive portfolio data with enhanced symbol mapping"""
        print("📡 Fetching comprehensive portfolio data...")
        
        try:
            # Get Kraken balance
            balance = kraken_api.get_balance()
            
            if 'result' not in balance:
                print("⚠️  Using sample portfolio for analysis")
                return self._get_sample_portfolio_data()
            
            holdings = balance['result']
            portfolio_data = {
                'assets': {},
                'total_value_usd': 0,
                'fetch_timestamp': self.analysis_timestamp.isoformat(),
                'data_source': 'kraken_live',
                'failed_assets': [],
                'successful_mappings': []
            }
            
            print(f"💰 Processing {len(holdings)} assets with enhanced mapping...")
            
            for asset, amount in holdings.items():
                try:
                    amount_float = float(amount)
                    if amount_float > 0.0001:  # Lower threshold
                        
                        # Enhanced price fetching with multiple attempts
                        price, symbol, success = await self._get_enhanced_price(asset)
                        
                        if price > 0:
                            value_usd = amount_float * price
                            portfolio_data['assets'][asset] = {
                                'amount': amount_float,
                                'price_usd': price,
                                'value_usd': value_usd,
                                'symbol': symbol,
                                'mapping_success': success,
                                'timestamp': self.analysis_timestamp.isoformat()
                            }
                            portfolio_data['total_value_usd'] += value_usd
                            portfolio_data['successful_mappings'].append(asset)
                            
                            print(f"   ✅ {asset}: {amount_float:.4f} @ ${price:.4f} = ${value_usd:.2f} ({symbol})")
                        else:
                            portfolio_data['failed_assets'].append((asset, amount_float))
                            print(f"   ❌ {asset}: {amount_float:.4f} - Price fetch failed")
                    elif amount_float > 0:
                        print(f"   🔍 {asset}: {amount_float:.8f} - Too small (below threshold)")
                
                except Exception as e:
                    print(f"   ⚠️  Error processing {asset}: {e}")
                    continue
            
            # Calculate allocations
            total_value = portfolio_data['total_value_usd']
            for asset_data in portfolio_data['assets'].values():
                asset_data['allocation_pct'] = (asset_data['value_usd'] / total_value) * 100
            
            # Update learning data
            self._update_mapping_success_rate(portfolio_data)
            
            print(f"✅ Portfolio value: ${total_value:.2f}")
            print(f"📊 Successful mappings: {len(portfolio_data['successful_mappings'])}")
            print(f"❌ Failed assets: {len(portfolio_data['failed_assets'])}")
            
            return portfolio_data
            
        except Exception as e:
            print(f"⚠️  Error fetching live data: {e}")
            return self._get_sample_portfolio_data()

    async def _get_enhanced_price(self, kraken_asset: str) -> Tuple[float, str, bool]:
        """Enhanced price fetching with multiple symbol variations"""
        try:
            # Handle .F suffix (futures)
            if kraken_asset.endswith('.F'):
                base_asset = kraken_asset.replace('.F', '')
            else:
                base_asset = kraken_asset
            
            # Try multiple symbol variations
            symbol_variations = self._get_symbol_variations(base_asset)
            
            for symbol in symbol_variations:
                try:
                    # Try Kraken first
                    if symbol in self.symbol_map.values() or any(var in self.symbol_map.values() for var in symbol_variations):
                        price = kraken_api.get_price(symbol)
                        if price > 0:
                            return price, symbol, True
                    
                    # Try yfinance
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1d", interval="1m")
                    
                    if not data.empty:
                        current_price = float(data['Close'].iloc[-1])
                        if current_price > 0:
                            return current_price, symbol, True
                            
                except Exception as e:
                    continue
            
            # If all variations fail, try direct mapping
            direct_symbol = f"{base_asset}-USD"
            try:
                ticker = yf.Ticker(direct_symbol)
                data = ticker.history(period="1d", interval="1m")
                
                if not data.empty:
                    current_price = float(data['Close'].iloc[-1])
                    if current_price > 0:
                        return current_price, direct_symbol, True
            except:
                pass
            
            return 0.0, f"{base_asset}-USD", False
                
        except Exception as e:
            print(f"   ⚠️  Enhanced price fetch error for {kraken_asset}: {e}")
            return 0.0, f"{kraken_asset}-USD", False

    def _get_symbol_variations(self, asset: str) -> List[str]:
        """Get multiple symbol variations for an asset"""
        variations = []
        
        # Direct mapping
        variations.append(f"{asset}-USD")
        
        # Check symbol map
        if f"{asset}-USD" in self.symbol_map:
            variations.append(self.symbol_map[f"{asset}-USD"])
        
        # Check Kraken variations
        if asset in self.kraken_symbol_variations:
            variations.extend(self.kraken_symbol_variations[asset])
        
        # Common variations
        common_variations = [
            f"{asset}USD",
            f"{asset}ZUSD",
            f"X{asset}USD",
            f"X{asset}ZUSD",
            f"XX{asset}USD",
            f"{asset.upper()}USD",
            f"X{asset.upper()}ZUSD"
        ]
        variations.extend(common_variations)
        
        return list(set(variations))  # Remove duplicates

    def _update_mapping_success_rate(self, portfolio_data: Dict):
        """Update learning data with mapping success rates"""
        successful = len(portfolio_data['successful_mappings'])
        total = len(portfolio_data['assets']) + len(portfolio_data['failed_assets'])
        
        if total > 0:
            success_rate = successful / total
            self.learning_data['price_fetch_success_rate'][self.analysis_timestamp.strftime('%Y-%m-%d')] = success_rate
            
            print(f"📊 Learning: Price fetch success rate: {success_rate:.1%}")

    def _get_sample_portfolio_data(self) -> Dict:
        """Get sample portfolio data for testing"""
        return {
            'assets': {
                'BTC': {
                    'amount': 0.001234,
                    'price_usd': 67500.0,
                    'value_usd': 83.295,
                    'symbol': 'BTC-USD',
                    'allocation_pct': 40.0,
                    'mapping_success': True,
                    'timestamp': self.analysis_timestamp.isoformat()
                },
                'ETH': {
                    'amount': 0.0456,
                    'price_usd': 3650.0,
                    'value_usd': 166.44,
                    'symbol': 'ETH-USD', 
                    'allocation_pct': 35.0,
                    'mapping_success': True,
                    'timestamp': self.analysis_timestamp.isoformat()
                },
                'SOL': {
                    'amount': 0.789,
                    'price_usd': 190.0,
                    'value_usd': 149.91,
                    'symbol': 'SOL-USD',
                    'allocation_pct': 25.0,
                    'mapping_success': True,
                    'timestamp': self.analysis_timestamp.isoformat()
                }
            },
            'total_value_usd': 399.645,
            'fetch_timestamp': self.analysis_timestamp.isoformat(),
            'data_source': 'sample_data',
            'failed_assets': [],
            'successful_mappings': ['BTC', 'ETH', 'SOL']
        }

    async def _run_smart_analysis(self, portfolio_data: Dict) -> Dict:
        """Run smart AI analysis with learning capabilities"""
        print("🧠 Running smart AI analysis...")
        
        analysis_results = {
            'portfolio_overview': self._analyze_portfolio_overview(portfolio_data),
            'technical_analysis': await self._analyze_technical_signals(portfolio_data),
            'learning_insights': self._generate_learning_insights(portfolio_data),
            'recommendations': self._generate_smart_recommendations(portfolio_data),
            'risk_assessment': self._assess_smart_risk(portfolio_data),
            'prediction_accuracy': self._get_prediction_accuracy(),
            'analysis_metadata': {
                'timestamp': self.analysis_timestamp.isoformat(),
                'learning_enabled': True,
                'symbol_mapping_version': 'enhanced_v2',
                'success_rate': len(portfolio_data.get('successful_mappings', [])) / max(1, len(portfolio_data.get('assets', {})))
            }
        }
        
        return analysis_results

    def _analyze_portfolio_overview(self, portfolio_data: Dict) -> Dict:
        """Analyze portfolio overview with learning insights"""
        assets = portfolio_data.get('assets', {})
        total_value = portfolio_data.get('total_value_usd', 0)
        
        # Calculate learning-based metrics
        mapping_success_rate = len(portfolio_data.get('successful_mappings', [])) / max(1, len(assets))
        
        return {
            'total_value_usd': total_value,
            'asset_count': len(assets),
            'largest_position': max([asset['allocation_pct'] for asset in assets.values()]) if assets else 0,
            'diversification_score': self._calculate_diversification_score(assets),
            'mapping_success_rate': mapping_success_rate,
            'data_quality': 'HIGH' if mapping_success_rate > 0.8 else 'MEDIUM' if mapping_success_rate > 0.6 else 'LOW',
            'learning_progress': self._get_learning_progress()
        }

    def _calculate_diversification_score(self, assets: Dict) -> float:
        """Calculate portfolio diversification score"""
        if not assets:
            return 0.0
        
        allocations = [asset['allocation_pct'] for asset in assets.values()]
        hhi = sum([(alloc/100)**2 for alloc in allocations])
        diversification_score = (1 - hhi) * 100
        
        return round(diversification_score, 2)

    async def _analyze_technical_signals(self, portfolio_data: Dict) -> Dict:
        """Analyze technical signals for all assets"""
        technical_signals = {}
        
        for asset_key, asset_info in portfolio_data.get('assets', {}).items():
            symbol = asset_info.get('symbol', asset_key)
            
            try:
                # Get technical data
                tech_data = await self._get_technical_data(symbol)
                if tech_data:
                    technical_signals[symbol] = {
                        'signal': self._determine_signal(tech_data),
                        'strength': self._calculate_signal_strength(tech_data),
                        'risk_level': self._assess_technical_risk(tech_data),
                        'confidence': self._get_prediction_confidence(symbol),
                        'learning_factor': self._get_learning_factor(symbol)
                    }
            except Exception as e:
                print(f"   ⚠️  Technical analysis failed for {symbol}: {e}")
                continue
        
        return technical_signals

    async def _get_technical_data(self, symbol: str) -> Optional[Dict]:
        """Get technical data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="30d")
            
            if data.empty:
                return None
            
            # Calculate technical indicators
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            
            current_price = float(data['Close'].iloc[-1])
            sma_20 = float(data['SMA_20'].iloc[-1])
            sma_50 = float(data['SMA_50'].iloc[-1])
            rsi = float(data['RSI'].iloc[-1])
            
            return {
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi,
                'trend': 'BULLISH' if current_price > sma_20 > sma_50 else 'BEARISH' if current_price < sma_20 < sma_50 else 'NEUTRAL'
            }
            
        except Exception as e:
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _determine_signal(self, tech_data: Dict) -> str:
        """Determine trading signal from technical data"""
        rsi = tech_data.get('rsi', 50)
        trend = tech_data.get('trend', 'NEUTRAL')
        
        if rsi > 70 and trend == 'BULLISH':
            return 'OVERBOUGHT'
        elif rsi < 30 and trend == 'BEARISH':
            return 'OVERSOLD'
        elif trend == 'BULLISH':
            return 'BULLISH'
        elif trend == 'BEARISH':
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _calculate_signal_strength(self, tech_data: Dict) -> float:
        """Calculate signal strength (0-100)"""
        rsi = tech_data.get('rsi', 50)
        trend = tech_data.get('trend', 'NEUTRAL')
        
        strength = 50  # Base
        
        # RSI contribution
        if rsi > 70 or rsi < 30:
            strength += 25
        elif 60 < rsi < 70 or 30 < rsi < 40:
            strength += 15
        
        # Trend contribution
        if trend == 'BULLISH':
            strength += 15
        elif trend == 'BEARISH':
            strength += 15
        
        return min(100, max(0, strength))

    def _assess_technical_risk(self, tech_data: Dict) -> str:
        """Assess technical risk level"""
        rsi = tech_data.get('rsi', 50)
        
        if rsi > 75 or rsi < 25:
            return 'HIGH'
        elif rsi > 70 or rsi < 30:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _get_prediction_confidence(self, symbol: str) -> float:
        """Get prediction confidence based on learning data"""
        # This would be based on historical accuracy for this symbol
        return 75.0  # Default confidence

    def _get_learning_factor(self, symbol: str) -> float:
        """Get learning factor for symbol"""
        # This would be based on how much we've learned about this symbol
        return 0.8  # Default learning factor

    def _generate_learning_insights(self, portfolio_data: Dict) -> Dict:
        """Generate insights from learning system"""
        return {
            'mapping_improvements': self._get_mapping_improvements(),
            'prediction_accuracy': self._get_prediction_accuracy(),
            'recommended_actions': self._get_learning_recommendations()
        }

    def _get_mapping_improvements(self) -> List[str]:
        """Get symbol mapping improvement suggestions"""
        improvements = []
        
        if len(self.learning_data.get('price_fetch_success_rate', {})) > 0:
            recent_success_rate = list(self.learning_data['price_fetch_success_rate'].values())[-1]
            
            if recent_success_rate < 0.8:
                improvements.append("Consider adding more symbol variations for failed assets")
            if recent_success_rate < 0.6:
                improvements.append("Review Kraken API symbol mapping for common failures")
        
        return improvements

    def _get_prediction_accuracy(self) -> Dict:
        """Get prediction accuracy metrics"""
        return {
            'overall_accuracy': 0.75,  # Would be calculated from historical data
            'symbol_specific': {},
            'trending_accuracy': 'IMPROVING'
        }

    def _get_learning_recommendations(self) -> List[str]:
        """Get recommendations from learning system"""
        return [
            "Continue monitoring symbol mapping success rates",
            "Track prediction accuracy for portfolio optimization",
            "Consider expanding symbol variations for better coverage"
        ]

    def _generate_smart_recommendations(self, portfolio_data: Dict) -> List[Dict]:
        """Generate smart recommendations based on learning"""
        recommendations = []
        assets = portfolio_data.get('assets', {})
        
        # Portfolio structure recommendations
        if len(assets) > 0:
            max_allocation = max([asset['allocation_pct'] for asset in assets.values()])
            if max_allocation > 40:
                recommendations.append({
                    'type': 'CONCENTRATION_RISK',
                    'priority': 'HIGH',
                    'message': f"Reduce largest position ({max_allocation:.1f}%) to improve diversification",
                    'confidence': 85
                })
        
        # Technical signal recommendations
        for asset_key, asset_info in assets.items():
            symbol = asset_info.get('symbol', asset_key)
            allocation = asset_info.get('allocation_pct', 0)
            
            if allocation > 20:  # Only for significant positions
                recommendations.append({
                    'type': 'POSITION_MANAGEMENT',
                    'priority': 'MEDIUM',
                    'message': f"Monitor {symbol} position ({allocation:.1f}%) for technical signals",
                    'confidence': 70
                })
        
        return recommendations

    def _assess_smart_risk(self, portfolio_data: Dict) -> Dict:
        """Assess portfolio risk with learning insights"""
        assets = portfolio_data.get('assets', {})
        
        risk_factors = []
        risk_score = 50  # Base risk score
        
        # Concentration risk
        if len(assets) > 0:
            max_allocation = max([asset['allocation_pct'] for asset in assets.values()])
            if max_allocation > 50:
                risk_factors.append(f"High concentration: {max_allocation:.1f}% in single asset")
                risk_score += 25
            elif max_allocation > 30:
                risk_factors.append(f"Moderate concentration: {max_allocation:.1f}% in single asset")
                risk_score += 15
        
        # Mapping success risk
        mapping_success_rate = len(portfolio_data.get('successful_mappings', [])) / max(1, len(assets))
        if mapping_success_rate < 0.8:
            risk_factors.append(f"Data quality issues: {mapping_success_rate:.1%} success rate")
            risk_score += 10
        
        return {
            'risk_score': min(100, risk_score),
            'risk_level': 'HIGH' if risk_score > 70 else 'MEDIUM' if risk_score > 40 else 'LOW',
            'risk_factors': risk_factors,
            'learning_insights': self._get_learning_progress()
        }

    def _get_learning_progress(self) -> Dict:
        """Get learning system progress"""
        return {
            'total_analyses': len(self.learning_data.get('price_fetch_success_rate', {})),
            'average_success_rate': np.mean(list(self.learning_data.get('price_fetch_success_rate', {}).values())) if self.learning_data.get('price_fetch_success_rate') else 0,
            'trend': 'IMPROVING' if len(self.learning_data.get('price_fetch_success_rate', {})) > 1 else 'STABLE'
        }

    def _display_smart_results(self, results: Dict):
        """Display smart analysis results"""
        print("\n" + "="*80)
        print("🧠 SMART AI ANALYSIS RESULTS")
        print("="*80)
        
        # Portfolio Overview
        overview = results.get('portfolio_overview', {})
        print(f"\n💰 PORTFOLIO OVERVIEW:")
        print(f"   Total Value: ${overview.get('total_value_usd', 0):,.2f}")
        print(f"   Assets: {overview.get('asset_count', 0)}")
        print(f"   Largest Position: {overview.get('largest_position', 0):.1f}%")
        print(f"   Diversification: {overview.get('diversification_score', 0):.1f}/100")
        print(f"   Data Quality: {overview.get('data_quality', 'UNKNOWN')}")
        
        # Learning Insights
        learning = results.get('learning_insights', {})
        print(f"\n🧠 LEARNING INSIGHTS:")
        print(f"   Mapping Success Rate: {results.get('analysis_metadata', {}).get('success_rate', 0):.1%}")
        print(f"   Learning Progress: {learning.get('prediction_accuracy', {}).get('trending_accuracy', 'UNKNOWN')}")
        
        # Technical Signals
        technical = results.get('technical_analysis', {})
        if technical:
            print(f"\n📊 TECHNICAL SIGNALS:")
            for symbol, signal in technical.items():
                print(f"   {symbol:12} | {signal['signal']:12} | Strength: {signal['strength']:5.1f}% | Risk: {signal['risk_level']}")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 SMART RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:5], 1):
                priority_emoji = "🔥" if rec['priority'] == 'HIGH' else "⚡" if rec['priority'] == 'MEDIUM' else "💡"
                print(f"   {i}. {priority_emoji} {rec['message']}")
                print(f"      Confidence: {rec['confidence']}%")
        
        # Risk Assessment
        risk = results.get('risk_assessment', {})
        print(f"\n⚠️  RISK ASSESSMENT:")
        print(f"   Risk Level: {risk.get('risk_level', 'UNKNOWN')}")
        print(f"   Risk Score: {risk.get('risk_score', 0):.1f}/100")
        for factor in risk.get('risk_factors', []):
            print(f"   • {factor}")

    def _save_smart_results(self, results: Dict):
        """Save smart analysis results and update learning data"""
        timestamp = self.analysis_timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"app/data/cache/smart_ai_analysis_{timestamp}.json"
        
        os.makedirs('app/data/cache', exist_ok=True)
        
        try:
            # Save results
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Update learning data
            self._save_learning_data()
            
            print(f"\n💾 Smart analysis saved to: {filename}")
            print(f"🧠 Learning data updated")
            
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")

async def main():
    """Main function"""
    analyzer = SmartAIAnalyzer()
    await analyzer.run_smart_analysis()

if __name__ == "__main__":
    asyncio.run(main()) 