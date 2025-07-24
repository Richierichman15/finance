#!/usr/bin/env python3
"""
🤖 AI PORTFOLIO ANALYSIS RUNNER
==============================
Run AI-powered analysis on your Kraken portfolio data
"""

import sys
import os
import asyncio
import json
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ai_market_analyzer import AIMarketAnalyzer
from services.kraken import kraken_api

class AIPortfolioRunner:
    def __init__(self):
        self.analyzer = AIMarketAnalyzer()
        
    async def run_full_analysis(self):
        """Run comprehensive AI analysis on current portfolio"""
        print("🤖 STARTING AI MARKET ANALYSIS...")
        print("=" * 60)
        
        try:
            # Get current portfolio data
            portfolio_data = await self._get_portfolio_data()
            
            # Run AI analysis
            analysis_results = await self.analyzer.analyze_portfolio_with_ai(portfolio_data)
            
            # Display results
            self._display_analysis_results(analysis_results)
            
            # Save results
            self._save_analysis_results(analysis_results)
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    async def _get_portfolio_data(self):
        """Get current portfolio data from Kraken"""
        try:
            # Get balance
            balance = kraken_api.get_balance()
            
            if 'result' not in balance:
                print("⚠️  Could not get balance, using sample data")
                return self._get_sample_portfolio_data()
            
            # Extract meaningful holdings
            holdings = balance['result']
            portfolio_allocations = {}
            total_value = 0
            
            for asset, amount in holdings.items():
                try:
                    amount_float = float(amount)
                    if amount_float > 0.01:  # Only include meaningful amounts
                        # Try to get USD value
                        if asset == 'ZUSD':
                            value = amount_float
                        else:
                            # This is simplified - in practice you'd get current prices
                            value = amount_float * 100  # Placeholder
                        
                        total_value += value
                        portfolio_allocations[asset] = value
                except:
                    continue
            
            # Convert to percentages
            if total_value > 0:
                for asset in portfolio_allocations:
                    portfolio_allocations[asset] = (portfolio_allocations[asset] / total_value) * 100
            
            return {
                'allocations': portfolio_allocations,
                'total_value': total_value,
                'source': 'kraken_live'
            }
            
        except Exception as e:
            print(f"⚠️  Kraken data error: {e}, using sample data")
            return self._get_sample_portfolio_data()
    
    def _get_sample_portfolio_data(self):
        """Get sample portfolio data for testing"""
        return {
            'allocations': {
                'BTC-USD': 35.0,
                'ETH-USD': 25.0,
                'ADA-USD': 10.0,
                'SOL-USD': 15.0,
                'QQQ': 10.0,
                'NVDA': 5.0
            },
            'total_value': 5000.0,
            'source': 'sample_data'
        }
    
    def _display_analysis_results(self, results):
        """Display comprehensive analysis results"""
        print("\n🧠 AI ANALYSIS COMPLETE!")
        print("=" * 60)
        
        # Overall conclusions
        conclusions = results['ai_conclusions']
        print(f"\n📊 OVERALL MARKET ASSESSMENT:")
        print(f"   Sentiment: {conclusions['overall_sentiment']} ({conclusions['confidence']:.1f}% confidence)")
        print(f"   Summary: {conclusions['summary']}")
        
        # Risk assessment
        risk = results['risk_assessment']
        print(f"\n⚠️  PORTFOLIO RISK ANALYSIS:")
        print(f"   Risk Level: {risk['risk_level']} (Score: {risk['risk_score']:.1f}/100)")
        if risk['high_risk_insights'] > 0:
            print(f"   ⚠️  {risk['high_risk_insights']} high-risk factors detected")
        
        # Top insights
        insights = results['ai_insights']
        if insights:
            print(f"\n🔍 KEY AI INSIGHTS ({len(insights)} total):")
            print("-" * 50)
            
            # Sort by impact score
            top_insights = sorted(insights, key=lambda x: x.impact_score, reverse=True)[:5]
            
            for i, insight in enumerate(top_insights, 1):
                print(f"\n{i}. 📈 {insight.category}")
                print(f"   Confidence: {insight.confidence:.1f}% | Impact: {insight.impact_score:.1f}/10 | Risk: {insight.risk_level}")
                print(f"   🔹 {insight.insight}")
                print(f"   📊 Math: {insight.mathematical_basis}")
                print(f"   💡 Action: {insight.action_recommendation}")
        
        # Action priorities
        priorities = results['action_priorities']
        if priorities:
            print(f"\n🎯 TOP ACTION PRIORITIES:")
            print("-" * 30)
            for i, action in enumerate(priorities[:3], 1):
                print(f"{i}. {action['action']}")
                print(f"   Priority Score: {action['priority_score']:.1f} | Risk: {action['risk_level']}")
        
        # Market data summary
        market_summary = results['market_data_summary']
        print(f"\n📊 ANALYSIS SCOPE:")
        print(f"   Assets Analyzed: {market_summary['total_assets_analyzed']}")
        print(f"   Crypto: {market_summary['crypto_assets']} | Stocks: {market_summary['stock_assets']}")
        print(f"   Data Quality: {market_summary['data_quality']}")
        
        print(f"\n⏰ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _save_analysis_results(self, results):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"app/data/cache/ai_analysis_{timestamp}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            # Convert MarketInsight objects to dictionaries for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if key == 'ai_insights':
                    serializable_results[key] = [
                        {
                            'category': insight.category,
                            'confidence': insight.confidence,
                            'insight': insight.insight,
                            'mathematical_basis': insight.mathematical_basis,
                            'action_recommendation': insight.action_recommendation,
                            'risk_level': insight.risk_level,
                            'impact_score': insight.impact_score
                        }
                        for insight in value
                    ]
                else:
                    serializable_results[key] = value
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            print(f"\n💾 Analysis saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")

async def main():
    """Main function"""
    runner = AIPortfolioRunner()
    await runner.run_full_analysis()

if __name__ == "__main__":
    asyncio.run(main()) 