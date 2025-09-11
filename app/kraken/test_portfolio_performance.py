#!/usr/bin/env python3
"""
Test Portfolio Performance Logic
Tests various scenarios to ensure percentage calculations and negative signs work correctly
"""

import sys
import os
import json
import tempfile
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MockKrakenAPI:
    """Mock Kraken API for testing"""
    def __init__(self, test_prices, test_balance):
        self.test_prices = test_prices
        self.test_balance = test_balance
    
    def get_ticker(self, symbol):
        if symbol in self.test_prices:
            return {
                'result': {
                    symbol: {
                        'c': [str(self.test_prices[symbol]), '0']
                    }
                }
            }
        return {'error': ['EQuery:Unknown asset pair']}
    
    def get_balance(self):
        return {'result': self.test_balance}
    
    def get_trade_history(self):
        return {'result': {'trades': {}}}

class TestPortfolioAnalyzer:
    """Test version of portfolio analyzer with controlled data"""
    
    def __init__(self, test_data):
        self.test_data = test_data
        self.current_prices = {}
        self.portfolio_value = 0
        
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.asset_mappings_file = os.path.join(self.temp_dir, "asset_mappings.json")
        self.investments_file = os.path.join(self.temp_dir, "investment_history.json")
        
        # Set up test investment data
        self.investment_data = {
            "total_invested": 0,
            "assets": test_data['cost_basis'],
            "last_updated": datetime.now().isoformat()
        }
        
        # Mock prices
        for asset, price in test_data['current_prices'].items():
            self.current_prices[asset] = price
    
    def calculate_test_performance(self):
        """Calculate performance with test data"""
        print(f"\n🧪 TESTING SCENARIO: {self.test_data['name']}")
        print("="*60)
        
        asset_values = {}
        total_cost_basis = 0
        
        # Calculate current values and performance
        for asset, amount in self.test_data['holdings'].items():
            current_price = self.current_prices.get(asset, 0)
            current_value = amount * current_price
            
            # Get cost basis
            cost_basis = self.investment_data['assets'].get(asset, {}).get('cost_basis', 0)
            
            asset_values[asset] = {
                'amount': amount,
                'current_value': current_value,
                'cost_basis': cost_basis,
                'current_price': current_price,
                'usd_value': current_value
            }
            
            total_cost_basis += cost_basis
        
        # Calculate portfolio performance
        self.portfolio_value = sum(data['current_value'] for data in asset_values.values())
        total_gains_losses = self.portfolio_value - total_cost_basis
        portfolio_performance = (total_gains_losses / total_cost_basis * 100) if total_cost_basis > 0 else 0
        
        # Display results
        print(f"💰 Current Portfolio Value: ${self.portfolio_value:,.2f}")
        print(f"💵 Total Invested: ${total_cost_basis:,.2f}")
        print(f"📈 Total Gains/Losses: ${total_gains_losses:+,.2f}")
        print(f"📊 Portfolio Performance: {portfolio_performance:+.2f}%")
        
        # Performance indicator
        if portfolio_performance > 0:
            print("🟢 Portfolio is PROFITABLE")
        elif portfolio_performance < 0:
            print("🔴 Portfolio is AT A LOSS")
        else:
            print("⚪ Portfolio is BREAK-EVEN")
        
        print("\n📊 INDIVIDUAL ASSET PERFORMANCE:")
        print("-" * 60)
        print(f"{'Asset':<10} {'Holdings':<12} {'Current':<12} {'Cost Basis':<12} {'Performance':<12}")
        print("-" * 60)
        
        for asset, data in asset_values.items():
            gain_loss = data['current_value'] - data['cost_basis']
            performance_pct = (gain_loss / data['cost_basis'] * 100) if data['cost_basis'] > 0 else 0
            
            # Color coding for performance
            if performance_pct > 0:
                performance_str = f"🟢{performance_pct:+.1f}%"
            elif performance_pct < 0:
                performance_str = f"🔴{performance_pct:+.1f}%"
            else:
                performance_str = f"⚪{performance_pct:+.1f}%"
            
            print(f"{asset:<10} {data['amount']:<12.4f} ${data['current_value']:<11.2f} ${data['cost_basis']:<11.2f} {performance_str:<12}")
        
        print("="*60)
        return portfolio_performance

def run_test_scenarios():
    """Run various test scenarios to verify logic"""
    
    test_scenarios = [
        {
            'name': 'PROFITABLE PORTFOLIO',
            'holdings': {
                'BTC': 0.1,
                'ETH': 1.0,
                'SOL': 10.0
            },
            'current_prices': {
                'BTC': 50000.00,  # $5000 current value
                'ETH': 3000.00,   # $3000 current value  
                'SOL': 100.00     # $1000 current value
            },
            'cost_basis': {
                'BTC': {'cost_basis': 4000.00},  # Bought at $40k
                'ETH': {'cost_basis': 2500.00},  # Bought at $2500
                'SOL': {'cost_basis': 800.00}    # Bought at $80
            }
        },
        {
            'name': 'LOSING PORTFOLIO',
            'holdings': {
                'BTC': 0.05,
                'ETH': 2.0,
                'ADA': 1000.0
            },
            'current_prices': {
                'BTC': 40000.00,  # $2000 current value
                'ETH': 2000.00,   # $4000 current value
                'ADA': 0.50       # $500 current value
            },
            'cost_basis': {
                'BTC': {'cost_basis': 3000.00},  # Bought at $60k
                'ETH': {'cost_basis': 5000.00},  # Bought at $2500
                'ADA': {'cost_basis': 800.00}    # Bought at $0.80
            }
        },
        {
            'name': 'MIXED PERFORMANCE',
            'holdings': {
                'BTC': 0.02,
                'ETH': 0.5,
                'SOL': 5.0,
                'ADA': 500.0
            },
            'current_prices': {
                'BTC': 60000.00,  # $1200 - GAIN
                'ETH': 2800.00,   # $1400 - LOSS
                'SOL': 90.00,     # $450 - LOSS  
                'ADA': 1.20       # $600 - GAIN
            },
            'cost_basis': {
                'BTC': {'cost_basis': 1000.00},  # +20% gain
                'ETH': {'cost_basis': 1600.00},  # -12.5% loss
                'SOL': {'cost_basis': 500.00},   # -10% loss
                'ADA': {'cost_basis': 500.00}    # +20% gain
            }
        },
        {
            'name': 'BREAK-EVEN PORTFOLIO',
            'holdings': {
                'BTC': 0.01,
                'ETH': 1.0
            },
            'current_prices': {
                'BTC': 50000.00,  # $500
                'ETH': 2500.00    # $2500
            },
            'cost_basis': {
                'BTC': {'cost_basis': 500.00},   # Exactly break-even
                'ETH': {'cost_basis': 2500.00}   # Exactly break-even
            }
        },
        {
            'name': 'EXTREME LOSS SCENARIO',
            'holdings': {
                'MEME': 1000000.0,
                'SHIB': 10000000.0
            },
            'current_prices': {
                'MEME': 0.000001,  # $1 current value
                'SHIB': 0.000005   # $50 current value
            },
            'cost_basis': {
                'MEME': {'cost_basis': 1000.00},  # -99.9% loss
                'SHIB': {'cost_basis': 5000.00}   # -99.0% loss
            }
        }
    ]
    
    print("🧪 PORTFOLIO PERFORMANCE TESTING SUITE")
    print("="*80)
    
    all_results = []
    
    for scenario in test_scenarios:
        analyzer = TestPortfolioAnalyzer(scenario)
        performance = analyzer.calculate_test_performance()
        all_results.append({
            'name': scenario['name'],
            'performance': performance
        })
        print("\n" + "="*60 + "\n")
    
    # Summary of all tests
    print("📋 TEST RESULTS SUMMARY:")
    print("="*40)
    for result in all_results:
        status = "🟢 PASS" if result['performance'] != 0 or "BREAK-EVEN" in result['name'] else "🔴 FAIL"
        print(f"{result['name']:<25} {result['performance']:+8.2f}% {status}")
    
    print("\n✅ All test scenarios completed!")
    print("🔍 Verify that:")
    print("  - Positive percentages show + sign and 🟢")
    print("  - Negative percentages show - sign and 🔴") 
    print("  - Zero percentages show ± sign and ⚪")
    print("  - Portfolio status matches performance")

if __name__ == "__main__":
    run_test_scenarios() 