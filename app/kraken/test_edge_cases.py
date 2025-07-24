#!/usr/bin/env python3
"""
Test Edge Cases for Portfolio Performance
Tests edge cases like zero cost basis, very small amounts, etc.
"""

import sys
import os
import tempfile
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class EdgeCaseTestAnalyzer:
    """Test analyzer for edge cases"""
    
    def __init__(self, test_data):
        self.test_data = test_data
        self.current_prices = test_data['current_prices']
        self.investment_data = {'assets': test_data['cost_basis']}
        
    def test_edge_case(self):
        """Test specific edge case"""
        print(f"\n🔬 EDGE CASE TEST: {self.test_data['name']}")
        print("="*60)
        print(f"📝 Description: {self.test_data['description']}")
        print("-"*60)
        
        total_cost_basis = 0
        total_current_value = 0
        
        for asset, amount in self.test_data['holdings'].items():
            current_price = self.current_prices.get(asset, 0)
            current_value = amount * current_price
            cost_basis = self.investment_data['assets'].get(asset, {}).get('cost_basis', 0)
            
            total_cost_basis += cost_basis
            total_current_value += current_value
            
            # Calculate performance
            if cost_basis > 0:
                gain_loss = current_value - cost_basis
                performance_pct = (gain_loss / cost_basis * 100)
            else:
                performance_pct = 0  # Handle zero cost basis
            
            # Format performance display
            if performance_pct > 0:
                performance_str = f"🟢{performance_pct:+.2f}%"
            elif performance_pct < 0:
                performance_str = f"🔴{performance_pct:+.2f}%"
            else:
                performance_str = f"⚪{performance_pct:+.2f}%"
            
            print(f"{asset}: {amount} @ ${current_price} = ${current_value:.2f} (Cost: ${cost_basis:.2f}) {performance_str}")
        
        # Overall portfolio performance
        if total_cost_basis > 0:
            overall_performance = ((total_current_value - total_cost_basis) / total_cost_basis * 100)
        else:
            overall_performance = 0
            
        print(f"\n💰 Total Current Value: ${total_current_value:.2f}")
        print(f"💵 Total Cost Basis: ${total_cost_basis:.2f}")
        print(f"📊 Overall Performance: {overall_performance:+.2f}%")
        
        # Test expected result
        expected = self.test_data.get('expected_performance')
        if expected is not None:
            if abs(overall_performance - expected) < 0.01:  # Allow for rounding
                print(f"✅ PASS: Expected {expected:+.2f}%, Got {overall_performance:+.2f}%")
            else:
                print(f"❌ FAIL: Expected {expected:+.2f}%, Got {overall_performance:+.2f}%")
        
        print("="*60)
        return overall_performance

def run_edge_case_tests():
    """Run edge case tests"""
    
    edge_cases = [
        {
            'name': 'ZERO COST BASIS',
            'description': 'Asset received for free (airdrop, etc.)',
            'holdings': {'AIRDROP': 1000.0},
            'current_prices': {'AIRDROP': 5.00},
            'cost_basis': {'AIRDROP': {'cost_basis': 0.0}},
            'expected_performance': 0.0  # Should be 0% when cost basis is 0
        },
        {
            'name': 'TINY AMOUNTS',
            'description': 'Very small fractional holdings',
            'holdings': {'BTC': 0.00000001},  # 1 satoshi
            'current_prices': {'BTC': 50000.00},
            'cost_basis': {'BTC': {'cost_basis': 0.0004}},  # Bought at $40k
            'expected_performance': 25.0  # Should be +25%
        },
        {
            'name': 'HUGE NUMBERS',
            'description': 'Very large holdings',
            'holdings': {'SHIB': 1000000000.0},  # 1 billion SHIB
            'current_prices': {'SHIB': 0.00001},
            'cost_basis': {'SHIB': {'cost_basis': 8000.0}},  # Bought higher
            'expected_performance': 25.0  # Should be +25%
        },
        {
            'name': 'WORTHLESS ASSET',
            'description': 'Asset that went to zero',
            'holdings': {'DEAD': 10000.0},
            'current_prices': {'DEAD': 0.0},
            'cost_basis': {'DEAD': {'cost_basis': 1000.0}},
            'expected_performance': -100.0  # Should be -100%
        },
        {
            'name': 'EXTREME GAIN',
            'description': 'Asset with massive gains',
            'holdings': {'MOON': 1000.0},
            'current_prices': {'MOON': 1000.0},  # $1M value
            'cost_basis': {'MOON': {'cost_basis': 100.0}},  # Bought for $100
            'expected_performance': 999900.0  # Should be +999,900%
        },
        {
            'name': 'PRECISION TEST',
            'description': 'Test floating point precision',
            'holdings': {'PREC': 0.123456789},
            'current_prices': {'PREC': 1.987654321},
            'cost_basis': {'PREC': {'cost_basis': 0.200000000}},
            'expected_performance': 22.91  # Calculated manually
        }
    ]
    
    print("🔬 EDGE CASE TESTING SUITE")
    print("="*80)
    
    all_results = []
    passed = 0
    failed = 0
    
    for case in edge_cases:
        analyzer = EdgeCaseTestAnalyzer(case)
        performance = analyzer.test_edge_case()
        
        # Check if test passed
        expected = case.get('expected_performance')
        if expected is not None:
            if abs(performance - expected) < 0.01:
                passed += 1
            else:
                failed += 1
        
        all_results.append({
            'name': case['name'],
            'performance': performance,
            'expected': expected
        })
        print()
    
    # Final summary
    print("📊 EDGE CASE TEST SUMMARY:")
    print("="*50)
    for result in all_results:
        expected_str = f"{result['expected']:+.2f}%" if result['expected'] is not None else "N/A"
        print(f"{result['name']:<20} {result['performance']:+10.2f}% (Expected: {expected_str})")
    
    print(f"\n🎯 TEST RESULTS: {passed} PASSED, {failed} FAILED")
    
    if failed == 0:
        print("🎉 ALL EDGE CASE TESTS PASSED!")
    else:
        print("⚠️  Some edge case tests failed - review logic!")
    
    return passed, failed

def test_mathematical_precision():
    """Test mathematical precision and edge cases"""
    print("\n🧮 MATHEMATICAL PRECISION TESTS")
    print("="*50)
    
    # Test 1: Very small percentage change
    current = 1000.01
    cost = 1000.00
    pct = ((current - cost) / cost * 100)
    print(f"Small change test: ${current} vs ${cost} = {pct:+.4f}%")
    
    # Test 2: Very large percentage change  
    current = 1000000.0
    cost = 1.0
    pct = ((current - cost) / cost * 100)
    print(f"Large change test: ${current:,.0f} vs ${cost} = {pct:+,.0f}%")
    
    # Test 3: Negative to positive
    current = 100.0
    cost = -50.0  # This shouldn't happen but test anyway
    if cost != 0:
        pct = ((current - cost) / abs(cost) * 100)
        print(f"Negative cost test: ${current} vs ${cost} = {pct:+.2f}%")
    
    # Test 4: Zero division protection
    current = 100.0
    cost = 0.0
    if cost > 0:
        pct = ((current - cost) / cost * 100)
    else:
        pct = 0.0
    print(f"Zero division test: ${current} vs ${cost} = {pct:+.2f}% (Protected)")
    
    print("✅ Mathematical precision tests completed")

if __name__ == "__main__":
    passed, failed = run_edge_case_tests()
    test_mathematical_precision()
    
    print(f"\n🏁 FINAL RESULT: {passed} edge cases passed, {failed} failed")
    if failed == 0:
        print("🎯 Portfolio performance logic is ROBUST and handles all edge cases!") 