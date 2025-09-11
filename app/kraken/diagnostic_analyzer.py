#!/usr/bin/env python3
"""
Diagnostic Portfolio Analyzer
Investigates the cost basis calculation issue and API errors
"""

import sys
import os
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.kraken import kraken_api

def diagnose_cost_basis_issue():
    """Diagnose why all assets show 25% performance"""
    print("🔍 DIAGNOSING COST BASIS CALCULATION ISSUE")
    print("="*60)
    
    # Simulate the problematic logic
    test_assets = [
        {'name': 'BTC', 'current_price': 50000, 'amount': 0.001},
        {'name': 'ETH', 'current_price': 3000, 'amount': 1.0},
        {'name': 'SOL', 'current_price': 150, 'amount': 10.0}
    ]
    
    print("🧮 CURRENT LOGIC (PROBLEMATIC):")
    print("-"*40)
    
    for asset in test_assets:
        current_value = asset['amount'] * asset['current_price']
        
        # This is the problematic logic from the code
        estimated_avg_price = asset['current_price'] * 0.8  # Always 80% of current
        cost_basis = asset['amount'] * estimated_avg_price
        
        performance = ((current_value - cost_basis) / cost_basis * 100)
        
        print(f"{asset['name']:<5}: Current ${asset['current_price']:>8,.2f}")
        print(f"      Estimated Purchase ${estimated_avg_price:>8,.2f} (80% of current)")
        print(f"      Performance: {performance:>8.1f}% (ALWAYS 25%!)")
        print()
    
    print("🎯 DIAGNOSIS:")
    print("The formula 'current_price * 0.8' GUARANTEES 25% performance")
    print("Performance = (current - cost) / cost * 100")
    print("Performance = (100 - 80) / 80 * 100 = 25%")
    print("This is why ALL assets show exactly 25%!")
    
def diagnose_api_errors():
    """Diagnose API errors for different asset types"""
    print("\n🔍 DIAGNOSING API ERRORS")
    print("="*60)
    
    test_symbols = [
        ('XXBT', 'XBTUSD', 'Crypto - Should work'),
        ('TSM.EQ', 'TSMUSD', 'Stock - Will fail (Kraken doesn\'t have stocks)'),
        ('AAPL.EQ', 'AAPLUSD', 'Stock - Will fail'),
        ('ETH.F', 'ETHUSD', 'Crypto Future - Should work'),
        ('NONEXISTENT', 'NONEXISTENTUSD', 'Made up - Will fail')
    ]
    
    print("Testing different asset types with Kraken API:")
    print("-"*60)
    
    for asset, test_symbol, description in test_symbols:
        try:
            result = kraken_api.get_ticker(test_symbol)
            if 'result' in result and result['result']:
                print(f"✅ {asset:<12} ({test_symbol:<15}) - {description}")
            else:
                print(f"❌ {asset:<12} ({test_symbol:<15}) - {description}")
        except Exception as e:
            print(f"❌ {asset:<12} ({test_symbol:<15}) - {description} - Error: {str(e)[:30]}...")
    
    print(f"\n📝 EXPLANATION:")
    print(f"• Kraken is a CRYPTO exchange - it only has cryptocurrency prices")
    print(f"• Stock symbols like TSM.EQ (Taiwan Semiconductor) will ALWAYS fail")
    print(f"• This is expected behavior - we need different APIs for stocks")

def suggest_fixes():
    """Suggest fixes for both issues"""
    print("\n💡 SUGGESTED FIXES")
    print("="*60)
    
    print("🔧 FIX 1: Cost Basis Calculation")
    print("-"*30)
    print("Instead of: estimated_avg_price = current_price * 0.8")
    print("Use:")
    print("1. REAL trade history from Kraken API")
    print("2. User-input purchase prices")
    print("3. Historical price data")
    print("4. OR mark as 'estimated' and use different multipliers")
    
    print("\n🔧 FIX 2: Asset Type Handling")
    print("-"*30)
    print("1. Skip stock symbols (*.EQ) from Kraken API calls")
    print("2. Use Yahoo Finance API for stocks")
    print("3. Add asset type detection")
    print("4. Show clear messages for unsupported assets")
    
    print("\n📊 RECOMMENDED CHANGES:")
    print("1. Fix cost basis to use actual purchase data")
    print("2. Add asset type detection (crypto vs stock)")
    print("3. Use appropriate APIs for each asset type")
    print("4. Show 'estimated' vs 'actual' performance")

def test_real_performance_calculation():
    """Test what performance would look like with real data"""
    print("\n🧪 TESTING WITH REALISTIC DATA")
    print("="*60)
    
    realistic_scenarios = [
        {'asset': 'BTC', 'current': 50000, 'purchase': 45000, 'amount': 0.001},
        {'asset': 'ETH', 'current': 3000, 'purchase': 3200, 'amount': 1.0},
        {'asset': 'SOL', 'current': 150, 'purchase': 100, 'amount': 10.0}
    ]
    
    print("With REAL purchase prices:")
    print("-"*30)
    
    for scenario in realistic_scenarios:
        current_value = scenario['amount'] * scenario['current']
        cost_basis = scenario['amount'] * scenario['purchase']
        performance = ((current_value - cost_basis) / cost_basis * 100)
        
        status = "🟢" if performance > 0 else "🔴" if performance < 0 else "⚪"
        
        print(f"{scenario['asset']}: ${scenario['current']:>8,.0f} vs ${scenario['purchase']:>8,.0f} = {status}{performance:+6.1f}%")
    
    print("\nThis shows realistic, varied performance percentages!")

if __name__ == "__main__":
    diagnose_cost_basis_issue()
    diagnose_api_errors()
    suggest_fixes()
    test_real_performance_calculation() 