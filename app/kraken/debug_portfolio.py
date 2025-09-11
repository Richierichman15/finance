#!/usr/bin/env python3
"""
🔍 DEBUG PORTFOLIO - See ALL Kraken Assets
==========================================
This script shows every single asset in your Kraken account,
including those that might be filtered out by the main analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.kraken import kraken_api

def debug_portfolio():
    """Debug portfolio to see all assets"""
    print("🔍 DEBUGGING YOUR KRAKEN PORTFOLIO")
    print("="*50)
    
    try:
        # Get raw balance
        balance = kraken_api.get_balance()
        
        if 'result' not in balance:
            print("❌ Could not get balance from Kraken")
            print(f"Response: {balance}")
            return
        
        holdings = balance['result']
        print(f"📊 RAW KRAKEN BALANCE - {len(holdings)} total entries:")
        print("-"*50)
        
        significant_assets = []
        tiny_assets = []
        zero_assets = []
        
        for asset, amount in holdings.items():
            try:
                amount_float = float(amount)
                
                if amount_float > 0.001:
                    significant_assets.append((asset, amount_float))
                elif amount_float > 0:
                    tiny_assets.append((asset, amount_float))
                else:
                    zero_assets.append((asset, amount_float))
                    
            except ValueError:
                print(f"⚠️  Could not parse amount for {asset}: {amount}")
        
        print(f"\n💰 SIGNIFICANT ASSETS (>{0.001}):")
        print("   Asset Code    Amount")
        print("   " + "-"*30)
        for asset, amount in significant_assets:
            print(f"   {asset:12} {amount:>15.8f}")
        
        print(f"\n🔍 TINY AMOUNTS (>0 but <0.001):")
        print("   Asset Code    Amount")
        print("   " + "-"*30)
        for asset, amount in tiny_assets:
            print(f"   {asset:12} {amount:>15.8f}")
        
        print(f"\n❌ ZERO BALANCE ASSETS:")
        for asset, amount in zero_assets[:10]:  # Show first 10
            print(f"   {asset:12} {amount:>15.8f}")
        if len(zero_assets) > 10:
            print(f"   ... and {len(zero_assets)-10} more zero balance assets")
        
        print(f"\n🎯 ASSET MAPPING CHECK:")
        print("   Checking how your assets would be mapped...")
        print("   " + "-"*50)
        
        # Check symbol mapping for significant assets
        symbol_map = {
            'XXBT': 'BTC-USD',
            'XETH': 'ETH-USD', 
            'XXRP': 'XRP-USD',
            'ADA': 'ADA-USD',
            'SOL': 'SOL-USD',
            'XXDG': 'DOGE-USD',
            'DOT': 'DOT-USD',
            'MATIC': 'MATIC-USD',
            'LINK': 'LINK-USD',
            'UNI': 'UNI-USD',
            'AVAX': 'AVAX-USD',
            'BONK': 'BONK-USD',
            'FLOKI': 'FLOKI-USD',
            'PEPE': 'PEPE-USD'
        }
        
        for asset, amount in significant_assets:
            # Handle .F suffix
            if asset.endswith('.F'):
                base_asset = asset.replace('.F', '')
                mapped_symbol = symbol_map.get(base_asset, f"{base_asset}-USD")
            else:
                mapped_symbol = symbol_map.get(asset, f"{asset}-USD")
            
            print(f"   {asset:12} → {mapped_symbol:15} (Amount: {amount:>10.6f})")
        
        print(f"\n🚨 POTENTIAL ISSUES:")
        
        # Check for Bitcoin
        bitcoin_assets = [asset for asset, _ in significant_assets if 'XBT' in asset or 'BTC' in asset]
        if not bitcoin_assets:
            print("   ⚠️  No Bitcoin found in significant assets!")
        else:
            print(f"   ✅ Bitcoin found: {bitcoin_assets}")
        
        # Check for UNI
        uni_assets = [asset for asset, _ in significant_assets if 'UNI' in asset]
        if not uni_assets:
            print("   ⚠️  No Uniswap (UNI) found in significant assets!")
        else:
            print(f"   ✅ Uniswap found: {uni_assets}")
        
        # Check for PEPE
        pepe_assets = [asset for asset, _ in significant_assets if 'PEPE' in asset]
        if not pepe_assets:
            print("   ⚠️  No PEPE found in significant assets!")
        else:
            print(f"   ✅ PEPE found: {pepe_assets}")
        
        print(f"\n💡 SOLUTION RECOMMENDATIONS:")
        if len(significant_assets) != len([asset for asset, _ in significant_assets]):
            print("   1. Some assets might have very small amounts (<0.001)")
            print("   2. Try lowering the minimum threshold in the analysis")
        
        print("   3. Check if missing assets have different Kraken symbols")
        print("   4. Some assets might need manual symbol mapping")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_portfolio() 