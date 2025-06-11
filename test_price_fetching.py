#!/usr/bin/env python3
"""
Test price fetching with new symbol mapping
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.trading_systems.pure_5k_system import Pure5KLiveTradingSystem

def test_price_fetching():
    print("\n🧪 Testing Price Fetching with New Symbol Mapping")
    print("=" * 50)
    
    system = Pure5KLiveTradingSystem()
    
    print("\n💰 Crypto Prices:")
    for symbol in ['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'ADA-USD', 'TRX-USD', 'XLM-USD']:
        kraken_symbol = system.symbol_map.get(symbol, 'N/A')
        price = system.get_current_price_online(symbol)
        status = '✅' if price > 0 else '❌'
        print(f"{status} {symbol} (Kraken: {kraken_symbol}): ${price:.2f}")
    
    print("\n📈 Stock Prices (Sample):")
    for symbol in system.tech_stocks[:3]:  # Test first 3 tech stocks
        price = system.get_current_price_online(symbol)
        status = '✅' if price > 0 else '❌'
        print(f"{status} {symbol}: ${price:.2f}")

if __name__ == "__main__":
    test_price_fetching() 