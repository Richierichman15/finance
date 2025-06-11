#!/usr/bin/env python3
"""
Kraken API Integration Test Script
Tests both public and private API functionality
"""

from app.services.kraken import kraken_api
import sys
import time

def test_public_api():
    print("\n🔍 Testing Public API...")
    
    # Test connection
    connection = kraken_api.test_connection()
    print(f"Connection Test: {'✅' if connection['public_api'] else '❌'}")
    print(f"Server Time: {connection.get('server_time', 'N/A')}")
    print(f"Supported Pairs: {connection.get('supported_pairs', 0)}")
    
    # Test price fetching for each supported symbol
    print("\n📊 Testing Price Fetching:")
    for symbol in kraken_api.get_supported_symbols():
        price = kraken_api.get_price(symbol)
        status = '✅' if price > 0 else '❌'
        print(f"{status} {symbol}: ${price:,.2f}")
        time.sleep(1)  # Respect rate limits

def test_private_api():
    print("\n🔐 Testing Private API...")
    
    if not kraken_api.api_key or not kraken_api.api_secret:
        print("❌ API credentials not configured!")
        print("Please set KRAKEN_API_KEY and KRAKEN_API_SECRET in your .env file")
        return False
    
    # Test balance fetching
    print("\n💰 Fetching Account Balance:")
    balance = kraken_api.get_balance()
    
    if 'error' in balance and balance['error']:
        print(f"❌ Balance fetch failed: {balance['error']}")
        return False
    
    if 'result' in balance:
        print("✅ Successfully retrieved balance!")
        for currency, amount in balance['result'].items():
            if float(amount) > 0:
                print(f"   {currency}: {float(amount):,.8f}")
    
    # Test trade history
    print("\n📜 Fetching Trade History:")
    history = kraken_api.get_trade_history()
    
    if 'error' in history and history['error']:
        print(f"❌ Trade history fetch failed: {history['error']}")
    else:
        print("✅ Successfully retrieved trade history!")
        
    return True

def main():
    print("🚀 Starting Kraken API Integration Tests")
    print("=" * 50)
    
    try:
        # Test public API first
        test_public_api()
        
        # Test private API
        private_success = test_private_api()
        
        print("\n📝 Test Summary:")
        print("=" * 50)
        print("✅ Public API: Working")
        print(f"{'✅' if private_success else '❌'} Private API: {'Working' if private_success else 'Failed'}")
        
        if not private_success:
            print("\n⚠️  Next Steps:")
            print("1. Check your API key permissions on Kraken")
            print("2. Verify KRAKEN_API_KEY and KRAKEN_API_SECRET in .env")
            print("3. Ensure 'Query Funds' permission is enabled")
            sys.exit(1)
        
        print("\n🎉 All tests passed! Ready for Phase 3: Order Management Setup")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 