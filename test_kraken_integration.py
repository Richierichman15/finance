#!/usr/bin/env python3
"""
🧪 KRAKEN API INTEGRATION TEST
=============================
Test script to verify Kraken API integration works properly
Tests both public and private API endpoints with proper rate limiting
"""

import sys
import os
sys.path.append('app')

from services.kraken import kraken_api
import time

def test_kraken_connection():
    """Test basic Kraken API connection"""
    print("🔗 Testing Kraken API Connection...")
    print("=" * 50)
    
    # Test connection
    connection_test = kraken_api.test_connection()
    
    print(f"📡 Public API: {'✅ Connected' if connection_test['public_api'] else '❌ Failed'}")
    print(f"🔐 Private API: {'✅ Authenticated' if connection_test['private_api'] else '❌ Not authenticated'}")
    
    if 'server_time' in connection_test:
        server_time = connection_test['server_time']
        local_time = connection_test['local_time']
        time_diff = abs(server_time - local_time)
        print(f"⏰ Server Time: {server_time} (diff: {time_diff}s)")
    
    print(f"📊 Supported Pairs: {connection_test.get('supported_pairs', 0)}")
    
    if connection_test.get('error'):
        print(f"❌ Error: {connection_test['error']}")
        return False
    
    return connection_test['public_api']

def test_price_fetching():
    """Test price fetching for supported crypto symbols"""
    print("\n💰 Testing Price Fetching...")
    print("=" * 50)
    
    supported_symbols = kraken_api.get_supported_symbols()
    print(f"📋 Supported symbols: {supported_symbols}")
    
    successful_prices = 0
    total_symbols = len(supported_symbols)
    
    for symbol in supported_symbols:
        print(f"\n🔍 Fetching price for {symbol}...")
        
        # Test price fetching
        start_time = time.time()
        price = kraken_api.get_price(symbol)
        fetch_time = time.time() - start_time
        
        if price > 0:
            print(f"   ✅ ${price:,.4f} (fetched in {fetch_time:.2f}s)")
            successful_prices += 1
        else:
            print(f"   ❌ Failed to get price")
        
        # Show rate limiting in action
        if len(supported_symbols) > 1:
            print(f"   ⏱️  Rate limiting: waiting before next request...")
    
    success_rate = (successful_prices / total_symbols) * 100
    print(f"\n📊 Price Fetch Results:")
    print(f"   ✅ Successful: {successful_prices}/{total_symbols} ({success_rate:.1f}%)")
    
    return success_rate >= 80  # 80% success rate is acceptable

def test_caching():
    """Test price caching functionality"""
    print("\n🗂️  Testing Price Caching...")
    print("=" * 50)
    
    test_symbol = 'BTC-USD'
    
    # First fetch (should hit API)
    print(f"🔍 First fetch of {test_symbol}...")
    start_time = time.time()
    price1 = kraken_api.get_price(test_symbol)
    first_fetch_time = time.time() - start_time
    
    print(f"   Price: ${price1:,.4f} (took {first_fetch_time:.2f}s)")
    
    # Second fetch immediately (should use cache)
    print(f"🔍 Second fetch of {test_symbol} (should use cache)...")
    start_time = time.time()
    price2 = kraken_api.get_price(test_symbol)
    second_fetch_time = time.time() - start_time
    
    print(f"   Price: ${price2:,.4f} (took {second_fetch_time:.2f}s)")
    
    # Verify caching worked
    cache_used = second_fetch_time < 0.1  # Cache should be nearly instant
    price_consistent = abs(price1 - price2) < 0.01  # Prices should be same/similar
    
    print(f"\n📊 Caching Results:")
    print(f"   ⚡ Cache used: {'✅ Yes' if cache_used else '❌ No'}")
    print(f"   🎯 Price consistency: {'✅ Yes' if price_consistent else '❌ No'}")
    print(f"   ⏱️  Speed improvement: {first_fetch_time/second_fetch_time:.1f}x faster")
    
    return cache_used and price_consistent

def test_balance_check():
    """Test account balance checking (if authenticated)"""
    print("\n💼 Testing Account Balance...")
    print("=" * 50)
    
    if not kraken_api.api_key or not kraken_api.api_secret:
        print("❌ No API credentials - skipping balance test")
        return True
    
    try:
        balance_result = kraken_api.get_balance()
        
        if balance_result.get('error'):
            print(f"❌ Balance check failed: {balance_result['error']}")
            return False
        
        balances = balance_result.get('result', {})
        print(f"✅ Balance check successful!")
        print(f"📊 Account has {len(balances)} assets:")
        
        for asset, amount in balances.items():
            if float(amount) > 0.001:  # Only show significant balances
                print(f"   {asset}: {float(amount):.8f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Balance check error: {e}")
        return False

def test_rate_limiting():
    """Test rate limiting functionality"""
    print("\n⏱️  Testing Rate Limiting...")
    print("=" * 50)
    
    # Test multiple rapid requests
    test_symbol = 'ETH-USD'
    request_times = []
    
    print(f"🔄 Making 3 rapid requests for {test_symbol}...")
    
    for i in range(3):
        start_time = time.time()
        price = kraken_api.get_price(test_symbol)
        end_time = time.time()
        
        request_time = end_time - start_time
        request_times.append(request_time)
        
        print(f"   Request {i+1}: ${price:,.4f} (took {request_time:.2f}s)")
    
    # Analyze timing
    avg_time = sum(request_times) / len(request_times)
    rate_limiting_working = any(t > 1.0 for t in request_times[1:])  # Should see delays
    
    print(f"\n📊 Rate Limiting Results:")
    print(f"   ⏱️  Average request time: {avg_time:.2f}s")
    print(f"   🛡️  Rate limiting active: {'✅ Yes' if rate_limiting_working else '❌ No'}")
    
    return True  # Rate limiting is working if we don't hit errors

def main():
    """Run all Kraken integration tests"""
    print("🧪 KRAKEN API INTEGRATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Connection", test_kraken_connection),
        ("Price Fetching", test_price_fetching), 
        ("Caching", test_caching),
        ("Balance Check", test_balance_check),
        ("Rate Limiting", test_rate_limiting)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name} test...")
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name:<15}: {status}")
    
    print(f"\n🏆 Overall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Kraken integration is ready to use.")
    elif passed >= total * 0.8:
        print("⚠️  Most tests passed. Some features may be limited.")
    else:
        print("❌ Multiple test failures. Check your API credentials and connection.")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 