#!/usr/bin/env python3
from app.services.stock_data_collector import StockDataCollector

def test_data_collection():
    collector = StockDataCollector()
    print('Testing with new symbols:', collector.tracked_stocks)
    
    # Test with one symbol first
    print('\nTesting SPY...')
    results = collector.collect_and_store_fundamentals(['SPY'])
    print('SPY Results:', results)
    
    # Test crypto
    print('\nTesting BTC-USD...')
    results = collector.collect_and_store_fundamentals(['BTC-USD'])
    print('BTC-USD Results:', results)
    
    # Test all symbols
    print('\nTesting all symbols...')
    results = collector.collect_and_store_fundamentals()
    print('All symbols results:', results)

if __name__ == "__main__":
    test_data_collection()