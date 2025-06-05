#!/usr/bin/env python3
"""
Stock Data Collection Script for Cron Jobs

This script collects fundamental data for specified stocks and stores them in the database.
Designed to be run 8 times per day via cron jobs.

Usage:
    python collect_stock_data.py
    python collect_stock_data.py --symbols AAPL MSFT GOOGL TSLA
"""

import sys
import os
import argparse
from datetime import datetime

# Add the app directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.stock_data_collector import StockDataCollector

def main():
    parser = argparse.ArgumentParser(description='Collect stock fundamental data')
    parser.add_argument('--symbols', nargs='+', 
                       default=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
                       help='Stock symbols to collect data for')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    print(f"[{datetime.now()}] Starting stock data collection...")
    print(f"Collecting data for: {', '.join(args.symbols)}")
    
    collector = StockDataCollector()
    results = collector.collect_and_store_fundamentals(args.symbols)
    
    # Print results
    print(f"\n=== Collection Results ===")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Successfully processed: {len(results['symbols_processed'])} symbols")
    print(f"Failed: {len(results['symbols_failed'])} symbols")
    print(f"Total records created: {results['total_records_created']}")
    
    if results['symbols_processed']:
        print(f"✅ Success: {', '.join(results['symbols_processed'])}")
    
    if results['symbols_failed']:
        print(f"❌ Failed: {', '.join(results['symbols_failed'])}")
        
    if args.verbose:
        # Show latest data for successful symbols
        for symbol in results['symbols_processed']:
            latest = collector.get_latest_fundamentals(symbol, 1)
            if latest:
                data = latest[0]
                print(f"\n📊 {symbol} Latest Data:")
                print(f"   Price: ${data['price']:.2f}")
                print(f"   P/E Ratio: {data['pe_ratio_ttm']}")
                print(f"   Market Cap: {data['market_cap']:,}" if data['market_cap'] else "   Market Cap: N/A")
    
    print(f"\n[{datetime.now()}] Collection completed.")
    
    # Exit with error code if any symbols failed
    if results['symbols_failed']:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 