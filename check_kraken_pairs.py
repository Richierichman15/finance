#!/usr/bin/env python3
"""
🔍 KRAKEN SYMBOL DISCOVERY
=========================
Discover correct symbol mappings for Kraken API
"""

import sys
import os
sys.path.append('app')

import requests
import json

def get_kraken_asset_pairs():
    """Get all available asset pairs from Kraken"""
    try:
        url = 'https://api.kraken.com/0/public/AssetPairs'
        response = requests.get(url)
        data = response.json()
        
        if data.get('error'):
            print(f"❌ Error: {data['error']}")
            return {}
        
        pairs = data.get('result', {})
        print(f"📊 Found {len(pairs)} asset pairs on Kraken")
        
        # Filter for USD pairs we care about
        crypto_symbols = ['BTC', 'ETH', 'XRP', 'SOL', 'ADA', 'TRX', 'XLM']
        relevant_pairs = {}
        
        for pair_id, pair_info in pairs.items():
            pair_name = pair_info.get('wsname', pair_id)
            
            # Look for our crypto symbols paired with USD
            for crypto in crypto_symbols:
                if crypto in pair_name and 'USD' in pair_name:
                    relevant_pairs[pair_id] = {
                        'wsname': pair_name,
                        'base': pair_info.get('base'),
                        'quote': pair_info.get('quote'),
                        'pair_decimals': pair_info.get('pair_decimals', 4),
                        'lot_decimals': pair_info.get('lot_decimals', 8)
                    }
                    break
        
        return relevant_pairs
        
    except Exception as e:
        print(f"❌ Error fetching pairs: {e}")
        return {}

def test_pair_prices(pairs):
    """Test getting prices for discovered pairs"""
    print(f"\n💰 Testing prices for {len(pairs)} pairs...")
    
    for pair_id, pair_info in pairs.items():
        try:
            url = f'https://api.kraken.com/0/public/Ticker?pair={pair_id}'
            response = requests.get(url)
            data = response.json()
            
            if data.get('error'):
                print(f"❌ {pair_id}: {data['error']}")
                continue
            
            if 'result' in data and pair_id in data['result']:
                price_data = data['result'][pair_id]
                last_price = float(price_data['c'][0])  # Last trade price
                
                print(f"✅ {pair_id:12} ({pair_info['wsname']:12}): ${last_price:>12.4f}")
            else:
                print(f"❌ {pair_id}: No price data")
                
        except Exception as e:
            print(f"❌ {pair_id}: Error - {e}")

def generate_symbol_mapping(pairs):
    """Generate correct symbol mapping for our trading system"""
    print(f"\n🔧 Generating symbol mapping...")
    
    mapping = {}
    
    for pair_id, pair_info in pairs.items():
        wsname = pair_info['wsname']
        base = pair_info['base']
        
        # Create our standard format
        if base == 'XXBT':  # Bitcoin special case
            our_symbol = 'BTC-USD'
        else:
            our_symbol = f"{base}-USD"
        
        mapping[our_symbol] = pair_id
        mapping[pair_id] = our_symbol  # Reverse mapping
    
    print("📋 Symbol mapping for trading system:")
    print("    symbol_map = {")
    
    # Our format -> Kraken format
    for our_symbol, kraken_pair in mapping.items():
        if '-USD' in our_symbol:
            print(f"        '{our_symbol}': '{kraken_pair}',")
    
    print()
    
    # Kraken format -> Our format  
    for kraken_pair, our_symbol in mapping.items():
        if '-USD' not in kraken_pair:
            print(f"        '{kraken_pair}': '{our_symbol}',")
    
    print("    }")
    
    return mapping

def main():
    print("🔍 KRAKEN SYMBOL DISCOVERY")
    print("=" * 50)
    
    # Get all asset pairs
    pairs = get_kraken_asset_pairs()
    
    if not pairs:
        print("❌ No pairs found - check connection")
        return
    
    print(f"\n📋 Relevant USD pairs found:")
    for pair_id, pair_info in pairs.items():
        print(f"   {pair_id:12} -> {pair_info['wsname']:12} (Base: {pair_info['base']})")
    
    # Test prices
    test_pair_prices(pairs)
    
    # Generate mapping
    mapping = generate_symbol_mapping(pairs)
    
    print(f"\n🎉 Discovery complete! Found {len(pairs)} relevant pairs.")

if __name__ == "__main__":
    main() 