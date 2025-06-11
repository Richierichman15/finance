#!/usr/bin/env python3
"""
Test script to check available Kraken pairs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.kraken import kraken_api
import json

def test_kraken_pairs():
    print("\n🧪 Testing Kraken Asset Pairs")
    print("=" * 50)
    
    # Get asset pairs from Kraken
    result = kraken_api.kraken_request('/0/public/AssetPairs')
    
    if result.get('error'):
        print(f"❌ Error: {result['error']}")
        return
    
    pairs = result.get('result', {})
    print(f"\n📊 Found {len(pairs)} trading pairs")
    
    # Filter USD pairs
    usd_pairs = {k: v for k, v in pairs.items() if 'USD' in k}
    print(f"\n💵 USD Pairs ({len(usd_pairs)}):")
    for pair, info in usd_pairs.items():
        print(f"   {pair}")
    
    # Save full response for analysis
    with open('kraken_pairs.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n💾 Full response saved to kraken_pairs.json")

if __name__ == "__main__":
    test_kraken_pairs() 