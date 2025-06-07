#!/usr/bin/env python3
"""
🚀 ENHANCED ULTRA-AGGRESSIVE TRADING SYSTEM V2 - LEGACY
========================================================
NOTE: This is the legacy system with daily additions feature.
For pure trading performance, use pure_5k_system.py instead.
"""

import sys
import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from typing import Dict, List, Tuple, Optional

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

class EnhancedUltraAggressiveV2:
    def __init__(self, initial_balance: float = 5000.0, daily_addition_base: float = 100.0):
        """
        LEGACY SYSTEM: Includes daily additions feature
        For pure trading, set daily_addition_base = 0
        """
        self.initial_balance = initial_balance
        self.daily_addition_base = daily_addition_base  # Base daily addition amount
        self.cash = initial_balance
        self.positions = {}  # {symbol: {'shares': float, 'avg_price': float}}
        self.trades = []
        self.daily_values = []
        self.daily_additions = []  # Track daily money additions
        self.historical_data_cache = {}  # Offline data storage
        self.logger = logging.getLogger(__name__)
        
        # Same symbol universe as pure system
        self.crypto_symbols = [
            'BTC-USD', 'XRP-USD', 'ETH-USD',
            'SOL-USD', 'TRX-USD', 'ADA-USD', 'XLM-USD'
        ]
        
        self.energy_stocks = [
            'XLE', 'XEG', 'KOLD', 'UNG', 'USO', 'NEE', 'DUK'
        ]
        
        self.tech_stocks = [
            'QQQ', 'NVDA', 'MSFT', 'GOOGL', 'TSLA', 'AMD'
        ]
        
        self.etf_symbols = [
            'SPY', 'VTI', 'GLD'
        ]
        
        self.all_symbols = self.crypto_symbols + self.energy_stocks + self.tech_stocks + self.etf_symbols
        
        # ENHANCED ALLOCATION STRATEGY
        self.crypto_allocation = 0.70
        self.energy_allocation = 0.15
        self.tech_allocation = 0.10
        self.etf_allocation = 0.05
        
        print(f"💰 ENHANCED ULTRA-AGGRESSIVE V2:")
        print(f"   💵 Initial Balance: ${self.initial_balance:,.2f}")
        print(f"   💰 Daily Addition Base: ${self.daily_addition_base}")
        print(f"   📊 Total Symbols: {len(self.all_symbols)}")
        if self.daily_addition_base == 0:
            print("   ⚠️  PURE TRADING MODE: No daily additions")

    def run_enhanced_backtest(self, days: int = 30) -> Dict:
        """Run enhanced backtest (legacy with daily additions)"""
        print(f"\n🎯 ENHANCED V2 BACKTEST ({days} DAYS)")
        print("=" * 60)
        
        if self.daily_addition_base > 0:
            print("⚠️  WARNING: This is LEGACY system with daily additions")
            print("   For pure trading performance, use Pure5KTradingSystem")
        
        # Implementation would go here...
        # For now, return basic structure
        results = {
            'initial_balance': self.initial_balance,
            'daily_addition_base': self.daily_addition_base,
            'system_type': 'enhanced_legacy',
            'note': 'Legacy system with daily additions - use Pure5KTradingSystem for pure trading'
        }
        
        return results

def main():
    """Legacy main function"""
    print("⚠️  This is the LEGACY enhanced system.")
    print("   For current pure trading system, use:")
    print("   python3 app/main_runner.py --system pure5k")
    
    system = EnhancedUltraAggressiveV2(initial_balance=5000.0, daily_addition_base=0)
    results = system.run_enhanced_backtest(days=30)
    return results

if __name__ == "__main__":
    main()