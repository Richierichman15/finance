#!/usr/bin/env python3
"""
🚀 MAIN TRADING SYSTEM RUNNER - ORGANIZED & CLEAN
=================================================
Entry point for all trading systems and functionality
Targeting 10% returns through ultra-aggressive strategies
"""

import sys
import os
from datetime import datetime
import argparse

# Add app directory to Python path
sys.path.append(os.path.dirname(__file__))

def main():
    parser = argparse.ArgumentParser(description='Ultra-Aggressive Trading System')
    parser.add_argument('--system', choices=['pure5k', 'enhanced', 'dashboard'], 
                       default='pure5k', help='Trading system to run')
    parser.add_argument('--days', type=int, default=30, help='Days to backtest')
    parser.add_argument('--balance', type=float, default=5000.0, help='Initial balance')
    
    args = parser.parse_args()
    
    print("🎯 ULTRA-AGGRESSIVE TRADING SYSTEM - TARGETING 10% RETURNS")
    print("=" * 60)
    print(f"System: {args.system}")
    print(f"Days: {args.days}")
    print(f"Initial Balance: ${args.balance:,.2f}")
    print(f"Target: 10% returns (stretch goal for best strategy)")
    print("=" * 60)
    
    if args.system == 'pure5k':
        from trading_systems.pure_5k_system import Pure5KTradingSystem
        
        system = Pure5KTradingSystem(initial_balance=args.balance)
        results = system.run_pure_5k_backtest(days=args.days)
        
        if results and not results.get('error'):
            print(f"\n✅ Pure $5K system completed successfully!")
            print(f"📊 Return: {results['return_percentage']:.2f}%")
            print(f"🎯 Target: {'✅ MET' if results['target_met'] else '❌ NOT MET'}")
            return results
        else:
            print("\n❌ Pure $5K system failed")
            return None
            
    elif args.system == 'enhanced':
        from trading_systems.enhanced_system import EnhancedUltraAggressiveV2
        
        # Enhanced system with daily additions (legacy)
        system = EnhancedUltraAggressiveV2(
            initial_balance=args.balance,
            daily_addition_base=0  # Set to 0 for pure trading
        )
        results = system.run_enhanced_backtest(days=args.days)
        return results
        
    elif args.system == 'dashboard':
        from dashboard.dashboard_runner import run_dashboard_with_system
        
        print("🌐 Starting dashboard with live trading system...")
        run_dashboard_with_system(args.balance, args.days)
        return None
    
    else:
        print(f"❌ Unknown system: {args.system}")
        return None

if __name__ == "__main__":
    main()