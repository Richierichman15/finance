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
    parser.add_argument('--system', choices=['pure5k', 'pure5kv3', 'pure5kv4ai', 'enhanced', 'dashboard'], 
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
        from trading_systems.pure_5k_system import Pure5KLiveTradingSystem
        
        system = Pure5KLiveTradingSystem(initial_balance=args.balance)
        results = system.run_pure_5k_backtest(days=args.days)
        
        if results and not results.get('error'):
            print(f"\n✅ Pure $5K system completed successfully!")
            print(f"📊 Return: {results['return_percentage']:.2f}%")
            print(f"🎯 Target: {'✅ MET' if results['target_met'] else '❌ NOT MET'}")
        else:
            print(f"\n❌ Pure $5K system failed: {results.get('error', 'Unknown error')}")
    
    elif args.system == 'pure5kv3':
        from trading_systems.pure_5k_v3_system import Pure5KV3TradingSystem
        
        system = Pure5KV3TradingSystem(initial_balance=args.balance)
        results = system.run_pure_5k_v3_backtest(days=args.days)
        
        if results and not results.get('error'):
            print(f"\n✅ Pure $5K V3 system completed successfully!")
            print(f"📊 Return: {results['return_percentage']:.2f}%")
            print(f"🎯 Target: {'✅ MET' if results['target_met'] else '❌ NOT MET'}")
        else:
            print(f"\n❌ Pure $5K V3 system failed: {results.get('error', 'Unknown error')}")
    
    elif args.system == 'pure5kv4ai':
        from trading_systems.pure_5k_v4_ai_system import Pure5KV4AITradingSystem
        
        system = Pure5KV4AITradingSystem(initial_balance=args.balance)
        results = system.run_ai_enhanced_backtest(days=args.days)
        
        if results and not results.get('error'):
            print(f"\n✅ Pure $5K V4 AI system completed successfully!")
            print(f"📊 Return: {results['return_percentage']:.2f}%")
            print(f"🎯 Target: {'✅ MET' if results['target_met'] else '❌ NOT MET'}")
            
            # Display AI metrics if available
            ai_metrics = results.get('ai_metrics', {})
            if ai_metrics and ai_metrics.get('sample_size', 0) > 0:
                print(f"\n🧠 AI Performance:")
                print(f"   📊 Direction Accuracy: {ai_metrics.get('direction_accuracy', 0):.1%}")
                print(f"   📈 Return MAE: {ai_metrics.get('return_mae', 0):.3f}")
                print(f"   📋 Predictions: {ai_metrics.get('sample_size', 0)}")
        else:
            print(f"\n❌ Pure $5K V4 AI system failed: {results.get('error', 'Unknown error')}")
    
    elif args.system == 'enhanced':
        print("🔧 Enhanced system not implemented yet")
    
    elif args.system == 'dashboard':
        print("📊 Dashboard not implemented yet")
    
    else:
        print(f"❌ Unknown system: {args.system}")

if __name__ == "__main__":
    main()