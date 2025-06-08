#!/usr/bin/env python3
"""
🚀 QUICK TEST RUNNER FOR LIVE TRADING SYSTEM
============================================
Run the comprehensive live trading system with all 5 safety recommendations
"""

import sys
import os

# Add the trading systems directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'trading_systems'))

from live_system import Pure5KLiveSystem

def quick_test():
    """Run a quick 30-minute test of the live system"""
    print("🚀 QUICK LIVE SYSTEM TEST")
    print("="*40)
    print("Running 30-minute test with all safety features enabled")
    print()
    
    # Initialize with all safety features
    system = Pure5KLiveSystem(
        initial_balance=2500.0,    # STEP 5: Smaller capital
        paper_trading=True         # STEP 1: Paper trading mode
    )
    
    try:
        # Quick 30-minute test
        system.start_monitoring(
            interval_minutes=2,     # Check every 2 minutes for faster testing
            duration_hours=0.5      # 30 minutes
        )
        
        print("\n✅ Quick test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def full_day_test():
    """Run a full 8-hour trading day test"""
    print("🚀 FULL DAY LIVE SYSTEM TEST")
    print("="*40)
    print("Running full 8-hour test with all safety features")
    print()
    
    system = Pure5KLiveSystem(
        initial_balance=2500.0,
        paper_trading=True
    )
    
    try:
        # Full trading day test
        system.start_monitoring(
            interval_minutes=5,     # Standard 5-minute cycles
            duration_hours=8        # Full trading day
        )
        
        print("\n✅ Full day test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

def main():
    """Main test menu"""
    print("🚀 PURE $5K LIVE TRADING SYSTEM TEST MENU")
    print("="*50)
    print("1. Quick Test (30 minutes)")
    print("2. Full Day Test (8 hours)")
    print("3. Exit")
    print()
    
    while True:
        choice = input("Select test type (1/2/3): ").strip()
        
        if choice == '1':
            return quick_test()
        elif choice == '2':
            return full_day_test()
        elif choice == '3':
            print("👋 Goodbye!")
            return True
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main() 