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
        initial_balance=5000.0,    # Using full amount for better testing
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
        initial_balance=5000.0,
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

def two_week_test():
    """Run a two-week trading test"""
    print("🚀 TWO WEEK LIVE SYSTEM TEST")
    print("="*40)
    print("Running two-week test with all safety features")
    print()
    
    system = Pure5KLiveSystem(
        initial_balance=5000.0,    # Using full amount for better comparison
        paper_trading=True         # Ensure paper trading
    )
    
    try:
        print("\n📈 Starting two-week live paper trading test...")
        print("⚠️  Press Ctrl+C to stop the test safely")
        
        # Start monitoring with standard settings
        system.start_monitoring(
            interval_minutes=5,     # 5-minute monitoring cycles
            duration_hours=336      # Two weeks (14 days * 24 hours)
        )
        
        print("\n✅ Two-week test completed successfully!")
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
    print("3. Two Week Test")
    print("4. Exit")
    print()
    
    while True:
        choice = input("Select test type (1/2/3/4): ").strip()
        
        if choice == '1':
            return quick_test()
        elif choice == '2':
            return full_day_test()
        elif choice == '3':
            return two_week_test()
        elif choice == '4':
            print("👋 Goodbye!")
            return True
        else:
            print("❌ Invalid choice. Please select 1, 2, 3, or 4.")

if __name__ == "__main__":
    main() 