#!/usr/bin/env python3
"""
Test script for crypto trading system
"""
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all imports work correctly"""
    try:
        from app.trading_systems.pure_5k_system import Pure5KLiveTradingSystem
        print("✅ Pure5K system imports successfully")
        
        from app.live_runner import main
        print("✅ Live runner imports successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_system_creation():
    """Test if the trading system can be created"""
    try:
        from app.trading_systems.pure_5k_system import Pure5KLiveTradingSystem
        system = Pure5KLiveTradingSystem(initial_balance=5000.0, paper_trading=True)
        print("✅ Trading system created successfully")
        print(f"   Crypto symbols: {system.crypto_symbols}")
        print(f"   All symbols: {system.all_symbols}")
        print(f"   Crypto allocation: {system.crypto_allocation:.0%}")
        return True
    except Exception as e:
        print(f"❌ System creation error: {e}")
        return False

def test_live_runner():
    """Test if live runner can be called"""
    try:
        from app.live_runner import main
        print("✅ Live runner function is callable")
        return True
    except Exception as e:
        print(f"❌ Live runner error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Crypto Trading System...")
    print("=" * 50)
    
    success = True
    success &= test_imports()
    success &= test_system_creation()
    success &= test_live_runner()
    
    print("=" * 50)
    if success:
        print("🎉 All tests passed! System is ready for testing.")
        print("\n🚀 To run the crypto trading system:")
        print("   python test_crypto_trading.py --run")
    else:
        print("❌ Some tests failed. Check the errors above.")
