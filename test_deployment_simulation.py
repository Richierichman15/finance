#!/usr/bin/env python3
"""
Test deployment simulation - verify both systems run together
"""
import sys
import os
import time
import threading
import requests
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_web_api_startup():
    """Test if web API can start and respond"""
    try:
        import subprocess
        import signal
        
        print("🌐 Testing web API startup...")
        
        # Start the web API in the background
        env = os.environ.copy()
        env["RUN_MODE"] = "web"
        env["PORT"] = "8001"  # Use different port to avoid conflicts
        
        process = subprocess.Popen(
            [sys.executable, "start.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a bit for startup
        time.sleep(5)
        
        # Test if API is responding
        try:
            response = requests.get("http://localhost:8001/api/health", timeout=5)
            if response.status_code == 200:
                print("✅ Web API responding correctly")
                success = True
            else:
                print(f"❌ Web API returned status {response.status_code}")
                success = False
        except requests.RequestException as e:
            print(f"❌ Web API not responding: {e}")
            success = False
        
        # Clean up
        process.terminate()
        process.wait(timeout=5)
        
        return success
        
    except Exception as e:
        print(f"❌ Web API startup test failed: {e}")
        return False

def test_trading_system_startup():
    """Test if trading system can start"""
    try:
        print("📈 Testing trading system startup...")
        
        from app.trading_systems.pure_5k_system import Pure5KLiveTradingSystem
        from app.shared_state import trading_state
        
        # Create trading system
        system = Pure5KLiveTradingSystem(initial_balance=5000.0, paper_trading=True)
        
        # Test shared state integration
        initial_state = trading_state.get_current_state()
        print(f"   Portfolio Value: ${initial_state['portfolio_value']:,.2f}")
        print(f"   Cash: ${initial_state['cash']:,.2f}")
        print(f"   Crypto Symbols: {len(initial_state['crypto_symbols'])}")
        
        print("✅ Trading system startup successful")
        return True
        
    except Exception as e:
        print(f"❌ Trading system startup failed: {e}")
        return False

def test_integration():
    """Test integration between web API and trading system"""
    try:
        print("🔗 Testing web API + trading system integration...")
        
        from app.shared_state import trading_state
        from app.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test trading endpoints
        status_response = client.get("/api/trading/status")
        positions_response = client.get("/api/trading/positions")
        trades_response = client.get("/api/trading/recent-trades")
        
        if all(r.status_code == 200 for r in [status_response, positions_response, trades_response]):
            print("✅ All trading endpoints working")
            
            # Check data structure
            status_data = status_response.json()
            if all(key in status_data for key in ['portfolio_value', 'cash', 'return_percentage']):
                print("✅ Trading data structure correct")
                return True
            else:
                print("❌ Trading data structure incomplete")
                return False
        else:
            print("❌ Some trading endpoints failed")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all deployment simulation tests"""
    print("🚀 DEPLOYMENT SIMULATION TEST")
    print("=" * 60)
    print(f"🕐 Testing at {datetime.now()}")
    print("=" * 60)
    
    tests = [
        ("Trading System Startup", test_trading_system_startup),
        ("Web API + Trading Integration", test_integration),
        ("Web API Standalone Startup", test_web_api_startup),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        result = test_func()
        results.append((test_name, result))
        print()
    
    print("=" * 60)
    print("📊 TEST RESULTS:")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        all_passed = all_passed and result
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - READY FOR DEPLOYMENT!")
        print("\n🚀 When deployed to Railway:")
        print("   ✅ Web dashboard will be accessible at your Railway URL")
        print("   ✅ Trading system will run automatically in the background")
        print("   ✅ Dashboard will show real-time trading data")
        print("   ✅ Portfolio updates every 5 minutes (300 seconds)")
        print("   ✅ All trades will be visible on the web interface")
        print("\n📊 Dashboard Features:")
        print("   • Live portfolio value and return percentage")
        print("   • Real-time crypto positions with P&L")
        print("   • Recent trades with timestamps and reasons")
        print("   • Interactive allocation chart")
        print("   • Market news and intelligence")
    else:
        print("❌ SOME TESTS FAILED - CHECK ISSUES ABOVE")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
