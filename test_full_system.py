#!/usr/bin/env python3
"""
Test the full system - web API + trading monitoring
"""
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_web_api():
    """Test if web API starts correctly"""
    try:
        from app.main import app
        print("✅ Web API imports successfully")
        return True
    except Exception as e:
        print(f"❌ Web API error: {e}")
        return False

def test_trading_endpoints():
    """Test if trading monitoring endpoints work"""
    try:
        from app.main import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test trading status endpoint
        response = client.get("/api/trading/status")
        if response.status_code == 200:
            print("✅ Trading status endpoint works")
        else:
            print(f"❌ Trading status endpoint failed: {response.status_code}")
            return False
        
        # Test trading positions endpoint
        response = client.get("/api/trading/positions")
        if response.status_code == 200:
            print("✅ Trading positions endpoint works")
        else:
            print(f"❌ Trading positions endpoint failed: {response.status_code}")
            return False
        
        # Test recent trades endpoint
        response = client.get("/api/trading/recent-trades")
        if response.status_code == 200:
            print("✅ Recent trades endpoint works")
        else:
            print(f"❌ Recent trades endpoint failed: {response.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Trading endpoints error: {e}")
        return False

def test_trading_system():
    """Test if trading system works"""
    try:
        from app.trading_systems.pure_5k_system import Pure5KLiveTradingSystem
        system = Pure5KLiveTradingSystem(initial_balance=5000.0, paper_trading=True)
        print("✅ Trading system creates successfully")
        return True
    except Exception as e:
        print(f"❌ Trading system error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Full System (Web API + Trading Monitoring)")
    print("=" * 60)
    
    success = True
    success &= test_web_api()
    success &= test_trading_endpoints()
    success &= test_trading_system()
    
    print("=" * 60)
    if success:
        print("🎉 All tests passed! System is ready for deployment.")
        print("\n🚀 To run the full system locally:")
        print("   python start.py")
        print("\n🌐 To access the web dashboard:")
        print("   http://localhost:8000")
        print("\n📊 The dashboard will show:")
        print("   - Live trading status")
        print("   - Active positions")
        print("   - Recent trades")
        print("   - Portfolio performance")
    else:
        print("❌ Some tests failed. Check the errors above.")
