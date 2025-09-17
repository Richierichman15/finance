#!/usr/bin/env python3
"""
Crypto Trading System Runner
Run this script to start the crypto-only trading system
"""
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run the crypto trading system"""
    try:
        from app.live_runner import main as run_trading
        print("🚀 Starting Crypto Trading System...")
        print("Press Ctrl+C to stop")
        run_trading()
    except KeyboardInterrupt:
        print("\n🛑 Trading system stopped by user")
    except Exception as e:
        print(f"❌ Error running trading system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
