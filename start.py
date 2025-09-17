#!/usr/bin/env python3
"""
Startup script for Railway deployment
Can run either the web API or the trading system based on environment variables
"""
import os
import sys
import subprocess
import threading
import time
from datetime import datetime

def run_web_api():
    """Run the FastAPI web application"""
    print("🌐 Starting FastAPI web server...")
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "app.main:app", 
        "--host", "0.0.0.0", 
        "--port", os.getenv("PORT", "8000")
    ]
    subprocess.run(cmd)

def run_trading_system():
    """Run the live crypto trading system"""
    print("📈 Starting live CRYPTO-ONLY trading system...")
    from app.live_runner import main
    main()

def run_both():
    """Run both web API and crypto trading system concurrently"""
    print("🚀 Starting both web API and CRYPTO-ONLY trading system...")
    
    # Start web API in a separate thread
    web_thread = threading.Thread(target=run_web_api, daemon=True)
    web_thread.start()
    
    # Give web API time to start
    time.sleep(5)
    
    # Start crypto trading system in main thread
    run_trading_system()

def main():
    """Main entry point"""
    print(f"🕐 Starting at {datetime.now()}")
    
    # Check environment variables to determine what to run
    run_mode = os.getenv("RUN_MODE", "web").lower()
    
    if run_mode == "web":
        run_web_api()
    elif run_mode == "trading":
        run_trading_system()
    elif run_mode == "both":
        run_both()
    else:
        print(f"❌ Unknown RUN_MODE: {run_mode}")
        print("Available modes: web, trading, both")
        sys.exit(1)

if __name__ == "__main__":
    main()
