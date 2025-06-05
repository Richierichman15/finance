#!/usr/bin/env python3
"""
Elite Financial Advisor AI - Startup Script
Launch your profit-maximizing AI advisor with ease!
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def print_banner():
    banner = """
    🚀 ELITE FINANCIAL ADVISOR AI 🚀
    ================================
    💰 The Most Aggressive Profit-Focused AI Financial Advisor! 💰
    
    Ready to turn your portfolio into a MONEY MACHINE? Let's GO!
    """
    print(banner)

def check_python_version():
    """Check if Python version is sufficient"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} - Good to go!")
    return True

def check_ollama():
    """Check if Ollama is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                print(f"✅ Ollama is running with {len(models)} model(s)")
                return True
            else:
                print("⚠️  Ollama is running but no models found")
                print("   Run: ollama pull llama2")
                return False
    except Exception as e:
        print("❌ Ollama not detected. Please:")
        print("   1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   2. Pull a model: ollama pull llama2")
        print("   3. Start server: ollama serve")
        return False

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def start_server():
    """Start the FastAPI server"""
    print("🚀 Launching Elite Financial Advisor AI...")
    print("💰 Get ready to make some SERIOUS MONEY!")
    print("\n" + "="*50)
    print("🌐 Access your advisor at: http://localhost:8000")
    print("💡 Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    try:
        # Start the server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.main:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Elite Financial Advisor AI shutting down...")
        print("💰 Keep making those gains! See you next time!")

def main():
    """Main startup function"""
    print_banner()
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check Ollama (optional - app can still run without it)
    ollama_running = check_ollama()
    if not ollama_running:
        print("\n⚠️  Continuing without Ollama - some AI features may be limited")
        time.sleep(2)
    
    # Start the server
    start_server()

if __name__ == "__main__":
    main()