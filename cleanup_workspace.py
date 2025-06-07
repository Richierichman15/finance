#!/usr/bin/env python3
"""
🧹 WORKSPACE CLEANUP SCRIPT
===========================
Removes scattered files and organizes everything into clean app structure
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_workspace():
    """Clean up the workspace and organize files"""
    
    print("🧹 CLEANING UP WORKSPACE")
    print("=" * 50)
    
    # Files to remove (scattered/duplicate files)
    files_to_remove = [
        'pure_5k_trading_system.py',  # Moved to app/trading_systems/
        'enhanced_ultra_aggressive_v2.py',  # Moved to app/trading_systems/
        'dashboard_integration.py',  # Replaced by app/dashboard/
        'dashboard.html',  # Replaced by app/dashboard/templates/
        'ultra_aggressive_simulation.py',  # Old simulation files
        'enhanced_trading_simulation.py',
        'aggressive_trading_simulation.py',
        'run_trading_simulation.py',
        'trading_system_summary.md',
        'collect_stock_data.py',  # Functionality in app/services/
        'test_collector.py',
        'start.py',  # Replaced by app/main_runner.py
        'setup_cron.sh',  # Not needed for current system
        'server.log',
        'TRADING_SYSTEM_SUCCESS_SUMMARY.md',  # Old documentation
        'FILE_USAGE_REVIEW.md',
        # Results files that are now in app/data/
        'pure_5k_results_*.json',
        'ultra_aggressive_SUCCESS_30days.json',
        'enhanced_simulation_results.json',
        'simulation_results.json',
        # Cache files that are now in app/data/cache/
        'historical_data_cache_*.pkl',
        'pure_5k_cache_*.pkl'
    ]
    
    # Remove old scattered files
    removed_count = 0
    for pattern in files_to_remove:
        for file_path in glob.glob(pattern):
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"🗑️  Removed: {file_path}")
                    removed_count += 1
                except Exception as e:
                    print(f"⚠️  Could not remove {file_path}: {e}")
    
    print(f"\n✅ Removed {removed_count} scattered files")
    
    # Create organized directory structure
    directories_to_create = [
        'app/data/cache',
        'app/data/results',
        'app/dashboard/data',
        'app/dashboard/templates',
        'app/trading_systems',
        'app/utils',
        'docs',
        'config'
    ]
    
    created_count = 0
    for directory in directories_to_create:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")
            created_count += 1
    
    print(f"\n✅ Created {created_count} organized directories")
    
    # Move important documentation to docs/
    docs_to_move = [
        ('FINAL_SUCCESS_SUMMARY.md', 'docs/FINAL_SUCCESS_SUMMARY.md'),
        ('CURRENT_SYSTEM_ANALYSIS.md', 'docs/CURRENT_SYSTEM_ANALYSIS.md'),
        ('README.md', 'docs/README.md'),
    ]
    
    moved_count = 0
    for src, dst in docs_to_move:
        if os.path.exists(src):
            try:
                shutil.move(src, dst)
                print(f"📄 Moved documentation: {src} → {dst}")
                moved_count += 1
            except Exception as e:
                print(f"⚠️  Could not move {src}: {e}")
    
    print(f"\n✅ Moved {moved_count} documentation files")
    
    # Create new organized README
    create_new_readme()
    
    print("\n🎉 WORKSPACE CLEANUP COMPLETE!")
    print("=" * 50)
    print("📁 NEW ORGANIZED STRUCTURE:")
    print("   app/")
    print("   ├── main_runner.py           # 🚀 Main entry point")
    print("   ├── trading_systems/")
    print("   │   ├── pure_5k_system.py    # 💰 Pure $5K trading")
    print("   │   └── enhanced_system.py   # 🔄 Legacy system")
    print("   ├── dashboard/")
    print("   │   ├── dashboard_runner.py  # 🌐 Dashboard integration")
    print("   │   └── templates/           # 📄 HTML templates")
    print("   ├── services/               # 🛠️  Backend services")
    print("   ├── models/                 # 📊 Database models")
    print("   ├── data/")
    print("   │   ├── cache/              # 💾 Historical data cache")
    print("   │   └── results/            # 📈 Trading results")
    print("   └── utils/                  # 🔧 Utility functions")
    print("   docs/                       # 📚 Documentation")
    print("   config/                     # ⚙️  Configuration files")
    print("\n🎯 TO RUN THE SYSTEM:")
    print("   python3 app/main_runner.py --system pure5k")
    print("   python3 app/main_runner.py --system dashboard")

def create_new_readme():
    """Create a new organized README"""
    
    readme_content = """# 🚀 Ultra-Aggressive Trading System

**Targeting 10% returns through ultra-aggressive trading strategies**

## 🎯 Quick Start

```bash
# Run pure $5K trading system (recommended)
python3 app/main_runner.py --system pure5k

# Run with dashboard visualization
python3 app/main_runner.py --system dashboard

# Run with custom parameters
python3 app/main_runner.py --system pure5k --days 60 --balance 10000
```

## 📁 Project Structure

```
app/
├── main_runner.py           # 🚀 Main entry point
├── trading_systems/
│   ├── pure_5k_system.py    # 💰 Pure trading (NO daily additions)
│   └── enhanced_system.py   # 🔄 Legacy system (with daily additions)
├── dashboard/
│   ├── dashboard_runner.py  # 🌐 Dashboard integration
│   └── templates/           # 📄 HTML templates
├── services/               # 🛠️  Backend services
├── models/                 # 📊 Database models
├── data/
│   ├── cache/              # 💾 Historical data cache
│   └── results/            # 📈 Trading results
└── utils/                  # 🔧 Utility functions
docs/                       # 📚 Documentation
config/                     # ⚙️  Configuration files
```

## 🎯 Trading Strategy

- **Universe**: 23 symbols (7 crypto, 7 energy, 6 tech, 3 ETFs)
- **Allocation**: 70% crypto, 15% energy, 10% tech, 5% ETFs
- **Approach**: Ultra-aggressive momentum trading
- **Target**: 10% returns (stretch goal for best strategy)
- **Capital**: Pure $5K trading (no daily additions)

## 🛠️ Features

✅ **Pure Trading Performance**: No artificial daily additions  
✅ **Cache System**: Offline historical data (90+ days)  
✅ **Fixed Timezone Issues**: Proper UTC handling  
✅ **Multiple Data Sources**: Robust fallback mechanisms  
✅ **Risk Management**: Protective sells and position sizing  
✅ **Dashboard**: Beautiful web visualization  
✅ **Clean Architecture**: Organized, modular codebase  

## 📊 Performance Tracking

The system tracks:
- Portfolio value over time
- Individual trade performance
- Return percentages vs 10% target
- Risk metrics (max drawdown, volatility)
- Trading activity and execution

## 🚨 Risk Warnings

⚠️ **High Risk**: Ultra-aggressive strategies can lose money quickly  
⚠️ **Past Performance**: Does not guarantee future results  
⚠️ **Volatility**: Crypto and momentum strategies are extremely volatile  
⚠️ **Paper Trading**: Test thoroughly before real money implementation  

## 📚 Documentation

- `docs/FINAL_SUCCESS_SUMMARY.md` - Complete system overview
- `docs/CURRENT_SYSTEM_ANALYSIS.md` - Technical analysis and real-money guide

---

*Built for maximum returns through disciplined ultra-aggressive trading* 🎯
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("📄 Created new organized README.md")

def main():
    """Main cleanup function"""
    cleanup_workspace()

if __name__ == "__main__":
    main()