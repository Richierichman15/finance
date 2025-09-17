#!/usr/bin/env python3
import sys
import os
import time
from datetime import datetime, timedelta
import pytz

# Add the parent directory to Python path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.trading_systems.pure_5k_system import Pure5KLiveTradingSystem

MARKET_TZ = pytz.timezone('America/New_York')


def is_market_hours(now_et: datetime) -> bool:
    # Weekdays 9:30 - 16:00 ET
    if now_et.weekday() >= 5:
        return False
    open_t = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now_et <= close_t


def main(loop_seconds: int = 300):
    system = Pure5KLiveTradingSystem(initial_balance=5000.0, paper_trading=True)
    print("🚀 Live runner started (CRYPTO-ONLY mode)")
    print(f"⏱️  Loop interval: {loop_seconds}s")
    print("📈 Trading only cryptocurrency - no stock market trading")

    last_report_date = None
    initialized = False

    try:
        while True:
            now_et = datetime.now(MARKET_TZ)
            date_str = now_et.strftime('%Y-%m-%d')

            # Check if we need to do initial allocation (Day 1)
            if not initialized and system.cash >= system.initial_balance * 0.95:  # Still have most of initial cash
                print(f"🎯 Running initial allocation for {date_str}")
                try:
                    system.execute_day_1_intelligent_allocation(date_str)
                    portfolio_value = system.calculate_portfolio_value_live()
                    active_positions = len([p for p in system.positions.values() if p['shares'] > 0])
                    print(f"✅ Initial allocation complete: ${portfolio_value:,.2f} | Positions: {active_positions}")
                    initialized = True
                except Exception as e:
                    print(f"⚠️  Initial allocation failed: {e}")
                    print("📈 Continuing with live monitoring...")
                    initialized = True  # Don't keep trying if it fails

            # Run full live monitoring cycle (includes buy/sell signals, risk assessment)
            print(f"🔄 Running full trading cycle at {now_et.strftime('%Y-%m-%d %H:%M:%S ET')}")
            try:
                system.run_live_monitoring_cycle()
            except Exception as e:
                print(f"⚠️  Live monitoring cycle failed: {e}")
                # Fallback to crypto check cycle
                system.run_crypto_check_cycle()

            # Once per day at 16:05 ET, export live CSV and save daily report
            today_tag = now_et.strftime('%Y%m%d')
            cutoff = now_et.replace(hour=16, minute=5, second=0, microsecond=0)
            if last_report_date != today_tag and now_et >= cutoff:
                # Export CSV of today's snapshots
                system.export_live_daily_csv()
                # Save text daily report into logs
                system.generate_and_send_daily_report()
                last_report_date = today_tag

            time.sleep(loop_seconds)
    except KeyboardInterrupt:
        print("\n🛑 Live runner stopped by user")


if __name__ == "__main__":
    main()
