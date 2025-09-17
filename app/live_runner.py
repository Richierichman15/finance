#!/usr/bin/env python3
import time
from datetime import datetime, timedelta
import pytz
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

    try:
        while True:
            now_et = datetime.now(MARKET_TZ)

            # Always run crypto-only trading (24/7)
            print(f"🔄 Running crypto trading cycle at {now_et.strftime('%Y-%m-%d %H:%M:%S ET')}")
            system.run_crypto_check_cycle()

            # Once per day at 00:00 UTC, export live CSV and save daily report
            today_tag = now_et.strftime('%Y%m%d')
            if last_report_date != today_tag and now_et.hour == 0 and now_et.minute < 5:
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
