#!/usr/bin/env python3
"""
🚀 PURE $5K LIVE TRADING SYSTEM - PAPER TRADING & MONITORING
===========================================================
Strategy: Live implementation with paper trading mode
Features: Real-time monitoring, risk management, daily reports, alerts
Recommendations Implementation: All 5 steps for safe live testing
"""

import sys
import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from typing import Dict, List, Tuple, Optional
import pytz
import time
import schedule
import threading
from pure_5k_system import Pure5KTradingSystem

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler('app/logs/live_trading.log'),
        logging.StreamHandler()
    ]
)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

class Pure5KLiveSystem(Pure5KTradingSystem):
    """
    STEP 1: Paper Trading Implementation
    STEP 2: Daily Monitoring System  
    STEP 3: Enhanced Logging & Monitoring
    STEP 4: Risk Management Rules
    STEP 5: Smaller Capital Testing
    """
    
    def __init__(self, initial_balance: float = 2500.0, paper_trading: bool = True):
        # STEP 5: Start with smaller capital ($2500 instead of $5000)
        super().__init__(initial_balance)
        
        # STEP 1: Paper Trading Mode
        self.paper_trading = paper_trading
        self.live_mode = not paper_trading
        self.monitoring_active = False
        
        # STEP 4: Risk Management Rules
        self.risk_rules = {
            'max_daily_loss_pct': -3.0,     # Stop if daily loss > 3%
            'max_total_loss_pct': -8.0,     # Stop if total loss > 8%
            'max_position_size_pct': 20.0,  # No single position > 20%
            'max_trades_per_day': 4,        # Max 4 trades per day
            'min_cash_reserve_pct': 15.0,   # Keep 15% minimum cash
            'max_drawdown_pct': -12.0       # Emergency stop at 12% drawdown
        }
        
        # STEP 2 & 3: Monitoring and Logging
        self.daily_start_value = self.initial_balance
        self.daily_trade_count = 0
        self.emergency_stop = False
        self.performance_log = []
        self.alert_log = []
        self.trade_signals_log = []
        
        # Enhanced tracking
        self.max_portfolio_value = self.initial_balance
        self.current_drawdown = 0.0
        self.consecutive_loss_days = 0
        
        # Create necessary directories
        os.makedirs('app/logs', exist_ok=True)
        os.makedirs('app/data/live', exist_ok=True)
        os.makedirs('app/reports', exist_ok=True)
        
        print(f"🚀 PURE $5K LIVE SYSTEM INITIALIZED")
        print(f"💰 Capital: ${self.initial_balance:,.2f} (STEP 5: Reduced for testing)")
        print(f"📝 Mode: {'PAPER TRADING' if paper_trading else 'LIVE TRADING'} (STEP 1)")
        print(f"📊 Monitoring: {'ACTIVE' if self.monitoring_active else 'READY'} (STEP 2)")
        print(f"📋 Logging: ENHANCED (STEP 3)")
        print(f"🛡️  Risk Rules: ACTIVE (STEP 4)")
        print("="*60)

    def get_live_price(self, symbol: str) -> float:
        """STEP 1: Get real-time price for paper trading"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Try multiple methods for current price
            for method in [
                lambda: ticker.history(period='1d', interval='1m')['Close'].iloc[-1],
                lambda: ticker.history(period='1d')['Close'].iloc[-1],
                lambda: ticker.info.get('regularMarketPrice', 0),
                lambda: ticker.info.get('previousClose', 0)
            ]:
                try:
                    price = float(method())
                    if price > 0:
                        return price
                except Exception:
                    continue
            
            # Fallback to cached data
            return self.get_price_from_cache(symbol)
            
        except Exception as e:
            self.logger.error(f"Live price fetch failed for {symbol}: {e}")
            return 0.0

    def check_risk_management_rules(self) -> Tuple[bool, List[str]]:
        """STEP 4: Comprehensive risk management system"""
        violations = []
        current_value = self.calculate_live_portfolio_value()
        
        # Daily loss check
        daily_return_pct = ((current_value - self.daily_start_value) / self.daily_start_value) * 100
        if daily_return_pct <= self.risk_rules['max_daily_loss_pct']:
            violations.append(f"Daily loss limit: {daily_return_pct:.1f}% <= {self.risk_rules['max_daily_loss_pct']}%")
        
        # Total loss check
        total_return_pct = ((current_value - self.initial_balance) / self.initial_balance) * 100
        if total_return_pct <= self.risk_rules['max_total_loss_pct']:
            violations.append(f"Total loss limit: {total_return_pct:.1f}% <= {self.risk_rules['max_total_loss_pct']}%")
        
        # Drawdown check
        self.max_portfolio_value = max(self.max_portfolio_value, current_value)
        self.current_drawdown = ((current_value - self.max_portfolio_value) / self.max_portfolio_value) * 100
        if self.current_drawdown <= self.risk_rules['max_drawdown_pct']:
            violations.append(f"Maximum drawdown: {self.current_drawdown:.1f}% <= {self.risk_rules['max_drawdown_pct']}%")
            self.emergency_stop = True
        
        # Daily trade limit
        if self.daily_trade_count >= self.risk_rules['max_trades_per_day']:
            violations.append(f"Daily trade limit: {self.daily_trade_count} >= {self.risk_rules['max_trades_per_day']}")
        
        # Cash reserve check
        cash_reserve_pct = (self.cash / current_value) * 100
        if cash_reserve_pct < self.risk_rules['min_cash_reserve_pct']:
            violations.append(f"Low cash reserve: {cash_reserve_pct:.1f}% < {self.risk_rules['min_cash_reserve_pct']}%")
        
        # Position concentration check
        for symbol, position in self.positions.items():
            if position['shares'] > 0:
                position_value = position['shares'] * self.get_live_price(symbol)
                position_pct = (position_value / current_value) * 100
                if position_pct > self.risk_rules['max_position_size_pct']:
                    violations.append(f"Position too large: {symbol} = {position_pct:.1f}% > {self.risk_rules['max_position_size_pct']}%")
        
        return len(violations) == 0, violations

    def send_alert(self, message: str, level: str = "WARNING") -> None:
        """STEP 3: Enhanced alert system with logging"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert = {
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'portfolio_value': self.calculate_live_portfolio_value(),
            'cash': self.cash
        }
        
        self.alert_log.append(alert)
        
        # Log to file
        with open('app/logs/alerts.log', 'a') as f:
            f.write(f"{timestamp} [{level}] {message}\n")
        
        # Console output
        print(f"🚨 {level}: {message}")
        
        # Log with appropriate level
        if level == "CRITICAL":
            self.logger.critical(message)
        elif level == "ERROR":
            self.logger.error(message)
        else:
            self.logger.warning(message)

    def calculate_live_portfolio_value(self) -> float:
        """Calculate current portfolio value using live prices"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if position['shares'] > 0:
                current_price = self.get_live_price(symbol)
                if current_price > 0:
                    total_value += position['shares'] * current_price
        
        return total_value

    def execute_paper_trade(self, symbol: str, action: str, shares: float, 
                           price: float, reason: str) -> bool:
        """STEP 1: Execute trade in paper trading mode with full logging"""
        if not self.paper_trading:
            self.send_alert("Paper trade attempted in live mode!", "ERROR")
            return False
        
        amount = shares * price
        
        # Pre-trade validation
        if action == 'BUY':
            if amount > self.cash:
                self.logger.warning(f"Insufficient cash for {symbol}: need ${amount:.2f}, have ${self.cash:.2f}")
                return False
            
            # Check if this would violate position size limits
            current_value = self.calculate_live_portfolio_value()
            future_position_value = amount
            future_position_pct = (future_position_value / (current_value + amount)) * 100
            
            if future_position_pct > self.risk_rules['max_position_size_pct']:
                self.logger.warning(f"Trade would exceed position size limit: {future_position_pct:.1f}%")
                return False
        
        elif action == 'SELL':
            if symbol not in self.positions or self.positions[symbol]['shares'] < shares:
                self.logger.warning(f"Insufficient shares to sell {symbol}")
                return False
        
        # Execute the paper trade
        try:
            if action == 'BUY':
                self.cash -= amount
                self._add_to_position(symbol, shares, price, self._get_symbol_category(symbol))
            
            elif action == 'SELL':
                self.positions[symbol]['shares'] -= shares
                self.cash += amount
            
            # Record trade with enhanced logging
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'symbol': symbol,
                'action': action,
                'shares': shares,
                'price': price,
                'amount': amount,
                'reason': reason,
                'portfolio_value_before': self.calculate_live_portfolio_value() - (amount if action == 'BUY' else -amount),
                'portfolio_value_after': self.calculate_live_portfolio_value(),
                'cash_after': self.cash,
                'paper_trade': True
            }
            
            self.trades.append(trade_record)
            self.daily_trade_count += 1
            
            # Enhanced logging
            self.logger.info(f"PAPER {action}: {shares:.6f} {symbol} @ ${price:.4f} = ${amount:.2f} - {reason}")
            
            return True
            
        except Exception as e:
            self.send_alert(f"Paper trade execution failed: {e}", "ERROR")
            return False

    def _get_symbol_category(self, symbol: str) -> str:
        """Get category for symbol"""
        if symbol in self.crypto_symbols:
            return 'crypto'
        elif symbol in self.energy_stocks:
            return 'energy'  
        elif symbol in self.tech_stocks:
            return 'tech'
        elif symbol in self.etf_symbols:
            return 'etf'
        return 'unknown'

    def detect_live_signals(self) -> Dict[str, str]:
        """STEP 2: Live signal detection with enhanced monitoring"""
        signals = {}
        signal_details = []
        
        for symbol in self.all_symbols:
            try:
                # Get recent data for signal analysis
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d', interval='1h')
                
                if len(hist) > 24:
                    # Calculate momentum indicators
                    recent_24h = hist['Close'].tail(24)
                    recent_6h = hist['Close'].tail(6)
                    
                    momentum_24h = (recent_24h.iloc[-1] - recent_24h.iloc[0]) / recent_24h.iloc[0]
                    momentum_6h = (recent_6h.iloc[-1] - recent_6h.iloc[0]) / recent_6h.iloc[0]
                    
                    # Volume analysis if available
                    volume_confirmed = True
                    if 'Volume' in hist.columns and len(hist['Volume']) > 20:
                        avg_volume = hist['Volume'].tail(20).mean()
                        current_volume = hist['Volume'].iloc[-1]
                        volume_confirmed = current_volume > (avg_volume * 1.2)
                    
                    # Signal classification
                    if momentum_6h > 0.08 and volume_confirmed:
                        signal = "EXPLOSIVE_UP"
                    elif momentum_24h > 0.05 and volume_confirmed:
                        signal = "STRONG_UP"
                    elif momentum_6h < -0.06:
                        signal = "STRONG_DOWN"
                    elif momentum_24h < -0.04:
                        signal = "REVERSAL_DOWN"
                    else:
                        signal = "NEUTRAL"
                    
                    signals[symbol] = signal
                    
                    # Log signal details
                    signal_details.append({
                        'symbol': symbol,
                        'signal': signal,
                        'momentum_6h': momentum_6h,
                        'momentum_24h': momentum_24h,
                        'volume_confirmed': volume_confirmed,
                        'current_price': recent_24h.iloc[-1]
                    })
                
                else:
                    signals[symbol] = "NO_DATA"
                    
            except Exception as e:
                self.logger.debug(f"Signal detection failed for {symbol}: {e}")
                signals[symbol] = "ERROR"
        
        # Store signal details for monitoring
        self.trade_signals_log.append({
            'timestamp': datetime.now().isoformat(),
            'signals': signal_details
        })
        
        return signals

    def run_monitoring_cycle(self) -> None:
        """STEP 2: Main monitoring cycle with comprehensive checks"""
        if self.emergency_stop:
            self.send_alert("Emergency stop active - skipping cycle", "CRITICAL")
            return
        
        current_time = datetime.now()
        
        # Check if we should trade (market hours for stocks, 24/7 for crypto focus)
        if not self._is_trading_time(current_time):
            self.logger.info("Outside trading hours")
            return
        
        # STEP 4: Risk management check
        risk_ok, violations = self.check_risk_management_rules()
        if not risk_ok:
            for violation in violations:
                self.send_alert(f"Risk violation: {violation}", "ERROR")
            
            if self.emergency_stop:
                self.send_alert("Emergency stop triggered!", "CRITICAL")
                return
            else:
                self.logger.warning("Risk violations detected - skipping trading cycle")
                return
        
        # Get live signals
        signals = self.detect_live_signals()
        
        # Execute trades based on signals
        for symbol, signal in signals.items():
            if self.daily_trade_count >= self.risk_rules['max_trades_per_day']:
                break
            
            current_price = self.get_live_price(symbol)
            if current_price <= 0:
                continue
            
            # Skip if in cooldown
            if self.is_in_cooldown(symbol, current_time.strftime('%Y-%m-%d')):
                continue
            
            # Execute trades based on signals
            if signal == "EXPLOSIVE_UP" and self.cash > 200:
                buy_amount = min(300, self.cash * 0.3)
                shares = buy_amount / current_price
                
                if self.execute_paper_trade(symbol, 'BUY', shares, current_price,
                                          f"Live explosive signal - 6h momentum > 8%"):
                    self.send_alert(f"Explosive buy executed: {symbol} @ ${current_price:.4f}")
            
            elif signal == "STRONG_UP" and self.cash > 150:
                buy_amount = min(200, self.cash * 0.25)
                shares = buy_amount / current_price
                
                if self.execute_paper_trade(symbol, 'BUY', shares, current_price,
                                          f"Live strong signal - 24h momentum > 5%"):
                    self.send_alert(f"Strong buy executed: {symbol} @ ${current_price:.4f}")
            
            # Handle sell signals
            elif signal in ["REVERSAL_DOWN", "STRONG_DOWN"] and symbol in self.positions:
                if self.positions[symbol]['shares'] > 0:
                    sell_ratio = 0.6 if signal == "REVERSAL_DOWN" else 0.4
                    shares_to_sell = self.positions[symbol]['shares'] * sell_ratio
                    
                    if self.execute_paper_trade(symbol, 'SELL', shares_to_sell, current_price,
                                              f"Live exit signal - {signal}"):
                        self.send_alert(f"Defensive sell executed: {symbol} @ ${current_price:.4f}")
        
        # Update performance tracking
        self._update_performance_log()

    def _is_trading_time(self, current_time: datetime) -> bool:
        """Check if it's appropriate trading time"""
        # For now, trade during market hours (9:30 AM - 4:00 PM ET) on weekdays
        if current_time.weekday() >= 5:  # Weekend
            return False
        
        et_time = current_time.astimezone(self.market_tz)
        market_open = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_open <= et_time <= market_close

    def _update_performance_log(self) -> None:
        """STEP 3: Update performance tracking"""
        current_value = self.calculate_live_portfolio_value()
        
        performance_entry = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': current_value,
            'cash': self.cash,
            'return_pct': ((current_value - self.initial_balance) / self.initial_balance) * 100,
            'daily_return_pct': ((current_value - self.daily_start_value) / self.daily_start_value) * 100,
            'drawdown_pct': self.current_drawdown,
            'active_positions': len([p for p in self.positions.values() if p['shares'] > 0]),
            'daily_trades': self.daily_trade_count
        }
        
        self.performance_log.append(performance_entry)

    def generate_daily_report(self) -> str:
        """STEP 2 & 3: Generate comprehensive daily report"""
        current_time = datetime.now()
        current_value = self.calculate_live_portfolio_value()
        total_return = current_value - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        daily_return_pct = ((current_value - self.daily_start_value) / self.daily_start_value) * 100
        
        # Active positions summary
        active_positions = [(s, p) for s, p in self.positions.items() if p['shares'] > 0]
        
        # Today's trades
        today_str = current_time.strftime('%Y-%m-%d')
        today_trades = [t for t in self.trades if t.get('date') == today_str]
        
        report = f"""
🚀 PURE $5K LIVE SYSTEM - DAILY REPORT
{'='*50}
📅 Date: {current_time.strftime('%Y-%m-%d %H:%M:%S')}
💰 Portfolio Value: ${current_value:,.2f}
💵 Cash Available: ${self.cash:,.2f}
📊 Total Return: ${total_return:+,.2f} ({return_pct:+.2f}%)
📈 Daily Return: {daily_return_pct:+.2f}%
📉 Max Drawdown: {self.current_drawdown:+.2f}%
🔄 Daily Trades: {self.daily_trade_count}/{self.risk_rules['max_trades_per_day']}
📈 Active Positions: {len(active_positions)}

🛡️  RISK MANAGEMENT STATUS:
   Daily Loss Limit: {daily_return_pct:.1f}% / {self.risk_rules['max_daily_loss_pct']}%
   Total Loss Limit: {return_pct:.1f}% / {self.risk_rules['max_total_loss_pct']}%
   Cash Reserve: {(self.cash/current_value)*100:.1f}% / {self.risk_rules['min_cash_reserve_pct']}%
   Max Drawdown: {self.current_drawdown:.1f}% / {self.risk_rules['max_drawdown_pct']}%

📊 ACTIVE POSITIONS:"""
        
        for symbol, position in active_positions:
            current_price = self.get_live_price(symbol)
            position_value = position['shares'] * current_price
            position_return = ((current_price - position['avg_price']) / position['avg_price']) * 100
            position_pct = (position_value / current_value) * 100
            
            report += f"\n   {symbol}: {position['shares']:.6f} @ ${current_price:.4f} = ${position_value:.2f} ({position_return:+.1f}%) [{position_pct:.1f}%]"
        
        if today_trades:
            report += f"\n\n🔄 TODAY'S TRADES:"
            for trade in today_trades:
                report += f"\n   {trade['action']} {trade['shares']:.6f} {trade['symbol']} @ ${trade['price']:.4f} - {trade['reason']}"
        
        if self.alert_log:
            recent_alerts = [a for a in self.alert_log if a['timestamp'].startswith(today_str)]
            if recent_alerts:
                report += f"\n\n🚨 TODAY'S ALERTS ({len(recent_alerts)}):"
                for alert in recent_alerts[-5:]:  # Last 5 alerts
                    report += f"\n   [{alert['level']}] {alert['message']}"
        
        return report

    def save_live_data(self) -> None:
        """STEP 3: Save all live trading data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save performance log
        perf_file = f"app/data/live/performance_{timestamp}.json"
        with open(perf_file, 'w') as f:
            json.dump(self.performance_log, f, indent=2)
        
        # Save trades
        trades_file = f"app/data/live/trades_{timestamp}.json"
        with open(trades_file, 'w') as f:
            json.dump(self.trades, f, indent=2, default=str)
        
        # Save positions
        positions_file = f"app/data/live/positions_{timestamp}.json"
        with open(positions_file, 'w') as f:
            json.dump(self.positions, f, indent=2)
        
        # Save alerts
        alerts_file = f"app/data/live/alerts_{timestamp}.json"
        with open(alerts_file, 'w') as f:
            json.dump(self.alert_log, f, indent=2)
        
        # Save signals log
        signals_file = f"app/data/live/signals_{timestamp}.json"
        with open(signals_file, 'w') as f:
            json.dump(self.trade_signals_log, f, indent=2)

    def start_monitoring(self, interval_minutes: int = 5, duration_hours: int = 8) -> None:
        """
        Start the live monitoring system for a specified duration
        
        Args:
            interval_minutes: How often to check for signals (default: 5 minutes)
            duration_hours: How long to run the monitoring (default: 8 hours)
        """
        self.monitoring_active = True
        self.daily_start_value = self.calculate_live_portfolio_value()
        
        print(f"\n🚀 STARTING LIVE MONITORING SYSTEM")
        print(f"{'='*50}")
        print(f"📝 Mode: {'PAPER TRADING' if self.paper_trading else 'LIVE TRADING'}")
        print(f"💰 Starting Value: ${self.daily_start_value:,.2f}")
        print(f"⏰ Monitoring Interval: {interval_minutes} minutes")
        print(f"⌛ Duration: {duration_hours} hours")
        print(f"🛡️  Risk Management: ACTIVE")
        print(f"📊 Enhanced Logging: ENABLED")
        print(f"📈 Daily Reports: ENABLED")
        
        # Execute initial portfolio allocation
        current_date = datetime.now().strftime('%Y-%m-%d')
        print("\n🔄 Executing initial portfolio allocation...")
        self.execute_day_1_intelligent_allocation(current_date)
        print(f"✅ Initial allocation complete")
        
        # Schedule daily report at market close
        schedule.every().day.at("16:05").do(self._generate_and_save_daily_report)
        
        # Schedule end-of-day reset
        schedule.every().day.at("16:30").do(self._reset_daily_counters)
        
        try:
            cycle_count = 0
            start_time = datetime.now()
            end_time = start_time + timedelta(hours=duration_hours)
            
            while self.monitoring_active and not self.emergency_stop:
                current_time = datetime.now()
                if current_time >= end_time:
                    print(f"\n⌛ Monitoring duration ({duration_hours} hours) completed")
                    break
                    
                cycle_count += 1
                print(f"\n⏰ Monitoring Cycle #{cycle_count} - {current_time.strftime('%H:%M:%S')}")
                
                # Run main monitoring cycle
                self.run_monitoring_cycle()
                
                # Run scheduled tasks
                schedule.run_pending()
                
                # Save data every hour (12 cycles at 5-minute intervals)
                if cycle_count % 12 == 0:
                    self.save_live_data()
                    print("💾 Data saved to disk")
                
                # Brief status update
                current_value = self.calculate_live_portfolio_value()
                return_pct = ((current_value - self.initial_balance) / self.initial_balance) * 100
                print(f"📊 Portfolio: ${current_value:,.2f} ({return_pct:+.2f}%) | Trades Today: {self.daily_trade_count}")
                
                # Calculate time remaining
                time_remaining = end_time - current_time
                hours_remaining = time_remaining.total_seconds() / 3600
                print(f"⏳ Time Remaining: {hours_remaining:.1f} hours")
                
                # Wait for next cycle
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Critical monitoring error: {e}")
            self.send_alert(f"Monitoring system crashed: {e}", "CRITICAL")
            raise  # Re-raise the exception for proper error handling
        finally:
            self.stop_monitoring()

    def _generate_and_save_daily_report(self) -> None:
        """Generate and save daily report"""
        report = self.generate_daily_report()
        
        # Save to file
        date_str = datetime.now().strftime('%Y%m%d')
        report_file = f"app/reports/daily_report_{date_str}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        self.logger.info(f"Daily report saved to {report_file}")

    def _reset_daily_counters(self) -> None:
        """Reset daily tracking counters"""
        self.daily_trade_count = 0
        self.daily_start_value = self.calculate_live_portfolio_value()
        self.logger.info("Daily counters reset for new trading day")

    def stop_monitoring(self) -> None:
        """Stop monitoring and generate final report"""
        self.monitoring_active = False
        
        # Save final data
        self.save_live_data()
        
        # Generate final report
        final_report = self.generate_daily_report()
        print(f"\n🏁 FINAL LIVE SYSTEM REPORT:")
        print(final_report)
        
        # Save final report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_report_file = f"app/reports/final_report_{timestamp}.txt"
        with open(final_report_file, 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        print(f"\n📊 Live monitoring stopped")
        print(f"💾 All data saved to app/data/live/")
        print(f"📋 Final report: {final_report_file}")

def main():
    """Main function to start live system"""
    print("🚀 PURE $5K LIVE TRADING SYSTEM")
    print("="*50)
    print("IMPLEMENTATION OF 5 RECOMMENDATIONS:")
    print("1. ✅ Paper Trading Mode (Safe Testing)")
    print("2. ✅ Daily Monitoring System")  
    print("3. ✅ Enhanced Logging & Monitoring")
    print("4. ✅ Risk Management Rules")
    print("5. ✅ Smaller Capital Testing ($2500)")
    print()
    
    # STEP 5: Start with smaller capital for testing
    # STEP 1: Paper trading mode enabled by default
    system = Pure5KLiveSystem(initial_balance=2500.0, paper_trading=True)
    
    try:
        # STEP 2: Start monitoring (5-minute intervals)
        system.start_monitoring(interval_minutes=5, duration_hours=8)
        
    except Exception as e:
        print(f"❌ System error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 