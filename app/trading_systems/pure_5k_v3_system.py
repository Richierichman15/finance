#!/usr/bin/env python3
"""
🚀 PURE $5K ULTRA-AGGRESSIVE TRADING SYSTEM - V3 (REALISTIC)
===========================================================
Strategy: Pure trading performance from $5,000 initial capital only
Focus: Diversified portfolio with realistic risk management
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

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

class Pure5KV3TradingSystem:
    def __init__(self, initial_balance: float = 5000.0):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.positions = {}  # {symbol: {'shares': float, 'avg_price': float, 'peak_price': float, 'category': str}}
        self.trades = []
        self.daily_values = []
        self.historical_data_cache = {}
        self.logger = logging.getLogger(__name__)
        self.trading_halted = False
        
        # V3 REALISTIC RISK MANAGEMENT
        self.risk_per_trade = 0.02
        self.max_position_value = initial_balance * 0.25
        self.portfolio_stop_loss = 0.20
        self.slippage_pct = 0.001
        self.transaction_cost_pct = 0.001
        self.atr_multipliers = {'crypto': 2.5, 'stocks': 2.0}
        self.profit_take_pct = 0.50
        self.cooldown_periods = 2
        self.rebalance_threshold = 0.05
        self.trailing_stops = {}

        # Symbol universe
        self.crypto_symbols = list(set(['BTC-USD', 'ETH-USD', 'XRP-USD', 'SOL-USD', 'TRX-USD', 'ADA-USD', 'XLM-USD', 'BNB-USD', 'USDC-USD', 'ARB-USD']))
        self.energy_stocks = ['XLE', 'KOLD', 'USO', 'ICLN', 'BE', 'LNG', 'XOM']
        self.tech_stocks = ['QQQ', 'NVDA', 'MSFT', 'GOOGL', 'TSLA', 'AMD', 'META', 'AAPL', 'AMZN']
        self.etf_symbols = ['SPY', 'VTI', 'GLD', 'QQQM', 'BIL']
        self.all_symbols = self.crypto_symbols + self.energy_stocks + self.tech_stocks + self.etf_symbols
        
        # Sector allocations
        self.allocations = {
            'crypto': {'target': 0.20, 'range': (0.15, 0.30)},
            'energy': {'target': 0.15, 'range': (0.10, 0.20)},
            'tech': {'target': 0.15, 'range': (0.10, 0.20)},
            'etf': {'target': 0.10, 'range': (0.05, 0.15)}
        }
        
        self.market_tz = pytz.timezone('America/New_York')
        self.utc = pytz.UTC
        
        print("🚀 PURE $5K TRADING SYSTEM V3 (REALISTIC)")
        print(f"   💵 Initial Capital: ${self.initial_balance:,.2f}")
        print(f"   🛡️ Risk per trade: {self.risk_per_trade:.1%}")
        print(f"   📉 Portfolio Stop Loss: {self.portfolio_stop_loss:.1%}")

    def standardize_datetime(self, dt) -> pd.Timestamp:
        if isinstance(dt, str): dt = pd.to_datetime(dt)
        if dt.tz is None: dt = self.market_tz.localize(dt)
        return dt.astimezone(self.utc)

    def cache_historical_data(self, days: int = 90):
        print(f"\n📥 Caching {days} days of historical data...")
        cache_file = f"app/data/cache/pure_5k_v3_cache_{days}days.pkl"
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f: self.historical_data_cache = pickle.load(f)
            print(f"✅ Loaded cached data for {len(self.historical_data_cache)} symbols")
            return

        end_date = datetime.now(self.utc)
        start_date = end_date - timedelta(days=days + 10)
        
        for symbol in self.all_symbols:
            try:
                hist_data = yf.Ticker(symbol).history(start=start_date.date(), end=end_date.date(), interval='1d')
                if not hist_data.empty:
                    self.historical_data_cache[symbol] = {'data': hist_data}
            except Exception as e:
                self.logger.error(f"Failed to cache {symbol}: {e}")

        with open(cache_file, 'wb') as f: pickle.dump(self.historical_data_cache, f)
        print(f"💾 Saved cache to {cache_file}")

    def get_price_from_cache(self, symbol: str, date: str) -> float:
        if symbol not in self.historical_data_cache: return 0.0
        try:
            target_date = pd.to_datetime(date)
            data = self.historical_data_cache[symbol]['data']
            return float(data.loc[data.index.asof(target_date)]['Close'])
        except Exception:
            return 0.0

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean().iloc[-1]

    def calculate_position_size(self, symbol: str, price: float, portfolio_value: float) -> float:
        data = self.historical_data_cache.get(symbol, {}).get('data')
        if data is None or len(data) < 20: return 0.0
        
        atr = self.calculate_atr(data)
        if atr == 0: return 0.0
        
        atr_mult = self.atr_multipliers['crypto' if '-USD' in symbol else 'stocks']
        stop_loss_price = price - (atr * atr_mult)
        if price <= stop_loss_price: return 0.0
        
        risk_amount = portfolio_value * self.risk_per_trade
        shares = risk_amount / (price - stop_loss_price)
        position_value = shares * price
        
        avg_volume_value = data['Volume'].tail(20).mean() * price
        if avg_volume_value > 0:
            position_value = min(position_value, avg_volume_value * 0.01)
        
        return min(position_value, self.max_position_value)

    def _execute_trade(self, date: str, symbol: str, action: str, shares: float, price: float, reason: str):
        category = next((cat for cat, params in self.allocations.items() if symbol in getattr(self, f"{cat}_symbols", [])), 'unknown')
        
        if action == 'BUY':
            exec_price = price * (1 + self.slippage_pct)
            cost = exec_price * shares * (1 + self.transaction_cost_pct)
            if self.cash >= cost:
                self.cash -= cost
                self._add_to_position(symbol, shares, exec_price, category)
                self._record_trade(date, symbol, 'BUY', shares, exec_price, cost, reason)
                return True
        elif action == 'SELL':
            exec_price = price * (1 - self.slippage_pct)
            proceeds = exec_price * shares * (1 - self.transaction_cost_pct)
            self.cash += proceeds
            if symbol in self.positions:
                self.positions[symbol]['shares'] -= shares
                if self.positions[symbol]['shares'] <= 1e-8: # floating point precision
                    del self.positions[symbol]
            self._record_trade(date, symbol, 'SELL', shares, exec_price, proceeds, reason)
            return True
        return False

    def _add_to_position(self, symbol: str, shares: float, price: float, category: str):
        if symbol in self.positions:
            pos = self.positions[symbol]
            total_cost = (pos['shares'] * pos['avg_price']) + (shares * price)
            total_shares = pos['shares'] + shares
            pos['avg_price'] = total_cost / total_shares
            pos['shares'] = total_shares
            pos['peak_price'] = max(pos['peak_price'], price)
        else:
            self.positions[symbol] = {'shares': shares, 'avg_price': price, 'peak_price': price, 'category': category}

    def _record_trade(self, date: str, symbol: str, action: str, shares: float, price: float, amount: float, strategy: str):
        self.trades.append({'date': date, 'symbol': symbol, 'action': action, 'shares': shares, 'price': price, 'amount': amount, 'strategy': strategy})

    def check_portfolio_stop_loss(self, portfolio_value: float):
        if portfolio_value < self.initial_balance * (1 - self.portfolio_stop_loss):
            print(f"❌ PORTFOLIO STOP LOSS HIT: Value {portfolio_value:,.2f} below threshold.")
            self.trading_halted = True

    def take_partial_profits(self, date: str):
        for symbol, pos in list(self.positions.items()):
            price = self.get_price_from_cache(symbol, date)
            if price > pos['avg_price'] * (1 + self.profit_take_pct):
                shares_to_sell = pos['shares'] * 0.5
                if self._execute_trade(date, symbol, 'SELL', shares_to_sell, price, "Profit Take"):
                    print(f"💰 PROFIT TAKE: Sold 50% of {symbol} at {price:.2f}")

    def update_trailing_stops(self, date: str):
        for symbol, pos in self.positions.items():
            price = self.get_price_from_cache(symbol, date)
            pos['peak_price'] = max(pos.get('peak_price', price), price)
            atr = self.calculate_atr(self.historical_data_cache[symbol]['data'])
            atr_mult = self.atr_multipliers['crypto' if '-USD' in symbol else 'stocks']
            stop_price = pos['peak_price'] - (atr * atr_mult)
            self.trailing_stops[symbol] = max(self.trailing_stops.get(symbol, 0), stop_price)

    def check_for_exits(self, date: str):
        for symbol, pos in list(self.positions.items()):
            price = self.get_price_from_cache(symbol, date)
            if price > 0 and price < self.trailing_stops.get(symbol, 0):
                if self._execute_trade(date, symbol, 'SELL', pos['shares'], price, "Trailing Stop"):
                    print(f"🔻 TRAILING STOP: Exited {symbol} at {price:.2f}")

    def execute_day_1_allocation_v3(self, date: str):
        """Executes a realistic Day 1 allocation based on signals and risk management."""
        print(f"\n🚀 DAY 1: Initializing portfolio based on best signals...")
        portfolio_value = self.calculate_portfolio_value(date) # This is just cash on day 1
        signals = self.detect_market_momentum_signals(date)
        
        strong_signals = {s: sig for s, sig in signals.items() if "UP" in sig}
        
        # Limit to top 10 signals to avoid over-diversification on day 1
        top_signals = dict(sorted(strong_signals.items(), key=lambda item: item[1], reverse=True)[:10])

        for symbol, signal in top_signals.items():
            price = self.get_price_from_cache(symbol, date)
            if price > 0:
                current_portfolio_value = self.calculate_portfolio_value(date)
                buy_amount = self.calculate_position_size(symbol, price, current_portfolio_value)
                if buy_amount > 0 and self.cash > buy_amount:
                    shares = buy_amount / price
                    if self._execute_trade(date, symbol, 'BUY', shares, price, f"Day1_{signal}"):
                        print(f"📈 DAY 1 BUY: {shares:.4f} shares of {symbol} for ${buy_amount:,.2f}")

    def detect_market_momentum_signals(self, date: str) -> Dict[str, str]:
        # Simplified signal logic for V3 example
        signals = {}
        for symbol in self.all_symbols:
            data = self.historical_data_cache.get(symbol, {}).get('data')
            if data is None or len(data) < 50: continue
            
            data = data.copy()
            data['EMA50'] = data['Close'].ewm(span=50).mean()
            data['EMA200'] = data['Close'].ewm(span=200).mean()

            if data['EMA50'].iloc[-1] > data['EMA200'].iloc[-1]:
                signals[symbol] = "STRONG_UP"
        return signals

    def simulate_pure_trading_day(self, date: str, is_first_day: bool = False):
        if self.trading_halted: return
        
        if is_first_day:
            self.execute_day_1_allocation_v3(date)
        else:
            portfolio_value = self.calculate_portfolio_value(date)
            self.check_portfolio_stop_loss(portfolio_value)
            if self.trading_halted: return
            
            print(f"\n📅 {date} | Portfolio: ${portfolio_value:,.2f} | Cash: ${self.cash:,.2f}")

            self.update_trailing_stops(date)
            self.check_for_exits(date)
            self.take_partial_profits(date)

            signals = self.detect_market_momentum_signals(date)
            for symbol, signal in signals.items():
                if "UP" in signal and symbol not in self.positions:
                    price = self.get_price_from_cache(symbol, date)
                    if price > 0:
                        buy_amount = self.calculate_position_size(symbol, price, portfolio_value)
                        if buy_amount > 0:
                            shares = buy_amount / price
                            if self._execute_trade(date, symbol, 'BUY', shares, price, signal):
                                 print(f"📈 BUY: {shares:.4f} shares of {symbol} for ${buy_amount:,.2f}")
        
        portfolio_value = self.calculate_portfolio_value(date)
        self.daily_values.append({'date': date, 'portfolio_value': portfolio_value})

    def calculate_portfolio_value(self, date: str) -> float:
        value = self.cash
        for symbol, pos in self.positions.items():
            value += pos['shares'] * self.get_price_from_cache(symbol, date)
        return value

    def run_pure_5k_backtest(self, days: int = 30):
        print(f"\n🎯 RUNNING PURE $5K V3 BACKTEST ({days} DAYS)")
        self.cache_historical_data(days + 50)
        
        end_date = datetime.now(self.utc)
        start_date = end_date - timedelta(days=days)
        trading_days = pd.bdate_range(start=start_date.date(), end=end_date.date(), freq='D')
        
        for i, trading_day in enumerate(trading_days):
            date_str = trading_day.strftime('%Y-%m-%d')
            self.simulate_pure_trading_day(date_str, is_first_day=(i == 0))
            if self.trading_halted:
                print("\n--- BACKTEST HALTED DUE TO PORTFOLIO STOP LOSS ---")
                break
        
        # Final results
        if not self.daily_values:
            print("No trading days were simulated.")
            return {'initial_balance': self.initial_balance, 'final_portfolio_value': self.initial_balance, 'return_percentage': 0, 'total_trades': 0, 'target_met': False}

        final_value = self.daily_values[-1]['portfolio_value']
        total_return = final_value - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        
        results = {
            'initial_balance': self.initial_balance,
            'final_portfolio_value': final_value,
            'return_percentage': return_pct,
            'total_trades': len(self.trades),
            'target_met': return_pct >= 10.0
        }
        
        print("\n🎯 V3 BACKTEST RESULTS")
        print("="*30)
        print(f"📈 Final Portfolio Value: ${final_value:,.2f}")
        print(f"📊 Return: {return_pct:.2f}%")
        print(f"🔄 Total Trades: {len(self.trades)}")
        return results

def main():
    system = Pure5KV3TradingSystem(initial_balance=5000.0)
    system.run_pure_5k_backtest(days=30)

if __name__ == "__main__":
    main() 