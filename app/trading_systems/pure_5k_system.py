#!/usr/bin/env python3
"""
🚀 PURE $5K ULTRA-AGGRESSIVE TRADING SYSTEM - LIVE TESTING VERSION
==================================================================
Strategy: Pure trading performance from $5,000 initial capital only
Focus: 23-symbol diversified portfolio with ultra-aggressive momentum trading
Features: Paper trading, live monitoring, risk management, daily alerts
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
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import schedule
import threading

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

class Pure5KLiveTradingSystem:
    def __init__(self, initial_balance: float = 5000.0, paper_trading: bool = True):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.positions = {}  # {symbol: {'shares': float, 'avg_price': float}}
        self.trades = []
        self.daily_values = []
        self.historical_data_cache = {}
        self.logger = logging.getLogger(__name__)
        
        # LIVE TRADING FEATURES
        self.paper_trading = paper_trading
        self.live_mode = not paper_trading
        self.monitoring_active = False
        self.daily_reports = []
        self.alert_thresholds = {
            'max_daily_loss': -5.0,  # Stop if daily loss > 5%
            'max_total_loss': -10.0,  # Stop if total loss > 10%
            'max_position_size': 0.25,  # No single position > 25%
            'max_trades_per_day': 5,   # Max 5 trades per day
            'min_cash_reserve': 500.0  # Keep $500 minimum cash
        }
        
        # Enhanced risk management
        self.last_trade_date = {}
        self.trailing_stops = {}
        self.cooldown_periods = 3
        self.daily_trade_count = 0
        self.daily_start_value = self.initial_balance
        self.emergency_stop = False
        
        # EXPANDED CRYPTO UNIVERSE - 7 cryptos
        self.crypto_symbols = [
            'BTC-USD', 'XRP-USD', 'ETH-USD',
            'SOL-USD', 'TRX-USD', 'ADA-USD', 'XLM-USD'
        ]
        
        # EXPANDED STOCK UNIVERSE
        self.energy_stocks = [
            'XLE', 'KOLD', 'UNG', 'USO', 'NEE', 'DUK', 'LNG', 'XOM', 'PLUG'
        ]
        
        self.tech_stocks = [
            'QQQ', 'NVDA', 'MSFT', 'GOOGL', 'TSLA', 'AMD'
        ]
        
        self.etf_symbols = [
            'SPY', 'VTI', 'GLD'
        ]
        
        self.all_symbols = self.crypto_symbols + self.energy_stocks + self.tech_stocks + self.etf_symbols
        
        # Portfolio allocation
        self.crypto_allocation = 0.70
        self.energy_allocation = 0.00
        self.tech_allocation = 0.30
        self.etf_allocation = 0.00
        
        # Market timezone handling
        self.market_tz = pytz.timezone('America/New_York')
        self.utc = pytz.UTC
        
        # Create necessary directories
        os.makedirs('app/logs', exist_ok=True)
        os.makedirs('app/data/live', exist_ok=True)
        
        print(f"🚀 PURE $5K LIVE TRADING SYSTEM - {'PAPER' if paper_trading else 'LIVE'} MODE")
        print(f"💰 Initial Capital: ${self.initial_balance:,.2f}")
        print(f"🎯 Risk Management: Active with multiple safeguards")
        print(f"📊 Daily Monitoring: Enabled")
        print(f"⚠️  Paper Trading: {'YES' if paper_trading else 'NO - REAL MONEY!'}")

    def standardize_datetime(self, dt) -> pd.Timestamp:
        """Standardize datetime to UTC for consistent comparison"""
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        
        if dt.tz is None:
            # Assume naive datetimes are in market timezone
            dt = self.market_tz.localize(dt)
        
        # Convert to UTC
        return dt.astimezone(self.utc)

    def cache_historical_data(self, days: int = 90) -> None:
        """Download and cache historical data with proper timezone handling - FIXED VERSION"""
        print(f"\n📥 Caching {days} days of historical data for offline use...")
        
        cache_file = f"app/data/cache/pure_5k_cache_{days}days.pkl"
        
        # Create cache directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # Try to load existing cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.historical_data_cache = pickle.load(f)
                print(f"✅ Loaded cached data for {len(self.historical_data_cache)} symbols")
                return
            except Exception as e:
                print(f"⚠️  Cache file corrupted ({e}), rebuilding...")
        
        # Build new cache with proper timezone handling
        end_date = datetime.now(self.utc)
        start_date = end_date - timedelta(days=days + 10)  # Extra buffer
        
        for symbol in self.all_symbols:
            try:
                print(f"📊 Caching {symbol}...")
                ticker = yf.Ticker(symbol)
                
                # Download with multiple fallback intervals
                hist_data = None
                intervals = ['1h', '2h', '1d']
                
                for interval in intervals:
                    try:
                        hist_data = ticker.history(
                            start=start_date.date(), 
                            end=end_date.date(), 
                            interval=interval,
                            auto_adjust=True,
                            prepost=True
                        )
                        if not hist_data.empty:
                            break
                    except Exception as interval_error:
                        self.logger.debug(f"Interval {interval} failed for {symbol}: {interval_error}")
                        continue
                
                if hist_data is not None and not hist_data.empty:
                    # FIXED: Standardize timezone properly
                    try:
                        if hist_data.index.tz is None:
                            if 'USD' in symbol:  # Crypto symbols
                                hist_data.index = hist_data.index.tz_localize(self.utc)
                            else:  # Stock symbols
                                hist_data.index = hist_data.index.tz_localize(self.market_tz).tz_convert(self.utc)
                        else:
                            hist_data.index = hist_data.index.tz_convert(self.utc)
                        
                        self.historical_data_cache[symbol] = {
                            'data': hist_data,
                            'last_updated': datetime.now(self.utc),
                            'symbol': symbol,
                            'interval': interval
                        }
                        print(f"✅ Cached {len(hist_data)} records for {symbol} ({interval})")
                    except Exception as tz_error:
                        self.logger.warning(f"Timezone handling failed for {symbol}: {tz_error}")
                        continue
                else:
                    print(f"❌ No data available for {symbol}")
                    
            except Exception as e:
                print(f"❌ Failed to cache {symbol}: {e}")
                continue
        
        # Save cache to file
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.historical_data_cache, f)
            print(f"💾 Saved cache to {cache_file}")
        except Exception as e:
            print(f"⚠️  Failed to save cache: {e}")

    def get_price_from_cache(self, symbol: str, date: str = None) -> float:
        """Get price from cached historical data with FIXED timezone handling"""
        if symbol not in self.historical_data_cache:
            return self.get_current_price_online(symbol, date)
        
        try:
            cache_entry = self.historical_data_cache[symbol]
            hist_data = cache_entry['data']
            
            if date:
                # FIXED: Standardize target date to UTC and handle comparison properly
                target_date = self.standardize_datetime(date)
                
                # Use proper pandas datetime comparison
                if len(hist_data) > 0:
                    # Find the closest date using proper pandas indexing
                    try:
                        # Get the closest index using get_indexer with method='nearest'
                        closest_idx = hist_data.index.get_indexer([target_date], method='nearest')[0]
                        
                        # Ensure valid index
                        if 0 <= closest_idx < len(hist_data):
                            # Check if within reasonable time window (2 days)
                            time_diff = abs((hist_data.index[closest_idx] - target_date).total_seconds())
                            if time_diff <= 2 * 24 * 3600:  # 2 days in seconds
                                return float(hist_data.iloc[closest_idx]['Close'])
                    except Exception as idx_error:
                        self.logger.debug(f"Index lookup failed for {symbol}: {idx_error}")
                        pass
            
            # Return most recent price
            if len(hist_data) > 0:
                return float(hist_data['Close'].iloc[-1])
            
        except Exception as e:
            self.logger.warning(f"Cache lookup failed for {symbol}: {e}")
        
        # Fallback to online
        return self.get_current_price_online(symbol, date)

    def get_current_price_online(self, symbol: str, date: str = None) -> float:
        """Fallback to online price fetching with multiple data sources"""
        # Primary: yfinance
        price = self._try_yfinance(symbol, date)
        if price > 0:
            return price
        
        # Fallback for crypto: try different suffixes
        if 'USD' in symbol and price <= 0:
            alt_symbol = symbol.replace('-USD', '-USDT')
            price = self._try_yfinance(alt_symbol, date)
            if price > 0:
                return price
        
        # Last resort: use cached data if available
        if symbol in self.historical_data_cache:
            try:
                cache_data = self.historical_data_cache[symbol]['data']
                if len(cache_data) > 0:
                    return float(cache_data['Close'].iloc[-1])
            except Exception as cache_fallback_error:
                self.logger.debug(f"Cache fallback failed for {symbol}: {cache_fallback_error}")
        
        # Only log error for non-problematic symbols
        if symbol not in ['XEG']:  # Add other known problematic symbols here
            self.logger.error(f"Could not get price for {symbol}")
        return 0.0

    def _try_yfinance(self, symbol: str, date: str = None) -> float:
        """Try to get price from yfinance with improved error handling"""
        try:
            ticker = yf.Ticker(symbol)
            
            if date:
                target_date = pd.to_datetime(date).date()
                end_date = target_date + timedelta(days=1)
                hist = ticker.history(start=target_date, end=end_date, interval='1d')
                if not hist.empty:
                    return float(hist['Close'].iloc[0])
            
            # Try recent data with fallbacks
            for period in ['1d', '5d', '1mo']:
                try:
                    hist = ticker.history(period=period)
                    if not hist.empty:
                        return float(hist['Close'].iloc[-1])
                except Exception as period_error:
                    # Only log debug messages for period failures to reduce noise
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(f"Period {period} failed for {symbol}: {period_error}")
                    continue
                    
        except Exception as e:
            # Only log if it's not a known delisted symbol
            if symbol not in ['XEG']:  # Add other known problematic symbols here
                self.logger.debug(f"yfinance failed for {symbol}: {e}")
        
        return 0.0

    def detect_market_momentum_signals(self, date: str) -> Dict[str, str]:
        """ENHANCED signal detection with volume, EMA trends, and better filtering"""
        signals = {}
        
        for symbol in self.all_symbols:
            try:
                if symbol in self.historical_data_cache:
                    cache_entry = self.historical_data_cache[symbol]
                    data = cache_entry['data']
                    
                    # Get data up to the target date
                    target_date = self.standardize_datetime(date)
                    
                    if len(data) > 0:
                        # Filter data up to target date
                        try:
                            recent_data = data[data.index <= target_date]
                        except Exception:
                            recent_data = data
                        
                        if len(recent_data) > 200:  # Need more data for EMAs
                            
                            # 1. CALCULATE EMAs FOR TREND BIAS
                            try:
                                # Create a proper copy to avoid pandas warnings
                                recent_data = recent_data.copy()
                                recent_data['EMA_50'] = recent_data['Close'].ewm(span=50).mean()
                                recent_data['EMA_200'] = recent_data['Close'].ewm(span=200).mean()
                                
                                # Current trend bias
                                current_ema_50 = recent_data['EMA_50'].iloc[-1]
                                current_ema_200 = recent_data['EMA_200'].iloc[-1]
                                trend_bias = "BULLISH" if current_ema_50 > current_ema_200 else "BEARISH"
                            except:
                                trend_bias = "NEUTRAL"
                            
                            # 2. CALCULATE VOLUME METRICS
                            try:
                                # Average volume over last 20 periods
                                avg_volume = recent_data['Volume'].tail(20).mean()
                                current_volume = recent_data['Volume'].iloc[-1]
                                volume_confirmed = current_volume > (avg_volume * 1.5)
                            except:
                                volume_confirmed = True  # Assume confirmed if no volume data
                            
                            # 3. MOMENTUM CALCULATIONS
                            try:
                                # Short-term momentum (6 periods)
                                recent_6 = recent_data['Close'].tail(6)
                                if len(recent_6) >= 2:
                                    momentum_6 = (recent_6.iloc[-1] - recent_6.iloc[0]) / recent_6.iloc[0]
                                else:
                                    momentum_6 = 0
                                
                                # Medium-term momentum (24 periods)
                                recent_24 = recent_data['Close'].tail(24)
                                if len(recent_24) >= 2:
                                    momentum_24 = (recent_24.iloc[-1] - recent_24.iloc[0]) / recent_24.iloc[0]
                                else:
                                    momentum_24 = 0
                                
                                # Long-term momentum (50 periods)
                                recent_50 = recent_data['Close'].tail(50)
                                if len(recent_50) >= 2:
                                    momentum_50 = (recent_50.iloc[-1] - recent_50.iloc[0]) / recent_50.iloc[0]
                                else:
                                    momentum_50 = 0
                            except:
                                momentum_6 = momentum_24 = momentum_50 = 0
                            
                            # 4. ENHANCED SIGNAL CLASSIFICATION
                            # Only trade in direction of trend bias for cleaner signals
                            if trend_bias == "BULLISH" and volume_confirmed:
                                if momentum_6 > 0.08:  # >8% in 6 periods + volume
                                    signals[symbol] = "EXPLOSIVE_UP"
                                elif momentum_24 > 0.12:  # >12% in 24 periods + volume
                                    signals[symbol] = "STRONG_UP"
                                elif momentum_50 > 0.15:  # >15% in 50 periods + volume
                                    signals[symbol] = "TREND_UP"
                                elif momentum_6 < -0.08 or momentum_24 < -0.12:
                                    signals[symbol] = "REVERSAL_DOWN"
                                else:
                                    signals[symbol] = "NEUTRAL"
                            
                            elif trend_bias == "BEARISH":
                                # In bearish trends, only take defensive actions
                                if momentum_6 < -0.05 or momentum_24 < -0.08:
                                    signals[symbol] = "STRONG_DOWN"
                                else:
                                    signals[symbol] = "BEARISH_HOLD"
                            
                            else:
                                signals[symbol] = "NEUTRAL"
                        else:
                            signals[symbol] = "NEUTRAL"
                    else:
                        signals[symbol] = "NEUTRAL"
                else:
                    signals[symbol] = "NEUTRAL"
                    
            except Exception as e:
                self.logger.warning(f"Enhanced signal detection failed for {symbol}: {e}")
                signals[symbol] = "NEUTRAL"
        
        return signals

    def update_trailing_stops(self, date: str) -> None:
        """Update trailing stop levels for all positions"""
        for symbol, position in self.positions.items():
            if position['shares'] > 0:
                current_price = self.get_price_from_cache(symbol, date)
                if current_price > 0:
                    # Initialize trailing stop at 85% of entry price
                    if symbol not in self.trailing_stops:
                        self.trailing_stops[symbol] = position['avg_price'] * 0.85
                    
                    # Update trailing stop to 85% of highest price seen
                    potential_new_stop = current_price * 0.85
                    if potential_new_stop > self.trailing_stops[symbol]:
                        self.trailing_stops[symbol] = potential_new_stop

    def check_trailing_stop_exit(self, symbol: str, current_price: float) -> bool:
        """Check if position should be closed due to trailing stop"""
        if symbol in self.trailing_stops and symbol in self.positions:
            if self.positions[symbol]['shares'] > 0:
                return current_price <= self.trailing_stops[symbol]
        return False

    def is_in_cooldown(self, symbol: str, current_date: str) -> bool:
        """Check if symbol is in trading cooldown period"""
        if symbol not in self.last_trade_date:
            return False
        
        try:
            last_trade = pd.to_datetime(self.last_trade_date[symbol])
            current = pd.to_datetime(current_date)
            days_since_trade = (current - last_trade).days
            return days_since_trade < self.cooldown_periods
        except:
            return False

    def calculate_position_return(self, symbol: str, current_price: float) -> float:
        """Calculate current return % for a position"""
        if symbol not in self.positions or self.positions[symbol]['shares'] <= 0:
            return 0.0
        
        avg_price = self.positions[symbol]['avg_price']
        return ((current_price - avg_price) / avg_price) * 100

    def execute_day_1_intelligent_allocation(self, date: str) -> None:
        """Execute intelligent Day 1 allocation across expanded universe"""
        print(f"\n🚀 DAY 1 PURE $5K ALLOCATION - {date}")
        print("=" * 60)
        
        # Calculate allocation amounts
        crypto_budget = self.cash * self.crypto_allocation
        energy_budget = self.cash * self.energy_allocation  
        tech_budget = self.cash * self.tech_allocation
        etf_budget = self.cash * self.etf_allocation
        
        crypto_per_symbol = crypto_budget / len(self.crypto_symbols)
        energy_per_symbol = energy_budget / len(self.energy_stocks)
        tech_per_symbol = tech_budget / len(self.tech_stocks)
        etf_per_symbol = etf_budget / len(self.etf_symbols)
        
        total_invested = 0.0
        
        # Invest in all cryptos
        print(f"\n🪙 CRYPTO INVESTMENTS ({self.crypto_allocation:.0%}):")
        for symbol in self.crypto_symbols:
            price = self.get_price_from_cache(symbol, date)
            if price > 0:
                investment_amount = crypto_per_symbol
                shares = investment_amount / price
                
                self.positions[symbol] = {
                    'shares': shares,
                    'avg_price': price,
                    'category': 'crypto'
                }
                
                self._record_trade(date, symbol, 'BUY', shares, price, investment_amount, 'Day1_Crypto_Allocation')
                total_invested += investment_amount
                
                print(f"  🪙 {symbol}: {shares:.8f} shares @ ${price:.4f} = ${investment_amount:.2f}")
        
        # Invest in energy stocks
        print(f"\n⚡ ENERGY INVESTMENTS ({self.energy_allocation:.0%}):")
        for symbol in self.energy_stocks:
            price = self.get_price_from_cache(symbol, date)
            if price > 0:
                investment_amount = energy_per_symbol
                shares = investment_amount / price
                
                self.positions[symbol] = {
                    'shares': shares,
                    'avg_price': price,
                    'category': 'energy'
                }
                
                self._record_trade(date, symbol, 'BUY', shares, price, investment_amount, 'Day1_Energy_Allocation')
                total_invested += investment_amount
                
                print(f"  ⚡ {symbol}: {shares:.4f} shares @ ${price:.2f} = ${investment_amount:.2f}")
        
        # Invest in tech stocks
        print(f"\n💻 TECH INVESTMENTS ({self.tech_allocation:.0%}):")
        for symbol in self.tech_stocks:
            price = self.get_price_from_cache(symbol, date)
            if price > 0:
                investment_amount = tech_per_symbol
                shares = investment_amount / price
                
                self.positions[symbol] = {
                    'shares': shares,
                    'avg_price': price,
                    'category': 'tech'
                }
                
                self._record_trade(date, symbol, 'BUY', shares, price, investment_amount, 'Day1_Tech_Allocation')
                total_invested += investment_amount
                
                print(f"  💻 {symbol}: {shares:.4f} shares @ ${price:.2f} = ${investment_amount:.2f}")
        
        # Invest in ETFs
        print(f"\n📈 ETF INVESTMENTS ({self.etf_allocation:.0%}):")
        for symbol in self.etf_symbols:
            price = self.get_price_from_cache(symbol, date)
            if price > 0:
                investment_amount = etf_per_symbol
                shares = investment_amount / price
                
                self.positions[symbol] = {
                    'shares': shares,
                    'avg_price': price,
                    'category': 'etf'
                }
                
                self._record_trade(date, symbol, 'BUY', shares, price, investment_amount, 'Day1_ETF_Allocation')
                total_invested += investment_amount
                
                print(f"  📈 {symbol}: {shares:.4f} shares @ ${price:.2f} = ${investment_amount:.2f}")
        
        self.cash -= total_invested
        print(f"\n💸 Total Day 1 Investment: ${total_invested:.2f}")
        print(f"💵 Remaining Cash: ${self.cash:.2f}")

    def _record_trade(self, date: str, symbol: str, action: str, shares: float, 
                     price: float, amount: float, strategy: str, reason: str = ""):
        """Helper to record trades and update tracking"""
        trade = {
            'date': date,
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': price,
            'amount': amount,
            'strategy': strategy,
            'reason': reason,
            'category': self.positions.get(symbol, {}).get('category', 'unknown')
        }
        self.trades.append(trade)
        
        # Update last trade date for cooldown tracking
        self.last_trade_date[symbol] = date

    def simulate_pure_trading_day(self, date: str, is_first_day: bool = False) -> None:
        """ENHANCED trading simulation with trailing stops, cooldowns, and better exits"""
        print(f"\n📅 {date}")
        print("-" * 50)
        
        if is_first_day:
            self.execute_day_1_intelligent_allocation(date)
        else:
            # Update trailing stops for all positions
            self.update_trailing_stops(date)
            
            # Check for trailing stop exits FIRST
            trailing_stop_exits = 0
            for symbol, position in list(self.positions.items()):
                if position['shares'] > 0:
                    current_price = self.get_price_from_cache(symbol, date)
                    if current_price > 0:
                        
                        # Check trailing stop
                        if self.check_trailing_stop_exit(symbol, current_price):
                            shares_to_sell = position['shares']
                            sell_amount = shares_to_sell * current_price
                            position_return = self.calculate_position_return(symbol, current_price)
                            
                            # Execute trailing stop exit
                            self.positions[symbol]['shares'] = 0
                            self.cash += sell_amount
                            
                            self._record_trade(date, symbol, 'SELL', shares_to_sell, current_price,
                                             sell_amount, 'Trailing_Stop', f"Trailing stop hit ({position_return:+.1f}%)")
                            trailing_stop_exits += 1
                            
                            print(f"  🔻 TRAILING STOP: {shares_to_sell:.6f} {symbol} @ ${current_price:.4f} ({position_return:+.1f}%)")
            
            # Detect market signals with enhanced analysis
            signals = self.detect_market_momentum_signals(date)
            
            # Execute trades based on enhanced signals
            trades_executed = 0
            
            for symbol, signal in signals.items():
                current_price = self.get_price_from_cache(symbol, date)
                if current_price <= 0:
                    continue
                
                # Skip if in cooldown period
                if self.is_in_cooldown(symbol, date):
                    continue
                
                # EXPLOSIVE UP signals - buy aggressively (with volume confirmation)
                if signal == "EXPLOSIVE_UP" and self.cash > 250:
                    category = self.positions.get(symbol, {}).get('category', 'unknown')
                    buy_amount = min(400 if category == 'crypto' else 300, self.cash * 0.4)
                    shares = buy_amount / current_price
                    
                    self._add_to_position(symbol, shares, current_price, category)
                    self.cash -= buy_amount
                    self.last_trade_date[symbol] = date
                    
                    self._record_trade(date, symbol, 'BUY', shares, current_price, 
                                     buy_amount, 'Explosive_Momentum', f"Volume-confirmed explosive surge")
                    trades_executed += 1
                    
                    print(f"  💥 EXPLOSIVE BUY: {shares:.6f} {symbol} @ ${current_price:.4f} (Volume confirmed)")
                
                # STRONG UP signals - buy moderately (trend + volume confirmed)
                elif signal == "STRONG_UP" and self.cash > 200:
                    category = self.positions.get(symbol, {}).get('category', 'unknown')
                    buy_amount = min(250 if category == 'crypto' else 200, self.cash * 0.3)
                    shares = buy_amount / current_price
                    
                    self._add_to_position(symbol, shares, current_price, category)
                    self.cash -= buy_amount
                    self.last_trade_date[symbol] = date
                    
                    self._record_trade(date, symbol, 'BUY', shares, current_price,
                                     buy_amount, 'Strong_Momentum', f"Trend + volume confirmed")
                    trades_executed += 1
                    
                    print(f"  🚀 STRONG BUY: {shares:.6f} {symbol} @ ${current_price:.4f} (Trend aligned)")
                
                # TREND UP signals - smaller position (long-term trend)
                elif signal == "TREND_UP" and self.cash > 150:
                    category = self.positions.get(symbol, {}).get('category', 'unknown')
                    buy_amount = min(150, self.cash * 0.2)
                    shares = buy_amount / current_price
                    
                    self._add_to_position(symbol, shares, current_price, category)
                    self.cash -= buy_amount
                    self.last_trade_date[symbol] = date
                    
                    self._record_trade(date, symbol, 'BUY', shares, current_price,
                                     buy_amount, 'Trend_Following', f"Long-term trend confirmed")
                    trades_executed += 1
                    
                    print(f"  📈 TREND BUY: {shares:.6f} {symbol} @ ${current_price:.4f} (Trend following)")
                
                # REVERSAL DOWN signals - quick exit before trailing stop
                elif signal == "REVERSAL_DOWN" and symbol in self.positions:
                    if self.positions[symbol]['shares'] > 0:
                        position_return = self.calculate_position_return(symbol, current_price)
                        
                        # Only exit if we're still profitable or loss is manageable
                        if position_return > -5:  # Don't panic sell big losses
                            shares_to_sell = self.positions[symbol]['shares'] * 0.6  # Sell 60%
                            sell_amount = shares_to_sell * current_price
                            
                            self.positions[symbol]['shares'] -= shares_to_sell
                            self.cash += sell_amount
                            self.last_trade_date[symbol] = date
                            
                            self._record_trade(date, symbol, 'SELL', shares_to_sell, current_price,
                                             sell_amount, 'Reversal_Exit', f"Momentum reversal detected ({position_return:+.1f}%)")
                            trades_executed += 1
                            
                            print(f"  ⚡ REVERSAL EXIT: {shares_to_sell:.6f} {symbol} @ ${current_price:.4f} ({position_return:+.1f}%)")
                
                # STRONG DOWN signals - defensive selling
                elif signal == "STRONG_DOWN" and symbol in self.positions:
                    if self.positions[symbol]['shares'] > 0:
                        shares_to_sell = self.positions[symbol]['shares'] * 0.4  # Sell 40%
                        sell_amount = shares_to_sell * current_price
                        position_return = self.calculate_position_return(symbol, current_price)
                        
                        self.positions[symbol]['shares'] -= shares_to_sell
                        self.cash += sell_amount
                        self.last_trade_date[symbol] = date
                        
                        self._record_trade(date, symbol, 'SELL', shares_to_sell, current_price,
                                         sell_amount, 'Defensive_Exit', f"Strong downtrend ({position_return:+.1f}%)")
                        trades_executed += 1
                        
                        print(f"  🛡️  DEFENSIVE SELL: {shares_to_sell:.6f} {symbol} @ ${current_price:.4f} ({position_return:+.1f}%)")
            
            if trades_executed == 0 and trailing_stop_exits == 0:
                print("  ⏸️  No trades executed - market in neutral zone or cooldown periods active")
            elif trailing_stop_exits > 0:
                print(f"  📊 Trailing stops executed: {trailing_stop_exits}")
        
        # Calculate and store portfolio metrics
        portfolio_value = self.calculate_portfolio_value(date)
        return_pct = ((portfolio_value - self.initial_balance) / self.initial_balance) * 100
        
        self.daily_values.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'return_pct': return_pct,
            'trades_count': len([t for t in self.trades if t['date'] == date])
        })
        
        print(f"  📊 Portfolio: ${portfolio_value:,.2f} | Cash: ${self.cash:.2f} | Return: {return_pct:+.2f}%")

    def _add_to_position(self, symbol: str, shares: float, price: float, category: str) -> None:
        """Helper to add shares to existing position or create new one"""
        if symbol in self.positions:
            # Calculate weighted average price
            total_shares = self.positions[symbol]['shares'] + shares
            weighted_avg = ((self.positions[symbol]['shares'] * self.positions[symbol]['avg_price']) + 
                           (shares * price)) / total_shares
            self.positions[symbol]['shares'] = total_shares
            self.positions[symbol]['avg_price'] = weighted_avg
        else:
            self.positions[symbol] = {
                'shares': shares,
                'avg_price': price,
                'category': category
            }

    def calculate_portfolio_value(self, date: str) -> float:
        """Calculate total portfolio value including cash"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            current_price = self.get_price_from_cache(symbol, date)
            if current_price > 0:
                total_value += position['shares'] * current_price
        
        return total_value

    def get_live_price(self, symbol: str) -> float:
        """Get real-time price for live trading"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Try multiple methods for live price
            methods = [
                lambda: ticker.history(period='1d', interval='1m').iloc[-1]['Close'],
                lambda: ticker.history(period='1d').iloc[-1]['Close'],
                lambda: ticker.info.get('regularMarketPrice', 0),
                lambda: ticker.info.get('previousClose', 0)
            ]
            
            for method in methods:
                try:
                    price = float(method())
                    if price > 0:
                        return price
                except Exception:
                    continue
            
            self.logger.warning(f"Could not get live price for {symbol}")
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Live price fetch failed for {symbol}: {e}")
            return 0.0

    def check_risk_management_rules(self, date: str) -> bool:
        """Check all risk management rules - return False if trading should stop"""
        current_value = self.calculate_portfolio_value_live(date)
        daily_return = ((current_value - self.daily_start_value) / self.daily_start_value) * 100
        total_return = ((current_value - self.initial_balance) / self.initial_balance) * 100
        
        # Check daily loss limit
        if daily_return <= self.alert_thresholds['max_daily_loss']:
            self.logger.warning(f"Daily loss limit hit: {daily_return:.2f}%")
            self.send_alert(f"DAILY LOSS ALERT: {daily_return:.2f}% loss today")
            return False
        
        # Check total loss limit  
        if total_return <= self.alert_thresholds['max_total_loss']:
            self.logger.warning(f"Total loss limit hit: {total_return:.2f}%")
            self.send_alert(f"TOTAL LOSS ALERT: {total_return:.2f}% total loss")
            self.emergency_stop = True
            return False
        
        # Check daily trade limit
        today_trades = len([t for t in self.trades if t['date'] == date])
        if today_trades >= self.alert_thresholds['max_trades_per_day']:
            self.logger.warning(f"Daily trade limit hit: {today_trades} trades")
            return False
        
        # Check minimum cash reserve
        if self.cash < self.alert_thresholds['min_cash_reserve']:
            self.logger.warning(f"Low cash reserve: ${self.cash:.2f}")
            return False
        
        # Check position concentration
        for symbol, position in self.positions.items():
            if position['shares'] > 0:
                position_value = position['shares'] * self.get_live_price(symbol)
                position_pct = position_value / current_value
                if position_pct > self.alert_thresholds['max_position_size']:
                    self.logger.warning(f"Position too large: {symbol} = {position_pct:.1%}")
                    return False
        
        return True

    def send_alert(self, message: str) -> None:
        """Send alert notification (placeholder for email/SMS)"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert_msg = f"[{timestamp}] PURE $5K ALERT: {message}"
        
        self.logger.warning(alert_msg)
        
        # Save alert to file
        with open('app/logs/alerts.log', 'a') as f:
            f.write(f"{alert_msg}\n")
        
        print(f"🚨 ALERT: {message}")

    def calculate_portfolio_value_live(self, date: str = None) -> float:
        """Calculate live portfolio value"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if position['shares'] > 0:
                current_price = self.get_live_price(symbol)
                if current_price > 0:
                    total_value += position['shares'] * current_price
        
        return total_value

    def execute_paper_trade(self, symbol: str, action: str, shares: float, price: float, reason: str) -> bool:
        """Execute trade in paper trading mode"""
        if not self.paper_trading:
            self.logger.error("Paper trade called in live mode!")
            return False
        
        amount = shares * price
        
        if action == 'BUY':
            if amount > self.cash:
                self.logger.warning(f"Insufficient cash for {symbol}: need ${amount:.2f}, have ${self.cash:.2f}")
                return False
            
            self.cash -= amount
            self._add_to_position(symbol, shares, price, self._get_symbol_category(symbol))
            
        elif action == 'SELL':
            if symbol not in self.positions or self.positions[symbol]['shares'] < shares:
                self.logger.warning(f"Insufficient shares to sell {symbol}")
                return False
            
            self.positions[symbol]['shares'] -= shares
            self.cash += amount
        
        # Record trade
        self._record_trade(datetime.now().strftime('%Y-%m-%d'), symbol, action, 
                          shares, price, amount, 'Live_Paper_Trading', reason)
        
        self.logger.info(f"PAPER {action}: {shares:.6f} {symbol} @ ${price:.4f} - {reason}")
        return True

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

    def run_live_monitoring_cycle(self) -> None:
        """Run one cycle of live monitoring and trading"""
        if self.emergency_stop:
            self.logger.warning("Emergency stop active - skipping trading cycle")
            return
        
        current_time = datetime.now(self.market_tz)
        date_str = current_time.strftime('%Y-%m-%d')
        
        # Check if market is open (simplified - weekdays 9:30 AM - 4:00 PM ET)
        if current_time.weekday() >= 5:  # Weekend
            self.logger.info("Market closed - weekend")
            return
        
        market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = current_time.replace(hour=16, minute=0, second=0, microsecond=0)
        
        if current_time < market_open or current_time > market_close:
            # Crypto trades 24/7, but let's focus on market hours for now
            self.logger.info("Outside market hours")
            return
        
        # Check risk management rules
        if not self.check_risk_management_rules(date_str):
            self.logger.warning("Risk management rules violated - stopping trading")
            return
        
        # Get live signals
        signals = self.detect_live_momentum_signals()
        
        # Execute trades based on signals
        trades_today = len([t for t in self.trades if t['date'] == date_str])
        
        for symbol, signal in signals.items():
            if trades_today >= self.alert_thresholds['max_trades_per_day']:
                break
            
            current_price = self.get_live_price(symbol)
            if current_price <= 0:
                continue
            
            # Skip if in cooldown
            if self.is_in_cooldown(symbol, date_str):
                continue
            
            # Execute trades based on signals (similar to backtest logic)
            if signal == "EXPLOSIVE_UP" and self.cash > 250:
                category = self._get_symbol_category(symbol)
                buy_amount = min(400 if category == 'crypto' else 300, self.cash * 0.4)
                shares = buy_amount / current_price
                
                if self.execute_paper_trade(symbol, 'BUY', shares, current_price, 
                                          f"Live explosive momentum - {signal}"):
                    trades_today += 1
            
            elif signal == "STRONG_UP" and self.cash > 200:
                category = self._get_symbol_category(symbol)
                buy_amount = min(250 if category == 'crypto' else 200, self.cash * 0.3)
                shares = buy_amount / current_price
                
                if self.execute_paper_trade(symbol, 'BUY', shares, current_price,
                                          f"Live strong momentum - {signal}"):
                    trades_today += 1
            
            # Handle sell signals
            elif signal in ["REVERSAL_DOWN", "STRONG_DOWN"] and symbol in self.positions:
                if self.positions[symbol]['shares'] > 0:
                    sell_ratio = 0.6 if signal == "REVERSAL_DOWN" else 0.4
                    shares_to_sell = self.positions[symbol]['shares'] * sell_ratio
                    
                    if self.execute_paper_trade(symbol, 'SELL', shares_to_sell, current_price,
                                              f"Live exit signal - {signal}"):
                        trades_today += 1
        
        # Update portfolio tracking
        portfolio_value = self.calculate_portfolio_value_live()
        return_pct = ((portfolio_value - self.initial_balance) / self.initial_balance) * 100
        
        self.daily_values.append({
            'timestamp': current_time.isoformat(),
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'return_pct': return_pct,
            'active_positions': len([p for p in self.positions.values() if p['shares'] > 0])
        })
        
        # Log current status
        self.logger.info(f"Portfolio: ${portfolio_value:,.2f} | Cash: ${self.cash:.2f} | Return: {return_pct:+.2f}%")

    def detect_live_momentum_signals(self) -> Dict[str, str]:
        """Detect momentum signals using live data"""
        signals = {}
        
        for symbol in self.all_symbols:
            try:
                # Get recent data for analysis
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='5d', interval='1h')
                
                if len(hist) > 50:
                    # Simple momentum analysis
                    recent_prices = hist['Close'].tail(24)  # Last 24 hours
                    if len(recent_prices) > 1:
                        momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                        
                        if momentum > 0.08:
                            signals[symbol] = "EXPLOSIVE_UP"
                        elif momentum > 0.04:
                            signals[symbol] = "STRONG_UP"
                        elif momentum < -0.05:
                            signals[symbol] = "STRONG_DOWN"
                        elif momentum < -0.03:
                            signals[symbol] = "REVERSAL_DOWN"
                        else:
                            signals[symbol] = "NEUTRAL"
                    else:
                        signals[symbol] = "NEUTRAL"
                else:
                    signals[symbol] = "NEUTRAL"
                    
            except Exception as e:
                self.logger.debug(f"Signal detection failed for {symbol}: {e}")
                signals[symbol] = "NEUTRAL"
        
        return signals

    def generate_daily_report(self) -> str:
        """Generate comprehensive daily report"""
        current_time = datetime.now()
        date_str = current_time.strftime('%Y-%m-%d')
        
        portfolio_value = self.calculate_portfolio_value_live()
        total_return = portfolio_value - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        
        # Count today's trades
        today_trades = [t for t in self.trades if t['date'] == date_str]
        
        # Active positions
        active_positions = [(symbol, pos) for symbol, pos in self.positions.items() if pos['shares'] > 0]
        
        report = f"""
🚀 PURE $5K DAILY REPORT - {date_str}
{'='*50}
💰 Portfolio Value: ${portfolio_value:,.2f}
💵 Cash Available: ${self.cash:,.2f} 
📊 Total Return: ${total_return:+,.2f} ({return_pct:+.2f}%)
🔄 Trades Today: {len(today_trades)}
📈 Active Positions: {len(active_positions)}

📊 POSITION BREAKDOWN:
"""
        
        for symbol, position in active_positions:
            current_price = self.get_live_price(symbol)
            position_value = position['shares'] * current_price
            position_return = ((current_price - position['avg_price']) / position['avg_price']) * 100
            
            report += f"   {symbol}: {position['shares']:.6f} @ ${current_price:.4f} = ${position_value:.2f} ({position_return:+.1f}%)\n"
        
        if today_trades:
            report += f"\n🔄 TODAY'S TRADES:\n"
            for trade in today_trades:
                report += f"   {trade['action']} {trade['shares']:.6f} {trade['symbol']} @ ${trade['price']:.4f} - {trade['reason']}\n"
        
        return report

    def save_daily_data(self) -> None:
        """Save daily trading data"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save positions
        positions_file = f"app/data/live/positions_{timestamp}.json"
        with open(positions_file, 'w') as f:
            json.dump(self.positions, f, indent=2)
        
        # Save trades
        trades_file = f"app/data/live/trades_{timestamp}.json"
        with open(trades_file, 'w') as f:
            json.dump(self.trades, f, indent=2, default=str)
        
        # Save daily values
        values_file = f"app/data/live/daily_values_{timestamp}.json"
        with open(values_file, 'w') as f:
            json.dump(self.daily_values, f, indent=2, default=str)

    def start_live_monitoring(self, monitoring_interval: int = 300) -> None:
        """Start live monitoring with specified interval (seconds)"""
        self.monitoring_active = True
        self.daily_start_value = self.calculate_portfolio_value_live()
        
        print(f"🚀 Starting live monitoring - {'PAPER' if self.paper_trading else 'LIVE'} trading mode")
        print(f"⏰ Monitoring interval: {monitoring_interval} seconds")
        print(f"📊 Risk management: Active")
        
        # Schedule daily report
        schedule.every().day.at("16:05").do(self.generate_and_send_daily_report)
        
        try:
            while self.monitoring_active and not self.emergency_stop:
                # Run monitoring cycle
                self.run_live_monitoring_cycle()
                
                # Run scheduled tasks
                schedule.run_pending()
                
                # Save data periodically
                if len(self.daily_values) % 12 == 0:  # Every hour if 5-minute intervals
                    self.save_daily_data()
                
                # Wait for next cycle
                time.sleep(monitoring_interval)
                
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
            self.send_alert(f"Monitoring system error: {e}")
        finally:
            self.stop_monitoring()

    def generate_and_send_daily_report(self) -> None:
        """Generate and send daily report"""
        report = self.generate_daily_report()
        print(report)
        
        # Save report
        date_str = datetime.now().strftime('%Y%m%d')
        report_file = f"app/logs/daily_report_{date_str}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Daily report saved to {report_file}")

    def stop_monitoring(self) -> None:
        """Stop live monitoring and save final data"""
        self.monitoring_active = False
        self.save_daily_data()
        
        final_report = self.generate_daily_report()
        print(f"\n🏁 FINAL REPORT:\n{final_report}")
        
        print("📊 Live monitoring stopped - all data saved")

def main():
    """Main execution function"""
    try:
        # Create pure $5K trading system
        system = Pure5KLiveTradingSystem(initial_balance=5000.0)
        
        # Run 30-day backtest
        results = system.run_pure_5k_backtest(days=30)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"app/data/results/pure_5k_results_{timestamp}.json"
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()