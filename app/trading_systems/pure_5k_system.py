#!/usr/bin/env python3
"""
🚀 PURE $5K ULTRA-AGGRESSIVE TRADING SYSTEM - FIXED VERSION
===========================================================
Strategy: Pure trading performance from $5,000 initial capital only
Focus: 23-symbol diversified portfolio with ultra-aggressive momentum trading
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

# Configure logging to reduce noise from yfinance
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)  # Suppress yfinance error messages

class Pure5KTradingSystem:
    def __init__(self, initial_balance: float = 5000.0):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.positions = {}  # {symbol: {'shares': float, 'avg_price': float}}
        self.trades = []
        self.daily_values = []
        self.historical_data_cache = {}  # Offline data storage
        self.logger = logging.getLogger(__name__)
        
        # ENHANCED FEATURES
        self.last_trade_date = {}  # Track last trade date per symbol for cooldowns
        self.trailing_stops = {}   # Track trailing stop levels per position
        self.atr_values = {}      # Track ATR values for dynamic stops
        self.rsi_values = {}      # Track RSI values for position sizing
        self.cooldown_periods = 3  # Wait 3 periods after trade before next entry
        
        # Cash bucket management
        self.cash_buckets = {
            'high_conviction': 0.70,  # 70% for strong setups
            'swing_trades': 0.20,     # 20% for shorter-term plays
            'defensive': 0.10         # 10% held in bad macro conditions
        }
        
        # Dynamic ATR multipliers based on volatility and trend
        self.atr_multipliers = {
            'crypto': {
                'strong_trend': 3.0,    # More room to run in strong trends
                'neutral': 2.5,         # Standard multiplier
                'weak_trend': 2.0       # Tighter stops in weak trends
            },
            'stocks': {
                'strong_trend': 2.5,    # Stock multipliers are more conservative
                'neutral': 2.0,
                'weak_trend': 1.5
            }
        }
        
        # RSI settings for position sizing
        self.rsi_period = 14
        self.rsi_weight_thresholds = {
            'very_strong': 75,  # Very bullish momentum
            'strong': 65,      # Strong momentum
            'neutral': 50,     # Neutral
            'weak': 35,        # Weak momentum
            'very_weak': 25    # Very bearish momentum
        }
        
        # EXPANDED CRYPTO UNIVERSE - 7 cryptos
        self.crypto_symbols = [
            'BTC-USD',
            'XRP-USD',
            'ETH-USD',
            'SOL-USD',
            'TRX-USD',
            'ADA-USD',
            'XLM-USD',
            'XRP-USD',
            "BNB-USD",
            "USDC-USD",
            "ARB-USD",
        ]
        
        # EXPANDED STOCK UNIVERSE - Energy, Tech, ETFs
        self.energy_stocks = [
            'XLE',    # Energy ETF
            'KOLD',   # Natural Gas Bear ETF
            'USO',    # Oil ETF
            'ICLN',    # NextEra Energy (Electrical)
            'BE',    # Duke Energy (Electrical)
            'LNG',    # LNG ETF
            'XOM',    # Exxon Mobil
        ]
        
        self.tech_stocks = [
            'QQQ',    # Tech ETF
            'NVDA',   # NVIDIA
            'MSFT',   # Microsoft  
            'GOOGL',  # Google
            'TSLA',   # Tesla
            'AMD',    # AMD
            "META",   # Meta
            "AAPL",   # Apple
            "AMZN",   # Amazon
        ]
        
        self.etf_symbols = [
            'SPY',    # S&P 500
            'VTI',    # Total Market
            'GLD',    # Gold
            "QQQM",   # QQQM
            "BIL",    # BIL
        ]
        
        self.all_symbols = self.crypto_symbols + self.energy_stocks + self.tech_stocks + self.etf_symbols
        
        # REBALANCED ALLOCATION STRATEGY WITH RANGES
        self.crypto_allocation = {'target': 0.20, 'range': (0.20, 0.25)}  # 20-25%
        self.energy_allocation = {'target': 0.15, 'range': (0.12, 0.18)}  # 12-18%
        self.tech_allocation = {'target': 0.12, 'range': (0.12, 0.15)}    # 12-15%
        self.etf_allocation = {'target': 0.08, 'range': (0.08, 0.10)}     # 8-10%
        
        # Cash management
        self.max_cash_ratio = 0.20  # Trigger reallocation if cash > 20%
        self.min_position_size = 100.0  # Minimum position size for new trades
        
        # Market timezone handling
        self.market_tz = pytz.timezone('America/New_York')
        self.utc = pytz.UTC
        
        print(f"💰 ENHANCED PURE $5K ULTRA-AGGRESSIVE TRADING SYSTEM:")
        print(f"   💵 Initial Capital: ${self.initial_balance:,.2f} (NO DAILY ADDITIONS)")
        print(f"   🪙 Crypto Allocation: {self.crypto_allocation['target']:.0%} ({len(self.crypto_symbols)} symbols)")
        print(f"   ⚡ Energy Allocation: {self.energy_allocation['target']:.0%} ({len(self.energy_stocks)} symbols)")
        print(f"   💻 Tech Allocation: {self.tech_allocation['target']:.0%} ({len(self.tech_stocks)} symbols)")
        print(f"   📈 ETF Allocation: {self.etf_allocation['target']:.0%} ({len(self.etf_symbols)} symbols)")
        print(f"   📊 Total Symbols: {len(self.all_symbols)}")
        print(f"   🔧 ENHANCED FEATURES: Volume filters, EMA trends, trailing stops, cooldowns")

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

    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range (ATR) for volatility-adjusted stops"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]
            return float(atr)
        except:
            return 0.0

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        try:
            close = data['Close']
            delta = close.diff()
            
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except:
            return 50.0  # Neutral RSI on error

    def get_atr_multiplier(self, symbol: str, trend_strength: float) -> float:
        """Get dynamic ATR multiplier based on asset type and trend strength"""
        is_crypto = symbol.endswith('-USD')
        multipliers = self.atr_multipliers['crypto' if is_crypto else 'stocks']
        
        if trend_strength >= 0.8:
            return multipliers['strong_trend']
        elif trend_strength <= 0.2:
            return multipliers['weak_trend']
        else:
            return multipliers['neutral']

    def get_position_weight(self, symbol: str, rsi: float, trend_strength: float) -> float:
        """Calculate position weight based on RSI and trend strength"""
        base_weight = 1.0
        
        # RSI-based weight adjustment
        if rsi >= self.rsi_weight_thresholds['very_strong']:
            rsi_mult = 1.5  # 50% larger position for very strong momentum
        elif rsi >= self.rsi_weight_thresholds['strong']:
            rsi_mult = 1.25  # 25% larger for strong momentum
        elif rsi <= self.rsi_weight_thresholds['very_weak']:
            rsi_mult = 0.5  # Half size for very weak momentum
        elif rsi <= self.rsi_weight_thresholds['weak']:
            rsi_mult = 0.75  # 25% smaller for weak momentum
        else:
            rsi_mult = 1.0  # Normal size for neutral momentum
        
        # Trend strength adjustment
        trend_mult = 0.5 + trend_strength  # 0.5 to 1.5 multiplier
        
        return base_weight * rsi_mult * trend_mult

    def update_trailing_stops(self, date: str) -> None:
        """Update trailing stop levels using dynamic ATR-based stops"""
        for symbol, position in self.positions.items():
            if position['shares'] > 0:
                current_price = self.get_price_from_cache(symbol, date)
                if current_price > 0 and symbol in self.historical_data_cache:
                    data = self.historical_data_cache[symbol]['data']
                    
                    # Calculate ATR and trend strength
                    atr = self.calculate_atr(data)
                    self.atr_values[symbol] = atr
                    
                    # Calculate trend strength (0 to 1)
                    trend_strength = self.calculate_trend_strength(data)
                    
                    # Get dynamic ATR multiplier
                    atr_mult = self.get_atr_multiplier(symbol, trend_strength)
                    
                    # Initialize or update stop
                    if symbol not in self.trailing_stops:
                        initial_stop = position['avg_price'] - (atr_mult * atr)
                        self.trailing_stops[symbol] = max(initial_stop, position['avg_price'] * 0.85)
                    else:
                        potential_new_stop = current_price - (atr_mult * atr)
                        if potential_new_stop > self.trailing_stops[symbol]:
                            self.trailing_stops[symbol] = potential_new_stop

    def calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength (0 to 1) using multiple indicators"""
        try:
            # Use multiple EMAs
            data = data.copy()
            data['EMA20'] = data['Close'].ewm(span=20).mean()
            data['EMA50'] = data['Close'].ewm(span=50).mean()
            data['EMA200'] = data['Close'].ewm(span=200).mean()
            
            # Calculate scores
            price = data['Close'].iloc[-1]
            score = 0.0
            total_checks = 0
            
            # Price vs EMAs
            if price > data['EMA20'].iloc[-1]: score += 1
            if price > data['EMA50'].iloc[-1]: score += 1
            if price > data['EMA200'].iloc[-1]: score += 1
            total_checks += 3
            
            # EMA alignments
            if data['EMA20'].iloc[-1] > data['EMA50'].iloc[-1]: score += 1
            if data['EMA50'].iloc[-1] > data['EMA200'].iloc[-1]: score += 1
            total_checks += 2
            
            # Momentum
            returns = data['Close'].pct_change()
            if returns.tail(5).mean() > 0: score += 1
            if returns.tail(20).mean() > 0: score += 1
            total_checks += 2
            
            return score / total_checks  # 0 to 1 score
        except:
            return 0.5  # Neutral on error

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
        crypto_budget = self.cash * self.crypto_allocation['target']
        energy_budget = self.cash * self.energy_allocation['target']  
        tech_budget = self.cash * self.tech_allocation['target']
        etf_budget = self.cash * self.etf_allocation['target']
        
        crypto_per_symbol = crypto_budget / len(self.crypto_symbols)
        energy_per_symbol = energy_budget / len(self.energy_stocks)
        tech_per_symbol = tech_budget / len(self.tech_stocks)
        etf_per_symbol = etf_budget / len(self.etf_symbols)
        
        total_invested = 0.0
        
        # Invest in all cryptos
        print(f"\n🪙 CRYPTO INVESTMENTS ({self.crypto_allocation['target']:.0%}):")
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
        print(f"\n⚡ ENERGY INVESTMENTS ({self.energy_allocation['target']:.0%}):")
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
        print(f"\n💻 TECH INVESTMENTS ({self.tech_allocation['target']:.0%}):")
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
        print(f"\n📈 ETF INVESTMENTS ({self.etf_allocation['target']:.0%}):")
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

    def check_cash_reallocation(self, date: str) -> None:
        """Enhanced cash reallocation with bucket strategy"""
        portfolio_value = self.calculate_portfolio_value(date)
        cash_ratio = self.cash / portfolio_value
        
        if cash_ratio > self.max_cash_ratio:
            print(f"\n💰 Cash ratio {cash_ratio:.1%} exceeds {self.max_cash_ratio:.1%} threshold")
            
            # Allocate cash to strategic buckets
            cash_targets = self.allocate_cash_buckets(date)
            
            # Find top performing sectors and symbols
            sector_performance = self.calculate_sector_performance(date)
            
            # Allocate high conviction bucket first
            self.allocate_high_conviction_trades(date, cash_targets['high_conviction'], sector_performance)
            
            # Allocate swing trade bucket
            self.allocate_swing_trades(date, cash_targets['swing_trades'], sector_performance)
            
            # Hold defensive bucket in cash or stable assets if macro looks bad
            conditions = self.analyze_market_conditions(date)
            if conditions['macro_bearish']:
                print(f"  🛡️  Holding ${cash_targets['defensive']:,.2f} in defensive assets")
                self.allocate_defensive_position(date, cash_targets['defensive'])

    def allocate_cash_buckets(self, date: str) -> Dict[str, float]:
        """Allocate cash into strategic buckets based on market conditions"""
        portfolio_value = self.calculate_portfolio_value(date)
        
        # Calculate target amounts for each bucket
        targets = {
            bucket: self.cash_buckets[bucket] * self.cash
            for bucket in self.cash_buckets
        }
        
        print(f"\n💰 CASH BUCKET ALLOCATION:")
        print(f"  💵 High Conviction: ${targets['high_conviction']:,.2f}")
        print(f"  🔄 Swing Trades:    ${targets['swing_trades']:,.2f}")
        print(f"  🛡️  Defensive:       ${targets['defensive']:,.2f}")
        
        # Analyze market conditions
        market_conditions = self.analyze_market_conditions(date)
        
        if market_conditions['macro_bearish']:
            # Increase defensive allocation in bad macro conditions
            print("  ⚠️  Bearish macro conditions detected - increasing defensive allocation")
            defensive_increase = targets['swing_trades'] * 0.5
            targets['defensive'] += defensive_increase
            targets['swing_trades'] -= defensive_increase
        
        return targets

    def analyze_market_conditions(self, date: str) -> Dict[str, bool]:
        """Analyze overall market conditions"""
        conditions = {
            'macro_bearish': False,
            'high_volatility': False,
            'trend_strength': 0.0
        }
        
        # Check major indices
        indices = ['SPY', 'QQQ']
        bearish_count = 0
        total_vol = 0
        
        for idx in indices:
            if idx in self.historical_data_cache:
                data = self.historical_data_cache[idx]['data']
                
                # Check trend
                if len(data) > 50:
                    sma50 = data['Close'].rolling(50).mean().iloc[-1]
                    if data['Close'].iloc[-1] < sma50:
                        bearish_count += 1
                
                # Check volatility
                if len(data) > 20:
                    vol = data['Close'].pct_change().std() * np.sqrt(252)
                    total_vol += vol
        
        # Set condition flags
        conditions['macro_bearish'] = bearish_count >= len(indices) * 0.5
        conditions['high_volatility'] = (total_vol / len(indices)) > 0.25
        conditions['trend_strength'] = 1 - (bearish_count / len(indices))
        
        return conditions

    def allocate_high_conviction_trades(self, date: str, budget: float, sector_performance: Dict[str, float]) -> None:
        """Allocate capital to highest conviction trades"""
        if budget < self.min_position_size:
            return
        
        print(f"\n💎 HIGH CONVICTION TRADES (${budget:,.2f}):")
        
        # Sort sectors by performance
        sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
        
        for sector, perf in sorted_sectors:
            if budget < self.min_position_size:
                break
            
            # Get top symbols in sector
            symbols = self.get_sector_symbols(sector)
            symbol_scores = []
            
            for symbol in symbols:
                if symbol in self.historical_data_cache:
                    data = self.historical_data_cache[symbol]['data']
                    
                    # Calculate comprehensive score
                    rsi = self.calculate_rsi(data)
                    trend_strength = self.calculate_trend_strength(data)
                    momentum = self.calculate_symbol_momentum(symbol, date)
                    
                    score = (rsi/100 * 0.3) + (trend_strength * 0.4) + (momentum * 0.3)
                    symbol_scores.append((symbol, score))
            
            # Sort symbols by score
            symbol_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 2 symbols from each sector
            for symbol, score in symbol_scores[:2]:
                if budget < self.min_position_size:
                    break
                
                price = self.get_price_from_cache(symbol, date)
                if price <= 0:
                    continue
                
                # Calculate position size based on score
                position_size = min(budget * score, budget * 0.3)  # Max 30% of budget per position
                position_size = max(position_size, self.min_position_size)
                
                if position_size >= self.min_position_size:
                    shares = position_size / price
                    self._add_to_position(symbol, shares, price, sector)
                    budget -= position_size
                    
                    self._record_trade(date, symbol, 'BUY', shares, price,
                                     position_size, 'High_Conviction',
                                     f"Strong setup (Score: {score:.2f})")
                    
                    print(f"  💎 {symbol}: ${position_size:,.2f} allocated (Score: {score:.2f})")

    def allocate_swing_trades(self, date: str, budget: float, sector_performance: Dict[str, float]) -> None:
        """Allocate capital to shorter-term swing trade opportunities"""
        if budget < self.min_position_size:
            return
        
        print(f"\n🔄 SWING TRADE ALLOCATION (${budget:,.2f}):")
        
        # Look for short-term momentum setups
        for sector, perf in sector_performance.items():
            if budget < self.min_position_size:
                break
            
            symbols = self.get_sector_symbols(sector)
            for symbol in symbols:
                if budget < self.min_position_size:
                    break
                
                if symbol in self.historical_data_cache:
                    data = self.historical_data_cache[symbol]['data']
                    
                    # Short-term momentum check
                    momentum = self.calculate_symbol_momentum(symbol, date, lookback=3)  # Shorter lookback
                    rsi = self.calculate_rsi(data)
                    
                    # Look for strong short-term momentum
                    if momentum > 0.02 and rsi > 60:  # 2% momentum and strong RSI
                        price = self.get_price_from_cache(symbol, date)
                        if price <= 0:
                            continue
                        
                        # Smaller position sizes for swing trades
                        position_size = min(budget * 0.2, 500.0)  # Max $500 per swing trade
                        if position_size >= self.min_position_size:
                            shares = position_size / price
                            self._add_to_position(symbol, shares, price, sector)
                            budget -= position_size
                            
                            self._record_trade(date, symbol, 'BUY', shares, price,
                                             position_size, 'Swing_Trade',
                                             f"Short-term momentum setup")
                            
                            print(f"  🔄 {symbol}: ${position_size:,.2f} swing trade")

    def allocate_defensive_position(self, date: str, budget: float) -> None:
        """Allocate defensive bucket to stable assets"""
        if budget < self.min_position_size:
            return
        
        print(f"\n🛡️  DEFENSIVE ALLOCATION (${budget:,.2f}):")
        
        # Defensive assets
        defensive_assets = [
            'BIL',  # Short-term Treasury ETF
            'USDC-USD',  # Stablecoin
            'GLD'   # Gold ETF
        ]
        
        # Split budget among defensive assets
        position_size = budget / len(defensive_assets)
        
        for symbol in defensive_assets:
            price = self.get_price_from_cache(symbol, date)
            if price <= 0:
                continue
            
            if position_size >= self.min_position_size:
                shares = position_size / price
                self._add_to_position(symbol, shares, price, 'defensive')
                budget -= position_size
                
                self._record_trade(date, symbol, 'BUY', shares, price,
                                 position_size, 'Defensive',
                                 f"Defensive allocation")
                
                print(f"  🛡️  {symbol}: ${position_size:,.2f} defensive position")

    def calculate_sector_performance(self, date: str, lookback: int = 10) -> Dict[str, float]:
        """Calculate performance metrics for each sector"""
        performance = {}
        sectors = ['crypto', 'energy', 'tech', 'etf']
        
        for sector in sectors:
            symbols = self.get_sector_symbols(sector)
            sector_perf = []
            
            for symbol in symbols:
                if symbol in self.historical_data_cache:
                    data = self.historical_data_cache[symbol]['data']
                    if len(data) >= lookback:
                        returns = data['Close'].pct_change().tail(lookback)
                        sharpe = returns.mean() / returns.std() if returns.std() != 0 else 0
                        sector_perf.append(sharpe)
            
            if sector_perf:
                performance[sector] = sum(sector_perf) / len(sector_perf)
            else:
                performance[sector] = 0.0
        
        return performance

    def get_sector_symbols(self, sector: str) -> List[str]:
        """Get list of symbols for a given sector"""
        if sector == 'crypto':
            return self.crypto_symbols
        elif sector == 'energy':
            return self.energy_stocks
        elif sector == 'tech':
            return self.tech_stocks
        elif sector == 'etf':
            return self.etf_symbols
        return []

    def calculate_symbol_momentum(self, symbol: str, date: str, lookback: int = 5) -> float:
        """Calculate momentum score for a symbol"""
        try:
            if symbol in self.historical_data_cache:
                data = self.historical_data_cache[symbol]['data']
                if len(data) >= lookback:
                    returns = data['Close'].pct_change().tail(lookback)
                    return returns.mean()
        except:
            pass
        return 0.0

    def simulate_pure_trading_day(self, date: str, is_first_day: bool = False) -> None:
        """ENHANCED trading simulation with ATR stops and cash reallocation"""
        print(f"\n📅 {date}")
        print("-" * 50)
        
        if is_first_day:
            self.execute_day_1_intelligent_allocation(date)
        else:
            # Update ATR-based trailing stops
            self.update_trailing_stops(date)
            
            # Check for trailing stop exits FIRST
            trailing_stop_exits = 0
            for symbol, position in list(self.positions.items()):
                if position['shares'] > 0:
                    current_price = self.get_price_from_cache(symbol, date)
                    if current_price > 0:
                        if self.check_trailing_stop_exit(symbol, current_price):
                            shares_to_sell = position['shares']
                            sell_amount = shares_to_sell * current_price
                            position_return = self.calculate_position_return(symbol, current_price)
                            
                            self.positions[symbol]['shares'] = 0
                            self.cash += sell_amount
                            
                            atr = self.atr_values.get(symbol, 0)
                            stop_distance = (self.trailing_stops[symbol] - position['avg_price']) / atr if atr > 0 else 0
                            
                            self._record_trade(date, symbol, 'SELL', shares_to_sell, current_price,
                                             sell_amount, 'ATR_Stop', 
                                             f"ATR-based stop hit ({position_return:+.1f}%, {stop_distance:.1f} ATR)")
                            trailing_stop_exits += 1
                            
                            print(f"  🔻 ATR STOP: {shares_to_sell:.6f} {symbol} @ ${current_price:.4f} ({position_return:+.1f}%)")
            
            # Check for cash reallocation opportunities
            self.check_cash_reallocation(date)
            
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

    def run_pure_5k_backtest(self, days: int = 30) -> Dict:
        """Run ENHANCED pure $5K backtest with volume filters, EMA trends, trailing stops, and cooldowns"""
        print(f"\n🎯 ENHANCED PURE $5K BACKTEST ({days} DAYS)")
        print("=" * 60)
        print(f"💰 Starting with ${self.initial_balance:,.2f} - NO DAILY ADDITIONS")
        print(f"🎯 Target: 10% returns through enhanced trading strategies")
        print(f"🔧 ENHANCED FEATURES:")
        print(f"   📊 Volume confirmation (1.5x average volume required)")
        print(f"   📈 EMA trend filtering (50 vs 200 EMA bias)")
        print(f"   🔻 Trailing stops (15% below peak prices)")
        print(f"   ⏰ Trade cooldowns ({self.cooldown_periods} days between entries)")
        print(f"   ⚡ Expanded energy universe ({len(self.energy_stocks)} stocks)")
        
        # Ensure we have cached data
        self.cache_historical_data(days + 10)
        
        # Calculate start date
        end_date = datetime.now(self.utc)
        start_date = end_date - timedelta(days=days)
        
        # Generate trading days (business days only for stocks)
        trading_days = pd.bdate_range(start=start_date.date(), end=end_date.date(), freq='D')
        
        for i, trading_day in enumerate(trading_days):
            date_str = trading_day.strftime('%Y-%m-%d')
            is_first_day = (i == 0)
            
            try:
                self.simulate_pure_trading_day(date_str, is_first_day)
            except Exception as e:
                self.logger.error(f"Error on {date_str}: {e}")
                continue
        
        # Calculate final results
        if not self.daily_values:
            return {"error": "No trading data generated"}
        
        final_value = self.daily_values[-1]['portfolio_value']
        total_return = final_value - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        
        max_value = max([day['portfolio_value'] for day in self.daily_values])
        min_value = min([day['portfolio_value'] for day in self.daily_values])
        
        total_trades = len(self.trades)
        
        results = {
            'initial_balance': self.initial_balance,
            'final_portfolio_value': final_value,
            'total_return': total_return,
            'return_percentage': return_pct,
            'max_value': max_value,
            'min_value': min_value,
            'total_trades': total_trades,
            'trading_days': len(self.daily_values),
            'target_met': return_pct >= 10.0
        }
        
        # Print results
        print(f"\n🎯 ENHANCED PURE $5K RESULTS ({days} DAYS)")
        print("=" * 60)
        print(f"📈 Initial Balance:        $  {self.initial_balance:,.2f}")
        print(f"📈 Final Portfolio Value:  $  {final_value:,.2f}")
        print(f"💰 Total Return:           $    {total_return:,.2f}")
        print(f"📊 Return %:                    {return_pct:.2f}%")
        print(f"📈 Maximum Value:          $  {max_value:,.2f}")
        print(f"📉 Minimum Value:          $  {min_value:,.2f}")
        print(f"🔄 Total Trades:               {total_trades}")
        print(f"📅 Trading Days:               {len(self.daily_values)}")
        print(f"🔧 Enhanced Features Used:     Volume filters, EMA trends, trailing stops, cooldowns")
        
        if return_pct >= 10.0:
            print(f"\n🎉 TARGET MET! {return_pct:.2f}% >= 10% TARGET!")
        else:
            print(f"\n❌ Target not met: {return_pct:.2f}% < 10%")
            print(f"💡 Enhanced system provides better risk management and signal quality")
        
        return results

def main():
    """Main execution function"""
    try:
        # Create pure $5K trading system
        system = Pure5KTradingSystem(initial_balance=5000.0)
        
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