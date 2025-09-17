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
from app.database import SessionLocal
from app.models.database_models import TradeRecord, DailySnapshot
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from typing import Dict, List, Tuple, Optional
import pytz
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule
import threading

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Kraken integration
from app.services.kraken import kraken_api

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
        # Backtest metric series: floats only (portfolio values)
        self.daily_values = []
        # Snapshot logs (dicts) for richer reporting
        self.backtest_daily_snapshots = []
        self.live_daily_snapshots = []
        # Per-trade equity curve for intraday risk realism
        self.intraday_equity_values = []  # floats
        self.intraday_equity_times = []   # timestamps or date strings
        # Trading frictions (stricter for robustness checks)
        self.trading_costs = {
            'fee_rate': 0.0025,      # 25 bps per side (0.25%)
            'slippage_bps': 25       # 25 bps price slippage (0.25%)
        }
        # Backtest mode flag (ensures historical-only pricing paths)
        self.backtest_mode = False
        # Risk controls
        self.risk_controls = {
            'daily_loss_halt_pct': 0.03,   # halt new buys if equity down >3% on the day
            'max_position_pct': 0.25       # cap any single position at 25% of equity
        }
        self._day_start_equity = None
        # Container for computed performance metrics
        self.performance_metrics = {
            'daily_returns': None,
            'cumulative_returns': None,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            # Day-based metrics
            'day_win_rate': 0.0,
            'day_profit_factor': 0.0,
            # Trade-based metrics
            'trade_win_rate': 0.0,
            'trade_profit_factor': 0.0,
            'recovery_factor': 0.0,
            'risk_adjusted_return': 0.0,
            'beta': 0.0,
            'alpha': 0.0,
            'correlation_matrix': None,
            # Sample sizes
            'n_daily': 0,
            'n_intraday': 0,
            'n_trades_closed': 0,
            'unstable_metrics_warning': ''
        }
        self.historical_data_cache = {}
        self.logger = logging.getLogger(__name__)
        
        # LIVE TRADING FEATURES
        self.paper_trading = paper_trading
        self.live_mode = not paper_trading
        self.monitoring_active = False
        self.daily_reports = []
        self.alert_thresholds = {
            'max_daily_loss': -5.0,
            'max_total_loss': -10.0,
            'max_position_size': 0.30,
            'max_trades_per_day': 10,
            'min_cash_reserve': 500.0
        }
        
        # Enhanced risk management
        self.last_trade_date = {}
        self.trailing_stops = {}
        self.cooldown_periods = 1  # Reduced from 2 to 1 day - allow more frequent trading
        self.daily_trade_count = 0
        self.daily_start_value = self.initial_balance
        self.emergency_stop = False
        
        # Volatility tracking for dynamic stops
        self.volatility_metrics = {}  # Store volatility data per symbol
        
        # EXPANDED CRYPTO UNIVERSE with correct Kraken symbols
        self.crypto_symbols = [
            'BTC-USD',  # Bitcoin - Working format
            'ETH-USD',  # Ethereum - Working format
            'SOL-USD',  # Solana - Working format
            'XRP-USD',  # Ripple - Working format
            'ADA-USD',  # Cardano - Working format
            'TRX-USD',  # Tron - Working format
            'XLM-USD'   # Stellar - Working format
        ]
        
        # Symbol mapping for price data (internal format -> external format)
        self.symbol_map = {
            'BTC-USD': 'XXBTZUSD',
            'ETH-USD': 'XETHZUSD',
            'SOL-USD': 'SOLUSD',
            'XRP-USD': 'XXRPZUSD',
            'ADA-USD': 'ADAUSD',
            'TRX-USD': 'TRXUSD',
            'XLM-USD': 'XXLMZUSD'
        }
        
        # CRYPTO-ONLY TRADING SYSTEM - NO STOCKS
        self.energy_stocks = []  # Disabled for crypto-only trading
        self.tech_stocks = []    # Disabled for crypto-only trading
        self.etf_symbols = []    # Disabled for crypto-only trading
        
        self.all_symbols = self.crypto_symbols  # Only crypto symbols
        
        # Portfolio allocation - CRYPTO-ONLY
        self.crypto_allocation = 1.00  # 100% crypto allocation
        self.energy_allocation = 0.00  # No energy stocks
        self.tech_allocation = 0.00    # No tech stocks
        self.etf_allocation = 0.00     # No ETFs
        
        # Update allocation explanation
        print(f"\n💰 CRYPTO-ONLY PORTFOLIO ALLOCATION STRATEGY:")
        print(f"🪙 Crypto: {self.crypto_allocation:.0%} - 100% cryptocurrency trading for maximum volatility returns")
        print(f"📈 Trading 7 major cryptocurrencies: BTC, ETH, SOL, XRP, ADA, TRX, XLM")
        print(f"🚫 No stock market trading - Pure cryptocurrency focus")
        
        # Market timezone handling
        self.market_tz = pytz.timezone('America/New_York')
        self.utc = pytz.UTC
        
        # Create necessary directories
        os.makedirs('app/logs', exist_ok=True)
        os.makedirs('app/data/live', exist_ok=True)
        
        # Enhanced cash management - MORE AGGRESSIVE
        self.cash_management = {
            'target_cash_ratio': 0.10,  # Reduced from 0.15 - keep only 10% cash
            'min_cash_ratio': 0.05,     # Reduced from 0.15 - allow down to 5%
            'max_cash_ratio': 0.15,     # Reduced from 0.20 - trigger reinvestment at 15%
            'reinvestment_threshold': 0.11  # Reduced from 0.16 - start reinvesting at 11%
        }
        
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
        
        # In backtest mode, never use online prices; return 0.0 to signal missing data
        if getattr(self, 'backtest_mode', False):
            return 0.0
        # Otherwise allow online fallback (e.g., live operations)
        return self.get_current_price_online(symbol, date)

    def get_current_price_online(self, symbol: str, date: str = None) -> float:
        """Get current price with enhanced Kraken integration for crypto"""
        # In backtest mode, never fetch online prices; use cached historical instead
        if getattr(self, 'backtest_mode', False):
            return self.get_price_from_cache(symbol, date)
        # For crypto symbols, try Kraken first
        internal_symbol = symbol
        kraken_symbol = None

        # Check if this is a crypto symbol that needs mapping to Kraken format
        if internal_symbol in self.crypto_symbols:
            # Map to Kraken format using the symbol_map
            kraken_symbol = self.symbol_map.get(internal_symbol, internal_symbol)
            self.logger.info(f"Mapping {internal_symbol} to Kraken format: {kraken_symbol}")

        if kraken_symbol and kraken_symbol in ['XXBTZUSD', 'XETHZUSD', 'XXRPZUSD', 'SOLUSD', 'ADAUSD', 'TRXUSD', 'XXLMZUSD']:
            max_retries = 3
            retry_delay = 1  # seconds
            last_error = None

            for attempt in range(max_retries):
                try:
                    self.logger.info(f"Fetching Kraken price for {internal_symbol} (Kraken: {kraken_symbol}) - Attempt {attempt + 1}/{max_retries}")
                    # IMPORTANT: pass the internal symbol; KrakenAPI.get_price maps to Kraken pair internally.
                    # Passing the Kraken pair here would cause a reverse mapping to a non-Kraken value and fail.
                    price = kraken_api.get_price(internal_symbol)
                    
                    if price > 0:
                        self.logger.info(f"Successfully got Kraken price for {internal_symbol}: ${price:.2f}")
                        return price
                    else:
                        last_error = "Zero price returned"
                        self.logger.warning(f"Kraken returned zero price for {internal_symbol} ({kraken_symbol})")
                
                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(f"Kraken price fetch failed for {internal_symbol} ({kraken_symbol}) - Attempt {attempt + 1}: {e}")
                    
                    if "Unknown asset pair" in str(e):
                        # No point retrying if the pair doesn't exist
                        break
                    
                    if attempt < max_retries - 1:  # Don't sleep on last attempt
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            
            if last_error:
                self.logger.error(f"All Kraken attempts failed for {internal_symbol} ({kraken_symbol}): {last_error}")
        
        # Fallback to yfinance for stocks or if Kraken fails
        try:
            ticker = yf.Ticker(internal_symbol)
            
            if date:
                target_date = pd.to_datetime(date).date()
                end_date = target_date + timedelta(days=1)
                hist = ticker.history(start=target_date, end=end_date, interval='1d')
                if not hist.empty:
                    price = float(hist['Close'].iloc[0])
                    self.logger.info(f"Got historical yfinance price for {internal_symbol}: ${price:.2f}")
                    return price
            
            # Try recent data with fallbacks
            for period in ['1d', '5d', '1mo']:
                try:
                    hist = ticker.history(period=period)
                    if not hist.empty:
                        price = float(hist['Close'].iloc[-1])
                        self.logger.info(f"Got yfinance price for {internal_symbol} (period: {period}): ${price:.2f}")
                        return price
                except Exception as period_error:
                    self.logger.debug(f"yfinance period {period} failed for {internal_symbol}: {period_error}")
                    continue
                
        except Exception as e:
            if symbol not in ['XEG']:  # Add other known problematic symbols here
                self.logger.error(f"yfinance failed for {internal_symbol}: {e}")
        
        return 0.0

    def detect_market_momentum_signals(self, date: str) -> Dict[str, str]:
        """ENHANCED signal detection with ultra-aggressive thresholds"""
        signals = {}
        
        for symbol in self.all_symbols:
            try:
                if symbol in self.historical_data_cache:
                    cache_entry = self.historical_data_cache[symbol]
                    data = cache_entry['data']
                    
                    # Get data up to the target date
                    target_date = self.standardize_datetime(date)
                    
                    if len(data) > 0:
                        try:
                            recent_data = data[data.index <= target_date]
                        except Exception:
                            recent_data = data
                        
                        if len(recent_data) > 200:
                            try:
                                # Create a proper copy to avoid pandas warnings
                                recent_data = recent_data.copy()
                                recent_data['EMA_50'] = recent_data['Close'].ewm(span=50).mean()
                                recent_data['EMA_200'] = recent_data['Close'].ewm(span=200).mean()
                                recent_data['EMA_20'] = recent_data['Close'].ewm(span=20).mean()  # Added shorter EMA
                                
                                # Enhanced trend bias using multiple EMAs
                                current_ema_20 = recent_data['EMA_20'].iloc[-1]
                                current_ema_50 = recent_data['EMA_50'].iloc[-1]
                                current_ema_200 = recent_data['EMA_200'].iloc[-1]
                                
                                # More sophisticated trend detection
                                if current_ema_20 > current_ema_50 and current_ema_50 > current_ema_200:
                                    trend_bias = "STRONG_BULLISH"
                                elif current_ema_20 > current_ema_50:
                                    trend_bias = "BULLISH"
                                elif current_ema_20 < current_ema_50 and current_ema_50 < current_ema_200:
                                    trend_bias = "STRONG_BEARISH"
                                else:
                                    trend_bias = "BEARISH"
                            except:
                                trend_bias = "NEUTRAL"
                            
                            try:
                                # Volume analysis with shorter window
                                avg_volume = recent_data['Volume'].tail(10).mean()  # Reduced from 20 to 10
                                current_volume = recent_data['Volume'].iloc[-1]
                                volume_confirmed = current_volume > (avg_volume * 1.1)  # Reduced from 1.2
                            except:
                                volume_confirmed = True
                            
                            try:
                                # More frequent momentum checks
                                recent_3 = recent_data['Close'].tail(3)  # Added 3-period momentum
                                if len(recent_3) >= 2:
                                    momentum_3 = (recent_3.iloc[-1] - recent_3.iloc[0]) / recent_3.iloc[0]
                                else:
                                    momentum_3 = 0
                                
                                recent_6 = recent_data['Close'].tail(6)
                                if len(recent_6) >= 2:
                                    momentum_6 = (recent_6.iloc[-1] - recent_6.iloc[0]) / recent_6.iloc[0]
                                else:
                                    momentum_6 = 0
                                
                                recent_12 = recent_data['Close'].tail(12)  # Reduced from 24
                                if len(recent_12) >= 2:
                                    momentum_12 = (recent_12.iloc[-1] - recent_12.iloc[0]) / recent_12.iloc[0]
                                else:
                                    momentum_12 = 0
                                
                                # Ultra-aggressive signal thresholds - MORE SENSITIVE
                                if trend_bias in ["STRONG_BULLISH", "BULLISH"]:
                                    if momentum_3 > 0.015:  # Reduced from 0.02 - more sensitive
                                        signals[symbol] = "EXPLOSIVE_UP"
                                    elif momentum_6 > 0.025:  # Reduced from 0.03
                                        signals[symbol] = "STRONG_UP"
                                    elif momentum_12 > 0.04:  # Reduced from 0.05
                                        signals[symbol] = "TREND_UP"
                                    elif momentum_3 < -0.015:  # Reduced from -0.02 - quicker reversal
                                        signals[symbol] = "REVERSAL_DOWN"
                                    else:
                                        signals[symbol] = "NEUTRAL"
                                
                                elif trend_bias in ["STRONG_BEARISH", "BEARISH"]:
                                    if momentum_3 < -0.015:  # Reduced from -0.02 - more sensitive
                                        signals[symbol] = "STRONG_DOWN"
                                    elif momentum_3 > 0.015:  # Reduced from 0.02 - quicker counter-trend
                                        signals[symbol] = "COUNTER_TREND_UP"
                                    else:
                                        signals[symbol] = "BEARISH_HOLD"
                                
                                else:
                                    # Even neutral trends can trigger on strong short-term moves
                                    if momentum_3 > 0.02:  # New trigger for neutral trends
                                        signals[symbol] = "SHORT_TERM_UP"
                                    elif momentum_3 < -0.02:
                                        signals[symbol] = "SHORT_TERM_DOWN"
                                    else:
                                        signals[symbol] = "NEUTRAL"
                                    
                            except:
                                signals[symbol] = "NEUTRAL"
                        else:
                            signals[symbol] = "NEUTRAL"
                    else:
                        signals[symbol] = "NEUTRAL"
                else:
                    signals[symbol] = "NEUTRAL"
                    
            except Exception as e:
                self.logger.warning(f"Signal detection failed for {symbol}: {e}")
                signals[symbol] = "NEUTRAL"
        
        return signals

    def update_trailing_stops(self, date: str) -> None:
        """Update trailing stop levels with adjusted volatility thresholds"""
        for symbol, position in list(self.positions.items()):
            if position['shares'] <= 0:
                continue
            current_price = self.get_price_from_cache(symbol, date)
            if current_price > 0:
                # Calculate volatility-based stop adjustment
                volatility = self._calculate_volatility(symbol, date)
                
                # Adjusted stop percentages
                if volatility > 0.03:  # High volatility
                    stop_percentage = 0.92  # Was 0.90
                elif volatility > 0.02:  # Medium volatility
                    stop_percentage = 0.90  # Was 0.88
                else:  # Low volatility
                    stop_percentage = 0.88  # Was 0.85
                
                # Initialize trailing stop
                if symbol not in self.trailing_stops:
                    self.trailing_stops[symbol] = position['avg_price'] * stop_percentage
                
                # Update trailing stop with dynamic adjustment
                potential_new_stop = current_price * stop_percentage
                if potential_new_stop > self.trailing_stops[symbol]:
                    self.trailing_stops[symbol] = potential_new_stop
                    
                    self.logger.info(f"Updated trailing stop for {symbol}: ${self.trailing_stops[symbol]:.2f} (Volatility: {volatility:.3f})")

    def _calculate_volatility(self, symbol: str, date: str, window: int = 14) -> float:
        """Calculate asset volatility for dynamic stop adjustment"""
        try:
            if symbol in self.historical_data_cache:
                data = self.historical_data_cache[symbol]['data']
                target_date = self.standardize_datetime(date)
                
                # Get data up to target date
                historical_data = data[data.index <= target_date].tail(window)
                
                if len(historical_data) > 0:
                    # Calculate daily returns
                    returns = historical_data['Close'].pct_change().dropna()
                    
                    # Calculate volatility (standard deviation of returns)
                    volatility = returns.std()
                    
                    # Store for reference
                    self.volatility_metrics[symbol] = {
                        'value': volatility,
                        'date': date
                    }
                    
                    return volatility
            
            return 0.02  # Default to medium volatility if calculation fails
            
        except Exception as e:
            self.logger.warning(f"Volatility calculation failed for {symbol}: {e}")
            return 0.02  # Default to medium volatility

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
        self.logger.info("Starting Day 1 portfolio allocation")
        
        print(f"\n🚀 DAY 1 PURE $5K ALLOCATION - {date}")
        print("=" * 60)
        
        # Calculate allocation amounts
        crypto_budget = self.cash * self.crypto_allocation
        energy_budget = self.cash * self.energy_allocation  
        tech_budget = self.cash * self.tech_allocation
        etf_budget = self.cash * self.etf_allocation
        
        print(f"\n💰 ALLOCATION BUDGETS:")
        print(f"   🪙 Crypto: ${crypto_budget:.2f} ({self.crypto_allocation:.0%})")
        print(f"   💻 Tech: ${tech_budget:.2f} ({self.tech_allocation:.0%})")
        print(f"   ⚡ Energy: ${energy_budget:.2f} ({self.energy_allocation:.0%})")
        print(f"   📈 ETFs: ${etf_budget:.2f} ({self.etf_allocation:.0%})")
        
        # Calculate per-symbol budgets
        crypto_per_symbol = crypto_budget / len(self.crypto_symbols)
        energy_per_symbol = energy_budget / len(self.energy_stocks) if len(self.energy_stocks) > 0 else 0
        tech_per_symbol = tech_budget / len(self.tech_stocks)
        etf_per_symbol = etf_budget / len(self.etf_symbols) if len(self.etf_symbols) > 0 else 0
        
        print(f"\n💵 PER-SYMBOL ALLOCATION:")
        print(f"   🪙 Crypto: ${crypto_per_symbol:.2f} per symbol")
        print(f"   💻 Tech: ${tech_per_symbol:.2f} per symbol")
        if energy_per_symbol > 0:
            print(f"   ⚡ Energy: ${energy_per_symbol:.2f} per symbol")
        if etf_per_symbol > 0:
            print(f"   📈 ETFs: ${etf_per_symbol:.2f} per symbol")
        
        total_invested = 0.0
        failed_symbols = []
        
        # Invest in all cryptos
        print(f"\n🪙 CRYPTO INVESTMENTS ({self.crypto_allocation:.0%}):")
        print("   Symbol      Shares           Price         Amount     % of Portfolio")
        print("   " + "-"*65)
        for symbol in self.crypto_symbols:
            price = self.get_price_from_cache(symbol, date)
            if price > 0:
                budget = crypto_per_symbol
                shares = self._max_shares_for_budget(budget, 'BUY', price)
                if shares <= 0:
                    failed_symbols.append(symbol)
                    continue
                exec_price = self._adjusted_price('BUY', price)
                fee_rate = self.trading_costs.get('fee_rate', 0.0)
                total_cost = shares * exec_price * (1.0 + fee_rate)
                if total_cost > self.cash:
                    # scale to actual available cash for this symbol
                    shares = self._max_shares_for_budget(self.cash, 'BUY', price)
                    exec_price = self._adjusted_price('BUY', price)
                    total_cost = shares * exec_price * (1.0 + fee_rate)
                if shares <= 0 or total_cost <= 0:
                    failed_symbols.append(symbol)
                    continue
                # Deduct cash and add position
                self.cash -= total_cost
                self._add_to_position(symbol, shares, exec_price, 'crypto')
                total_invested += total_cost
                portfolio_pct = (total_cost / self.initial_balance) * 100
                # Record trade with adjusted execution price and cash impact
                self._record_trade(date, symbol, 'BUY', shares, exec_price, total_cost, 'Day1_Crypto_Allocation')
                print(f"   {symbol:<10} {shares:>12.8f} @ ${exec_price:>8.4f} = ${total_cost:>8.2f} ({portfolio_pct:>5.1f}%)")
                self.logger.info(f"Allocated {shares:.8f} {symbol} @ ${exec_price:.4f} = ${total_cost:.2f}")
            else:
                failed_symbols.append(symbol)
                self.logger.warning(f"Failed to get price for {symbol}")
        
        # Invest in tech stocks
        print(f"\n💻 TECH INVESTMENTS ({self.tech_allocation:.0%}):")
        print("   Symbol      Shares           Price         Amount     % of Portfolio")
        print("   " + "-"*65)
        for symbol in self.tech_stocks:
            price = self.get_price_from_cache(symbol, date)
            if price > 0:
                budget = tech_per_symbol
                shares = self._max_shares_for_budget(budget, 'BUY', price)
                if shares <= 0:
                    failed_symbols.append(symbol)
                    continue
                exec_price = self._adjusted_price('BUY', price)
                fee_rate = self.trading_costs.get('fee_rate', 0.0)
                total_cost = shares * exec_price * (1.0 + fee_rate)
                if total_cost > self.cash:
                    shares = self._max_shares_for_budget(self.cash, 'BUY', price)
                    exec_price = self._adjusted_price('BUY', price)
                    total_cost = shares * exec_price * (1.0 + fee_rate)
                if shares <= 0 or total_cost <= 0:
                    failed_symbols.append(symbol)
                    continue
                self.cash -= total_cost
                self._add_to_position(symbol, shares, exec_price, 'tech')
                total_invested += total_cost
                portfolio_pct = (total_cost / self.initial_balance) * 100
                self._record_trade(date, symbol, 'BUY', shares, exec_price, total_cost, 'Day1_Tech_Allocation')
                print(f"   {symbol:<10} {shares:>12.8f} @ ${exec_price:>8.4f} = ${total_cost:>8.2f} ({portfolio_pct:>5.1f}%)")
                self.logger.info(f"Allocated {shares:.8f} {symbol} @ ${exec_price:.4f} = ${total_cost:.2f}")
            else:
                failed_symbols.append(symbol)
                self.logger.warning(f"Failed to get price for {symbol}")
        
        # Summary (fees+slippage already applied per-trade above when adding positions)
        # total_invested here is a nominal target; cash was deducted per trade with costs
        # so we avoid double-counting by not reducing cash again.
        print("\n📊 ALLOCATION SUMMARY:")
        print(f"   💰 Total Investment: ${total_invested:.2f} ({(total_invested/self.initial_balance)*100:.1f}% of portfolio)")
        print(f"    Remaining Cash: ${self.cash:.2f} ({(self.cash/self.initial_balance)*100:.1f}% of portfolio)")
        print(f"   📈 Number of Positions: {len([p for p in self.positions.values() if p['shares'] > 0])}")
        
        if failed_symbols:
            print(f"\n⚠️  Failed to allocate to {len(failed_symbols)} symbols:")
            for symbol in failed_symbols:
                print(f"   - {symbol}")
        
        self.logger.info(f"Day 1 allocation complete: ${total_invested:.2f} invested, ${self.cash:.2f} cash remaining")

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
        # Persist to DB
        try:
            db = SessionLocal()
            db.add(TradeRecord(
                timestamp=datetime.now(),
                date=date,
                symbol=symbol,
                action=action,
                shares=shares,
                price=price,
                amount=amount,
                strategy=strategy,
                reason=reason,
                category=trade['category']
            ))
            db.commit()
            db.close()
        except Exception as e:
            self.logger.debug(f"Failed to persist trade to DB: {e}")
        
        # Update last trade date for cooldown tracking
        self.last_trade_date[symbol] = date

        # Update intraday equity curve (post-trade) for more realistic drawdown
        try:
            eq_value = self.calculate_portfolio_value(date)
            self.intraday_equity_values.append(float(eq_value))
            self.intraday_equity_times.append(date)
        except Exception as e:
            self.logger.debug(f"Failed updating intraday equity after trade: {e}")

    def _exceeded_daily_loss_halt(self, date: str) -> bool:
        """Return True if today's equity drawdown exceeds the configured daily loss halt percent."""
        try:
            if self._day_start_equity is None or self._day_start_equity <= 0:
                return False
            current = float(self.calculate_portfolio_value(date))
            dd = (self._day_start_equity - current) / self._day_start_equity
            return dd >= float(self.risk_controls.get('daily_loss_halt_pct', 0.0))
        except Exception:
            return False

    def _cap_position_size(self, symbol: str, desired_amount: float, date: str) -> float:
        """Cap an intended buy amount so that the resulting position does not exceed max_position_pct of equity."""
        try:
            equity = float(self.calculate_portfolio_value(date))
            cap_pct = float(self.risk_controls.get('max_position_pct', 1.0))
            max_value = equity * cap_pct
            current_price = self.get_price_from_cache(symbol, date)
            current_value = 0.0
            pos = self.positions.get(symbol)
            if pos and current_price > 0:
                current_value = float(pos['shares']) * float(current_price)
            remaining = max_value - current_value
            if remaining <= 0:
                return 0.0
            return min(desired_amount, remaining)
        except Exception:
            return desired_amount

    def _adjusted_price(self, action: str, price: float) -> float:
        """Apply slippage to execution price based on action.
        Buys pay up (price increases), Sells receive less (price decreases)."""
        slip = self.trading_costs.get('slippage_bps', 0) / 10000.0
        if action == 'BUY':
            return price * (1.0 + slip)
        elif action == 'SELL':
            return price * (1.0 - slip)
        return price

    def _max_shares_for_budget(self, budget: float, action: str, price: float) -> float:
        """Compute shares purchasable given a cash budget including fees and slippage."""
        adj_price = self._adjusted_price(action, price)
        fee_rate = self.trading_costs.get('fee_rate', 0.0)
        denom = adj_price * (1.0 + fee_rate)
        return 0.0 if denom <= 0 else budget / denom

    def manage_cash_position(self, date: str) -> None:
        """Actively manage cash position with aggressive reinvestment"""
        try:
            portfolio_value = self.calculate_portfolio_value(date)
            current_cash_ratio = self.cash / portfolio_value
            
            # Check if we have excess cash
            if current_cash_ratio > self.cash_management['reinvestment_threshold']:
                target_cash = portfolio_value * self.cash_management['target_cash_ratio']
                excess_cash = self.cash - target_cash
                
                if excess_cash > 100:  # Minimum reinvestment amount reduced to $100
                    self.logger.info(f"Excess cash detected: ${excess_cash:.2f}")
                    
                    # Get current signals
                    signals = self.detect_market_momentum_signals(date)
                    
                    # Find best opportunities (respecting allocation settings)
                    opportunities = []
                    for symbol, signal in signals.items():
                        if signal in ["EXPLOSIVE_UP", "STRONG_UP", "TREND_UP", "COUNTER_TREND_UP"]:
                            # Check if symbol is in allowed categories based on allocations
                            if (symbol in self.crypto_symbols and self.crypto_allocation > 0) or \
                               (symbol in self.tech_stocks and self.tech_allocation > 0) or \
                               (symbol in self.energy_stocks and self.energy_allocation > 0) or \
                               (symbol in self.etf_symbols and self.etf_allocation > 0):
                                # Calculate recent performance
                                performance = self._calculate_recent_performance(symbol, date)
                                opportunities.append((symbol, signal, performance))
                    
                    # Sort by performance
                    opportunities.sort(key=lambda x: x[2], reverse=True)
                    
                    # Reinvest in top opportunities
                    cash_to_deploy = excess_cash
                    max_position_size = portfolio_value * 0.30  # Max 30% per position
                    
                    for symbol, signal, perf in opportunities[:5]:  # Top 5 opportunities
                        if cash_to_deploy < 100:  # Stop if remaining cash too small
                            break
                            
                        # Calculate position size based on signal strength
                        if signal == "EXPLOSIVE_UP":
                            position_size = min(cash_to_deploy * 0.4, 400)
                        elif signal == "COUNTER_TREND_UP":
                            position_size = min(cash_to_deploy * 0.3, 300)
                        else:  # STRONG_UP or TREND_UP
                            position_size = min(cash_to_deploy * 0.25, 250)
                        
                        current_price = self.get_price_from_cache(symbol, date)
                        if current_price > 0:
                            # Check existing position size
                            existing_value = 0
                            if symbol in self.positions:
                                existing_value = self.positions[symbol]['shares'] * current_price
                            
                            # Adjust position size to respect max position size
                            if existing_value + position_size > max_position_size:
                                position_size = max(0, max_position_size - existing_value)
                            
                            if position_size >= 100:  # Minimum trade size
                                shares = position_size / current_price
                                
                                # Execute reinvestment with costs and slippage
                                exec_price = self._adjusted_price('BUY', current_price)
                                fee_rate = self.trading_costs.get('fee_rate', 0.0)
                                shares = self._max_shares_for_budget(position_size, 'BUY', current_price)
                                total_cost = shares * exec_price * (1.0 + fee_rate)
                                if total_cost <= self.cash and shares > 0:
                                    self._add_to_position(symbol, shares, exec_price, 
                                                        self._get_symbol_category(symbol))
                                    self.cash -= total_cost
                                cash_to_deploy -= position_size
                                
                                self._record_trade(date, symbol, 'BUY', shares, current_price,
                                                 position_size, 'Cash_Reinvestment', 
                                                 f"Aggressive cash reinvestment ({signal})")
                                
                                self.logger.info(f"Reinvested ${position_size:.2f} in {symbol}")
                    
                    # Update cash ratios
                    new_cash_ratio = self.cash / self.calculate_portfolio_value(date)
                    self.logger.info(f"New cash ratio: {new_cash_ratio:.1%}")
                    
        except Exception as e:
            self.logger.error(f"Cash management error: {e}")

    def _calculate_recent_performance(self, symbol: str, date: str, days: int = 5) -> float:
        """Calculate recent performance for opportunity ranking"""
        try:
            if symbol in self.historical_data_cache:
                data = self.historical_data_cache[symbol]['data']
                target_date = self.standardize_datetime(date)
                
                # Get recent data
                recent_data = data[data.index <= target_date].tail(days)
                
                if len(recent_data) > 1:
                    start_price = recent_data['Close'].iloc[0]
                    end_price = recent_data['Close'].iloc[-1]
                    return ((end_price - start_price) / start_price) * 100
                    
            return 0.0
            
        except Exception as e:
            self.logger.debug(f"Performance calculation failed for {symbol}: {e}")
            return 0.0

    def simulate_pure_trading_day(self, date: str, is_first_day: bool = False) -> None:
        """ENHANCED trading simulation with improved cash management"""
        print(f"\n📅 {date}")
        print("-" * 50)
        
        # Establish day start equity for daily loss halt
        try:
            self._day_start_equity = float(self.calculate_portfolio_value(date))
        except Exception:
            self._day_start_equity = float(self.initial_balance)

        if is_first_day:
            self.execute_day_1_intelligent_allocation(date)
        else:
            # Update trailing stops for all positions
            self.update_trailing_stops(date)
            
            # Manage cash position actively
            self.manage_cash_position(date)
            
            # Check for trailing stop exits FIRST
            trailing_stop_exits = 0
            for symbol, position in list(self.positions.items()):
                if position['shares'] > 0:
                    # Use Kraken API for crypto, cache for stocks
                    if symbol in self.crypto_symbols:
                        current_price = self.get_current_price_online(symbol, date)
                    else:
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
                # Use Kraken API for crypto, cache for stocks
                if symbol in self.crypto_symbols:
                    current_price = self.get_current_price_online(symbol, date)
                else:
                    current_price = self.get_price_from_cache(symbol, date)
                    
                if current_price <= 0:
                    continue
                
                # Skip if in cooldown period
                if self.is_in_cooldown(symbol, date):
                    continue
                
                # EXPLOSIVE UP signals - buy aggressively
                if signal in ["EXPLOSIVE_UP", "COUNTER_TREND_UP", "SHORT_TERM_UP"] and self.cash > 200:
                    category = self.positions.get(symbol, {}).get('category', 'unknown')
                    buy_amount = min(500 if category == 'crypto' else 400, self.cash * 0.5)  # Increased from 400/300
                    # Compute shares to fit budget including costs/slippage
                    shares = self._max_shares_for_budget(buy_amount, 'BUY', current_price)
                    exec_price = self._adjusted_price('BUY', current_price)
                    fee_rate = self.trading_costs.get('fee_rate', 0.0)
                    total_cost = shares * exec_price * (1.0 + fee_rate)
                    if total_cost <= self.cash and shares > 0:
                        self._add_to_position(symbol, shares, exec_price, category)
                        self.cash -= total_cost
                    self.last_trade_date[symbol] = date
                    
                    signal_type = "counter-trend" if signal == "COUNTER_TREND_UP" else "momentum"
                    self._record_trade(date, symbol, 'BUY', shares, current_price, 
                                     buy_amount, f'{signal_type.capitalize()}_Entry', f"Strong {signal_type} signal")
                    trades_executed += 1
                    
                    print(f"  💥 {signal_type.upper()} BUY: {shares:.6f} {symbol} @ ${current_price:.4f}")
                
                # STRONG UP and TREND UP signals - buy moderately
                elif signal in ["STRONG_UP", "TREND_UP"] and self.cash > 150:
                    category = self.positions.get(symbol, {}).get('category', 'unknown')
                    # Daily loss halt: skip new buys if exceeded
                    if self._exceeded_daily_loss_halt(date):
                        continue
                    # Desired buy amount
                    buy_amount = min(350 if category == 'crypto' else 300, self.cash * 0.4)
                    # Per-position cap
                    buy_amount = self._cap_position_size(symbol, buy_amount, date)
                    if buy_amount <= 0:
                        continue
                    shares = buy_amount / current_price
                    
                    self._add_to_position(symbol, shares, current_price, category)
                    self.cash -= buy_amount
                    self.last_trade_date[symbol] = date
                    
                    self._record_trade(date, symbol, 'BUY', shares, current_price,
                                     buy_amount, 'Trend_Following', f"{signal} signal confirmed")
                    trades_executed += 1
                    
                    print(f"  🚀 TREND BUY: {shares:.6f} {symbol} @ ${current_price:.4f}")
                
                # Handle defensive sells and reversals - MORE AGGRESSIVE
                elif signal in ["REVERSAL_DOWN", "STRONG_DOWN", "SHORT_TERM_DOWN"] and symbol in self.positions:
                    if self.positions[symbol]['shares'] > 0:
                        # More aggressive selling on strong down signals
                        sell_ratio = 0.8 if signal == "STRONG_DOWN" else 0.6 if signal == "REVERSAL_DOWN" else 0.4  # Increased ratios
                        shares_to_sell = self.positions[symbol]['shares'] * sell_ratio
                        sell_amount = shares_to_sell * current_price
                        position_return = self.calculate_position_return(symbol, current_price)
                        
                        self.positions[symbol]['shares'] -= shares_to_sell
                        self.cash += sell_amount
                        self.last_trade_date[symbol] = date
                        
                        self._record_trade(date, symbol, 'SELL', shares_to_sell, current_price,
                                         sell_amount, 'Defensive_Exit', f"{signal} detected ({position_return:+.1f}%)")
                        trades_executed += 1
                        
                        print(f"  🛡️  DEFENSIVE SELL: {shares_to_sell:.6f} {symbol} @ ${current_price:.4f} ({position_return:+.1f}%)")
            
            if trades_executed == 0 and trailing_stop_exits == 0:
                print("  ⏸️  No trades executed - market in neutral zone or cooldown periods active")
            elif trailing_stop_exits > 0:
                print(f"  📊 Trailing stops executed: {trailing_stop_exits}")
        
        # Calculate and store portfolio metrics
        portfolio_value = self.calculate_portfolio_value(date)
        return_pct = ((portfolio_value - self.initial_balance) / self.initial_balance) * 100
        
        # Store structured snapshot separately; keep daily_values as numeric series only
        self.backtest_daily_snapshots.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'return_pct': return_pct,
            'active_positions': len([p for p in self.positions.values() if p['shares'] > 0])
        })
        
        self.daily_values.append(portfolio_value)
        
        print(f"  📊 Portfolio: ${portfolio_value:,.2f} | Cash: ${self.cash:.2f} | Return: {return_pct:+.2f}%")

    def _add_to_position(self, symbol: str, shares: float, price: float, category: str) -> None:
        """Enhanced position management with volatility-based sizing"""
        try:
            # Calculate volatility-based position adjustment
            volatility = self._calculate_volatility(symbol, datetime.now(self.utc).strftime('%Y-%m-%d'))
            
            # Adjust position size based on volatility
            if volatility > 0.03:  # High volatility
                shares *= 0.8  # Reduce position size
            elif volatility < 0.01:  # Low volatility
                shares *= 1.2  # Increase position size
            
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
                
            # Log position adjustments
            self.logger.info(f"Added to {symbol}: {shares:.6f} shares (Volatility: {volatility:.3f})")
            
        except Exception as e:
            self.logger.error(f"Position adjustment failed for {symbol}: {e}")
            # Fallback to original position sizing
            if symbol in self.positions:
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
            # Backtest should use historical OHLC for all assets (no live quotes)
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
        current_value = self.calculate_portfolio_value(date)
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

        fee_rate = self.trading_costs.get('fee_rate', 0.0)

        if action == 'BUY':
            # Apply slippage and fees and check cash
            exec_price = self._adjusted_price('BUY', price)
            total_cost = shares * exec_price * (1.0 + fee_rate)
            if total_cost > self.cash:
                # Scale shares to available cash
                shares = self._max_shares_for_budget(self.cash, 'BUY', price)
                exec_price = self._adjusted_price('BUY', price)
                total_cost = shares * exec_price * (1.0 + fee_rate)
            if shares <= 0:
                self.logger.warning("BUY aborted: insufficient cash after costs")
                return False
            self.cash -= total_cost
            self._add_to_position(symbol, shares, exec_price, self._get_symbol_category(symbol))
            recorded_amount = total_cost

        elif action == 'SELL':
            if symbol not in self.positions or self.positions[symbol]['shares'] < shares:
                self.logger.warning(f"Insufficient shares to sell {symbol}")
                return False
            exec_price = self._adjusted_price('SELL', price)
            self.positions[symbol]['shares'] -= shares
            proceeds_net = shares * exec_price * (1.0 - fee_rate)
            self.cash += proceeds_net
            recorded_amount = proceeds_net
        else:
            self.logger.warning(f"Unknown action: {action}")
            return False

        # Record trade with execution-adjusted price and actual cash amount
        self._record_trade(datetime.now().strftime('%Y-%m-%d'), symbol, action,
                           shares, exec_price, recorded_amount, 'Live_Paper_Trading', reason)
        self.logger.info(f"PAPER {action}: {shares:.6f} {symbol} @ ${exec_price:.4f} - {reason}")
        return True

    def _get_symbol_category(self, symbol: str) -> str:
        """Get category for symbol"""
        if symbol in getattr(self, 'crypto_symbols', []):
            return 'crypto'
        if symbol in getattr(self, 'energy_stocks', []):
            return 'energy'
        if symbol in getattr(self, 'tech_stocks', []):
            return 'tech'
        if symbol in getattr(self, 'etf_symbols', []):
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
        
        # Store live structured snapshot separately and persist to DB
        live_snap = {
            'timestamp': current_time.isoformat(),
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'return_pct': return_pct,
            'active_positions': len([p for p in self.positions.values() if p['shares'] > 0])
        }
        self.live_daily_snapshots.append(live_snap)
        try:
            db = SessionLocal()
            db.add(DailySnapshot(
                date=date_str,
                portfolio_value=float(portfolio_value),
                cash=float(self.cash),
                return_pct=float(return_pct),
                active_positions=int(live_snap['active_positions'])
            ))
            db.commit()
            db.close()
        except Exception as e:
            self.logger.debug(f"Failed to persist live daily snapshot: {e}")
        
        # Log current status
        self.logger.info(f"Portfolio: ${portfolio_value:,.2f} | Cash: ${self.cash:.2f} | Return: {return_pct:+.2f}%")

    def run_crypto_check_cycle(self) -> None:
        """Run a lightweight cycle primarily for crypto outside stock market hours.
        Applies the same risk controls. Records a snapshot and persists it."""
        current_time = datetime.now(self.market_tz)
        date_str = current_time.strftime('%Y-%m-%d')

        # Update trailing stops and manage cash opportunistically for crypto symbols only
        try:
            self.update_trailing_stops(date_str)
            # Optional: light reinvest only into crypto when signals are strong
            signals = self.detect_market_momentum_signals(date_str)
            for symbol in self.crypto_symbols:
                sig = signals.get(symbol, 'NEUTRAL')
                if sig in ("STRONG_UP", "TREND_UP") and self.cash > 50:
                    if self._exceeded_daily_loss_halt(date_str):
                        continue
                    price = self.get_live_price(symbol)
                    if price <= 0:
                        continue
                    buy_amount = min(200, self.cash * 0.2)
                    buy_amount = self._cap_position_size(symbol, buy_amount, date_str)
                    if buy_amount <= 0:
                        continue
                    shares = buy_amount / price
                    self._add_to_position(symbol, shares, price, 'crypto')
                    self.cash -= buy_amount
                    self._record_trade(date_str, symbol, 'BUY', shares, price,
                                       buy_amount, 'Crypto_OffHours', f"{sig} crypto check")
        except Exception as e:
            self.logger.debug(f"Crypto check encountered an issue: {e}")

        # Snapshot and persist
        try:
            portfolio_value = self.calculate_portfolio_value_live()
            return_pct = ((portfolio_value - self.initial_balance) / self.initial_balance) * 100
            live_snap = {
                'timestamp': current_time.isoformat(),
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'return_pct': return_pct,
                'active_positions': len([p for p in self.positions.values() if p['shares'] > 0])
            }
            self.live_daily_snapshots.append(live_snap)
            db = SessionLocal()
            db.add(DailySnapshot(
                date=date_str,
                portfolio_value=float(portfolio_value),
                cash=float(self.cash),
                return_pct=float(return_pct),
                active_positions=int(live_snap['active_positions'])
            ))
            db.commit()
            db.close()
            self.logger.info(f"[Crypto Cycle] Portfolio: ${portfolio_value:,.2f} | Cash: ${self.cash:.2f} | Return: {return_pct:+.2f}%")
        except Exception as e:
            self.logger.debug(f"Failed crypto snapshot persist: {e}")

    def export_live_daily_csv(self) -> Optional[str]:
        """Export accumulated live snapshots for the current day to CSV. Returns filepath or None."""
        try:
            os.makedirs('app/data/exports', exist_ok=True)
            today = datetime.now(self.market_tz).strftime('%Y%m%d')
            fp = f"app/data/exports/live_equity_{today}.csv"
            import csv
            with open(fp, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'portfolio_value', 'cash', 'return_pct', 'active_positions'])
                for snap in self.live_daily_snapshots:
                    writer.writerow([
                        snap.get('timestamp'),
                        snap.get('portfolio_value'),
                        snap.get('cash'),
                        snap.get('return_pct'),
                        snap.get('active_positions')
                    ])
            self.logger.info(f"Exported live equity CSV to {fp}")
            return fp
        except Exception as e:
            self.logger.debug(f"Failed to export live CSV: {e}")
            return None

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
        
        # Save live snapshot values
        live_values_file = f"app/data/live/daily_values_{timestamp}.json"
        with open(live_values_file, 'w') as f:
            json.dump(self.live_daily_snapshots, f, indent=2, default=str)
        
        # Save backtest snapshots if present
        if self.backtest_daily_snapshots:
            backtest_values_file = f"app/data/live/backtest_daily_values_{timestamp}.json"
            with open(backtest_values_file, 'w') as f:
                json.dump(self.backtest_daily_snapshots, f, indent=2, default=str)

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

    def calculate_advanced_metrics(self) -> Dict:
        """Calculate advanced performance metrics"""
        try:
            self.logger.info("Calculating advanced performance metrics...")
            
            # Calculate daily returns
            portfolio_values = pd.Series(self.daily_values, dtype=float)
            daily_returns = portfolio_values.pct_change().dropna()
            self.performance_metrics['daily_returns'] = daily_returns
            self.performance_metrics['n_daily'] = int(len(daily_returns))
            
            # Calculate cumulative returns
            cumulative_returns = (1 + daily_returns).cumprod()
            self.performance_metrics['cumulative_returns'] = cumulative_returns
            
            # 1. Sharpe Ratio
            risk_free_rate = 0.04  # 4% annual risk-free rate
            avg_return = daily_returns.mean() * 252
            std_dev = daily_returns.std() * np.sqrt(252)
            self.performance_metrics['sharpe_ratio'] = (avg_return - risk_free_rate) / std_dev if std_dev > 0 else 0
            
            # 2. Sortino Ratio
            # Use daily risk-free threshold for downside; more conservative than 0
            rf_daily = risk_free_rate / 252.0
            downside_returns = daily_returns[daily_returns < rf_daily]
            downside_std = np.sqrt(np.mean(downside_returns ** 2)) * np.sqrt(252)
            self.performance_metrics['sortino_ratio'] = (avg_return - risk_free_rate) / downside_std if downside_std > 0 else 0
            
            # 3. Maximum Drawdown
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = cumulative_returns / rolling_max - 1
            self.performance_metrics['max_drawdown'] = float(drawdowns.min())

            # 3b. Intraday risk metrics from per-trade equity curve
            if len(self.intraday_equity_values) > 1:
                intraday_series = pd.Series(self.intraday_equity_values, dtype=float)
                intraday_returns = intraday_series.pct_change().dropna()
                self.performance_metrics['n_intraday'] = int(len(intraday_returns))
                # Intraday max drawdown on equity curve
                intraday_cum = (1 + intraday_returns).cumprod()
                intraday_roll_max = intraday_cum.expanding().max()
                intraday_dd = intraday_cum / intraday_roll_max - 1
                self.performance_metrics['intraday_max_drawdown'] = float(intraday_dd.min())
                # Intraday volatility (annualized proxy using per-trade steps ~ not time-normalized)
                self.performance_metrics['intraday_volatility'] = float(intraday_returns.std())
            else:
                self.performance_metrics['n_intraday'] = 0
                self.performance_metrics['intraday_max_drawdown'] = 0.0
                self.performance_metrics['intraday_volatility'] = 0.0
            
            # 4. Day-based Win Rate
            winning_days = len(daily_returns[daily_returns > 0])
            total_days = len(daily_returns)
            self.performance_metrics['day_win_rate'] = winning_days / total_days if total_days > 0 else 0
            
            # 5. Day-based Profit Factor (kept for context)
            gains = daily_returns[daily_returns > 0].sum()
            losses = abs(daily_returns[daily_returns < 0].sum())
            self.performance_metrics['day_profit_factor'] = gains / losses if losses != 0 else 0

            # 5b. Trade-based metrics using FIFO realized PnL
            realized_pnls = []  # per-closed-lot PnL values
            fifo_books: Dict[str, list] = {}
            # Ensure chronological order; trades store date 'YYYY-MM-DD'
            sorted_trades = sorted(self.trades, key=lambda t: (t.get('date', ''), t.get('symbol', ''), 0 if t.get('action')=='BUY' else 1))
            for tr in sorted_trades:
                sym = tr.get('symbol')
                action = tr.get('action')
                shares = float(tr.get('shares', 0) or 0)
                price = float(tr.get('price', 0) or 0)
                if not sym or shares <= 0 or price <= 0:
                    continue
                fifo_books.setdefault(sym, [])
                if action == 'BUY':
                    fifo_books[sym].append([shares, price])
                elif action == 'SELL':
                    remaining = shares
                    while remaining > 0 and fifo_books[sym]:
                        lot_shares, lot_price = fifo_books[sym][0]
                        take = min(remaining, lot_shares)
                        pnl = (price - lot_price) * take
                        realized_pnls.append(pnl)
                        lot_shares -= take
                        remaining -= take
                        if lot_shares <= 1e-12:
                            fifo_books[sym].pop(0)
                        else:
                            fifo_books[sym][0][0] = lot_shares
                    # If no lots to sell against, skip remainder (short sales not modeled)
            closed = len(realized_pnls)
            self.performance_metrics['n_trades_closed'] = int(closed)
            if closed > 0:
                wins = [p for p in realized_pnls if p > 0]
                losses_p = [-p for p in realized_pnls if p < 0]
                self.performance_metrics['trade_win_rate'] = len(wins) / closed if closed > 0 else 0
                total_win = float(np.sum(wins)) if wins else 0.0
                total_loss = float(np.sum(losses_p)) if losses_p else 0.0
                self.performance_metrics['trade_profit_factor'] = (total_win / total_loss) if total_loss > 0 else (0 if total_win == 0 else float('inf'))
            else:
                self.performance_metrics['trade_win_rate'] = 0.0
                self.performance_metrics['trade_profit_factor'] = 0.0
            
            # 6. Recovery Factor
            total_return = cumulative_returns.iloc[-1] - 1
            max_dd = abs(self.performance_metrics['max_drawdown'])
            self.performance_metrics['recovery_factor'] = total_return / max_dd if max_dd != 0 else 0
            
            # 7. Risk-Adjusted Return
            downside_risk = downside_returns.std() * np.sqrt(252)
            self.performance_metrics['risk_adjusted_return'] = avg_return / downside_risk if downside_risk != 0 else 0
            
            # 8. Alpha & Beta vs BTC
            try:
                # Only attempt alpha/beta if we can infer a date range
                if isinstance(portfolio_values.index, pd.DatetimeIndex) and len(portfolio_values) > 2:
                    start_dt = portfolio_values.index[0]
                    end_dt = portfolio_values.index[-1]
                    btc_data = yf.download('BTC-USD', start=start_dt, end=end_dt)
                    if not btc_data.empty:
                        btc_returns = btc_data['Close'].pct_change().dropna()
                        if not btc_returns.empty and len(daily_returns) > 1:
                            # Align lengths if needed
                            min_len = min(len(daily_returns), len(btc_returns))
                            dr = daily_returns.tail(min_len)
                            br = btc_returns.tail(min_len)
                            covariance = dr.cov(br)
                            market_variance = br.var()
                            beta = covariance / market_variance if market_variance != 0 else 0
                            self.performance_metrics['beta'] = float(beta)
                            market_return = br.mean() * 252
                            alpha = avg_return - (risk_free_rate + beta * (market_return - risk_free_rate))
                            self.performance_metrics['alpha'] = float(alpha)
                else:
                    # No datetime context in backtest list; skip alpha/beta
                    self.performance_metrics['beta'] = 0.0
                    self.performance_metrics['alpha'] = 0.0
            except Exception as e:
                self.performance_metrics['beta'] = 0.0
                self.performance_metrics['alpha'] = 0.0
                self.logger.warning(f"Could not calculate alpha/beta: {e}")
            
            # 9. Asset Correlations
            returns_df = pd.DataFrame()
            for symbol in self.positions.keys():
                try:
                    price_data = self.historical_data_cache[symbol]['data']['Close']
                    returns_df[symbol] = price_data.pct_change()
                except:
                    continue
            
            if not returns_df.empty:
                self.performance_metrics['correlation_matrix'] = returns_df.corr()
            
            # Stability warnings
            warnings = []
            if self.performance_metrics['n_daily'] < 30:
                warnings.append("Daily sample < 30; Sharpe/Sortino/Drawdown may be unstable")
            if self.performance_metrics['n_trades_closed'] < 10:
                warnings.append("Closed trades < 10; trade metrics may be unstable")
            if self.performance_metrics.get('n_intraday', 0) < 20:
                warnings.append("Intraday equity points < 20; intraday drawdown/vol may be unstable")
            self.performance_metrics['unstable_metrics_warning'] = '; '.join(warnings)

            return self.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced metrics: {e}")
            return self.performance_metrics

    def display_advanced_metrics(self):
        """Display advanced performance metrics"""
        try:
            print("\n=== ADVANCED PERFORMANCE METRICS ===")
            print("-" * 40)
            
            print(f"\n📈 Risk-Adjusted Returns:")
            print(f"   Sharpe Ratio: {self.performance_metrics.get('sharpe_ratio', 0.0):.2f}")
            print(f"   Sortino Ratio: {self.performance_metrics.get('sortino_ratio', 0.0):.2f}")
            print(f"   Risk-Adjusted Return: {self.performance_metrics.get('risk_adjusted_return', 0.0):.2f}")
            
            print(f"\n📊 Risk Metrics:")
            print(f"   Maximum Drawdown: {self.performance_metrics.get('max_drawdown', 0.0)*100:.1f}%")
            print(f"   Intraday Max Drawdown (per-trade): {self.performance_metrics.get('intraday_max_drawdown', 0.0)*100:.1f}% (points={self.performance_metrics.get('n_intraday', 0)})")
            print(f"   Intraday Volatility (per-trade std): {self.performance_metrics.get('intraday_volatility', 0.0):.4f}")
            print(f"   Beta vs BTC: {self.performance_metrics.get('beta', 0.0):.2f}")
            print(f"   Alpha (annualized): {self.performance_metrics.get('alpha', 0.0)*100:.1f}%")
            
            print(f"\n🎯 Trading Metrics:")
            print(f"   Day Win Rate: {self.performance_metrics.get('day_win_rate', 0.0)*100:.1f}% (n={self.performance_metrics.get('n_daily', 0)})")
            print(f"   Day Profit Factor: {self.performance_metrics.get('day_profit_factor', 0.0):.2f}")
            print(f"   Trade Win Rate: {self.performance_metrics.get('trade_win_rate', 0.0)*100:.1f}% (closed={self.performance_metrics.get('n_trades_closed', 0)})")
            tpf = self.performance_metrics.get('trade_profit_factor', 0.0)
            print(f"   Trade Profit Factor: {'∞' if tpf == float('inf') else f'{tpf:.2f}'}")
            print(f"   Recovery Factor: {self.performance_metrics.get('recovery_factor', 0.0):.2f}")
            
            if self.performance_metrics.get('correlation_matrix') is not None:
                print("\n📐 Asset Correlations:")
                print(self.performance_metrics['correlation_matrix'].round(2))
            warn = self.performance_metrics.get('unstable_metrics_warning')
            if warn:
                print(f"\n⚠️  Warnings: {warn}")
            
            # Cost summary if in backtest mode
            if getattr(self, 'backtest_mode', False):
                fee_cost, slip_cost, total_cost = self._compute_cost_summary()
                print("\n💸 Execution Costs (Backtest mode: ON)")
                print(f"   Fees paid: ${fee_cost:,.2f}")
                print(f"   Slippage cost: ${slip_cost:,.2f}")
                print(f"   Total drag: ${total_cost:,.2f}")

        except Exception as e:
            self.logger.error(f"Error displaying advanced metrics: {e}")

    def _compute_cost_summary(self):
        """Compute total fee and slippage costs from recorded trades.
        Assumes recorded trade 'price' is execution-adjusted and fee_rate/slippage_bps are constant."""
        fee_rate = float(self.trading_costs.get('fee_rate', 0.0))
        slip_bps = float(self.trading_costs.get('slippage_bps', 0.0))
        slip = slip_bps / 10000.0
        total_fee = 0.0
        total_slip = 0.0
        for t in self.trades:
            px_exec = float(t.get('price', 0.0))
            sh = float(t.get('shares', 0.0))
            act = t.get('action', '')
            if px_exec <= 0 or sh <= 0:
                continue
            # Fees: symmetric per side
            total_fee += sh * px_exec * fee_rate
            # Slippage: reconstruct reference price implied by slippage model
            if act == 'BUY':
                ref = px_exec / (1.0 + slip) if slip > 0 else px_exec
                total_slip += (px_exec - ref) * sh
            elif act == 'SELL':
                ref = px_exec / (1.0 - slip) if slip > 0 else px_exec
                total_slip += (ref - px_exec) * sh
        return total_fee, total_slip, (total_fee + total_slip)

    def _export_equity_curves(self, label: str) -> Dict[str, str]:
        """Export daily and intraday equity curves to CSV for quick review.
        Returns a dict with file paths for 'daily' and 'intraday'."""
        try:
            os.makedirs('app/data/exports', exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            base = f"app/data/exports/{label}_{ts}"

            # Daily: prefer backtest_daily_snapshots if available
            daily_fp = f"{base}_daily_equity.csv"
            if self.backtest_daily_snapshots:
                import csv
                with open(daily_fp, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['date', 'portfolio_value', 'cash', 'return_pct'])
                    for snap in self.backtest_daily_snapshots:
                        writer.writerow([
                            snap.get('date'),
                            snap.get('portfolio_value'),
                            snap.get('cash'),
                            snap.get('return_pct')
                        ])
            else:
                # Fallback: write index-based daily values
                import csv
                with open(daily_fp, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['index', 'portfolio_value'])
                    for i, v in enumerate(self.daily_values):
                        writer.writerow([i, v])

            # Intraday equity (post-trade snapshots)
            intraday_fp = f"{base}_intraday_equity.csv"
            import csv
            with open(intraday_fp, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time', 'equity_value'])
                for t, v in zip(self.intraday_equity_times, self.intraday_equity_values):
                    writer.writerow([t, v])

            self.logger.info(f"Exported equity curves: daily={daily_fp}, intraday={intraday_fp}")
            return {'daily': daily_fp, 'intraday': intraday_fp}
        except Exception as e:
            self.logger.error(f"Failed to export equity curves: {e}")
            return {}

    def _save_backtest_results(self, results: Dict) -> None:
        """Persist backtest results to JSON file"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = f"app/data/results/pure_5k_results_{timestamp}.json"
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info(f"Backtest results saved to {results_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save backtest results: {e}")


    def run_pure_5k_backtest(self, days: int = 30) -> Dict:
        """Run complete backtest over the last `days` business days with advanced analytics."""
        try:
            print(f"\n🚀 Running Pure $5K Trading System Backtest ({days} days)")
            print("=" * 80)
            print("🔧 Backtest mode: ON (historical OHLC only; no live price fallbacks)")

            # Ensure we have historical data cached for the requested span
            self.cache_historical_data(days)

            # Reset state
            self.cash = self.initial_balance
            self.positions = {}
            self.trades = []
            self.daily_values = []
            self.backtest_daily_snapshots = []
            self.intraday_equity_values = []
            self.intraday_equity_times = []
            self.backtest_mode = True

            # Build business-day date range
            end_date = datetime.now(pytz.UTC)
            start_date = end_date - timedelta(days=days)
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')

            for i, date in enumerate(date_range):
                date_str = date.strftime('%Y-%m-%d')
                is_first = (i == 0)
                self.simulate_pure_trading_day(date_str, is_first_day=is_first)

                # Track daily portfolio value (EOD)
                pv = self.calculate_portfolio_value(date_str)
                self.daily_values.append(float(pv))
                snapshot = {
                    'date': date_str,
                    'portfolio_value': float(pv),
                    'cash': float(self.cash),
                    'return_pct': ((float(pv) - float(self.initial_balance)) / float(self.initial_balance)) * 100.0
                }
                self.backtest_daily_snapshots.append(snapshot)
                # Persist to DB
                try:
                    db = SessionLocal()
                    db.add(DailySnapshot(
                        date=snapshot['date'],
                        portfolio_value=snapshot['portfolio_value'],
                        cash=snapshot['cash'],
                        return_pct=snapshot['return_pct'],
                        active_positions=len([p for p in self.positions.values() if p['shares'] > 0])
                    ))
                    db.commit()
                    db.close()
                except Exception as e:
                    self.logger.debug(f"Failed to persist daily snapshot: {e}")

            final_value = self.calculate_portfolio_value(date_range[-1].strftime('%Y-%m-%d')) if len(date_range) > 0 else self.initial_balance
            total_return = final_value - self.initial_balance
            return_percentage = (total_return / self.initial_balance) * 100
            target_met = return_percentage >= 10.0

            # Metrics
            self.calculate_advanced_metrics()
            self.display_advanced_metrics()
            # Cost summary
            fee_cost, slip_cost, total_cost = self._compute_cost_summary()

            results = {
                'initial_balance': self.initial_balance,
                'final_value': final_value,
                'total_return': total_return,
                'return_percentage': return_percentage,
                'target_met': target_met,
                'trades': self.trades,
                'daily_values': self.daily_values,
                'performance_metrics': self.performance_metrics,
                'costs': {
                    'fee_cost': fee_cost,
                    'slippage_cost': slip_cost,
                    'total_drag': total_cost,
                    'fee_rate': self.trading_costs.get('fee_rate', 0.0),
                    'slippage_bps': self.trading_costs.get('slippage_bps', 0.0)
                }
            }

            # Export equity curves
            exports = self._export_equity_curves(label=f"pure5k_bt_{days}d")
            if exports:
                results['exports'] = exports

            self._save_backtest_results(results)
            print("\n✅ Backtest completed successfully!")
            print("=" * 80)
            return results
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            return {}

    def run_backtest_between(self, start_date: datetime, end_date: datetime) -> Dict:
        """Run a backtest over a specific [start_date, end_date] range (business days).
        Assumes historical data cache already covers this span. Resets internal state."""
        try:
            # Reset state
            self.cash = self.initial_balance
            self.positions = {}
            self.trades = []
            self.daily_values = []
            self.backtest_daily_snapshots = []
            self.intraday_equity_values = []
            self.intraday_equity_times = []

            # Build date range (business days) with correct tz handling
            if not isinstance(start_date, pd.Timestamp):
                start_date = pd.Timestamp(start_date)
            if not isinstance(end_date, pd.Timestamp):
                end_date = pd.Timestamp(end_date)
            # Localize or convert to UTC
            start_date = start_date.tz_localize(pytz.UTC) if start_date.tz is None else start_date.tz_convert(pytz.UTC)
            end_date = end_date.tz_localize(pytz.UTC) if end_date.tz is None else end_date.tz_convert(pytz.UTC)
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')

            for i, date in enumerate(date_range):
                date_str = date.strftime('%Y-%m-%d')
                is_first = (i == 0)
                self.simulate_pure_trading_day(date_str, is_first_day=is_first)
                # Track daily portfolio value (EOD)
                pv = self.calculate_portfolio_value(date_str)
                self.daily_values.append(float(pv))
                snapshot = {
                    'date': date_str,
                    'portfolio_value': float(pv),
                    'cash': float(self.cash),
                    'return_pct': ((float(pv) - float(self.initial_balance)) / float(self.initial_balance)) * 100.0
                }
                self.backtest_daily_snapshots.append(snapshot)
                # Persist to DB
                try:
                    db = SessionLocal()
                    db.add(DailySnapshot(
                        date=snapshot['date'],
                        portfolio_value=snapshot['portfolio_value'],
                        cash=snapshot['cash'],
                        return_pct=snapshot['return_pct'],
                        active_positions=len([p for p in self.positions.values() if p['shares'] > 0])
                    ))
                    db.commit()
                    db.close()
                except Exception as e:
                    self.logger.debug(f"Failed to persist daily snapshot: {e}")

            final_value = self.calculate_portfolio_value(date_range[-1].strftime('%Y-%m-%d')) if len(date_range) > 0 else self.initial_balance
            total_return = final_value - self.initial_balance
            return_percentage = (total_return / self.initial_balance) * 100
            target_met = return_percentage >= 20.0

            # Compute metrics
            self.calculate_advanced_metrics()
            self.display_advanced_metrics()
            fee_cost, slip_cost, total_cost = self._compute_cost_summary()

            results = {
                'start_date': str(start_date),
                'end_date': str(end_date),
                'initial_balance': self.initial_balance,
                'final_value': final_value,
                'total_return': total_return,
                'return_percentage': return_percentage,
                'target_met': target_met,
                'trades': self.trades,
                'daily_values': self.daily_values,
                'performance_metrics': self.performance_metrics,
                'costs': {
                    'fee_cost': fee_cost,
                    'slippage_cost': slip_cost,
                    'total_drag': total_cost,
                    'fee_rate': self.trading_costs.get('fee_rate', 0.0),
                    'slippage_bps': self.trading_costs.get('slippage_bps', 0.0)
                }
            }
            return results
        except Exception as e:
            self.logger.error(f"Range backtest failed: {e}")
            return {}

    def walk_forward_backtest(self, total_days: int, train_days: int, test_days: int) -> Dict:
        """Perform a walk-forward evaluation by rolling train/test windows across total_days.
        We do not fit parameters; the 'train' period is used only to define the split and avoid lookahead.
        Each test window starts with fresh capital and results are compounded across windows."""
        try:
            # Ensure cache covers the entire span
            self.cache_historical_data(total_days)

            end_all = datetime.now(pytz.UTC)
            start_all = end_all - timedelta(days=total_days)
            full_dates = pd.date_range(start=start_all, end=end_all, freq='B')
            if len(full_dates) == 0:
                raise ValueError("No business days in requested range")

            splits = []
            i = 0
            while True:
                train_start = start_all + timedelta(days=i)
                train_end = train_start + timedelta(days=train_days)
                test_end = train_end + timedelta(days=test_days)
                if test_end > end_all:
                    break
                splits.append((train_start, train_end, train_end, test_end))
                i += test_days

            compounded = 1.0
            window_results = []

            for (tr_start, tr_end, te_start, te_end) in splits:
                # Use a fresh system per test window to avoid state bleed
                sys = Pure5KLiveTradingSystem(initial_balance=self.initial_balance, paper_trading=True)
                # Reuse cached data to avoid re-downloading
                sys.historical_data_cache = getattr(self, 'historical_data_cache', {})
                # Run test-only window
                res = sys.run_backtest_between(te_start, te_end)
                if not res:
                    continue
                rp = float(res.get('return_percentage', 0.0)) / 100.0
                compounded *= (1.0 + rp)
                window_results.append({
                    'train_start': str(tr_start),
                    'train_end': str(tr_end),
                    'test_start': str(te_start),
                    'test_end': str(te_end),
                    'return_pct': rp * 100.0,
                    'final_value': res.get('final_value'),
                    'n_trades': len(res.get('trades', [])),
                    'metrics': {
                        'sharpe': res.get('performance_metrics', {}).get('sharpe_ratio'),
                        'sortino': res.get('performance_metrics', {}).get('sortino_ratio'),
                        'max_dd': res.get('performance_metrics', {}).get('max_drawdown'),
                        'intraday_max_dd': res.get('performance_metrics', {}).get('intraday_max_drawdown')
                    }
                })

            overall = {
                'strategy': 'Pure5KLiveTradingSystem',
                'walk_forward': {
                    'total_days': total_days,
                    'train_days': train_days,
                    'test_days': test_days,
                    'splits': window_results,
                    'compounded_return_pct': (compounded - 1.0) * 100.0
                }
            }
            # Save a summary artifact
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                out_file = f"app/data/results/pure_5k_walk_forward_{timestamp}.json"
                os.makedirs(os.path.dirname(out_file), exist_ok=True)
                with open(out_file, 'w') as f:
                    json.dump(overall, f, indent=2)
                self.logger.info(f"Walk-forward results saved to {out_file}")
            except Exception as save_err:
                self.logger.warning(f"Failed to save walk-forward results: {save_err}")

            # Print a brief summary
            print("\n🧪 WALK-FORWARD SUMMARY")
            print("-" * 60)
            print(f"Splits: {len(window_results)} | Compounded return: {(compounded - 1.0) * 100.0:.2f}%")
            for idx, wr in enumerate(window_results[:5]):
                print(f"  [{idx+1}] Test {wr['test_start']}→{wr['test_end']} | Return: {wr['return_pct']:.2f}% | Trades: {wr['n_trades']}")

            return overall
        except Exception as e:
            self.logger.error(f"Walk-forward backtest failed: {e}")
            return {}

def main():
    """Main execution function"""
    try:
        # Create pure $5K trading system
        system = Pure5KLiveTradingSystem(initial_balance=5000.0)
        
        # Run 45-day backtest
        results = system.run_pure_5k_backtest(days=45)
        
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