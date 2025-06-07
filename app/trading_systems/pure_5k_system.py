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

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

class Pure5KTradingSystem:
    def __init__(self, initial_balance: float = 5000.0):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.positions = {}  # {symbol: {'shares': float, 'avg_price': float}}
        self.trades = []
        self.daily_values = []
        self.historical_data_cache = {}  # Offline data storage
        self.logger = logging.getLogger(__name__)
        
        # EXPANDED CRYPTO UNIVERSE - 7 cryptos
        self.crypto_symbols = [
            'BTC-USD', 'XRP-USD', 'ETH-USD',  # Original 3
            'SOL-USD', 'TRX-USD', 'ADA-USD', 'XLM-USD'  # New 4
        ]
        
        # EXPANDED STOCK UNIVERSE - Energy, Tech, ETFs
        self.energy_stocks = [
            'XLE',    # Energy ETF
            'XEG',    # Energy Equipment ETF  
            'KOLD',   # Natural Gas Bear ETF
            'UNG',    # Natural Gas ETF
            'USO',    # Oil ETF
            'NEE',    # NextEra Energy (Electrical)
            'DUK',    # Duke Energy (Electrical)
        ]
        
        self.tech_stocks = [
            'QQQ',    # Tech ETF
            'NVDA',   # NVIDIA
            'MSFT',   # Microsoft  
            'GOOGL',  # Google
            'TSLA',   # Tesla
            'AMD',    # AMD
        ]
        
        self.etf_symbols = [
            'SPY',    # S&P 500
            'VTI',    # Total Market
            'GLD',    # Gold
        ]
        
        self.all_symbols = self.crypto_symbols + self.energy_stocks + self.tech_stocks + self.etf_symbols
        
        # ULTRA-AGGRESSIVE ALLOCATION STRATEGY
        self.crypto_allocation = 0.70     # 70% crypto
        self.energy_allocation = 0.15     # 15% energy sector
        self.tech_allocation = 0.10       # 10% tech sector  
        self.etf_allocation = 0.05        # 5% ETFs for stability
        
        # Market timezone handling
        self.market_tz = pytz.timezone('America/New_York')
        self.utc = pytz.UTC
        
        print(f"💰 PURE $5K ULTRA-AGGRESSIVE TRADING SYSTEM:")
        print(f"   💵 Initial Capital: ${self.initial_balance:,.2f} (NO DAILY ADDITIONS)")
        print(f"   🪙 Crypto Allocation: {self.crypto_allocation:.0%} ({len(self.crypto_symbols)} symbols)")
        print(f"   ⚡ Energy Allocation: {self.energy_allocation:.0%} ({len(self.energy_stocks)} symbols)")
        print(f"   💻 Tech Allocation: {self.tech_allocation:.0%} ({len(self.tech_stocks)} symbols)")
        print(f"   📈 ETF Allocation: {self.etf_allocation:.0%} ({len(self.etf_symbols)} symbols)")
        print(f"   📊 Total Symbols: {len(self.all_symbols)}")

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
                    self.logger.debug(f"Period {period} failed for {symbol}: {period_error}")
                    continue
                    
        except Exception as e:
            self.logger.debug(f"yfinance failed for {symbol}: {e}")
        
        return 0.0

    def detect_market_momentum_signals(self, date: str) -> Dict[str, str]:
        """Detect various momentum signals across all assets with FIXED data handling"""
        signals = {}
        
        for symbol in self.all_symbols:
            try:
                if symbol in self.historical_data_cache:
                    cache_entry = self.historical_data_cache[symbol]
                    data = cache_entry['data']
                    
                    # FIXED: Get data up to the target date with proper handling
                    target_date = self.standardize_datetime(date)
                    
                    if len(data) > 0:
                        # Filter data up to target date
                        try:
                            recent_data = data[data.index <= target_date]
                        except Exception as filter_error:
                            # Fallback: just use all data
                            recent_data = data
                        
                        if len(recent_data) > 24:  # Need at least 24 periods
                            
                            # Short-term momentum (6 periods)
                            try:
                                recent_6 = recent_data['Close'].tail(6)
                                if len(recent_6) >= 2:
                                    momentum_6 = (recent_6.iloc[-1] - recent_6.iloc[0]) / recent_6.iloc[0]
                                else:
                                    momentum_6 = 0
                            except:
                                momentum_6 = 0
                            
                            # Medium-term momentum (24 periods) 
                            try:
                                recent_24 = recent_data['Close'].tail(24)
                                if len(recent_24) >= 2:
                                    momentum_24 = (recent_24.iloc[-1] - recent_24.iloc[0]) / recent_24.iloc[0]
                                else:
                                    momentum_24 = 0
                            except:
                                momentum_24 = 0
                            
                            # Classify signals
                            if momentum_6 > 0.05:  # >5% in 6 periods
                                signals[symbol] = "EXPLOSIVE_UP"
                            elif momentum_6 < -0.05:  # <-5% in 6 periods
                                signals[symbol] = "EXPLOSIVE_DOWN"
                            elif momentum_24 > 0.08:  # >8% in 24 periods
                                signals[symbol] = "STRONG_UP"
                            elif momentum_24 < -0.08:  # <-8% in 24 periods
                                signals[symbol] = "STRONG_DOWN"
                            elif momentum_24 > 0.03:  # >3% in 24 periods
                                signals[symbol] = "MODERATE_UP"
                            else:
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
        """Helper to record trades"""
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

    def simulate_pure_trading_day(self, date: str, is_first_day: bool = False) -> None:
        """Simulate one pure trading day - no daily additions, just trading"""
        print(f"\n📅 {date}")
        print("-" * 50)
        
        if is_first_day:
            self.execute_day_1_intelligent_allocation(date)
        else:
            # Detect market signals
            signals = self.detect_market_momentum_signals(date)
            
            # Execute trades based on signals (using only existing cash)
            trades_executed = 0
            
            for symbol, signal in signals.items():
                current_price = self.get_price_from_cache(symbol, date)
                if current_price <= 0:
                    continue
                
                # EXPLOSIVE UP signals - buy aggressively (if we have cash)
                if signal == "EXPLOSIVE_UP" and self.cash > 200:
                    category = self.positions.get(symbol, {}).get('category', 'unknown')
                    buy_amount = min(300 if category == 'crypto' else 200, self.cash * 0.4)
                    shares = buy_amount / current_price
                    
                    self._add_to_position(symbol, shares, current_price, category)
                    self.cash -= buy_amount
                    
                    self._record_trade(date, symbol, 'BUY', shares, current_price, 
                                     buy_amount, 'Explosive_Momentum', f"6h surge detected")
                    trades_executed += 1
                    
                    print(f"  💥 EXPLOSIVE BUY: {shares:.6f} {symbol} @ ${current_price:.4f}")
                
                # STRONG UP signals - buy moderately (if we have cash)
                elif signal == "STRONG_UP" and self.cash > 150:
                    category = self.positions.get(symbol, {}).get('category', 'unknown')
                    buy_amount = min(200 if category == 'crypto' else 150, self.cash * 0.3)
                    shares = buy_amount / current_price
                    
                    self._add_to_position(symbol, shares, current_price, category)
                    self.cash -= buy_amount
                    
                    self._record_trade(date, symbol, 'BUY', shares, current_price,
                                     buy_amount, 'Strong_Momentum', f"24h strength detected")
                    trades_executed += 1
                    
                    print(f"  🚀 STRONG BUY: {shares:.6f} {symbol} @ ${current_price:.4f}")
                
                # Take profits on explosive down moves
                elif signal in ["EXPLOSIVE_DOWN", "STRONG_DOWN"] and symbol in self.positions:
                    if self.positions[symbol]['shares'] > 0:
                        # Sell 40% of position to preserve capital
                        shares_to_sell = self.positions[symbol]['shares'] * 0.4
                        sell_amount = shares_to_sell * current_price
                        
                        self.positions[symbol]['shares'] -= shares_to_sell
                        self.cash += sell_amount
                        
                        self._record_trade(date, symbol, 'SELL', shares_to_sell, current_price,
                                         sell_amount, 'Risk_Management', f"Downtrend protection")
                        trades_executed += 1
                        
                        print(f"  🛡️  PROTECTIVE SELL: {shares_to_sell:.6f} {symbol} @ ${current_price:.4f}")
            
            if trades_executed == 0:
                print("  ⏸️  No trades executed - market in neutral zone")
        
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
        """Run pure $5K backtest with no daily additions"""
        print(f"\n🎯 PURE $5K BACKTEST ({days} DAYS)")
        print("=" * 60)
        print(f"💰 Starting with ${self.initial_balance:,.2f} - NO DAILY ADDITIONS")
        print(f"🎯 Target: 10% returns through pure trading skill")
        
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
        print(f"\n🎯 PURE $5K RESULTS ({days} DAYS)")
        print("=" * 60)
        print(f"📈 Initial Balance:        $  {self.initial_balance:,.2f}")
        print(f"📈 Final Portfolio Value:  $  {final_value:,.2f}")
        print(f"💰 Total Return:           $    {total_return:,.2f}")
        print(f"📊 Return %:                    {return_pct:.2f}%")
        print(f"📈 Maximum Value:          $  {max_value:,.2f}")
        print(f"📉 Minimum Value:          $  {min_value:,.2f}")
        print(f"🔄 Total Trades:               {total_trades}")
        print(f"📅 Trading Days:               {len(self.daily_values)}")
        
        if return_pct >= 10.0:
            print(f"\n🎉 TARGET MET! {return_pct:.2f}% >= 10% TARGET!")
        else:
            print(f"\n❌ Target not met: {return_pct:.2f}% < 10%")
        
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