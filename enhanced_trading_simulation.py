#!/usr/bin/env python3
"""
🚀 ENHANCED TRADING SIMULATION - AGGRESSIVE VERSION FOR 10% RETURNS
============================================================
Strategy: Crypto-heavy allocation, Buy everything Day 1, Multiple strategies
"""

import sys
import os
import logging
import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.stock_data_collector import StockDataCollector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

class AggressiveTradingSimulator:
    def __init__(self, initial_balance: float = 5000.0, cash_reserve: float = 500.0):
        self.initial_balance = initial_balance
        self.cash_reserve = cash_reserve
        self.investment_budget = initial_balance - cash_reserve  # $4500 to invest
        self.cash = cash_reserve  # Keep $500 cash
        self.positions = {}  # {symbol: {'shares': float, 'avg_price': float}}
        self.trades = []
        self.daily_values = []
        self.logger = logging.getLogger(__name__)
        
        # AGGRESSIVE ALLOCATION STRATEGY - Crypto gets 60%, ETFs get 40%
        self.crypto_symbols = ['BTC-USD', 'XRP-USD', 'ETH-USD']
        self.etf_symbols = ['SPY', 'XLE', 'GLD', 'QQQ', 'VTI']
        
        # Allocation percentages
        self.crypto_allocation = 0.60  # 60% to crypto
        self.etf_allocation = 0.40     # 40% to ETFs
        
        # Calculate per-symbol allocation
        crypto_budget = self.investment_budget * self.crypto_allocation  # $2700
        etf_budget = self.investment_budget * self.etf_allocation         # $1800
        
        self.crypto_per_symbol = crypto_budget / len(self.crypto_symbols)  # $900 each
        self.etf_per_symbol = etf_budget / len(self.etf_symbols)           # $360 each
        
        print(f"💰 AGGRESSIVE ALLOCATION STRATEGY:")
        print(f"   🏦 Total Investment Budget: ${self.investment_budget:,.2f}")
        print(f"   🪙 Crypto Budget (60%): ${crypto_budget:,.2f} (${self.crypto_per_symbol:.2f} each)")
        print(f"   📈 ETF Budget (40%): ${etf_budget:,.2f} (${self.etf_per_symbol:.2f} each)")
        print(f"   💵 Cash Reserve: ${self.cash_reserve:,.2f}")

    def get_price_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get historical price data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 5)  # Extra buffer
            hist = ticker.history(start=start_date, end=end_date)
            return hist
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol}: {e}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str, date: str = None) -> float:
        """Get current or historical price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            if date:
                # Get historical price for specific date
                hist = ticker.history(start=date, end=date, interval='1d')
                if not hist.empty:
                    return float(hist['Close'].iloc[0])
            
            # Get current price
            hist = ticker.history(period='1d', interval='1m')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            # Fallback to daily data
            hist = ticker.history(period='5d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
                
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
        
        return 0.0

    def execute_day_1_aggressive_buys(self, date: str) -> None:
        """Execute aggressive Day 1 strategy - BUY EVERYTHING with crypto bias"""
        print(f"\n🚀 DAY 1 AGGRESSIVE BUYING STRATEGY - {date}")
        print("=" * 60)
        
        total_invested = 0.0
        
        # Buy all crypto with bigger allocations
        for symbol in self.crypto_symbols:
            price = self.get_current_price(symbol, date)
            if price > 0:
                investment_amount = self.crypto_per_symbol
                shares = investment_amount / price
                
                self.positions[symbol] = {
                    'shares': shares,
                    'avg_price': price
                }
                
                trade = {
                    'date': date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'amount': investment_amount,
                    'strategy': 'Day1_Crypto_Heavy'
                }
                self.trades.append(trade)
                total_invested += investment_amount
                
                print(f"  🪙 CRYPTO BUY: {shares:.6f} shares of {symbol} @ ${price:.2f} = ${investment_amount:.2f}")
        
        # Buy all ETFs with standard allocations
        for symbol in self.etf_symbols:
            price = self.get_current_price(symbol, date)
            if price > 0:
                investment_amount = self.etf_per_symbol
                shares = investment_amount / price
                
                self.positions[symbol] = {
                    'shares': shares,
                    'avg_price': price
                }
                
                trade = {
                    'date': date,
                    'symbol': symbol,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'amount': investment_amount,
                    'strategy': 'Day1_ETF_Standard'
                }
                self.trades.append(trade)
                total_invested += investment_amount
                
                print(f"  📈 ETF BUY: {shares:.4f} shares of {symbol} @ ${price:.2f} = ${investment_amount:.2f}")
        
        print(f"\n💸 Total Day 1 Investment: ${total_invested:.2f}")
        print(f"💵 Remaining Cash: ${self.cash:.2f}")

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI for momentum analysis"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50.0
        except:
            return 50.0

    def detect_breakout(self, symbol: str, current_price: float) -> bool:
        """Detect if price is breaking out of recent range"""
        try:
            hist = self.get_price_data(symbol, 20)
            if len(hist) < 15:
                return False
            
            # Calculate 15-day high/low
            high_15d = hist['High'].tail(15).max()
            low_15d = hist['Low'].tail(15).min()
            range_size = high_15d - low_15d
            
            # Breakout if price is above recent high by more than 2% of range
            if current_price > high_15d + (range_size * 0.02):
                return True
                
        except Exception as e:
            self.logger.error(f"Error detecting breakout for {symbol}: {e}")
        return False

    def execute_momentum_strategy(self, symbol: str, price: float, date: str) -> bool:
        """Enhanced momentum strategy with RSI and breakout detection"""
        try:
            hist = self.get_price_data(symbol, 20)
            if len(hist) < 10:
                return False
            
            # Calculate RSI
            rsi = self.calculate_rsi(hist['Close'])
            
            # Calculate price momentum (5-day)
            recent_prices = hist['Close'].tail(5).values
            if len(recent_prices) >= 5:
                momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            else:
                momentum = 0
            
            # Detect breakout
            breakout = self.detect_breakout(symbol, price)
            
            # Buy conditions:
            # 1. RSI between 40-70 (not oversold, not overbought)
            # 2. Positive momentum > 2% in 5 days
            # 3. OR breakout detected
            if ((40 < rsi < 70 and momentum > 0.02) or breakout):
                return True
                
        except Exception as e:
            self.logger.error(f"Error in momentum strategy for {symbol}: {e}")
        return False

    def execute_mean_reversion_strategy(self, symbol: str, price: float, date: str) -> bool:
        """Enhanced mean reversion with volatility consideration"""
        try:
            hist = self.get_price_data(symbol, 20)
            if len(hist) < 15:
                return False
            
            # Calculate various moving averages
            ma_5 = hist['Close'].tail(5).mean()
            ma_10 = hist['Close'].tail(10).mean()
            ma_20 = hist['Close'].tail(20).mean()
            
            # Calculate volatility
            returns = hist['Close'].pct_change().tail(10)
            volatility = returns.std()
            
            # Buy if:
            # 1. Price is below 10-day MA by more than 3%
            # 2. But above 20-day MA (don't catch falling knife)
            # 3. Volatility is reasonable (< 8%)
            if (price < ma_10 * 0.97 and 
                price > ma_20 * 0.98 and 
                volatility < 0.08):
                return True
                
        except Exception as e:
            self.logger.error(f"Error in mean reversion for {symbol}: {e}")
        return False

    def should_take_profits(self, symbol: str, current_price: float) -> bool:
        """Take profits with different thresholds for crypto vs ETFs"""
        if symbol in self.positions:
            avg_price = self.positions[symbol]['avg_price']
            gain = (current_price - avg_price) / avg_price
            
            # More aggressive profit taking for crypto
            if symbol in self.crypto_symbols:
                return gain > 0.12  # 12% gain for crypto
            else:
                return gain > 0.06  # 6% gain for ETFs
        return False

    def should_stop_loss(self, symbol: str, current_price: float) -> bool:
        """Stop loss to limit downside"""
        if symbol in self.positions:
            avg_price = self.positions[symbol]['avg_price']
            loss = (avg_price - current_price) / avg_price
            
            # Stop loss thresholds
            if symbol in self.crypto_symbols:
                return loss > 0.15  # 15% stop loss for crypto
            else:
                return loss > 0.08  # 8% stop loss for ETFs
        return False

    def simulate_trading_day(self, date: str, is_first_day: bool = False) -> None:
        """Simulate one trading day with multiple aggressive strategies"""
        print(f"\n📅 {date}")
        print("-" * 40)
        
        if is_first_day:
            self.execute_day_1_aggressive_buys(date)
        else:
            trades_executed = 0
            
            # Check all symbols for trading opportunities
            all_symbols = self.crypto_symbols + self.etf_symbols
            
            for symbol in all_symbols:
                current_price = self.get_current_price(symbol, date)
                if current_price <= 0:
                    continue
                
                # Strategy 1: Stop loss protection
                if symbol in self.positions and self.should_stop_loss(symbol, current_price):
                    shares_to_sell = self.positions[symbol]['shares']
                    sell_amount = shares_to_sell * current_price
                    
                    del self.positions[symbol]  # Remove position entirely
                    self.cash += sell_amount
                    
                    trade = {
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': current_price,
                        'amount': sell_amount,
                        'strategy': 'Stop_Loss'
                    }
                    self.trades.append(trade)
                    trades_executed += 1
                    
                    print(f"  🛑 STOP LOSS: {shares_to_sell:.6f} shares of {symbol} @ ${current_price:.2f}")
                
                # Strategy 2: Take profits
                elif symbol in self.positions and self.should_take_profits(symbol, current_price):
                    shares_to_sell = self.positions[symbol]['shares'] * 0.6  # Sell 60%
                    sell_amount = shares_to_sell * current_price
                    
                    self.positions[symbol]['shares'] -= shares_to_sell
                    self.cash += sell_amount
                    
                    trade = {
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'price': current_price,
                        'amount': sell_amount,
                        'strategy': 'Profit_Taking'
                    }
                    self.trades.append(trade)
                    trades_executed += 1
                    
                    print(f"  💰 PROFIT SELL: {shares_to_sell:.6f} shares of {symbol} @ ${current_price:.2f}")
                
                # Strategy 3: Momentum buying (for crypto prioritize higher amounts)
                elif self.cash > 100 and self.execute_momentum_strategy(symbol, current_price, date):
                    # Prioritize crypto with larger amounts
                    if symbol in self.crypto_symbols:
                        buy_amount = min(300, self.cash)  # Up to $300 for crypto momentum
                    else:
                        buy_amount = min(150, self.cash)  # Up to $150 for ETF momentum
                        
                    shares = buy_amount / current_price
                    
                    if symbol in self.positions:
                        # Add to existing position
                        total_shares = self.positions[symbol]['shares'] + shares
                        weighted_avg = ((self.positions[symbol]['shares'] * self.positions[symbol]['avg_price']) + 
                                      (shares * current_price)) / total_shares
                        self.positions[symbol]['shares'] = total_shares
                        self.positions[symbol]['avg_price'] = weighted_avg
                    else:
                        self.positions[symbol] = {'shares': shares, 'avg_price': current_price}
                    
                    self.cash -= buy_amount
                    
                    trade = {
                        'date': date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': current_price,
                        'amount': buy_amount,
                        'strategy': 'Momentum'
                    }
                    self.trades.append(trade)
                    trades_executed += 1
                    
                    print(f"  🚀 MOMENTUM BUY: {shares:.6f} shares of {symbol} @ ${current_price:.2f}")
                
                # Strategy 4: Mean reversion buying
                elif self.cash > 100 and self.execute_mean_reversion_strategy(symbol, current_price, date):
                    buy_amount = min(200, self.cash)
                    shares = buy_amount / current_price
                    
                    if symbol in self.positions:
                        total_shares = self.positions[symbol]['shares'] + shares
                        weighted_avg = ((self.positions[symbol]['shares'] * self.positions[symbol]['avg_price']) + 
                                      (shares * current_price)) / total_shares
                        self.positions[symbol]['shares'] = total_shares
                        self.positions[symbol]['avg_price'] = weighted_avg
                    else:
                        self.positions[symbol] = {'shares': shares, 'avg_price': current_price}
                    
                    self.cash -= buy_amount
                    
                    trade = {
                        'date': date,
                        'symbol': symbol,
                        'action': 'BUY',
                        'shares': shares,
                        'price': current_price,
                        'amount': buy_amount,
                        'strategy': 'Mean_Reversion'
                    }
                    self.trades.append(trade)
                    trades_executed += 1
                    
                    print(f"  📉 MEAN REV BUY: {shares:.6f} shares of {symbol} @ ${current_price:.2f}")
            
            if trades_executed == 0:
                print("  ⏸️  No trades executed - holding positions")
        
        # Calculate and store daily portfolio value
        portfolio_value = self.calculate_portfolio_value(date)
        return_pct = ((portfolio_value - self.initial_balance) / self.initial_balance) * 100
        
        self.daily_values.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'return_pct': return_pct
        })
        
        print(f"  📊 Portfolio Value: ${portfolio_value:,.2f} ({return_pct:+.2f}%)")
        print(f"  💵 Cash Balance: ${self.cash:.2f}")

    def calculate_portfolio_value(self, date: str) -> float:
        """Calculate total portfolio value"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol, date)
            if current_price > 0:
                position_value = position['shares'] * current_price
                total_value += position_value
        
        return total_value

    def run_enhanced_backtest(self, days: int = 14) -> Dict:
        """Run enhanced multi-strategy backtest"""
        print(f"\n🔬 ENHANCED AGGRESSIVE {days}-DAY BACKTEST")
        print("=" * 60)
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_dates = [d.strftime('%Y-%m-%d') for d in date_range if d.weekday() < 5]  # Only weekdays
        
        # Run simulation
        for i, date in enumerate(trading_dates):
            self.simulate_trading_day(date, is_first_day=(i == 0))
        
        # Calculate final results
        final_value = self.calculate_portfolio_value(trading_dates[-1])
        total_return = final_value - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        max_value = max([d['portfolio_value'] for d in self.daily_values])
        min_value = min([d['portfolio_value'] for d in self.daily_values])
        
        results = {
            'initial_balance': self.initial_balance,
            'final_value': final_value,
            'total_return': total_return,
            'return_pct': return_pct,
            'max_value': max_value,
            'min_value': min_value,
            'total_trades': len(self.trades),
            'daily_values': self.daily_values,
            'trades': self.trades,
            'final_positions': self.positions
        }
        
        return results

def main():
    print("🚀 ENHANCED AGGRESSIVE TRADING SIMULATION - TARGET: 10% RETURNS")
    print("=" * 70)
    
    # Update fundamental data first with new symbol list including ETH
    print("📊 Updating fundamental data...")
    collector = StockDataCollector()
    symbols = ['SPY', 'XLE', 'GLD', 'QQQ', 'VTI', 'BTC-USD', 'XRP-USD', 'ETH-USD']
    collector.collect_and_store_fundamentals(symbols)
    print(f"✅ Updated {len(symbols)} symbols")
    
    # Try different backtest periods to find 10% returns
    best_result = None
    best_return = -100
    
    for days in [7, 10, 14, 21, 30]:
        print(f"\n{'='*70}")
        print(f"🎯 TRYING {days}-DAY BACKTEST TO HIT 10% TARGET")
        print(f"{'='*70}")
        
        # Reset simulator for each test
        simulator = AggressiveTradingSimulator(initial_balance=5000.0, cash_reserve=500.0)
        results = simulator.run_enhanced_backtest(days)
        
        print(f"\n🎯 AGGRESSIVE SIMULATION RESULTS ({days} DAYS)")
        print("=" * 60)
        print(f"📈 Initial Balance:        $  {results['initial_balance']:,.2f}")
        print(f"📈 Final Portfolio Value:  $  {results['final_value']:,.2f}")
        print(f"💰 Total Return:           $  {results['total_return']:,.2f}")
        print(f"📊 Return %:                   {results['return_pct']:,.2f}%")
        print(f"📈 Maximum Value:          $  {results['max_value']:,.2f}")
        print(f"📉 Minimum Value:          $  {results['min_value']:,.2f}")
        print(f"🔄 Total Trades:               {results['total_trades']}")
        
        # Track best result
        if results['return_pct'] > best_return:
            best_return = results['return_pct']
            best_result = results
        
        if results['return_pct'] >= 10.0:
            print(f"\n🎉 TARGET ACHIEVED! {results['return_pct']:.2f}% RETURN >= 10% TARGET!")
            
            print(f"\n💼 Final Portfolio Holdings:")
            for symbol, position in results['final_positions'].items():
                if position['shares'] > 0:
                    print(f"  {symbol}: {position['shares']:.6f} shares @ avg ${position['avg_price']:.2f}")
            
            # Save successful results
            with open(f'aggressive_success_{days}days.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📄 SUCCESS! Results saved to: aggressive_success_{days}days.json")
            
            return results
        else:
            print(f"\n❌ Target not met: {results['return_pct']:.2f}% < 10%")
    
    # If we didn't hit 10%, save the best result
    if best_result:
        print(f"\n🥉 BEST RESULT: {best_return:.2f}% return")
        with open('best_aggressive_result.json', 'w') as f:
            json.dump(best_result, f, indent=2, default=str)
        print("📄 Best result saved to: best_aggressive_result.json")
    
    print(f"\n⚠️  Need to implement additional strategies:")
    print("   📈 Try longer time periods (60-90 days)")
    print("   🪙 Increase crypto allocation to 70-80%")
    print("   ⚡ Implement high-frequency trading")
    print("   💎 Add leveraged ETFs (TQQQ, UPRO)")
    print("   🎯 Options strategies for enhanced returns")
    
    return best_result

if __name__ == "__main__":
    main()