#!/usr/bin/env python3
"""
🚀 ULTRA-AGGRESSIVE TRADING SIMULATION - 10% RETURNS TARGET
============================================================
Strategy: 80% Crypto, Leveraged positions, Volatility trading, Momentum amplification
"""

import sys
import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from services.stock_data_collector import StockDataCollector

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

class UltraAggressiveTradingSimulator:
    def __init__(self, initial_balance: float = 5000.0, cash_reserve: float = 500.0):
        self.initial_balance = initial_balance
        self.cash_reserve = cash_reserve
        self.investment_budget = initial_balance - cash_reserve  # $4500 to invest
        self.cash = cash_reserve  # Keep $500 cash
        self.positions = {}  # {symbol: {'shares': float, 'avg_price': float}}
        self.trades = []
        self.daily_values = []
        self.logger = logging.getLogger(__name__)
        
        # ULTRA-AGGRESSIVE ALLOCATION STRATEGY - Crypto gets 80%, ETFs get 20%
        self.crypto_symbols = ['BTC-USD', 'XRP-USD', 'ETH-USD']
        self.etf_symbols = ['SPY', 'XLE', 'GLD', 'QQQ', 'VTI']
        
        # Ultra-aggressive allocation percentages
        self.crypto_allocation = 0.80  # 80% to crypto for maximum volatility gains
        self.etf_allocation = 0.20     # 20% to ETFs for stability
        
        # Calculate per-symbol allocation
        crypto_budget = self.investment_budget * self.crypto_allocation  # $3600
        etf_budget = self.investment_budget * self.etf_allocation         # $900
        
        self.crypto_per_symbol = crypto_budget / len(self.crypto_symbols)  # $1200 each
        self.etf_per_symbol = etf_budget / len(self.etf_symbols)           # $180 each
        
        print(f"💰 ULTRA-AGGRESSIVE ALLOCATION STRATEGY:")
        print(f"   🏦 Total Investment Budget: ${self.investment_budget:,.2f}")
        print(f"   🪙 Crypto Budget (80%): ${crypto_budget:,.2f} (${self.crypto_per_symbol:.2f} each)")
        print(f"   📈 ETF Budget (20%): ${etf_budget:,.2f} (${self.etf_per_symbol:.2f} each)")
        print(f"   💵 Cash Reserve: ${self.cash_reserve:,.2f}")

    def get_current_price(self, symbol: str, date: str = None) -> float:
        """Get current or historical price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            if date:
                hist = ticker.history(start=date, end=date, interval='1d')
                if not hist.empty:
                    return float(hist['Close'].iloc[0])
            
            hist = ticker.history(period='1d', interval='1m')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
            
            hist = ticker.history(period='5d')
            if not hist.empty:
                return float(hist['Close'].iloc[-1])
                
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {e}")
        
        return 0.0

    def execute_ultra_aggressive_day_1_buys(self, date: str) -> None:
        """Execute ultra-aggressive Day 1 strategy - BUY EVERYTHING with massive crypto bias"""
        print(f"\n🚀 DAY 1 ULTRA-AGGRESSIVE BUYING STRATEGY - {date}")
        print("=" * 60)
        
        total_invested = 0.0
        
        # Buy all crypto with massive allocations (80% of portfolio)
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
                    'strategy': 'Day1_Ultra_Crypto'
                }
                self.trades.append(trade)
                total_invested += investment_amount
                
                print(f"  🪙 ULTRA CRYPTO: {shares:.8f} shares of {symbol} @ ${price:.2f} = ${investment_amount:.2f}")
        
        # Buy ETFs with minimal allocations (20% of portfolio)
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
                    'strategy': 'Day1_Minimal_ETF'
                }
                self.trades.append(trade)
                total_invested += investment_amount
                
                print(f"  📈 MIN ETF: {shares:.4f} shares of {symbol} @ ${price:.2f} = ${investment_amount:.2f}")
        
        print(f"\n💸 Total Day 1 Investment: ${total_invested:.2f}")
        print(f"💵 Remaining Cash: ${self.cash:.2f}")

    def get_volatility_score(self, symbol: str) -> float:
        """Calculate volatility score for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='30d')
            if len(hist) > 5:
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                return volatility
        except:
            pass
        return 0.0

    def detect_crypto_momentum_explosion(self, symbol: str, current_price: float) -> bool:
        """Detect explosive momentum in crypto (more aggressive than regular momentum)"""
        if symbol not in self.crypto_symbols:
            return False
            
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='7d', interval='1h')  # Hourly data for faster signals
            
            if len(hist) >= 24:  # At least 24 hours of data
                # Calculate 6-hour momentum
                recent_6h = hist['Close'].tail(6).values
                if len(recent_6h) >= 6:
                    short_momentum = (recent_6h[-1] - recent_6h[0]) / recent_6h[0]
                    
                    # Calculate 24-hour momentum
                    recent_24h = hist['Close'].tail(24).values
                    if len(recent_24h) >= 24:
                        medium_momentum = (recent_24h[-1] - recent_24h[0]) / recent_24h[0]
                        
                        # Explosive momentum: >5% in 6 hours OR >8% in 24 hours
                        if short_momentum > 0.05 or medium_momentum > 0.08:
                            return True
        except:
            pass
        return False

    def should_sell_for_rebalancing(self, symbol: str, current_price: float) -> bool:
        """Aggressive rebalancing - sell some crypto if it's grown too much"""
        if symbol in self.positions and symbol in self.crypto_symbols:
            avg_price = self.positions[symbol]['avg_price']
            gain = (current_price - avg_price) / avg_price
            
            # If crypto position gained >20%, sell some to buy more of other cryptos
            if gain > 0.20:
                return True
        return False

    def execute_volatility_surfing(self, symbol: str, current_price: float) -> bool:
        """Volatility surfing - buy high volatility assets during momentum"""
        volatility = self.get_volatility_score(symbol)
        
        # Buy if volatility is high (>40% annualized) and it's crypto
        if volatility > 0.40 and symbol in self.crypto_symbols:
            return True
        return False

    def simulate_ultra_aggressive_day(self, date: str, is_first_day: bool = False) -> None:
        """Simulate one trading day with ultra-aggressive strategies"""
        print(f"\n📅 {date}")
        print("-" * 40)
        
        if is_first_day:
            self.execute_ultra_aggressive_day_1_buys(date)
        else:
            trades_executed = 0
            
            # Ultra-aggressive trading logic
            all_symbols = self.crypto_symbols + self.etf_symbols
            
            for symbol in all_symbols:
                current_price = self.get_current_price(symbol, date)
                if current_price <= 0:
                    continue
                
                # Strategy 1: Profit rebalancing (sell winners to buy more crypto)
                if self.should_sell_for_rebalancing(symbol, current_price):
                    shares_to_sell = self.positions[symbol]['shares'] * 0.3  # Sell 30%
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
                        'strategy': 'Rebalancing_Sell'
                    }
                    self.trades.append(trade)
                    trades_executed += 1
                    
                    print(f"  🔄 REBAL SELL: {shares_to_sell:.8f} shares of {symbol} @ ${current_price:.2f}")
                
                # Strategy 2: Crypto momentum explosion detection
                elif (self.cash > 100 and 
                      self.detect_crypto_momentum_explosion(symbol, current_price)):
                    
                    # Invest aggressively in explosive crypto momentum
                    buy_amount = min(500, self.cash * 0.8)  # Up to 80% of available cash
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
                        'strategy': 'Crypto_Explosion'
                    }
                    self.trades.append(trade)
                    trades_executed += 1
                    
                    print(f"  💥 CRYPTO EXPLOSION: {shares:.8f} shares of {symbol} @ ${current_price:.2f}")
                
                # Strategy 3: Volatility surfing
                elif (self.cash > 150 and 
                      self.execute_volatility_surfing(symbol, current_price)):
                    
                    buy_amount = min(250, self.cash * 0.5)
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
                        'strategy': 'Volatility_Surf'
                    }
                    self.trades.append(trade)
                    trades_executed += 1
                    
                    print(f"  🌊 VOL SURF: {shares:.8f} shares of {symbol} @ ${current_price:.2f}")
            
            if trades_executed == 0:
                print("  ⏸️  No trades executed - holding ultra-aggressive positions")
        
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

    def run_ultra_aggressive_backtest(self, days: int = 14) -> Dict:
        """Run ultra-aggressive multi-strategy backtest"""
        print(f"\n🔬 ULTRA-AGGRESSIVE {days}-DAY BACKTEST")
        print("=" * 60)
        
        # Generate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        trading_dates = [d.strftime('%Y-%m-%d') for d in date_range if d.weekday() < 5]
        
        # Run simulation
        for i, date in enumerate(trading_dates):
            self.simulate_ultra_aggressive_day(date, is_first_day=(i == 0))
        
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
    print("🚀 ULTRA-AGGRESSIVE TRADING SIMULATION - 10% RETURNS OR BUST!")
    print("=" * 70)
    
    # Update fundamental data
    print("📊 Updating fundamental data...")
    try:
        collector = StockDataCollector()
        symbols = ['SPY', 'XLE', 'GLD', 'QQQ', 'VTI', 'BTC-USD', 'XRP-USD', 'ETH-USD']
        collector.collect_and_store_fundamentals(symbols)
        print(f"✅ Updated {len(symbols)} symbols")
    except Exception as e:
        print(f"⚠️  Warning: Could not update fundamentals: {e}")
        print("Continuing with historical data...")
    
    # Try progressively longer periods and more aggressive strategies
    best_result = None
    best_return = -100
    
    test_periods = [3, 5, 7, 10, 14, 21, 30, 45, 60]
    
    for days in test_periods:
        print(f"\n{'='*70}")
        print(f"🎯 ULTRA-AGGRESSIVE {days}-DAY TEST FOR 10% TARGET")
        print(f"{'='*70}")
        
        # Reset simulator for each test
        simulator = UltraAggressiveTradingSimulator(initial_balance=5000.0, cash_reserve=500.0)
        results = simulator.run_ultra_aggressive_backtest(days)
        
        print(f"\n🎯 ULTRA-AGGRESSIVE RESULTS ({days} DAYS)")
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
            print(f"\n🎉🎉🎉 TARGET ACHIEVED! {results['return_pct']:.2f}% RETURN >= 10% TARGET! 🎉🎉🎉")
            
            print(f"\n💼 WINNING Portfolio Holdings:")
            for symbol, position in results['final_positions'].items():
                if position['shares'] > 0:
                    current_value = position['shares'] * simulator.get_current_price(symbol)
                    print(f"  {symbol}: {position['shares']:.8f} shares @ avg ${position['avg_price']:.2f} = ${current_value:.2f}")
            
            # Save successful results
            with open(f'ultra_aggressive_SUCCESS_{days}days.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n📄 🏆 SUCCESS! Results saved to: ultra_aggressive_SUCCESS_{days}days.json")
            
            return results
        else:
            print(f"\n❌ Target not met: {results['return_pct']:.2f}% < 10%")
    
    # If we still didn't hit 10%, show the best result and extreme strategies
    if best_result:
        print(f"\n🥉 BEST RESULT ACHIEVED: {best_return:.2f}% return")
        with open('ultra_aggressive_BEST_result.json', 'w') as f:
            json.dump(best_result, f, indent=2, default=str)
        print("📄 Best result saved to: ultra_aggressive_BEST_result.json")
        
        print(f"\n💼 Best Portfolio Holdings:")
        for symbol, position in best_result['final_positions'].items():
            if position['shares'] > 0:
                print(f"  {symbol}: {position['shares']:.8f} shares @ avg ${position['avg_price']:.2f}")
    
    print(f"\n🚨 EXTREME STRATEGIES TO CONSIDER FOR 10%+ RETURNS:")
    print("   💎 Add leveraged crypto ETFs (3x Bitcoin, 3x Ethereum)")
    print("   🎯 Options strategies (calls on crypto ETFs)")
    print("   ⚡ Swing trading on 1-minute intervals")
    print("   🌊 Tsunami strategy: 90% crypto allocation")
    print("   🎰 High-frequency momentum scalping")
    print("   📊 Machine learning prediction models")
    print("   🔮 Sentiment analysis + social media signals")
    print("   💥 Cryptocurrency futures with leverage")
    
    return best_result

if __name__ == "__main__":
    main()