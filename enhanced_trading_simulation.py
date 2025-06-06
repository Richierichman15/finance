#!/usr/bin/env python3
from app.services.trading_simulator import TradingSimulator
from app.services.stock_data_collector import StockDataCollector
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json

class EnhancedTradingSimulator(TradingSimulator):
    def __init__(self, initial_balance: float = 5000.0):
        super().__init__(initial_balance)
        self.daily_portfolio_values = []
        
    def run_backtest_simulation(self, days: int = 7) -> dict:
        """
        Run a backtest simulation over multiple days to show performance
        """
        print(f"🔬 Running {days}-day backtest simulation...")
        print("=" * 60)
        
        # Collect historical data for backtesting
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days+5)  # Extra days for data
        
        historical_prices = {}
        for symbol in self.collector.tracked_stocks:
            try:
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(start=start_date, end=end_date)
                if not hist_data.empty:
                    historical_prices[symbol] = hist_data
                    print(f"📈 Loaded {len(hist_data)} days of data for {symbol}")
            except Exception as e:
                print(f"❌ Failed to load data for {symbol}: {e}")
        
        # Get fundamentals for initial decisions
        fundamentals_data = {}
        for symbol in self.collector.tracked_stocks:
            fund_data = self.get_latest_fundamentals(symbol)
            if fund_data:
                fundamentals_data[symbol] = fund_data
        
        # Simulate trading over each day
        simulation_days = []
        available_dates = list(historical_prices[list(historical_prices.keys())[0]].index[-days:])
        
        for i, current_date in enumerate(available_dates):
            day_info = {
                "date": current_date.strftime("%Y-%m-%d"),
                "day": i + 1,
                "actions": [],
                "portfolio_value": 0,
                "cash_balance": self.current_balance
            }
            
            print(f"\n📅 Day {i+1}: {current_date.strftime('%Y-%m-%d')}")
            print("-" * 40)
            
            # Check each symbol for trading opportunities
            for symbol in historical_prices.keys():
                if current_date not in historical_prices[symbol].index:
                    continue
                    
                current_price = historical_prices[symbol].loc[current_date]['Close']
                fund_data = fundamentals_data.get(symbol)
                
                if not fund_data:
                    continue
                
                # Update the price in fundamentals for current day
                fund_data['price'] = current_price
                
                # Trading logic
                if symbol not in self.portfolio and self.should_buy(symbol, fund_data):
                    if self.execute_buy(symbol, current_price, f"Day {i+1} buy signal"):
                        day_info["actions"].append(f"BUY {symbol} @ ${current_price:.2f}")
                        print(f"  ✅ BOUGHT {symbol} @ ${current_price:.2f}")
                
                elif symbol in self.portfolio and self.should_sell(symbol, fund_data):
                    if self.execute_sell(symbol, current_price, f"Day {i+1} sell signal"):
                        day_info["actions"].append(f"SELL {symbol} @ ${current_price:.2f}")
                        print(f"  💰 SOLD {symbol} @ ${current_price:.2f}")
            
            # Calculate portfolio value for this day
            portfolio_value = self.current_balance
            for symbol, position in self.portfolio.items():
                if current_date in historical_prices[symbol].index:
                    current_price = historical_prices[symbol].loc[current_date]['Close']
                    position_value = position["shares"] * current_price
                    portfolio_value += position_value
            
            day_info["portfolio_value"] = portfolio_value
            day_info["cash_balance"] = self.current_balance
            day_info["return_pct"] = ((portfolio_value - self.initial_balance) / self.initial_balance) * 100
            
            self.daily_portfolio_values.append(portfolio_value)
            simulation_days.append(day_info)
            
            print(f"  📊 Portfolio Value: ${portfolio_value:,.2f} ({day_info['return_pct']:+.2f}%)")
            print(f"  💵 Cash Balance: ${self.current_balance:,.2f}")
            
            if not day_info["actions"]:
                print(f"  ⏸️  No trades executed")
        
        # Final results
        final_portfolio_value = self.daily_portfolio_values[-1] if self.daily_portfolio_values else self.initial_balance
        total_return = final_portfolio_value - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        
        results = {
            "simulation_days": simulation_days,
            "initial_balance": self.initial_balance,
            "final_portfolio_value": final_portfolio_value,
            "total_return": total_return,
            "return_percentage": return_pct,
            "total_trades": len(self.trade_history),
            "current_portfolio": self.portfolio,
            "daily_values": self.daily_portfolio_values,
            "max_value": max(self.daily_portfolio_values) if self.daily_portfolio_values else self.initial_balance,
            "min_value": min(self.daily_portfolio_values) if self.daily_portfolio_values else self.initial_balance
        }
        
        return results

def main():
    print("🚀 ENHANCED TRADING SIMULATION")
    print("=" * 60)
    
    # Update data first
    print("📊 Updating fundamental data...")
    collector = StockDataCollector()
    data_results = collector.collect_and_store_fundamentals()
    print(f"✅ Updated {len(data_results['symbols_processed'])} symbols\n")
    
    # Run enhanced simulation
    simulator = EnhancedTradingSimulator(initial_balance=5000.0)
    results = simulator.run_backtest_simulation(days=7)
    
    # Display results
    print("\n" + "=" * 60)
    print("🎯 ENHANCED SIMULATION RESULTS") 
    print("=" * 60)
    
    print(f"📈 Initial Balance:        ${results['initial_balance']:>10,.2f}")
    print(f"📈 Final Portfolio Value:  ${results['final_portfolio_value']:>10,.2f}")
    print(f"💰 Total Return:           ${results['total_return']:>10,.2f}")
    print(f"📊 Return %:               {results['return_percentage']:>10.2f}%")
    print(f"📈 Maximum Value:          ${results['max_value']:>10,.2f}")
    print(f"📉 Minimum Value:          ${results['min_value']:>10,.2f}")
    print(f"🔄 Total Trades:           {results['total_trades']:>10}")
    
    if results['return_percentage'] > 0:
        print(f"\n🎉 PROFIT! You would have made ${results['total_return']:.2f}!")
    elif results['return_percentage'] < 0:
        print(f"\n📉 Loss: You would have lost ${abs(results['total_return']):.2f}")
    else:
        print(f"\n🔄 Break-even")
    
    # Show daily progression
    print(f"\n📊 Daily Portfolio Values:")
    for i, day in enumerate(results['simulation_days']):
        actions_str = ", ".join(day['actions']) if day['actions'] else "No trades"
        print(f"  Day {day['day']}: ${day['portfolio_value']:>8,.2f} ({day['return_pct']:+6.2f}%) - {actions_str}")
    
    # Current holdings
    if results['current_portfolio']:
        print(f"\n💼 Final Portfolio Holdings:")
        for symbol, position in results['current_portfolio'].items():
            print(f"  {symbol}: {position['shares']:.4f} shares @ avg ${position['avg_price']:.2f}")
    
    # Save enhanced results
    with open('enhanced_simulation_results.json', 'w') as f:
        # Convert dates and other objects for JSON
        json_results = results.copy()
        json_results['simulation_days'] = []
        for day in results['simulation_days']:
            day_copy = day.copy()
            json_results['simulation_days'].append(day_copy)
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n📄 Enhanced results saved to: enhanced_simulation_results.json")

if __name__ == "__main__":
    main()