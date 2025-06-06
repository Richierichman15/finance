#!/usr/bin/env python3
from app.services.trading_simulator import TradingSimulator
from app.services.stock_data_collector import StockDataCollector
import json

def main():
    print("🏦 Elite Trading Simulation System")
    print("=" * 60)
    
    # First, make sure we have fresh data
    print("📊 Updating stock data...")
    collector = StockDataCollector()
    data_results = collector.collect_and_store_fundamentals()
    
    print(f"✅ Updated data for {len(data_results['symbols_processed'])} symbols")
    if data_results['symbols_failed']:
        print(f"⚠️  Failed to update: {data_results['symbols_failed']}")
    
    print("\n" + "=" * 60)
    
    # Run the trading simulation
    simulator = TradingSimulator(initial_balance=5000.0)
    
    print("🎯 Trading Criteria:")
    print("BUY when:")
    print(f"  • P/E Ratio ≤ {simulator.buy_criteria['max_pe_ratio']}")
    print(f"  • FCF Yield ≥ {simulator.buy_criteria['min_fcf_yield']}%")
    print(f"  • Debt/Equity ≤ {simulator.buy_criteria['max_debt_to_equity']}")
    print(f"  • Current Ratio ≥ {simulator.buy_criteria['min_current_ratio']}")
    print(f"  • At least 60% of criteria must be met")
    
    print("\nSELL when:")
    print(f"  • P/E Ratio > {simulator.sell_criteria['max_pe_ratio']}")
    print(f"  • FCF Yield < {simulator.sell_criteria['min_fcf_yield']}%")
    print(f"  • Debt/Equity > {simulator.sell_criteria['max_debt_to_equity']}")
    print(f"  • Current Ratio < {simulator.sell_criteria['min_current_ratio']}")
    print(f"  • Any one criteria triggers a sell")
    
    print(f"\n💰 Position Size: {simulator.position_size_pct*100}% of available cash per trade")
    
    print("\n" + "=" * 60)
    
    # Run simulation
    results = simulator.run_simulation(days=14)
    
    if "error" in results:
        print(f"❌ Simulation failed: {results['error']}")
        return
    
    # Save results to file
    with open('simulation_results.json', 'w') as f:
        # Convert datetime objects to strings for JSON serialization
        json_results = results.copy()
        json_results['trade_history'] = []
        for trade in results['trade_history']:
            trade_copy = trade.copy()
            trade_copy['timestamp'] = trade['timestamp'].isoformat()
            json_results['trade_history'].append(trade_copy)
        
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: simulation_results.json")
    
    # Performance analysis
    print("\n" + "=" * 60)
    print("📈 PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    if results['return_percentage'] > 0:
        print(f"🎉 PROFIT! You made ${results['total_return']:.2f} ({results['return_percentage']:.2f}%)")
    elif results['return_percentage'] < 0:
        print(f"📉 Loss: You lost ${abs(results['total_return']):.2f} ({results['return_percentage']:.2f}%)")
    else:
        print("🔄 Break-even: No gain or loss")
    
    # Compare to buy-and-hold strategy
    print(f"\n💡 Analysis:")
    print(f"• Total trades executed: {results['total_trades']}")
    print(f"• Active positions: {results['portfolio_positions']}")
    print(f"• Cash remaining: ${results['final_balance']:.2f}")
    
    if results['total_trades'] == 0:
        print("• No trades were executed - criteria too strict or insufficient data")
    
    print(f"\n🔍 Symbols analyzed: {', '.join(results['symbols_analyzed'])}")

if __name__ == "__main__":
    main()