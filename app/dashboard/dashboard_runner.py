#!/usr/bin/env python3
"""
🌐 DASHBOARD RUNNER - TRADING SYSTEM VISUALIZATION
=================================================
Consolidated dashboard that integrates with all trading systems
"""

import sys
import os
from datetime import datetime
import json

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def run_dashboard_with_system(initial_balance: float = 5000.0, days: int = 30):
    """Run dashboard with integrated trading system"""
    
    print("🌐 DASHBOARD INTEGRATION")
    print("=" * 50)
    print("📊 Running Pure $5K trading system...")
    
    # Import and run the pure trading system
    from trading_systems.pure_5k_system import Pure5KTradingSystem
    
    system = Pure5KTradingSystem(initial_balance=initial_balance)
    results = system.run_pure_5k_backtest(days=days)
    
    if results and not results.get('error'):
        print("✅ Trading system completed successfully!")
        
        # Save results for dashboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dashboard_data_file = f"app/dashboard/data/latest_results.json"
        
        # Create dashboard data directory
        os.makedirs(os.path.dirname(dashboard_data_file), exist_ok=True)
        
        # Prepare dashboard data
        dashboard_data = {
            'timestamp': timestamp,
            'system_type': 'pure_5k',
            'results': results,
            'trades': system.trades,
            'daily_values': system.daily_values,
            'positions': system.positions
        }
        
        with open(dashboard_data_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        print(f"💾 Dashboard data saved to: {dashboard_data_file}")
        
        # Generate HTML dashboard
        generate_html_dashboard(dashboard_data)
        
        print("\n🌐 Dashboard generated!")
        print("📁 Open: app/dashboard/templates/live_dashboard.html")
        
        return True
    else:
        print("❌ Trading system failed")
        return False

def generate_html_dashboard(data):
    """Generate HTML dashboard from trading data"""
    
    results = data['results']
    trades = data['trades']
    daily_values = data['daily_values']
    
    # Create HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultra-Aggressive Trading Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            margin: 0;
            padding: 20px;
            color: white;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(0, 0, 0, 0.8);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: linear-gradient(145deg, #2c3e50, #34495e);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .trades-section {{
            margin-top: 30px;
        }}
        .trade-item {{
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
        }}
        .target-status {{
            font-size: 1.5em;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
        }}
        .target-met {{
            background: linear-gradient(145deg, #27ae60, #2ecc71);
        }}
        .target-not-met {{
            background: linear-gradient(145deg, #e74c3c, #c0392b);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Ultra-Aggressive Trading Dashboard</h1>
            <h2>Pure $5K Trading System Results</h2>
            <p>Generated: {data['timestamp']}</p>
        </div>
        
        <div class="target-status {'target-met' if results['target_met'] else 'target-not-met'}">
            🎯 10% TARGET: {'✅ MET' if results['target_met'] else '❌ NOT MET'}
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <h3>💵 Initial Balance</h3>
                <div class="metric-value">${results['initial_balance']:,.2f}</div>
            </div>
            <div class="metric-card">
                <h3>📈 Final Value</h3>
                <div class="metric-value">${results['final_portfolio_value']:,.2f}</div>
            </div>
            <div class="metric-card">
                <h3>💰 Total Return</h3>
                <div class="metric-value {'positive' if results['total_return'] > 0 else 'negative'}">${results['total_return']:,.2f}</div>
            </div>
            <div class="metric-card">
                <h3>📊 Return %</h3>
                <div class="metric-value {'positive' if results['return_percentage'] > 0 else 'negative'}">{results['return_percentage']:+.2f}%</div>
            </div>
            <div class="metric-card">
                <h3>📈 Max Value</h3>
                <div class="metric-value">${results['max_value']:,.2f}</div>
            </div>
            <div class="metric-card">
                <h3>📉 Min Value</h3>
                <div class="metric-value">${results['min_value']:,.2f}</div>
            </div>
            <div class="metric-card">
                <h3>🔄 Total Trades</h3>
                <div class="metric-value">{results['total_trades']}</div>
            </div>
            <div class="metric-card">
                <h3>📅 Trading Days</h3>
                <div class="metric-value">{results['trading_days']}</div>
            </div>
        </div>
        
        <div class="trades-section">
            <h3>🔄 Recent Trades</h3>
            <div style="max-height: 300px; overflow-y: auto;">
"""
    
    # Add recent trades (last 10)
    recent_trades = trades[-10:] if len(trades) > 10 else trades
    for trade in recent_trades:
        action_emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
        html_content += f"""
                <div class="trade-item">
                    <span>{action_emoji} {trade['action']} {trade['symbol']}</span>
                    <span>{trade['shares']:.6f} @ ${trade['price']:.4f}</span>
                    <span>{trade['date']}</span>
                </div>"""
    
    html_content += """
            </div>
        </div>
        
        <div style="text-align: center; margin-top: 30px;">
            <p>🎯 <strong>Targeting 10% returns through ultra-aggressive strategies</strong></p>
            <p>🚀 Pure trading performance - no daily additions</p>
        </div>
    </div>
</body>
</html>"""
    
    # Save HTML file
    html_file = "app/dashboard/templates/live_dashboard.html"
    os.makedirs(os.path.dirname(html_file), exist_ok=True)
    
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    print(f"🌐 HTML dashboard generated: {html_file}")

def main():
    """Main dashboard runner"""
    print("🌐 Running dashboard with default settings...")
    run_dashboard_with_system(5000.0, 30)

if __name__ == "__main__":
    main()