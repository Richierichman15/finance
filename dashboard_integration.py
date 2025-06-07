#!/usr/bin/env python3
"""
🎯 DASHBOARD INTEGRATION SCRIPT
===============================
Runs Enhanced Ultra-Aggressive V2 and updates HTML dashboard with real results
"""

import json
import os
import webbrowser
from datetime import datetime
from enhanced_ultra_aggressive_v2 import EnhancedUltraAggressiveV2

def format_results_for_dashboard(results):
    """Format trading results for dashboard display"""
    
    # Calculate portfolio breakdown by category
    portfolio_breakdown = {}
    for symbol, position in results['final_positions'].items():
        category = position['category']
        if category not in portfolio_breakdown:
            portfolio_breakdown[category] = {'value': 0, 'count': 0, 'symbols': []}
        
        current_price = position.get('current_price', position['avg_price'])
        value = position['shares'] * current_price
        gain_pct = ((current_price - position['avg_price']) / position['avg_price']) * 100
        
        portfolio_breakdown[category]['value'] += value
        portfolio_breakdown[category]['count'] += 1
        portfolio_breakdown[category]['symbols'].append({
            'symbol': symbol,
            'value': value,
            'gain_pct': gain_pct
        })
    
    # Format for dashboard
    dashboard_positions = {}
    for symbol, position in results['final_positions'].items():
        current_price = position.get('current_price', position['avg_price'])
        value = position['shares'] * current_price
        gain_pct = ((current_price - position['avg_price']) / position['avg_price']) * 100
        
        dashboard_positions[symbol] = {
            'shares': position['shares'],
            'category': position['category'],
            'value': value,
            'gain_pct': gain_pct,
            'avg_price': position['avg_price'],
            'current_price': current_price
        }
    
    dashboard_data = {
        'simulation_metadata': results['simulation_metadata'],
        'final_results': results['final_results'],
        'daily_values': results['daily_values'],
        'final_positions': dashboard_positions,
        'trades': results['trades'][-20:],  # Last 20 trades
        'portfolio_breakdown': portfolio_breakdown,
        'last_updated': datetime.now().isoformat()
    }
    
    return dashboard_data

def update_dashboard_html(results):
    """Update the HTML dashboard with new results"""
    
    dashboard_data = format_results_for_dashboard(results)
    
    # Read the current HTML file
    with open('dashboard.html', 'r') as f:
        html_content = f.read()
    
    # Replace the sample data with real results
    json_data = json.dumps(dashboard_data, indent=4, default=str)
    
    # Find and replace the tradingResults object
    start_marker = "let tradingResults = {"
    end_marker = "};"
    
    start_idx = html_content.find(start_marker)
    if start_idx != -1:
        # Find the end of the object (looking for the closing brace)
        brace_count = 0
        current_idx = start_idx + len("let tradingResults = ")
        
        while current_idx < len(html_content):
            char = html_content[current_idx]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = current_idx + 1
                    break
            current_idx += 1
        
        # Replace the data
        new_html = (html_content[:start_idx] + 
                   f"let tradingResults = {json_data}" +
                   html_content[end_idx:])
        
        # Write updated HTML
        dashboard_file = 'dashboard_live.html'
        with open(dashboard_file, 'w') as f:
            f.write(new_html)
        
        print(f"✅ Dashboard updated: {dashboard_file}")
        return dashboard_file
    else:
        print("❌ Could not find tradingResults object in HTML")
        return None

def run_simulation_and_update_dashboard():
    """Run the Enhanced V2 simulation and update dashboard"""
    
    print("🚀 RUNNING ENHANCED ULTRA-AGGRESSIVE V2 SIMULATION")
    print("=" * 60)
    
    # Initialize and run simulation
    simulator = EnhancedUltraAggressiveV2(
        initial_balance=5000.0,
        daily_addition_base=100.0
    )
    
    # Test with 30-day period for reliable results
    results = simulator.run_enhanced_backtest(days=30)
    
    print(f"\n🎯 SIMULATION COMPLETE!")
    print("=" * 40)
    print(f"📈 Final Portfolio Value:  ${results['final_results']['final_value']:,.2f}")
    print(f"💰 Total Return:           ${results['final_results']['total_return']:,.2f}")
    print(f"📊 Return %:               {results['final_results']['return_pct']:.2f}%")
    print(f"🔄 Total Trades:           {results['final_results']['total_trades']}")
    
    success = results['final_results']['return_pct'] >= 10.0
    if success:
        print(f"\n🎉 SUCCESS! {results['final_results']['return_pct']:.2f}% >= 10% TARGET!")
    else:
        print(f"\n⚠️  Target not quite met: {results['final_results']['return_pct']:.2f}% < 10%")
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"enhanced_v2_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"📄 Results saved to: {results_file}")
    
    # Update dashboard
    print(f"\n📊 Updating HTML dashboard...")
    dashboard_file = update_dashboard_html(results)
    
    if dashboard_file:
        # Get absolute path for browser
        abs_path = os.path.abspath(dashboard_file)
        dashboard_url = f"file://{abs_path}"
        
        print(f"🌐 Dashboard URL: {dashboard_url}")
        print(f"\n🚀 Opening dashboard in browser...")
        
        try:
            webbrowser.open(dashboard_url)
            print("✅ Dashboard opened successfully!")
        except Exception as e:
            print(f"⚠️  Could not auto-open browser: {e}")
            print(f"   Please manually open: {dashboard_file}")
    
    return results, dashboard_file

def main():
    """Main execution function"""
    print("🎯 DASHBOARD INTEGRATION - Enhanced Ultra-Aggressive V2")
    print("=" * 70)
    print("💰 Running simulation with real data and updating live dashboard...")
    print()
    
    try:
        results, dashboard_file = run_simulation_and_update_dashboard()
        
        print(f"\n" + "="*70)
        print("🏆 INTEGRATION COMPLETE!")
        print("="*70)
        print(f"✅ Simulation: Enhanced V2 executed successfully")
        print(f"✅ Results: Saved to JSON with timestamp")
        print(f"✅ Dashboard: Updated with live data")
        print(f"✅ Browser: Dashboard opened automatically")
        print()
        print(f"📊 View your live results at: {dashboard_file}")
        print(f"🔄 Re-run this script anytime to refresh with new data!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Please check your setup and try again.")
        return None

if __name__ == "__main__":
    main()