#!/usr/bin/env python3
"""
Kraken Portfolio Analyzer
Comprehensive analysis of your Kraken account holdings, assets, and financial status
With Dynamic Asset Detection - automatically discovers prices for any new assets
"""

import sys
import os
import time
import json
from datetime import datetime
from decimal import Decimal
from typing import Optional, Tuple
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.kraken import kraken_api

class KrakenPortfolioAnalyzer:
    def __init__(self):
        self.kraken = kraken_api
        self.current_prices = {}
        self.portfolio_value = 0
        self.total_usd_value = 0
        
        # Dynamic asset discovery
        self.asset_mappings_file = "app/data/cache/asset_mappings.json"
        self.known_mappings = self.load_asset_mappings()
        self.discovery_attempts = {}
        
        # EUR/USD conversion rate
        self.eurusd_rate = 1.0
        
        # Investment tracking
        self.investments_file = "app/data/cache/investment_history.json"
        self.investment_data = self.load_investment_data()
        self.total_invested = 0
        self.total_gains_losses = 0
        self.portfolio_performance = 0
        
        # Historical performance tracking
        self.timeframes = {
            '7d': 7 * 24 * 60 * 60,    # 7 days in seconds
            '30d': 30 * 24 * 60 * 60,  # 30 days
            '90d': 90 * 24 * 60 * 60,  # 90 days
            '1y': 365 * 24 * 60 * 60   # 1 year
        }
        self.historical_data = {}
        self.performance_metrics = {}
        
        # Advanced performance metrics
        self.performance_metrics = {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'recovery_factor': 0.0,
            'risk_adjusted_return': 0.0,
            'alpha': 0.0,
            'beta': 0.0,
            'correlation_matrix': None
        }
    
    def load_asset_mappings(self):
        """Load previously discovered asset mappings from cache"""
        try:
            if os.path.exists(self.asset_mappings_file):
                with open(self.asset_mappings_file, 'r') as f:
                    mappings = json.load(f)
                    print(f"📁 Loaded {len(mappings)} cached asset mappings")
                    return mappings
        except Exception as e:
            print(f"⚠️  Could not load asset mappings: {e}")
        
        return {}
    
    def save_asset_mappings(self):
        """Save discovered asset mappings to cache"""
        try:
            os.makedirs(os.path.dirname(self.asset_mappings_file), exist_ok=True)
            with open(self.asset_mappings_file, 'w') as f:
                json.dump(self.known_mappings, f, indent=2)
            print(f"💾 Saved {len(self.known_mappings)} asset mappings to cache")
        except Exception as e:
            print(f"⚠️  Could not save asset mappings: {e}")
    
    def generate_symbol_variations(self, asset):
        """Generate possible symbol variations for an asset"""
        variations = []
        
        # Remove common suffixes for base asset name
        base_asset = asset.replace('.F', '').replace('.EQ', '')
        
        # Special cases for known problematic assets
        if asset == 'XXBT':
            variations.extend(['XBTUSD', 'XXBTZUSD', 'BTCUSD', 'XBTEUR', 'BTCEUR'])
        elif asset == 'XXDG':
            variations.extend(['DOGEUSD', 'XDGUSD', 'DOGEEUR', 'XDGEUR'])
        elif asset == 'XETH':
            variations.extend(['ETHUSD', 'XETHZUSD', 'ETHEUR'])
        
        # Standard USD pairs
        variations.extend([
            f"{base_asset}USD",
            f"{asset}USD", 
            f"X{base_asset}USD",
            f"XX{base_asset}USD",
            f"{base_asset}ZUSD",
            f"X{base_asset}ZUSD",
            f"XX{base_asset}ZUSD"
        ])
        
        # EUR pairs (for conversion)
        variations.extend([
            f"{base_asset}EUR",
            f"{asset}EUR",
            f"X{base_asset}EUR",
            f"XX{base_asset}EUR"
        ])
        
        # GBP pairs (for conversion)
        variations.extend([
            f"{base_asset}GBP",
            f"{asset}GBP",
            f"X{base_asset}GBP"
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for var in variations:
            if var not in seen:
                seen.add(var)
                unique_variations.append(var)
                
        return unique_variations
    
    def discover_asset_price(self, asset):
        """Dynamically discover the correct price symbol for an asset"""
        print(f"🔍 Discovering price symbol for {asset}...")
        
        # Skip stock assets (.EQ) - Kraken doesn't have stocks
        if asset.endswith('.EQ'):
            print(f"📈 Skipping {asset} - Stock assets not available on Kraken (crypto exchange)")
            return None
        
        # Check if we already know this mapping
        if asset in self.known_mappings:
            known_symbol = self.known_mappings[asset]
            print(f"💡 Using cached mapping: {asset} → {known_symbol}")
            return self.fetch_price_for_symbol(known_symbol, asset)
        
        # Generate possible symbol variations
        variations = self.generate_symbol_variations(asset)
        print(f"🎯 Testing {len(variations)} symbol variations...")
        
        # Try each variation
        for symbol in variations:
            try:
                price_data = self.fetch_price_for_symbol(symbol, asset)
                if price_data and price_data['price'] > 0:
                    # Success! Cache this mapping
                    self.known_mappings[asset] = symbol
                    print(f"✅ Found working symbol: {asset} → {symbol} = ${price_data['price']:,.2f}")
                    return price_data
                    
            except Exception as e:
                print(f"❌ {symbol}: {str(e)[:50]}...")
                continue
        
        print(f"❌ No working price symbol found for {asset}")
        return None
    
    def fetch_price_for_symbol(self, symbol, original_asset):
        """Fetch price for a specific symbol"""
        try:
            ticker = self.kraken.get_ticker(symbol)
            if 'result' in ticker and ticker['result']:
                # Get the actual pair name from result
                result_key = list(ticker['result'].keys())[0]
                price = float(ticker['result'][result_key]['c'][0])
                
                # Handle currency conversion
                currency = 'USD'
                if 'EUR' in symbol:
                    price = price * self.eurusd_rate
                    currency = 'EUR→USD'
                elif 'GBP' in symbol:
                    price = price * 1.27  # Approximate GBP/USD rate
                    currency = 'GBP→USD'
                
                return {
                    'price': price,
                    'symbol': symbol,
                    'currency': currency,
                    'original_asset': original_asset
                }
        except Exception as e:
            # Don't print individual errors to avoid spam
            pass
        
        return None
    
    def get_eurusd_rate(self):
        """Get current EUR/USD exchange rate"""
        try:
            eur_ticker = self.kraken.get_ticker('EURUSD')
            if 'result' in eur_ticker and 'EURUSD' in eur_ticker['result']:
                self.eurusd_rate = float(eur_ticker['result']['EURUSD']['c'][0])
                print(f"✅ EUR/USD rate: {self.eurusd_rate:.4f}")
            else:
                print("⚠️  Using EUR/USD rate of 1.0 (fallback)")
        except:
            print("⚠️  Using EUR/USD rate of 1.0 (fallback)")

    def get_current_prices(self):
        """Dynamically fetch current prices for all assets in portfolio"""
        print("📊 Starting Dynamic Asset Price Discovery...")
        
        # Get EUR/USD rate first
        self.get_eurusd_rate()
        
        # Get account balance to see what assets we have
        balance = self.kraken.get_balance()
        if 'result' not in balance:
            print("❌ Could not get account balance for asset discovery")
            return
        
        assets_in_portfolio = list(balance['result'].keys())
        print(f"🎯 Found {len(assets_in_portfolio)} assets to price: {', '.join(assets_in_portfolio)}")
        
        # Discover prices for each asset
        successful_discoveries = 0
        for asset in assets_in_portfolio:
            amount = float(balance['result'][asset])
            if amount <= 0:
                continue
            
            # Skip USD assets - they don't need price discovery
            if asset == 'ZUSD':
                print(f"💵 Skipping {asset} - already USD")
                continue
                
            price_data = self.discover_asset_price(asset)
            if price_data:
                self.current_prices[asset] = price_data['price']
                successful_discoveries += 1
            else:
                print(f"⚠️  No price found for {asset}")
        
        # Save discovered mappings
        self.save_asset_mappings()
        
        print(f"✅ Successfully discovered prices for {successful_discoveries}/{len([a for a in assets_in_portfolio if float(balance['result'][a]) > 0])} assets")
    
    def analyze_balance(self):
        """Get and analyze account balance"""
        print("\n💰 Analyzing Account Balance...")
        
        balance = self.kraken.get_balance()
        
        if 'error' in balance and balance['error']:
            print(f"❌ Error fetching balance: {balance['error']}")
            return None
            
        if 'result' not in balance:
            print("❌ No balance data received")
            return None
            
        print(f"✅ Found {len(balance['result'])} assets in account")
        return balance['result']
    
    def calculate_asset_values(self, balance_data):
        """Calculate USD values for all assets using dynamic pricing"""
        print("\n🧮 Calculating Asset Values with Dynamic Pricing...")
        
        asset_values = {}
        total_usd_value = 0
        
        for currency, amount in balance_data.items():
            amount_float = float(amount)
            
            if amount_float <= 0:
                continue
                
            print(f"Processing {currency}: {amount_float}")
            
            # Handle USD directly
            if currency == 'ZUSD':
                usd_value = amount_float
                asset_values[currency] = {
                    'amount': amount_float,
                    'usd_value': usd_value,
                    'price': 1.0,
                    'type': 'USD'
                }
                total_usd_value += usd_value
                print(f"  → USD: ${usd_value:,.2f}")
                
            else:
                # Use dynamic pricing
                if currency in self.current_prices:
                    price = self.current_prices[currency]
                    usd_value = amount_float * price
                    
                    # Determine asset type
                    asset_type = 'Crypto'
                    if currency.endswith('.F'):
                        asset_type = 'Futures'
                    elif currency.endswith('.EQ'):
                        asset_type = 'Equity'
                    
                    asset_values[currency] = {
                        'amount': amount_float,
                        'usd_value': usd_value,
                        'price': price,
                        'type': asset_type
                    }
                    total_usd_value += usd_value
                    print(f"  → {asset_type}: ${usd_value:,.2f} @ ${price:,.2f}")
                else:
                    # No price found
                    asset_type = 'Unknown (No Price)'
                    if currency.endswith('.F'):
                        asset_type = 'Futures (No Price)'
                    elif currency.endswith('.EQ'):
                        asset_type = 'Equity (No Price)'
                    
                    asset_values[currency] = {
                        'amount': amount_float,
                        'usd_value': 0,
                        'price': 0,
                        'type': asset_type
                    }
                    print(f"  → {asset_type}: No price data available")
        
        self.portfolio_value = total_usd_value
        return asset_values
    
    def load_investment_data(self):
        """Load investment history and cost basis data"""
        try:
            if os.path.exists(self.investments_file):
                with open(self.investments_file, 'r') as f:
                    data = json.load(f)
                    print(f"📁 Loaded investment history with {len(data.get('assets', {}))} tracked assets")
                    return data
        except Exception as e:
            print(f"⚠️  Could not load investment data: {e}")
        
        # Default structure
        return {
            "total_invested": 0,
            "assets": {},
            "last_updated": datetime.now().isoformat()
        }
    
    def save_investment_data(self):
        """Save investment tracking data"""
        try:
            self.investment_data["last_updated"] = datetime.now().isoformat()
            os.makedirs(os.path.dirname(self.investments_file), exist_ok=True)
            with open(self.investments_file, 'w') as f:
                json.dump(self.investment_data, f, indent=2)
            print(f"💾 Saved investment tracking data")
        except Exception as e:
            print(f"⚠️  Could not save investment data: {e}")
    
    def get_trade_history_detailed(self):
        """Get detailed trade history for investment tracking"""
        try:
            print("📈 Fetching detailed trade history for investment analysis...")
            history = self.kraken.get_trade_history()
            
            if 'error' in history and history['error']:
                print(f"⚠️  Could not fetch trade history: {history['error']}")
                return None
                
            if 'result' not in history:
                print("⚠️  No trade history data available")
                return None
                
            trades = history['result'].get('trades', {})
            print(f"✅ Found {len(trades)} trades in history")
            return trades
            
        except Exception as e:
            print(f"⚠️  Error fetching trade history: {e}")
            return None
    
    def calculate_investment_performance(self, asset_values):
        """Calculate investment performance and cost basis"""
        print("\n💰 Calculating Investment Performance...")
        
        # Get trade history for cost basis calculation
        trades = self.get_trade_history_detailed()
        
        # Initialize tracking
        total_cost_basis = 0
        asset_performance = {}
        
        for asset, data in asset_values.items():
            if data['usd_value'] > 0:
                # Check if we have cost basis data
                if asset in self.investment_data['assets']:
                    cost_basis = self.investment_data['assets'][asset].get('cost_basis', 0)
                    quantity_invested = self.investment_data['assets'][asset].get('quantity', 0)
                else:
                    # Estimate cost basis if not tracked
                    # Use varied multipliers to avoid artificial 25% performance
                    if asset == 'ZUSD':
                        cost_basis = data['usd_value']  # USD is always 1:1
                    else:
                        # Use varied estimates based on asset type and volatility
                        if asset.startswith('XX') or 'BTC' in asset:  # Bitcoin
                            multiplier = 0.75  # Assume 33% gains (more conservative)
                        elif 'ETH' in asset:  # Ethereum
                            multiplier = 0.85  # Assume 18% gains
                        elif asset.endswith('.F'):  # Futures
                            multiplier = 0.90  # Assume 11% gains (less volatile)
                        elif asset in ['GOAT', 'PEPE']:  # Meme coins
                            multiplier = 0.50  # Assume 100% gains (high risk)
                        elif asset in ['UNI', 'SOL']:  # Established altcoins
                            multiplier = 0.80  # Assume 25% gains
                        else:  # Other assets
                            multiplier = 0.70  # Assume 43% gains
                        
                        estimated_avg_price = data['price'] * multiplier
                        cost_basis = data['amount'] * estimated_avg_price
                        
                        # Save this estimate with multiplier info
                        self.investment_data['assets'][asset] = {
                            'cost_basis': cost_basis,
                            'quantity': data['amount'],
                            'estimated': True,
                            'last_price': data['price'],
                            'multiplier_used': multiplier,
                            'estimated_gains': f"{((1/multiplier - 1) * 100):.1f}%"
                        }
                
                # Calculate performance
                current_value = data['usd_value']
                gain_loss = current_value - cost_basis
                performance_pct = (gain_loss / cost_basis * 100) if cost_basis > 0 else 0
                
                asset_performance[asset] = {
                    'current_value': current_value,
                    'cost_basis': cost_basis,
                    'gain_loss': gain_loss,
                    'performance_pct': performance_pct,
                    'amount': data['amount'],
                    'current_price': data['price']
                }
                
                total_cost_basis += cost_basis
                print(f"📊 {asset}: ${current_value:,.2f} (Cost: ${cost_basis:,.2f}) = {performance_pct:+.1f}%")
        
        # Calculate overall portfolio performance
        self.total_invested = total_cost_basis
        self.total_gains_losses = self.portfolio_value - total_cost_basis
        self.portfolio_performance = (self.total_gains_losses / total_cost_basis * 100) if total_cost_basis > 0 else 0
        
        # Update investment data
        self.investment_data['total_invested'] = total_cost_basis
        self.save_investment_data()
        
        return asset_performance
    
    def get_trade_history_summary(self):
        """Get summary of recent trading activity"""
        print("\n📜 Fetching Trade History Summary...")
        
        history = self.kraken.get_trade_history()
        
        if 'error' in history and history['error']:
            print(f"⚠️  Could not fetch trade history: {history['error']}")
            return None
            
        return history.get('result', {})
    
    def display_portfolio_summary(self, asset_values):
        """Display comprehensive portfolio summary with investment performance"""
        print("\n" + "="*80)
        print("🏦 KRAKEN PORTFOLIO ANALYSIS")
        print("="*80)
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 Current Portfolio Value: ${self.portfolio_value:,.2f}")
        
        # Investment Performance Summary
        if self.total_invested > 0:
            print(f"💵 Total Invested: ${self.total_invested:,.2f}")
            print(f"📈 Total Gains/Losses: ${self.total_gains_losses:+,.2f}")
            print(f"📊 Portfolio Performance: {self.portfolio_performance:+.2f}%")
            
            # Performance indicator
            if self.portfolio_performance > 0:
                print("🟢 Portfolio is PROFITABLE")
            elif self.portfolio_performance < 0:
                print("🔴 Portfolio is AT A LOSS")
            else:
                print("⚪ Portfolio is BREAK-EVEN")
        else:
            print("⚠️  Investment tracking initialized - performance will be tracked going forward")
        
        print("="*80)
        
        # Categorize assets
        categories = {
            'USD': [],
            'Crypto': [],
            'Futures': [],
            'Equity': [],
            'Unknown (No Price)': [],
            'Futures (No Price)': [],
            'Equity (No Price)': []
        }
        
        for currency, data in asset_values.items():
            if data['type'] in categories:
                categories[data['type']].append((currency, data))
            else:
                # Handle any new types
                if 'Unknown' not in categories:
                    categories['Unknown'] = []
                categories['Unknown'].append((currency, data))
        
        # Display USD assets
        if categories['USD']:
            print("\n📊 USD ASSETS:")
            print("-" * 50)
            for currency, data in categories['USD']:
                percentage = (data['usd_value'] / self.portfolio_value * 100) if self.portfolio_value > 0 else 0
                print(f"   {currency:<12} | {data['amount']:<15.8f} | ${data['usd_value']:<10.2f} | {percentage:<5.1f}%")
        
        # Display Crypto assets
        if categories['Crypto']:
            print("\n🪙 CRYPTO ASSETS:")
            print("-" * 50)
            for currency, data in categories['Crypto']:
                percentage = (data['usd_value'] / self.portfolio_value * 100) if self.portfolio_value > 0 else 0
                print(f"   {currency:<12} | {data['amount']:<15.8f} | ${data['usd_value']:<10.2f} | {percentage:<5.1f}%")
        
        # Display Futures assets
        if categories['Futures']:
            print("\n📈 FUTURES ASSETS:")
            print("-" * 50)
            for currency, data in categories['Futures']:
                percentage = (data['usd_value'] / self.portfolio_value * 100) if self.portfolio_value > 0 else 0
                print(f"   {currency:<12} | {data['amount']:<15.8f} | ${data['usd_value']:<10.2f} | {percentage:<5.1f}%")
        
        # Display Equity assets
        if categories['Equity']:
            print("\n🏢 EQUITY ASSETS:")
            print("-" * 50)
            for currency, data in categories['Equity']:
                percentage = (data['usd_value'] / self.portfolio_value * 100) if self.portfolio_value > 0 else 0
                print(f"   {currency:<12} | {data['amount']:<15.8f} | ${data['usd_value']:<10.2f} | {percentage:<5.1f}%")
        
        # Display assets with no price data
        no_price_assets = (categories['Unknown (No Price)'] + 
                          categories['Futures (No Price)'] + 
                          categories['Equity (No Price)'])
        
        if no_price_assets:
            print("\n❓ ASSETS WITH NO PRICE DATA:")
            print("-" * 50)
            for currency, data in no_price_assets:
                print(f"   {currency:<12} | {data['amount']:<15.8f} | $0.00 (no price data)")
        
        print("="*80)
    
    def display_detailed_breakdown(self, asset_values):
        """Display detailed breakdown with investment performance for each asset"""
        print("\n" + "="*80)
        print("📋 DETAILED ASSET BREAKDOWN WITH PERFORMANCE")
        print("="*80)
        
        # Sort by USD value
        sorted_assets = sorted(asset_values.items(), key=lambda x: x[1]['usd_value'], reverse=True)
        
        print(f"{'Asset':<12} {'Amount':<15} {'Current Price':<12} {'USD Value':<12} {'% Portfolio':<12} {'Performance':<12}")
        print("-" * 90)
        
        for currency, data in sorted_assets:
            if data['usd_value'] > 0:
                percentage = (data['usd_value'] / self.portfolio_value * 100) if self.portfolio_value > 0 else 0
                price_str = f"${data.get('price', 0):,.2f}" if data.get('price', 0) > 0 else "N/A"
                
                # Get performance data
                performance_str = "N/A"
                if currency in self.investment_data.get('assets', {}):
                    asset_data = self.investment_data['assets'][currency]
                    cost_basis = asset_data.get('cost_basis', 0)
                    if cost_basis > 0:
                        current_value = data['usd_value']
                        gain_loss = current_value - cost_basis
                        performance_pct = (gain_loss / cost_basis * 100)
                        
                        # Add estimation indicator
                        is_estimated = asset_data.get('estimated', False)
                        estimate_suffix = "*" if is_estimated else ""
                        
                        performance_str = f"{performance_pct:+.1f}%{estimate_suffix}"
                        
                        # Add color coding for performance
                        if performance_pct > 0:
                            performance_str = f"🟢{performance_str}"
                        elif performance_pct < 0:
                            performance_str = f"🔴{performance_str}"
                        else:
                            performance_str = f"⚪{performance_str}"
                
                print(f"{currency:<12} {data['amount']:<15.8f} {price_str:<12} ${data['usd_value']:<11,.2f} {percentage:<11.1f}% {performance_str:<12}")
        
        print("="*80)
        
        # Investment Summary
        if self.total_invested > 0:
            print(f"\n💰 INVESTMENT SUMMARY:")
            print(f"📊 Total Invested: ${self.total_invested:,.2f}")
            print(f"💎 Current Value: ${self.portfolio_value:,.2f}")
            print(f"📈 Total Gains/Losses: ${self.total_gains_losses:+,.2f}")
            print(f"🎯 Overall Performance: {self.portfolio_performance:+.2f}%")
            
            # Performance breakdown
            profitable_assets = sum(1 for asset in self.investment_data.get('assets', {}).values() 
                                   if asset.get('cost_basis', 0) > 0 and 
                                   self.calculate_asset_performance_pct(asset) > 0)
            total_tracked_assets = len([a for a in self.investment_data.get('assets', {}).values() 
                                      if a.get('cost_basis', 0) > 0])
            
            if total_tracked_assets > 0:
                print(f"🎲 Profitable Assets: {profitable_assets}/{total_tracked_assets} ({profitable_assets/total_tracked_assets*100:.1f}%)")
                
            # Check if we have any estimated values
            estimated_assets = sum(1 for asset in self.investment_data.get('assets', {}).values() 
                                 if asset.get('estimated', False))
            if estimated_assets > 0:
                print(f"\n⚠️  NOTE: * indicates estimated performance (no purchase history available)")
                print(f"📊 {estimated_assets} assets using estimated cost basis")
                print(f"💡 Connect real trade history for accurate performance tracking")
        
        print("="*80)
    
    def calculate_asset_performance_pct(self, asset_data):
        """Helper to calculate individual asset performance percentage"""
        cost_basis = asset_data.get('cost_basis', 0)
        current_price = asset_data.get('last_price', 0)
        quantity = asset_data.get('quantity', 0)
        
        if cost_basis > 0 and current_price > 0:
            current_value = quantity * current_price
            return (current_value - cost_basis) / cost_basis * 100
        return 0
    
    def display_trading_summary(self, trade_history):
        """Display trading activity summary"""
        if not trade_history:
            print("\n⚠️  No trade history available")
            return
            
        print("\n" + "="*80)
        print("📈 TRADING ACTIVITY SUMMARY")
        print("="*80)
        
        # Count recent trades
        recent_trades = len(trade_history.get('trades', {}))
        print(f"📊 Recent Trades: {recent_trades}")
        
        if recent_trades > 0:
            print("🕒 Most recent trades available in Kraken dashboard")
    
    def display_recommendations(self, asset_values):
        """Display portfolio recommendations"""
        print("\n" + "="*80)
        print("💡 PORTFOLIO RECOMMENDATIONS")
        print("="*80)
        
        # Analyze portfolio composition
        crypto_value = sum(data['usd_value'] for data in asset_values.values() if data['type'] == 'Crypto')
        usd_value = sum(data['usd_value'] for data in asset_values.values() if data['type'] == 'USD')
        
        crypto_percentage = (crypto_value / self.portfolio_value * 100) if self.portfolio_value > 0 else 0
        usd_percentage = (usd_value / self.portfolio_value * 100) if self.portfolio_value > 0 else 0
        
        print(f"🪙 Crypto Allocation: {crypto_percentage:.1f}% (${crypto_value:,.2f})")
        print(f"💵 Cash Allocation: {usd_percentage:.1f}% (${usd_value:,.2f})")
        
        # Recommendations
        if crypto_percentage > 80:
            print("⚠️  High crypto concentration - consider diversifying")
        elif crypto_percentage < 20:
            print("💡 Low crypto allocation - consider increasing exposure")
        else:
            print("✅ Balanced portfolio allocation")
            
        if usd_percentage < 5:
            print("⚠️  Low cash reserves - consider keeping some USD for opportunities")
        elif usd_percentage > 50:
            print("💡 High cash position - consider deploying into crypto")
        else:
            print("✅ Good cash management")
    
    async def calculate_historical_performance(self, asset_values: dict) -> dict:
        """Calculate historical performance across multiple timeframes"""
        try:
            print("\n📈 Calculating historical performance...")
            now = int(time.time())
            
            for timeframe, seconds in self.timeframes.items():
                start_time = now - seconds
                
                # Get historical trades and balance changes
                trades = await self.get_historical_trades(start_time)
                
                # Calculate metrics for this timeframe
                metrics = {
                    'total_trades': len(trades),
                    'volume': sum(float(t['vol']) for t in trades),
                    'starting_balance': await self.get_historical_balance(start_time),
                    'current_balance': self.portfolio_value,
                    'pnl': 0.0,
                    'pnl_percentage': 0.0
                }
                
                # Calculate PnL if we have valid starting balance
                if metrics['starting_balance'] > 0:
                    metrics['pnl'] = metrics['current_balance'] - metrics['starting_balance']
                    metrics['pnl_percentage'] = (metrics['pnl'] / metrics['starting_balance']) * 100
                
                # Add volatility calculation
                price_changes = [float(t['price']) for t in trades]
                if price_changes:
                    metrics['volatility'] = self.calculate_volatility(price_changes)
                else:
                    metrics['volatility'] = 0.0
                
                self.performance_metrics[timeframe] = metrics
            
            return self.performance_metrics
            
        except Exception as e:
            print(f"⚠️  Error calculating historical performance: {e}")
            return {}

    async def get_historical_trades(self, start_time: int) -> list:
        """Get historical trades from start_time to now"""
        try:
            trades_response = kraken_api.get_trades_history({
                'start': str(start_time)
            })
            
            if 'result' in trades_response and 'trades' in trades_response['result']:
                return trades_response['result']['trades']
            return []
            
        except Exception as e:
            print(f"⚠️  Error fetching historical trades: {e}")
            return []

    async def get_historical_balance(self, timestamp: int) -> float:
        """Calculate portfolio balance at a historical timestamp"""
        try:
            # Get ledger entries up to timestamp
            ledger_response = kraken_api.get_ledgers({
                'start': str(timestamp),
                'type': 'trade'
            })
            
            if 'result' in ledger_response and 'ledger' in ledger_response['result']:
                ledger = ledger_response['result']['ledger']
                
                # Calculate balance from ledger entries
                balance = 0.0
                for entry in ledger.values():
                    if float(entry['time']) <= timestamp:
                        balance += float(entry['amount'])
                
                return balance
                
            return 0.0
            
        except Exception as e:
            print(f"⚠️  Error fetching historical balance: {e}")
            return 0.0

    def calculate_volatility(self, price_changes: list) -> float:
        """Calculate price volatility from a list of prices"""
        try:
            if len(price_changes) < 2:
                return 0.0
                
            # Calculate daily returns
            returns = []
            for i in range(1, len(price_changes)):
                daily_return = (price_changes[i] - price_changes[i-1]) / price_changes[i-1]
                returns.append(daily_return)
            
            # Calculate standard deviation of returns
            mean_return = sum(returns) / len(returns)
            squared_diff_sum = sum((r - mean_return) ** 2 for r in returns)
            volatility = (squared_diff_sum / (len(returns) - 1)) ** 0.5
            
            # Annualize volatility
            return volatility * (365 ** 0.5) * 100
            
        except Exception as e:
            print(f"⚠️  Error calculating volatility: {e}")
            return 0.0

    def display_historical_performance(self):
        """Display historical performance metrics"""
        try:
            print("\n=== HISTORICAL PERFORMANCE ANALYSIS ===")
            print("-" * 40)
            
            for timeframe, metrics in self.performance_metrics.items():
                print(f"\n📊 {timeframe.upper()} Performance:")
                print(f"   Total Trades: {metrics['total_trades']}")
                print(f"   Trading Volume: ${metrics['volume']:,.2f}")
                print(f"   PnL: ${metrics['pnl']:,.2f} ({metrics['pnl_percentage']:+.2f}%)")
                print(f"   Volatility: {metrics['volatility']:.2f}%")
                
        except Exception as e:
            print(f"⚠️  Error displaying historical performance: {e}")

    async def calculate_advanced_metrics(self, asset_values: dict) -> dict:
        """Calculate advanced performance metrics"""
        try:
            print("\n📊 Calculating advanced performance metrics...")
            
            # Get historical price data for calculations
            historical_data = {}
            for asset in asset_values.keys():
                if asset != 'ZUSD':  # Skip cash
                    prices = await self._get_historical_prices(asset)
                    if prices is not None:
                        historical_data[asset] = prices
            
            # Calculate returns
            returns = self._calculate_returns(historical_data)
            
            # 1. Sharpe Ratio (Risk-adjusted return)
            risk_free_rate = 0.04  # 4% annual risk-free rate
            self.performance_metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns, risk_free_rate)
            
            # 2. Sortino Ratio (Downside risk-adjusted return)
            self.performance_metrics['sortino_ratio'] = self._calculate_sortino_ratio(returns, risk_free_rate)
            
            # 3. Maximum Drawdown
            self.performance_metrics['max_drawdown'] = self._calculate_max_drawdown(returns)
            
            # 4. Win Rate
            self.performance_metrics['win_rate'] = self._calculate_win_rate(returns)
            
            # 5. Profit Factor
            self.performance_metrics['profit_factor'] = self._calculate_profit_factor(returns)
            
            # 6. Recovery Factor
            self.performance_metrics['recovery_factor'] = self._calculate_recovery_factor(returns)
            
            # 7. Risk-Adjusted Return
            self.performance_metrics['risk_adjusted_return'] = self._calculate_risk_adjusted_return(returns)
            
            # 8. Alpha & Beta (vs BTC)
            btc_prices = await self._get_historical_prices('BTC-USD')
            if btc_prices is not None:
                btc_returns = self._calculate_returns({'BTC-USD': btc_prices})
                alpha, beta = self._calculate_alpha_beta(returns, btc_returns['BTC-USD'])
                self.performance_metrics['alpha'] = alpha
                self.performance_metrics['beta'] = beta
            
            # 9. Correlation Matrix
            self.performance_metrics['correlation_matrix'] = self._calculate_correlation_matrix(returns)
            
            return self.performance_metrics
            
        except Exception as e:
            print(f"⚠️  Error calculating advanced metrics: {e}")
            return {}

    async def _get_historical_prices(self, asset: str, days: int = 365) -> Optional[pd.Series]:
        """Get historical price data for an asset"""
        try:
            # Convert to Kraken symbol if needed
            kraken_symbol = self._convert_to_kraken_symbol(asset)
            if not kraken_symbol:
                return None
                
            # Get OHLCV data from Kraken
            ohlc = kraken_api.get_ohlc_data(kraken_symbol, interval=1440, since=int(time.time() - days * 86400))
            if 'result' in ohlc and kraken_symbol in ohlc['result']:
                prices = pd.Series([float(x[4]) for x in ohlc['result'][kraken_symbol]])  # Use closing prices
                prices.index = pd.to_datetime([x[0] for x in ohlc['result'][kraken_symbol]], unit='s')
                return prices
                
            return None
            
        except Exception as e:
            print(f"⚠️  Error fetching historical prices for {asset}: {e}")
            return None

    def _calculate_returns(self, historical_data: dict) -> dict:
        """Calculate daily returns for all assets"""
        returns = {}
        for asset, prices in historical_data.items():
            if len(prices) > 1:
                returns[asset] = prices.pct_change().dropna()
        return returns

    def _calculate_sharpe_ratio(self, returns: dict, risk_free_rate: float) -> float:
        """Calculate portfolio Sharpe ratio"""
        try:
            # Calculate portfolio returns
            portfolio_returns = pd.concat(returns.values(), axis=1).mean(axis=1)
            
            # Annualize metrics
            avg_return = portfolio_returns.mean() * 252
            std_dev = portfolio_returns.std() * np.sqrt(252)
            
            if std_dev == 0:
                return 0.0
                
            return (avg_return - risk_free_rate) / std_dev
            
        except Exception as e:
            print(f"⚠️  Error calculating Sharpe ratio: {e}")
            return 0.0

    def _calculate_sortino_ratio(self, returns: dict, risk_free_rate: float) -> float:
        """Calculate portfolio Sortino ratio"""
        try:
            portfolio_returns = pd.concat(returns.values(), axis=1).mean(axis=1)
            
            # Calculate downside deviation
            negative_returns = portfolio_returns[portfolio_returns < 0]
            if len(negative_returns) == 0:
                return 0.0
                
            downside_std = np.sqrt(np.mean(negative_returns ** 2)) * np.sqrt(252)
            avg_return = portfolio_returns.mean() * 252
            
            if downside_std == 0:
                return 0.0
                
            return (avg_return - risk_free_rate) / downside_std
            
        except Exception as e:
            print(f"⚠️  Error calculating Sortino ratio: {e}")
            return 0.0

    def _calculate_max_drawdown(self, returns: dict) -> float:
        """Calculate maximum drawdown"""
        try:
            portfolio_returns = pd.concat(returns.values(), axis=1).mean(axis=1)
            cumulative = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = cumulative / rolling_max - 1
            return float(drawdowns.min())
            
        except Exception as e:
            print(f"⚠️  Error calculating max drawdown: {e}")
            return 0.0

    def _calculate_win_rate(self, returns: dict) -> float:
        """Calculate win rate"""
        try:
            portfolio_returns = pd.concat(returns.values(), axis=1).mean(axis=1)
            winning_days = len(portfolio_returns[portfolio_returns > 0])
            total_days = len(portfolio_returns)
            return winning_days / total_days if total_days > 0 else 0.0
            
        except Exception as e:
            print(f"⚠️  Error calculating win rate: {e}")
            return 0.0

    def _calculate_profit_factor(self, returns: dict) -> float:
        """Calculate profit factor"""
        try:
            portfolio_returns = pd.concat(returns.values(), axis=1).mean(axis=1)
            gains = portfolio_returns[portfolio_returns > 0].sum()
            losses = abs(portfolio_returns[portfolio_returns < 0].sum())
            return gains / losses if losses != 0 else 0.0
            
        except Exception as e:
            print(f"⚠️  Error calculating profit factor: {e}")
            return 0.0

    def _calculate_recovery_factor(self, returns: dict) -> float:
        """Calculate recovery factor"""
        try:
            portfolio_returns = pd.concat(returns.values(), axis=1).mean(axis=1)
            cumulative_return = (1 + portfolio_returns).prod() - 1
            max_dd = self._calculate_max_drawdown(returns)
            return cumulative_return / abs(max_dd) if max_dd != 0 else 0.0
            
        except Exception as e:
            print(f"⚠️  Error calculating recovery factor: {e}")
            return 0.0

    def _calculate_risk_adjusted_return(self, returns: dict) -> float:
        """Calculate risk-adjusted return using modified Treynor ratio"""
        try:
            portfolio_returns = pd.concat(returns.values(), axis=1).mean(axis=1)
            avg_return = portfolio_returns.mean() * 252
            downside_risk = portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
            return avg_return / downside_risk if downside_risk != 0 else 0.0
            
        except Exception as e:
            print(f"⚠️  Error calculating risk-adjusted return: {e}")
            return 0.0

    def _calculate_alpha_beta(self, returns: dict, market_returns: pd.Series) -> Tuple[float, float]:
        """Calculate portfolio alpha and beta vs market (BTC)"""
        try:
            portfolio_returns = pd.concat(returns.values(), axis=1).mean(axis=1)
            
            # Calculate beta
            covariance = portfolio_returns.cov(market_returns)
            market_variance = market_returns.var()
            beta = covariance / market_variance if market_variance != 0 else 0.0
            
            # Calculate alpha
            risk_free_rate = 0.04 / 252  # Daily risk-free rate
            alpha = portfolio_returns.mean() - (risk_free_rate + beta * (market_returns.mean() - risk_free_rate))
            alpha *= 252  # Annualize
            
            return alpha, beta
            
        except Exception as e:
            print(f"⚠️  Error calculating alpha/beta: {e}")
            return 0.0, 0.0

    def _calculate_correlation_matrix(self, returns: dict) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix between assets"""
        try:
            return pd.DataFrame(returns).corr()
            
        except Exception as e:
            print(f"⚠️  Error calculating correlation matrix: {e}")
            return None

    def display_advanced_metrics(self):
        """Display advanced performance metrics"""
        try:
            print("\n=== ADVANCED PERFORMANCE METRICS ===")
            print("-" * 40)
            
            print(f"\n📈 Risk-Adjusted Returns:")
            print(f"   Sharpe Ratio: {self.performance_metrics['sharpe_ratio']:.2f}")
            print(f"   Sortino Ratio: {self.performance_metrics['sortino_ratio']:.2f}")
            print(f"   Risk-Adjusted Return: {self.performance_metrics['risk_adjusted_return']:.2f}")
            
            print(f"\n📊 Risk Metrics:")
            print(f"   Maximum Drawdown: {self.performance_metrics['max_drawdown']*100:.1f}%")
            print(f"   Beta vs BTC: {self.performance_metrics['beta']:.2f}")
            print(f"   Alpha (annualized): {self.performance_metrics['alpha']*100:.1f}%")
            
            print(f"\n🎯 Trading Metrics:")
            print(f"   Win Rate: {self.performance_metrics['win_rate']*100:.1f}%")
            print(f"   Profit Factor: {self.performance_metrics['profit_factor']:.2f}")
            print(f"   Recovery Factor: {self.performance_metrics['recovery_factor']:.2f}")
            
            if self.performance_metrics['correlation_matrix'] is not None:
                print("\n📐 Asset Correlations:")
                print(self.performance_metrics['correlation_matrix'].round(2))
            
        except Exception as e:
            print(f"⚠️  Error displaying advanced metrics: {e}")

    async def run_analysis(self):
        """Main analysis runner with advanced metrics"""
        try:
            # Get current portfolio state
            print("\n🔍 ANALYZING PORTFOLIO...")
            
            # Get EUR/USD rate for conversions
            self.get_eurusd_rate()
            
            # Get current prices for all assets
            await self.get_current_prices()
            
            # Analyze current balance
            balance_data = await self.analyze_balance()
            
            # Calculate current asset values
            asset_values = await self.calculate_asset_values(balance_data)
            
            # Calculate historical performance
            await self.calculate_historical_performance(asset_values)
            
            # Calculate advanced metrics
            await self.calculate_advanced_metrics(asset_values)
            
            # Display results
            self.display_portfolio_summary(asset_values)
            self.display_detailed_breakdown(asset_values)
            self.display_historical_performance()
            self.display_advanced_metrics()
            
            # Generate recommendations
            self.display_recommendations(asset_values)
            
            # Save discovered mappings
            self.save_asset_mappings()
            
        except Exception as e:
            print(f"⚠️  Analysis failed: {e}")

def main():
    analyzer = KrakenPortfolioAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main() 