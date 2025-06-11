#!/usr/bin/env python3
"""
Kraken Paper Trading Order Test Script
Tests order placement, cancellation, and position tracking without using real money
"""

from app.services.kraken import kraken_api
import time
from datetime import datetime
import json
import os

class KrakenPaperTrader:
    def __init__(self):
        self.orders = []
        self.positions = {}
        self.order_id_counter = 1
        
        # Load previous paper trading data if exists
        self.load_paper_trading_data()
        
    def load_paper_trading_data(self):
        """Load previous paper trading data"""
        try:
            if os.path.exists('paper_trading_data.json'):
                with open('paper_trading_data.json', 'r') as f:
                    data = json.load(f)
                    self.orders = data.get('orders', [])
                    self.positions = data.get('positions', {})
                    self.order_id_counter = data.get('order_id_counter', 1)
                print("📝 Loaded previous paper trading data")
        except Exception as e:
            print(f"⚠️  Failed to load paper trading data: {e}")

    def save_paper_trading_data(self):
        """Save paper trading data"""
        try:
            data = {
                'orders': self.orders,
                'positions': self.positions,
                'order_id_counter': self.order_id_counter
            }
            with open('paper_trading_data.json', 'w') as f:
                json.dump(data, f, indent=2)
            print("💾 Saved paper trading data")
        except Exception as e:
            print(f"⚠️  Failed to save paper trading data: {e}")

    def place_order(self, symbol: str, order_type: str, side: str, amount: float):
        """Place a paper trading order"""
        try:
            # Get current price from Kraken
            current_price = kraken_api.get_price(symbol)
            if current_price <= 0:
                print(f"❌ Could not get price for {symbol}")
                return None
            
            # Create paper order
            order = {
                'id': f"PAPER_{self.order_id_counter}",
                'symbol': symbol,
                'type': order_type,
                'side': side,
                'amount': amount,
                'price': current_price,
                'status': 'open',
                'timestamp': datetime.now().isoformat(),
                'filled': 0.0
            }
            
            self.order_id_counter += 1
            self.orders.append(order)
            
            print(f"\n📝 Paper Order Placed:")
            print(f"   Symbol: {symbol}")
            print(f"   Type: {order_type}")
            print(f"   Side: {side}")
            print(f"   Amount: {amount}")
            print(f"   Price: ${current_price:,.2f}")
            print(f"   Order ID: {order['id']}")
            
            # Auto-fill market orders
            if order_type == 'market':
                self._fill_order(order)
            
            self.save_paper_trading_data()
            return order
            
        except Exception as e:
            print(f"❌ Failed to place paper order: {e}")
            return None

    def _fill_order(self, order):
        """Simulate order filling"""
        try:
            symbol = order['symbol']
            side = order['side']
            amount = order['amount']
            price = order['price']
            
            # Update position
            if symbol not in self.positions:
                self.positions[symbol] = {
                    'amount': 0,
                    'avg_price': 0
                }
            
            position = self.positions[symbol]
            
            if side == 'buy':
                # Calculate new average price
                total_cost = (position['amount'] * position['avg_price']) + (amount * price)
                total_amount = position['amount'] + amount
                new_avg_price = total_cost / total_amount if total_amount > 0 else price
                
                position['amount'] = total_amount
                position['avg_price'] = new_avg_price
                
            else:  # sell
                position['amount'] -= amount
                if position['amount'] <= 0:
                    position['amount'] = 0
                    position['avg_price'] = 0
            
            # Update order
            order['status'] = 'filled'
            order['filled'] = amount
            order['fill_price'] = price
            order['fill_time'] = datetime.now().isoformat()
            
            print(f"\n✅ Order Filled:")
            print(f"   Fill Price: ${price:,.2f}")
            print(f"   Fill Amount: {amount}")
            print(f"   New Position: {position['amount']} {symbol}")
            print(f"   Average Price: ${position['avg_price']:,.2f}")
            
        except Exception as e:
            print(f"❌ Failed to fill paper order: {e}")

    def cancel_order(self, order_id: str):
        """Cancel a paper trading order"""
        for order in self.orders:
            if order['id'] == order_id and order['status'] == 'open':
                order['status'] = 'cancelled'
                print(f"\n❌ Order Cancelled: {order_id}")
                self.save_paper_trading_data()
                return True
        return False

    def get_open_orders(self):
        """Get all open paper trading orders"""
        return [order for order in self.orders if order['status'] == 'open']

    def get_positions(self):
        """Get current paper trading positions"""
        return self.positions

def test_paper_trading():
    """Run paper trading tests"""
    trader = KrakenPaperTrader()
    
    print("\n🚀 Starting Paper Trading Tests")
    print("=" * 50)
    
    # 1. Test market buy
    print("\n1️⃣  Testing Market Buy Order")
    buy_order = trader.place_order('BTC-USD', 'market', 'buy', 0.001)
    
    # 2. Test limit buy
    print("\n2️⃣  Testing Limit Buy Order")
    limit_buy = trader.place_order('ETH-USD', 'limit', 'buy', 0.01)
    
    # 3. Test order cancellation
    if limit_buy:
        print("\n3️⃣  Testing Order Cancellation")
        trader.cancel_order(limit_buy['id'])
    
    # 4. Test market sell
    print("\n4️⃣  Testing Market Sell Order")
    if buy_order:
        trader.place_order('BTC-USD', 'market', 'sell', 0.0005)
    
    # Show current positions
    print("\n📊 Current Positions:")
    positions = trader.get_positions()
    for symbol, pos in positions.items():
        if pos['amount'] > 0:
            print(f"   {symbol}: {pos['amount']} @ ${pos['avg_price']:,.2f}")
    
    # Show open orders
    print("\n📝 Open Orders:")
    open_orders = trader.get_open_orders()
    for order in open_orders:
        print(f"   {order['id']}: {order['side']} {order['amount']} {order['symbol']} @ ${order['price']:,.2f}")

if __name__ == "__main__":
    test_paper_trading() 