#!/usr/bin/env python3
"""
🚀 PURE $5K TRADING SYSTEM - KRAKEN INTEGRATION
Combines the Pure5K trading strategy with Kraken paper trading execution
"""

from app.trading_systems.pure_5k_system import Pure5KLiveTradingSystem
from app.kraken.test_kraken_orders import KrakenPaperTrader
import logging
from datetime import datetime
import os

class Pure5KKrakenSystem(Pure5KLiveTradingSystem):
    def __init__(self, initial_balance: float = 5000.0):
        # Initialize the main trading system
        super().__init__(initial_balance=initial_balance, paper_trading=True)
        
        # Initialize Kraken paper trader
        self.kraken_trader = KrakenPaperTrader()
        
        # Enhanced logging
        self.setup_logging()
        
        print(f"\n🚀 PURE $5K KRAKEN TRADING SYSTEM")
        print(f"💰 Initial Balance: ${initial_balance:,.2f}")
        print(f"📈 Mode: Paper Trading")
        print(f"🔄 Trading: {len(self.crypto_symbols)} crypto pairs")
        
    def setup_logging(self):
        """Setup enhanced logging"""
        log_dir = 'app/logs/kraken'
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(f'{log_dir}/kraken_trading_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def _execute_trade(self, symbol: str, action: str, shares: float, price: float, reason: str) -> bool:
        """Override the trade execution to use Kraken paper trading"""
        try:
            # Convert action to Kraken format
            side = 'buy' if action == 'BUY' else 'sell'
            
            # Place order through Kraken paper trader
            order = self.kraken_trader.place_order(
                symbol=symbol,
                order_type='market',  # Using market orders for now
                side=side,
                amount=shares
            )
            
            if order and order['status'] == 'filled':
                # Update internal tracking (parent class)
                if side == 'buy':
                    self.cash -= shares * price
                else:
                    self.cash += shares * price
                
                # Record trade in the main system
                self._record_trade(
                    date=datetime.now().strftime('%Y-%m-%d'),
                    symbol=symbol,
                    action=action,
                    shares=shares,
                    price=price,
                    amount=shares * price,
                    strategy='Kraken_' + reason
                )
                
                self.logger.info(f"Executed {action} {shares} {symbol} @ ${price:,.2f} through Kraken")
                return True
            
            self.logger.warning(f"Failed to execute {action} {shares} {symbol}")
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return False
    
    def get_current_price(self, symbol: str) -> float:
        """Override to use Kraken prices for supported symbols"""
        if symbol in self.crypto_symbols:
            return self.kraken_trader.kraken_api.get_price(symbol)
        return super().get_current_price(symbol)
    
    def calculate_portfolio_value(self, date: str = None) -> float:
        """Override to use Kraken positions for crypto"""
        total_value = self.cash
        
        # Get Kraken positions
        kraken_positions = self.kraken_trader.get_positions()
        for symbol, position in kraken_positions.items():
            if position['amount'] > 0:
                current_price = self.get_current_price(symbol)
                if current_price > 0:
                    total_value += position['amount'] * current_price
        
        # Add non-crypto positions from main system
        for symbol, position in self.positions.items():
            if symbol not in self.crypto_symbols and position['shares'] > 0:
                current_price = self.get_current_price(symbol)
                if current_price > 0:
                    total_value += position['shares'] * current_price
        
        return total_value
    
    def get_position_summary(self) -> str:
        """Get a summary of all positions"""
        summary = "\n📊 PORTFOLIO SUMMARY\n"
        summary += "=" * 50 + "\n"
        
        # Kraken crypto positions
        summary += "\n🪙 CRYPTO POSITIONS (Kraken):\n"
        kraken_positions = self.kraken_trader.get_positions()
        for symbol, position in kraken_positions.items():
            if position['amount'] > 0:
                current_price = self.get_current_price(symbol)
                position_value = position['amount'] * current_price
                profit_loss = ((current_price - position['avg_price']) / position['avg_price']) * 100
                summary += f"   {symbol}: {position['amount']:.8f} @ ${position['avg_price']:,.2f} = ${position_value:,.2f} ({profit_loss:+.2f}%)\n"
        
        # Non-crypto positions
        summary += "\n📈 OTHER POSITIONS:\n"
        for symbol, position in self.positions.items():
            if symbol not in self.crypto_symbols and position['shares'] > 0:
                current_price = self.get_current_price(symbol)
                position_value = position['shares'] * current_price
                profit_loss = ((current_price - position['avg_price']) / position['avg_price']) * 100
                summary += f"   {symbol}: {position['shares']:.6f} @ ${position['avg_price']:,.2f} = ${position_value:,.2f} ({profit_loss:+.2f}%)\n"
        
        # Portfolio stats
        total_value = self.calculate_portfolio_value()
        return_pct = ((total_value - self.initial_balance) / self.initial_balance) * 100
        
        summary += f"\n💰 PORTFOLIO STATS:\n"
        summary += f"   Total Value: ${total_value:,.2f}\n"
        summary += f"   Cash: ${self.cash:,.2f}\n"
        summary += f"   Return: {return_pct:+.2f}%\n"
        
        return summary

def main():
    """Main execution function"""
    try:
        # Create integrated trading system
        system = Pure5KKrakenSystem(initial_balance=5000.0)
        
        # Start paper trading
        print("\n🚀 Starting Pure5K Kraken Paper Trading...")
        system.start_live_monitoring()
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 