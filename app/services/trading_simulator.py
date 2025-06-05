import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from ..database import SessionLocal
from ..models.database_models import StockFundamentals
from .stock_data_collector import StockDataCollector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSimulator:
    def __init__(self, initial_balance: float = 5000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.portfolio = {}  # symbol: {"shares": quantity, "avg_price": price}
        self.trade_history = []
        self.collector = StockDataCollector()
        
        # Trading criteria based on fundamental analysis
        self.buy_criteria = {
            "max_pe_ratio": 25.0,        # P/E ratio should be reasonable
            "min_fcf_yield": 5.0,        # FCF yield should be > 5%
            "max_debt_to_equity": 1.0,   # D/E ratio should be < 1.0
            "min_current_ratio": 1.2,    # Current ratio should be > 1.2
        }
        
        self.sell_criteria = {
            "max_pe_ratio": 35.0,        # Sell if P/E gets too high
            "min_fcf_yield": 3.0,        # Sell if FCF yield drops below 3%
            "max_debt_to_equity": 1.5,   # Sell if D/E gets too high
            "min_current_ratio": 1.0,    # Sell if current ratio drops too low
        }
        
        # Position sizing (what % of available cash to use per trade)
        self.position_size_pct = 0.15  # Use 15% of available cash per position
        
        # Track performance
        self.daily_values = []
        
    def collect_historical_data(self, days: int = 14) -> Dict[str, Any]:
        """
        Collect historical price data for the last N days for all tracked symbols
        """
        results = {
            "symbols_processed": [],
            "symbols_failed": [],
            "historical_data": {}
        }
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for symbol in self.collector.tracked_stocks:
            try:
                logger.info(f"Collecting {days} days of historical data for {symbol}")
                ticker = yf.Ticker(symbol)
                
                # Get historical price data
                hist_data = ticker.history(start=start_date, end=end_date)
                
                if not hist_data.empty:
                    results["historical_data"][symbol] = hist_data
                    results["symbols_processed"].append(symbol)
                    logger.info(f"Collected {len(hist_data)} days of data for {symbol}")
                else:
                    results["symbols_failed"].append(symbol)
                    logger.warning(f"No historical data found for {symbol}")
                    
            except Exception as e:
                logger.error(f"Error collecting historical data for {symbol}: {str(e)}")
                results["symbols_failed"].append(symbol)
                
        return results
    
    def get_latest_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest fundamental data for a symbol from database
        """
        fundamentals = self.collector.get_latest_fundamentals(symbol, 1)
        return fundamentals[0] if fundamentals else None
    
    def should_buy(self, symbol: str, fundamentals: Dict[str, Any]) -> bool:
        """
        Determine if we should buy a stock based on fundamental criteria
        """
        if not fundamentals:
            return False
            
        # Skip crypto for fundamental analysis (they don't have traditional metrics)
        if "-USD" in symbol:
            # For crypto, use simple momentum/technical analysis
            return self._should_buy_crypto(symbol)
        
        pe_ratio = fundamentals.get("pe_ratio_ttm")
        fcf_yield = fundamentals.get("fcf_yield") 
        debt_to_equity = fundamentals.get("debt_to_equity")
        current_ratio = fundamentals.get("current_ratio")
        
        # Check each criterion
        criteria_met = 0
        total_criteria = 0
        
        # P/E Ratio check
        if pe_ratio is not None:
            total_criteria += 1
            if pe_ratio <= self.buy_criteria["max_pe_ratio"]:
                criteria_met += 1
                logger.info(f"{symbol}: P/E {pe_ratio:.2f} ✓ (< {self.buy_criteria['max_pe_ratio']})")
            else:
                logger.info(f"{symbol}: P/E {pe_ratio:.2f} ✗ (> {self.buy_criteria['max_pe_ratio']})")
        
        # FCF Yield check
        if fcf_yield is not None:
            total_criteria += 1
            if fcf_yield >= self.buy_criteria["min_fcf_yield"]:
                criteria_met += 1
                logger.info(f"{symbol}: FCF Yield {fcf_yield:.2f}% ✓ (> {self.buy_criteria['min_fcf_yield']}%)")
            else:
                logger.info(f"{symbol}: FCF Yield {fcf_yield:.2f}% ✗ (< {self.buy_criteria['min_fcf_yield']}%)")
        
        # Debt to Equity check
        if debt_to_equity is not None:
            total_criteria += 1
            if debt_to_equity <= self.buy_criteria["max_debt_to_equity"]:
                criteria_met += 1
                logger.info(f"{symbol}: D/E {debt_to_equity:.2f} ✓ (< {self.buy_criteria['max_debt_to_equity']})")
            else:
                logger.info(f"{symbol}: D/E {debt_to_equity:.2f} ✗ (> {self.buy_criteria['max_debt_to_equity']})")
        
        # Current Ratio check
        if current_ratio is not None:
            total_criteria += 1
            if current_ratio >= self.buy_criteria["min_current_ratio"]:
                criteria_met += 1
                logger.info(f"{symbol}: Current Ratio {current_ratio:.2f} ✓ (> {self.buy_criteria['min_current_ratio']})")
            else:
                logger.info(f"{symbol}: Current Ratio {current_ratio:.2f} ✗ (< {self.buy_criteria['min_current_ratio']})")
        
        # Need at least 60% of available criteria to be met
        if total_criteria == 0:
            return False
            
        success_rate = criteria_met / total_criteria
        should_buy = success_rate >= 0.6
        
        logger.info(f"{symbol}: {criteria_met}/{total_criteria} criteria met ({success_rate:.1%}) - {'BUY' if should_buy else 'PASS'}")
        return should_buy
    
    def _should_buy_crypto(self, symbol: str) -> bool:
        """
        Simple crypto trading logic based on momentum
        """
        # For now, implement a basic strategy: buy if we don't own any crypto yet
        crypto_holdings = sum(1 for s in self.portfolio.keys() if "-USD" in s)
        max_crypto_positions = 2  # Max 2 crypto positions
        
        return crypto_holdings < max_crypto_positions
    
    def should_sell(self, symbol: str, fundamentals: Dict[str, Any]) -> bool:
        """
        Determine if we should sell a stock based on fundamental criteria
        """
        if symbol not in self.portfolio:
            return False
            
        if not fundamentals:
            return False
            
        # Skip crypto for fundamental analysis
        if "-USD" in symbol:
            return self._should_sell_crypto(symbol)
        
        pe_ratio = fundamentals.get("pe_ratio_ttm")
        fcf_yield = fundamentals.get("fcf_yield")
        debt_to_equity = fundamentals.get("debt_to_equity") 
        current_ratio = fundamentals.get("current_ratio")
        
        # Check sell criteria (any one trigger = sell)
        sell_triggers = []
        
        if pe_ratio is not None and pe_ratio > self.sell_criteria["max_pe_ratio"]:
            sell_triggers.append(f"P/E too high: {pe_ratio:.2f}")
            
        if fcf_yield is not None and fcf_yield < self.sell_criteria["min_fcf_yield"]:
            sell_triggers.append(f"FCF yield too low: {fcf_yield:.2f}%")
            
        if debt_to_equity is not None and debt_to_equity > self.sell_criteria["max_debt_to_equity"]:
            sell_triggers.append(f"D/E too high: {debt_to_equity:.2f}")
            
        if current_ratio is not None and current_ratio < self.sell_criteria["min_current_ratio"]:
            sell_triggers.append(f"Current ratio too low: {current_ratio:.2f}")
        
        if sell_triggers:
            logger.info(f"{symbol}: SELL triggered - {', '.join(sell_triggers)}")
            return True
            
        return False
    
    def _should_sell_crypto(self, symbol: str) -> bool:
        """
        Simple crypto selling logic 
        """
        # For now, just hold crypto (buy and hold strategy)
        return False
    
    def execute_buy(self, symbol: str, price: float, reason: str = "") -> bool:
        """
        Execute a buy order
        """
        available_cash = self.current_balance
        position_value = available_cash * self.position_size_pct
        
        if position_value < 100:  # Minimum $100 position
            logger.info(f"Insufficient funds for {symbol}: need ${position_value:.2f}")
            return False
        
        shares = position_value / price
        total_cost = shares * price
        
        if total_cost > self.current_balance:
            logger.info(f"Insufficient funds for {symbol}: need ${total_cost:.2f}, have ${self.current_balance:.2f}")
            return False
        
        # Execute the trade
        self.current_balance -= total_cost
        
        if symbol in self.portfolio:
            # Add to existing position
            existing_shares = self.portfolio[symbol]["shares"]
            existing_cost = existing_shares * self.portfolio[symbol]["avg_price"]
            new_avg_price = (existing_cost + total_cost) / (existing_shares + shares)
            
            self.portfolio[symbol] = {
                "shares": existing_shares + shares,
                "avg_price": new_avg_price
            }
        else:
            # New position
            self.portfolio[symbol] = {
                "shares": shares,
                "avg_price": price
            }
        
        # Record the trade
        trade = {
            "timestamp": datetime.now(),
            "action": "BUY", 
            "symbol": symbol,
            "shares": shares,
            "price": price,
            "total": total_cost,
            "balance_after": self.current_balance,
            "reason": reason
        }
        self.trade_history.append(trade)
        
        logger.info(f"BUY: {shares:.4f} shares of {symbol} @ ${price:.2f} = ${total_cost:.2f} | Balance: ${self.current_balance:.2f}")
        return True
    
    def execute_sell(self, symbol: str, price: float, reason: str = "") -> bool:
        """
        Execute a sell order (sell entire position)
        """
        if symbol not in self.portfolio:
            return False
        
        shares = self.portfolio[symbol]["shares"]
        total_proceeds = shares * price
        
        # Execute the trade
        self.current_balance += total_proceeds
        
        # Record the trade
        avg_price = self.portfolio[symbol]["avg_price"]
        profit_loss = (price - avg_price) * shares
        
        trade = {
            "timestamp": datetime.now(),
            "action": "SELL",
            "symbol": symbol, 
            "shares": shares,
            "price": price,
            "total": total_proceeds,
            "balance_after": self.current_balance,
            "profit_loss": profit_loss,
            "reason": reason
        }
        self.trade_history.append(trade)
        
        # Remove from portfolio
        del self.portfolio[symbol]
        
        logger.info(f"SELL: {shares:.4f} shares of {symbol} @ ${price:.2f} = ${total_proceeds:.2f} | P/L: ${profit_loss:.2f} | Balance: ${self.current_balance:.2f}")
        return True
    
    def get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value (cash + holdings)
        """
        total_value = self.current_balance
        
        for symbol, position in self.portfolio.items():
            try:
                # Get current price
                ticker = yf.Ticker(symbol)
                current_price = ticker.history(period="1d")['Close'].iloc[-1]
                position_value = position["shares"] * current_price
                total_value += position_value
            except:
                logger.warning(f"Could not get current price for {symbol}")
                
        return total_value
    
    def run_simulation(self, days: int = 14) -> Dict[str, Any]:
        """
        Run the complete trading simulation
        """
        logger.info(f"🚀 Starting Trading Simulation with ${self.initial_balance:,.2f}")
        logger.info("=" * 80)
        
        # Collect historical data
        logger.info("📈 Collecting historical data...")
        historical_data = self.collect_historical_data(days)
        
        if not historical_data["symbols_processed"]:
            logger.error("No historical data collected. Simulation cannot proceed.")
            return {"error": "No historical data available"}
        
        # Collect latest fundamentals for each symbol
        logger.info("📊 Analyzing fundamentals for trading decisions...")
        
        for symbol in historical_data["symbols_processed"]:
            logger.info(f"\n--- Analyzing {symbol} ---")
            
            # Get latest fundamentals
            fundamentals = self.get_latest_fundamentals(symbol)
            
            if not fundamentals:
                logger.warning(f"No fundamental data available for {symbol}")
                continue
            
            # Get current price
            try:
                current_price = fundamentals["price"]
                logger.info(f"Current Price: ${current_price:.2f}")
                
                # Check if we should buy
                if self.should_buy(symbol, fundamentals):
                    reason = "Fundamentals analysis: Buy criteria met"
                    self.execute_buy(symbol, current_price, reason)
                elif symbol in self.portfolio and self.should_sell(symbol, fundamentals):
                    reason = "Fundamentals analysis: Sell criteria triggered"
                    self.execute_sell(symbol, current_price, reason)
                else:
                    logger.info(f"{symbol}: No action taken")
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
        
        # Calculate final results
        final_portfolio_value = self.get_portfolio_value()
        total_return = final_portfolio_value - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        
        # Generate summary
        summary = {
            "initial_balance": self.initial_balance,
            "final_balance": self.current_balance,
            "portfolio_positions": len(self.portfolio),
            "final_portfolio_value": final_portfolio_value,
            "total_return": total_return,
            "return_percentage": return_pct,
            "total_trades": len(self.trade_history),
            "trade_history": self.trade_history,
            "current_portfolio": self.portfolio,
            "symbols_analyzed": historical_data["symbols_processed"]
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("🎯 SIMULATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Initial Balance:      ${self.initial_balance:>10,.2f}")
        logger.info(f"Final Cash Balance:   ${self.current_balance:>10,.2f}")
        logger.info(f"Portfolio Value:      ${final_portfolio_value:>10,.2f}")
        logger.info(f"Total Return:         ${total_return:>10,.2f}")
        logger.info(f"Return %:             {return_pct:>10.2f}%")
        logger.info(f"Total Trades:         {len(self.trade_history):>10}")
        logger.info(f"Active Positions:     {len(self.portfolio):>10}")
        
        if self.portfolio:
            logger.info("\n📈 Current Holdings:")
            for symbol, position in self.portfolio.items():
                logger.info(f"  {symbol}: {position['shares']:.4f} shares @ avg ${position['avg_price']:.2f}")
        
        if self.trade_history:
            logger.info("\n📋 Trade History:")
            for trade in self.trade_history:
                action = trade['action']
                symbol = trade['symbol']
                shares = trade['shares']
                price = trade['price']
                total = trade['total']
                reason = trade.get('reason', '')
                logger.info(f"  {action} {shares:.4f} {symbol} @ ${price:.2f} = ${total:.2f} ({reason})")
        
        return summary