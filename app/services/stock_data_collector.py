import yfinance as yf
from sqlalchemy.orm import Session
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from ..database import SessionLocal
from ..models.database_models import StockFundamentals

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enhanced list of symbols to track - V2 Expanded Universe
SYMBOLS = [
    # Original ETFs
    'SPY',     # S&P 500 ETF
    'XLE',     # Energy ETF
    'GLD',     # Gold ETF
    'QQQ',     # NASDAQ ETF
    'VTI',     # Total Stock Market ETF
    
    # Original Cryptocurrencies
    'BTC-USD', # Bitcoin
    'XRP-USD', # XRP (Ripple)
    'ETH-USD', # Ethereum
    
    # NEW Cryptocurrencies (V2 Expansion)
    'SOL-USD', # Solana
    'TRX-USD', # Tron
    'ADA-USD', # Cardano
    'XLM-USD', # Stellar Lumens
    
    # NEW Energy Sector Stocks (V2 Expansion)
    'XEG',     # Energy Equipment ETF
    'KOLD',    # Natural Gas Bear ETF
    'UNG',     # Natural Gas ETF
    'USO',     # Oil ETF
    'NEE',     # NextEra Energy (Electrical)
    'DUK',     # Duke Energy (Electrical)
    
    # NEW Tech Sector Stocks (V2 Expansion)
    'NVDA',    # NVIDIA
    'MSFT',    # Microsoft
    'GOOGL',   # Google/Alphabet
    'TSLA',    # Tesla
    'AMD',     # AMD
]

class StockDataCollector:
    def __init__(self):
        # Updated portfolio with ETFs and crypto
        self.tracked_stocks = SYMBOLS
    
    def collect_and_store_fundamentals(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Collect fundamental data for specified stocks and store in database
        """
        if symbols is None:
            symbols = self.tracked_stocks
        
        db = SessionLocal()
        results = {
            "timestamp": datetime.now(),
            "symbols_processed": [],
            "symbols_failed": [],
            "total_records_created": 0
        }
        
        try:
            for symbol in symbols:
                try:
                    logger.info(f"Collecting data for {symbol}")
                    fundamentals_data = self._fetch_stock_fundamentals(symbol)
                    
                    if fundamentals_data:
                        # Store in database
                        db_record = StockFundamentals(**fundamentals_data)
                        db.add(db_record)
                        db.commit()
                        
                        results["symbols_processed"].append(symbol)
                        results["total_records_created"] += 1
                        logger.info(f"Successfully stored data for {symbol}")
                    else:
                        results["symbols_failed"].append(symbol)
                        logger.warning(f"No data retrieved for {symbol}")
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    results["symbols_failed"].append(symbol)
                    db.rollback()
                    
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            db.rollback()
        finally:
            db.close()
            
        return results
    
    def _fetch_stock_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch specific fundamental metrics using yfinance
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get current price
            hist = ticker.history(period="1d")
            current_price = hist['Close'].iloc[-1] if not hist.empty else None
            
            if current_price is None:
                logger.warning(f"No price data available for {symbol}")
                return None
            
            # Extract fundamental metrics with safe gets and type conversion
            fundamentals = {
                "symbol": symbol,
                "price": self._safe_float(current_price),
                "market_cap": self._safe_float(info.get('marketCap')),
                "pe_ratio_ttm": self._safe_float(info.get('trailingPE')),
                "peg_ratio": self._safe_float(info.get('pegRatio')),
                "eps": self._safe_float(info.get('trailingEps')),
                "free_cash_flow": self._safe_float(info.get('freeCashflow')),
                "fcf_yield": self._calculate_fcf_yield(
                    info.get('freeCashflow'), 
                    info.get('marketCap')
                ),
                "debt_to_equity": self._safe_float(info.get('debtToEquity')),
                "current_ratio": self._safe_float(info.get('currentRatio')),
                "dividend_yield": self._safe_float(info.get('dividendYield')),
                "roe": self._safe_float(info.get('returnOnEquity')),
                "timestamp": datetime.now()
            }
            
            # Log the collected data
            logger.info(f"Collected fundamentals for {symbol}: Price=${fundamentals['price']:.2f}, P/E={fundamentals['pe_ratio_ttm']}")
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {str(e)}")
            return None
    
    def _safe_float(self, value) -> Optional[float]:
        """
        Safely convert value to float, handling None and invalid values
        """
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                # Remove any non-numeric characters except decimal point and minus
                cleaned = ''.join(c for c in value if c.isdigit() or c in '.-')
                return float(cleaned) if cleaned else None
            else:
                return float(value)
        except (ValueError, TypeError):
            return None
    
    def _calculate_fcf_yield(self, free_cash_flow, market_cap) -> Optional[float]:
        """
        Calculate FCF Yield = Free Cash Flow / Market Cap
        """
        try:
            if free_cash_flow and market_cap and market_cap != 0:
                return (free_cash_flow / market_cap) * 100  # Convert to percentage
            return None
        except (TypeError, ZeroDivisionError):
            return None
    
    def get_latest_fundamentals(self, symbol: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get latest fundamental records for a symbol from database
        """
        db = SessionLocal()
        try:
            records = db.query(StockFundamentals)\
                       .filter(StockFundamentals.symbol == symbol)\
                       .order_by(StockFundamentals.timestamp.desc())\
                       .limit(limit)\
                       .all()
            
            return [
                {
                    "symbol": record.symbol,
                    "price": record.price,
                    "market_cap": record.market_cap,
                    "pe_ratio_ttm": record.pe_ratio_ttm,
                    "peg_ratio": record.peg_ratio,
                    "eps": record.eps,
                    "free_cash_flow": record.free_cash_flow,
                    "fcf_yield": record.fcf_yield,
                    "debt_to_equity": record.debt_to_equity,
                    "current_ratio": record.current_ratio,
                    "dividend_yield": record.dividend_yield,
                    "roe": record.roe,
                    "timestamp": record.timestamp
                }
                for record in records
            ]
        finally:
            db.close()
    
    def get_all_latest_fundamentals(self, limit_per_symbol: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get latest fundamentals for all tracked symbols
        """
        result = {}
        for symbol in self.tracked_stocks:
            result[symbol] = self.get_latest_fundamentals(symbol, limit_per_symbol)
        return result 