#!/usr/bin/env python3
"""
Shared state manager for live trading system
Allows web dashboard to access real-time trading data
"""
import threading
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

class TradingState:
    """Singleton class to manage shared trading state"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(TradingState, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self.data_lock = threading.Lock()
        self.state_file = 'app/data/live/trading_state.json'
        
        # Initialize state
        self.reset_state()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        
        # Load existing state if available
        self.load_state()
    
    def reset_state(self):
        """Reset trading state to initial values"""
        with self.data_lock:
            self.portfolio_value = 5000.0
            self.cash = 5000.0
            self.return_percentage = 0.0
            self.positions = {}
            self.recent_trades = []
            self.daily_snapshots = []
            self.last_update = datetime.now().isoformat()
            self.trading_active = True
            self.crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'TRX-USD', 'XLM-USD']
    
    def update_portfolio(self, portfolio_value: float, cash: float, positions: Dict[str, Any]):
        """Update portfolio data"""
        with self.data_lock:
            self.portfolio_value = portfolio_value
            self.cash = cash
            self.return_percentage = ((portfolio_value - 5000.0) / 5000.0) * 100
            self.positions = positions.copy()
            self.last_update = datetime.now().isoformat()
            self.save_state()
    
    def add_trade(self, trade_data: Dict[str, Any]):
        """Add a new trade to the recent trades list"""
        with self.data_lock:
            # Add timestamp if not present
            if 'timestamp' not in trade_data:
                trade_data['timestamp'] = datetime.now().isoformat()
            
            # Add to front of list
            self.recent_trades.insert(0, trade_data)
            
            # Keep only last 20 trades
            self.recent_trades = self.recent_trades[:20]
            self.save_state()
    
    def add_daily_snapshot(self, snapshot_data: Dict[str, Any]):
        """Add a daily portfolio snapshot"""
        with self.data_lock:
            self.daily_snapshots.insert(0, snapshot_data)
            
            # Keep only last 30 days
            self.daily_snapshots = self.daily_snapshots[:30]
            self.save_state()
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current trading state"""
        with self.data_lock:
            return {
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'return_percentage': self.return_percentage,
                'positions': self.positions.copy(),
                'recent_trades': self.recent_trades.copy(),
                'daily_snapshots': self.daily_snapshots.copy(),
                'last_update': self.last_update,
                'trading_active': self.trading_active,
                'crypto_symbols': self.crypto_symbols.copy(),
                'active_positions': len([p for p in self.positions.values() if p.get('shares', 0) > 0])
            }
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get formatted position data for dashboard"""
        with self.data_lock:
            positions = []
            for symbol, pos in self.positions.items():
                if pos.get('shares', 0) > 0:
                    positions.append({
                        'symbol': symbol,
                        'shares': pos.get('shares', 0),
                        'avg_price': pos.get('avg_price', 0),
                        'current_price': pos.get('current_price', 0),
                        'market_value': pos.get('shares', 0) * pos.get('current_price', 0),
                        'unrealized_pnl': (pos.get('shares', 0) * pos.get('current_price', 0)) - (pos.get('shares', 0) * pos.get('avg_price', 0))
                    })
            return positions
    
    def save_state(self):
        """Save current state to file"""
        try:
            state_data = {
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'return_percentage': self.return_percentage,
                'positions': self.positions,
                'recent_trades': self.recent_trades,
                'daily_snapshots': self.daily_snapshots,
                'last_update': self.last_update,
                'trading_active': self.trading_active,
                'crypto_symbols': self.crypto_symbols
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save trading state: {e}")
    
    def load_state(self):
        """Load state from file if it exists"""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)
                
                with self.data_lock:
                    self.portfolio_value = state_data.get('portfolio_value', 5000.0)
                    self.cash = state_data.get('cash', 5000.0)
                    self.return_percentage = state_data.get('return_percentage', 0.0)
                    self.positions = state_data.get('positions', {})
                    self.recent_trades = state_data.get('recent_trades', [])
                    self.daily_snapshots = state_data.get('daily_snapshots', [])
                    self.last_update = state_data.get('last_update', datetime.now().isoformat())
                    self.trading_active = state_data.get('trading_active', True)
                    self.crypto_symbols = state_data.get('crypto_symbols', ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'TRX-USD', 'XLM-USD'])
        except Exception as e:
            print(f"Warning: Could not load trading state: {e}")

# Global instance
trading_state = TradingState()
