"""
Trading Systems Package
======================
Ultra-aggressive trading strategies targeting 10% returns
"""

from .pure_5k_system import Pure5KTradingSystem
from .pure_5k_v3_system import Pure5KV3TradingSystem

__all__ = ['Pure5KTradingSystem', 'Pure5KV3TradingSystem']