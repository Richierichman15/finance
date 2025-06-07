"""
Trading Systems Package
======================
Ultra-aggressive trading strategies targeting 10% returns
"""

from .pure_5k_system import Pure5KTradingSystem
from .pure_5k_v3_system import Pure5KV3TradingSystem
from .pure_5k_v4_ai_system import Pure5KV4AITradingSystem

__all__ = ['Pure5KTradingSystem', 'Pure5KV3TradingSystem', 'Pure5KV4AITradingSystem']