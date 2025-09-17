"""
Trading Systems Package
Contains various trading system implementations
"""

# Import only modules that exist to avoid ModuleNotFoundError at package import time.
from .pure_5k_system import Pure5KLiveTradingSystem

# Optional systems (uncomment when available)
# from .pure_5k_v3_system import Pure5KV3TradingSystem
# from .pure_5k_v4_ai_system import Pure5KV4AITradingSystem

__all__ = ['Pure5KLiveTradingSystem']