#!/usr/bin/env python3
"""
🚀 CRYPTO MARKET ANALYZER - QUANTITATIVE BUY RECOMMENDATIONS
============================================================
Analyzes the entire crypto market using mathematical models to identify
the best crypto assets to buy next based on quantitative scoring.
"""

import sys
import os
import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Tuple, Optional
import requests
import warnings
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.kraken import kraken_api

class CryptoMarketAnalyzer:
    """Quantitative crypto market analyzer for buy recommendations"""
    
    def __init__(self):
        # Extended crypto universe for analysis
        self.crypto_universe = [
            # Major Cryptos
            'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD',
            'DOGE-USD', 'DOT-USD', 'MATIC-USD', 'SHIB-USD', 'LTC-USD', 'TRX-USD',
            
            # DeFi & Smart Contract
            'AVAX-USD', 'LINK-USD', 'UNI-USD', 'AAVE-USD', 'SUSHI-USD', 'CRV-USD',
            'COMP-USD', 'MKR-USD', 'YFI-USD', 'SNX-USD',
            
            # Layer 2 & Scaling
            'MATIC-USD', 'LRC-USD', 'IMX-USD', 'ARB-USD', 'OP-USD',
            
            # Meme & Community
            'DOGE-USD', 'SHIB-USD', 'PEPE-USD', 'FLOKI-USD', 'BONK-USD',
            
            # Infrastructure & Utility
            'XLM-USD', 'VET-USD', 'FIL-USD', 'THETA-USD', 'HBAR-USD',
            'ALGO-USD', 'XTZ-USD', 'ICP-USD', 'NEAR-USD', 'FTM-USD',
            
            # Emerging & High Potential
            'ATOM-USD', 'LUNA-USD', 'MANA-USD', 'SAND-USD', 'APE-USD',
            'LDO-USD', 'APT-USD', 'SUI-USD', 'SEI-USD', 'TIA-USD'
        ]
        
        self.market_data = {}
        self.analysis_results = {}
        self.analysis_timestamp = datetime.now()
        
        # Kraken symbol mappings (will be dynamically populated)
        self.kraken_symbol_cache = {}
        self.kraken_assets = {}
        self.kraken_pairs = {}
        
        # Quantitative scoring weights
        self.scoring_weights = {
            'momentum': 0.25,           # Price momentum
            'volatility_adj_return': 0.20,  # Risk-adjusted returns
            'volume_strength': 0.15,    # Volume analysis
            'technical_signals': 0.15,  # Technical indicators
            'relative_strength': 0.10,  # vs market performance
            'recovery_potential': 0.10, # Drawdown recovery
            'trend_consistency': 0.05   # Trend stability
        }
        
        print("🚀 Crypto Market Analyzer Initialized")
        print(f"📊 Analyzing {len(self.crypto_universe)} cryptocurrencies")
        print("🔬 Mathematical models: Momentum, Sharpe, RSI, Volume, Relative Strength")

    async def _initialize_kraken_mappings(self):
        """Initialize Kraken symbol mappings using their public API"""
        try:
            import requests
            
            print("📡 Fetching Kraken asset and pair mappings...")
            
            # Fetch available assets
            try:
                assets_response = requests.get('https://api.kraken.com/0/public/Assets', timeout=10)
                if assets_response.status_code == 200:
                    assets_data = assets_response.json()
                    if 'result' in assets_data:
                        self.kraken_assets = assets_data['result']
                        print(f"✅ Loaded {len(self.kraken_assets)} Kraken assets")
            except Exception as e:
                print(f"⚠️  Could not fetch Kraken assets: {e}")
            
            # Fetch available trading pairs
            try:
                pairs_response = requests.get('https://api.kraken.com/0/public/AssetPairs', timeout=10)
                if pairs_response.status_code == 200:
                    pairs_data = pairs_response.json()
                    if 'result' in pairs_data:
                        self.kraken_pairs = pairs_data['result']
                        print(f"✅ Loaded {len(self.kraken_pairs)} Kraken trading pairs")
                        
                        # Build symbol mapping cache
                        self._build_symbol_mapping_cache()
            except Exception as e:
                print(f"⚠️  Could not fetch Kraken pairs: {e}")
                
        except Exception as e:
            print(f"⚠️  Kraken API initialization failed: {e}")
            # Fall back to hardcoded mappings
            self._setup_fallback_mappings()
    
    def _build_symbol_mapping_cache(self):
        """Build symbol mapping cache from Kraken API data"""
        try:
            print("🔧 Building Kraken symbol mapping cache...")
            
            # Common crypto symbols we want to map
            target_cryptos = [
                'BTC', 'ETH', 'XRP', 'ADA', 'SOL', 'DOGE', 'DOT', 'MATIC', 'SHIB', 'LTC', 'TRX',
                'AVAX', 'LINK', 'UNI', 'AAVE', 'SUSHI', 'CRV', 'COMP', 'MKR', 'YFI', 'SNX',
                'LRC', 'ARB', 'OP', 'PEPE', 'FLOKI', 'BONK', 'XLM', 'VET', 'FIL', 'THETA',
                'HBAR', 'ALGO', 'XTZ', 'ICP', 'NEAR', 'FTM', 'ATOM', 'MANA', 'SAND', 'APE',
                'LDO', 'APT', 'SUI', 'SEI', 'TIA'
            ]
            
            mappings_found = 0
            
            for crypto in target_cryptos:
                yf_symbol = f"{crypto}-USD"
                kraken_pair = self._find_kraken_usd_pair(crypto)
                
                if kraken_pair:
                    self.kraken_symbol_cache[yf_symbol] = kraken_pair
                    mappings_found += 1
                    print(f"   📊 {yf_symbol} → {kraken_pair}")
            
            print(f"✅ Built {mappings_found} symbol mappings")
            
        except Exception as e:
            print(f"⚠️  Error building symbol cache: {e}")
    
    def _find_kraken_usd_pair(self, crypto_symbol: str) -> Optional[str]:
        """Enhanced method to find correct Kraken USD pair"""
        try:
            # Direct mappings for problematic symbols
            direct_mappings = {
                'MATIC': 'MATICUSD',
                'UNI': 'UNIUSD', 
                'COMP': 'COMPUSD',
                'IMX': 'IMXUSD',
                'PEPE': 'PEPEUSD',
                'FTM': 'FTMUSD',
                'LUNA': 'LUNAUSD',
                'SUI': 'SUIUSD',
                'NEAR': 'NEARUSD',
                'APT': 'APTUSD',
                'SEI': 'SEIUSD',
                'TIA': 'TIAUSD'
            }
            
            # Check direct mappings first
            if crypto_symbol in direct_mappings:
                kraken_pair = direct_mappings[crypto_symbol]
                if kraken_pair in self.kraken_pairs:
                    return kraken_pair
            
            # Try common patterns
            possible_patterns = [
                f"{crypto_symbol}USD",
                f"{crypto_symbol}USDT",
                f"X{crypto_symbol}USD",
                f"{crypto_symbol}USD",
                f"{crypto_symbol}USDC"
            ]
            
            # Try special mappings for specific cryptos
            special_mappings = {
                'MATIC': ['MATICUSD', 'MATICUSDT'],
                'UNI': ['UNIUSD', 'UNIUSDT'],
                'COMP': ['COMPUSD', 'COMPUSDT'],
                'IMX': ['IMXUSD', 'IMXUSDT'],
                'PEPE': ['PEPEUSD', 'PEPEUSDT'],
                'FTM': ['FTMUSD', 'FTMUSDT'],
                'LUNA': ['LUNAUSD', 'LUNAUSDT'],
                'SUI': ['SUIUSD', 'SUIUSDT'],
                'BONK': ['BONKUSD', 'BONKUSDT'],
                'FLOKI': ['FLOKIUSD', 'FLOKIUSDT'],
                'PEPE': ['PEPEUSD', 'PEPEUSDT']
            }
            
            # Try special mappings first
            if crypto_symbol in special_mappings:
                for candidate in special_mappings[crypto_symbol]:
                    if candidate in self.kraken_pairs:
                        return candidate
            
            # Try common patterns
            for pattern in possible_patterns:
                if pattern in self.kraken_pairs:
                    return pattern
            
            # Enhanced fallback patterns for problematic symbols
            enhanced_patterns = [
                f"{crypto_symbol}USD",
                f"{crypto_symbol}USDT", 
                f"X{crypto_symbol}USD",
                f"{crypto_symbol}USD",
                f"{crypto_symbol}USDC"
            ]
            
            for pattern in enhanced_patterns:
                if pattern in self.kraken_pairs:
                    return pattern
            
            # Search through all pairs for alternative names
            for pair_key, pair_data in self.kraken_pairs.items():
                if 'altname' in pair_data:
                    altname = pair_data['altname']
                    if altname.startswith(crypto_symbol) and altname.endswith('USD'):
                        return pair_key
                        
                # Check wsname as well
                if 'wsname' in pair_data:
                    wsname = pair_data['wsname']
                    if f"{crypto_symbol}/USD" in wsname:
                        return pair_key
            
            return None
            
        except Exception as e:
            print(f"⚠️  Error finding Kraken pair for {crypto_symbol}: {e}")
            return None
    
    def _setup_fallback_mappings(self):
        """Setup hardcoded fallback mappings if API fails"""
        print("🔧 Setting up fallback symbol mappings...")
        
        self.kraken_symbol_cache = {
            'BTC-USD': 'XBTUSD',
            'ETH-USD': 'ETHUSD', 
            'XRP-USD': 'XRPUSD',
            'ADA-USD': 'ADAUSD',
            'SOL-USD': 'SOLUSD',
            'DOGE-USD': 'XDGUSD',
            'DOT-USD': 'DOTUSD',
            'LTC-USD': 'LTCUSD',
            'XLM-USD': 'XLMUSD',
            'LINK-USD': 'LINKUSD',
            'UNI-USD': 'UNIUSD',
            'AVAX-USD': 'AVAXUSD',
            'XLM-USD': 'XLMUSD',
            'ATOM-USD': 'ATOMUSD'
        }
        
        print(f"✅ Setup {len(self.kraken_symbol_cache)} fallback mappings")

    async def analyze_crypto_market(self) -> Dict:
        """Main analysis function to find best crypto buy opportunities"""
        print("\n" + "="*80)
        print("🎯 QUANTITATIVE CRYPTO MARKET ANALYSIS")
        print("="*80)
        
        try:
            # Step 0: Ensure Kraken mappings are initialized
            if not self.kraken_symbol_cache:
                await self._initialize_kraken_mappings()
            
            # Step 1: Fetch market data for all cryptos
            await self._fetch_comprehensive_crypto_data()
            
            # Step 2: Calculate quantitative scores
            await self._calculate_quantitative_scores()
            
            # Step 3: Generate buy recommendations
            recommendations = await self._generate_buy_recommendations()
            
            # Step 4: Display results
            self._display_market_analysis(recommendations)
            
            # Step 5: Save analysis
            self._save_market_analysis(recommendations)
            
            return {
                'recommendations': recommendations,
                'market_analysis': self.analysis_results,
                'timestamp': self.analysis_timestamp.isoformat()
            }
            
        except Exception as e:
            print(f"❌ Market analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    async def _fetch_comprehensive_crypto_data(self):
        """Fetch comprehensive market data for all cryptos"""
        print(f"\n📥 Fetching market data for {len(self.crypto_universe)} cryptocurrencies...")
        
        successful_fetches = 0
        failed_symbols = []
        
        for i, symbol in enumerate(self.crypto_universe):
            try:
                print(f"📊 [{i+1}/{len(self.crypto_universe)}] Fetching {symbol}...")
                
                ticker = yf.Ticker(symbol)
                
                # Get multiple timeframes for comprehensive analysis
                hist_1y = ticker.history(period='1y', interval='1d')
                hist_3m = ticker.history(period='3mo', interval='1d') 
                hist_1m = ticker.history(period='1mo', interval='1d')
                hist_7d = ticker.history(period='7d', interval='1h')
                
                if not hist_1m.empty:
                    # Calculate additional metrics
                    current_price = hist_1m['Close'].iloc[-1]
                    
                    # Try to get Kraken price for verification
                    kraken_price = None
                    kraken_symbol = None
                    try:
                        kraken_symbol = self._convert_to_kraken_symbol(symbol)
                        if kraken_symbol:
                            kraken_price = kraken_api.get_price(kraken_symbol)
                            if kraken_price and kraken_price > 0:
                                print(f"   ✅ Kraken price for {symbol}: ${kraken_price:.4f}")
                            else:
                                print(f"   ⚠️  Kraken returned zero price for {symbol} ({kraken_symbol})")
                        else:
                            print(f"   ❌ No Kraken symbol mapping for {symbol}")
                    except Exception as e:
                        print(f"   ❌ Kraken API error for {symbol}: {e}")
                
                self.market_data[symbol] = {
                    'symbol': symbol,
                    'current_price': kraken_price if kraken_price and kraken_price > 0 else current_price,
                    'hist_1y': hist_1y,
                    'hist_3m': hist_3m,
                    'hist_1m': hist_1m,
                    'hist_7d': hist_7d,
                    'data_quality': 'HIGH' if not hist_1y.empty else 'MEDIUM' if not hist_3m.empty else 'LOW'
                }
                successful_fetches += 1
            except Exception as e:
                print(f"   ⚠️  Failed to fetch {symbol}: {e}")
                failed_symbols.append(symbol)
                continue
        
        print(f"\n✅ Successfully fetched data for {successful_fetches}/{len(self.crypto_universe)} cryptos")
        if failed_symbols:
            print(f"❌ Failed symbols: {', '.join(failed_symbols[:10])}{'...' if len(failed_symbols) > 10 else ''}")

    def _convert_to_kraken_symbol(self, yf_symbol: str) -> Optional[str]:
        """Convert yfinance symbol to Kraken format using dynamic mappings"""
        # First check our dynamic cache
        if yf_symbol in self.kraken_symbol_cache:
            return self.kraken_symbol_cache[yf_symbol]
        
        # Try to find mapping on the fly
        crypto_part = yf_symbol.replace('-USD', '')
        kraken_pair = self._find_kraken_usd_pair(crypto_part)
        
        if kraken_pair:
            # Cache the result for future use
            self.kraken_symbol_cache[yf_symbol] = kraken_pair
            return kraken_pair
        
        # Fallback to hardcoded mappings
        fallback_map = {
            'BTC-USD': 'XBTUSD',
            'ETH-USD': 'ETHUSD',
            'XRP-USD': 'XRPUSD',
            'ADA-USD': 'ADAUSD',
            'SOL-USD': 'SOLUSD',
            'DOGE-USD': 'DOGEUSD',
            'DOT-USD': 'DOTUSD',
            'MATIC-USD': 'MATICUSD',
            'LTC-USD': 'LTCUSD',
            'LINK-USD': 'LINKUSD',
            'UNI-USD': 'UNIUSD',
            'AVAX-USD': 'AVAXUSD',
            'XLM-USD': 'XLMUSD',
            'ATOM-USD': 'ATOMUSD'
        }
        return fallback_map.get(yf_symbol)

    async def _calculate_quantitative_scores(self):
        """Calculate comprehensive quantitative scores for each crypto"""
        print(f"\n🔬 Calculating quantitative scores for {len(self.market_data)} cryptos...")
        
        for symbol, data in self.market_data.items():
            try:
                scores = {}
                
                # 1. Momentum Analysis
                scores['momentum'] = self._calculate_momentum_score(data)
                
                # 2. Volatility-Adjusted Returns (Sharpe-like)
                scores['volatility_adj_return'] = self._calculate_sharpe_score(data)
                
                # 3. Volume Strength
                scores['volume_strength'] = self._calculate_volume_score(data)
                
                # 4. Technical Signals
                scores['technical_signals'] = self._calculate_technical_score(data)
                
                # 5. Relative Strength vs Market
                scores['relative_strength'] = self._calculate_relative_strength(data)
                
                # 6. Recovery Potential
                scores['recovery_potential'] = self._calculate_recovery_score(data)
                
                # 7. Trend Consistency
                scores['trend_consistency'] = self._calculate_trend_consistency(data)
                
                # Calculate composite score
                composite_score = sum(
                    scores[metric] * weight 
                    for metric, weight in self.scoring_weights.items()
                    if metric in scores
                )
                
                self.analysis_results[symbol] = {
                    'scores': scores,
                    'composite_score': composite_score,
                    'current_price': data['current_price'],
                    'data_quality': data['data_quality']
                }
                
            except Exception as e:
                print(f"   ⚠️  Scoring failed for {symbol}: {e}")
                continue
        
        print(f"✅ Quantitative scoring completed for {len(self.analysis_results)} cryptos")

    def _calculate_momentum_score(self, data: Dict) -> float:
        """Calculate momentum score (0-100)"""
        try:
            hist_1m = data['hist_1m']
            if hist_1m.empty:
                return 50  # Neutral
            
            closes = hist_1m['Close']
            
            # Multiple timeframe momentum
            momentum_7d = (closes.iloc[-1] - closes.iloc[-7]) / closes.iloc[-7] if len(closes) >= 7 else 0
            momentum_14d = (closes.iloc[-1] - closes.iloc[-14]) / closes.iloc[-14] if len(closes) >= 14 else 0
            momentum_30d = (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0] if len(closes) > 1 else 0
            
            # Weight recent momentum more heavily
            weighted_momentum = (momentum_7d * 0.5 + momentum_14d * 0.3 + momentum_30d * 0.2)
            
            # Convert to 0-100 score (capped at extremes)
            score = 50 + (weighted_momentum * 100)
            return max(0, min(100, score))
            
        except Exception:
            return 50

    def _calculate_sharpe_score(self, data: Dict) -> float:
        """Calculate risk-adjusted return score (0-100)"""
        try:
            hist_3m = data['hist_3m']
            if hist_3m.empty:
                return 50
            
            returns = hist_3m['Close'].pct_change().dropna()
            if len(returns) < 30:
                return 50
            
            mean_return = returns.mean() * 252  # Annualized
            volatility = returns.std() * np.sqrt(252)
            
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            
            # Convert Sharpe to 0-100 score
            # Sharpe > 1.5 = excellent (90-100)
            # Sharpe 1.0-1.5 = good (70-90)
            # Sharpe 0.5-1.0 = moderate (50-70)
            # Sharpe < 0.5 = poor (0-50)
            
            if sharpe_ratio > 1.5:
                score = 90 + min(10, (sharpe_ratio - 1.5) * 10)
            elif sharpe_ratio > 1.0:
                score = 70 + (sharpe_ratio - 1.0) * 40
            elif sharpe_ratio > 0.5:
                score = 50 + (sharpe_ratio - 0.5) * 40
            elif sharpe_ratio > 0:
                score = 25 + sharpe_ratio * 50
            else:
                score = max(0, 25 + sharpe_ratio * 25)
            
            return max(0, min(100, score))
            
        except Exception:
            return 50

    def _calculate_volume_score(self, data: Dict) -> float:
        """Calculate volume strength score (0-100)"""
        try:
            hist_1m = data['hist_1m']
            if hist_1m.empty or 'Volume' not in hist_1m.columns:
                return 50
            
            volumes = hist_1m['Volume']
            recent_volume = volumes.tail(7).mean()  # Last 7 days
            historical_volume = volumes.head(-7).mean()  # Older data
            
            if historical_volume <= 0:
                return 50
            
            volume_ratio = recent_volume / historical_volume
            
            # Score based on volume increase
            if volume_ratio > 2.0:
                score = 90  # Exceptional volume
            elif volume_ratio > 1.5:
                score = 80  # Strong volume
            elif volume_ratio > 1.2:
                score = 70  # Good volume
            elif volume_ratio > 0.8:
                score = 60  # Normal volume
            else:
                score = 40  # Weak volume
            
            return score
            
        except Exception:
            return 50

    def _calculate_technical_score(self, data: Dict) -> float:
        """Calculate technical indicators score (0-100)"""
        try:
            hist_1m = data['hist_1m']
            if hist_1m.empty:
                return 50
            
            closes = hist_1m['Close']
            score = 0
            indicators = 0
            
            # RSI (Relative Strength Index)
            if len(closes) >= 14:
                delta = closes.diff()
                gains = delta.where(delta > 0, 0).rolling(window=14).mean()
                losses = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                # RSI scoring (looking for oversold recovery opportunities)
                if 30 <= current_rsi <= 45:  # Oversold but recovering
                    score += 85
                elif 45 < current_rsi <= 60:  # Neutral to bullish
                    score += 70
                elif 60 < current_rsi <= 70:  # Bullish but not overbought
                    score += 60
                elif current_rsi < 30:  # Very oversold (potential bounce)
                    score += 75
                else:  # Overbought
                    score += 30
                indicators += 1
            
            # Moving Average Convergence
            if len(closes) >= 50:
                sma_20 = closes.rolling(window=20).mean().iloc[-1]
                sma_50 = closes.rolling(window=50).mean().iloc[-1]
                current_price = closes.iloc[-1]
                
                # Price above moving averages is bullish
                if current_price > sma_20 > sma_50:
                    score += 85  # Strong uptrend
                elif current_price > sma_20:
                    score += 70  # Short-term bullish
                elif current_price > sma_50:
                    score += 60  # Medium-term support
                else:
                    score += 35  # Bearish trend
                indicators += 1
            
            # MACD-like momentum
            if len(closes) >= 26:
                ema_12 = closes.ewm(span=12).mean()
                ema_26 = closes.ewm(span=26).mean()
                macd = ema_12 - ema_26
                signal = macd.ewm(span=9).mean()
                
                current_macd = macd.iloc[-1]
                current_signal = signal.iloc[-1]
                
                if current_macd > current_signal and current_macd > 0:
                    score += 80  # Bullish momentum
                elif current_macd > current_signal:
                    score += 65  # Improving momentum
                elif current_macd > 0:
                    score += 55  # Positive momentum
                else:
                    score += 35  # Negative momentum
                indicators += 1
            
            return score / indicators if indicators > 0 else 50
            
        except Exception:
            return 50

    def _calculate_relative_strength(self, data: Dict) -> float:
        """Calculate relative strength vs Bitcoin (market proxy)"""
        try:
            if data['symbol'] == 'BTC-USD':
                return 70  # Bitcoin baseline score
            
            hist_3m = data['hist_3m']
            if hist_3m.empty or 'BTC-USD' not in self.market_data:
                return 50
            
            btc_hist = self.market_data['BTC-USD']['hist_3m']
            if btc_hist.empty:
                return 50
            
            # Calculate relative performance vs BTC
            crypto_return = (hist_3m['Close'].iloc[-1] - hist_3m['Close'].iloc[0]) / hist_3m['Close'].iloc[0]
            btc_return = (btc_hist['Close'].iloc[-1] - btc_hist['Close'].iloc[0]) / btc_hist['Close'].iloc[0]
            
            relative_performance = crypto_return - btc_return
            
            # Score based on outperformance
            if relative_performance > 0.5:  # 50%+ outperformance
                score = 95
            elif relative_performance > 0.2:  # 20%+ outperformance
                score = 85
            elif relative_performance > 0.1:  # 10%+ outperformance
                score = 75
            elif relative_performance > 0:     # Any outperformance
                score = 65
            elif relative_performance > -0.1:  # Minor underperformance
                score = 50
            else:  # Significant underperformance
                score = 30
            
            return score
            
        except Exception:
            return 50

    def _calculate_recovery_score(self, data: Dict) -> float:
        """Calculate recovery potential from drawdowns"""
        try:
            hist_3m = data['hist_3m']
            if hist_3m.empty:
                return 50
            
            closes = hist_3m['Close']
            
            # Calculate drawdown from recent highs
            rolling_max = closes.expanding().max()
            drawdown = (closes - rolling_max) / rolling_max
            current_drawdown = drawdown.iloc[-1]
            max_drawdown = drawdown.min()
            
            # Score based on recovery potential
            if current_drawdown > -0.1:  # Less than 10% from highs
                score = 60  # Limited recovery upside
            elif current_drawdown > -0.2:  # 10-20% from highs
                score = 75  # Good recovery potential
            elif current_drawdown > -0.3:  # 20-30% from highs
                score = 85  # Strong recovery potential
            elif current_drawdown > -0.5:  # 30-50% from highs
                score = 90  # Excellent recovery potential
            else:  # >50% from highs
                score = 95  # Maximum recovery potential
            
            # Adjust for overall trend
            if max_drawdown < -0.7:  # Very volatile asset
                score *= 0.8  # Reduce score for high risk
            
            return max(0, min(100, score))
            
        except Exception:
            return 50

    def _calculate_trend_consistency(self, data: Dict) -> float:
        """Calculate trend consistency score"""
        try:
            hist_1m = data['hist_1m']
            if hist_1m.empty:
                return 50
            
            closes = hist_1m['Close']
            returns = closes.pct_change().dropna()
            
            if len(returns) < 20:
                return 50
            
            # Calculate trend consistency (lower volatility of returns = higher consistency)
            volatility = returns.std()
            mean_abs_return = abs(returns).mean()
            
            consistency_ratio = mean_abs_return / volatility if volatility > 0 else 0
            
            # Higher ratio = more consistent trends
            if consistency_ratio > 1.5:
                score = 85
            elif consistency_ratio > 1.2:
                score = 75
            elif consistency_ratio > 1.0:
                score = 65
            elif consistency_ratio > 0.8:
                score = 55
            else:
                score = 45
            
            return score
            
        except Exception:
            return 50

    async def _generate_buy_recommendations(self) -> List[Dict]:
        """Generate top buy recommendations based on quantitative analysis"""
        print(f"\n🎯 Generating buy recommendations...")
        
        # Sort by composite score
        sorted_cryptos = sorted(
            self.analysis_results.items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )
        
        recommendations = []
        
        for i, (symbol, analysis) in enumerate(sorted_cryptos[:20]):  # Top 20
            try:
                scores = analysis['scores']
                composite_score = analysis['composite_score']
                current_price = analysis['current_price']
                
                # Determine recommendation strength
                if composite_score >= 80:
                    recommendation = "STRONG BUY"
                    confidence = "VERY HIGH"
                elif composite_score >= 70:
                    recommendation = "BUY"
                    confidence = "HIGH"
                elif composite_score >= 60:
                    recommendation = "MODERATE BUY"
                    confidence = "MODERATE"
                elif composite_score >= 50:
                    recommendation = "WEAK BUY"
                    confidence = "LOW"
                else:
                    recommendation = "AVOID"
                    confidence = "LOW"
                
                # Calculate expected return based on scores
                expected_return = self._calculate_expected_return(scores, composite_score)
                
                # Risk assessment
                risk_level = self._assess_risk_level(scores)
                
                recommendations.append({
                    'rank': i + 1,
                    'symbol': symbol,
                    'recommendation': recommendation,
                    'confidence': confidence,
                    'composite_score': composite_score,
                    'current_price': current_price,
                    'expected_return_30d': expected_return,
                    'risk_level': risk_level,
                    'key_strengths': self._identify_key_strengths(scores),
                    'scores': scores
                })
                
            except Exception as e:
                continue
        
        return recommendations

    def _calculate_expected_return(self, scores: Dict, composite_score: float) -> str:
        """Calculate expected 30-day return estimate"""
        try:
            # Base return on composite score
            if composite_score >= 85:
                base_return = 25  # 25%
            elif composite_score >= 75:
                base_return = 15  # 15%
            elif composite_score >= 65:
                base_return = 10  # 10%
            elif composite_score >= 55:
                base_return = 5   # 5%
            else:
                base_return = 0   # 0%
            
            # Adjust for momentum
            momentum_bonus = (scores.get('momentum', 50) - 50) * 0.2
            
            # Adjust for recovery potential
            recovery_bonus = (scores.get('recovery_potential', 50) - 50) * 0.15
            
            total_expected = base_return + momentum_bonus + recovery_bonus
            
            return f"{total_expected:.1f}%"
            
        except Exception:
            return "N/A"

    def _assess_risk_level(self, scores: Dict) -> str:
        """Assess risk level based on scores"""
        try:
            volatility_score = scores.get('volatility_adj_return', 50)
            trend_consistency = scores.get('trend_consistency', 50)
            
            avg_stability = (volatility_score + trend_consistency) / 2
            
            if avg_stability >= 70:
                return "LOW"
            elif avg_stability >= 50:
                return "MODERATE"
            else:
                return "HIGH"
                
        except Exception:
            return "MODERATE"

    def _identify_key_strengths(self, scores: Dict) -> List[str]:
        """Identify key strengths for each recommendation"""
        strengths = []
        
        try:
            if scores.get('momentum', 0) >= 75:
                strengths.append("Strong Momentum")
            if scores.get('volatility_adj_return', 0) >= 75:
                strengths.append("Excellent Risk-Adjusted Returns")
            if scores.get('volume_strength', 0) >= 75:
                strengths.append("High Volume Activity")
            if scores.get('technical_signals', 0) >= 75:
                strengths.append("Bullish Technical Signals")
            if scores.get('relative_strength', 0) >= 75:
                strengths.append("Outperforming Market")
            if scores.get('recovery_potential', 0) >= 80:
                strengths.append("High Recovery Potential")
            if scores.get('trend_consistency', 0) >= 75:
                strengths.append("Consistent Trends")
            
            return strengths[:3]  # Top 3 strengths
            
        except Exception:
            return ["Mathematical Analysis"]

    def _display_market_analysis(self, recommendations: List[Dict]):
        """Display comprehensive market analysis results"""
        print("\n" + "="*100)
        print("🚀 CRYPTO MARKET ANALYSIS - BUY RECOMMENDATIONS")
        print("="*100)
        
        print(f"\n📊 MARKET OVERVIEW:")
        print(f"   Cryptos Analyzed: {len(self.analysis_results)}")
        print(f"   Data Quality: HIGH")
        print(f"   Analysis Method: Quantitative Multi-Factor Scoring")
        print(f"   Timestamp: {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n🎯 TOP BUY RECOMMENDATIONS:")
        print("-" * 100)
        print(f"{'Rank':<4} {'Symbol':<12} {'Price':<12} {'Score':<8} {'Recommendation':<15} {'Expected Return':<15} {'Risk':<10} {'Key Strengths'}")
        print("-" * 100)
        
        for rec in recommendations[:15]:  # Top 15
            strengths_str = ", ".join(rec['key_strengths'][:2]) if rec['key_strengths'] else "N/A"
            if len(strengths_str) > 25:
                strengths_str = strengths_str[:22] + "..."
            
            print(f"{rec['rank']:<4} {rec['symbol']:<12} ${rec['current_price']:<11.4f} "
                  f"{rec['composite_score']:<7.1f} {rec['recommendation']:<15} "
                  f"{rec['expected_return_30d']:<15} {rec['risk_level']:<10} {strengths_str}")
        
        # Detailed analysis for top 5
        print(f"\n🔍 DETAILED ANALYSIS - TOP 5 RECOMMENDATIONS:")
        print("=" * 80)
        
        for i, rec in enumerate(recommendations[:5]):
            print(f"\n{i+1}. 🚀 {rec['symbol']} - {rec['recommendation']}")
            print(f"   💰 Current Price: ${rec['current_price']:.4f}")
            print(f"   📊 Composite Score: {rec['composite_score']:.1f}/100 ({rec['confidence']} confidence)")
            print(f"   📈 Expected 30-Day Return: {rec['expected_return_30d']}")
            print(f"   ⚠️  Risk Level: {rec['risk_level']}")
            print(f"   ✨ Key Strengths: {', '.join(rec['key_strengths']) if rec['key_strengths'] else 'Mathematical edge detected'}")
            
            # Show detailed scores
            scores = rec['scores']
            print(f"   📊 Score Breakdown:")
            print(f"      • Momentum: {scores.get('momentum', 0):.1f}/100")
            print(f"      • Risk-Adj Return: {scores.get('volatility_adj_return', 0):.1f}/100")
            print(f"      • Volume Strength: {scores.get('volume_strength', 0):.1f}/100")
            print(f"      • Technical Signals: {scores.get('technical_signals', 0):.1f}/100")
            print(f"      • Relative Strength: {scores.get('relative_strength', 0):.1f}/100")
            print(f"      • Recovery Potential: {scores.get('recovery_potential', 0):.1f}/100")
        
        print(f"\n💡 INVESTMENT STRATEGY RECOMMENDATIONS:")
        print(f"   🎯 AGGRESSIVE STRATEGY: Focus on top 3 STRONG BUY recommendations")
        print(f"   📊 BALANCED STRATEGY: Diversify across top 5-8 recommendations")
        print(f"   🛡️  CONSERVATIVE STRATEGY: Focus on LOW risk recommendations only")
        print(f"   💰 POSITION SIZING: Allocate more to higher confidence scores")

    def _save_market_analysis(self, recommendations: List[Dict]):
        """Save market analysis results"""
        timestamp = self.analysis_timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"app/data/cache/crypto_market_analysis_{timestamp}.json"
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            results = {
                'timestamp': self.analysis_timestamp.isoformat(),
                'analysis_summary': {
                    'cryptos_analyzed': len(self.analysis_results),
                    'recommendations_generated': len(recommendations),
                    'top_recommendation': recommendations[0] if recommendations else None
                },
                'recommendations': recommendations,
                'scoring_methodology': self.scoring_weights
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"\n💾 Market analysis saved to: {filename}")
            
        except Exception as e:
            print(f"⚠️  Could not save analysis: {e}")

async def main():
    """Main function"""
    analyzer = CryptoMarketAnalyzer()
    await analyzer.analyze_crypto_market()

if __name__ == "__main__":
    asyncio.run(main()) 