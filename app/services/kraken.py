import time, hashlib, hmac, base64, urllib.parse
import requests
from dotenv import load_dotenv
import os
import logging
from typing import Dict, Optional, List
import json
from datetime import datetime, timedelta

load_dotenv()

API_KEY = os.getenv("KRAKEN_API_KEY")
API_SECRET = os.getenv("KRAKEN_API_SECRET").encode() if os.getenv("KRAKEN_API_SECRET") else None

KRAKEN_API_URL = 'https://api.kraken.com'

# Setup logging
logger = logging.getLogger(__name__)

class KrakenAPI:
    """
    Kraken API client with rate limiting and caching
    Respects Kraken's rate limits: 1 call/second for public, counter-based for private
    """
    
    def __init__(self):
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        
        # Rate limiting - Conservative approach
        self.last_public_call = 0
        self.min_public_interval = 1.1  # 1.1 seconds between public calls (conservative)
        self.private_call_counter = 0
        self.last_counter_reset = time.time()
        
        # Caching for market data
        self.price_cache = {}
        self.cache_expiry = 30  # 30 seconds cache for prices
        
        # Kraken symbol mapping (our format -> Kraken REST API format) - CORRECTED
        self.symbol_map = {
            # Crypto pairs (our format -> Kraken REST API format)
            'BTC-USD': 'XXBTZUSD',  # Bitcoin (actual Kraken pair name)
            'ETH-USD': 'XETHZUSD',  # Ethereum (actual Kraken pair name)
            'XRP-USD': 'XXRPZUSD',  # Ripple (actual Kraken pair name)
            'SOL-USD': 'SOLUSD',    # Solana (actual Kraken pair name)
            'ADA-USD': 'ADAUSD',    # Cardano (actual Kraken pair name)
            'TRX-USD': 'TRXUSD',    # Tron (actual Kraken pair name)
            'XLM-USD': 'XXLMZUSD',  # Stellar (actual Kraken pair name)
            
            # Reverse mapping (Kraken format -> our format)
            'XXBTZUSD': 'BTC-USD',
            'XETHZUSD': 'ETH-USD',
            'XXRPZUSD': 'XRP-USD', 
            'SOLUSD': 'SOL-USD',
            'ADAUSD': 'ADA-USD',
            'TRXUSD': 'TRX-USD',
            'XXLMZUSD': 'XLM-USD'
        }
        
        # Supported pairs on Kraken (actual pair names)
        self.supported_pairs = [
            'XXBTZUSD',  # Bitcoin
            'XETHZUSD',  # Ethereum
            'XXRPZUSD',  # Ripple
            'SOLUSD',    # Solana
            'ADAUSD',    # Cardano
            'TRXUSD',    # Tron
            'XXLMZUSD'   # Stellar
        ]
        
        logger.info(f"Kraken API initialized - {'Authenticated' if self.api_key and self.api_secret else 'Public only'}")
        logger.info(f"Supported pairs: {self.supported_pairs}")

    def _wait_for_rate_limit(self, is_private: bool = False):
        """Ensure we don't exceed rate limits"""
        if is_private:
            # Private API: Counter-based system
            # Reset counter every 15 seconds (conservative)
            current_time = time.time()
            if current_time - self.last_counter_reset > 15:
                self.private_call_counter = 0
                self.last_counter_reset = current_time
            
            # Wait if counter is too high (conservative limit of 10 calls per 15 seconds)
            if self.private_call_counter >= 10:
                wait_time = 15 - (current_time - self.last_counter_reset)
                if wait_time > 0:
                    logger.info(f"Rate limit protection: waiting {wait_time:.1f}s for private API")
                    time.sleep(wait_time)
                    self.private_call_counter = 0
                    self.last_counter_reset = time.time()
        else:
            # Public API: 1 call per second max
            current_time = time.time()
            time_since_last = current_time - self.last_public_call
            
            if time_since_last < self.min_public_interval:
                wait_time = self.min_public_interval - time_since_last
                logger.debug(f"Rate limit protection: waiting {wait_time:.1f}s for public API")
                time.sleep(wait_time)
            
            self.last_public_call = time.time()

    def kraken_request(self, uri_path, data=None, headers=None, is_private=False):
        """Make authenticated request to Kraken API with rate limiting"""
        if is_private and (not self.api_key or not self.api_secret):
            raise ValueError("Private API calls require API key and secret")
        
        # Apply rate limiting
        self._wait_for_rate_limit(is_private)
        
        url = KRAKEN_API_URL + uri_path
        
        if is_private:
            # Private API call
            nonce = str(int(time.time() * 1000))
            data = data or {}
            data['nonce'] = nonce
            
            # Generate signature
            postdata = urllib.parse.urlencode(data)
            encoded = (str(nonce) + postdata).encode()
            message = uri_path.encode() + hashlib.sha256(encoded).digest()
            
            mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
            sigdigest = base64.b64encode(mac.digest())
            
            headers = {
                'API-Key': self.api_key,
                'API-Sign': sigdigest.decode()
            }
            
            response = requests.post(url, headers=headers, data=data)
        else:
            # Public API call
            params = data if data else {}
            response = requests.get(url, params=params)
        
        try:
            result = response.json()
            if result.get('error'):
                logger.error(f"Kraken API error: {result['error']}")
            return result
        except Exception as e:
            logger.error(f"Failed to parse Kraken response: {e}")
            return {'error': [str(e)]}

    def get_ticker(self, pair: str) -> Dict:
        """Get ticker data for a pair with caching"""
        # Convert our symbol format to Kraken format
        kraken_pair = self.symbol_map.get(pair, pair)
        
        # Check cache first
        cache_key = f"ticker_{kraken_pair}"
        current_time = time.time()
        
        if cache_key in self.price_cache:
            cached_data, timestamp = self.price_cache[cache_key]
            if current_time - timestamp < self.cache_expiry:
                return cached_data
        
        # Fetch from API
        result = self.kraken_request(f'/0/public/Ticker?pair={kraken_pair}')
        
        # Cache the result
        if not result.get('error'):
            self.price_cache[cache_key] = (result, current_time)
        
        return result

    def get_price(self, symbol: str) -> float:
        """Get current price for a symbol (compatible with trading system)"""
        try:
            # Only try Kraken for supported crypto pairs
            if symbol not in self.symbol_map:
                return 0.0
            
            ticker_data = self.get_ticker(symbol)
            
            if ticker_data.get('error'):
                logger.debug(f"Kraken price fetch failed for {symbol}: {ticker_data['error']}")
                return 0.0
            
            # Extract price from Kraken response
            kraken_pair = self.symbol_map[symbol]
            if 'result' in ticker_data and kraken_pair in ticker_data['result']:
                # Get the last trade price (most current)
                price_data = ticker_data['result'][kraken_pair]
                last_price = float(price_data['c'][0])  # 'c' is last trade [price, volume]
                
                logger.debug(f"Kraken price for {symbol}: ${last_price:.4f}")
                return last_price
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    def get_balance(self) -> Dict:
        """Get account balance (private API)"""
        if not self.api_key or not self.api_secret:
            return {'error': ['API credentials not configured']}
        
        return self.kraken_request('/0/private/Balance', is_private=True)

    def get_trade_history(self) -> Dict:
        """Get trade history (private API)"""
        if not self.api_key or not self.api_secret:
            return {'error': ['API credentials not configured']}
        
        return self.kraken_request('/0/private/TradesHistory', is_private=True)

    def get_supported_symbols(self) -> List[str]:
        """Get list of symbols supported by this Kraken integration"""
        # Return our standard symbol format
        return [symbol for symbol in self.symbol_map.keys() if '-USD' in symbol]

    def is_supported(self, symbol: str) -> bool:
        """Check if a symbol is supported by Kraken integration"""
        return symbol in self.symbol_map

    def test_connection(self) -> Dict:
        """Test Kraken API connection"""
        try:
            # Test public API
            result = self.kraken_request('/0/public/Time')
            
            if result.get('error'):
                return {
                    'public_api': False,
                    'private_api': False,
                    'error': result['error']
                }
            
            public_ok = True
            server_time = result.get('result', {}).get('unixtime', 0)
            
            # Test private API if credentials available
            private_ok = False
            if self.api_key and self.api_secret:
                try:
                    balance_result = self.get_balance()
                    private_ok = not balance_result.get('error')
                except Exception:
                    private_ok = False
            
            return {
                'public_api': public_ok,
                'private_api': private_ok,
                'server_time': server_time,
                'local_time': int(time.time()),
                'supported_pairs': len(self.supported_pairs)
            }
            
        except Exception as e:
            return {
                'public_api': False,
                'private_api': False,
                'error': str(e)
            }

# Global instance
kraken_api = KrakenAPI()

# Legacy functions for backward compatibility
def kraken_request(uri_path, data=None, headers=None):
    """Legacy function - use KrakenAPI class instead"""
    return kraken_api.kraken_request(uri_path, data, headers, is_private=True)

def get_balance():
    """Legacy function - use KrakenAPI class instead"""
    return kraken_api.get_balance()

def get_trade_history():
    """Legacy function - use KrakenAPI class instead"""
    return kraken_api.get_trade_history()

def get_ticker(pair):
    """Legacy function - use KrakenAPI class instead"""
    return kraken_api.get_ticker(pair)
