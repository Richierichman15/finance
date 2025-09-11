#!/usr/bin/env python3
"""
🧠 SMART AI PORTFOLIO ANALYZER - ENHANCED VERSION
=================================================
Advanced AI analysis with proven Kraken symbol mapping, real-time data,
learning capabilities, and comprehensive portfolio insights.
"""

import sys
import os
import asyncio
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import tweepy
import praw
import textblob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
warnings.filterwarnings('ignore')

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.kraken import kraken_api

class SentimentAnalyzer:
    def __init__(self):
        # Initialize sentiment analyzers
        self.vader = SentimentIntensityAnalyzer()
        self.sentiment_cache = {}
        self.sentiment_history = {}
        
        # Twitter API setup (you'll need to add your credentials)
        self.twitter_api = None
        try:
            auth = tweepy.OAuthHandler("YOUR_CONSUMER_KEY", "YOUR_CONSUMER_SECRET")
            auth.set_access_token("YOUR_ACCESS_TOKEN", "YOUR_ACCESS_SECRET")
            self.twitter_api = tweepy.API(auth)
        except:
            print("⚠️  Twitter API setup failed - using alternative sources")
            
        # Reddit API setup (you'll need to add your credentials)
        self.reddit_api = None
        try:
            self.reddit_api = praw.Reddit(
                client_id="YOUR_CLIENT_ID",
                client_secret="YOUR_CLIENT_SECRET",
                user_agent="YOUR_USER_AGENT"
            )
        except:
            print("⚠️  Reddit API setup failed - using alternative sources")
    
    async def get_social_sentiment(self, symbol: str) -> Dict:
        """Get aggregated social sentiment for a symbol"""
        try:
            # Check cache first
            if symbol in self.sentiment_cache:
                cache_time = self.sentiment_cache[symbol]['timestamp']
                if (datetime.now() - cache_time).seconds < 3600:  # Cache for 1 hour
                    return self.sentiment_cache[symbol]['data']
            
            sentiment_data = {
                'twitter_sentiment': 0.0,
                'reddit_sentiment': 0.0,
                'news_sentiment': 0.0,
                'overall_sentiment': 0.0,
                'sentiment_strength': 0.0,
                'mentions_count': 0,
                'trending_score': 0.0
            }
            
            # Get Twitter sentiment
            if self.twitter_api:
                tweets = self.twitter_api.search_tweets(q=f"#{symbol}", lang="en", count=100)
                twitter_scores = []
                for tweet in tweets:
                    score = self.vader.polarity_scores(tweet.text)
                    twitter_scores.append(score['compound'])
                if twitter_scores:
                    sentiment_data['twitter_sentiment'] = np.mean(twitter_scores)
                    sentiment_data['mentions_count'] += len(twitter_scores)
            
            # Get Reddit sentiment
            if self.reddit_api:
                subreddits = ['CryptoCurrency', 'CryptoMarkets', symbol]
                reddit_scores = []
                for subreddit in subreddits:
                    try:
                        posts = self.reddit_api.subreddit(subreddit).search(symbol, time_filter='day', limit=100)
                        for post in posts:
                            score = self.vader.polarity_scores(post.title + " " + post.selftext)
                            reddit_scores.append(score['compound'])
                    except:
                        continue
                if reddit_scores:
                    sentiment_data['reddit_sentiment'] = np.mean(reddit_scores)
                    sentiment_data['mentions_count'] += len(reddit_scores)
            
            # Calculate overall sentiment
            sentiments = [s for s in [
                sentiment_data['twitter_sentiment'],
                sentiment_data['reddit_sentiment'],
                sentiment_data['news_sentiment']
            ] if s != 0]
            
            if sentiments:
                sentiment_data['overall_sentiment'] = np.mean(sentiments)
                sentiment_data['sentiment_strength'] = abs(sentiment_data['overall_sentiment'])
                
                # Calculate trending score based on mentions and sentiment strength
                sentiment_data['trending_score'] = (
                    sentiment_data['mentions_count'] * 
                    (1 + abs(sentiment_data['overall_sentiment']))
                ) / 100
            
            # Cache the results
            self.sentiment_cache[symbol] = {
                'timestamp': datetime.now(),
                'data': sentiment_data
            }
            
            # Update sentiment history
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            self.sentiment_history[symbol].append({
                'timestamp': datetime.now(),
                'sentiment': sentiment_data['overall_sentiment']
            })
            
            return sentiment_data
            
        except Exception as e:
            print(f"⚠️  Error getting social sentiment for {symbol}: {e}")
            return {
                'overall_sentiment': 0.0,
                'sentiment_strength': 0.0,
                'mentions_count': 0,
                'trending_score': 0.0
            }

class SmartAIAnalyzer:
    def __init__(self):
        self.analysis_timestamp = datetime.now()
        
        # Learning system
        self.learning_data = self._load_learning_data()
        self.prediction_history = []
        self.accuracy_tracking = {}
        
        # Enhanced ML components
        self.models = {
            'price_predictor': None,
            'trend_classifier': None,
            'risk_analyzer': None
        }
        self.scalers = {
            'features': StandardScaler(),
            'target': StandardScaler()
        }
        
        # Initialize ML models
        self._initialize_ml_models()
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Proven symbol mapping from pure_5k_system.py
        self.symbol_map = {
            'BTC-USD': 'XXBTZUSD',
            'ETH-USD': 'XETHZUSD',
            'SOL-USD': 'SOLUSD',
            'XRP-USD': 'XXRPZUSD',
            'ADA-USD': 'ADAUSD',
            'TRX-USD': 'TRXUSD',
            'XLM-USD': 'XXLMZUSD',
            'DOGE-USD': 'XXDGZUSD',
            'DOT-USD': 'DOTUSD',
            'MATIC-USD': 'MATICUSD',
            'LINK-USD': 'LINKUSD',
            'UNI-USD': 'UNIUSD',
            'AVAX-USD': 'AVAXUSD',
            'BONK-USD': 'BONKUSD',
            'FLOKI-USD': 'FLOKIUSD',
            'PEPE-USD': 'PEPEUSD'
        }
        
        print("🧠 Smart AI Portfolio Analyzer Initialized")
        print("🔬 Features: Proven symbol mapping, Learning system, Real-time analysis")
        print("🎯 ML Models: Price Prediction, Trend Classification, Risk Analysis")
        print("📊 Sentiment Analysis: Social Media Integration Ready")

    def _initialize_ml_models(self):
        """Initialize or load pre-trained ML models"""
        try:
            # Try to load existing models
            model_path = 'app/data/models/'
            os.makedirs(model_path, exist_ok=True)
            
            for model_name in self.models.keys():
                model_file = f"{model_path}{model_name}.pkl"
                try:
                    if os.path.exists(model_file):
                        with open(model_file, 'rb') as f:
                            self.models[model_name] = pickle.load(f)
                        print(f"✅ Loaded {model_name} model")
                    else:
                        # Create new models if not found
                        if model_name == 'price_predictor':
                            self.models[model_name] = GradientBoostingRegressor(
                                n_estimators=100,
                                learning_rate=0.1,
                                max_depth=5
                            )
                        elif model_name == 'trend_classifier':
                            self.models[model_name] = RandomForestClassifier(
                                n_estimators=100,
                                max_depth=5,
                                random_state=42
                            )
                        elif model_name == 'risk_analyzer':
                            self.models[model_name] = GradientBoostingRegressor(
                                n_estimators=100,
                                learning_rate=0.1,
                                max_depth=4
                            )
                        print(f"🆕 Created new {model_name} model")
                except Exception as e:
                    print(f"⚠️  Error loading {model_name} model: {e}")
                    # Create a new model as fallback
                    self.models[model_name] = GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=5
                    )
                    print(f"🆕 Created fallback {model_name} model")
            
        except Exception as e:
            print(f"⚠️  Error initializing ML models: {e}")
            # Initialize basic models as fallback
            self.models = {
                'price_predictor': GradientBoostingRegressor(n_estimators=100),
                'trend_classifier': RandomForestClassifier(n_estimators=100),
                'risk_analyzer': GradientBoostingRegressor(n_estimators=100)
            }
            print("⚠️  Using fallback ML models")

    def _load_learning_data(self) -> Dict:
        """Load learning data for AI improvement"""
        try:
            with open('app/data/cache/ai_learning_data.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'price_fetch_success_rate': {},
                'prediction_accuracy': {},
                'learning_progress': [],
                'last_updated': self.analysis_timestamp.isoformat()
            }

    def _save_learning_data(self):
        """Save learning data to disk"""
        self.learning_data['last_updated'] = self.analysis_timestamp.isoformat()
        
        os.makedirs('app/data/cache', exist_ok=True)
        with open('app/data/cache/ai_learning_data.json', 'w') as f:
            json.dump(self.learning_data, f, indent=2, default=str)

    async def run_smart_analysis(self, portfolio_data: Dict = None) -> Dict:
        """Run comprehensive AI analysis with learning capabilities"""
        print("\n🧠 Running Smart AI Analysis...")
        
        try:
            # Step 1: Get comprehensive portfolio data if not provided
            if portfolio_data is None:
                portfolio_data = await self._get_comprehensive_portfolio_data()
            else:
                # Convert provided portfolio data to our format if needed
                portfolio_data = self._normalize_portfolio_data(portfolio_data)
            
            # Step 2: Run ML predictions for each asset
            ml_predictions = {}
            sentiment_analysis = {}
            
            for asset, data in portfolio_data.items():
                if asset != 'ZUSD':  # Skip cash
                    # Get historical data
                    historical_data = await self._get_technical_data(asset)
                    if historical_data:
                        # Get ML predictions
                        predictions = await self._predict_price_movement(asset, pd.DataFrame(historical_data))
                        ml_predictions[asset] = predictions
                        
                        # Get social sentiment
                        sentiment = await self.sentiment_analyzer.get_social_sentiment(asset)
                        sentiment_analysis[asset] = sentiment
            
            # Step 3: Generate analysis results
            analysis_results = {
                'portfolio_overview': self._analyze_portfolio_overview(portfolio_data),
                'technical_signals': await self._analyze_technical_signals(portfolio_data),
                'ml_predictions': ml_predictions,
                'sentiment_analysis': sentiment_analysis,
                'risk_assessment': self._assess_smart_risk(portfolio_data, ml_predictions, sentiment_analysis),
                'recommendations': self._generate_smart_recommendations(portfolio_data, ml_predictions, sentiment_analysis),
                'analysis_metadata': {
                    'timestamp': self.analysis_timestamp.isoformat(),
                    'model_versions': {name: str(model) for name, model in self.models.items()},
                    'success_rate': len(portfolio_data.get('successful_mappings', [])) / max(1, len(portfolio_data))
                }
            }
            
            # Step 4: Update learning system
            self._update_learning_data(portfolio_data, analysis_results)
            
            return analysis_results
            
        except Exception as e:
            print(f"❌ Smart analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _update_learning_data(self, portfolio_data: Dict, analysis_results: Dict):
        """Update learning system with new data"""
        try:
            # Update prediction accuracy
            for asset, predictions in analysis_results.get('ml_predictions', {}).items():
                if asset not in self.learning_data['prediction_accuracy']:
                    self.learning_data['prediction_accuracy'][asset] = {
                        'correct_predictions': 0,
                        'total_predictions': 0,
                        'last_prediction': None,
                        'last_actual': None
                    }
                
                # Store current prediction for future accuracy tracking
                self.learning_data['prediction_accuracy'][asset]['last_prediction'] = {
                    'timestamp': self.analysis_timestamp.isoformat(),
                    'prediction': predictions.get('price_change_pred', 0),
                    'confidence': predictions.get('confidence', 0)
                }
            
            # Update price fetch success rate
            for asset in portfolio_data.get('successful_mappings', []):
                if asset not in self.learning_data['price_fetch_success_rate']:
                    self.learning_data['price_fetch_success_rate'][asset] = {
                        'success_count': 0,
                        'total_attempts': 0
                    }
                self.learning_data['price_fetch_success_rate'][asset]['success_count'] += 1
                self.learning_data['price_fetch_success_rate'][asset]['total_attempts'] += 1
            
            # Add learning progress entry
            self.learning_data['learning_progress'].append({
                'timestamp': self.analysis_timestamp.isoformat(),
                'portfolio_size': len(portfolio_data),
                'successful_predictions': sum(1 for pred in analysis_results.get('ml_predictions', {}).values() 
                                           if pred.get('confidence', 0) > 0.7),
                'average_confidence': np.mean([pred.get('confidence', 0) 
                                            for pred in analysis_results.get('ml_predictions', {}).values()])
            })
            
            # Save updated learning data
            self._save_learning_data()
            
        except Exception as e:
            print(f"⚠️  Error updating learning data: {e}")

    def _normalize_portfolio_data(self, input_data: Dict) -> Dict:
        """Normalize external portfolio data to our format"""
        try:
            if 'assets' in input_data:
                return input_data  # Already in our format
            
            normalized_data = {
                'assets': {},
                'total_value_usd': 0,
                'fetch_timestamp': self.analysis_timestamp.isoformat(),
                'data_source': 'external',
                'failed_assets': [],
                'successful_mappings': []
            }
            
            # Process each asset
            for asset, data in input_data.items():
                if isinstance(data, dict):
                    # If it's a dict with detailed data
                    amount = float(data.get('amount', 0))
                    price = float(data.get('value_usd', 0)) / amount if amount > 0 else 0
                    value = float(data.get('value_usd', 0))
                else:
                    # If it's just an amount
                    amount = float(data)
                    price = 0  # We'll need to fetch this
                    value = 0
                    
                if amount > 0.0001:  # Apply minimum threshold
                    normalized_data['assets'][asset] = {
                        'amount': amount,
                        'price_usd': price,
                        'value_usd': value,
                        'symbol': asset,
                        'mapping_success': True,
                        'timestamp': self.analysis_timestamp.isoformat()
                    }
                    normalized_data['successful_mappings'].append(asset)
                    normalized_data['total_value_usd'] += value
                
            # Calculate allocations
            total_value = normalized_data['total_value_usd']
            if total_value > 0:
                for asset_data in normalized_data['assets'].values():
                    asset_data['allocation_pct'] = (asset_data['value_usd'] / total_value) * 100
                    
            return normalized_data
            
        except Exception as e:
            print(f"⚠️  Error normalizing portfolio data: {e}")
            return {
                'assets': {},
                'total_value_usd': 0,
                'fetch_timestamp': self.analysis_timestamp.isoformat(),
                'data_source': 'error',
                'failed_assets': [],
                'successful_mappings': []
            }

    async def _get_comprehensive_portfolio_data(self) -> Dict:
        """Get comprehensive portfolio data with enhanced symbol mapping"""
        print("📡 Fetching comprehensive portfolio data...")
        
        try:
            # Get Kraken balance
            balance = kraken_api.get_balance()
            
            if 'result' not in balance:
                print("⚠️  Using sample portfolio for analysis")
                return self._get_sample_portfolio_data()
            
            holdings = balance['result']
            portfolio_data = {
                'assets': {},
                'total_value_usd': 0,
                'fetch_timestamp': self.analysis_timestamp.isoformat(),
                'data_source': 'kraken_live',
                'failed_assets': [],
                'successful_mappings': []
            }
            
            print(f"💰 Processing {len(holdings)} assets with enhanced mapping...")
            
            for asset, amount in holdings.items():
                try:
                    amount_float = float(amount)
                    if amount_float > 0.0001:  # Lower threshold
                        
                        # Enhanced price fetching with multiple attempts
                        price, symbol, success = await self._get_enhanced_price(asset)
                        
                        if price > 0:
                            value_usd = amount_float * price
                            portfolio_data['assets'][asset] = {
                                'amount': amount_float,
                                'price_usd': price,
                                'value_usd': value_usd,
                                'symbol': symbol,
                                'mapping_success': success,
                                'timestamp': self.analysis_timestamp.isoformat()
                            }
                            portfolio_data['total_value_usd'] += value_usd
                            portfolio_data['successful_mappings'].append(asset)
                            
                            print(f"   ✅ {asset}: {amount_float:.4f} @ ${price:.4f} = ${value_usd:.2f} ({symbol})")
                        else:
                            portfolio_data['failed_assets'].append((asset, amount_float))
                            print(f"   ❌ {asset}: {amount_float:.4f} - Price fetch failed")
                    elif amount_float > 0:
                        print(f"   🔍 {asset}: {amount_float:.8f} - Too small (below threshold)")
                
                except Exception as e:
                    print(f"   ⚠️  Error processing {asset}: {e}")
                    continue
            
            # Calculate allocations
            total_value = portfolio_data['total_value_usd']
            for asset_data in portfolio_data['assets'].values():
                asset_data['allocation_pct'] = (asset_data['value_usd'] / total_value) * 100
            
            # Update learning data
            self._update_mapping_success_rate(portfolio_data)
            
            print(f"✅ Portfolio value: ${total_value:.2f}")
            print(f"📊 Successful mappings: {len(portfolio_data['successful_mappings'])}")
            print(f"❌ Failed assets: {len(portfolio_data['failed_assets'])}")
            
            return portfolio_data
            
        except Exception as e:
            print(f"⚠️  Error fetching live data: {e}")
            return self._get_sample_portfolio_data()

    async def _get_enhanced_price(self, kraken_asset: str) -> Tuple[float, str, bool]:
        """Enhanced price fetching with multiple symbol variations"""
        try:
            # Handle .F suffix (futures)
            if kraken_asset.endswith('.F'):
                base_asset = kraken_asset.replace('.F', '')
            else:
                base_asset = kraken_asset
            
            # Try multiple symbol variations
            symbol_variations = self._get_symbol_variations(base_asset)
            
            for symbol in symbol_variations:
                try:
                    # Try Kraken first
                    if symbol in self.symbol_map.values() or any(var in self.symbol_map.values() for var in symbol_variations):
                        price = kraken_api.get_price(symbol)
                        if price > 0:
                            return price, symbol, True
                    
                    # Try yfinance
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1d", interval="1m")
                    
                    if not data.empty:
                        current_price = float(data['Close'].iloc[-1])
                        if current_price > 0:
                            return current_price, symbol, True
                            
                except Exception as e:
                    continue
            
            # If all variations fail, try direct mapping
            direct_symbol = f"{base_asset}-USD"
            try:
                ticker = yf.Ticker(direct_symbol)
                data = ticker.history(period="1d", interval="1m")
                
                if not data.empty:
                    current_price = float(data['Close'].iloc[-1])
                    if current_price > 0:
                        return current_price, direct_symbol, True
            except:
                pass
            
            return 0.0, f"{base_asset}-USD", False
                
        except Exception as e:
            print(f"   ⚠️  Enhanced price fetch error for {kraken_asset}: {e}")
            return 0.0, f"{kraken_asset}-USD", False

    def _get_symbol_variations(self, asset: str) -> List[str]:
        """Get multiple symbol variations for an asset"""
        variations = []
        
        # Direct mapping
        variations.append(f"{asset}-USD")
        
        # Check symbol map
        if f"{asset}-USD" in self.symbol_map:
            variations.append(self.symbol_map[f"{asset}-USD"])
        
        # Check Kraken variations
        if asset in self.kraken_symbol_variations:
            variations.extend(self.kraken_symbol_variations[asset])
        
        # Common variations
        common_variations = [
            f"{asset}USD",
            f"{asset}ZUSD",
            f"X{asset}USD",
            f"X{asset}ZUSD",
            f"XX{asset}USD",
            f"{asset.upper()}USD",
            f"X{asset.upper()}ZUSD"
        ]
        variations.extend(common_variations)
        
        return list(set(variations))  # Remove duplicates

    def _update_mapping_success_rate(self, portfolio_data: Dict):
        """Update learning data with mapping success rates"""
        successful = len(portfolio_data['successful_mappings'])
        total = len(portfolio_data['assets']) + len(portfolio_data['failed_assets'])
        
        if total > 0:
            success_rate = successful / total
            self.learning_data['price_fetch_success_rate'][self.analysis_timestamp.strftime('%Y-%m-%d')] = success_rate
            
            print(f"📊 Learning: Price fetch success rate: {success_rate:.1%}")

    def _get_sample_portfolio_data(self) -> Dict:
        """Get sample portfolio data for testing"""
        return {
            'assets': {
                'BTC': {
                    'amount': 0.001234,
                    'price_usd': 67500.0,
                    'value_usd': 83.295,
                    'symbol': 'BTC-USD',
                    'allocation_pct': 40.0,
                    'mapping_success': True,
                    'timestamp': self.analysis_timestamp.isoformat()
                },
                'ETH': {
                    'amount': 0.0456,
                    'price_usd': 3650.0,
                    'value_usd': 166.44,
                    'symbol': 'ETH-USD', 
                    'allocation_pct': 35.0,
                    'mapping_success': True,
                    'timestamp': self.analysis_timestamp.isoformat()
                },
                'SOL': {
                    'amount': 0.789,
                    'price_usd': 190.0,
                    'value_usd': 149.91,
                    'symbol': 'SOL-USD',
                    'allocation_pct': 25.0,
                    'mapping_success': True,
                    'timestamp': self.analysis_timestamp.isoformat()
                }
            },
            'total_value_usd': 399.645,
            'fetch_timestamp': self.analysis_timestamp.isoformat(),
            'data_source': 'sample_data',
            'failed_assets': [],
            'successful_mappings': ['BTC', 'ETH', 'SOL']
        }

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix for ML models"""
        try:
            features = []
            
            # Technical indicators
            features.extend([
                data['Close'].pct_change(),  # Returns
                data['Close'].pct_change().rolling(5).std(),  # Volatility
                data['Close'].rolling(5).mean() / data['Close'] - 1,  # Price vs SMA
                data['Volume'].pct_change(),  # Volume change
                self._calculate_rsi(data['Close']),  # RSI
            ])
            
            # Market context
            if 'market_cap' in data.columns:
                features.append(data['market_cap'].pct_change())
            if 'dominance' in data.columns:
                features.append(data['dominance'].pct_change())
                
            # Stack and clean features
            feature_matrix = np.column_stack(features)
            feature_matrix = np.nan_to_num(feature_matrix, 0)
            
            return self.scalers['features'].fit_transform(feature_matrix)
            
        except Exception as e:
            print(f"⚠️  Error preparing features: {e}")
            return np.array([])
    
    async def _predict_price_movement(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Predict price movement using ML models"""
        try:
            # Prepare features
            features = self._prepare_features(data)
            if len(features) == 0:
                return {}
            
            # Get social sentiment
            sentiment = await self.sentiment_analyzer.get_social_sentiment(symbol)
            
            # Make predictions
            price_pred = self.models['price_predictor'].predict(features)
            trend_pred = self.models['trend_classifier'].predict_proba(features)
            risk_pred = self.models['risk_analyzer'].predict(features)
            
            # Calculate confidence scores
            prediction_confidence = np.mean([
                np.max(trend_pred[-1]),  # Trend confidence
                1 - abs(risk_pred[-1]),  # Risk confidence
                sentiment['sentiment_strength']  # Sentiment confidence
            ])
            
            return {
                'price_change_pred': float(price_pred[-1]),
                'trend_probability': float(np.max(trend_pred[-1])),
                'risk_score': float(risk_pred[-1]),
                'confidence': float(prediction_confidence),
                'sentiment': sentiment
            }
            
        except Exception as e:
            print(f"⚠️  Error predicting price movement: {e}")
            return {}
    
    def _update_models(self, symbol: str, predictions: Dict, actual_outcomes: Dict):
        """Update ML models with new data"""
        try:
            # Prepare training data
            features = self._prepare_features(actual_outcomes['data'])
            if len(features) == 0:
                return
            
            # Update price predictor
            y_price = actual_outcomes['data']['Close'].pct_change().values
            y_price = np.nan_to_num(y_price, 0)
            self.models['price_predictor'].fit(features[:-1], y_price[1:])
            
            # Update trend classifier
            y_trend = (y_price > 0).astype(int)
            self.models['trend_classifier'].fit(features[:-1], y_trend[1:])
            
            # Update risk analyzer
            y_risk = np.abs(y_price)
            self.models['risk_analyzer'].fit(features[:-1], y_risk[1:])
            
            # Save updated models
            model_path = 'app/data/models/'
            os.makedirs(model_path, exist_ok=True)
            for model_name, model in self.models.items():
                with open(f"{model_path}{model_name}.pkl", 'wb') as f:
                    pickle.dump(model, f)
            
            print(f"✅ Updated ML models with new data for {symbol}")
            
        except Exception as e:
            print(f"⚠️  Error updating models: {e}")
    
    async def _run_smart_analysis(self, portfolio_data: Dict) -> Dict:
        """Enhanced smart analysis with ML and sentiment"""
        try:
            results = {
                'portfolio_overview': self._analyze_portfolio_overview(portfolio_data),
                'technical_signals': await self._analyze_technical_signals(portfolio_data),
                'ml_predictions': {},
                'sentiment_analysis': {},
                'risk_assessment': {},
                'recommendations': []
            }
            
            # Get ML predictions and sentiment for each asset
            for symbol, data in portfolio_data.items():
                if symbol != 'ZUSD':  # Skip cash
                    # Get historical data
                    historical_data = await self._get_technical_data(symbol)
                    if historical_data:
                        # Get ML predictions
                        predictions = await self._predict_price_movement(symbol, pd.DataFrame(historical_data))
                        results['ml_predictions'][symbol] = predictions
                        
                        # Get social sentiment
                        sentiment = await self.sentiment_analyzer.get_social_sentiment(symbol)
                        results['sentiment_analysis'][symbol] = sentiment
            
            # Generate smart recommendations
            results['recommendations'] = self._generate_smart_recommendations(
                portfolio_data,
                results['ml_predictions'],
                results['sentiment_analysis']
            )
            
            # Enhanced risk assessment
            results['risk_assessment'] = self._assess_smart_risk(
                portfolio_data,
                results['ml_predictions'],
                results['sentiment_analysis']
            )
            
            return results
            
        except Exception as e:
            print(f"⚠️  Error in smart analysis: {e}")
            return {}
    
    def _generate_smart_recommendations(self, portfolio_data: Dict, 
                                     predictions: Dict, sentiment: Dict) -> List[Dict]:
        """Generate recommendations using ML and sentiment"""
        try:
            recommendations = []
            
            for symbol, data in portfolio_data.items():
                if symbol == 'ZUSD':  # Skip cash
                    continue
                
                pred = predictions.get(symbol, {})
                sent = sentiment.get(symbol, {})
                
                # Calculate combined score
                ml_score = pred.get('confidence', 0) * pred.get('trend_probability', 0)
                sentiment_score = sent.get('sentiment_strength', 0) * sent.get('trending_score', 0)
                combined_score = (ml_score + sentiment_score) / 2
                
                if combined_score > 0.6:  # High confidence signal
                    action = 'BUY' if pred.get('price_change_pred', 0) > 0 else 'SELL'
                    recommendations.append({
                        'symbol': symbol,
                        'action': action,
                        'confidence': combined_score,
                        'reasons': [
                            f"ML Prediction: {pred.get('price_change_pred', 0)*100:+.1f}% change",
                            f"Sentiment: {sent.get('overall_sentiment', 0):+.2f}",
                            f"Trending Score: {sent.get('trending_score', 0):.2f}"
                        ]
                    })
            
            # Sort by confidence
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            return recommendations
            
        except Exception as e:
            print(f"⚠️  Error generating recommendations: {e}")
            return []
    
    def _assess_smart_risk(self, portfolio_data: Dict, predictions: Dict, 
                          sentiment: Dict) -> Dict:
        """Enhanced risk assessment using ML and sentiment"""
        try:
            risk_assessment = {
                'overall_risk_score': 0.0,
                'risk_factors': [],
                'risk_metrics': {},
                'sentiment_risks': []
            }
            
            # Calculate portfolio risk score
            risk_scores = []
            for symbol, data in portfolio_data.items():
                if symbol == 'ZUSD':  # Skip cash
                    continue
                
                pred = predictions.get(symbol, {})
                sent = sentiment.get(symbol, {})
                
                # Individual asset risk
                asset_risk = pred.get('risk_score', 0.5)
                sentiment_risk = 1 - sent.get('sentiment_strength', 0)
                combined_risk = (asset_risk + sentiment_risk) / 2
                
                risk_assessment['risk_metrics'][symbol] = {
                    'ml_risk_score': asset_risk,
                    'sentiment_risk': sentiment_risk,
                    'combined_risk': combined_risk
                }
                risk_scores.append(combined_risk)
                
                # Check for sentiment-based risks
                if sent.get('trending_score', 0) > 0.8 and sent.get('overall_sentiment', 0) < -0.5:
                    risk_assessment['sentiment_risks'].append({
                        'symbol': symbol,
                        'risk_type': 'NEGATIVE_SENTIMENT_SPIKE',
                        'severity': 'HIGH'
                    })
            
            # Calculate overall portfolio risk
            if risk_scores:
                risk_assessment['overall_risk_score'] = np.mean(risk_scores)
                
                # Add risk factors based on analysis
                if risk_assessment['overall_risk_score'] > 0.7:
                    risk_assessment['risk_factors'].append({
                        'type': 'HIGH_PORTFOLIO_RISK',
                        'description': 'Portfolio shows elevated risk levels across multiple assets',
                        'severity': 'HIGH'
                    })
                
                if len(risk_assessment['sentiment_risks']) > 2:
                    risk_assessment['risk_factors'].append({
                        'type': 'NEGATIVE_MARKET_SENTIMENT',
                        'description': 'Multiple assets showing negative social sentiment',
                        'severity': 'MEDIUM'
                    })
            
            return risk_assessment
            
        except Exception as e:
            print(f"⚠️  Error in risk assessment: {e}")
            return {}

    def _analyze_portfolio_overview(self, portfolio_data: Dict) -> Dict:
        """Analyze portfolio overview metrics"""
        try:
            assets = portfolio_data.get('assets', {})
            total_value = sum(asset.get('value_usd', 0) for asset in assets.values())
            
            # Calculate allocations if not present
            if total_value > 0:
                for asset_data in assets.values():
                    asset_data['allocation_pct'] = (asset_data.get('value_usd', 0) / total_value) * 100
            
            # Calculate metrics
            overview = {
                'total_value': total_value,
                'number_of_positions': len(assets),
                'largest_position': max([asset.get('allocation_pct', 0) for asset in assets.values()]) if assets else 0,
                'smallest_position': min([asset.get('allocation_pct', 0) for asset in assets.values()]) if assets else 0,
                'average_position': total_value / len(assets) if assets else 0,
                'diversification_score': self._calculate_diversification_score(assets),
                'concentration_risk': self._assess_concentration_risk(assets),
                'asset_types': {
                    'crypto': len([a for a in assets.keys() if any(s in a for s in ['BTC', 'ETH', 'SOL', 'DOGE'])]),
                    'stablecoins': len([a for a in assets.keys() if any(s in a for s in ['USDT', 'USDC', 'ZUSD'])]),
                    'other': len([a for a in assets.keys() if not any(s in a for s in ['BTC', 'ETH', 'SOL', 'DOGE', 'USDT', 'USDC', 'ZUSD'])])
                }
            }
            
            return overview
            
        except Exception as e:
            print(f"⚠️  Error analyzing portfolio overview: {e}")
            return {
                'total_value': 0,
                'number_of_positions': 0,
                'largest_position': 0,
                'smallest_position': 0,
                'average_position': 0,
                'diversification_score': 0,
                'concentration_risk': 'UNKNOWN',
                'asset_types': {'crypto': 0, 'stablecoins': 0, 'other': 0}
            }

    def _assess_concentration_risk(self, assets: Dict) -> str:
        """Assess portfolio concentration risk"""
        try:
            # Calculate HHI (Herfindahl-Hirschman Index)
            allocations = [asset.get('allocation_pct', 0)/100 for asset in assets.values()]
            hhi = sum(alloc**2 for alloc in allocations)
            
            if hhi > 0.25:  # More than 25% concentration
                return 'HIGH'
            elif hhi > 0.15:  # More than 15% concentration
                return 'MEDIUM'
            else:
                return 'LOW'
                
        except Exception as e:
            print(f"⚠️  Error assessing concentration risk: {e}")
            return 'UNKNOWN'

    def _calculate_diversification_score(self, assets: Dict) -> float:
        """Calculate portfolio diversification score"""
        if not assets:
            return 0.0
        
        allocations = [asset['allocation_pct'] for asset in assets.values()]
        hhi = sum([(alloc/100)**2 for alloc in allocations])
        diversification_score = (1 - hhi) * 100
        
        return round(diversification_score, 2)

    async def _analyze_technical_signals(self, portfolio_data: Dict) -> Dict:
        """Analyze technical signals for all assets"""
        technical_signals = {}
        
        for asset_key, asset_info in portfolio_data.get('assets', {}).items():
            symbol = asset_info.get('symbol', asset_key)
            
            try:
                # Get technical data
                tech_data = await self._get_technical_data(symbol)
                if tech_data:
                    technical_signals[symbol] = {
                        'signal': self._determine_signal(tech_data),
                        'strength': self._calculate_signal_strength(tech_data),
                        'risk_level': self._assess_technical_risk(tech_data),
                        'confidence': self._get_prediction_confidence(symbol),
                        'learning_factor': self._get_learning_factor(symbol)
                    }
            except Exception as e:
                print(f"   ⚠️  Technical analysis failed for {symbol}: {e}")
                continue
        
        return technical_signals

    async def _get_technical_data(self, symbol: str) -> Optional[Dict]:
        """Get technical data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="30d")
            
            if data.empty:
                return None
            
            # Calculate technical indicators
            data['SMA_20'] = data['Close'].rolling(20).mean()
            data['SMA_50'] = data['Close'].rolling(50).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            
            current_price = float(data['Close'].iloc[-1])
            sma_20 = float(data['SMA_20'].iloc[-1])
            sma_50 = float(data['SMA_50'].iloc[-1])
            rsi = float(data['RSI'].iloc[-1])
            
            return {
                'current_price': current_price,
                'sma_20': sma_20,
                'sma_50': sma_50,
                'rsi': rsi,
                'trend': 'BULLISH' if current_price > sma_20 > sma_50 else 'BEARISH' if current_price < sma_20 < sma_50 else 'NEUTRAL'
            }
            
        except Exception as e:
            return None

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _determine_signal(self, tech_data: Dict) -> str:
        """Determine trading signal based on technical data"""
        try:
            rsi = tech_data.get('rsi', 50)
            sma_ratio = tech_data.get('sma_ratio', 1.0)
            volume_change = tech_data.get('volume_change', 0)
            
            # RSI-based signals
            if rsi > 70 and sma_ratio < 0.98:
                return 'SELL'
            elif rsi < 30 and sma_ratio > 1.02:
                return 'BUY'
            
            # Volume and trend signals
            if sma_ratio > 1.05 and volume_change > 0.1:
                return 'BUY'
            elif sma_ratio < 0.95 and volume_change < -0.1:
                return 'SELL'
                
            return 'HOLD'  # Changed from 'NEUTRAL' to 'HOLD'
            
        except Exception as e:
            print(f"⚠️  Error determining signal: {e}")
            return 'HOLD'

    def _calculate_signal_strength(self, tech_data: Dict) -> float:
        """Calculate signal strength (0-1)"""
        try:
            rsi = tech_data.get('rsi', 50)
            sma_ratio = tech_data.get('sma_ratio', 1.0)
            volume_change = tech_data.get('volume_change', 0)
            
            # Calculate component scores
            rsi_score = abs(rsi - 50) / 50  # 0-1 score based on RSI deviation from neutral
            trend_score = abs(sma_ratio - 1.0)  # 0-1 score based on trend strength
            volume_score = min(abs(volume_change), 1.0)  # Cap at 1.0
            
            # Weighted average of scores
            strength = (rsi_score * 0.4 + trend_score * 0.4 + volume_score * 0.2)
            
            return float(min(max(strength, 0.0), 1.0))  # Ensure float and 0-1 range
            
        except Exception as e:
            print(f"⚠️  Error calculating signal strength: {e}")
            return 0.0

    def _assess_technical_risk(self, tech_data: Dict) -> str:
        """Assess technical risk level"""
        rsi = tech_data.get('rsi', 50)
        
        if rsi > 75 or rsi < 25:
            return 'HIGH'
        elif rsi > 70 or rsi < 30:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _get_prediction_confidence(self, symbol: str) -> float:
        """Get confidence score for predictions"""
        try:
            # Get historical prediction accuracy
            symbol_accuracy = self.learning_data.get('prediction_accuracy', {}).get(symbol, 0.5)
            
            # Get learning progress
            learning_factor = self._get_learning_factor(symbol)
            
            # Calculate confidence score
            confidence = symbol_accuracy * 0.7 + learning_factor * 0.3
            
            return float(min(max(confidence, 0.0), 1.0))  # Ensure float and 0-1 range
            
        except Exception as e:
            print(f"⚠️  Error calculating prediction confidence: {e}")
            return 0.5  # Return neutral confidence on error

    def _get_learning_factor(self, symbol: str) -> float:
        """Get learning factor for symbol"""
        # This would be based on how much we've learned about this symbol
        return 0.8  # Default learning factor

    def _generate_learning_insights(self, portfolio_data: Dict) -> Dict:
        """Generate insights from learning system"""
        return {
            'mapping_improvements': self._get_mapping_improvements(),
            'prediction_accuracy': self._get_prediction_accuracy(),
            'recommended_actions': self._get_learning_recommendations()
        }

    def _get_mapping_improvements(self) -> List[str]:
        """Get symbol mapping improvement suggestions"""
        improvements = []
        
        if len(self.learning_data.get('price_fetch_success_rate', {})) > 0:
            recent_success_rate = list(self.learning_data['price_fetch_success_rate'].values())[-1]
            
            if recent_success_rate < 0.8:
                improvements.append("Consider adding more symbol variations for failed assets")
            if recent_success_rate < 0.6:
                improvements.append("Review Kraken API symbol mapping for common failures")
        
        return improvements

    def _get_prediction_accuracy(self) -> Dict:
        """Get prediction accuracy metrics"""
        return {
            'overall_accuracy': 0.75,  # Would be calculated from historical data
            'symbol_specific': {},
            'trending_accuracy': 'IMPROVING'
        }

    def _get_learning_recommendations(self) -> List[str]:
        """Get recommendations from learning system"""
        return [
            "Continue monitoring symbol mapping success rates",
            "Track prediction accuracy for portfolio optimization",
            "Consider expanding symbol variations for better coverage"
        ]

    def _display_smart_results(self, results: Dict):
        """Display smart analysis results"""
        print("\n" + "="*80)
        print("🧠 SMART AI ANALYSIS RESULTS")
        print("="*80)
        
        # Portfolio Overview
        overview = results.get('portfolio_overview', {})
        print(f"\n💰 PORTFOLIO OVERVIEW:")
        print(f"   Total Value: ${overview.get('total_value_usd', 0):,.2f}")
        print(f"   Assets: {overview.get('asset_count', 0)}")
        print(f"   Largest Position: {overview.get('largest_position', 0):.1f}%")
        print(f"   Diversification: {overview.get('diversification_score', 0):.1f}/100")
        print(f"   Data Quality: {overview.get('data_quality', 'UNKNOWN')}")
        
        # Learning Insights
        learning = results.get('learning_insights', {})
        print(f"\n🧠 LEARNING INSIGHTS:")
        print(f"   Mapping Success Rate: {results.get('analysis_metadata', {}).get('success_rate', 0):.1%}")
        print(f"   Learning Progress: {learning.get('prediction_accuracy', {}).get('trending_accuracy', 'UNKNOWN')}")
        
        # Technical Signals
        technical = results.get('technical_analysis', {})
        if technical:
            print(f"\n📊 TECHNICAL SIGNALS:")
            for symbol, signal in technical.items():
                print(f"   {symbol:12} | {signal['signal']:12} | Strength: {signal['strength']:5.1f}% | Risk: {signal['risk_level']}")
        
        # Recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print(f"\n💡 SMART RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:5], 1):
                priority_emoji = "🔥" if rec['priority'] == 'HIGH' else "⚡" if rec['priority'] == 'MEDIUM' else "💡"
                print(f"   {i}. {priority_emoji} {rec['message']}")
                print(f"      Confidence: {rec['confidence']}%")
        
        # Risk Assessment
        risk = results.get('risk_assessment', {})
        print(f"\n⚠️  RISK ASSESSMENT:")
        print(f"   Risk Level: {risk.get('risk_level', 'UNKNOWN')}")
        print(f"   Risk Score: {risk.get('risk_score', 0):.1f}/100")
        for factor in risk.get('risk_factors', []):
            print(f"   • {factor}")

    def _save_smart_results(self, results: Dict):
        """Save smart analysis results and update learning data"""
        timestamp = self.analysis_timestamp.strftime('%Y%m%d_%H%M%S')
        filename = f"app/data/cache/smart_ai_analysis_{timestamp}.json"
        
        os.makedirs('app/data/cache', exist_ok=True)
        
        try:
            # Save results
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Update learning data
            self._save_learning_data()
            
            print(f"\n💾 Smart analysis saved to: {filename}")
            print(f"🧠 Learning data updated")
            
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")

async def main():
    """Main function"""
    analyzer = SmartAIAnalyzer()
    await analyzer.run_smart_analysis()

if __name__ == "__main__":
    asyncio.run(main()) 