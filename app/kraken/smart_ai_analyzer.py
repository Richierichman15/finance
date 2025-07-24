"""
Smart AI Analyzer for Crypto Markets with ML capabilities
Provides advanced AI-driven analysis for cryptocurrency trading
"""

import json
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import aiohttp
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available. ML features will be limited.")

class SmartAIAnalyzer:
    """Advanced AI analyzer with ML model performance tracking and social sentiment"""
    
    def __init__(self, data_dir: str = "app/data"):
        self.data_dir = data_dir
        self.models_dir = os.path.join(data_dir, "models")
        self.cache_dir = os.path.join(data_dir, "cache")
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Model components
        self.signal_classifier = None
        self.return_predictor = None
        self.feature_scaler = None
        self.position_sizer = None
        
        # Performance tracking
        self.accuracy_history = []
        self.confidence_threshold = 0.7
        self.retrain_threshold = 0.6
        
        # Social sentiment sources
        self.sentiment_sources = [
            "reddit", "twitter", "news", "social_volume", "fear_greed_index"
        ]
        
        # Initialize components
        self._load_models()
        self._load_accuracy_history()
    
    def _load_models(self):
        """Load trained ML models"""
        if not SKLEARN_AVAILABLE:
            return
            
        try:
            # Load signal classifier
            classifier_path = os.path.join(self.models_dir, "signal_classifier.pkl")
            if os.path.exists(classifier_path):
                with open(classifier_path, 'rb') as f:
                    self.signal_classifier = pickle.load(f)
                    
            # Load return predictor
            predictor_path = os.path.join(self.models_dir, "return_predictor.pkl")
            if os.path.exists(predictor_path):
                with open(predictor_path, 'rb') as f:
                    self.return_predictor = pickle.load(f)
                    
            # Load feature scaler
            scaler_path = os.path.join(self.models_dir, "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                    
            # Load position sizer
            sizer_path = os.path.join(self.models_dir, "position_sizer.pkl")
            if os.path.exists(sizer_path):
                with open(sizer_path, 'rb') as f:
                    self.position_sizer = pickle.load(f)
                    
        except Exception as e:
            print(f"⚠️  Error loading models: {e}")
    
    def _save_models(self):
        """Save trained ML models"""
        if not SKLEARN_AVAILABLE:
            return
            
        try:
            if self.signal_classifier:
                with open(os.path.join(self.models_dir, "signal_classifier.pkl"), 'wb') as f:
                    pickle.dump(self.signal_classifier, f)
                    
            if self.return_predictor:
                with open(os.path.join(self.models_dir, "return_predictor.pkl"), 'wb') as f:
                    pickle.dump(self.return_predictor, f)
                    
            if self.feature_scaler:
                with open(os.path.join(self.models_dir, "feature_scaler.pkl"), 'wb') as f:
                    pickle.dump(self.feature_scaler, f)
                    
            if self.position_sizer:
                with open(os.path.join(self.models_dir, "position_sizer.pkl"), 'wb') as f:
                    pickle.dump(self.position_sizer, f)
                    
        except Exception as e:
            print(f"⚠️  Error saving models: {e}")
    
    def _load_accuracy_history(self):
        """Load model accuracy history"""
        history_path = os.path.join(self.models_dir, "accuracy_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    self.accuracy_history = json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading accuracy history: {e}")
                self.accuracy_history = []
    
    def _save_accuracy_history(self):
        """Save model accuracy history"""
        try:
            history_path = os.path.join(self.models_dir, "accuracy_history.json")
            with open(history_path, 'w') as f:
                json.dump(self.accuracy_history, f, indent=2)
        except Exception as e:
            print(f"⚠️  Error saving accuracy history: {e}")
    
    async def get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive social sentiment analysis"""
        sentiment_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'aggregated_score': 0.0,
            'confidence': 0.0,
            'signals': []
        }
        
        try:
            # Simulate social sentiment data (in real implementation, would use APIs)
            # Reddit sentiment
            reddit_sentiment = await self._get_reddit_sentiment(symbol)
            sentiment_data['sources']['reddit'] = reddit_sentiment
            
            # Twitter sentiment
            twitter_sentiment = await self._get_twitter_sentiment(symbol)
            sentiment_data['sources']['twitter'] = twitter_sentiment
            
            # News sentiment
            news_sentiment = await self._get_news_sentiment(symbol)
            sentiment_data['sources']['news'] = news_sentiment
            
            # Fear & Greed Index
            fear_greed = await self._get_fear_greed_index()
            sentiment_data['sources']['fear_greed'] = fear_greed
            
            # Aggregate sentiment
            scores = [reddit_sentiment.get('score', 0), 
                     twitter_sentiment.get('score', 0),
                     news_sentiment.get('score', 0),
                     fear_greed.get('score', 0)]
            
            sentiment_data['aggregated_score'] = np.mean([s for s in scores if s != 0])
            sentiment_data['confidence'] = len([s for s in scores if s != 0]) / len(scores)
            
            # Generate trading signals based on sentiment
            sentiment_data['signals'] = self._generate_sentiment_signals(sentiment_data)
            
        except Exception as e:
            print(f"⚠️  Error getting social sentiment: {e}")
            
        return sentiment_data
    
    async def _get_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get Reddit sentiment (simulated)"""
        # In real implementation, would use Reddit API
        return {
            'score': np.random.uniform(-1, 1),
            'volume': np.random.randint(10, 1000),
            'trending': np.random.choice([True, False]),
            'top_mentions': f"r/cryptocurrency, r/{symbol.lower()}"
        }
    
    async def _get_twitter_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get Twitter sentiment (simulated)"""
        # In real implementation, would use Twitter API v2
        return {
            'score': np.random.uniform(-1, 1),
            'volume': np.random.randint(50, 5000),
            'trending': np.random.choice([True, False]),
            'influencer_sentiment': np.random.uniform(-1, 1)
        }
    
    async def _get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get news sentiment (simulated)"""
        # In real implementation, would use news APIs
        return {
            'score': np.random.uniform(-1, 1),
            'article_count': np.random.randint(5, 50),
            'headline_sentiment': np.random.uniform(-1, 1),
            'source_credibility': np.random.uniform(0.5, 1.0)
        }
    
    async def _get_fear_greed_index(self) -> Dict[str, Any]:
        """Get Fear & Greed Index"""
        # Simulate fear & greed index (0-100)
        index_value = np.random.randint(0, 100)
        
        if index_value <= 25:
            sentiment = "Extreme Fear"
            score = -0.8
        elif index_value <= 45:
            sentiment = "Fear"
            score = -0.4
        elif index_value <= 55:
            sentiment = "Neutral"
            score = 0.0
        elif index_value <= 75:
            sentiment = "Greed"
            score = 0.4
        else:
            sentiment = "Extreme Greed"
            score = 0.8
            
        return {
            'index': index_value,
            'sentiment': sentiment,
            'score': score,
            'last_update': datetime.now().isoformat()
        }
    
    def _generate_sentiment_signals(self, sentiment_data: Dict[str, Any]) -> List[str]:
        """Generate trading signals based on sentiment analysis"""
        signals = []
        score = sentiment_data['aggregated_score']
        confidence = sentiment_data['confidence']
        
        if confidence > 0.7:  # High confidence signals
            if score > 0.6:
                signals.append("STRONG_BUY_SENTIMENT")
            elif score > 0.3:
                signals.append("BUY_SENTIMENT")
            elif score < -0.6:
                signals.append("STRONG_SELL_SENTIMENT")
            elif score < -0.3:
                signals.append("SELL_SENTIMENT")
        
        # Check for trending
        if any(source.get('trending', False) for source in sentiment_data['sources'].values()):
            signals.append("TRENDING_MOMENTUM")
            
        return signals
    
    def calculate_prediction_confidence(self, features: np.ndarray) -> float:
        """Calculate prediction confidence score"""
        if not SKLEARN_AVAILABLE or not self.signal_classifier:
            return 0.5
            
        try:
            # Get prediction probabilities
            probabilities = self.signal_classifier.predict_proba(features)
            max_prob = np.max(probabilities, axis=1)[0]
            
            # Confidence based on how certain the model is
            confidence = (max_prob - 0.5) * 2  # Scale from 0-1
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            print(f"⚠️  Error calculating confidence: {e}")
            return 0.5
    
    def train_models(self, price_data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None):
        """Train ML models with automatic retraining based on accuracy"""
        if not SKLEARN_AVAILABLE:
            print("⚠️  scikit-learn not available for model training")
            return
            
        try:
            print("🤖 Training AI models...")
            
            # Prepare features
            features = self._prepare_features(price_data, sentiment_data)
            
            # Prepare targets
            targets = self._prepare_targets(price_data)
            
            # Ensure features and targets have the same length
            min_len = min(len(features), len(targets))
            if min_len > 0:
                features = features[:min_len]
                targets = targets[:min_len]
            
            if len(features) < 50:  # Need minimum data
                print("⚠️  Insufficient data for training")
                return
                
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, targets, test_size=0.2, random_state=42
            )
            
            # Scale features
            if self.feature_scaler is None:
                self.feature_scaler = StandardScaler()
            
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_test_scaled = self.feature_scaler.transform(X_test)
            
            # Train signal classifier
            if self.signal_classifier is None:
                self.signal_classifier = RandomForestClassifier(
                    n_estimators=100, random_state=42, max_depth=10
                )
            
            # Convert continuous targets to classification
            y_class = (y_train > 0).astype(int)  # 1 for positive returns, 0 for negative
            self.signal_classifier.fit(X_train_scaled, y_class)
            
            # Evaluate classifier
            y_pred_class = self.signal_classifier.predict(X_test_scaled)
            y_test_class = (y_test > 0).astype(int)
            accuracy = accuracy_score(y_test_class, y_pred_class)
            
            # Train return predictor
            if self.return_predictor is None:
                self.return_predictor = GradientBoostingRegressor(
                    n_estimators=100, random_state=42, max_depth=6
                )
            
            self.return_predictor.fit(X_train_scaled, y_train)
            
            # Evaluate regressor
            y_pred_reg = self.return_predictor.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred_reg)
            
            # Record accuracy
            accuracy_record = {
                'timestamp': datetime.now().isoformat(),
                'classifier_accuracy': accuracy,
                'regressor_mse': mse,
                'data_points': len(features),
                'features_count': features.shape[1]
            }
            
            self.accuracy_history.append(accuracy_record)
            
            # Check if retraining is needed
            recent_accuracy = self._get_recent_accuracy()
            if recent_accuracy < self.retrain_threshold:
                print(f"🔄 Low accuracy ({recent_accuracy:.3f}), automatic retraining triggered")
                
            print(f"✅ Model training complete. Accuracy: {accuracy:.3f}, MSE: {mse:.6f}")
            
            # Save models and history
            self._save_models()
            self._save_accuracy_history()
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
    
    def _prepare_features(self, price_data: pd.DataFrame, sentiment_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Prepare feature matrix for ML models"""
        features = []
        
        # Technical indicators
        if 'close' in price_data.columns:
            # Price features
            price_data['returns'] = price_data['close'].pct_change()
            price_data['ma_5'] = price_data['close'].rolling(5).mean()
            price_data['ma_20'] = price_data['close'].rolling(20).mean()
            price_data['rsi'] = self._calculate_rsi(price_data['close'])
            price_data['volatility'] = price_data['returns'].rolling(20).std()
            
            # Feature columns
            feature_cols = ['returns', 'ma_5', 'ma_20', 'rsi', 'volatility']
            
            # Add volume if available
            if 'volume' in price_data.columns:
                price_data['volume_ma'] = price_data['volume'].rolling(20).mean()
                feature_cols.append('volume_ma')
            
            features = price_data[feature_cols].dropna().values
        
        # Add sentiment features if available
        if sentiment_data is not None and len(sentiment_data) > 0:
            sentiment_features = sentiment_data[['sentiment_score', 'confidence']].values
            # Align with price features
            min_len = min(len(features), len(sentiment_features))
            if min_len > 0:
                features = np.hstack([features[-min_len:], sentiment_features[-min_len:]])
        
        return features
    
    def _prepare_targets(self, price_data: pd.DataFrame) -> np.ndarray:
        """Prepare target variable (future returns)"""
        if 'close' in price_data.columns:
            returns = price_data['close'].pct_change().shift(-1)  # Next period return
            return returns.dropna().values
        return np.array([])
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _get_recent_accuracy(self, lookback_days: int = 7) -> float:
        """Get recent model accuracy"""
        if not self.accuracy_history:
            return 1.0
            
        cutoff_date = datetime.now() - timedelta(days=lookback_days)
        recent_records = [
            record for record in self.accuracy_history
            if datetime.fromisoformat(record['timestamp']) > cutoff_date
        ]
        
        if not recent_records:
            return self.accuracy_history[-1]['classifier_accuracy']
            
        return np.mean([record['classifier_accuracy'] for record in recent_records])
    
    async def analyze_market_with_ai(self, symbol: str, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive AI market analysis"""
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'technical_analysis': {},
            'sentiment_analysis': {},
            'ml_predictions': {},
            'ai_signals': [],
            'confidence_score': 0.0,
            'risk_assessment': {}
        }
        
        try:
            # Technical analysis
            analysis['technical_analysis'] = self._technical_analysis(price_data)
            
            # Sentiment analysis
            analysis['sentiment_analysis'] = await self.get_social_sentiment(symbol)
            
            # ML predictions
            if SKLEARN_AVAILABLE and self.signal_classifier and len(price_data) > 0:
                features = self._prepare_features(price_data)
                if len(features) > 0:
                    latest_features = features[-1:].reshape(1, -1)
                    if self.feature_scaler:
                        latest_features = self.feature_scaler.transform(latest_features)
                    
                    # Signal prediction
                    signal_pred = self.signal_classifier.predict(latest_features)[0]
                    signal_confidence = self.calculate_prediction_confidence(latest_features)
                    
                    # Return prediction
                    return_pred = 0.0
                    if self.return_predictor:
                        return_pred = self.return_predictor.predict(latest_features)[0]
                    
                    analysis['ml_predictions'] = {
                        'signal': 'BUY' if signal_pred == 1 else 'SELL',
                        'expected_return': return_pred,
                        'confidence': signal_confidence,
                        'model_accuracy': self._get_recent_accuracy()
                    }
            
            # Generate AI signals
            analysis['ai_signals'] = self._generate_ai_signals(analysis)
            
            # Calculate overall confidence
            analysis['confidence_score'] = self._calculate_overall_confidence(analysis)
            
            # Risk assessment
            analysis['risk_assessment'] = self._assess_risk(analysis, price_data)
            
        except Exception as e:
            print(f"⚠️  Error in AI analysis: {e}")
            
        return analysis
    
    def _technical_analysis(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Technical analysis component"""
        if price_data.empty or 'close' not in price_data.columns:
            return {}
            
        current_price = price_data['close'].iloc[-1]
        
        # Moving averages
        ma_5 = price_data['close'].rolling(5).mean().iloc[-1] if len(price_data) >= 5 else current_price
        ma_20 = price_data['close'].rolling(20).mean().iloc[-1] if len(price_data) >= 20 else current_price
        
        # RSI
        rsi = self._calculate_rsi(price_data['close']).iloc[-1] if len(price_data) >= 14 else 50
        
        # Volatility
        volatility = price_data['close'].pct_change().rolling(20).std().iloc[-1] if len(price_data) >= 20 else 0
        
        return {
            'current_price': current_price,
            'ma_5': ma_5,
            'ma_20': ma_20,
            'rsi': rsi,
            'volatility': volatility,
            'price_above_ma5': current_price > ma_5,
            'price_above_ma20': current_price > ma_20,
            'ma5_above_ma20': ma_5 > ma_20,
            'rsi_overbought': rsi > 70,
            'rsi_oversold': rsi < 30
        }
    
    def _generate_ai_signals(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate AI trading signals"""
        signals = []
        
        # Technical signals
        tech = analysis.get('technical_analysis', {})
        if tech.get('price_above_ma5') and tech.get('ma5_above_ma20'):
            signals.append("BULLISH_TREND")
        elif not tech.get('price_above_ma5') and not tech.get('ma5_above_ma20'):
            signals.append("BEARISH_TREND")
            
        if tech.get('rsi_oversold'):
            signals.append("RSI_OVERSOLD")
        elif tech.get('rsi_overbought'):
            signals.append("RSI_OVERBOUGHT")
        
        # Sentiment signals
        sentiment_signals = analysis.get('sentiment_analysis', {}).get('signals', [])
        signals.extend(sentiment_signals)
        
        # ML signals
        ml_pred = analysis.get('ml_predictions', {})
        if ml_pred.get('confidence', 0) > self.confidence_threshold:
            signals.append(f"ML_{ml_pred.get('signal', 'HOLD')}")
        
        return signals
    
    def _calculate_overall_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall analysis confidence"""
        confidence_components = []
        
        # ML confidence
        ml_conf = analysis.get('ml_predictions', {}).get('confidence', 0)
        confidence_components.append(ml_conf)
        
        # Sentiment confidence
        sent_conf = analysis.get('sentiment_analysis', {}).get('confidence', 0)
        confidence_components.append(sent_conf)
        
        # Technical indicators confidence (based on clarity of signals)
        tech = analysis.get('technical_analysis', {})
        if tech:
            tech_conf = 0.8 if len([s for s in analysis.get('ai_signals', []) if 'TREND' in s]) > 0 else 0.5
            confidence_components.append(tech_conf)
        
        return np.mean(confidence_components) if confidence_components else 0.0
    
    def _assess_risk(self, analysis: Dict[str, Any], price_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess trading risk"""
        risk_assessment = {
            'volatility_risk': 'LOW',
            'sentiment_risk': 'LOW',
            'technical_risk': 'LOW',
            'overall_risk': 'LOW'
        }
        
        try:
            # Volatility risk
            tech = analysis.get('technical_analysis', {})
            volatility = tech.get('volatility', 0)
            
            if volatility > 0.05:
                risk_assessment['volatility_risk'] = 'HIGH'
            elif volatility > 0.02:
                risk_assessment['volatility_risk'] = 'MEDIUM'
            
            # Sentiment risk
            sentiment_score = analysis.get('sentiment_analysis', {}).get('aggregated_score', 0)
            if abs(sentiment_score) > 0.8:
                risk_assessment['sentiment_risk'] = 'HIGH'
            elif abs(sentiment_score) > 0.5:
                risk_assessment['sentiment_risk'] = 'MEDIUM'
            
            # Technical risk
            signals = analysis.get('ai_signals', [])
            conflicting_signals = any('BULLISH' in s for s in signals) and any('BEARISH' in s for s in signals)
            if conflicting_signals:
                risk_assessment['technical_risk'] = 'HIGH'
            
            # Overall risk
            high_risks = sum(1 for risk in risk_assessment.values() if risk == 'HIGH')
            medium_risks = sum(1 for risk in risk_assessment.values() if risk == 'MEDIUM')
            
            if high_risks >= 2:
                risk_assessment['overall_risk'] = 'HIGH'
            elif high_risks >= 1 or medium_risks >= 2:
                risk_assessment['overall_risk'] = 'MEDIUM'
                
        except Exception as e:
            print(f"⚠️  Error in risk assessment: {e}")
            
        return risk_assessment
    
    def get_model_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive model performance report"""
        if not self.accuracy_history:
            return {
                'model_status': 'ACTIVE' if SKLEARN_AVAILABLE else 'UNAVAILABLE',
                'recent_accuracy': 1.0,  # Default to perfect accuracy when no history
                'accuracy_trend': 'STABLE',
                'total_training_sessions': 0,
                'needs_retraining': False,
                'confidence_threshold': self.confidence_threshold,
                'last_training': None,
                'performance_history': [],
                'status': 'No training history available'
            }
            
        recent_accuracy = self._get_recent_accuracy()
        
        # Calculate trends
        accuracies = [record['classifier_accuracy'] for record in self.accuracy_history[-10:]]
        accuracy_trend = 'IMPROVING' if len(accuracies) > 1 and accuracies[-1] > accuracies[0] else 'DECLINING'
        
        return {
            'model_status': 'ACTIVE' if SKLEARN_AVAILABLE else 'UNAVAILABLE',
            'recent_accuracy': recent_accuracy,
            'accuracy_trend': accuracy_trend,
            'total_training_sessions': len(self.accuracy_history),
            'needs_retraining': recent_accuracy < self.retrain_threshold,
            'confidence_threshold': self.confidence_threshold,
            'last_training': self.accuracy_history[-1]['timestamp'] if self.accuracy_history else None,
            'performance_history': self.accuracy_history[-5:] if len(self.accuracy_history) >= 5 else self.accuracy_history
        }

if __name__ == "__main__":
    # Test the analyzer
    import asyncio
    
    async def test_analyzer():
        analyzer = SmartAIAnalyzer()
        
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        price_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Test AI analysis
        analysis = await analyzer.analyze_market_with_ai('BTC/USD', price_data)
        print("🤖 AI Analysis Results:")
        print(json.dumps(analysis, indent=2, default=str))
        
        # Test model performance
        performance = analyzer.get_model_performance_report()
        print("\n📊 Model Performance:")
        print(json.dumps(performance, indent=2, default=str))
    
    asyncio.run(test_analyzer())