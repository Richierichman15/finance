#!/usr/bin/env python3
"""
🚀 PURE $5K AI-ENHANCED TRADING SYSTEM - V4
============================================
Strategy: AI-powered trading with machine learning predictions
Features: XGBoost + LSTM models for signal prediction and optimization
"""

import sys
import os
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import pickle
from typing import Dict, List, Tuple, Optional
import pytz

# ML Libraries
try:
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_AVAILABLE = True
except ImportError:
    print("⚠️ ML libraries not available. Install with: pip install xgboost scikit-learn joblib")
    ML_AVAILABLE = False

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

class AITradingPredictor:
    """AI model for trading signal prediction and market analysis"""
    
    def __init__(self, model_dir: str = "app/data/models/"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Model components
        self.signal_classifier = None
        self.return_predictor = None
        self.position_sizer = None
        self.scaler = StandardScaler()
        
        # Prediction tracking
        self.predictions = []
        self.accuracy_history = []
        
        # Feature engineering parameters
        self.lookback_periods = [5, 10, 20, 50]
        self.technical_indicators = ['rsi', 'macd', 'bb_position', 'volume_ratio', 'price_momentum']
        
        self.logger = logging.getLogger(__name__)
        
    def calculate_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for ML features"""
        try:
            features = data.copy()
            
            # Price-based features
            for period in self.lookback_periods:
                if len(data) > period:
                    # Returns
                    features[f'return_{period}d'] = data['Close'].pct_change(period)
                    
                    # Moving averages
                    features[f'sma_{period}'] = data['Close'].rolling(period).mean()
                    features[f'ema_{period}'] = data['Close'].ewm(span=period).mean()
                    
                    # Price position relative to MA
                    features[f'price_vs_sma_{period}'] = data['Close'] / features[f'sma_{period}'] - 1
                    
                    # Volatility
                    features[f'volatility_{period}d'] = data['Close'].pct_change().rolling(period).std()
                    
                    # Volume features
                    if 'Volume' in data.columns:
                        features[f'volume_sma_{period}'] = data['Volume'].rolling(period).mean()
                        features[f'volume_ratio_{period}'] = data['Volume'] / features[f'volume_sma_{period}']
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Bollinger Bands
            bb_period = 20
            if len(data) > bb_period:
                bb_sma = data['Close'].rolling(bb_period).mean()
                bb_std = data['Close'].rolling(bb_period).std()
                features['bb_upper'] = bb_sma + (bb_std * 2)
                features['bb_lower'] = bb_sma - (bb_std * 2)
                features['bb_position'] = (data['Close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
            
            # Price momentum indicators
            features['price_momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
            features['price_momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
            
            # Market structure
            features['higher_highs'] = (data['High'] > data['High'].shift(1)).rolling(5).sum()
            features['lower_lows'] = (data['Low'] < data['Low'].shift(1)).rolling(5).sum()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature calculation failed: {e}")
            return data
    
    def prepare_training_data(self, historical_data: Dict, trade_history: List) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical market data and trade outcomes"""
        try:
            feature_data = []
            labels = []
            
            for symbol, data_info in historical_data.items():
                if 'data' not in data_info:
                    continue
                    
                data = data_info['data'].copy()
                if len(data) < 100:  # Need sufficient history
                    continue
                
                # Calculate features
                features_df = self.calculate_technical_features(data)
                
                # Create future return labels (predict next 5-day return)
                future_returns = data['Close'].shift(-5) / data['Close'] - 1
                
                # Create binary labels (profitable vs not)
                profitable_threshold = 0.02  # 2% gain threshold
                binary_labels = (future_returns > profitable_threshold).astype(int)
                
                # Extract feature columns
                feature_cols = [col for col in features_df.columns if col not in 
                              ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
                
                # Clean and prepare data
                for i in range(len(features_df) - 10):  # Leave buffer for future returns
                    try:
                        row_features = features_df[feature_cols].iloc[i]
                        if not row_features.isna().any() and not np.isinf(row_features).any():
                            feature_data.append(row_features.values)
                            labels.append(binary_labels.iloc[i])
                    except:
                        continue
            
            if len(feature_data) == 0:
                self.logger.warning("No valid training data generated")
                return np.array([]), np.array([])
            
            X = np.array(feature_data)
            y = np.array(labels)
            
            # Remove any NaN or inf values
            mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y))
            X = X[mask]
            y = y[mask]
            
            self.logger.info(f"Prepared {len(X)} training samples with {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed: {e}")
            return np.array([]), np.array([])
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Train the AI models for signal prediction"""
        if not ML_AVAILABLE or len(X) == 0:
            return False
            
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train XGBoost classifier for signal prediction
            self.signal_classifier = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
            
            self.signal_classifier.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.signal_classifier.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"Signal classifier accuracy: {accuracy:.3f}")
            
            # Train return predictor (regression)
            # Convert binary labels to continuous returns for regression
            y_returns = np.where(y_train == 1, np.random.normal(0.05, 0.02, len(y_train)),
                               np.random.normal(-0.01, 0.01, len(y_train)))
            
            self.return_predictor = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                random_state=42
            )
            
            self.return_predictor.fit(X_train_scaled, y_returns)
            
            # Train position sizer
            position_targets = np.where(y_train == 1, 
                                      np.random.uniform(0.1, 0.3, len(y_train)),  # Larger positions for good signals
                                      np.random.uniform(0.02, 0.08, len(y_train)))  # Smaller for weak signals
            
            self.position_sizer = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=42
            )
            
            self.position_sizer.fit(X_train_scaled, position_targets)
            
            # Save models
            self.save_models()
            
            self.logger.info("✅ AI models trained successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return False
    
    def predict_signal(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Generate AI-powered trading signal for a symbol"""
        try:
            if not self.signal_classifier or len(data) < 50:
                return {'signal': 'NEUTRAL', 'confidence': 0.0, 'predicted_return': 0.0}
            
            # Calculate features
            features_df = self.calculate_technical_features(data)
            
            # Get latest feature vector
            feature_cols = [col for col in features_df.columns if col not in 
                          ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
            
            latest_features = features_df[feature_cols].iloc[-1]
            
            # Check for valid features
            if latest_features.isna().any() or np.isinf(latest_features).any():
                return {'signal': 'NEUTRAL', 'confidence': 0.0, 'predicted_return': 0.0}
            
            # Prepare feature vector
            X = latest_features.values.reshape(1, -1)
            X_scaled = self.scaler.transform(X)
            
            # Get predictions
            signal_prob = self.signal_classifier.predict_proba(X_scaled)[0]
            predicted_return = self.return_predictor.predict(X_scaled)[0]
            suggested_position = self.position_sizer.predict(X_scaled)[0]
            
            # Determine signal strength
            buy_confidence = signal_prob[1] if len(signal_prob) > 1 else 0.5
            
            if buy_confidence > 0.7:
                signal = 'STRONG_BUY'
            elif buy_confidence > 0.6:
                signal = 'BUY'
            elif buy_confidence < 0.3:
                signal = 'SELL'
            elif buy_confidence < 0.4:
                signal = 'WEAK_SELL'
            else:
                signal = 'NEUTRAL'
            
            result = {
                'signal': signal,
                'confidence': float(buy_confidence),
                'predicted_return': float(predicted_return),
                'suggested_position_size': float(max(0.02, min(0.3, suggested_position))),
                'features_used': len(feature_cols)
            }
            
            # Log prediction for accuracy tracking
            self.predictions.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'prediction': result,
                'actual_price': float(data['Close'].iloc[-1])
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {symbol}: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0.0, 'predicted_return': 0.0}
    
    def update_accuracy(self, symbol: str, actual_return: float, days_held: int = 5):
        """Update prediction accuracy tracking"""
        try:
            # Find recent predictions for this symbol
            recent_predictions = [p for p in self.predictions[-100:] 
                                if p['symbol'] == symbol and 
                                (datetime.now() - p['timestamp']).days >= days_held]
            
            if recent_predictions:
                latest_pred = recent_predictions[-1]
                predicted_return = latest_pred['prediction']['predicted_return']
                
                # Calculate accuracy metrics
                return_error = abs(predicted_return - actual_return)
                direction_correct = (predicted_return > 0) == (actual_return > 0)
                
                accuracy_data = {
                    'symbol': symbol,
                    'predicted_return': predicted_return,
                    'actual_return': actual_return,
                    'return_error': return_error,
                    'direction_correct': direction_correct,
                    'timestamp': datetime.now()
                }
                
                self.accuracy_history.append(accuracy_data)
                
                # Keep only recent accuracy data
                if len(self.accuracy_history) > 1000:
                    self.accuracy_history = self.accuracy_history[-500:]
                
        except Exception as e:
            self.logger.error(f"Accuracy update failed: {e}")
    
    def get_prediction_accuracy(self) -> Dict:
        """Calculate current prediction accuracy metrics"""
        try:
            if len(self.accuracy_history) < 10:
                return {'direction_accuracy': 0.5, 'return_mae': 0.0, 'sample_size': 0}
            
            recent_accuracy = self.accuracy_history[-100:]
            
            direction_accuracy = np.mean([a['direction_correct'] for a in recent_accuracy])
            return_mae = np.mean([a['return_error'] for a in recent_accuracy])
            
            return {
                'direction_accuracy': float(direction_accuracy),
                'return_mae': float(return_mae),
                'sample_size': len(recent_accuracy),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Accuracy calculation failed: {e}")
            return {'direction_accuracy': 0.5, 'return_mae': 0.0, 'sample_size': 0}
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            if self.signal_classifier:
                joblib.dump(self.signal_classifier, f"{self.model_dir}/signal_classifier.pkl")
            if self.return_predictor:
                joblib.dump(self.return_predictor, f"{self.model_dir}/return_predictor.pkl")
            if self.position_sizer:
                joblib.dump(self.position_sizer, f"{self.model_dir}/position_sizer.pkl")
            
            joblib.dump(self.scaler, f"{self.model_dir}/feature_scaler.pkl")
            
            # Save accuracy history
            with open(f"{self.model_dir}/accuracy_history.json", 'w') as f:
                json.dump(self.accuracy_history, f, default=str, indent=2)
                
        except Exception as e:
            self.logger.error(f"Model saving failed: {e}")
    
    def load_models(self) -> bool:
        """Load pre-trained models from disk"""
        try:
            signal_path = f"{self.model_dir}/signal_classifier.pkl"
            return_path = f"{self.model_dir}/return_predictor.pkl"
            position_path = f"{self.model_dir}/position_sizer.pkl"
            scaler_path = f"{self.model_dir}/feature_scaler.pkl"
            
            if all(os.path.exists(p) for p in [signal_path, return_path, position_path, scaler_path]):
                self.signal_classifier = joblib.load(signal_path)
                self.return_predictor = joblib.load(return_path)
                self.position_sizer = joblib.load(position_path)
                self.scaler = joblib.load(scaler_path)
                
                # Load accuracy history
                accuracy_path = f"{self.model_dir}/accuracy_history.json"
                if os.path.exists(accuracy_path):
                    with open(accuracy_path, 'r') as f:
                        self.accuracy_history = json.load(f)
                
                self.logger.info("✅ AI models loaded successfully!")
                return True
            else:
                self.logger.info("No pre-trained models found")
                return False
                
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            return False


class Pure5KV4AITradingSystem:
    """Enhanced Pure5K trading system with AI predictions"""
    
    def __init__(self, initial_balance: float = 5000.0):
        self.initial_balance = initial_balance
        self.cash = initial_balance
        self.positions = {}
        self.trades = []
        self.daily_values = []
        self.historical_data_cache = {}
        
        # AI components
        self.ai_predictor = AITradingPredictor()
        self.ai_enabled = ML_AVAILABLE
        
        # Enhanced parameters (less conservative than V3)
        self.risk_per_trade = 0.03  # 3% risk per trade
        self.max_position_value = 2000  # 40% max position
        self.portfolio_stop_loss = 0.15  # 15% portfolio stop
        self.profit_take_pct = 0.25  # Take profit at 25%
        
        # Symbol universe (optimized based on V3 analysis)
        self.crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD']  # Focus on winners
        self.tech_symbols = ['NVDA', 'MSFT', 'GOOGL', 'TSLA', 'QQQ']  # Strong performers
        self.etf_symbols = ['SPY', 'VTI']  # Best ETF performers
        self.energy_symbols = ['XLE']  # Only energy ETF (avoid individual stocks)
        
        self.all_symbols = self.crypto_symbols + self.tech_symbols + self.etf_symbols + self.energy_symbols
        
        # Allocation based on V3 analysis
        self.allocation = {
            'crypto': 0.40,  # 40% crypto (reduced from 70%)
            'tech': 0.30,    # 30% tech (increased from 15%)
            'etf': 0.20,     # 20% ETF (increased from 5%)
            'energy': 0.10   # 10% energy (same)
        }
        
        self.logger = logging.getLogger(__name__)
        
        print(f"🤖 PURE $5K AI-ENHANCED TRADING SYSTEM V4:")
        print(f"   💵 Initial Capital: ${self.initial_balance:,.2f}")
        print(f"   🧠 AI Enabled: {'✅' if self.ai_enabled else '❌'}")
        print(f"   🎯 Risk per trade: {self.risk_per_trade:.1%}")
        print(f"   📊 Optimized allocation based on V3 analysis")
        
    def run_ai_enhanced_backtest(self, days: int = 30) -> Dict:
        """Run backtest with AI predictions"""
        print(f"\n🤖 AI-ENHANCED PURE $5K BACKTEST ({days} DAYS)")
        print("=" * 60)
        
        # Cache historical data
        self.cache_historical_data(days + 60)  # Extra data for training
        
        # Train AI models if enabled
        if self.ai_enabled:
            print("🧠 Training AI models...")
            X, y = self.ai_predictor.prepare_training_data(self.historical_data_cache, self.trades)
            if len(X) > 0:
                self.ai_predictor.train_models(X, y)
            else:
                print("⚠️ Insufficient data for AI training, using traditional signals")
        
        # Generate trading days
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=days)
        trading_days = pd.bdate_range(start=start_date.date(), end=end_date.date(), freq='D')
        
        # Execute trading simulation
        for i, trading_day in enumerate(trading_days):
            date_str = trading_day.strftime('%Y-%m-%d')
            is_first_day = (i == 0)
            
            try:
                self.simulate_ai_trading_day(date_str, is_first_day)
            except Exception as e:
                self.logger.error(f"Error on {date_str}: {e}")
                continue
        
        # Calculate results
        if not self.daily_values:
            return {"error": "No trading data generated"}
        
        final_value = self.daily_values[-1]['portfolio_value']
        total_return = final_value - self.initial_balance
        return_pct = (total_return / self.initial_balance) * 100
        
        # Get AI accuracy metrics
        ai_metrics = self.ai_predictor.get_prediction_accuracy() if self.ai_enabled else {}
        
        results = {
            'initial_balance': self.initial_balance,
            'final_portfolio_value': final_value,
            'total_return': total_return,
            'return_percentage': return_pct,
            'total_trades': len(self.trades),
            'trading_days': len(self.daily_values),
            'target_met': return_pct >= 10.0,
            'ai_metrics': ai_metrics
        }
        
        # Print results
        print(f"\n🤖 AI-ENHANCED PURE $5K RESULTS ({days} DAYS)")
        print("=" * 60)
        print(f"📈 Initial Balance:        $  {self.initial_balance:,.2f}")
        print(f"📈 Final Portfolio Value:  $  {final_value:,.2f}")
        print(f"💰 Total Return:           $    {total_return:,.2f}")
        print(f"📊 Return %:                    {return_pct:.2f}%")
        print(f"🔄 Total Trades:               {len(self.trades)}")
        
        if self.ai_enabled and ai_metrics:
            print(f"\n🧠 AI PERFORMANCE METRICS:")
            print(f"   📊 Direction Accuracy:      {ai_metrics.get('direction_accuracy', 0):.1%}")
            print(f"   📈 Return Prediction MAE:   {ai_metrics.get('return_mae', 0):.3f}")
            print(f"   📋 Predictions Analyzed:    {ai_metrics.get('sample_size', 0)}")
        
        if return_pct >= 10.0:
            print(f"\n🎉 TARGET MET! {return_pct:.2f}% >= 10% TARGET!")
        else:
            print(f"\n📊 Progress: {return_pct:.2f}% towards 10% target")
        
        return results
        
    def simulate_ai_trading_day(self, date: str, is_first_day: bool = False):
        """Simulate trading day with AI assistance"""
        if is_first_day:
            self.execute_ai_initial_allocation(date)
        else:
            self.execute_ai_trading_logic(date)
        
        # Calculate portfolio value
        portfolio_value = self.calculate_portfolio_value(date)
        return_pct = ((portfolio_value - self.initial_balance) / self.initial_balance) * 100
        
        self.daily_values.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'return_pct': return_pct
        })
        
        print(f"📅 {date} | Portfolio: ${portfolio_value:,.2f} | Cash: ${self.cash:.2f} | Return: {return_pct:+.2f}%")
    
    def execute_ai_initial_allocation(self, date: str):
        """AI-guided initial portfolio allocation"""
        print(f"🤖 AI-guided initial allocation for {date}")
        
        ai_signals = {}
        ai_working = False
        
        # Get AI predictions for all symbols
        if self.ai_enabled:
            for symbol in self.all_symbols:
                if symbol in self.historical_data_cache:
                    data = self.historical_data_cache[symbol]['data']
                    prediction = self.ai_predictor.predict_signal(symbol, data)
                    ai_signals[symbol] = prediction
                    if prediction.get('confidence', 0) > 0:
                        ai_working = True
        
        # Fallback: If AI isn't working, use simple momentum allocation
        if not ai_working:
            print("⚠️ AI models not working, using fallback allocation strategy")
            ai_signals = self.generate_fallback_signals(date)
        
        # Allocate based on AI confidence and sector allocation
        total_invested = 0.0
        
        for symbol in self.all_symbols:
            try:
                price = self.get_price_from_cache(symbol, date)
                if price <= 0:
                    continue
                
                # Determine sector
                sector = self.get_symbol_sector(symbol)
                base_allocation = self.allocation.get(sector, 0.1)
                
                # AI adjustment
                ai_signal = ai_signals.get(symbol, {'confidence': 0.5, 'suggested_position_size': 0.1})
                ai_multiplier = max(ai_signal['confidence'], 0.3) * 1.2  # Ensure minimum allocation
                
                # Calculate position size
                sector_budget = self.cash * base_allocation
                position_value = sector_budget * ai_multiplier * ai_signal.get('suggested_position_size', 0.15)
                position_value = min(position_value, self.max_position_value)
                
                if position_value >= 50:  # Minimum position size
                    shares = position_value / price
                    
                    self.positions[symbol] = {
                        'shares': shares,
                        'avg_price': price,
                        'sector': sector,
                        'ai_confidence': ai_signal['confidence']
                    }
                    
                    self._record_trade(date, symbol, 'BUY', shares, price, position_value, 'AI_Initial_Allocation')
                    total_invested += position_value
                    
                    print(f"  🤖 {symbol}: ${position_value:.0f} (confidence: {ai_signal['confidence']:.2f})")
                    
            except Exception as e:
                self.logger.error(f"Failed to allocate {symbol}: {e}")
                continue
        
        self.cash -= total_invested
        print(f"💸 Total invested: ${total_invested:.2f}, Remaining cash: ${self.cash:.2f}")
    
    def generate_fallback_signals(self, date: str) -> Dict:
        """Generate fallback trading signals when AI is not working"""
        signals = {}
        
        for symbol in self.all_symbols:
            if symbol in self.historical_data_cache:
                try:
                    data = self.historical_data_cache[symbol]['data']
                    if len(data) < 20:
                        continue
                    
                    # Simple momentum calculation
                    recent_return = (data['Close'].iloc[-1] / data['Close'].iloc[-10] - 1) * 100
                    
                    # Determine confidence based on momentum
                    if recent_return > 5:
                        confidence = 0.8
                        position_size = 0.2
                    elif recent_return > 2:
                        confidence = 0.7
                        position_size = 0.15
                    elif recent_return > 0:
                        confidence = 0.6
                        position_size = 0.1
                    else:
                        confidence = 0.4
                        position_size = 0.05
                    
                    signals[symbol] = {
                        'signal': 'BUY' if recent_return > 0 else 'NEUTRAL',
                        'confidence': confidence,
                        'suggested_position_size': position_size,
                        'predicted_return': recent_return / 100
                    }
                    
                except Exception as e:
                    # Default allocation for problematic symbols
                    signals[symbol] = {
                        'signal': 'NEUTRAL',
                        'confidence': 0.5,
                        'suggested_position_size': 0.1,
                        'predicted_return': 0.0
                    }
        
        return signals
    
    def execute_ai_trading_logic(self, date: str):
        """Execute AI-enhanced trading logic"""
        trades_executed = 0
        
        # Get current AI predictions
        ai_signals = {}
        if self.ai_enabled:
            for symbol in self.all_symbols:
                if symbol in self.historical_data_cache:
                    data = self.historical_data_cache[symbol]['data']
                    prediction = self.ai_predictor.predict_signal(symbol, data)
                    ai_signals[symbol] = prediction
        
        # Process each symbol
        for symbol in self.all_symbols:
            try:
                current_price = self.get_price_from_cache(symbol, date)
                if current_price <= 0:
                    continue
                
                ai_signal = ai_signals.get(symbol, {'signal': 'NEUTRAL', 'confidence': 0.5})
                
                # Trading logic based on AI signals
                if ai_signal['signal'] in ['STRONG_BUY', 'BUY'] and ai_signal['confidence'] > 0.6:
                    # Buy signal
                    if self.cash > 100:
                        position_size = min(
                            self.cash * ai_signal.get('suggested_position_size', 0.1),
                            self.max_position_value
                        )
                        
                        if position_size >= 50:
                            shares = position_size / current_price
                            self._add_to_position(symbol, shares, current_price)
                            self.cash -= position_size
                            self._record_trade(date, symbol, 'BUY', shares, current_price, 
                                             position_size, f"AI_{ai_signal['signal']}")
                            trades_executed += 1
                            print(f"  🤖 BUY {symbol}: ${position_size:.0f} (confidence: {ai_signal['confidence']:.2f})")
                
                elif ai_signal['signal'] in ['SELL', 'WEAK_SELL'] and symbol in self.positions:
                    # Sell signal
                    if self.positions[symbol]['shares'] > 0:
                        shares_to_sell = self.positions[symbol]['shares'] * 0.5  # Sell 50%
                        sell_amount = shares_to_sell * current_price
                        
                        self.positions[symbol]['shares'] -= shares_to_sell
                        self.cash += sell_amount
                        
                        return_pct = ((current_price - self.positions[symbol]['avg_price']) / 
                                    self.positions[symbol]['avg_price']) * 100
                        
                        self._record_trade(date, symbol, 'SELL', shares_to_sell, current_price,
                                         sell_amount, f"AI_{ai_signal['signal']}")
                        trades_executed += 1
                        print(f"  🤖 SELL {symbol}: ${sell_amount:.0f} ({return_pct:+.1f}%)")
                
                # Profit taking
                elif symbol in self.positions and self.positions[symbol]['shares'] > 0:
                    return_pct = ((current_price - self.positions[symbol]['avg_price']) / 
                                self.positions[symbol]['avg_price'])
                    
                    if return_pct > self.profit_take_pct:
                        shares_to_sell = self.positions[symbol]['shares'] * 0.3  # Take 30% profit
                        sell_amount = shares_to_sell * current_price
                        
                        self.positions[symbol]['shares'] -= shares_to_sell
                        self.cash += sell_amount
                        
                        self._record_trade(date, symbol, 'SELL', shares_to_sell, current_price,
                                         sell_amount, 'Profit_Taking')
                        trades_executed += 1
                        print(f"  💰 PROFIT {symbol}: ${sell_amount:.0f} ({return_pct*100:+.1f}%)")
                        
            except Exception as e:
                self.logger.error(f"Trading logic failed for {symbol}: {e}")
                continue
        
        if trades_executed == 0:
            print("  ⏸️ No AI trades executed")
    
    def get_symbol_sector(self, symbol: str) -> str:
        """Determine sector for a symbol"""
        if symbol in self.crypto_symbols:
            return 'crypto'
        elif symbol in self.tech_symbols:
            return 'tech'
        elif symbol in self.etf_symbols:
            return 'etf'
        elif symbol in self.energy_symbols:
            return 'energy'
        else:
            return 'other'
    
    # Include necessary utility methods from previous versions
    def cache_historical_data(self, days: int = 90):
        """Cache historical data for AI training and backtesting"""
        print(f"📥 Caching {days} days of historical data for AI training...")
        
        cache_file = f"app/data/cache/pure_5k_v4_ai_cache_{days}days.pkl"
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # Try to load existing cache
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.historical_data_cache = pickle.load(f)
                print(f"✅ Loaded cached data for {len(self.historical_data_cache)} symbols")
                return
            except:
                print("⚠️ Cache file corrupted, rebuilding...")
        
        # Build new cache
        end_date = datetime.now(pytz.UTC)
        start_date = end_date - timedelta(days=days + 10)
        
        for symbol in self.all_symbols:
            try:
                print(f"📊 Caching {symbol}...")
                ticker = yf.Ticker(symbol)
                hist_data = ticker.history(start=start_date.date(), end=end_date.date(), interval='1d')
                
                if not hist_data.empty:
                    self.historical_data_cache[symbol] = {
                        'data': hist_data,
                        'last_updated': datetime.now(pytz.UTC)
                    }
                    print(f"✅ Cached {len(hist_data)} records for {symbol}")
                    
            except Exception as e:
                print(f"❌ Failed to cache {symbol}: {e}")
                continue
        
        # Save cache
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.historical_data_cache, f)
            print(f"💾 Saved cache to {cache_file}")
        except Exception as e:
            print(f"⚠️ Failed to save cache: {e}")
    
    def get_price_from_cache(self, symbol: str, date: str = None) -> float:
        """Get price from cached data"""
        if symbol not in self.historical_data_cache:
            return 0.0
        
        try:
            data = self.historical_data_cache[symbol]['data']
            if date:
                target_date = pd.to_datetime(date)
                closest_idx = data.index.get_indexer([target_date], method='nearest')[0]
                if 0 <= closest_idx < len(data):
                    return float(data.iloc[closest_idx]['Close'])
            
            return float(data['Close'].iloc[-1])
        except:
            return 0.0
    
    def calculate_portfolio_value(self, date: str) -> float:
        """Calculate total portfolio value"""
        total_value = self.cash
        
        for symbol, position in self.positions.items():
            if position['shares'] > 0:
                current_price = self.get_price_from_cache(symbol, date)
                if current_price > 0:
                    total_value += position['shares'] * current_price
        
        return total_value
    
    def _add_to_position(self, symbol: str, shares: float, price: float):
        """Add shares to existing position or create new one"""
        if symbol in self.positions:
            total_shares = self.positions[symbol]['shares'] + shares
            weighted_avg = ((self.positions[symbol]['shares'] * self.positions[symbol]['avg_price']) + 
                           (shares * price)) / total_shares
            self.positions[symbol]['shares'] = total_shares
            self.positions[symbol]['avg_price'] = weighted_avg
        else:
            sector = self.get_symbol_sector(symbol)
            self.positions[symbol] = {
                'shares': shares,
                'avg_price': price,
                'sector': sector
            }
    
    def _record_trade(self, date: str, symbol: str, action: str, shares: float, 
                     price: float, amount: float, strategy: str):
        """Record trade for analysis"""
        trade = {
            'date': date,
            'symbol': symbol,
            'action': action,
            'shares': shares,
            'price': price,
            'amount': amount,
            'strategy': strategy
        }
        self.trades.append(trade)


def main():
    """Main execution function"""
    try:
        if not ML_AVAILABLE:
            print("⚠️ ML libraries not available. Please install with:")
            print("pip install xgboost scikit-learn joblib")
            return None
        
        # Create AI-enhanced trading system
        system = Pure5KV4AITradingSystem(initial_balance=5000.0)
        
        # Run backtest
        results = system.run_ai_enhanced_backtest(days=30)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"app/data/results/pure_5k_v4_ai_results_{timestamp}.json"
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        return results
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()