import unittest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from smart_ai_analyzer import SmartAIAnalyzer, SentimentAnalyzer

class TestAIAndSentimentImprovements(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.ai_analyzer = SmartAIAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Sample portfolio data
        self.sample_portfolio = {
            'BTC-USD': {
                'amount': 0.5,
                'value_usd': 20000,
                'allocation_pct': 40
            },
            'ETH-USD': {
                'amount': 5,
                'value_usd': 15000,
                'allocation_pct': 30
            },
            'SOL-USD': {
                'amount': 100,
                'value_usd': 10000,
                'allocation_pct': 20
            },
            'ZUSD': {
                'amount': 5000,
                'value_usd': 5000,
                'allocation_pct': 10
            }
        }
        
        # Sample historical data
        self.sample_historical = pd.DataFrame({
            'Close': [100, 102, 98, 103, 105],
            'Volume': [1000, 1200, 800, 1500, 1300],
            'market_cap': [1e9, 1.1e9, 0.95e9, 1.05e9, 1.15e9],
            'dominance': [40, 41, 39, 42, 43]
        })
        
    def test_ml_model_initialization(self):
        """Test ML model initialization"""
        self.assertIsNotNone(self.ai_analyzer.models['price_predictor'])
        self.assertIsNotNone(self.ai_analyzer.models['trend_classifier'])
        self.assertIsNotNone(self.ai_analyzer.models['risk_analyzer'])
        self.assertIsNotNone(self.ai_analyzer.scalers['features'])
        
    def test_feature_preparation(self):
        """Test feature preparation for ML models"""
        features = self.ai_analyzer._prepare_features(self.sample_historical)
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(features.shape[0] == len(self.sample_historical))
        
    @patch('smart_ai_analyzer.SentimentAnalyzer.get_social_sentiment')
    async def test_sentiment_analysis(self, mock_sentiment):
        """Test sentiment analysis functionality"""
        mock_sentiment.return_value = {
            'overall_sentiment': 0.5,
            'sentiment_strength': 0.7,
            'mentions_count': 100,
            'trending_score': 0.8
        }
        
        sentiment = await self.sentiment_analyzer.get_social_sentiment('BTC-USD')
        self.assertIsInstance(sentiment, dict)
        self.assertIn('overall_sentiment', sentiment)
        self.assertIn('sentiment_strength', sentiment)
        self.assertIn('mentions_count', sentiment)
        self.assertIn('trending_score', sentiment)
        
    @patch('smart_ai_analyzer.SmartAIAnalyzer._get_technical_data')
    async def test_price_prediction(self, mock_tech_data):
        """Test price prediction functionality"""
        mock_tech_data.return_value = self.sample_historical.to_dict()
        
        predictions = await self.ai_analyzer._predict_price_movement('BTC-USD', self.sample_historical)
        self.assertIn('price_change_pred', predictions)
        self.assertIn('trend_probability', predictions)
        self.assertIn('risk_score', predictions)
        self.assertIn('confidence', predictions)
        
    def test_risk_assessment(self):
        """Test risk assessment functionality"""
        predictions = {
            'BTC-USD': {
                'risk_score': 0.6,
                'confidence': 0.8
            }
        }
        sentiment = {
            'BTC-USD': {
                'sentiment_strength': 0.7,
                'trending_score': 0.9,
                'overall_sentiment': -0.6
            }
        }
        
        risk_assessment = self.ai_analyzer._assess_smart_risk(
            self.sample_portfolio, predictions, sentiment
        )
        
        self.assertIn('overall_risk_score', risk_assessment)
        self.assertIn('risk_factors', risk_assessment)
        self.assertIn('risk_metrics', risk_assessment)
        self.assertIn('sentiment_risks', risk_assessment)
        
    def test_recommendation_generation(self):
        """Test recommendation generation"""
        predictions = {
            'BTC-USD': {
                'price_change_pred': 0.05,
                'trend_probability': 0.8,
                'confidence': 0.75
            }
        }
        sentiment = {
            'BTC-USD': {
                'sentiment_strength': 0.7,
                'trending_score': 0.8,
                'overall_sentiment': 0.6
            }
        }
        
        recommendations = self.ai_analyzer._generate_smart_recommendations(
            self.sample_portfolio, predictions, sentiment
        )
        
        self.assertIsInstance(recommendations, list)
        if recommendations:  # If we got recommendations
            self.assertIn('symbol', recommendations[0])
            self.assertIn('action', recommendations[0])
            self.assertIn('confidence', recommendations[0])
            self.assertIn('reasons', recommendations[0])
            
    def test_model_update(self):
        """Test model update functionality"""
        predictions = {
            'price_change_pred': 0.05,
            'trend_probability': 0.8
        }
        actual_outcomes = {
            'data': self.sample_historical
        }
        
        # This shouldn't raise any exceptions
        self.ai_analyzer._update_models('BTC-USD', predictions, actual_outcomes)
        
    @patch('smart_ai_analyzer.SmartAIAnalyzer._get_technical_data')
    async def test_full_analysis_pipeline(self, mock_tech_data):
        """Test the complete analysis pipeline"""
        mock_tech_data.return_value = self.sample_historical.to_dict()
        
        results = await self.ai_analyzer._run_smart_analysis(self.sample_portfolio)
        
        self.assertIn('portfolio_overview', results)
        self.assertIn('technical_signals', results)
        self.assertIn('ml_predictions', results)
        self.assertIn('sentiment_analysis', results)
        self.assertIn('risk_assessment', results)
        self.assertIn('recommendations', results)

def run_tests():
    """Run all tests"""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAIAndSentimentImprovements)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    # Run async tests
    loop = asyncio.get_event_loop()
    loop.run_until_complete(unittest.main()) 