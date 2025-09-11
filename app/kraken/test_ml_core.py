import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from smart_ai_analyzer import SmartAIAnalyzer

class TestMLCore(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.ai_analyzer = SmartAIAnalyzer()
        
        # Sample data
        self.sample_data = pd.DataFrame({
            'Close': [100, 102, 98, 103, 105],
            'Volume': [1000, 1200, 800, 1500, 1300]
        })
        
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105, 104, 106])
        rsi = self.ai_analyzer._calculate_rsi(prices)
        self.assertIsInstance(rsi, pd.Series)
        self.assertTrue(all(0 <= x <= 100 for x in rsi.dropna()))
        
    def test_signal_determination(self):
        """Test signal determination"""
        tech_data = {
            'rsi': 65,
            'sma_ratio': 1.02,
            'volume_change': 0.15
        }
        signal = self.ai_analyzer._determine_signal(tech_data)
        self.assertIn(signal, ['BUY', 'SELL', 'HOLD'])
        
    def test_signal_strength(self):
        """Test signal strength calculation"""
        tech_data = {
            'rsi': 70,
            'sma_ratio': 1.05,
            'volume_change': 0.2
        }
        strength = self.ai_analyzer._calculate_signal_strength(tech_data)
        self.assertIsInstance(strength, float)
        self.assertTrue(0 <= strength <= 1)
        
    def test_technical_risk(self):
        """Test technical risk assessment"""
        tech_data = {
            'rsi': 75,
            'volatility': 0.15,
            'volume_change': 0.3
        }
        risk = self.ai_analyzer._assess_technical_risk(tech_data)
        self.assertIn(risk, ['LOW', 'MEDIUM', 'HIGH'])
        
    def test_prediction_confidence(self):
        """Test prediction confidence calculation"""
        symbol = 'BTC-USD'
        confidence = self.ai_analyzer._get_prediction_confidence(symbol)
        self.assertIsInstance(confidence, float)
        self.assertTrue(0 <= confidence <= 1)

def run_tests():
    """Run all tests"""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMLCore)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    run_tests() 