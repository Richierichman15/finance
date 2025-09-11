#!/usr/bin/env python3
import unittest
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kraken.smart_ai_analyzer import SmartAIAnalyzer

class TestMLCore(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        print("🔬 Initializing ML test environment...")
        self.ai_analyzer = SmartAIAnalyzer()
        
        # Sample data
        self.sample_data = pd.DataFrame({
            'Close': [100, 102, 98, 103, 105],
            'Volume': [1000, 1200, 800, 1500, 1300]
        })
        
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        print("\n📊 Testing RSI calculation...")
        prices = pd.Series([100, 102, 101, 103, 102, 104, 103, 105, 104, 106])
        rsi = self.ai_analyzer._calculate_rsi(prices)
        self.assertIsInstance(rsi, pd.Series)
        self.assertTrue(all(0 <= x <= 100 for x in rsi.dropna()))
        print("✅ RSI calculation test passed")
        
    def test_signal_determination(self):
        """Test signal determination"""
        print("\n🎯 Testing signal determination...")
        tech_data = {
            'rsi': 65,
            'sma_ratio': 1.02,
            'volume_change': 0.15
        }
        signal = self.ai_analyzer._determine_signal(tech_data)
        self.assertIn(signal, ['BUY', 'SELL', 'HOLD'])
        print(f"✅ Signal determination test passed - Got signal: {signal}")
        
    def test_signal_strength(self):
        """Test signal strength calculation"""
        print("\n💪 Testing signal strength...")
        tech_data = {
            'rsi': 70,
            'sma_ratio': 1.05,
            'volume_change': 0.2
        }
        strength = self.ai_analyzer._calculate_signal_strength(tech_data)
        self.assertIsInstance(strength, float)
        self.assertTrue(0 <= strength <= 1)
        print(f"✅ Signal strength test passed - Strength: {strength:.2f}")
        
    def test_technical_risk(self):
        """Test technical risk assessment"""
        print("\n⚠️ Testing risk assessment...")
        tech_data = {
            'rsi': 75,
            'volatility': 0.15,
            'volume_change': 0.3
        }
        risk = self.ai_analyzer._assess_technical_risk(tech_data)
        self.assertIn(risk, ['LOW', 'MEDIUM', 'HIGH'])
        print(f"✅ Risk assessment test passed - Risk level: {risk}")
        
    def test_prediction_confidence(self):
        """Test prediction confidence calculation"""
        print("\n🎲 Testing prediction confidence...")
        symbol = 'BTC-USD'
        confidence = self.ai_analyzer._get_prediction_confidence(symbol)
        self.assertIsInstance(confidence, float)
        self.assertTrue(0 <= confidence <= 1)
        print(f"✅ Prediction confidence test passed - Confidence: {confidence:.2f}")

def run_tests():
    """Run all tests"""
    print("\n🧪 Starting ML Core Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestMLCore)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    print("\n📝 Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    return len(result.failures) + len(result.errors) == 0

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 