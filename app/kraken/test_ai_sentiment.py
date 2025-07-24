"""
Test Suite for AI Sentiment Analysis
Tests the SmartAIAnalyzer class for social sentiment and AI predictions
"""

import asyncio
import json
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from smart_ai_analyzer import SmartAIAnalyzer
except ImportError:
    try:
        # Try relative import from current directory
        from .smart_ai_analyzer import SmartAIAnalyzer
    except ImportError:
        print("⚠️  Could not import SmartAIAnalyzer. Ensure the module is in the correct path.")
        sys.exit(1)

class TestAISentiment:
    """Test class for AI sentiment analysis functionality"""
    
    def __init__(self):
        self.analyzer = SmartAIAnalyzer()
        self.test_symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'SOL/USD']
        self.test_results = []
    
    async def test_social_sentiment_analysis(self):
        """Test social sentiment analysis for multiple cryptocurrencies"""
        print("🔍 Testing Social Sentiment Analysis...")
        
        for symbol in self.test_symbols:
            try:
                print(f"\n📊 Analyzing sentiment for {symbol}...")
                
                sentiment_data = await self.analyzer.get_social_sentiment(symbol)
                
                # Validate sentiment data structure
                required_keys = ['symbol', 'timestamp', 'sources', 'aggregated_score', 'confidence', 'signals']
                missing_keys = [key for key in required_keys if key not in sentiment_data]
                
                if missing_keys:
                    print(f"❌ Missing keys in sentiment data: {missing_keys}")
                    return False
                
                # Validate aggregated score range
                score = sentiment_data['aggregated_score']
                if not -1 <= score <= 1:
                    print(f"❌ Aggregated score {score} out of range [-1, 1]")
                    return False
                
                # Validate confidence range
                confidence = sentiment_data['confidence']
                if not 0 <= confidence <= 1:
                    print(f"❌ Confidence {confidence} out of range [0, 1]")
                    return False
                
                # Validate sources
                expected_sources = ['reddit', 'twitter', 'news', 'fear_greed']
                for source in expected_sources:
                    if source not in sentiment_data['sources']:
                        print(f"❌ Missing source: {source}")
                        return False
                
                print(f"✅ {symbol}: Score={score:.3f}, Confidence={confidence:.3f}, Signals={len(sentiment_data['signals'])}")
                
                # Store result for later analysis
                self.test_results.append({
                    'symbol': symbol,
                    'sentiment_data': sentiment_data,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"❌ Error testing sentiment for {symbol}: {e}")
                return False
        
        print("✅ Social sentiment analysis tests passed!")
        return True
    
    async def test_fear_greed_index(self):
        """Test Fear & Greed Index functionality"""
        print("\n🔍 Testing Fear & Greed Index...")
        
        try:
            fear_greed_data = await self.analyzer._get_fear_greed_index()
            
            # Validate structure
            required_keys = ['index', 'sentiment', 'score', 'last_update']
            missing_keys = [key for key in required_keys if key not in fear_greed_data]
            
            if missing_keys:
                print(f"❌ Missing keys in fear & greed data: {missing_keys}")
                return False
            
            # Validate index range
            index_value = fear_greed_data['index']
            if not 0 <= index_value <= 100:
                print(f"❌ Fear & Greed index {index_value} out of range [0, 100]")
                return False
            
            # Validate score range
            score = fear_greed_data['score']
            if not -1 <= score <= 1:
                print(f"❌ Fear & Greed score {score} out of range [-1, 1]")
                return False
            
            # Validate sentiment labels
            valid_sentiments = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
            if fear_greed_data['sentiment'] not in valid_sentiments:
                print(f"❌ Invalid sentiment label: {fear_greed_data['sentiment']}")
                return False
            
            print(f"✅ Fear & Greed Index: {index_value} ({fear_greed_data['sentiment']}) Score: {score}")
            
        except Exception as e:
            print(f"❌ Error testing Fear & Greed Index: {e}")
            return False
        
        print("✅ Fear & Greed Index tests passed!")
        return True
    
    def test_sentiment_signal_generation(self):
        """Test sentiment signal generation logic"""
        print("\n🔍 Testing Sentiment Signal Generation...")
        
        try:
            # Test various sentiment scenarios
            test_scenarios = [
                {
                    'name': 'Strong Bullish',
                    'aggregated_score': 0.8,
                    'confidence': 0.9,
                    'expected_signals': ['STRONG_BUY_SENTIMENT']
                },
                {
                    'name': 'Moderate Bullish',
                    'aggregated_score': 0.4,
                    'confidence': 0.8,
                    'expected_signals': ['BUY_SENTIMENT']
                },
                {
                    'name': 'Strong Bearish',
                    'aggregated_score': -0.7,
                    'confidence': 0.85,
                    'expected_signals': ['STRONG_SELL_SENTIMENT']
                },
                {
                    'name': 'Low Confidence',
                    'aggregated_score': 0.6,
                    'confidence': 0.5,
                    'expected_signals': []  # Should not generate signals due to low confidence
                }
            ]
            
            for scenario in test_scenarios:
                sentiment_data = {
                    'aggregated_score': scenario['aggregated_score'],
                    'confidence': scenario['confidence'],
                    'sources': {
                        'reddit': {'trending': False},
                        'twitter': {'trending': False},
                        'news': {'trending': False},
                        'fear_greed': {'trending': False}
                    }
                }
                
                signals = self.analyzer._generate_sentiment_signals(sentiment_data)
                
                # Check if expected signals are present
                for expected_signal in scenario['expected_signals']:
                    if expected_signal not in signals:
                        print(f"❌ Missing expected signal '{expected_signal}' in scenario '{scenario['name']}'")
                        return False
                
                print(f"✅ {scenario['name']}: Generated signals: {signals}")
            
        except Exception as e:
            print(f"❌ Error testing sentiment signal generation: {e}")
            return False
        
        print("✅ Sentiment signal generation tests passed!")
        return True
    
    async def test_comprehensive_ai_analysis(self):
        """Test comprehensive AI market analysis"""
        print("\n🔍 Testing Comprehensive AI Analysis...")
        
        try:
            # Create sample price data
            dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
            price_data = pd.DataFrame({
                'close': np.random.randn(100).cumsum() + 50000,  # BTC-like prices
                'volume': np.random.randint(1000, 10000, 100)
            }, index=dates)
            
            # Run comprehensive analysis
            analysis = await self.analyzer.analyze_market_with_ai('BTC/USD', price_data)
            
            # Validate analysis structure
            required_sections = [
                'symbol', 'timestamp', 'technical_analysis', 'sentiment_analysis',
                'ml_predictions', 'ai_signals', 'confidence_score', 'risk_assessment'
            ]
            
            missing_sections = [section for section in required_sections if section not in analysis]
            if missing_sections:
                print(f"❌ Missing analysis sections: {missing_sections}")
                return False
            
            # Validate technical analysis
            tech_analysis = analysis['technical_analysis']
            required_tech_keys = ['current_price', 'ma_5', 'ma_20', 'rsi', 'volatility']
            missing_tech_keys = [key for key in required_tech_keys if key not in tech_analysis]
            
            if missing_tech_keys:
                print(f"❌ Missing technical analysis keys: {missing_tech_keys}")
                return False
            
            # Validate confidence score
            confidence = analysis['confidence_score']
            if not 0 <= confidence <= 1:
                print(f"❌ Confidence score {confidence} out of range [0, 1]")
                return False
            
            # Validate risk assessment
            risk_assessment = analysis['risk_assessment']
            required_risk_keys = ['volatility_risk', 'sentiment_risk', 'technical_risk', 'overall_risk']
            missing_risk_keys = [key for key in required_risk_keys if key not in risk_assessment]
            
            if missing_risk_keys:
                print(f"❌ Missing risk assessment keys: {missing_risk_keys}")
                return False
            
            print(f"✅ Comprehensive analysis completed for {analysis['symbol']}")
            print(f"   Technical: Price=${tech_analysis.get('current_price', 0):.2f}, RSI={tech_analysis.get('rsi', 0):.1f}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Signals: {len(analysis['ai_signals'])} generated")
            
        except Exception as e:
            print(f"❌ Error in comprehensive AI analysis test: {e}")
            return False
        
        print("✅ Comprehensive AI analysis tests passed!")
        return True
    
    def test_prediction_confidence_scoring(self):
        """Test prediction confidence scoring"""
        print("\n🔍 Testing Prediction Confidence Scoring...")
        
        try:
            # Test with dummy features (since we might not have trained models)
            dummy_features = np.random.randn(1, 5)  # 1 sample, 5 features
            
            confidence = self.analyzer.calculate_prediction_confidence(dummy_features)
            
            # Validate confidence range
            if not 0 <= confidence <= 1:
                print(f"❌ Confidence score {confidence} out of range [0, 1]")
                return False
            
            print(f"✅ Prediction confidence calculated: {confidence:.3f}")
            
            # Test multiple scenarios
            for i in range(5):
                test_features = np.random.randn(1, 5)
                conf = self.analyzer.calculate_prediction_confidence(test_features)
                
                if not 0 <= conf <= 1:
                    print(f"❌ Invalid confidence in test {i+1}: {conf}")
                    return False
                
                print(f"   Test {i+1}: Confidence = {conf:.3f}")
            
        except Exception as e:
            print(f"❌ Error testing prediction confidence: {e}")
            return False
        
        print("✅ Prediction confidence scoring tests passed!")
        return True
    
    def test_model_performance_tracking(self):
        """Test model performance tracking functionality"""
        print("\n🔍 Testing Model Performance Tracking...")
        
        try:
            # Get performance report
            performance_report = self.analyzer.get_model_performance_report()
            
            # Validate report structure
            required_keys = [
                'model_status', 'recent_accuracy', 'accuracy_trend',
                'total_training_sessions', 'needs_retraining', 'confidence_threshold'
            ]
            
            missing_keys = [key for key in required_keys if key not in performance_report]
            if missing_keys:
                print(f"❌ Missing performance report keys: {missing_keys}")
                return False
            
            # Validate model status
            valid_statuses = ['ACTIVE', 'UNAVAILABLE']
            if performance_report['model_status'] not in valid_statuses:
                print(f"❌ Invalid model status: {performance_report['model_status']}")
                return False
            
            print(f"✅ Model Status: {performance_report['model_status']}")
            print(f"   Recent Accuracy: {performance_report['recent_accuracy']:.3f}")
            print(f"   Training Sessions: {performance_report['total_training_sessions']}")
            print(f"   Needs Retraining: {performance_report['needs_retraining']}")
            
        except Exception as e:
            print(f"❌ Error testing model performance tracking: {e}")
            return False
        
        print("✅ Model performance tracking tests passed!")
        return True
    
    async def run_all_tests(self):
        """Run all AI sentiment tests"""
        print("🚀 Starting AI Sentiment Analysis Test Suite\n")
        print("=" * 60)
        
        test_methods = [
            self.test_social_sentiment_analysis,
            self.test_fear_greed_index,
            self.test_sentiment_signal_generation,
            self.test_comprehensive_ai_analysis,
            self.test_prediction_confidence_scoring,
            self.test_model_performance_tracking
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
                if asyncio.iscoroutinefunction(test_method):
                    result = await test_method()
                else:
                    result = test_method()
                
                if result:
                    passed_tests += 1
                else:
                    print(f"❌ Test failed: {test_method.__name__}")
                    
            except Exception as e:
                print(f"❌ Test error in {test_method.__name__}: {e}")
        
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All AI sentiment tests passed successfully!")
        else:
            print("⚠️  Some tests failed. Please review the output above.")
        
        # Save test results
        self.save_test_results(passed_tests, total_tests)
        
        return passed_tests == total_tests
    
    def save_test_results(self, passed: int, total: int):
        """Save test results to file"""
        try:
            results_data = {
                'test_suite': 'AI Sentiment Analysis',
                'timestamp': datetime.now().isoformat(),
                'passed_tests': passed,
                'total_tests': total,
                'success_rate': passed / total if total > 0 else 0,
                'test_data': self.test_results
            }
            
            # Ensure results directory exists
            results_dir = "app/data/test_results"
            os.makedirs(results_dir, exist_ok=True)
            
            filename = f"ai_sentiment_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(results_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            print(f"💾 Test results saved to: {filepath}")
            
        except Exception as e:
            print(f"⚠️  Could not save test results: {e}")

async def main():
    """Main test execution"""
    print("🤖 AI Sentiment Analysis Test Suite")
    print("Testing smart AI analyzer with social sentiment integration\n")
    
    tester = TestAISentiment()
    success = await tester.run_all_tests()
    
    if success:
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    # Run the tests
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)