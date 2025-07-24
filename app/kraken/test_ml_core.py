"""
Test Suite for ML Core Functionality
Tests the PortfolioAnalyzer class and ML model training/prediction features
"""

import json
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from portfolio_analyzer import PortfolioAnalyzer
    from smart_ai_analyzer import SmartAIAnalyzer
except ImportError:
    try:
        # Try relative imports from current directory
        from .portfolio_analyzer import PortfolioAnalyzer
        from .smart_ai_analyzer import SmartAIAnalyzer
    except ImportError:
        print("⚠️  Could not import analyzer modules. Ensure the modules are in the correct path.")
        sys.exit(1)

class TestMLCore:
    """Test class for ML core functionality"""
    
    def __init__(self):
        self.portfolio_analyzer = PortfolioAnalyzer()
        self.ai_analyzer = SmartAIAnalyzer()
        self.test_results = []
        
        # Create comprehensive test data
        self.sample_portfolio = {
            'positions': {
                'BTC/USD': {'quantity': 0.5, 'value': 25000},
                'ETH/USD': {'quantity': 10, 'value': 15000},
                'ADA/USD': {'quantity': 1000, 'value': 500},
                'SOL/USD': {'quantity': 50, 'value': 2500},
                'LINK/USD': {'quantity': 100, 'value': 1000},
                'UNI/USD': {'quantity': 200, 'value': 1500}
            }
        }
        
        # Create realistic price history
        self.price_history = self._generate_test_price_data()
    
    def _generate_test_price_data(self):
        """Generate realistic test price data for multiple timeframes"""
        price_history = {}
        
        # Create 2 years of daily data
        dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
        
        for symbol in self.sample_portfolio['positions'].keys():
            # Generate price movements with some correlation to BTC
            if symbol == 'BTC/USD':
                # BTC base prices
                base_price = 40000
                returns = np.random.normal(0.001, 0.03, len(dates))  # Small daily drift with volatility
            elif symbol == 'ETH/USD':
                # ETH correlated with BTC but more volatile
                base_price = 2500
                btc_returns = np.random.normal(0.001, 0.03, len(dates))
                eth_specific = np.random.normal(0, 0.02, len(dates))
                returns = btc_returns * 1.2 + eth_specific  # Higher correlation and volatility
            else:
                # Altcoins with varying correlations
                base_price = 10 if 'ADA' in symbol else 100
                btc_returns = np.random.normal(0.001, 0.03, len(dates))
                alt_specific = np.random.normal(0, 0.04, len(dates))
                correlation = 0.6  # Moderate correlation with BTC
                returns = btc_returns * correlation + alt_specific
            
            # Generate price series
            prices = [base_price]
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Generate volume with some trend
            volume = np.random.lognormal(10, 1, len(dates))
            
            price_history[symbol] = pd.DataFrame({
                'close': prices,
                'volume': volume,
                'high': np.array(prices) * (1 + np.random.uniform(0, 0.02, len(prices))),
                'low': np.array(prices) * (1 - np.random.uniform(0, 0.02, len(prices))),
                'open': np.roll(prices, 1)  # Previous day's close as open
            }, index=dates)
        
        return price_history
    
    def test_portfolio_overview(self):
        """Test basic portfolio overview calculations"""
        print("🔍 Testing Portfolio Overview...")
        
        try:
            overview = self.portfolio_analyzer._calculate_portfolio_overview(self.sample_portfolio)
            
            # Validate overview structure
            required_keys = [
                'total_value', 'total_assets', 'largest_position',
                'cash_percentage', 'crypto_percentage', 'diversification_score'
            ]
            
            missing_keys = [key for key in required_keys if key not in overview]
            if missing_keys:
                print(f"❌ Missing overview keys: {missing_keys}")
                return False
            
            # Validate calculations
            expected_total_value = sum(pos['value'] for pos in self.sample_portfolio['positions'].values())
            if abs(overview['total_value'] - expected_total_value) > 0.01:
                print(f"❌ Total value mismatch: {overview['total_value']} vs {expected_total_value}")
                return False
            
            # Validate diversification score
            if not 0 <= overview['diversification_score'] <= 100:
                print(f"❌ Diversification score out of range: {overview['diversification_score']}")
                return False
            
            print(f"✅ Total Value: ${overview['total_value']:,.2f}")
            print(f"   Assets: {overview['total_assets']}")
            print(f"   Largest Position: {overview['largest_position']['symbol']} ({overview['largest_position']['percentage']:.1f}%)")
            print(f"   Diversification Score: {overview['diversification_score']:.1f}")
            
        except Exception as e:
            print(f"❌ Error testing portfolio overview: {e}")
            return False
        
        print("✅ Portfolio overview tests passed!")
        return True
    
    def test_multi_timeframe_analysis(self):
        """Test multi-timeframe analysis (7d, 30d, 90d, 1y)"""
        print("\n🔍 Testing Multi-Timeframe Analysis...")
        
        try:
            timeframe_analysis = self.portfolio_analyzer._multi_timeframe_analysis(
                self.sample_portfolio, self.price_history
            )
            
            # Validate all timeframes are present
            expected_timeframes = ['7d', '30d', '90d', '1y']
            missing_timeframes = [tf for tf in expected_timeframes if tf not in timeframe_analysis]
            
            if missing_timeframes:
                print(f"❌ Missing timeframes: {missing_timeframes}")
                return False
            
            # Validate each timeframe analysis
            for timeframe, analysis in timeframe_analysis.items():
                required_keys = [
                    'return', 'volatility', 'sharpe_ratio', 'max_drawdown',
                    'win_rate', 'best_performer', 'worst_performer'
                ]
                
                missing_keys = [key for key in required_keys if key not in analysis]
                if missing_keys:
                    print(f"❌ Missing keys in {timeframe} analysis: {missing_keys}")
                    return False
                
                # Validate ranges
                if not -100 <= analysis['return'] <= 1000:  # Allow for crypto volatility
                    print(f"❌ {timeframe} return out of reasonable range: {analysis['return']}")
                    return False
                
                if not 0 <= analysis['win_rate'] <= 100:
                    print(f"❌ {timeframe} win rate out of range: {analysis['win_rate']}")
                    return False
                
                print(f"✅ {timeframe}: Return={analysis['return']:.2f}%, Volatility={analysis['volatility']:.1f}%, Sharpe={analysis['sharpe_ratio']:.2f}")
            
        except Exception as e:
            print(f"❌ Error testing multi-timeframe analysis: {e}")
            return False
        
        print("✅ Multi-timeframe analysis tests passed!")
        return True
    
    def test_sharpe_ratio_calculations(self):
        """Test Sharpe ratio calculations across timeframes"""
        print("\n🔍 Testing Sharpe Ratio Calculations...")
        
        try:
            risk_metrics = self.portfolio_analyzer._calculate_risk_metrics(
                self.sample_portfolio, self.price_history
            )
            
            # Validate Sharpe ratios
            sharpe_ratios = risk_metrics.get('sharpe_ratio', {})
            expected_periods = ['30d', '90d', '1y']
            
            for period in expected_periods:
                if period not in sharpe_ratios:
                    print(f"❌ Missing Sharpe ratio for {period}")
                    return False
                
                sharpe = sharpe_ratios[period]
                if not -5 <= sharpe <= 5:  # Reasonable range for Sharpe ratios
                    print(f"❌ {period} Sharpe ratio out of reasonable range: {sharpe}")
                    return False
                
                print(f"✅ {period} Sharpe Ratio: {sharpe:.3f}")
            
            # Test Sortino ratios
            sortino_ratios = risk_metrics.get('sortino_ratio', {})
            for period in expected_periods:
                if period in sortino_ratios:
                    print(f"✅ {period} Sortino Ratio: {sortino_ratios[period]:.3f}")
            
        except Exception as e:
            print(f"❌ Error testing Sharpe ratio calculations: {e}")
            return False
        
        print("✅ Sharpe ratio calculation tests passed!")
        return True
    
    def test_drawdown_analysis(self):
        """Test comprehensive drawdown analysis"""
        print("\n🔍 Testing Drawdown Analysis...")
        
        try:
            drawdown_analysis = self.portfolio_analyzer._calculate_drawdown_analysis(
                self.sample_portfolio, self.price_history
            )
            
            # Validate drawdown analysis structure
            required_keys = [
                'current_drawdown', 'max_drawdown_1y', 'max_drawdown_duration',
                'avg_drawdown', 'drawdown_frequency', 'recovery_time_avg',
                'underwater_periods', 'drawdown_severity'
            ]
            
            missing_keys = [key for key in required_keys if key not in drawdown_analysis]
            if missing_keys:
                print(f"❌ Missing drawdown analysis keys: {missing_keys}")
                return False
            
            # Validate drawdown values
            max_drawdown = drawdown_analysis['max_drawdown_1y']
            if max_drawdown > 0:  # Drawdowns should be negative
                print(f"❌ Max drawdown should be negative: {max_drawdown}")
                return False
            
            if not -100 <= max_drawdown <= 0:
                print(f"❌ Max drawdown out of reasonable range: {max_drawdown}")
                return False
            
            # Validate severity assessment
            valid_severities = ['LOW', 'MEDIUM', 'HIGH', 'EXTREME']
            if drawdown_analysis['drawdown_severity'] not in valid_severities:
                print(f"❌ Invalid drawdown severity: {drawdown_analysis['drawdown_severity']}")
                return False
            
            print(f"✅ Current Drawdown: {drawdown_analysis['current_drawdown']:.2f}%")
            print(f"   Max Drawdown (1y): {drawdown_analysis['max_drawdown_1y']:.2f}%")
            print(f"   Average Recovery Time: {drawdown_analysis['recovery_time_avg']:.1f} days")
            print(f"   Severity: {drawdown_analysis['drawdown_severity']}")
            
        except Exception as e:
            print(f"❌ Error testing drawdown analysis: {e}")
            return False
        
        print("✅ Drawdown analysis tests passed!")
        return True
    
    def test_beta_calculations(self):
        """Test portfolio beta calculations vs BTC/ETH"""
        print("\n🔍 Testing Beta Calculations vs BTC/ETH...")
        
        try:
            beta_analysis = self.portfolio_analyzer._calculate_beta_analysis(
                self.sample_portfolio, self.price_history
            )
            
            # Validate beta analysis structure
            required_keys = [
                'beta_btc', 'beta_eth', 'beta_interpretation',
                'systematic_risk', 'idiosyncratic_risk'
            ]
            
            missing_keys = [key for key in required_keys if key not in beta_analysis]
            if missing_keys:
                print(f"❌ Missing beta analysis keys: {missing_keys}")
                return False
            
            # Validate beta values for different timeframes
            for benchmark in ['beta_btc', 'beta_eth']:
                beta_values = beta_analysis[benchmark]
                expected_periods = ['30d', '90d', '1y']
                
                for period in expected_periods:
                    if period not in beta_values:
                        print(f"❌ Missing {benchmark} for {period}")
                        return False
                    
                    beta = beta_values[period]
                    if not -2 <= beta <= 3:  # Reasonable range for crypto betas
                        print(f"❌ {benchmark} {period} out of reasonable range: {beta}")
                        return False
                
                print(f"✅ {benchmark.upper()}: 30d={beta_values['30d']:.2f}, 90d={beta_values['90d']:.2f}, 1y={beta_values['1y']:.2f}")
            
            # Validate systematic vs idiosyncratic risk
            sys_risk = beta_analysis['systematic_risk']
            idio_risk = beta_analysis['idiosyncratic_risk']
            
            if not 0 <= sys_risk <= 100:
                print(f"❌ Systematic risk out of range: {sys_risk}")
                return False
            
            if not 0 <= idio_risk <= 100:
                print(f"❌ Idiosyncratic risk out of range: {idio_risk}")
                return False
            
            # Should approximately sum to 100%
            total_risk = sys_risk + idio_risk
            if not 90 <= total_risk <= 110:  # Allow some tolerance
                print(f"❌ Risk components don't sum properly: {total_risk}")
                return False
            
            print(f"✅ Risk Decomposition: Systematic={sys_risk:.1f}%, Idiosyncratic={idio_risk:.1f}%")
            
        except Exception as e:
            print(f"❌ Error testing beta calculations: {e}")
            return False
        
        print("✅ Beta calculation tests passed!")
        return True
    
    def test_sector_exposure_analysis(self):
        """Test sector/category exposure analysis"""
        print("\n🔍 Testing Sector Exposure Analysis...")
        
        try:
            sector_analysis = self.portfolio_analyzer._analyze_sector_exposure(self.sample_portfolio)
            
            # Validate sector analysis structure
            required_keys = [
                'sector_allocation', 'sector_diversification_score', 'largest_sector',
                'sector_concentration_risk', 'recommended_rebalancing'
            ]
            
            missing_keys = [key for key in required_keys if key not in sector_analysis]
            if missing_keys:
                print(f"❌ Missing sector analysis keys: {missing_keys}")
                return False
            
            # Validate sector allocations
            sector_allocation = sector_analysis['sector_allocation']
            if not sector_allocation:
                print("❌ No sector allocation data")
                return False
            
            # Check that percentages sum to approximately 100%
            total_allocation = sum(sector_allocation.values())
            if not 95 <= total_allocation <= 105:  # Allow some tolerance
                print(f"❌ Sector allocations don't sum to 100%: {total_allocation}")
                return False
            
            # Validate diversification score
            div_score = sector_analysis['sector_diversification_score']
            if not 0 <= div_score <= 100:
                print(f"❌ Diversification score out of range: {div_score}")
                return False
            
            # Validate concentration risk
            valid_risks = ['LOW', 'MEDIUM', 'HIGH']
            if sector_analysis['sector_concentration_risk'] not in valid_risks:
                print(f"❌ Invalid concentration risk: {sector_analysis['sector_concentration_risk']}")
                return False
            
            print(f"✅ Sector Allocation:")
            for sector, percentage in sector_allocation.items():
                print(f"   {sector}: {percentage:.1f}%")
            
            print(f"✅ Diversification Score: {div_score:.1f}")
            print(f"   Concentration Risk: {sector_analysis['sector_concentration_risk']}")
            print(f"   Largest Sector: {sector_analysis['largest_sector']['name']} ({sector_analysis['largest_sector']['percentage']:.1f}%)")
            
        except Exception as e:
            print(f"❌ Error testing sector exposure analysis: {e}")
            return False
        
        print("✅ Sector exposure analysis tests passed!")
        return True
    
    def test_ml_model_training(self):
        """Test ML model training functionality"""
        print("\n🔍 Testing ML Model Training...")
        
        try:
            # Create training data
            dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
            training_data = pd.DataFrame({
                'close': np.random.randn(200).cumsum() + 50000,
                'volume': np.random.randint(1000, 10000, 200)
            }, index=dates)
            
            # Test model training
            print("🤖 Training ML models...")
            self.ai_analyzer.train_models(training_data)
            
            # Test model performance report
            performance_report = self.ai_analyzer.get_model_performance_report()
            
            # Validate performance report
            required_keys = [
                'model_status', 'recent_accuracy', 'accuracy_trend',
                'total_training_sessions', 'needs_retraining'
            ]
            
            missing_keys = [key for key in required_keys if key not in performance_report]
            if missing_keys:
                print(f"❌ Missing performance report keys: {missing_keys}")
                return False
            
            print(f"✅ Model Status: {performance_report['model_status']}")
            print(f"   Training Sessions: {performance_report['total_training_sessions']}")
            print(f"   Recent Accuracy: {performance_report['recent_accuracy']:.3f}")
            
        except Exception as e:
            print(f"❌ Error testing ML model training: {e}")
            return False
        
        print("✅ ML model training tests passed!")
        return True
    
    def test_comprehensive_portfolio_analysis(self):
        """Test the complete comprehensive portfolio analysis"""
        print("\n🔍 Testing Comprehensive Portfolio Analysis...")
        
        try:
            # Run full comprehensive analysis
            comprehensive_analysis = self.portfolio_analyzer.analyze_portfolio_comprehensive(
                self.sample_portfolio, self.price_history
            )
            
            # Validate comprehensive analysis structure
            required_sections = [
                'timestamp', 'portfolio_overview', 'multi_timeframe_analysis',
                'risk_metrics', 'performance_metrics', 'beta_analysis',
                'sector_analysis', 'drawdown_analysis', 'recommendations'
            ]
            
            missing_sections = [section for section in required_sections if section not in comprehensive_analysis]
            if missing_sections:
                print(f"❌ Missing comprehensive analysis sections: {missing_sections}")
                return False
            
            # Validate that recommendations are generated
            recommendations = comprehensive_analysis['recommendations']
            if not isinstance(recommendations, list):
                print("❌ Recommendations should be a list")
                return False
            
            print(f"✅ Comprehensive analysis completed with {len(recommendations)} recommendations")
            
            # Store results for later analysis
            self.test_results.append({
                'test_type': 'comprehensive_analysis',
                'timestamp': datetime.now().isoformat(),
                'analysis': comprehensive_analysis
            })
            
        except Exception as e:
            print(f"❌ Error testing comprehensive portfolio analysis: {e}")
            return False
        
        print("✅ Comprehensive portfolio analysis tests passed!")
        return True
    
    def test_performance_metrics(self):
        """Test advanced performance metrics"""
        print("\n🔍 Testing Performance Metrics...")
        
        try:
            performance_metrics = self.portfolio_analyzer._calculate_performance_metrics(
                self.sample_portfolio, self.price_history
            )
            
            # Validate performance metrics structure
            required_keys = [
                'total_return_1y', 'annualized_return', 'monthly_returns',
                'rolling_returns', 'alpha_vs_btc', 'alpha_vs_eth',
                'information_ratio_btc', 'information_ratio_eth',
                'calmar_ratio', 'sterling_ratio'
            ]
            
            missing_keys = [key for key in required_keys if key not in performance_metrics]
            if missing_keys:
                print(f"❌ Missing performance metrics keys: {missing_keys}")
                return False
            
            # Validate monthly returns
            monthly_returns = performance_metrics['monthly_returns']
            if not isinstance(monthly_returns, list):
                print("❌ Monthly returns should be a list")
                return False
            
            # Validate rolling returns
            rolling_returns = performance_metrics['rolling_returns']
            if not isinstance(rolling_returns, dict):
                print("❌ Rolling returns should be a dict")
                return False
            
            required_rolling_keys = ['mean', 'std', 'max', 'min']
            missing_rolling_keys = [key for key in required_rolling_keys if key not in rolling_returns]
            if missing_rolling_keys:
                print(f"❌ Missing rolling returns keys: {missing_rolling_keys}")
                return False
            
            print(f"✅ Total Return (1y): {performance_metrics['total_return_1y']:.2f}%")
            print(f"   Annualized Return: {performance_metrics['annualized_return']:.2f}%")
            print(f"   Alpha vs BTC: {performance_metrics['alpha_vs_btc']:.2f}%")
            print(f"   Alpha vs ETH: {performance_metrics['alpha_vs_eth']:.2f}%")
            print(f"   Calmar Ratio: {performance_metrics['calmar_ratio']:.3f}")
            
        except Exception as e:
            print(f"❌ Error testing performance metrics: {e}")
            return False
        
        print("✅ Performance metrics tests passed!")
        return True
    
    def run_all_tests(self):
        """Run all ML core tests"""
        print("🚀 Starting ML Core Test Suite\n")
        print("=" * 60)
        
        test_methods = [
            self.test_portfolio_overview,
            self.test_multi_timeframe_analysis,
            self.test_sharpe_ratio_calculations,
            self.test_drawdown_analysis,
            self.test_beta_calculations,
            self.test_sector_exposure_analysis,
            self.test_performance_metrics,
            self.test_ml_model_training,
            self.test_comprehensive_portfolio_analysis
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_method in test_methods:
            try:
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
            print("🎉 All ML core tests passed successfully!")
        else:
            print("⚠️  Some tests failed. Please review the output above.")
        
        # Save test results
        self.save_test_results(passed_tests, total_tests)
        
        return passed_tests == total_tests
    
    def save_test_results(self, passed: int, total: int):
        """Save test results to file"""
        try:
            results_data = {
                'test_suite': 'ML Core Functionality',
                'timestamp': datetime.now().isoformat(),
                'passed_tests': passed,
                'total_tests': total,
                'success_rate': passed / total if total > 0 else 0,
                'test_data': self.test_results
            }
            
            # Ensure results directory exists
            results_dir = "app/data/test_results"
            os.makedirs(results_dir, exist_ok=True)
            
            filename = f"ml_core_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(results_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            
            print(f"💾 Test results saved to: {filepath}")
            
        except Exception as e:
            print(f"⚠️  Could not save test results: {e}")

def main():
    """Main test execution"""
    print("📊 ML Core Functionality Test Suite")
    print("Testing portfolio analyzer with advanced metrics and ML features\n")
    
    tester = TestMLCore()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All tests completed successfully!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    # Run the tests
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)