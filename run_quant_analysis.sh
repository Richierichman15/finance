#!/bin/bash
# 🎯 QUANTITATIVE ADVISOR LAUNCHER
# ===============================
# Simple script to run portfolio analysis

echo "🚀 Starting Quantitative Portfolio Advisor..."
echo "📊 Analyzing your Kraken portfolio with professional-grade mathematical models"
echo ""

# Set Python path and run the advisor
PYTHONPATH=/Users/gitonga-nyaga/github/finance python3 app/kraken/quant_advisor.py

echo ""
echo "✅ Analysis complete! Check the results above."
echo "💾 Analysis data saved to app/data/cache/"
echo ""
echo "🔄 To run again: ./run_quant_analysis.sh" 