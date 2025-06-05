#!/bin/bash

# Setup script for stock data collection cron jobs
# This will run the data collection 8 times per day (every 3 hours)

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_PATH="$PROJECT_DIR/venv/bin/python"
SCRIPT_PATH="$PROJECT_DIR/collect_stock_data.py"
LOG_DIR="$PROJECT_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "Setting up stock data collection cron jobs..."
echo "Project directory: $PROJECT_DIR"
echo "Python path: $PYTHON_PATH"
echo "Script path: $SCRIPT_PATH"

# Create the cron job entries
CRON_JOBS="
# Stock Data Collection - 8 times per day (every 3 hours)
# Collect fundamental data for AAPL, MSFT, GOOGL, TSLA
0 0 * * * cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH >> $LOG_DIR/stock_data.log 2>&1
0 3 * * * cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH >> $LOG_DIR/stock_data.log 2>&1
0 6 * * * cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH >> $LOG_DIR/stock_data.log 2>&1
0 9 * * * cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH >> $LOG_DIR/stock_data.log 2>&1
0 12 * * * cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH >> $LOG_DIR/stock_data.log 2>&1
0 15 * * * cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH >> $LOG_DIR/stock_data.log 2>&1
0 18 * * * cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH >> $LOG_DIR/stock_data.log 2>&1
0 21 * * * cd $PROJECT_DIR && $PYTHON_PATH $SCRIPT_PATH >> $LOG_DIR/stock_data.log 2>&1
"

echo "Cron jobs to be added:"
echo "$CRON_JOBS"

# Check if user wants to install the cron jobs
read -p "Do you want to install these cron jobs? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Add to crontab
    (crontab -l 2>/dev/null; echo "$CRON_JOBS") | crontab -
    echo "✅ Cron jobs installed successfully!"
    echo ""
    echo "The stock data will be collected at:"
    echo "  - 12:00 AM (midnight)"
    echo "  - 3:00 AM"
    echo "  - 6:00 AM"
    echo "  - 9:00 AM"
    echo "  - 12:00 PM (noon)"
    echo "  - 3:00 PM"
    echo "  - 6:00 PM"
    echo "  - 9:00 PM"
    echo ""
    echo "Logs will be stored in: $LOG_DIR/stock_data.log"
    echo ""
    echo "To view current cron jobs: crontab -l"
    echo "To remove these cron jobs: crontab -e (and delete the stock data lines)"
else
    echo "❌ Cron jobs not installed."
fi

echo ""
echo "Manual commands:"
echo "  Test collection: $PYTHON_PATH $SCRIPT_PATH --verbose"
echo "  View logs: tail -f $LOG_DIR/stock_data.log"
echo "  Test specific stocks: $PYTHON_PATH $SCRIPT_PATH --symbols AAPL TSLA" 