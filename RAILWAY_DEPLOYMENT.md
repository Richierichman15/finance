# Railway Deployment Guide

## 🚀 Quick Deploy to Railway

Your trading system is now ready for Railway deployment! Here's what I've set up:

### Files Created/Modified:
- ✅ `Procfile` - Defines start commands for Railway
- ✅ `railway.json` - Railway-specific configuration
- ✅ `nixpacks.toml` - Nixpacks build configuration
- ✅ `start.py` - Flexible startup script
- ✅ `requirements.txt` - Updated with missing dependencies

### Deployment Steps:

1. **Push to GitHub** (if not already done):
   ```bash
   git add .
   git commit -m "Add Railway deployment configuration"
   git push origin kraken_old_symbols
   ```

2. **Deploy on Railway**:
   - Go to [Railway.app](https://railway.app)
   - Connect your GitHub repository
   - Select this project
   - Railway will automatically detect the configuration

3. **Set Environment Variables** in Railway dashboard:
   ```
   RUN_MODE=web
   PORT=8000
   PAPER_TRADING=true
   INITIAL_BALANCE=5000.0
   ```

4. **Optional API Keys** (for live trading):
   ```
   KRAKEN_API_KEY=your_key_here
   KRAKEN_API_SECRET=your_secret_here
   ```

### Service Types:

- **Web Service**: Runs the FastAPI dashboard and API endpoints
- **Worker Service**: Runs the live trading system (optional, for background trading)

### Monitoring:

- **Health Check**: `https://your-app.railway.app/api/health`
- **Dashboard**: `https://your-app.railway.app/`
- **API Docs**: `https://your-app.railway.app/docs`

### Trading System Features:

- 📊 Real-time portfolio monitoring
- 🤖 AI-powered trading decisions
- 📈 Live performance tracking
- 💰 Paper trading mode (safe testing)
- 📱 Web dashboard for monitoring

### Troubleshooting:

If deployment fails:
1. Check Railway logs for specific errors
2. Ensure all environment variables are set
3. Verify database connectivity
4. Check API key permissions

Your trading algorithm will now run live for a month and you can monitor its performance through the web dashboard!
