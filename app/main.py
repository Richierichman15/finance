from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import uvicorn
from typing import List, Dict, Any
import asyncio
from datetime import datetime
from sqlalchemy.orm import Session

from .services.ollama_service import OllamaService
from .services.financial_advisor import FinancialAdvisor
from .services.news_collector import NewsCollector
from .services.portfolio_manager import PortfolioManager
from .services.stock_data_collector import StockDataCollector
from .models.schemas import (
    PortfolioRequest, 
    AdviceRequest, 
    PortfolioResponse,
    AdviceResponse,
    NewsResponse
)
from . import models
from .database import engine, get_db

# Create database tables
models.database_models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Elite Financial Advisor AI",
    description="An aggressive, profit-focused AI financial advisor powered by Ollama",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Initialize services
ollama_service = OllamaService()
financial_advisor = FinancialAdvisor(ollama_service)
news_collector = NewsCollector()
portfolio_manager = PortfolioManager()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await ollama_service.initialize()
    print("🚀 Elite Financial Advisor AI is ready to make you RICH!")

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "alive", "message": "Ready to make insane profits! 💰"}

@app.post("/api/portfolio/analyze", response_model=PortfolioResponse)
async def analyze_portfolio(portfolio_data: PortfolioRequest):
    """Analyze portfolio and get aggressive recommendations"""
    try:
        analysis = await portfolio_manager.analyze_portfolio(portfolio_data.dict())
        advice = await financial_advisor.get_portfolio_advice(analysis)
        
        return PortfolioResponse(
            analysis=analysis,
            recommendations=advice["recommendations"],
            risk_assessment=advice["risk_assessment"],
            profit_potential=advice["profit_potential"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/advice", response_model=AdviceResponse)
async def get_financial_advice(request: AdviceRequest):
    """Get aggressive financial advice from the AI"""
    try:
        # Get current market context
        news = await news_collector.get_market_news()
        market_data = await news_collector.get_market_sentiment()
        
        # Generate advice
        advice = await financial_advisor.generate_advice(
            question=request.question,
            portfolio_value=request.portfolio_value,
            risk_tolerance=request.risk_tolerance,
            market_context={"news": news, "sentiment": market_data}
        )
        
        return AdviceResponse(
            advice=advice["response"],
            confidence_level=advice["confidence"],
            action_items=advice["actions"],
            timestamp=datetime.now()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advice generation failed: {str(e)}")

@app.get("/api/news", response_model=NewsResponse)
async def get_market_news():
    """Get latest financial news and market analysis"""
    try:
        news = await news_collector.get_comprehensive_news()
        return NewsResponse(
            headlines=news["headlines"],
            market_sentiment=news["sentiment"],
            sector_analysis=news["sectors"],
            recommendations=news["opportunities"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News collection failed: {str(e)}")

@app.get("/api/portfolio/default")
async def get_default_portfolio():
    """Get the default aggressive portfolio allocation"""
    return {
        "allocation": {
            "US_Equities": {
                "percentage": 40,
                "etfs": ["VTI", "SCHB"]
            },
            "Energy_Sector": {
                "percentage": 20,
                "etfs": ["XLE", "AMLP"]
            },
            "Global_Infrastructure": {
                "percentage": 15,
                "etfs": ["IGF", "BIPC"]
            },
            "Technology": {
                "percentage": 15,
                "etfs": ["QQQ", "VGT"]
            },
            "Real_Estate": {
                "percentage": 5,
                "etfs": ["VNQ"]
            },
            "Crypto": {
                "percentage": 5,
                "exchanges": ["Coinbase", "Kraken"]
            }
        }
    }

@app.post("/api/rebalance")
async def rebalance_portfolio(portfolio_data: PortfolioRequest):
    """Get rebalancing recommendations"""
    try:
        rebalance_plan = await portfolio_manager.generate_rebalance_plan(portfolio_data.dict())
        ai_commentary = await financial_advisor.comment_on_rebalancing(rebalance_plan)
        
        return {
            "rebalance_plan": rebalance_plan,
            "ai_commentary": ai_commentary,
            "execution_priority": "IMMEDIATE - Markets wait for no one!"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Rebalancing failed: {str(e)}")

@app.post("/api/portfolio")
def create_portfolio(name: str, db: Session = Depends(get_db)):
    portfolio = models.database_models.Portfolio(name=name)
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    return portfolio

@app.get("/api/news/store")
def store_news_in_db(db: Session = Depends(get_db)):
    news_collector = NewsCollector()
    news_items = news_collector.get_latest_news()
    
    # Store news items in database
    for item in news_items:
        db_item = models.database_models.NewsItem(**item.dict())
        db.add(db_item)
    db.commit()
    
    return news_items

@app.get("/api/portfolio/{portfolio_id}/holdings")
def get_portfolio_holdings(portfolio_id: int, db: Session = Depends(get_db)):
    portfolio = db.query(models.database_models.Portfolio).filter(models.database_models.Portfolio.id == portfolio_id).first()
    if not portfolio:
        return {"error": "Portfolio not found"}
    return portfolio.holdings

@app.get("/api/stock/{symbol}/fundamentals")
def get_stock_fundamentals(symbol: str, limit: int = 10):
    """Get latest fundamental data for a specific stock"""
    collector = StockDataCollector()
    data = collector.get_latest_fundamentals(symbol.upper(), limit)
    if not data:
        raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
    return {
        "symbol": symbol.upper(),
        "records_count": len(data),
        "data": data
    }

@app.get("/api/stocks/fundamentals")
def get_all_stock_fundamentals(limit: int = 5):
    """Get latest fundamental data for all tracked stocks"""
    collector = StockDataCollector()
    data = collector.get_all_latest_fundamentals(limit)
    return {
        "tracked_stocks": list(data.keys()),
        "data": data
    }

@app.post("/api/stocks/collect")
def collect_stock_data_now(symbols: list = None):
    """Manually trigger stock data collection"""
    collector = StockDataCollector()
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"]
    
    results = collector.collect_and_store_fundamentals(symbols)
    return {
        "message": "Data collection completed",
        "results": results
    }

@app.get("/api/test/fundamentals")
def test_fundamentals_endpoint():
    """Test endpoint to debug routing"""
    try:
        collector = StockDataCollector()
        data = collector.get_latest_fundamentals('AAPL', 1)
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)