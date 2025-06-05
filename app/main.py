from fastapi import FastAPI, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
import uvicorn
from typing import List, Dict, Any
import asyncio
from datetime import datetime

from .services.ollama_service import OllamaService
from .services.financial_advisor import FinancialAdvisor
from .services.news_collector import NewsCollector
from .services.portfolio_manager import PortfolioManager
from .models.schemas import (
    PortfolioRequest, 
    AdviceRequest, 
    PortfolioResponse,
    AdviceResponse,
    NewsResponse
)

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
        "allocations": [
            {"name": "Broad U.S. Equities", "percentage": 40, "etfs": ["VTI", "SCHB"]},
            {"name": "U.S. Energy Sector", "percentage": 20, "etfs": ["XLE", "AMLP"]},
            {"name": "Global Infrastructure", "percentage": 15, "etfs": ["IGF", "BIPC"]},
            {"name": "Technology Growth", "percentage": 15, "etfs": ["QQQ", "VGT"]},
            {"name": "Real Estate", "percentage": 5, "etfs": ["VNQ"]},
            {"name": "Speculative Crypto", "percentage": 5, "description": "High-risk, high-reward"}
        ],
        "philosophy": "Aggressive growth with diversified risk. We're here to make BANK! 💰"
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

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)