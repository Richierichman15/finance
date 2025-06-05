from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class RiskTolerance(str, Enum):
    conservative = "conservative"
    moderate = "moderate"
    aggressive = "aggressive"
    yolo = "yolo"  # For the most aggressive investors

class PortfolioRequest(BaseModel):
    current_value: float = Field(..., description="Current portfolio value in USD")
    allocations: Dict[str, float] = Field(..., description="Current allocation percentages")
    cash_available: float = Field(default=0, description="Available cash for investment")
    investment_timeline: str = Field(default="long_term", description="Investment timeline")

class AdviceRequest(BaseModel):
    question: str = Field(..., description="Financial question or scenario")
    portfolio_value: float = Field(..., description="Current portfolio value")
    risk_tolerance: RiskTolerance = Field(default=RiskTolerance.aggressive)
    additional_context: Optional[str] = Field(None, description="Any additional context")

class PortfolioAnalysis(BaseModel):
    current_allocation: Dict[str, float]
    recommended_allocation: Dict[str, float]
    performance_metrics: Dict[str, Any]
    risk_score: float
    diversification_score: float

class PortfolioResponse(BaseModel):
    analysis: Dict[str, Any]
    recommendations: List[str]
    risk_assessment: str
    profit_potential: str
    timestamp: datetime = Field(default_factory=datetime.now)

class AdviceResponse(BaseModel):
    advice: str
    confidence_level: float = Field(..., ge=0, le=100, description="Confidence percentage")
    action_items: List[str]
    timestamp: datetime
    market_outlook: Optional[str] = None

class NewsItem(BaseModel):
    title: str
    summary: str
    source: str
    url: str
    published_date: datetime
    sentiment: str  # positive, negative, neutral
    relevance_score: float

class MarketSentiment(BaseModel):
    overall_sentiment: str
    sentiment_score: float  # -1 to 1
    key_factors: List[str]
    volatility_indicator: str

class NewsResponse(BaseModel):
    headlines: List[NewsItem]
    market_sentiment: MarketSentiment
    sector_analysis: Dict[str, Any]
    recommendations: List[str]
    last_updated: datetime = Field(default_factory=datetime.now)

class ETFInfo(BaseModel):
    symbol: str
    name: str
    sector: str
    expense_ratio: float
    current_price: Optional[float] = None
    daily_change: Optional[float] = None
    recommendation_strength: int = Field(..., ge=1, le=10)

class AggressivePlay(BaseModel):
    strategy: str
    description: str
    potential_return: str
    risk_level: int = Field(..., ge=1, le=10)
    time_horizon: str
    entry_conditions: List[str]

class MarketOpportunity(BaseModel):
    opportunity_type: str
    description: str
    symbols: List[str]
    confidence: float
    time_sensitivity: str
    profit_target: str