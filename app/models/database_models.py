from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime

from ..database import Base

class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    holdings = relationship("Holding", back_populates="portfolio")

class Holding(Base):
    __tablename__ = "holdings"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"))
    symbol = Column(String, index=True)
    quantity = Column(Float)
    purchase_price = Column(Float)
    purchase_date = Column(DateTime, default=datetime.utcnow)
    
    portfolio = relationship("Portfolio", back_populates="holdings")
    price_history = relationship("StockPrice", back_populates="holding")

class StockPrice(Base):
    __tablename__ = "stock_prices"

    id = Column(Integer, primary_key=True, index=True)
    holding_id = Column(Integer, ForeignKey("holdings.id"))
    price = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    holding = relationship("Holding", back_populates="price_history")

class StockFundamentals(Base):
    __tablename__ = "stock_fundamentals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    price = Column(Float)
    market_cap = Column(Float)
    pe_ratio_ttm = Column(Float)
    peg_ratio = Column(Float)
    eps = Column(Float)
    free_cash_flow = Column(Float)
    fcf_yield = Column(Float)
    debt_to_equity = Column(Float)
    current_ratio = Column(Float)
    dividend_yield = Column(Float)
    roe = Column(Float)  # Return on Equity
    timestamp = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

class NewsItem(Base):
    __tablename__ = "news_items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    summary = Column(String)
    source = Column(String)
    url = Column(String)
    published_date = Column(DateTime)
    sentiment = Column(String)  # positive, negative, neutral
    relevance_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    value = Column(String)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow) 