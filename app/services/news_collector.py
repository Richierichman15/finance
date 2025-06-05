import asyncio
import aiohttp
import feedparser
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import re
from bs4 import BeautifulSoup
import json

class NewsCollector:
    def __init__(self):
        self.news_sources = {
            "reuters_finance": "https://feeds.reuters.com/reuters/businessNews",
            "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
            "bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
            "cnbc": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069"
        }
        
        self.market_symbols = {
            "spy": "S&P 500",
            "qqq": "NASDAQ", 
            "vti": "Total Stock Market",
            "xle": "Energy Sector",
            "vgt": "Technology",
            "vnq": "Real Estate",
            "igf": "Global Infrastructure"
        }

    async def get_market_news(self) -> List[Dict[str, Any]]:
        """Collect latest financial news from multiple sources"""
        all_news = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source_name, url in self.news_sources.items():
                tasks.append(self._fetch_rss_news(session, source_name, url))
            
            news_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in news_results:
                if isinstance(result, list):
                    all_news.extend(result)
        
        # Sort by recency and relevance
        all_news = sorted(all_news, key=lambda x: x.get('published_date', datetime.min), reverse=True)
        
        # Return top 20 most recent
        return all_news[:20]

    async def _fetch_rss_news(self, session: aiohttp.ClientSession, source: str, url: str) -> List[Dict[str, Any]]:
        """Fetch news from RSS feed"""
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Parse RSS feed
                    feed = feedparser.parse(content)
                    news_items = []
                    
                    for entry in feed.entries[:10]:  # Limit to 10 per source
                        news_item = {
                            "title": entry.get('title', ''),
                            "summary": self._clean_text(entry.get('summary', entry.get('description', ''))),
                            "source": source,
                            "url": entry.get('link', ''),
                            "published_date": self._parse_date(entry.get('published', '')),
                            "sentiment": self._analyze_sentiment(entry.get('title', '') + ' ' + entry.get('summary', '')),
                            "relevance_score": self._calculate_relevance(entry.get('title', '') + ' ' + entry.get('summary', ''))
                        }
                        news_items.append(news_item)
                    
                    return news_items
                        
        except Exception as e:
            print(f"Error fetching news from {source}: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        """Clean HTML and extra characters from text"""
        if not text:
            return ""
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        cleaned = soup.get_text()
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned[:500]  # Limit length

    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats"""
        try:
            # Try common RSS date formats
            import email.utils
            return datetime(*email.utils.parsedate(date_str)[:6])
        except:
            return datetime.now()

    def _analyze_sentiment(self, text: str) -> str:
        """Basic sentiment analysis based on keywords"""
        positive_words = ['up', 'gain', 'profit', 'growth', 'bull', 'surge', 'rally', 'boost', 'rise', 'soar']
        negative_words = ['down', 'loss', 'crash', 'bear', 'fall', 'drop', 'decline', 'plunge', 'tumble', 'slump']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"

    def _calculate_relevance(self, text: str) -> float:
        """Calculate relevance score based on financial keywords"""
        financial_keywords = [
            'market', 'stock', 'etf', 'portfolio', 'investment', 'trading', 'profit', 'earnings',
            'fed', 'interest rate', 'inflation', 'gdp', 'economy', 'sector', 'energy', 'technology',
            'nasdaq', 'sp500', 's&p', 'dow', 'bitcoin', 'crypto', 'oil', 'gas'
        ]
        
        text_lower = text.lower()
        relevance_score = sum(1 for keyword in financial_keywords if keyword in text_lower)
        return min(relevance_score / 5.0, 1.0)  # Normalize to 0-1

    async def get_market_sentiment(self) -> Dict[str, Any]:
        """Analyze overall market sentiment from multiple indicators"""
        try:
            # Get market data
            market_data = await self._get_market_indicators()
            
            # Calculate sentiment score
            sentiment_score = self._calculate_market_sentiment_score(market_data)
            
            return {
                "overall_sentiment": self._get_sentiment_label(sentiment_score),
                "sentiment_score": sentiment_score,
                "key_factors": self._identify_key_factors(market_data),
                "volatility_indicator": self._assess_volatility(market_data),
                "market_data": market_data
            }
            
        except Exception as e:
            print(f"Error getting market sentiment: {e}")
            return {
                "overall_sentiment": "neutral",
                "sentiment_score": 0.0,
                "key_factors": ["Market data temporarily unavailable"],
                "volatility_indicator": "unknown"
            }

    async def _get_market_indicators(self) -> Dict[str, Any]:
        """Get key market indicators"""
        indicators = {}
        
        try:
            # Get major index data
            for symbol, name in self.market_symbols.items():
                ticker = yf.Ticker(symbol.upper())
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change_pct = ((current_price - prev_price) / prev_price) * 100
                    
                    indicators[symbol] = {
                        "name": name,
                        "current_price": current_price,
                        "daily_change": change_pct,
                        "weekly_change": ((current_price - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                    }
                    
        except Exception as e:
            print(f"Error fetching market indicators: {e}")
            
        return indicators

    def _calculate_market_sentiment_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate overall sentiment score from market data"""
        if not market_data:
            return 0.0
            
        daily_changes = [data.get('daily_change', 0) for data in market_data.values()]
        avg_daily_change = sum(daily_changes) / len(daily_changes) if daily_changes else 0
        
        # Normalize to -1 to 1 scale
        sentiment_score = max(-1.0, min(1.0, avg_daily_change / 5.0))
        return sentiment_score

    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label"""
        if score > 0.3:
            return "bullish"
        elif score < -0.3:
            return "bearish"
        else:
            return "neutral"

    def _identify_key_factors(self, market_data: Dict[str, Any]) -> List[str]:
        """Identify key market driving factors"""
        factors = []
        
        for symbol, data in market_data.items():
            daily_change = data.get('daily_change', 0)
            if abs(daily_change) > 2:  # Significant move
                direction = "up" if daily_change > 0 else "down"
                factors.append(f"{data['name']} moving {direction} ({daily_change:.1f}%)")
        
        if not factors:
            factors = ["Market showing steady trading patterns"]
            
        return factors[:5]  # Top 5 factors

    def _assess_volatility(self, market_data: Dict[str, Any]) -> str:
        """Assess market volatility"""
        if not market_data:
            return "unknown"
            
        daily_changes = [abs(data.get('daily_change', 0)) for data in market_data.values()]
        avg_volatility = sum(daily_changes) / len(daily_changes) if daily_changes else 0
        
        if avg_volatility > 3:
            return "high"
        elif avg_volatility > 1.5:
            return "moderate"
        else:
            return "low"

    async def get_comprehensive_news(self) -> Dict[str, Any]:
        """Get comprehensive news analysis for the AI advisor"""
        # Get news and market data in parallel
        news_task = self.get_market_news()
        sentiment_task = self.get_market_sentiment()
        
        news, sentiment = await asyncio.gather(news_task, sentiment_task)
        
        # Analyze sectors from news
        sector_analysis = self._analyze_sector_trends(news)
        
        # Generate opportunities based on news and sentiment
        opportunities = self._identify_opportunities(news, sentiment, sector_analysis)
        
        return {
            "headlines": news,
            "sentiment": sentiment,
            "sectors": sector_analysis,
            "opportunities": opportunities,
            "last_updated": datetime.now()
        }

    def _analyze_sector_trends(self, news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sector trends from news"""
        sectors = {
            "technology": {"mentions": 0, "sentiment": "neutral"},
            "energy": {"mentions": 0, "sentiment": "neutral"},
            "finance": {"mentions": 0, "sentiment": "neutral"},
            "healthcare": {"mentions": 0, "sentiment": "neutral"},
            "real_estate": {"mentions": 0, "sentiment": "neutral"}
        }
        
        sector_keywords = {
            "technology": ["tech", "ai", "software", "semiconductor", "apple", "microsoft", "google"],
            "energy": ["oil", "gas", "energy", "renewable", "solar", "wind", "crude"],
            "finance": ["bank", "financial", "fed", "interest", "credit", "loan"],
            "healthcare": ["health", "pharma", "drug", "medical", "biotech"],
            "real_estate": ["real estate", "reit", "housing", "property", "mortgage"]
        }
        
        for news_item in news:
            text = (news_item.get('title', '') + ' ' + news_item.get('summary', '')).lower()
            
            for sector, keywords in sector_keywords.items():
                if any(keyword in text for keyword in keywords):
                    sectors[sector]["mentions"] += 1
                    
                    # Aggregate sentiment
                    if news_item.get('sentiment') == 'positive':
                        sectors[sector]["sentiment"] = "positive"
                    elif news_item.get('sentiment') == 'negative' and sectors[sector]["sentiment"] != "positive":
                        sectors[sector]["sentiment"] = "negative"
        
        return sectors

    def _identify_opportunities(self, news: List[Dict[str, Any]], sentiment: Dict[str, Any], 
                             sectors: Dict[str, Any]) -> List[str]:
        """Identify investment opportunities based on news and market analysis"""
        opportunities = []
        
        # Market sentiment based opportunities
        if sentiment.get('overall_sentiment') == 'bullish':
            opportunities.append("BULLISH MOMENTUM: Consider increasing growth allocations (QQQ, VGT)")
        elif sentiment.get('overall_sentiment') == 'bearish':
            opportunities.append("BEARISH PROTECTION: Time to increase defensive positions and wait for dip buying opportunities")
        
        # Sector-specific opportunities
        for sector, data in sectors.items():
            if data['mentions'] > 2 and data['sentiment'] == 'positive':
                if sector == 'technology':
                    opportunities.append(f"TECH SURGE: {sector.upper()} showing strong momentum - QQQ/VGT looking HOT! 🚀")
                elif sector == 'energy':
                    opportunities.append(f"ENERGY BOOM: {sector.upper()} in the spotlight - XLE/AMLP ready to PUMP! ⛽")
                elif sector == 'real_estate':
                    opportunities.append(f"REAL ESTATE PLAY: {sector.upper()} gaining traction - VNQ opportunity! 🏢")
        
        # Volatility opportunities
        volatility = sentiment.get('volatility_indicator')
        if volatility == 'high':
            opportunities.append("HIGH VOLATILITY: Perfect for swing trading - time to make QUICK PROFITS! 💰")
        elif volatility == 'low':
            opportunities.append("LOW VOLATILITY: Accumulation phase - DCA into quality positions")
        
        if not opportunities:
            opportunities = ["STEADY MARKETS: Perfect time to rebalance and optimize allocations"]
            
        return opportunities[:5]  # Top 5 opportunities