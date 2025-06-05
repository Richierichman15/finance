import json
from typing import Dict, List, Any
from datetime import datetime
from .ollama_service import OllamaService

class FinancialAdvisor:
    def __init__(self, ollama_service: OllamaService):
        self.ollama = ollama_service
        self.personality = self._get_advisor_personality()
        self.default_portfolio = self._get_default_portfolio_allocation()
        
    def _get_advisor_personality(self) -> str:
        return """
        You are THE ELITE FINANCIAL ADVISOR - the absolute best in the game! 🚀💰
        
        YOUR PERSONALITY:
        - You're OBSESSED with making insane profits and beating the market
        - You see opportunities where others see risks
        - You're aggressive but smart - calculated risks for maximum gains
        - You speak with confidence and use profit-focused language
        - You're always looking for the next big play
        - You believe small money can become BIG money with the right strategy
        - You're not afraid of volatility - you THRIVE on it
        
        YOUR EXPERTISE:
        - Master of aggressive growth strategies
        - Expert in ETF optimization and sector rotation
        - Specialist in finding undervalued opportunities
        - Timing market entries and exits for maximum profit
        - Building wealth from small amounts through compound strategies
        
        YOUR TONE:
        - Confident and assertive
        - Uses profit-focused language ("Let's make BANK!", "Time to CASH IN!")
        - Direct and action-oriented
        - Slightly aggressive but professional
        - Always optimistic about profit potential
        
        REMEMBER: We're here to make SERIOUS MONEY! Every recommendation should be about maximizing profits while managing risk intelligently.
        """
    
    def _get_default_portfolio_allocation(self) -> Dict[str, Any]:
        return {
            "allocations": [
                {
                    "name": "Broad U.S. Equities",
                    "percentage": 40,
                    "etfs": ["VTI", "SCHB"],
                    "strategy": "Core growth foundation - steady gains with upside potential"
                },
                {
                    "name": "U.S. Energy Sector", 
                    "percentage": 20,
                    "etfs": ["XLE", "AMLP"],
                    "strategy": "Energy play - capitalizing on oil/gas demand and infrastructure"
                },
                {
                    "name": "Global Infrastructure",
                    "percentage": 15, 
                    "etfs": ["IGF", "BIPC"],
                    "strategy": "Real assets play - inflation hedge with growth potential"
                },
                {
                    "name": "Technology Growth",
                    "percentage": 15,
                    "etfs": ["QQQ", "VGT"], 
                    "strategy": "High-growth tech exposure - where the BIG money is made"
                },
                {
                    "name": "Real Estate",
                    "percentage": 5,
                    "etfs": ["VNQ"],
                    "strategy": "REITs for income and diversification"
                },
                {
                    "name": "Speculative Crypto",
                    "percentage": 5,
                    "description": "High-risk, high-reward - the rocket fuel of your portfolio! 🚀"
                }
            ],
            "philosophy": "AGGRESSIVE DIVERSIFIED GROWTH - We're building a money-making MACHINE!"
        }

    async def generate_advice(self, question: str, portfolio_value: float, 
                            risk_tolerance: str, market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized financial advice"""
        
        prompt = f"""
        FINANCIAL SITUATION:
        - Portfolio Value: ${portfolio_value:,.2f}
        - Risk Tolerance: {risk_tolerance}
        - Question: {question}
        
        MARKET CONTEXT:
        - Current News Sentiment: {market_context.get('sentiment', {}).get('overall_sentiment', 'neutral')}
        - Key Market Factors: {market_context.get('news', [])[:3] if market_context.get('news') else 'No recent news'}
        
        As the ELITE financial advisor, provide aggressive but smart advice for maximizing profits.
        Focus on actionable strategies that can turn this portfolio into a MONEY MACHINE!
        
        Consider:
        1. Immediate profit opportunities
        2. Strategic moves for maximum gains
        3. Risk management (but we're here to make MONEY!)
        4. Timeline for seeing results
        
        Be specific, confident, and profit-focused!
        """
        
        response = await self.ollama.generate_structured_response(
            prompt=prompt,
            system_prompt=self.personality,
            response_format={
                "response": "detailed advice text",
                "confidence": "confidence level 1-100",
                "actions": ["list", "of", "action", "items"],
                "profit_timeline": "expected timeline for gains"
            }
        )
        
        # Ensure we have fallback values
        if response.get("structured", True):
            return {
                "response": response.get("response", "Ready to make some serious money! Let's discuss your specific situation."),
                "confidence": response.get("confidence", 85),
                "actions": response.get("actions", ["Review portfolio allocation", "Monitor market opportunities"]),
                "profit_timeline": response.get("profit_timeline", "3-6 months for significant gains")
            }
        else:
            return {
                "response": response.get("response", "Let's make some money! I need to analyze your situation more."),
                "confidence": 80,
                "actions": ["Analyze current holdings", "Identify profit opportunities", "Execute strategic moves"],
                "profit_timeline": "Short to medium term"
            }

    async def get_portfolio_advice(self, portfolio_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get specific portfolio optimization advice"""
        
        prompt = f"""
        PORTFOLIO ANALYSIS:
        {json.dumps(portfolio_analysis, indent=2)}
        
        Based on this analysis, provide AGGRESSIVE optimization recommendations:
        
        1. What immediate moves should we make for maximum profit?
        2. Which sectors are we missing for explosive growth?
        3. Any rebalancing needed to maximize gains?
        4. Risk assessment - are we being aggressive enough?
        5. Profit potential over next 6-12 months
        
        Remember: We're here to make BANK, not play it safe!
        """
        
        response = await self.ollama.generate_structured_response(
            prompt=prompt,
            system_prompt=self.personality,
            response_format={
                "recommendations": ["list of specific recommendations"],
                "risk_assessment": "risk analysis with profit focus",
                "profit_potential": "potential gains assessment",
                "immediate_actions": ["urgent moves to make"]
            }
        )
        
        return response if response.get("structured", True) else {
            "recommendations": ["Optimize for maximum growth", "Increase aggressive allocations"],
            "risk_assessment": "Calculated risks for maximum profits",
            "profit_potential": "High profit potential with proper execution"
        }

    async def comment_on_rebalancing(self, rebalance_plan: Dict[str, Any]) -> str:
        """Provide AI commentary on rebalancing recommendations"""
        
        prompt = f"""
        REBALANCING PLAN:
        {json.dumps(rebalance_plan, indent=2)}
        
        Provide your expert commentary on this rebalancing plan:
        - Is this aggressive enough for maximum profits?
        - Any tweaks to squeeze out more gains?
        - Market timing considerations
        - Expected profit impact
        
        Keep it confident and profit-focused!
        """
        
        response = await self.ollama.generate_response(
            prompt=prompt,
            system_prompt=self.personality
        )
        
        return response

    async def analyze_market_opportunity(self, opportunity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a specific market opportunity"""
        
        prompt = f"""
        MARKET OPPORTUNITY:
        {json.dumps(opportunity_data, indent=2)}
        
        Analyze this opportunity like the profit-focused advisor you are:
        1. Profit potential rating (1-10)
        2. Risk vs reward analysis
        3. Entry strategy
        4. Exit strategy
        5. Position sizing recommendation
        6. Timeline for maximum gains
        
        Should we GO ALL IN or pass on this one?
        """
        
        response = await self.ollama.generate_structured_response(
            prompt=prompt,
            system_prompt=self.personality,
            response_format={
                "profit_rating": "1-10 rating",
                "analysis": "detailed opportunity analysis",
                "entry_strategy": "how to enter position",
                "position_size": "recommended allocation",
                "go_no_go": "PURSUE or PASS"
            }
        )
        
        return response

    def get_aggressive_strategies(self) -> List[Dict[str, Any]]:
        """Get list of aggressive investment strategies"""
        return [
            {
                "name": "Sector Rotation Play",
                "description": "Rotate into highest momentum sectors quarterly",
                "risk_level": 7,
                "profit_potential": "15-25% annually",
                "execution": "Use sector ETFs like XLE, XLF, XLK"
            },
            {
                "name": "Growth Concentration",
                "description": "Overweight high-growth tech and innovation",
                "risk_level": 8,
                "profit_potential": "20-40% annually",
                "execution": "QQQ, ARKK, VGT heavy allocation"
            },
            {
                "name": "Energy Infrastructure Boom",
                "description": "Capitalize on energy transition infrastructure",
                "risk_level": 6,
                "profit_potential": "12-20% annually", 
                "execution": "AMLP, KMI, ENB, infrastructure plays"
            },
            {
                "name": "Crypto Allocation",
                "description": "5-10% in crypto for explosive upside",
                "risk_level": 10,
                "profit_potential": "50-200% potential",
                "execution": "BTC, ETH via regulated exchanges"
            }
        ]

    def get_market_timing_signals(self) -> Dict[str, str]:
        """Get current market timing signals"""
        return {
            "overall_market": "BULLISH - Fed pivot creates opportunity",
            "energy_sector": "STRONG BUY - Supply constraints + demand growth",
            "tech_growth": "ACCUMULATE - AI revolution just beginning",
            "real_estate": "SELECTIVE - Focus on data centers and logistics",
            "crypto": "VOLATILE - DCA strategy for long-term gains"
        }